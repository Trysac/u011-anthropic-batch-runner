"""Tests for the command line interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from helpers import FakeTransport, ok, simple

from anthropic_batch_runner import BatchError, BatchJob, HttpBatchTransport
from anthropic_batch_runner import __main__ as cli

SENTINEL = "KEYSENTINEL-0123456789"


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch) -> FakeTransport:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL)
    fake = FakeTransport()
    monkeypatch.setattr(cli, "_make_transport", lambda api_key: fake)
    return fake


def submit_args(tmp_path: Path, items: str | None = None) -> list[str]:
    items_file = tmp_path / "items.jsonl"
    content = items
    if content is None:
        content = '{"key": "a1", "user_content": "hi"}\n\n{"key":"b2","user_content":"yo"}\n'
    items_file.write_text(content, encoding="utf-8")
    (tmp_path / "system.txt").write_text("You are terse.", encoding="utf-8")
    return [
        "submit",
        "--state",
        str(tmp_path / "state.json"),
        "--items",
        str(items_file),
        "--model",
        "m",
        "--system",
        str(tmp_path / "system.txt"),
        "--max-tokens",
        "100",
    ]


def test_unset_key_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert cli.main(["collect", "--state", str(tmp_path / "s.json")]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
    monkeypatch.setenv("ANTHROPIC_API_KEY", "  ")
    assert cli.main(["collect", "--state", str(tmp_path / "s.json")]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_argument_errors_and_help(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert cli.main([]) == 2
    assert cli.main(["bogus"]) == 2
    assert cli.main(["submit"]) == 2
    assert cli.main(["--help"]) == 0
    capsys.readouterr()


def test_submit_collect_requeue(
    transport: FakeTransport, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state = tmp_path / "state.json"
    assert cli.main(submit_args(tmp_path)) == 0
    assert capsys.readouterr().out == "msgbatch_1\n"
    assert [r["custom_id"] for r in transport.created[0]] == ["a1--a1", "b2--a1"]
    assert transport.created[0][0]["params"]["system"][0]["text"] == "You are terse."
    assert {s.status for s in BatchJob.load(state).items().values()} == {"submitted"}
    assert SENTINEL not in state.read_text(encoding="utf-8")

    transport.status = "in_progress"
    assert cli.main(["collect", "--state", str(state)]) == 1
    assert "not_ended" in capsys.readouterr().err

    transport.status = "ended"
    transport.lines["msgbatch_1"] = [simple("a1--a1", "expired"), ok("b2--a1")]
    assert cli.main(["collect", "--state", str(state)]) == 0
    counts = json.loads(capsys.readouterr().out)
    assert counts["expired"] == 1
    assert counts["succeeded"] == 1
    assert counts["ignored"] == 0

    assert cli.main(["requeue", "--state", str(state)]) == 0
    assert capsys.readouterr().out == "msgbatch_2\n"
    transport.lines["msgbatch_2"] = [ok("a1--a2")]
    assert cli.main(["collect", "--state", str(state)]) == 0
    capsys.readouterr()
    assert cli.main(["requeue", "--state", str(state)]) == 0
    assert capsys.readouterr().out == "nothing to requeue\n"


def test_submit_failure_removes_the_state_file(
    transport: FakeTransport, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    transport.create_error = BatchError("http_error", "boom", status=500)
    assert cli.main(submit_args(tmp_path)) == 1
    assert "http_error" in capsys.readouterr().err
    assert not (tmp_path / "state.json").exists()


def test_submit_refuses_existing_state(
    transport: FakeTransport, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(submit_args(tmp_path)) == 0
    assert cli.main(submit_args(tmp_path)) == 2
    assert "state_file" in capsys.readouterr().err
    assert len(transport.created) == 1


@pytest.mark.parametrize(
    "items",
    [
        '{"key": "a b", "user_content": "hi"}\n',
        "not json\n",
        '{"key": "a"}\n',
        '{"key": 5, "user_content": "x"}\n',
        "",
    ],
)
def test_invalid_items_are_usage_errors(
    transport: FakeTransport,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    items: str,
) -> None:
    assert cli.main(submit_args(tmp_path, items)) == 2
    assert capsys.readouterr().err.startswith("error")
    assert transport.created == []
    assert not (tmp_path / "state.json").exists()


def test_invalid_input_files(
    transport: FakeTransport,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = submit_args(tmp_path)
    (tmp_path / "items.jsonl").write_bytes(b"\xff\xfe")
    assert cli.main(args) == 2
    (tmp_path / "items.jsonl").unlink()
    assert cli.main(args) == 2
    assert "FileNotFoundError" in capsys.readouterr().err
    args = submit_args(tmp_path)
    monkeypatch.setattr(cli, "MAX_ITEMS_FILE_BYTES", 5)
    assert cli.main(args) == 2
    monkeypatch.setattr(cli, "MAX_ITEMS_FILE_BYTES", 1024)
    monkeypatch.setattr(cli, "MAX_SYSTEM_FILE_BYTES", 3)
    assert cli.main(args) == 2
    monkeypatch.setattr(cli, "MAX_SYSTEM_FILE_BYTES", 1024)
    (tmp_path / "system.txt").unlink()
    assert cli.main(args) == 2
    (tmp_path / "system.txt").write_bytes(b"\xff")
    assert cli.main(args) == 2
    assert transport.created == []


def test_bad_numbers(transport: FakeTransport, tmp_path: Path) -> None:
    args = submit_args(tmp_path)
    assert cli.main([*args[:-1], "abc"]) == 2
    assert cli.main([*args[:-1], "0"]) == 2
    assert cli.main([*args, "--max-attempts", "10"]) == 2
    assert transport.created == []


def test_collect_with_missing_or_corrupt_state(
    transport: FakeTransport, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state = tmp_path / "state.json"
    assert cli.main(["collect", "--state", str(state)]) == 2
    state.write_text("{corrupt", encoding="utf-8")
    assert cli.main(["requeue", "--state", str(state)]) == 2
    assert "state_file" in capsys.readouterr().err


def test_state_path_that_is_a_directory(
    transport: FakeTransport, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["collect", "--state", str(tmp_path)]) == 2
    capsys.readouterr()


def test_default_transport_is_http(monkeypatch: pytest.MonkeyPatch) -> None:
    made = cli._make_transport(SENTINEL)
    assert isinstance(made, HttpBatchTransport)
    assert SENTINEL not in repr(made)
