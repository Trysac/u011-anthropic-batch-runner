"""Tests for build_requests and BatchJob."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
from helpers import BASE, FakeOpener, FakeResponse, FakeTransport, err, ok, simple

import anthropic_batch_runner as pkg
from anthropic_batch_runner import (
    BatchError,
    BatchJob,
    HttpBatchTransport,
    ItemState,
    JobItem,
    build_requests,
)
from anthropic_batch_runner import job as job_module

SENTINEL = "KEYSENTINEL-0123456789"


def make_job(
    tmp_path: Path, keys: tuple[str, ...] = ("a1", "b2"), max_attempts: int = 3
) -> BatchJob:
    items = [JobItem(key, f"content {key}") for key in keys]
    return BatchJob.create(
        tmp_path / "state.json",
        items,
        model="m",
        system="S",
        max_tokens=100,
        max_attempts=max_attempts,
    )


def reject_bad(text: str) -> None:
    if text == "bad":
        raise ValueError("rejected")


def test_version() -> None:
    assert pkg.__version__ == "0.1.0"


def test_build_requests_exact() -> None:
    result = build_requests([JobItem("a1", "hi")], model="m", system="S", max_tokens=100)
    assert result == [
        {
            "custom_id": "a1--a1",
            "params": {
                "model": "m",
                "max_tokens": 100,
                "system": [{"type": "text", "text": "S", "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": "hi"}],
            },
        }
    ]


@pytest.mark.parametrize("key", ["a b", "x" * 60, "", "a\n", "é", "a.b", None])
def test_build_requests_invalid_keys(key: Any) -> None:
    with pytest.raises(BatchError) as info:
        build_requests([JobItem(key, "hi")], model="m", system="S", max_tokens=100)
    assert info.value.code == "invalid_item_key"
    assert len(str(info.value)) <= 200


def test_build_requests_acceptance_literals() -> None:
    """Acceptance text, verbatim.

    build_requests([JobItem('a1', 'hi')], model='m', system='S', max_tokens=100)
    returns [{'custom_id': 'a1--a1'}]
    fragments: ', max_tokens=100) returns [{' and '}]}}], and key ' and '
    ' * 60 raises BatchError with code '
    """
    result = build_requests([JobItem("a1", "hi")], model="m", system="S", max_tokens=100)
    assert result[0]["custom_id"] == "a1--a1"
    for key in ("a b", "x" * 60):
        with pytest.raises(BatchError) as info:
            build_requests([JobItem(key, "hi")], model="m", system="S", max_tokens=100)
        assert info.value.code == "invalid_item_key"


def test_build_requests_key_and_attempt_boundaries() -> None:
    key = "x" * 59
    result = build_requests([JobItem(key, "hi")], model="m", system="S", max_tokens=1, attempt=9)
    assert result[0]["custom_id"] == f"{key}--a9"
    assert len(result[0]["custom_id"]) == 63


def test_build_requests_cache_off_and_independent_blocks() -> None:
    items = [JobItem("a", "1"), JobItem("b", "2")]
    result = build_requests(items, model="m", system="S", max_tokens=1, cache_system=False)
    assert result[0]["params"]["system"] == [{"type": "text", "text": "S"}]
    result[0]["params"]["system"][0]["text"] = "changed"
    assert result[1]["params"]["system"][0]["text"] == "S"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": ""},
        {"model": 5},
        {"system": ""},
        {"system": chr(0xD800)},
        {"max_tokens": 0},
        {"max_tokens": True},
        {"attempt": 0},
        {"attempt": 10},
        {"attempt": True},
    ],
)
def test_build_requests_rejects_bad_parameters(kwargs: dict[str, Any]) -> None:
    params: dict[str, Any] = {"model": "m", "system": "S", "max_tokens": 100}
    params.update(kwargs)
    with pytest.raises(BatchError) as info:
        build_requests([JobItem("a1", "hi")], **params)
    assert info.value.code == "config"


@pytest.mark.parametrize(
    "items",
    [
        [],
        [JobItem("a", "")],
        [JobItem("a", "   ")],
        [JobItem("a", 5)],  # type: ignore[arg-type]
        [JobItem("a", chr(0xD800))],
        ["a"],
        [JobItem("a", "1"), JobItem("a", "2")],
    ],
)
def test_build_requests_rejects_bad_items(items: list[Any]) -> None:
    with pytest.raises(BatchError) as info:
        build_requests(items, model="m", system="S", max_tokens=100)
    assert info.value.code == "config"


def test_build_requests_too_many_items(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(job_module, "MAX_REQUESTS", 1)
    items = [JobItem("a", "1"), JobItem("b", "2")]
    with pytest.raises(BatchError) as info:
        build_requests(items, model="m", system="S", max_tokens=100)
    assert info.value.code == "too_large"


def test_item_state_is_frozen() -> None:
    state = ItemState("pending", 0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.attempts = 2  # type: ignore[misc]
    assert state.output is None
    assert state.error is None


def test_create_writes_state(tmp_path: Path) -> None:
    job = make_job(tmp_path)
    document = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert document["schema"] == 1
    assert document["model"] == "m"
    assert document["system_sha256"] == hashlib.sha256(b"S").hexdigest()
    assert document["max_tokens"] == 100
    assert document["batches"] == []
    assert set(document["items"]) == {"a1", "b2"}
    assert job.items() == {"a1": ItemState("pending", 0), "b2": ItemState("pending", 0)}
    assert "BatchJob(" in repr(job)


def test_create_accepts_string_path(tmp_path: Path) -> None:
    path = str(tmp_path / "s.json")
    BatchJob.create(path, [JobItem("a", "x")], model="m", system="S", max_tokens=1)
    assert Path(path).exists()


def test_create_refuses_existing_file(tmp_path: Path) -> None:
    make_job(tmp_path)
    with pytest.raises(BatchError) as info:
        make_job(tmp_path)
    assert info.value.code == "state_file"


@pytest.mark.parametrize("max_attempts", [0, 10, True])
def test_create_rejects_bad_max_attempts(tmp_path: Path, max_attempts: Any) -> None:
    with pytest.raises(BatchError) as info:
        make_job(tmp_path, max_attempts=max_attempts)
    assert info.value.code == "config"


def test_create_rejects_bad_input(tmp_path: Path) -> None:
    with pytest.raises(BatchError) as info:
        BatchJob.create(tmp_path / "s.json", [], model="m", system="S", max_tokens=1)
    assert info.value.code == "config"
    with pytest.raises(BatchError) as info:
        BatchJob.create(
            tmp_path / "s.json", [JobItem("a", "x")], model="", system="S", max_tokens=1
        )
    assert info.value.code == "config"
    assert not (tmp_path / "s.json").exists()


def test_create_in_missing_directory_is_state_file_error(tmp_path: Path) -> None:
    path = tmp_path / "missing" / "s.json"
    with pytest.raises(BatchError) as info:
        BatchJob.create(path, [JobItem("a", "x")], model="m", system="S", max_tokens=1)
    assert info.value.code == "state_file"


def test_create_too_large_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(job_module, "MAX_STATE_BYTES", 10)
    with pytest.raises(BatchError) as info:
        make_job(tmp_path)
    assert info.value.code == "too_large"
    assert list(tmp_path.iterdir()) == []


def test_submit_marks_items_and_persists(tmp_path: Path) -> None:
    job = make_job(tmp_path)
    transport = FakeTransport()
    assert job.submit(transport) == "msgbatch_1"
    assert [r["custom_id"] for r in transport.created[0]] == ["a1--a1", "b2--a1"]
    assert transport.created[0][0]["params"]["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert set(job.items().values()) == {ItemState("submitted", 1)}
    assert BatchJob.load(tmp_path / "state.json").items() == job.items()
    with pytest.raises(BatchError) as info:
        job.submit(transport)
    assert info.value.code == "config"


def test_submit_failure_keeps_state(tmp_path: Path) -> None:
    job = make_job(tmp_path)
    before = (tmp_path / "state.json").read_bytes()
    transport = FakeTransport()
    transport.create_error = BatchError("http_error", "boom", status=500)
    with pytest.raises(BatchError):
        job.submit(transport)
    assert set(job.items().values()) == {ItemState("pending", 0)}
    assert (tmp_path / "state.json").read_bytes() == before


def test_submit_rejects_invalid_batch_id(tmp_path: Path) -> None:
    class BadIdTransport(FakeTransport):
        def create(self, requests: list[dict[str, Any]]) -> str:
            return "bad id!"

    job = make_job(tmp_path)
    with pytest.raises(BatchError) as info:
        job.submit(BadIdTransport())
    assert info.value.code == "http_error"
    assert set(job.items().values()) == {ItemState("pending", 0)}


def test_collect_not_ended_then_ended(tmp_path: Path) -> None:
    job = make_job(tmp_path)
    transport = FakeTransport()
    job.submit(transport)
    transport.lines["msgbatch_1"] = [ok("a1--a1"), ok("b2--a1")]
    for status in ("in_progress", "canceling"):
        transport.status = status
        with pytest.raises(BatchError) as info:
            job.collect(transport)
        assert info.value.code == "not_ended"
        assert set(job.items().values()) == {ItemState("submitted", 1)}
    transport.status = "ended"
    counts = job.collect(transport)
    assert counts["succeeded"] == 2
    assert counts["ignored"] == 0


def test_collect_maps_results_in_any_order(tmp_path: Path) -> None:
    job = make_job(tmp_path, ("a", "b", "c", "d", "e"))
    transport = FakeTransport()
    job.submit(transport)
    split = {
        "custom_id": "b--a1",
        "result": {
            "type": "succeeded",
            "message": {
                "content": [{"type": "text", "text": "Hel"}, {"type": "text", "text": "lo"}]
            },
        },
    }
    transport.lines["msgbatch_1"] = [
        simple("e--a1", "expired"),
        err("c--a1", "overloaded_error"),
        ok("a--a1", "fine"),
        simple("d--a1", "canceled"),
        split,
    ]
    counts = job.collect(transport)
    assert job.items() == {
        "a": ItemState("succeeded", 1, "fine"),
        "b": ItemState("succeeded", 1, "Hello"),
        "c": ItemState("errored", 1, None, "overloaded_error"),
        "d": ItemState("canceled", 1),
        "e": ItemState("expired", 1),
    }
    assert counts == {
        "pending": 0,
        "submitted": 0,
        "succeeded": 2,
        "errored": 1,
        "canceled": 1,
        "expired": 1,
        "invalid_output": 0,
        "failed": 0,
        "exhausted": 0,
        "ignored": 0,
    }
    assert BatchJob.load(tmp_path / "state.json").items() == job.items()
    assert "details that must not be stored" not in (tmp_path / "state.json").read_text()


def test_collect_validator(tmp_path: Path) -> None:
    job = make_job(tmp_path, ("a", "b"))
    transport = FakeTransport()
    job.submit(transport)
    transport.lines["msgbatch_1"] = [ok("a--a1", "bad"), ok("b--a1", "good")]
    job.collect(transport, validator=reject_bad)
    assert job.items()["a"] == ItemState("invalid_output", 1, "bad", "ValueError")
    assert job.items()["b"] == ItemState("succeeded", 1, "good")


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ("not a dict", ItemState("errored", 1, None, "malformed_result")),
        (
            {"type": "succeeded", "message": {"content": "no"}},
            ItemState("errored", 1, None, "malformed_result"),
        ),
        ({"type": "succeeded"}, ItemState("errored", 1, None, "malformed_result")),
        ({"type": "weird"}, ItemState("errored", 1, None, "malformed_result")),
        (
            {
                "type": "succeeded",
                "message": {
                    "content": [
                        {"type": "tool_use"},
                        {"type": "text", "text": 5},
                        "str",
                        {"type": "text", "text": "o"},
                        {"type": "text", "text": "k"},
                    ]
                },
            },
            ItemState("succeeded", 1, "ok"),
        ),
        (
            {
                "type": "succeeded",
                "message": {"content": [{"type": "text", "text": "x" * 100_001}]},
            },
            ItemState("invalid_output", 1, None, "output_too_large"),
        ),
        (
            {
                "type": "succeeded",
                "message": {"content": [{"type": "text", "text": "x" * 100_000}]},
            },
            ItemState("succeeded", 1, "x" * 100_000),
        ),
        (
            {
                "type": "errored",
                "error": {"type": "error", "error": {"type": "invalid_request_error"}},
            },
            ItemState("errored", 1, None, "invalid_request_error"),
        ),
        (
            {"type": "errored", "error": {"type": "overloaded_error"}},
            ItemState("errored", 1, None, "overloaded_error"),
        ),
        (
            {"type": "errored", "error": {"type": "error", "error": {"type": "bad type!"}}},
            ItemState("errored", 1, None, "unknown_error"),
        ),
        ({"type": "errored"}, ItemState("errored", 1, None, "unknown_error")),
        ({"type": "canceled"}, ItemState("canceled", 1)),
        ({"type": "expired"}, ItemState("expired", 1)),
    ],
)
def test_collect_result_shapes(tmp_path: Path, result: Any, expected: ItemState) -> None:
    job = make_job(tmp_path, ("a1",))
    transport = FakeTransport()
    job.submit(transport)
    transport.lines["msgbatch_1"] = [{"custom_id": "a1--a1", "result": result}]
    job.collect(transport)
    assert job.items()["a1"] == expected


def test_collect_ignores_stale_unknown_and_duplicate_lines(tmp_path: Path) -> None:
    job = make_job(tmp_path, ("a1", "b2"))
    transport = FakeTransport()
    job.submit(transport)
    transport.lines["msgbatch_1"] = [simple("a1--a1", "expired"), ok("b2--a1", "kept")]
    job.collect(transport)
    assert job.requeue_failed(transport) == "msgbatch_2"
    assert job.items()["a1"] == ItemState("submitted", 2)
    transport.lines["msgbatch_2"] = [
        ok("a1--a1", "stale"),
        ok("zzz--a1", "unknown"),
        ok("b2--a1", "rewrite attempt"),
        ok("a1--a2", "first"),
        ok("a1--a2", "second"),
        "not a dict",
        {"custom_id": 5},
        {"custom_id": "bad id"},
        {"custom_id": "x" * 70 + "--a1"},
        {},
    ]
    counts = job.collect(transport)
    assert counts["ignored"] == 9
    assert job.items()["a1"] == ItemState("succeeded", 2, "first")
    assert job.items()["b2"] == ItemState("succeeded", 1, "kept")


def test_collect_with_nothing_pending(tmp_path: Path) -> None:
    job = make_job(tmp_path)
    transport = FakeTransport()
    counts = job.collect(transport)
    assert counts["pending"] == 2
    assert counts["ignored"] == 0
    assert transport.retrieved == []
    job.submit(transport)
    transport.lines["msgbatch_1"] = [ok("a1--a1"), ok("b2--a1")]
    job.collect(transport)
    retrieved = len(transport.retrieved)
    assert job.collect(transport)["succeeded"] == 2
    assert len(transport.retrieved) == retrieved


def test_collect_failure_while_reading_changes_nothing(tmp_path: Path) -> None:
    class Exploding(FakeTransport):
        def results(self, batch: dict[str, Any]) -> Any:
            yield ok("a1--a1")
            raise BatchError("http_error", "stream failed")

    job = make_job(tmp_path)
    transport = Exploding()
    job.submit(transport)
    with pytest.raises(BatchError):
        job.collect(transport)
    assert set(job.items().values()) == {ItemState("submitted", 1)}
    assert BatchJob.load(tmp_path / "state.json").items() == job.items()


def test_requeue_rules_and_exhaustion(tmp_path: Path) -> None:
    keys = ("e", "c", "v", "o", "r", "s")
    job = make_job(tmp_path, keys, max_attempts=2)
    transport = FakeTransport()
    job.submit(transport)
    transport.lines["msgbatch_1"] = [
        simple("e--a1", "expired"),
        simple("c--a1", "canceled"),
        ok("v--a1", "bad"),
        err("o--a1", "overloaded_error"),
        err("r--a1", "invalid_request_error"),
        ok("s--a1", "fine"),
    ]
    job.collect(transport, validator=reject_bad)
    assert job.requeue_failed(transport) == "msgbatch_2"
    ids = {r["custom_id"] for r in transport.created[1]}
    assert ids == {"e--a2", "c--a2", "v--a2", "o--a2"}
    for key in ("e", "c", "v", "o"):
        assert job.items()[key] == ItemState("submitted", 2)
    assert job.items()["r"] == ItemState("failed", 1, None, "invalid_request_error")
    assert job.items()["s"] == ItemState("succeeded", 1, "fine")
    transport.lines["msgbatch_2"] = [simple(f"{key}--a2", "expired") for key in "ecvo"]
    job.collect(transport)
    assert job.requeue_failed(transport) is None
    for key in ("e", "c", "v", "o"):
        assert job.items()[key].status == "exhausted"
    snapshot = dict(job.items())
    assert job.requeue_failed(transport) is None
    assert job.items() == snapshot
    assert len(transport.created) == 2
    assert BatchJob.load(tmp_path / "state.json").items() == snapshot


def test_requeue_with_only_terminal_transitions(tmp_path: Path) -> None:
    job = make_job(tmp_path, ("r",))
    transport = FakeTransport()
    assert job.requeue_failed(transport) is None
    job.submit(transport)
    assert job.requeue_failed(transport) is None
    transport.lines["msgbatch_1"] = [err("r--a1", "invalid_request_error")]
    job.collect(transport)
    assert job.requeue_failed(transport) is None
    assert job.items()["r"].status == "failed"
    assert BatchJob.load(tmp_path / "state.json").items()["r"].status == "failed"
    assert len(transport.created) == 1


def test_requeue_mixed_attempts_use_their_own_suffix(tmp_path: Path) -> None:
    job = make_job(tmp_path, ("a", "b"))
    transport = FakeTransport()
    job.submit(transport)
    transport.lines["msgbatch_1"] = [simple("a--a1", "expired"), ok("b--a1")]
    job.collect(transport)
    job.requeue_failed(transport)
    transport.lines["msgbatch_2"] = [simple("a--a2", "expired")]
    job.collect(transport)
    assert job.requeue_failed(transport) == "msgbatch_3"
    assert [r["custom_id"] for r in transport.created[2]] == ["a--a3"]


def test_requeue_create_failure_changes_nothing(tmp_path: Path) -> None:
    job = make_job(tmp_path, ("a", "r"))
    transport = FakeTransport()
    job.submit(transport)
    transport.lines["msgbatch_1"] = [
        simple("a--a1", "expired"),
        err("r--a1", "invalid_request_error"),
    ]
    job.collect(transport)
    before = dict(job.items())
    transport.create_error = BatchError("http_error", "boom", status=503)
    with pytest.raises(BatchError):
        job.requeue_failed(transport)
    assert job.items() == before


def test_events(tmp_path: Path) -> None:
    job = make_job(tmp_path)
    events: list[dict[str, Any]] = []
    job.on_event = events.append
    transport = FakeTransport()
    job.submit(transport)
    transport.lines["msgbatch_1"] = [simple("a1--a1", "expired"), ok("b2--a1")]
    job.collect(transport)
    job.requeue_failed(transport)
    assert [e["type"] for e in events] == ["submitted", "collected", "requeued"]
    assert [e["batch_id"] for e in events] == ["msgbatch_1", "msgbatch_1", "msgbatch_2"]
    assert set(events[0]) == {"type", "batch_id", "counts"}
    assert events[0]["counts"]["submitted"] == 2
    assert events[1]["counts"]["expired"] == 1
    assert events[1]["counts"]["succeeded"] == 1
    assert events[1]["counts"]["ignored"] == 0
    assert events[2]["counts"]["submitted"] == 1


def test_failing_event_handler_does_not_break_the_job(tmp_path: Path) -> None:
    def boom(event: dict[str, Any]) -> None:
        raise RuntimeError("handler failure")

    job = make_job(tmp_path)
    job.on_event = boom
    assert job.submit(FakeTransport()) == "msgbatch_1"
    assert BatchJob.load(tmp_path / "state.json").items() == job.items()


def test_state_is_replaced_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "state.json"
    job = make_job(tmp_path)
    calls: list[tuple[Any, Any]] = []
    real_replace = os.replace

    def spy(src: Any, dst: Any) -> None:
        calls.append((src, dst))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    job.submit(FakeTransport())
    assert len(calls) == 1
    assert Path(calls[0][1]) == path
    assert Path(calls[0][0]).parent == path.parent
    assert list(tmp_path.iterdir()) == [path]


def test_failed_replace_keeps_old_file_and_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    job = make_job(tmp_path)
    transport = FakeTransport()
    job.submit(transport)
    before = path.read_bytes()
    transport.lines["msgbatch_1"] = [ok("a1--a1"), ok("b2--a1")]

    def boom(src: Any, dst: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(BatchError) as info:
        job.collect(transport)
    assert info.value.code == "state_file"
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]
    assert set(job.items().values()) == {ItemState("submitted", 1)}


def test_load_roundtrip_keeps_content_and_resumes(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    BatchJob.create(
        path, [JobItem("a", "héllo ☃")], model="m", system="Sys ☃", max_tokens=7, max_attempts=4
    )
    loaded = BatchJob.load(path)
    assert loaded.items() == {"a": ItemState("pending", 0)}
    transport = FakeTransport()
    loaded.submit(transport)
    params = transport.created[0][0]["params"]
    assert params["messages"][0]["content"] == "héllo ☃"
    assert params["system"][0]["text"] == "Sys ☃"
    assert params["max_tokens"] == 7
    transport.lines["msgbatch_1"] = [ok("a--a1", "done")]
    resumed = BatchJob.load(path)
    assert resumed.collect(transport)["succeeded"] == 1
    assert BatchJob.load(path).items() == resumed.items()


def test_load_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def code_of(path: Path) -> str:
        with pytest.raises(BatchError) as info:
            BatchJob.load(path)
        return info.value.code

    assert code_of(tmp_path / "missing.json") == "state_file"
    assert code_of(tmp_path) == "state_file"
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert code_of(corrupt) == "state_file"
    corrupt.write_bytes(b"\xff\xfe\x00")
    assert code_of(corrupt) == "state_file"
    corrupt.write_text("[]", encoding="utf-8")
    assert code_of(corrupt) == "state_file"
    monkeypatch.setattr(job_module, "MAX_STATE_BYTES", 10)
    corrupt.write_text("x" * 100, encoding="utf-8")
    assert code_of(corrupt) == "too_large"


MUTATIONS: dict[str, Any] = {
    "schema2": lambda d: d.update(schema=2),
    "schema_bool": lambda d: d.update(schema=True),
    "model": lambda d: d.update(model=5),
    "system_empty": lambda d: d.update(system=""),
    "sha": lambda d: d.update(system_sha256="0" * 64),
    "surrogate_system": lambda d: d.update(system=chr(0xD800)),
    "max_tokens": lambda d: d.update(max_tokens=0),
    "max_attempts": lambda d: d.update(max_attempts=10),
    "batches_type": lambda d: d.update(batches={}),
    "batches_many": lambda d: d.update(batches=[{"id": "b", "collected": True}] * 65),
    "batch_entry": lambda d: d.update(batches=[5]),
    "batch_id": lambda d: d.update(batches=[{"id": "bad id", "collected": False}]),
    "batch_flag": lambda d: d.update(batches=[{"id": "msgbatch_1", "collected": "no"}]),
    "items_empty": lambda d: d.update(items={}),
    "items_type": lambda d: d.update(items=[]),
    "bad_key": lambda d: d["items"].update({"a b": d["items"]["a1"]}),
    "entry_type": lambda d: d["items"].update(a1=5),
    "status": lambda d: d["items"]["a1"].update(status="weird"),
    "attempts_negative": lambda d: d["items"]["a1"].update(attempts=-1),
    "attempts_pending": lambda d: d["items"]["a1"].update(attempts=1),
    "attempts_over": lambda d: d["items"]["a1"].update(status="submitted", attempts=4),
    "output": lambda d: d["items"]["a1"].update(output=5),
    "error": lambda d: d["items"]["a1"].update(error=5),
    "content": lambda d: d["items"]["a1"].update(user_content=""),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_load_rejects_invalid_documents(tmp_path: Path, name: str) -> None:
    make_job(tmp_path)
    path = tmp_path / "state.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    MUTATIONS[name](document)
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(BatchError) as info:
        BatchJob.load(path)
    assert info.value.code == "state_file"
    assert len(str(info.value)) <= 200


def test_api_key_never_leaks(tmp_path: Path) -> None:
    lines = [ok("a1--a1", "done"), err("b2--a1", "overloaded_error")]
    results = "\n".join(json.dumps(line) for line in lines) + "\n"
    ended = {
        "id": "msgbatch_1",
        "processing_status": "ended",
        "results_url": BASE + "/v1/messages/batches/msgbatch_1/results",
    }
    opener = FakeOpener(
        FakeResponse(json.dumps({"id": "msgbatch_1"}).encode()),
        FakeResponse(json.dumps(ended).encode()),
        FakeResponse(results.encode()),
        FakeResponse(json.dumps({"id": "msgbatch_2"}).encode()),
    )
    transport = HttpBatchTransport(api_key=SENTINEL, opener=opener)
    job = make_job(tmp_path)
    events: list[dict[str, Any]] = []
    job.on_event = events.append
    job.submit(transport)
    counts = job.collect(transport)
    assert counts["succeeded"] == 1
    assert job.requeue_failed(transport) == "msgbatch_2"
    headers = {k.lower(): v for k, v in opener.requests[0].header_items()}
    assert headers["x-api-key"] == SENTINEL
    state_text = (tmp_path / "state.json").read_text(encoding="utf-8")
    assert SENTINEL not in state_text
    assert SENTINEL not in repr(job)
    assert SENTINEL not in repr(transport)
    assert SENTINEL not in json.dumps(events)
    assert len(events) == 3


def test_api_key_not_in_http_error_messages(tmp_path: Path) -> None:
    body = json.dumps(
        {"type": "error", "error": {"type": "authentication_error", "message": SENTINEL}}
    )
    opener = FakeOpener(FakeResponse(body.encode(), status=401))
    transport = HttpBatchTransport(api_key=SENTINEL, opener=opener)
    job = make_job(tmp_path)
    with pytest.raises(BatchError) as info:
        job.submit(transport)
    assert SENTINEL not in str(info.value)
    assert SENTINEL not in repr(info.value)
    assert info.value.status == 401
    assert info.value.error_type == "authentication_error"
