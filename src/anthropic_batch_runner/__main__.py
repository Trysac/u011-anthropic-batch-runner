"""Command line interface: ``python -m anthropic_batch_runner``."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

from ._util import loads_object
from .errors import BatchError
from .job import BatchJob, JobItem
from .transport import BatchTransport, HttpBatchTransport

API_KEY_ENV = "ANTHROPIC_API_KEY"
MAX_ITEMS_FILE_BYTES = 64 * 1024 * 1024
MAX_SYSTEM_FILE_BYTES = 1024 * 1024
_USAGE_CODES = frozenset({"invalid_item_key", "state_file", "config"})


def _make_transport(api_key: str) -> BatchTransport:
    return HttpBatchTransport(api_key=api_key)


def _read_capped(path: Path, limit: int) -> bytes:
    with path.open("rb") as handle:
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise BatchError("config", "input file is too large")
    return raw


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        raise BatchError("config", "input file is not valid UTF-8") from None


def _read_items(path: Path) -> list[JobItem]:
    text = _decode(_read_capped(path, MAX_ITEMS_FILE_BYTES))
    items: list[JobItem] = []
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        obj = loads_object(line)
        if obj is None:
            raise BatchError("config", f"items line {number} is not a JSON object")
        key = obj.get("key")
        content = obj.get("user_content")
        if not isinstance(key, str) or not isinstance(content, str):
            raise BatchError("config", f"items line {number} needs string key and user_content")
        items.append(JobItem(key, content))
    return items


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anthropic-batch-runner",
        description="Run jobs on the Anthropic Message Batches API.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("submit", help="create a job and submit its first batch")
    submit.add_argument("--state", required=True, help="state file to create")
    submit.add_argument("--items", required=True, help="JSONL file with key and user_content")
    submit.add_argument("--model", required=True)
    submit.add_argument("--system", required=True, help="file holding the system prompt")
    submit.add_argument("--max-tokens", required=True, type=int)
    submit.add_argument("--max-attempts", type=int, default=3)
    collect = commands.add_parser("collect", help="collect the results of ended batches")
    collect.add_argument("--state", required=True)
    requeue = commands.add_parser("requeue", help="resubmit retryable items")
    requeue.add_argument("--state", required=True)
    return parser


def _run(args: argparse.Namespace, api_key: str) -> int:
    transport = _make_transport(api_key)
    if args.command == "submit":
        items = _read_items(Path(args.items))
        system = _decode(_read_capped(Path(args.system), MAX_SYSTEM_FILE_BYTES))
        job = BatchJob.create(
            args.state,
            items,
            model=args.model,
            system=system,
            max_tokens=args.max_tokens,
            max_attempts=args.max_attempts,
        )
        try:
            batch_id = job.submit(transport)
        except BatchError:
            with contextlib.suppress(OSError):
                Path(args.state).unlink()
            raise
        print(batch_id)
    elif args.command == "collect":
        counts = BatchJob.load(args.state).collect(transport)
        print(json.dumps(counts, sort_keys=True))
    else:
        requeued = BatchJob.load(args.state).requeue_failed(transport)
        print(requeued if requeued is not None else "nothing to requeue")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns 0 on success, 1 for a job error, 2 for usage, IO or parse errors."""
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, int):
            return code
        return 0 if code is None else 2
    api_key = os.environ.get(API_KEY_ENV, "")
    if not api_key.strip():
        print(f"error: the {API_KEY_ENV} environment variable is not set", file=sys.stderr)
        return 2
    try:
        return _run(args, api_key)
    except BatchError as exc:
        print(f"error [{exc.code}]: {exc}", file=sys.stderr)
        return 2 if exc.code in _USAGE_CODES else 1
    except OSError as exc:
        print(f"error: {type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
