"""Resumable per-item job state for the Message Batches API."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from itertools import islice
from pathlib import Path
from typing import Any, Literal, NamedTuple, TypeGuard

from ._util import loads_object, sanitize_error_type
from .errors import BatchError
from .transport import BATCH_ID_RE, MAX_REQUESTS, BatchTransport

logger = logging.getLogger(__name__)

Status = Literal[
    "pending",
    "submitted",
    "succeeded",
    "errored",
    "canceled",
    "expired",
    "invalid_output",
    "failed",
    "exhausted",
]
STATUSES: tuple[Status, ...] = (
    "pending",
    "submitted",
    "succeeded",
    "errored",
    "canceled",
    "expired",
    "invalid_output",
    "failed",
    "exhausted",
)

SCHEMA_VERSION = 1
MAX_ATTEMPTS_LIMIT = 9
MAX_OUTPUT_CHARS = 100_000
MAX_STATE_BYTES = 64 * 1024 * 1024
MAX_BATCHES = 64
_KEY_RE = re.compile(r"[A-Za-z0-9_-]{1,59}")
_CUSTOM_ID_RE = re.compile(r"([A-Za-z0-9_-]{1,59})--a([1-9])")
_RETRYABLE = frozenset({"expired", "canceled", "invalid_output", "errored"})


@dataclass(frozen=True)
class JobItem:
    """One unit of work: a key unique within the job and the user message text."""

    key: str
    user_content: str


@dataclass(frozen=True)
class ItemState:
    """State of one item: its status, attempt number, output and API error type."""

    status: Status
    attempts: int
    output: str | None = None
    error: str | None = None


class _Batch(NamedTuple):
    id: str
    collected: bool


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _encodable(text: str) -> bool:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _check_params(model: object, system: object, max_tokens: object) -> None:
    if not isinstance(model, str) or not model.strip() or not _encodable(model):
        raise BatchError("config", "model must be a non-empty string")
    if not isinstance(system, str) or not system.strip() or not _encodable(system):
        raise BatchError("config", "system must be a non-empty string")
    if not _is_int(max_tokens) or max_tokens < 1:
        raise BatchError("config", "max_tokens must be a positive integer")


def _check_item(item: object) -> JobItem:
    if not isinstance(item, JobItem):
        raise BatchError("config", "items must be JobItem instances")
    if not isinstance(item.key, str) or _KEY_RE.fullmatch(item.key) is None:
        raise BatchError("invalid_item_key", "item key must match ^[A-Za-z0-9_-]{1,59}$")
    content = item.user_content
    if not isinstance(content, str) or not content.strip() or not _encodable(content):
        raise BatchError("config", "item user_content must be a non-empty string")
    return item


def _collect_items(items: Iterable[JobItem]) -> list[JobItem]:
    batch = list(islice(items, MAX_REQUESTS + 1))
    if len(batch) > MAX_REQUESTS:
        raise BatchError("too_large", "a job holds at most 100000 items")
    if not batch:
        raise BatchError("config", "at least one item is required")
    seen: set[str] = set()
    for item in batch:
        checked = _check_item(item)
        if checked.key in seen:
            raise BatchError("config", "item keys must be unique")
        seen.add(checked.key)
    return batch


def _system_blocks(system: str, cache_system: bool) -> list[dict[str, Any]]:
    block: dict[str, Any] = {"type": "text", "text": system}
    if cache_system:
        block["cache_control"] = {"type": "ephemeral"}
    return [block]


def build_requests(
    items: Iterable[JobItem],
    *,
    model: str,
    system: str,
    max_tokens: int,
    cache_system: bool = True,
    attempt: int = 1,
) -> list[dict[str, Any]]:
    """Build Message Batches requests with ``custom_id`` ``<key>--a<attempt>``.

    The shared system prompt is sent as one block marked ``cache_control: ephemeral`` unless
    ``cache_system`` is false. Caching in batches is best effort and only applies when the
    prompt reaches the model's minimum cacheable length.

    Raises:
        BatchError: ``invalid_item_key``, ``config`` or ``too_large``.
    """
    _check_params(model, system, max_tokens)
    if not _is_int(attempt) or not 1 <= attempt <= MAX_ATTEMPTS_LIMIT:
        raise BatchError("config", "attempt must be an integer from 1 to 9")
    requests: list[dict[str, Any]] = []
    for item in _collect_items(items):
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": _system_blocks(system, cache_system),
            "messages": [{"role": "user", "content": item.user_content}],
        }
        requests.append({"custom_id": f"{item.key}--a{attempt}", "params": params})
    return requests


def _count(items: Mapping[str, ItemState]) -> dict[str, int]:
    counts: dict[str, int] = dict.fromkeys(STATUSES, 0)
    for state in items.values():
        counts[state.status] += 1
    return counts


def _checked_batch_id(batch_id: object) -> str:
    if not isinstance(batch_id, str) or BATCH_ID_RE.fullmatch(batch_id) is None:
        raise BatchError("http_error", "transport returned an invalid batch id")
    return batch_id


def _extract_text(message: object) -> str | None:
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _error_type(result: dict[str, Any]) -> str:
    error = result.get("error")
    if not isinstance(error, dict):
        return "unknown_error"
    inner = error.get("error")
    source = inner if isinstance(inner, dict) else error
    return sanitize_error_type(source.get("type")) or "unknown_error"


def _succeeded_state(
    state: ItemState, result: dict[str, Any], validator: Callable[[str], object] | None
) -> ItemState:
    attempts = state.attempts
    text = _extract_text(result.get("message"))
    if text is None:
        return ItemState("errored", attempts, None, "malformed_result")
    if len(text) > MAX_OUTPUT_CHARS:
        return ItemState("invalid_output", attempts, None, "output_too_large")
    if validator is not None:
        try:
            validator(text)
        except Exception as exc:
            return ItemState("invalid_output", attempts, text, type(exc).__name__[:128])
    return ItemState("succeeded", attempts, text, None)


def _result_state(
    state: ItemState, result: object, validator: Callable[[str], object] | None
) -> ItemState:
    attempts = state.attempts
    if not isinstance(result, dict):
        return ItemState("errored", attempts, None, "malformed_result")
    kind = result.get("type")
    if kind == "succeeded":
        return _succeeded_state(state, result, validator)
    if kind == "errored":
        return ItemState("errored", attempts, None, _error_type(result))
    if kind == "canceled":
        return ItemState("canceled", attempts)
    if kind == "expired":
        return ItemState("expired", attempts)
    return ItemState("errored", attempts, None, "malformed_result")


def _apply_line(
    items: dict[str, ItemState], line: object, validator: Callable[[str], object] | None
) -> bool:
    """Apply one result line; return False when the line must be ignored.

    Because only ``submitted`` items accept a result and the status changes on the first
    accepted line, a second line for the same custom_id is ignored.
    """
    if not isinstance(line, dict):
        return False
    custom_id = line.get("custom_id")
    if not isinstance(custom_id, str) or len(custom_id) > 64:
        return False
    match = _CUSTOM_ID_RE.fullmatch(custom_id)
    if match is None:
        return False
    key = match.group(1)
    state = items.get(key)
    if state is None or state.status != "submitted" or state.attempts != int(match.group(2)):
        return False
    items[key] = _result_state(state, line.get("result"), validator)
    return True


def _bad(reason: str) -> BatchError:
    return BatchError("state_file", f"invalid state file: {reason}")


def _fsync_directory(directory: Path) -> None:
    if os.name == "posix":
        with contextlib.suppress(OSError):
            fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)


def _write_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` through an fsynced temp file and ``os.replace``."""
    if len(data) > MAX_STATE_BYTES:
        raise BatchError("too_large", "state file would exceed the size limit")
    tmp_name: str | None = None
    try:
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except OSError as exc:
        if tmp_name is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
        reason = type(exc).__name__
        raise BatchError("state_file", f"cannot write state file: {reason}") from None
    _fsync_directory(path.parent)


def _parse_batch(entry: object) -> _Batch:
    if not isinstance(entry, dict):
        raise _bad("batch entry")
    batch_id = entry.get("id")
    collected = entry.get("collected")
    if not isinstance(batch_id, str) or BATCH_ID_RE.fullmatch(batch_id) is None:
        raise _bad("batch id")
    if not isinstance(collected, bool):
        raise _bad("batch collected flag")
    return _Batch(batch_id, collected)


def _parse_item(key: str, entry: object, max_attempts: int) -> tuple[ItemState, str]:
    if _KEY_RE.fullmatch(key) is None or not isinstance(entry, dict):
        raise _bad("item entry")
    status = entry.get("status")
    attempts = entry.get("attempts")
    output = entry.get("output")
    error = entry.get("error")
    content = entry.get("user_content")
    if status not in STATUSES:
        raise _bad("item status")
    if not _is_int(attempts) or not 0 <= attempts <= max_attempts:
        raise _bad("item attempts")
    if (status == "pending") != (attempts == 0):
        raise _bad("item attempts do not match status")
    if output is not None and not isinstance(output, str):
        raise _bad("item output")
    if error is not None and not isinstance(error, str):
        raise _bad("item error")
    if not isinstance(content, str) or not content:
        raise _bad("item content")
    return ItemState(status, attempts, output, error), content


class BatchJob:
    """A resumable job: per-item state kept in an atomically replaced local file.

    Use :meth:`create` for a new job and :meth:`load` to resume one. The state file never
    holds the API key. ``on_event`` may be set to a callable receiving dicts with the keys
    ``type`` (``submitted``, ``collected`` or ``requeued``), ``batch_id`` and ``counts``.
    """

    on_event: Callable[[dict[str, Any]], None] | None

    def __init__(
        self,
        *,
        path: Path,
        model: str,
        system: str,
        max_tokens: int,
        max_attempts: int,
        items: dict[str, ItemState],
        contents: dict[str, str],
        batches: list[_Batch],
    ) -> None:
        self._path = path
        self._model = model
        self._system = system
        self._max_tokens = max_tokens
        self._max_attempts = max_attempts
        self._items = items
        self._contents = contents
        self._batches = batches
        self.on_event = None

    def __repr__(self) -> str:
        return (
            f"BatchJob(path={str(self._path)!r}, items={len(self._items)}, "
            f"batches={len(self._batches)})"
        )

    @classmethod
    def create(
        cls,
        state_path: str | os.PathLike[str],
        items: Iterable[JobItem],
        *,
        model: str,
        system: str,
        max_tokens: int,
        max_attempts: int = 3,
    ) -> BatchJob:
        """Create a job with every item ``pending`` and write its state file.

        Raises:
            BatchError: ``config``, ``invalid_item_key``, ``too_large`` or ``state_file``
                (including when the state file already exists).
        """
        _check_params(model, system, max_tokens)
        if not _is_int(max_attempts) or not 1 <= max_attempts <= MAX_ATTEMPTS_LIMIT:
            raise BatchError("config", "max_attempts must be an integer from 1 to 9")
        batch = _collect_items(items)
        path = Path(state_path)
        if path.exists():
            raise BatchError("state_file", "state file already exists")
        job = cls(
            path=path,
            model=model,
            system=system,
            max_tokens=max_tokens,
            max_attempts=max_attempts,
            items={item.key: ItemState("pending", 0) for item in batch},
            contents={item.key: item.user_content for item in batch},
            batches=[],
        )
        job._save()
        return job

    @classmethod
    def load(cls, state_path: str | os.PathLike[str]) -> BatchJob:
        """Load a job from its state file.

        Raises:
            BatchError: ``state_file`` for unreadable, corrupt or unsupported files and
                ``too_large`` above 64 MiB.
        """
        path = Path(state_path)
        try:
            with path.open("rb") as handle:
                raw = handle.read(MAX_STATE_BYTES + 1)
        except OSError as exc:
            reason = type(exc).__name__
            raise BatchError("state_file", f"cannot read state file: {reason}") from None
        if len(raw) > MAX_STATE_BYTES:
            raise BatchError("too_large", "state file exceeds the size limit")
        document = loads_object(raw)
        if document is None:
            raise _bad("not a JSON object")
        return cls._from_document(path, document)

    @classmethod
    def _from_document(cls, path: Path, document: dict[str, Any]) -> BatchJob:
        schema = document.get("schema")
        if not _is_int(schema) or schema != SCHEMA_VERSION:
            raise _bad("unsupported schema version")
        model = document.get("model")
        system = document.get("system")
        digest = document.get("system_sha256")
        max_tokens = document.get("max_tokens")
        max_attempts = document.get("max_attempts")
        if not (isinstance(model, str) and model and isinstance(system, str) and system):
            raise _bad("model or system")
        if not _is_int(max_tokens) or max_tokens < 1:
            raise _bad("max_tokens")
        if not _is_int(max_attempts) or not 1 <= max_attempts <= MAX_ATTEMPTS_LIMIT:
            raise _bad("max_attempts")
        if not _encodable(system) or _sha256(system) != digest:
            raise _bad("system digest")
        raw_batches = document.get("batches")
        raw_items = document.get("items")
        if not isinstance(raw_batches, list) or len(raw_batches) > MAX_BATCHES:
            raise _bad("batches")
        if not isinstance(raw_items, dict) or not 1 <= len(raw_items) <= MAX_REQUESTS:
            raise _bad("items")
        batches = [_parse_batch(entry) for entry in raw_batches]
        items: dict[str, ItemState] = {}
        contents: dict[str, str] = {}
        for key, entry in raw_items.items():
            items[key], contents[key] = _parse_item(key, entry, max_attempts)
        return cls(
            path=path,
            model=model,
            system=system,
            max_tokens=max_tokens,
            max_attempts=max_attempts,
            items=items,
            contents=contents,
            batches=batches,
        )

    def _save(self) -> None:
        document = {
            "schema": SCHEMA_VERSION,
            "model": self._model,
            "system_sha256": _sha256(self._system),
            "system": self._system,
            "max_tokens": self._max_tokens,
            "max_attempts": self._max_attempts,
            "batches": [{"id": b.id, "collected": b.collected} for b in self._batches],
            "items": {
                key: {
                    "status": state.status,
                    "attempts": state.attempts,
                    "output": state.output,
                    "error": state.error,
                    "user_content": self._contents[key],
                }
                for key, state in self._items.items()
            },
        }
        try:
            data = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError):
            raise BatchError("state_file", "state cannot be serialised") from None
        _write_atomic(self._path, data)

    def _commit(self, items: dict[str, ItemState], batches: list[_Batch]) -> None:
        """Install new state and persist it; restore the old state if persisting fails."""
        previous = (self._items, self._batches)
        self._items, self._batches = items, batches
        try:
            self._save()
        except BatchError:
            self._items, self._batches = previous
            raise

    def _emit(self, kind: str, batch_id: str, counts: dict[str, int]) -> None:
        logger.debug("%s batch with counts %s", kind, counts)
        handler = self.on_event
        if handler is None:
            return
        try:
            handler({"type": kind, "batch_id": batch_id, "counts": dict(counts)})
        except Exception as exc:
            logger.warning("on_event handler failed: %s", type(exc).__name__)

    def _requests_for(self, entries: list[tuple[str, int]]) -> list[dict[str, Any]]:
        requests: list[dict[str, Any]] = []
        for attempt in sorted({number for _, number in entries}):
            group = [JobItem(key, self._contents[key]) for key, n in entries if n == attempt]
            requests.extend(
                build_requests(
                    group,
                    model=self._model,
                    system=self._system,
                    max_tokens=self._max_tokens,
                    attempt=attempt,
                )
            )
        return requests

    def items(self) -> Mapping[str, ItemState]:
        """Return a snapshot of the per-item states."""
        return dict(self._items)

    def submit(self, transport: BatchTransport) -> str:
        """Submit every ``pending`` item as one batch (attempt 1) and return the batch id."""
        pending = [(key, 1) for key, state in self._items.items() if state.status == "pending"]
        if not pending:
            raise BatchError("config", "no pending items to submit")
        batch_id = _checked_batch_id(transport.create(self._requests_for(pending)))
        items = dict(self._items)
        for key, attempt in pending:
            items[key] = ItemState("submitted", attempt)
        self._commit(items, [*self._batches, _Batch(batch_id, False)])
        self._emit("submitted", batch_id, _count(self._items))
        return batch_id

    def collect(
        self,
        transport: BatchTransport,
        validator: Callable[[str], object] | None = None,
    ) -> dict[str, int]:
        """Collect results of every uncollected batch that has ended.

        Returns the count per status plus ``ignored`` (result lines that were not applied).
        If any uncollected batch has not ended, ``not_ended`` is raised and nothing changes.
        A validator that raises turns the item into ``invalid_output`` whose error is the
        exception type name. State changes only after all results were read.
        """
        pending = [batch for batch in self._batches if not batch.collected]
        infos: list[dict[str, Any]] = []
        for batch in pending:
            info = transport.retrieve(batch.id)
            if not isinstance(info, dict) or info.get("processing_status") != "ended":
                raise BatchError("not_ended", "batch has not ended yet")
            infos.append(info)
        items = dict(self._items)
        ignored_total = 0
        snapshots: list[tuple[str, dict[str, int]]] = []
        for batch, info in zip(pending, infos, strict=True):
            ignored = 0
            for line in transport.results(info):
                if not _apply_line(items, line, validator):
                    ignored += 1
            ignored_total += ignored
            counts = _count(items)
            counts["ignored"] = ignored
            snapshots.append((batch.id, counts))
        if pending:
            done = {batch.id for batch in pending}
            batches = [b._replace(collected=True) if b.id in done else b for b in self._batches]
            self._commit(items, batches)
        for batch_id, counts in snapshots:
            self._emit("collected", batch_id, counts)
        result = _count(self._items)
        result["ignored"] = ignored_total
        return result

    def requeue_failed(self, transport: BatchTransport) -> str | None:
        """Resubmit retryable items as a new batch and return its id, or ``None``.

        Expired, canceled, invalid_output and errored items are retried with ``attempt + 1``
        unless their error type is ``invalid_request_error`` (they become ``failed``) or they
        reached ``max_attempts`` (they become ``exhausted``).
        """
        updates: dict[str, ItemState] = {}
        retry: list[tuple[str, int]] = []
        for key, state in self._items.items():
            if state.status not in _RETRYABLE:
                continue
            if state.error == "invalid_request_error":
                updates[key] = replace(state, status="failed")
            elif state.attempts >= self._max_attempts:
                updates[key] = replace(state, status="exhausted")
            else:
                retry.append((key, state.attempts + 1))
        batch_id: str | None = None
        batches = list(self._batches)
        if retry:
            batch_id = _checked_batch_id(transport.create(self._requests_for(retry)))
            batches.append(_Batch(batch_id, False))
            for key, attempt in retry:
                updates[key] = ItemState("submitted", attempt)
        if not updates:
            return None
        self._commit({**self._items, **updates}, batches)
        if batch_id is not None:
            self._emit("requeued", batch_id, _count(self._items))
        return batch_id
