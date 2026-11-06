"""Test doubles shared by the test modules."""

from __future__ import annotations

import io
from collections.abc import Iterator
from typing import Any

BASE = "https://api.anthropic.com"


class FakeResponse(io.BytesIO):
    """In-memory HTTP response."""

    def __init__(self, body: bytes = b"", status: int = 200) -> None:
        super().__init__(body)
        self.status = status


class FakeOpener:
    """Opener double replaying queued responses or raising queued exceptions."""

    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.requests: list[Any] = []
        self.timeouts: list[float | None] = []

    def open(self, request: Any, data: Any = None, timeout: float | None = None) -> Any:
        self.requests.append(request)
        self.timeouts.append(timeout)
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeTransport:
    """BatchTransport double with scripted batches."""

    def __init__(self) -> None:
        self.created: list[list[dict[str, Any]]] = []
        self.status = "ended"
        self.lines: dict[str, list[Any]] = {}
        self.create_error: Exception | None = None
        self.retrieved: list[str] = []

    def create(self, requests: list[dict[str, Any]]) -> str:
        if self.create_error is not None:
            raise self.create_error
        self.created.append(requests)
        return f"msgbatch_{len(self.created)}"

    def retrieve(self, batch_id: str) -> dict[str, Any]:
        self.retrieved.append(batch_id)
        return {
            "id": batch_id,
            "processing_status": self.status,
            "results_url": f"{BASE}/results/{batch_id}",
        }

    def results(self, batch: dict[str, Any]) -> Iterator[Any]:
        return iter(self.lines.get(batch["id"], []))


def ok(custom_id: str, text: str = "out") -> dict[str, Any]:
    """A succeeded result line."""
    message = {"content": [{"type": "text", "text": text}]}
    return {"custom_id": custom_id, "result": {"type": "succeeded", "message": message}}


def err(custom_id: str, error_type: str) -> dict[str, Any]:
    """An errored result line in the API's nested shape."""
    inner = {"type": error_type, "message": "details that must not be stored"}
    result = {"type": "errored", "error": {"type": "error", "error": inner}}
    return {"custom_id": custom_id, "result": result}


def simple(custom_id: str, kind: str) -> dict[str, Any]:
    """A canceled or expired result line."""
    return {"custom_id": custom_id, "result": {"type": kind}}
