"""Typed errors for anthropic_batch_runner."""

from __future__ import annotations

from typing import Literal

ErrorCode = Literal[
    "invalid_item_key",
    "http_error",
    "untrusted_url",
    "not_ended",
    "state_file",
    "too_large",
    "config",
]

MAX_MESSAGE_CHARS = 200


class BatchError(Exception):
    """Base error of the library.

    Attributes:
        code: Stable machine-readable error code.
        status: HTTP status for ``http_error``, otherwise ``None``.
        error_type: API error type (for example ``rate_limit_error``) when known.
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        status: int | None = None,
        error_type: str | None = None,
    ) -> None:
        if len(message) > MAX_MESSAGE_CHARS:
            limit = MAX_MESSAGE_CHARS - 3
            message = message[:limit] + "..."
        super().__init__(message)
        self.code: ErrorCode = code
        self.status = status
        self.error_type = error_type
