"""Small shared helpers (not part of the public API)."""

from __future__ import annotations

import json
import re
from typing import Any

_ERROR_TYPE_RE = re.compile(r"[A-Za-z0-9_.-]{1,64}")


def sanitize_error_type(value: object) -> str | None:
    """Return ``value`` if it looks like an API error type, otherwise ``None``."""
    if isinstance(value, str) and _ERROR_TYPE_RE.fullmatch(value):
        return value
    return None


def loads_object(raw: bytes | str) -> dict[str, Any] | None:
    """Parse ``raw`` as a JSON object; return ``None`` for anything else.

    The C JSON decoder enforces the interpreter recursion limit, which acts as the nesting
    depth limit; ``RecursionError`` is mapped to ``None``.
    """
    try:
        data = json.loads(raw)
    except (ValueError, RecursionError):
        return None
    if isinstance(data, dict):
        return data
    return None
