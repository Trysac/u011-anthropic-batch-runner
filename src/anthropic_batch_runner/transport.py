"""Transports for the Anthropic Message Batches API."""

from __future__ import annotations

import http.client
import json
import logging
import re
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any, Protocol
from urllib.parse import urlsplit

from ._util import loads_object, sanitize_error_type
from .errors import BatchError

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
MAX_REQUESTS = 100_000
MAX_BATCH_BYTES = 256_000_000
MAX_RESPONSE_BYTES = 1_048_576
MAX_ERROR_BODY_BYTES = 65_536
BATCH_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,128}")


class BatchTransport(Protocol):
    """What a job needs from the Message Batches API."""

    def create(self, requests: list[dict[str, Any]]) -> str:
        """Create a batch and return its id."""
        ...

    def retrieve(self, batch_id: str) -> dict[str, Any]:
        """Return the batch object (with ``processing_status``)."""
        ...

    def results(self, batch: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Yield result lines of an ended batch, in any order.

        Lines that cannot be parsed as a JSON object are yielded as ``{}``.
        """
        ...


def _default_opener() -> urllib.request.OpenerDirector:
    """Build an opener without a redirect handler so credentials are never forwarded."""
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPSHandler())
    opener.add_handler(urllib.request.HTTPDefaultErrorHandler())
    opener.add_handler(urllib.request.HTTPErrorProcessor())
    return opener


def _api_error(status: int, response: Any) -> BatchError:
    """Build an error carrying only the HTTP status and the API error type."""
    try:
        raw = response.read(MAX_ERROR_BODY_BYTES)
    except (OSError, http.client.HTTPException):
        raw = b""
    data = loads_object(raw)
    error_type: str | None = None
    if data is not None:
        error = data.get("error")
        if isinstance(error, dict):
            error_type = sanitize_error_type(error.get("type"))
    message = f"HTTP {status}"
    if error_type is not None:
        message = f"{message}: {error_type}"
    return BatchError("http_error", message, status=status, error_type=error_type)


def _iter_lines(response: Any, limit: int) -> Iterator[bytes | None]:
    """Yield lines of at most ``limit`` bytes; oversize lines are drained and yield ``None``."""
    while True:
        line = response.readline(limit + 1)
        if not line:
            return
        if len(line) > limit and not line.endswith(b"\n"):
            while True:
                rest = response.readline(limit + 1)
                if not rest or rest.endswith(b"\n"):
                    break
            yield None
            continue
        yield line


class HttpBatchTransport:
    """``BatchTransport`` over HTTPS using only the standard library.

    The default opener has no redirect handler, so any redirect is an ``http_error`` and the
    API key is never forwarded. ``base_url`` must be https and ``results_url`` must share its
    scheme, host and port.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        opener: urllib.request.OpenerDirector | None = None,
        timeout: float = 60.0,
        max_line_bytes: int = 10_485_760,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise BatchError("config", "api_key must be a non-empty string")
        if not isinstance(base_url, str):
            raise BatchError("config", "base_url must be a string")
        try:
            parts = urlsplit(base_url)
            port = parts.port
        except ValueError:
            raise BatchError("config", "base_url is not a valid URL") from None
        if (
            parts.scheme != "https"
            or not parts.hostname
            or parts.username
            or parts.password
            or parts.query
            or parts.fragment
        ):
            raise BatchError("config", "base_url must be a plain https URL")
        if isinstance(timeout, bool) or not isinstance(timeout, int | float) or timeout <= 0:
            raise BatchError("config", "timeout must be a positive number")
        if isinstance(max_line_bytes, bool) or not isinstance(max_line_bytes, int):
            raise BatchError("config", "max_line_bytes must be a positive integer")
        if max_line_bytes < 1:
            raise BatchError("config", "max_line_bytes must be a positive integer")
        self._api_key = api_key
        self._base = base_url.rstrip("/")
        self._host = parts.hostname
        self._port = port or 443
        self._opener = opener if opener is not None else _default_opener()
        self._timeout = float(timeout)
        self._max_line_bytes = max_line_bytes

    def __repr__(self) -> str:
        return f"HttpBatchTransport(base_url={self._base!r})"

    def _request(self, method: str, url: str, body: bytes | None) -> Any:
        headers = {"x-api-key": self._api_key, "anthropic-version": ANTHROPIC_VERSION}
        if body is not None:
            headers["content-type"] = "application/json"
        request = urllib.request.Request(url, data=body, headers=headers, method=method)  # noqa: S310, RUF100
        try:
            response = self._opener.open(request, timeout=self._timeout)
        except urllib.error.HTTPError as exc:
            error = _api_error(exc.code, exc)
            exc.close()
            raise error from None
        except (OSError, http.client.HTTPException) as exc:
            reason = type(exc).__name__
            raise BatchError("http_error", f"request failed: {reason}") from None
        status = response.status
        if not 200 <= status < 300:
            error = _api_error(status, response)
            response.close()
            raise error
        return response

    def _get_json(self, method: str, url: str, body: bytes | None) -> dict[str, Any]:
        response = self._request(method, url, body)
        try:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        except (OSError, http.client.HTTPException):
            raise BatchError("http_error", "could not read the response") from None
        finally:
            response.close()
        if len(raw) > MAX_RESPONSE_BYTES:
            raise BatchError("too_large", "response body is too large")
        data = loads_object(raw)
        if data is None:
            raise BatchError("http_error", "response is not a JSON object")
        return data

    def create(self, requests: list[dict[str, Any]]) -> str:
        """POST the requests as one batch and return the batch id."""
        if not isinstance(requests, list) or not requests:
            raise BatchError("config", "requests must be a non-empty list")
        if len(requests) > MAX_REQUESTS:
            raise BatchError("too_large", "a batch holds at most 100000 requests")
        try:
            payload = {"requests": requests}
            document = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            body = document.encode("utf-8")
        except (TypeError, ValueError):
            raise BatchError("config", "requests cannot be serialised as JSON") from None
        if len(body) > MAX_BATCH_BYTES:
            raise BatchError("too_large", "batch body exceeds the size limit")
        data = self._get_json("POST", f"{self._base}/v1/messages/batches", body)
        batch_id = data.get("id")
        if not isinstance(batch_id, str) or BATCH_ID_RE.fullmatch(batch_id) is None:
            raise BatchError("http_error", "response has no valid batch id")
        logger.debug("created a batch with %d requests", len(requests))
        return batch_id

    def retrieve(self, batch_id: str) -> dict[str, Any]:
        """GET the batch object for ``batch_id``."""
        if not isinstance(batch_id, str) or BATCH_ID_RE.fullmatch(batch_id) is None:
            raise BatchError("config", "batch id has an invalid format")
        return self._get_json("GET", f"{self._base}/v1/messages/batches/{batch_id}", None)

    def _check_trusted(self, url: str) -> None:
        try:
            parts = urlsplit(url)
            port = parts.port
        except ValueError:
            raise BatchError("untrusted_url", "results_url is not a valid URL") from None
        if (
            parts.scheme != "https"
            or parts.hostname != self._host
            or (port or 443) != self._port
            or parts.username
            or parts.password
        ):
            raise BatchError("untrusted_url", "results_url does not match base_url")

    def results(self, batch: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Open the results stream of an ended batch.

        Validation and the request happen eagerly; lines are parsed lazily. Oversize, non-JSON
        and non-object lines are yielded as ``{}``.
        """
        url = batch.get("results_url") if isinstance(batch, dict) else None
        if not isinstance(url, str) or not url:
            raise BatchError("not_ended", "batch has no results_url")
        self._check_trusted(url)
        return self._iter_results(self._request("GET", url, None))

    def _iter_results(self, response: Any) -> Iterator[dict[str, Any]]:
        try:
            for line in _iter_lines(response, self._max_line_bytes):
                if line is None:
                    yield {}
                    continue
                stripped = line.strip()
                if not stripped:
                    continue
                yield loads_object(stripped) or {}
        except (OSError, http.client.HTTPException) as exc:
            reason = type(exc).__name__
            raise BatchError("http_error", f"results stream failed: {reason}") from None
        finally:
            response.close()
