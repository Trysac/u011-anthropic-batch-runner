"""Tests for HttpBatchTransport."""

from __future__ import annotations

import email.message
import http.client
import io
import json
import urllib.error
import urllib.request
from typing import Any

import pytest
from helpers import BASE, FakeOpener, FakeResponse

from anthropic_batch_runner import BatchError, HttpBatchTransport
from anthropic_batch_runner import transport as transport_module

SENTINEL = "KEYSENTINEL-0123456789"
RESULTS_URL = BASE + "/v1/messages/batches/msgbatch_01/results"


def make(*responses: object, **kwargs: Any) -> tuple[HttpBatchTransport, FakeOpener]:
    opener = FakeOpener(*responses)
    return HttpBatchTransport(api_key=SENTINEL, opener=opener, **kwargs), opener


def http_error(code: int, body: bytes = b"") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(BASE, code, "msg", email.message.Message(), io.BytesIO(body))


def test_create_sends_one_post_with_headers() -> None:
    transport, opener = make(FakeResponse(b'{"id": "msgbatch_01"}'))
    requests = [{"custom_id": "a1--a1", "params": {"model": "m"}}]
    assert transport.create(requests) == "msgbatch_01"
    (request,) = opener.requests
    assert request.full_url == "https://api.anthropic.com/v1/messages/batches"
    assert request.get_method() == "POST"
    headers = {k.lower(): v for k, v in request.header_items()}
    assert headers["x-api-key"] == SENTINEL
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["content-type"] == "application/json"
    assert json.loads(request.data) == {"requests": requests}
    assert opener.timeouts == [60.0]


def test_create_redirect_status_is_an_error_without_second_request() -> None:
    transport, opener = make(FakeResponse(b"", status=302), FakeResponse(b'{"id": "x"}'))
    with pytest.raises(BatchError) as info:
        transport.create([{"custom_id": "a1--a1"}])
    assert info.value.code == "http_error"
    assert info.value.status == 302
    assert len(opener.requests) == 1


def test_create_redirect_raised_by_opener_is_an_error() -> None:
    transport, opener = make(http_error(302), FakeResponse(b'{"id": "x"}'))
    with pytest.raises(BatchError) as info:
        transport.create([{"custom_id": "a1--a1"}])
    assert info.value.code == "http_error"
    assert info.value.status == 302
    assert len(opener.requests) == 1


def test_http_errors_carry_status_and_api_error_type_only() -> None:
    body = b'{"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}'
    transport, _ = make(FakeResponse(body, status=429), http_error(500, body))
    for status in (429, 500):
        with pytest.raises(BatchError) as info:
            transport.create([{"custom_id": "a1--a1"}])
        assert info.value.status == status
        assert info.value.error_type == "rate_limit_error"
        assert "slow down" not in str(info.value)


@pytest.mark.parametrize(
    "body",
    [b"<html>", b"[]", b'{"error": 5}', b'{"error": {"type": "bad type!"}}', b""],
)
def test_http_error_with_unusable_body(body: bytes) -> None:
    transport, _ = make(FakeResponse(body, status=400))
    with pytest.raises(BatchError) as info:
        transport.create([{"custom_id": "a1--a1"}])
    assert info.value.status == 400
    assert info.value.error_type is None
    assert str(info.value) == "HTTP 400"


def test_http_error_body_read_failure() -> None:
    class Broken(FakeResponse):
        def read(self, size: int | None = -1) -> bytes:
            raise OSError("reset")

    transport, _ = make(Broken(b"", status=502))
    with pytest.raises(BatchError) as info:
        transport.create([{"custom_id": "a1--a1"}])
    assert info.value.status == 502


@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.URLError("unreachable " + SENTINEL),
        TimeoutError("slow " + SENTINEL),
        http.client.HTTPException("protocol " + SENTINEL),
    ],
)
def test_network_failures_are_http_errors(failure: BaseException) -> None:
    transport, _ = make(failure)
    with pytest.raises(BatchError) as info:
        transport.create([{"custom_id": "a1--a1"}])
    assert info.value.code == "http_error"
    assert info.value.status is None
    assert SENTINEL not in str(info.value)


@pytest.mark.parametrize(
    "body",
    [b"not json", b"[]", b"{}", b'{"id": 5}', b'{"id": "bad id"}', b"[" * 100_000],
)
def test_create_rejects_malformed_responses(body: bytes) -> None:
    transport, _ = make(FakeResponse(body))
    with pytest.raises(BatchError) as info:
        transport.create([{"custom_id": "a1--a1"}])
    assert info.value.code == "http_error"


def test_response_read_failure_and_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    class Broken(FakeResponse):
        def read(self, size: int | None = -1) -> bytes:
            raise http.client.IncompleteRead(b"")

    transport, _ = make(Broken(b""))
    with pytest.raises(BatchError) as info:
        transport.create([{"custom_id": "a1--a1"}])
    assert info.value.code == "http_error"
    monkeypatch.setattr(transport_module, "MAX_RESPONSE_BYTES", 5)
    transport, _ = make(FakeResponse(b'{"id": "msgbatch_01"}'))
    with pytest.raises(BatchError) as info:
        transport.create([{"custom_id": "a1--a1"}])
    assert info.value.code == "too_large"


def test_create_validates_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, opener = make()
    for bad in ([], "x", [{"x": object()}], [{"x": chr(0xD800)}]):
        with pytest.raises(BatchError) as info:
            transport.create(bad)  # type: ignore[arg-type]
        assert info.value.code == "config"
    monkeypatch.setattr(transport_module, "MAX_REQUESTS", 1)
    with pytest.raises(BatchError) as info:
        transport.create([{"a": 1}, {"b": 2}])
    assert info.value.code == "too_large"
    monkeypatch.setattr(transport_module, "MAX_REQUESTS", 100_000)
    monkeypatch.setattr(transport_module, "MAX_BATCH_BYTES", 10)
    with pytest.raises(BatchError) as info:
        transport.create([{"custom_id": "a1--a1"}])
    assert info.value.code == "too_large"
    assert opener.requests == []


def test_retrieve() -> None:
    body = b'{"id": "msgbatch_01", "processing_status": "ended"}'
    transport, opener = make(FakeResponse(body))
    assert transport.retrieve("msgbatch_01")["processing_status"] == "ended"
    (request,) = opener.requests
    assert request.full_url == BASE + "/v1/messages/batches/msgbatch_01"
    assert request.get_method() == "GET"
    assert request.data is None
    headers = {k.lower(): v for k, v in request.header_items()}
    assert "content-type" not in headers
    assert headers["x-api-key"] == SENTINEL


@pytest.mark.parametrize("batch_id", ["", "a/b", "../x", "x" * 129, "a b", 5])
def test_retrieve_rejects_bad_ids(batch_id: Any) -> None:
    transport, opener = make()
    with pytest.raises(BatchError) as info:
        transport.retrieve(batch_id)
    assert info.value.code == "config"
    assert opener.requests == []


@pytest.mark.parametrize(
    "url",
    [
        "http://api.anthropic.com/r",
        "https://evil.example.test/r",
        "https://api.anthropic.com:8443/r",
        "https://user@api.anthropic.com/r",
        "https://user:pw@api.anthropic.com/r",
        "https://api.anthropic.com:bad/r",
        "https://api.anthropic.com.evil.test/r",
        "https://evil.test/api.anthropic.com",
        "/relative/path",
    ],
)
def test_results_url_must_share_scheme_and_host(url: str) -> None:
    transport, opener = make()
    with pytest.raises(BatchError) as info:
        transport.results({"results_url": url})
    assert info.value.code == "untrusted_url"
    assert opener.requests == []


@pytest.mark.parametrize("batch", [{}, {"results_url": None}, {"results_url": ""}, [], None])
def test_results_without_url(batch: Any) -> None:
    transport, opener = make()
    with pytest.raises(BatchError) as info:
        transport.results(batch)
    assert info.value.code == "not_ended"
    assert opener.requests == []


def test_results_parse_lines() -> None:
    good = b'{"custom_id":"a1--a1","result":{"type":"expired"}}\n'
    body = b"not json\n" + b"[1,2]\n" + b"\n" + good + b"   \n" + b"[" * 100_000 + b"\n"
    response = FakeResponse(body)
    transport, opener = make(response)
    lines = list(transport.results({"results_url": "HTTPS://API.ANTHROPIC.COM/v1/r"}))
    assert lines == [{}, {}, json.loads(good), {}]
    assert opener.requests[0].get_method() == "GET"
    assert opener.requests[0].data is None
    assert response.closed


def test_results_oversize_lines_are_drained() -> None:
    small = b'{"custom_id":"a"}\n'
    body = small + b"x" * 200 + b"\n" + small + b"y" * 300
    transport, _ = make(FakeResponse(body), max_line_bytes=64)
    lines = list(transport.results({"results_url": RESULTS_URL}))
    assert lines == [{"custom_id": "a"}, {}, {"custom_id": "a"}, {}]


def test_results_line_exactly_at_limit_is_kept() -> None:
    content = b'{"k":"' + b"a" * 10 + b'"}'
    transport, _ = make(FakeResponse(content + b"\n"), max_line_bytes=len(content))
    assert list(transport.results({"results_url": RESULTS_URL})) == [{"k": "a" * 10}]


def test_results_http_error() -> None:
    transport, _ = make(FakeResponse(b"", status=404))
    with pytest.raises(BatchError) as info:
        transport.results({"results_url": RESULTS_URL})
    assert info.value.status == 404


def test_results_stream_failure_is_wrapped_and_closed() -> None:
    class Broken(FakeResponse):
        def readline(self, size: int | None = -1) -> bytes:
            raise OSError("connection reset")

    response = Broken(b"")
    transport, _ = make(response)
    with pytest.raises(BatchError) as info:
        list(transport.results({"results_url": RESULTS_URL}))
    assert info.value.code == "http_error"
    assert response.closed


def test_custom_base_url_and_trailing_slash() -> None:
    transport = HttpBatchTransport(
        api_key=SENTINEL,
        base_url="https://proxy.example.test:8443/",
        opener=FakeOpener(FakeResponse(b'{"id": "msgbatch_01"}'), FakeResponse(b"")),
        timeout=5,
    )
    assert transport.create([{"a": 1}]) == "msgbatch_01"
    url = transport._opener.requests[0].full_url  # type: ignore[attr-defined]
    assert url == "https://proxy.example.test:8443/v1/messages/batches"
    with pytest.raises(BatchError):
        transport.results({"results_url": "https://proxy.example.test/r"})
    assert list(transport.results({"results_url": "https://proxy.example.test:8443/r"})) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"api_key": ""},
        {"api_key": "   "},
        {"api_key": 5},
        {"base_url": "http://api.anthropic.com"},
        {"base_url": "https://"},
        {"base_url": 5},
        {"base_url": "https://user:pw@api.anthropic.com"},
        {"base_url": "https://api.anthropic.com/?x=1"},
        {"base_url": "https://api.anthropic.com/#f"},
        {"base_url": "https://api.anthropic.com:99999"},
        {"timeout": 0},
        {"timeout": True},
        {"timeout": "1"},
        {"max_line_bytes": 0},
        {"max_line_bytes": True},
    ],
)
def test_constructor_validation(kwargs: dict[str, Any]) -> None:
    params: dict[str, Any] = {"api_key": SENTINEL}
    params.update(kwargs)
    with pytest.raises(BatchError) as info:
        HttpBatchTransport(**params)
    assert info.value.code == "config"
    assert SENTINEL not in str(info.value)


def test_default_opener_has_no_redirect_handler() -> None:
    transport = HttpBatchTransport(api_key=SENTINEL)
    opener = transport._opener
    assert isinstance(opener, urllib.request.OpenerDirector)
    assert not any(isinstance(h, urllib.request.HTTPRedirectHandler) for h in opener.handlers)
    assert any(isinstance(h, urllib.request.HTTPSHandler) for h in opener.handlers)
    assert SENTINEL not in repr(transport)
    assert "api.anthropic.com" in repr(transport)
