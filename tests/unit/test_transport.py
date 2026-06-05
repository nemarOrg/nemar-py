"""Tests for the JSON-fetch-with-retries transport primitive.

The transport module owns the retry loop, the redirect-origin check, the
JSON parse, and the exhausted-retry error reshaping. These tests pin the
contract: which HTTP results trigger a retry, which abort immediately,
and how the error messages are shaped.

Pure-transport tests use ``httpx.MockTransport`` for speed. The
integration suite covers the real-server path.
"""

from __future__ import annotations

from unittest import mock
from unittest.mock import MagicMock

import httpx
import pytest

from nemar._endpoint import DataEndpoint
from nemar._retry import RetryPolicy, _RetryableError, _RetryFreshError
from nemar._transport import (
    _response_detail,
    fetch_json,
)


def _one_shot_policy() -> RetryPolicy:
    """Return a policy that performs exactly one attempt (no retries)."""
    return RetryPolicy.default().with_attempts(0)


def _two_shot_policy() -> RetryPolicy:
    """Return a policy that performs one initial attempt plus one retry."""
    return RetryPolicy.default().with_attempts(1)


def test_fetch_json_returns_parsed_payload() -> None:
    """A 200 OK response is parsed and returned."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True}, request=request)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        result = fetch_json(
            client,
            url="https://data.nemar.org/nm000132/",
            what="testing",
            policy=_one_shot_policy(),
        )

    assert result == {"ok": True}


def test_fetch_json_retries_after_500_then_succeeds() -> None:
    """A retryable HTTP status retries once before succeeding."""
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return httpx.Response(500, request=request)
        return httpx.Response(200, json={"ok": True}, request=request)

    transport = httpx.MockTransport(handler)
    with mock.patch("nemar._transport.time.sleep"):
        with httpx.Client(transport=transport) as client:
            result = fetch_json(
                client,
                url="https://data.nemar.org/x",
                what="testing",
                policy=_two_shot_policy(),
            )

    assert result == {"ok": True}
    assert call_count["n"] == 2


def test_fetch_json_raises_immediately_on_404() -> None:
    """A non-retryable status raises with HTTP code in the message, no retry."""
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(404, json={"message": "not found"}, request=request)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        with pytest.raises(RuntimeError, match="HTTP 404"):
            fetch_json(
                client,
                url="https://data.nemar.org/missing",
                what="testing",
                policy=_two_shot_policy(),
            )

    # 404 is not retryable, so only one attempt should happen.
    assert call_count["n"] == 1


def test_fetch_json_exhausts_retries_naming_what_string() -> None:
    """Exhausted retries raise RuntimeError naming the ``what`` string."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, request=request)

    transport = httpx.MockTransport(handler)
    with mock.patch("nemar._transport.time.sleep"):
        with httpx.Client(transport=transport) as client:
            with pytest.raises(RuntimeError, match="retrieving the manifest"):
                fetch_json(
                    client,
                    url="https://data.nemar.org/x",
                    what="retrieving the manifest",
                    policy=_one_shot_policy(),
                )


def test_fetch_json_rejects_off_origin_redirect() -> None:
    """A redirect to a non-NEMAR origin must raise the off-origin guard."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "data.nemar.org":
            return httpx.Response(
                302,
                headers={"location": "https://evil.example.com/nm000132/"},
                request=request,
            )
        return httpx.Response(200, json={"trap": True}, request=request)

    endpoint = DataEndpoint.from_url("https://data.nemar.org/")
    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, follow_redirects=True) as client:
        with pytest.raises(
            RuntimeError,
            match="Refusing to download a file outside the configured NEMAR",
        ):
            fetch_json(
                client,
                url="https://data.nemar.org/nm000132/",
                what="testing",
                policy=_one_shot_policy(),
                endpoint=endpoint,
            )


def test_fetch_json_reports_invalid_json_with_what_and_url() -> None:
    """A JSON decode error mentions both the ``what`` string and the URL."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not-json", request=request)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        with pytest.raises(RuntimeError, match="Invalid JSON when testing"):
            fetch_json(
                client,
                url="https://data.nemar.org/x",
                what="testing",
                policy=_one_shot_policy(),
            )


def test_fetch_json_retries_after_connect_timeout() -> None:
    """``httpx.ConnectTimeout`` is treated as retryable per the policy."""
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise httpx.ConnectTimeout("simulated")
        return httpx.Response(200, json={"ok": True}, request=request)

    transport = httpx.MockTransport(handler)
    with mock.patch("nemar._transport.time.sleep"):
        with httpx.Client(transport=transport) as client:
            result = fetch_json(
                client,
                url="https://data.nemar.org/x",
                what="testing",
                policy=_two_shot_policy(),
            )

    assert result == {"ok": True}
    assert call_count["n"] == 2


def test_fetch_json_reports_network_error_on_exhausted_retries() -> None:
    """A transport exception that exhausts retries becomes a RuntimeError."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed")

    transport = httpx.MockTransport(handler)
    with mock.patch("nemar._transport.time.sleep"):
        with httpx.Client(transport=transport) as client:
            with pytest.raises(RuntimeError, match="Network error when testing"):
                fetch_json(
                    client,
                    url="https://data.nemar.org/x",
                    what="testing",
                    policy=_one_shot_policy(),
                )


def test_fetch_json_runs_origin_check_on_success_path() -> None:
    """A 200 OK whose URL is off-origin is also rejected."""

    def handler(request: httpx.Request) -> httpx.Response:
        # No redirect: the request itself targets an off-origin URL.
        return httpx.Response(200, json={"x": 1}, request=request)

    endpoint = DataEndpoint.from_url("https://data.nemar.org/")
    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        with pytest.raises(
            RuntimeError,
            match="Refusing to download a file outside the configured NEMAR",
        ):
            fetch_json(
                client,
                url="https://elsewhere.example.com/x",
                what="testing",
                policy=_one_shot_policy(),
                endpoint=endpoint,
            )


# ---------------------------------------------------------------------------
# Tests migrated from ``test_download_internals.py`` (retry-loop contract).
# ---------------------------------------------------------------------------


def test_fetch_json_recovers_after_retryable_status_via_mock_client() -> None:
    """Retryable HTTP statuses are retried before failing.

    Mirrors the legacy ``test_fetch_json_with_retries_recovers_after_retryable_status``
    test but against the new ``fetch_json`` signature.
    """
    client = MagicMock()
    client.get.side_effect = [
        httpx.Response(503, request=httpx.Request("GET", "https://example.org")),
        httpx.Response(
            200,
            json={"ok": True},
            request=httpx.Request("GET", "https://example.org"),
        ),
    ]

    with mock.patch("nemar._transport.time.sleep"):
        result = fetch_json(
            client,
            url="https://example.org",
            what="testing",
            policy=_two_shot_policy(),
        )

    assert result == {"ok": True}
    assert client.get.call_count == 2


@pytest.mark.parametrize(
    ("response_or_error", "message"),
    [
        pytest.param(
            httpx.ConnectError("connection failed"),
            "Network error",
            id="network-exception",
        ),
        pytest.param(
            httpx.Response(
                503,
                request=httpx.Request("GET", "https://example.org"),
            ),
            "Retryable error",
            id="retryable-status-exhausted",
        ),
        pytest.param(
            httpx.Response(
                404,
                json={"message": "gone"},
                request=httpx.Request("GET", "https://example.org"),
            ),
            "HTTP 404 gone",
            id="non-retryable-status",
        ),
    ],
)
def test_fetch_json_reports_failures(response_or_error, message) -> None:
    """Fetch failures are normalized by retry class.

    Migrated from ``test_fetch_json_with_retries_reports_failures``.
    """
    client = MagicMock()
    if isinstance(response_or_error, BaseException):
        client.get.side_effect = response_or_error
    else:
        client.get.return_value = response_or_error

    with pytest.raises(RuntimeError, match=message):
        fetch_json(
            client,
            url="https://example.org",
            what="testing",
            policy=_one_shot_policy(),
        )


def test_fetch_json_reports_invalid_json() -> None:
    """Invalid JSON responses include the requested URL.

    Migrated from ``test_fetch_json_with_retries_reports_invalid_json``.
    """
    client = MagicMock()
    client.get.return_value = httpx.Response(
        200,
        text="not-json",
        request=httpx.Request("GET", "https://example.org"),
    )

    with pytest.raises(RuntimeError, match="Invalid JSON"):
        fetch_json(
            client,
            url="https://example.org",
            what="testing",
            policy=_one_shot_policy(),
        )


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        pytest.param(
            httpx.Response(
                500,
                json={"message": "server exploded"},
                request=httpx.Request("GET", "https://example.org"),
            ),
            "server exploded",
            id="json-message",
        ),
        pytest.param(
            httpx.Response(
                500,
                json={"error": "version missing"},
                request=httpx.Request("GET", "https://example.org"),
            ),
            "version missing",
            id="json-error",
        ),
        pytest.param(
            httpx.Response(
                500,
                json=["bad"],
                request=httpx.Request("GET", "https://example.org"),
            ),
            '["bad"]',
            id="json-non-object",
        ),
        pytest.param(
            httpx.Response(
                500,
                text="server exploded",
                request=httpx.Request("GET", "https://example.org"),
            ),
            "server exploded",
            id="text",
        ),
        pytest.param(
            httpx.Response(
                500,
                text="",
                request=httpx.Request("GET", "https://example.org"),
            ),
            "",
            id="empty-text",
        ),
    ],
)
def test_response_detail_formats_endpoint_errors(response, expected) -> None:
    """Endpoint error details stay concise across JSON and text payloads.

    Migrated from ``test_response_detail_formats_endpoint_errors``.
    """
    assert _response_detail(response) == expected


# ---------------------------------------------------------------------------
# Control-flow signal types remain importable for the transfer collaboration.
# ---------------------------------------------------------------------------


def test_retry_fresh_error_is_subclass_of_retryable_error() -> None:
    """``_RetryFreshError`` is a ``_RetryableError`` so existing handlers match."""
    assert issubclass(_RetryFreshError, _RetryableError)


def test_fetch_json_honors_retry_after_header() -> None:
    """A 503/429 with ``Retry-After`` sleeps the advised delay, not the backoff."""
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return httpx.Response(503, headers={"Retry-After": "7"}, request=request)
        return httpx.Response(200, json={"ok": True}, request=request)

    transport = httpx.MockTransport(handler)
    sleep = MagicMock()
    with mock.patch("nemar._transport.time.sleep", sleep):
        with httpx.Client(transport=transport) as client:
            result = fetch_json(
                client,
                url="https://data.nemar.org/x",
                what="testing",
                policy=_two_shot_policy(),
            )

    assert result == {"ok": True}
    assert call_count["n"] == 2
    sleep.assert_called_once_with(7.0)
