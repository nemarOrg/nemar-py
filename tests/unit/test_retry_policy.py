"""Tests for the ``RetryPolicy`` value type.

``RetryPolicy`` encapsulates the retry contract previously expressed as three
floating module-level constants (status set, exception tuple, backoff helper).
The contract under test mirrors what the JSON-fetch and per-file-stream retry
loops have always done, so behavior is preserved across the refactor.
"""

from __future__ import annotations

import statistics

import httpx
import pytest

from nemar._retry import RetryPolicy


def test_default_policy_carries_expected_status_codes() -> None:
    """The default policy advertises the same status set the loops used."""
    policy = RetryPolicy.default()

    assert policy.retryable_status == frozenset(
        {408, 429, 500, 502, 503, 504, 522, 524}
    )


def test_default_policy_carries_expected_exception_types() -> None:
    """The default policy advertises the httpx exception tuple."""
    policy = RetryPolicy.default()

    assert policy.retryable_exceptions == (
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.ReadError,
        httpx.ConnectError,
        httpx.RemoteProtocolError,
        httpx.PoolTimeout,
        httpx.WriteError,
    )


def test_default_policy_defaults_to_unbounded_backoff() -> None:
    """The default policy preserves the unbounded backoff curve."""
    policy = RetryPolicy.default()

    assert policy.max_backoff is None
    assert policy.base_backoff == pytest.approx(0.5)


def test_default_policy_defaults_to_six_attempts() -> None:
    """The default policy mirrors ``max_retries=5`` (one initial + five retries)."""
    policy = RetryPolicy.default()

    assert policy.max_attempts == 6


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        pytest.param(408, True, id="408"),
        pytest.param(429, True, id="429"),
        pytest.param(500, True, id="500"),
        pytest.param(502, True, id="502"),
        pytest.param(503, True, id="503"),
        pytest.param(504, True, id="504"),
        pytest.param(522, True, id="522"),
        pytest.param(524, True, id="524"),
        pytest.param(200, False, id="200"),
        pytest.param(404, False, id="404"),
        pytest.param(418, False, id="418"),
    ],
)
def test_should_retry_status_round_trips(code: int, expected: bool) -> None:
    """``should_retry_status`` matches membership in the policy's status set."""
    assert RetryPolicy.default().should_retry_status(code) is expected


def test_should_retry_exception_matches_retryable_types() -> None:
    """``should_retry_exception`` is true for known transient httpx errors."""
    policy = RetryPolicy.default()

    assert policy.should_retry_exception(httpx.ConnectTimeout("x"))
    assert policy.should_retry_exception(httpx.ReadTimeout("x"))
    assert policy.should_retry_exception(httpx.ReadError("x"))
    assert policy.should_retry_exception(httpx.ConnectError("x"))
    assert policy.should_retry_exception(httpx.RemoteProtocolError("x"))


def test_should_retry_exception_rejects_unrelated_errors() -> None:
    """``should_retry_exception`` is false for unrelated exceptions."""
    policy = RetryPolicy.default()

    assert not policy.should_retry_exception(ValueError("x"))
    assert not policy.should_retry_exception(KeyError("x"))


def test_next_delay_lives_in_full_jitter_window() -> None:
    """``next_delay`` uses ``base * 2**attempt`` jittered into ``[b, 1.5b)``."""
    policy = RetryPolicy.default()
    base = policy.base_backoff

    for attempt in range(5):
        expected_base = base * (2**attempt)
        upper = expected_base + expected_base / 2.0
        samples = [policy.next_delay(attempt) for _ in range(200)]
        assert all(expected_base <= v < upper for v in samples), (
            f"attempt {attempt} produced out-of-band samples"
        )
        assert statistics.stdev(samples) > 0.0


def test_with_attempts_returns_copy_with_attempt_count() -> None:
    """``with_attempts(n)`` produces a new policy capped at ``n`` attempts."""
    policy = RetryPolicy.default()

    one_shot = policy.with_attempts(0)

    # ``max_retries=0`` in the public API means "try once, no retry" --
    # so ``with_attempts(0)`` MUST produce ``max_attempts=1``.
    assert one_shot.max_attempts == 1
    # Original policy is untouched (frozen dataclass semantics).
    assert policy.max_attempts == 6


def test_with_attempts_preserves_other_fields() -> None:
    """``with_attempts`` only changes ``max_attempts``."""
    policy = RetryPolicy.default()
    copy = policy.with_attempts(3)

    assert copy.max_attempts == 4
    assert copy.retryable_status == policy.retryable_status
    assert copy.retryable_exceptions == policy.retryable_exceptions
    assert copy.base_backoff == policy.base_backoff
    assert copy.max_backoff == policy.max_backoff


def test_policy_is_frozen() -> None:
    """``RetryPolicy`` is a frozen value type."""
    policy = RetryPolicy.default()

    with pytest.raises((AttributeError, TypeError)):
        policy.max_attempts = 99  # type: ignore[misc]


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        pytest.param("30", 30.0, id="integer-seconds"),
        pytest.param("0", 0.0, id="zero"),
        pytest.param("99999", 120.0, id="capped-at-max"),
        pytest.param("  12  ", 12.0, id="whitespace-trimmed"),
        pytest.param("not-a-date", None, id="unparseable"),
        pytest.param("", None, id="empty"),
        pytest.param(None, None, id="absent"),
    ],
)
def test_parse_retry_after(header, expected) -> None:
    """``parse_retry_after`` handles integer seconds, caps, and junk."""
    from nemar._retry import parse_retry_after

    assert parse_retry_after(header) == expected


def test_parse_retry_after_http_date_in_the_past_is_zero() -> None:
    """An HTTP-date already in the past yields a zero (immediate) delay."""
    from nemar._retry import parse_retry_after

    assert parse_retry_after("Wed, 21 Oct 1990 07:28:00 GMT") == 0.0
