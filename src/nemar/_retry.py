"""Retry policy value type for ``data.nemar.org`` requests.

The retry contract used by the JSON-fetch and per-file-stream loops is
expressed as a single frozen value: which HTTP statuses to retry, which
transient ``httpx`` errors to retry, and how to schedule the backoff
between attempts. Keeping it in a value type makes the contract testable
in isolation and removes duplication between the two retry loops.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx

# Cap on a server-advised ``Retry-After`` so a hostile or misconfigured
# origin cannot stall the client for an unbounded time.
_MAX_RETRY_AFTER = 120.0


def parse_retry_after(value: str | None) -> float | None:
    """Parse an HTTP ``Retry-After`` header into a delay in seconds.

    Honours both forms RFC 9110 allows — a delay in integer seconds
    (``Retry-After: 30``) and an HTTP-date (``Retry-After: Wed, 21 Oct
    2026 07:28:00 GMT``). Returns ``None`` when the header is absent or
    unparseable (the caller then falls back to its computed backoff). The
    result is clamped to ``[0, _MAX_RETRY_AFTER]`` so a bad value cannot
    hang the client.
    """
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return min(float(value), _MAX_RETRY_AFTER)
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    delta = (when - datetime.now(timezone.utc)).total_seconds()
    if delta <= 0:
        return 0.0
    return min(delta, _MAX_RETRY_AFTER)

_DEFAULT_RETRYABLE_STATUS: frozenset[int] = frozenset(
    {408, 429, 500, 502, 503, 504, 522, 524}
)
_DEFAULT_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.ReadError,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    # PoolTimeout: under saturated connection pool (max_connections =
    # 2x concurrency, large datasets) httpx raises this when no slot
    # frees up in time. Transient -- the next attempt will likely have
    # room.
    httpx.PoolTimeout,
    # WriteError: TCP write failed mid-stream (connection reset by the
    # peer, broken pipe). Transient on flaky links.
    httpx.WriteError,
)
_DEFAULT_BASE_BACKOFF = 0.5
# ``max_retries=5`` in the public API means "one initial attempt + five retries",
# so the default policy carries six attempts total.
_DEFAULT_MAX_ATTEMPTS = 6


def _next_backoff(base: float) -> float:
    """Return ``base`` plus jitter to avoid synchronized retry storms.

    Full-jitter to half: ``base + uniform(0, base/2)``.
    """
    return base + random.uniform(0.0, base / 2.0)


class _RetryableError(Exception):
    """Raised when a request can be retried.

    Internal control-flow signal shared by the JSON-fetch loop
    (:mod:`nemar._transport`) and the per-file streaming loop
    (:mod:`nemar._streaming`). Callers should not catch it; each loop
    translates it into a final transport/transfer error once retries
    exhaust. Lives here so both loops share the same vocabulary without
    importing each other.
    """


class _RetryFreshError(_RetryableError):
    """Raised when the retry must abandon any local partial bytes.

    Subclass of :class:`_RetryableError` so existing handlers still treat
    it as retryable; the per-file streaming driver distinguishes it to set
    ``force_fresh=True`` on the next attempt (a one-shot recovery after
    HTTP 416).
    """


@dataclass(frozen=True)
class RetryPolicy:
    """How retries are decided and scheduled for one request loop.

    ``max_attempts`` is ``max_retries + 1`` -- one initial attempt plus the
    number of retries permitted. ``base_backoff`` is the seed for the
    exponential-with-jitter schedule. ``max_backoff`` caps the (post-jitter)
    delay; ``None`` means unbounded, preserving the historical behavior.
    """

    max_attempts: int
    base_backoff: float
    max_backoff: float | None
    retryable_status: frozenset[int]
    retryable_exceptions: tuple[type[BaseException], ...]

    @classmethod
    def default(cls) -> RetryPolicy:
        """Return the policy used by the production retry loops."""
        return cls(
            max_attempts=_DEFAULT_MAX_ATTEMPTS,
            base_backoff=_DEFAULT_BASE_BACKOFF,
            max_backoff=None,
            retryable_status=_DEFAULT_RETRYABLE_STATUS,
            retryable_exceptions=_DEFAULT_RETRYABLE_EXCEPTIONS,
        )

    def should_retry_status(self, code: int) -> bool:
        """Return ``True`` when an HTTP status warrants a retry."""
        return code in self.retryable_status

    def should_retry_exception(self, exc: BaseException) -> bool:
        """Return ``True`` when a transport exception warrants a retry."""
        return isinstance(exc, self.retryable_exceptions)

    def next_delay(self, attempt: int) -> float:
        """Return the (jittered) sleep before the ``attempt``-th retry.

        ``attempt`` is zero-indexed: ``next_delay(0)`` returns the wait
        between attempts 0 and 1, ``next_delay(1)`` between attempts 1
        and 2, and so on. The schedule is ``base_backoff * 2**attempt``
        plus full-jitter into the upper half, optionally clamped to
        ``max_backoff``.
        """
        scaled = self.base_backoff * (2**attempt)
        delay = _next_backoff(scaled)
        if self.max_backoff is not None and delay > self.max_backoff:
            return self.max_backoff
        return delay

    def with_attempts(self, max_retries: int) -> RetryPolicy:
        """Return a copy with ``max_attempts = max_retries + 1``.

        This keeps the public ``download(max_retries=N)`` semantics intact:
        callers think in "additional retries", the policy thinks in
        "total attempts".
        """
        return replace(self, max_attempts=max_retries + 1)
