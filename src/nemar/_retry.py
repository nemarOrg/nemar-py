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

import httpx

_DEFAULT_RETRYABLE_STATUS: frozenset[int] = frozenset(
    {408, 429, 500, 502, 503, 504, 522, 524}
)
_DEFAULT_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.ReadError,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
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

    def with_max_backoff(self, cap: float | None) -> RetryPolicy:
        """Return a copy with ``max_backoff`` set to ``cap``."""
        return replace(self, max_backoff=cap)
