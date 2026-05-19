"""Verify that retry backoff is not perfectly deterministic.

Without jitter, N parallel clients retrying at the same time would all land
in the same millisecond bucket -- a retry storm. We don't measure absolute
timing here; we sample the backoff sequence many times and require some
spread.
"""

from __future__ import annotations

import statistics

import pytest

from nemar._download import _next_backoff


def test_next_backoff_introduces_jitter() -> None:
    """``_next_backoff`` must spread retries with jitter, not return a
    deterministic value. This drives bug fix #1."""
    samples = [_next_backoff(base=1.0) for _ in range(100)]
    # All values should sit in [base, base * 1.5) (full-jitter cap).
    assert all(1.0 <= v < 1.5 for v in samples)
    # And they should not all be the same.
    assert statistics.stdev(samples) > 0.0
