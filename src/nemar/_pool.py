"""Run one callable over many items concurrently, reporting every failure.

A leaf module — no ``nemar`` imports — so both file-writing backends
(:mod:`nemar.s3`, :mod:`nemar._streaming`) can share it without either
importing the other or the layer that composes them.

Both backends need the same thing: fan a per-item worker out across a thread
pool, let every item run even if one fails, then raise once with a summary.
Written twice it drifts — the second copy silently lost exception chaining,
so the underlying botocore error never reached the traceback.
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable, Iterable
from typing import TypeVar

__all__ = ["run_batch"]

T = TypeVar("T")

#: How many failure messages to quote before summarising the rest. Enough to
#: see a pattern, few enough that one bad batch does not bury the terminal.
_QUOTED_ERRORS = 3


def run_batch(
    worker: Callable[[T], None],
    items: Iterable[T],
    *,
    workers: int,
    error_cls: type[Exception],
    label: str,
    fail_fast: bool = False,
) -> None:
    """Apply ``worker`` to every item, raising ``error_cls`` if any failed.

    By default every item is attempted even when an earlier one fails: a
    partially transferred batch is still progress, and the caller learns about
    all of the failures at once rather than one re-run at a time.

    ``fail_fast`` inverts that for callers whose batch is all-or-nothing. There,
    finishing the batch after a failure is pure waste -- the work is discarded
    and redone by whatever handles the failure -- so pending items are cancelled
    as soon as one raises.

    Raises
    ------
    error_cls
        If any item raised, with the first few messages quoted and a count of
        the rest. The first failure is chained as ``__cause__`` so its
        traceback survives.

    """
    errors: list[BaseException] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, item) for item in items]
        for future in concurrent.futures.as_completed(futures):
            exc = future.exception()
            if exc is None:
                continue
            errors.append(exc)
            if fail_fast:
                for pending in futures:
                    pending.cancel()
                break
    if not errors:
        return
    head = "; ".join(str(exc) for exc in errors[:_QUOTED_ERRORS])
    extra = len(errors) - _QUOTED_ERRORS
    more = f" (+{extra} more)" if extra > 0 else ""
    raise error_cls(
        f"{len(errors)} file(s) failed during {label}: {head}{more}"
    ) from errors[0]
