"""The transfer seam: the interface every backend satisfies + the policy enum.

This module is the shared interface that both the streaming engine
(:mod:`nemar._streaming`) and the backend composer (:mod:`nemar._transfer`)
depend on. Splitting it out keeps the dependency graph a DAG:
``_backend`` ← ``_streaming`` ← ``_transfer``, and ``_backend`` ← ``_request``
(the validator reads ``VALID_BACKENDS``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from nemar._models import DatasetFile
from nemar._retry import RetryPolicy
from nemar._verification import VerifyPolicy

VALID_BACKENDS = frozenset({"auto", "python", "datalad", "s3"})
"""The accepted values for ``TransferOptions.backend``.

Co-resident with the seam it parameterizes and imported by
:func:`~nemar._request._validate` (validator) and
:func:`~nemar._transfer.select_backend` (selector), so a new backend
addition does not drift between the two.
"""


@dataclass(frozen=True)
class TransferOptions:
    """Transfer-backend knobs that travel with one download request.

    Bundles the three runtime values the orchestrator forwards to the
    transfer phase: which backend to select, how much concurrency to
    allow, and the per-stream timeout for the Python backend.
    """

    backend: str
    max_concurrent_downloads: int
    stream_timeout: float


class TransferBackend(Protocol):
    """Adapter contract for the post-selection bytes-on-the-wire phase.

    Implementations receive the list of files that the pre-transfer
    verifier already filtered (so every entry needs transfer), the
    target directory (created by the caller), the runtime options
    bundle, the verify policy, and the retry policy. They are
    responsible for the network work only; the orchestrator runs
    :func:`~nemar._verification.assert_all_present` after this returns
    to gate on the final hash + size sweep.
    """

    def transfer(
        self,
        files: Sequence[DatasetFile],
        *,
        target_dir: Path,
        options: TransferOptions,
        verify: VerifyPolicy,
        retry: RetryPolicy,
    ) -> None:
        """Move ``files`` into ``target_dir``.

        Raises :class:`~nemar.errors.TransferError` on any irrecoverable
        failure. The orchestrator does not catch it; it is the
        user-facing failure path.
        """
