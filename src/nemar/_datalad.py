"""DataLad transfer adapter — one layer of the layered transfer chain.

The :class:`DataLadBackend` clones the dataset's DataLad repo (advertised
by the NEMAR data endpoint as ``index.datalad_url``) and runs
``datalad get`` against the BIDS-selected files. Composition with the
S3 / HTTPS layers happens in :class:`~nemar._transfer.LayeredBackend`
(which lives next to the :class:`~nemar._transfer.TransferBackend`
Protocol it composes); see :func:`nemar._transfer.select_backend` for
the chain assembly.

DataLad is an optional dependency. The import happens lazily through
:func:`_import_datalad` so the module itself imports cheaply, and so
that callers who never select the DataLad path do not pay for the
``datalad`` package's heavy import surface (which itself pulls in
``git-annex``, ``rdflib``, and friends). A missing dependency surfaces
as :class:`DataLadError`, which the layered wrapper treats as a
fallback signal.

Exception scoping
-----------------

The DataLad adapters narrow their ``except`` clauses to the DataLad
exception classes returned by :func:`_import_datalad`
(``CommandError`` and ``IncompleteResultsError``). Programming bugs
(``TypeError``, ``AttributeError``, …) bubble up to the caller
unchanged, so a real defect in this module never silently routes
through the HTTPS fallback — it surfaces immediately as the bug it is.

Policy scope
------------

``DataLadBackend`` accepts the full :class:`TransferBackend` protocol
arguments (``options``, ``verify``, ``retry``) but only ``options``
informs its behavior (``max_concurrent_downloads`` → DataLad's
``jobs``). ``verify`` and ``retry`` are honored end-to-end only by the
HTTPS fallback; DataLad's per-file git-annex hash check and the
orchestrator's post-transfer ``assert_all_present`` sweep are the
real verification gates.

Test hook
---------

:func:`_import_datalad` is the single seam tests monkey-patch to
substitute a fake module bundle. Production code never substitutes it.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from nemar._models import DatasetFile
from nemar._retry import RetryPolicy
from nemar._verification import VerifyPolicy
from nemar.errors import DataLadError

if TYPE_CHECKING:
    # ``TransferOptions`` is used only as a type annotation. Guarding it
    # behind ``TYPE_CHECKING`` keeps the dependency direction one-way
    # (``_datalad`` → ``_transfer`` only for type checkers) and documents
    # that this module has no runtime coupling to the HTTPS / S3 backends.
    from nemar._backend import TransferOptions


@dataclass(frozen=True)
class _DataLadModules:
    """Bundle of the DataLad API surfaces the backend touches.

    Returned by :func:`_import_datalad` so tests substitute the entire
    surface (api, ``Dataset`` constructor, exception tuple) in one place
    instead of monkey-patching three separate import sites.
    """

    api: ModuleType | Any
    Dataset: Any
    exceptions: tuple[type[BaseException], ...]


def _import_datalad() -> _DataLadModules:
    """Return the DataLad module bundle or raise :class:`DataLadError`.

    Single seam for the optional-dependency import. Production code calls
    this once per :meth:`DataLadBackend.transfer` invocation; Python's
    import cache makes the second call free. Tests monkey-patch this
    function to inject a fake bundle so the DataLad code path runs
    without the real ``datalad`` package installed.

    The bundle carries the DataLad exception classes the backend
    catches so the ``except`` clauses do not need a separate, conditional
    import that might fail when ``datalad`` is absent.
    """
    try:
        api = importlib.import_module("datalad.api")
        Dataset = importlib.import_module("datalad.distribution.dataset").Dataset
        exceptions_module = importlib.import_module("datalad.support.exceptions")
        command_error = exceptions_module.CommandError
        incomplete_results_error = exceptions_module.IncompleteResultsError
    except ImportError as exc:
        raise DataLadError(
            "DataLad is not installed. Install with: pip install nemar-py[datalad]"
        ) from exc
    return _DataLadModules(
        api=api,
        Dataset=Dataset,
        exceptions=(command_error, incomplete_results_error),
    )


@dataclass(frozen=True)
class DataLadBackend:
    """Adapter that materializes files through a DataLad / git-annex clone.

    The clone source is the URL advertised by the NEMAR dataset index
    (``index.datalad_url``). When ``revision`` is provided (the resolved
    version tag), it is checked out before ``datalad get`` runs against
    the BIDS-selected paths. Any failure — missing optional dep, clone
    error, checkout error, ``get`` error — raises :class:`DataLadError`,
    which the wrapping :class:`LayeredBackend` catches to trigger the
    HTTPS fallback.

    Idempotency is explicit: when ``target_dir / ".datalad"`` already
    exists the existing clone is opened via
    :class:`datalad.distribution.dataset.Dataset` rather than
    re-installed, because DataLad's ``install`` is not guaranteed to be
    a no-op against a pre-existing path. This is what lets re-runs reuse
    a cached clone instead of always falling back to HTTPS.

    The orchestrator's post-transfer ``assert_all_present`` sweep is the
    real verification gate; this backend trusts the per-file hashes
    git-annex already validates internally and leaves the manifest-level
    re-check to the caller.
    """

    datalad_url: str
    revision: str | None = None

    def transfer(
        self,
        files: Sequence[DatasetFile],
        *,
        target_dir: Path,
        options: TransferOptions,
        verify: VerifyPolicy,
        retry: RetryPolicy,
    ) -> None:
        """Clone (or open) the DataLad sibling and fetch ``files``.

        Steps: load the DataLad bundle → open the existing clone if
        ``target_dir / ".datalad"`` is present, otherwise install →
        optionally check out ``revision`` → ``datalad get`` the
        BIDS-selected paths.

        ``options.max_concurrent_downloads`` is forwarded as DataLad's
        ``jobs`` argument. The two are semantically related but not
        identical: ``max_concurrent_downloads`` caps simultaneous HTTP
        streams in the Python backend, while ``jobs`` caps
        simultaneous git-annex content retrievals (each of which may
        itself open multiple HTTP connections). They are deliberately
        bound to the same knob so the user has one budget to tune.

        ``verify`` and ``retry`` are accepted to satisfy the
        :class:`TransferBackend` protocol but are not consulted here.
        DataLad owns its own retry and integrity behavior via
        git-annex; the orchestrator's post-transfer
        :func:`~nemar._verification.assert_all_present` is the real
        verification gate. Callers who set non-default
        ``RetryPolicy`` / ``VerifyPolicy`` see them honored only on
        the HTTPS fallback path.
        """
        modules = _import_datalad()
        dataset = self._open_or_install(modules, target_dir)
        if self.revision is not None:
            self._checkout(dataset, modules)
        if not files:
            return
        try:
            dataset.get(
                path=[file.path for file in files],
                jobs=options.max_concurrent_downloads,
            )
        except modules.exceptions as exc:
            raise DataLadError(
                f"datalad get failed for {len(files)} file(s): {exc}"
            ) from exc

    def _open_or_install(
        self, modules: _DataLadModules, target_dir: Path
    ) -> Any:
        """Open an existing clone at ``target_dir`` or install a new one.

        The presence of ``target_dir / ".datalad"`` is the signal that
        ``target_dir`` already holds a DataLad clone; calling
        ``Dataset(target_dir)`` then reuses it without touching the
        network. Otherwise we clone via :func:`datalad.api.install`.
        Either failure mode surfaces as :class:`DataLadError` to the
        layered wrapper.
        """
        if (target_dir / ".datalad").is_dir():
            try:
                return modules.Dataset(str(target_dir))
            except modules.exceptions as exc:
                raise DataLadError(
                    f"failed to open existing DataLad clone at {target_dir}: {exc}"
                ) from exc
        try:
            return modules.api.install(
                source=self.datalad_url, path=str(target_dir)
            )
        except modules.exceptions as exc:
            raise DataLadError(
                f"datalad install failed for {self.datalad_url}: {exc}"
            ) from exc

    def _checkout(self, dataset: Any, modules: _DataLadModules) -> None:
        """Check out ``self.revision`` on ``dataset``.

        A separate method so the ``except`` clause stays narrow without
        forcing a deeply-nested ``try`` in :meth:`transfer`.
        """
        try:
            dataset.repo.checkout(self.revision)
        except modules.exceptions as exc:
            raise DataLadError(
                f"datalad failed to check out revision "
                f"{self.revision!r}: {exc}"
            ) from exc


