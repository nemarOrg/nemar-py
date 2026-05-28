"""DataLad / git-annex transfer backend — first layer of the transfer chain.

:class:`DataLadBackend` clones the dataset's DataLad repo (advertised as
``index.datalad_url``) and runs ``datalad get`` against the BIDS-selected
files. The layered chain in :mod:`nemar._transfer` tries it first and falls
back to HTTPS when any step raises :class:`DataLadError`.

DataLad is a hard dependency but a heavy one (it pulls in ``git-annex``,
``rdflib``, and friends), so the import is lazy via :func:`_import_datalad`:
importing it at module top would tax every consumer that imports
``nemar`` (including eegdash), even those that never select the DataLad
path. ``import datalad.api`` alone costs ~0.5s, so deferring it until
:meth:`DataLadBackend.transfer` actually runs keeps ``import nemar`` fast.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from nemar._backend import TransferOptions
from nemar._models import DatasetFile
from nemar._retry import RetryPolicy
from nemar._verification import VerifyPolicy
from nemar.errors import DataLadError


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
    """Lazily import the DataLad surfaces so ``import nemar`` stays cheap.

    DataLad is an optional extra (``pip install nemar-py[datalad]``). When
    it is not installed, the import fails and we convert the ``ImportError``
    into a :class:`DataLadError` — the exception the wrapping
    :class:`LayeredBackend` catches to fall through to the S3 / HTTPS
    layers. Without this conversion a missing optional dependency would
    raise a bare ``ImportError`` that escapes the fallback and aborts the
    whole download.
    """
    try:
        import datalad.api as api
        from datalad.distribution.dataset import Dataset
        from datalad.support.exceptions import CommandError, IncompleteResultsError
    except ImportError as exc:
        raise DataLadError(
            "DataLad is not installed. Install the optional extra with "
            "'pip install nemar-py[datalad]', or use downloader='s3' / "
            "'python' (the auto chain already falls back to HTTPS)."
        ) from exc

    return _DataLadModules(
        api=api,
        Dataset=Dataset,
        exceptions=(CommandError, IncompleteResultsError),
    )


@dataclass(frozen=True)
class DataLadBackend:
    """Adapter that materializes files through a DataLad / git-annex clone.

    The clone source is the URL advertised by the NEMAR dataset index
    (``index.datalad_url``). When ``revision`` is provided (the resolved
    version tag), it is checked out before ``datalad get`` runs against
    the BIDS-selected paths. Any failure — clone error, checkout error,
    ``get`` error — raises :class:`DataLadError`, which the wrapping
    :class:`LayeredBackend` catches to trigger the HTTPS fallback.

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
            try:
                dataset.repo.checkout(self.revision)
            except modules.exceptions as exc:
                raise DataLadError(
                    f"datalad failed to check out revision "
                    f"{self.revision!r}: {exc}"
                ) from exc
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


