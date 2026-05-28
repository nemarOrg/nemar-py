"""Backend composition + selection + the bulk download primitive.

The orchestrator in :mod:`nemar._download` asks :func:`select_backend`
for a (possibly layered) :class:`~nemar._backend.TransferBackend`, then
calls ``.transfer(...)``. This module owns that selection
(:func:`select_backend`), the fallback composition
(:class:`LayeredBackend`), and the bulk entry point
(:func:`download_files`). The actual HTTPS mechanics live in
:mod:`nemar._streaming`; the seam they share lives in
:mod:`nemar._backend`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from tqdm.auto import tqdm

from nemar._backend import TransferBackend, TransferOptions
from nemar._datalad import DataLadBackend
from nemar._endpoint import DataEndpoint
from nemar._models import DatasetFile
from nemar._retry import RetryPolicy
from nemar._streaming import PythonBackend
from nemar._verification import (
    VerifyPolicy,
    assert_all_present,
    partition_pending,
)
from nemar.errors import DataLadError, S3Error
from nemar.s3 import S3Backend


class LayeredBackend:
    """Two-layer transfer: try primary, fall back to secondary on a narrow error.

    The fallback fires only on exceptions whose class is in
    ``fallback_on``. Everything else (HTTP errors, verification
    mismatches, retries exhausted) propagates from the primary. The
    fallback runs over the same file set with the same policies; the
    post-transfer
    :func:`~nemar._verification.assert_all_present` sweep that the
    orchestrator runs after this function returns is the real gate
    either way.

    Construction is policy-driven by :func:`select_backend` and not
    part of the public seam — callers configure the policy via
    :class:`~nemar._backend.TransferOptions` and the index's
    ``datalad_url``, not by building this wrapper directly.

    The default ``fallback_on=(DataLadError,)`` preserves the
    pre-S3 contract for callers that built the wrapper before the
    parameter existed.
    """

    def __init__(
        self,
        primary: TransferBackend,
        fallback: TransferBackend,
        *,
        fallback_on: tuple[type[BaseException], ...] = (DataLadError,),
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.fallback_on = fallback_on

    def transfer(
        self,
        files: Sequence[DatasetFile],
        *,
        target_dir: Path,
        options: TransferOptions,
        verify: VerifyPolicy,
        retry: RetryPolicy,
    ) -> None:
        """Run primary; on any ``fallback_on`` error, run fallback over ``files``."""
        try:
            self.primary.transfer(
                files,
                target_dir=target_dir,
                options=options,
                verify=verify,
                retry=retry,
            )
        except self.fallback_on as exc:
            tqdm.write(
                f"primary backend failed; falling back to next layer: {exc}"
            )
            self.fallback.transfer(
                files,
                target_dir=target_dir,
                options=options,
                verify=verify,
                retry=retry,
            )


def select_backend(
    options: TransferOptions,
    *,
    dataset: str | None = None,
    datalad_url: str | None = None,
    revision: str | None = None,
) -> TransferBackend:
    """Resolve ``options.backend`` into a concrete (possibly layered) backend.

    The chain shape is: **S3 → DataLad → HTTPS**. Each layer earns a
    place only when its precondition holds (S3 needs a ``dataset`` id
    to derive content-addressed keys; the DataLad layer requires
    ``datalad_url``). The composition is a tiny list-builder so adding
    a future backend (Azure / GCS / mirror) is one
    ``layers.append(...)`` line, not three new branches.

    * ``"python"`` → bare :class:`PythonBackend`. Skip every other layer.
    * ``"s3"`` → bare :class:`~nemar.s3.S3Backend`. No fallback. Requires
      ``dataset``.
    * ``"auto"`` with ``dataset`` → S3 → DataLad (when advertised) → HTTPS.
    * ``"auto"`` without ``dataset`` (bulk ``download_files`` API) →
      DataLad (when advertised) → HTTPS. The S3 layer is skipped because
      there is no dataset id to derive an annex key against.
    * ``"datalad"`` and ``datalad_url`` is set → DataLad → HTTPS.
    * ``"datalad"`` and no ``datalad_url`` → degrade to plain
      :class:`PythonBackend` with a tqdm notice.
    """
    requested = options.backend
    https = PythonBackend()

    if requested == "python":
        return https
    if requested == "s3":
        if dataset is None:
            raise ValueError(
                'downloader="s3" requires a dataset id to derive bucket keys; '
                "use downloader=\"python\" for the bulk download_files API."
            )
        return S3Backend(dataset=dataset)

    layers: list[tuple[TransferBackend, tuple[type[BaseException], ...]]] = []
    if requested == "auto" and dataset is not None:
        layers.append((S3Backend(dataset=dataset), (S3Error,)))
    if datalad_url is not None and requested in {"auto", "datalad"}:
        layers.append(
            (
                DataLadBackend(datalad_url=datalad_url, revision=revision),
                (DataLadError,),
            )
        )

    if not layers:
        if requested == "datalad":
            tqdm.write(
                "DataLad backend requested but the dataset index did not advertise "
                "a datalad_url; using the HTTPS downloader."
            )
        return https

    chain: TransferBackend = https
    for primary, on in reversed(layers):
        chain = LayeredBackend(primary, chain, fallback_on=on)
    return chain


def download_files(
    files: Sequence[DatasetFile],
    target_dir: Path | str,
    *,
    options: TransferOptions | None = None,
    verify: VerifyPolicy | None = None,
    retry: RetryPolicy | None = None,
    endpoint: DataEndpoint | None = None,
) -> None:
    """Download a list of :class:`DatasetFile` entries to ``target_dir``.

    The bulk variant of :func:`download_one`. Callers that already have
    their own selection logic (e.g., eegdash composing a custom file
    list from a parsed :class:`~nemar._models.VersionManifest`) reach for
    this primitive instead of the full :func:`nemar.download`
    orchestrator: there is no dataset index, no metadata fetch, no BIDS
    query — only the bytes-on-the-wire phase plus the verify gates.

    The same three-step pipeline the orchestrator runs after manifest
    parsing fires here too: pre-transfer
    :func:`~nemar._verification.partition_pending` (size-only filter,
    skips already-complete files), the selected
    :class:`TransferBackend`, and post-transfer
    :func:`~nemar._verification.assert_all_present` (full size + hash
    sweep over every file in ``files``).

    Parameters
    ----------
    files
        Sequence of :class:`DatasetFile` entries to transfer. An empty
        sequence is a no-op (``target_dir`` is still created so callers
        can chain follow-ups safely).
    target_dir
        Destination directory. Created if it does not exist. ``str`` and
        :class:`~pathlib.Path` are both accepted.
    options
        Transfer-backend knobs (concurrency, stream timeout, backend
        policy). Defaults to a python-backend bundle with 16-way
        concurrency. The DataLad layer is not exercised here — bulk
        file lists do not carry a ``datalad_url`` — so a
        ``backend="datalad"`` value degrades to plain HTTPS with a
        ``tqdm`` notice.
    verify
        Verification policy applied by both the pre-transfer partition
        and the post-transfer sweep. Defaults to size + hash.
    retry
        Per-file retry policy. Defaults to :meth:`RetryPolicy.default`.
    endpoint
        Optional :class:`~nemar._endpoint.DataEndpoint` used to enforce
        origin scoping. When supplied, every file's ``url`` must share
        the endpoint's scheme + netloc; otherwise
        :class:`~nemar.errors.EndpointError` is raised before any
        bytes move. When ``None`` (default) no origin check runs —
        matches :func:`download_one`'s default and trusts the caller's
        own scoping (typically inherited from
        :meth:`~nemar._models.VersionManifest.parse`).

    Raises
    ------
    EndpointError
        When ``endpoint`` is supplied and any file's ``url`` is
        off-origin.
    TransferError
        On irrecoverable transport failure for any file.
    VerificationError
        When the post-transfer sweep finds a file on disk that does not
        satisfy its manifest entry (size or hash mismatch, error
        sentinel, missing file after a swallowed failure).

    """
    target = Path(target_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    if options is None:
        options = TransferOptions(
            backend="python",
            max_concurrent_downloads=16,
            stream_timeout=60.0,
        )
    if verify is None:
        verify = VerifyPolicy()
    if retry is None:
        retry = RetryPolicy.default()
    if endpoint is not None:
        # Check *all* URLs before any bytes move. Otherwise the first
        # files in the list could land on disk before a later off-origin
        # one is rejected, leaving partial state for callers to clean up.
        for file in files:
            endpoint.assert_within(file.url)
    if not files:
        return
    # Bulk file lists do not carry a DataLad URL, so the layered backend
    # never applies here. ``select_backend`` with ``datalad_url=None``
    # returns the plain HTTPS adapter.
    backend = select_backend(options, datalad_url=None)
    pending = partition_pending(
        list(files),
        target_dir=target,
        policy=verify,
        pre_transfer=True,
    )
    if pending:
        backend.transfer(
            pending,
            target_dir=target,
            options=options,
            verify=verify,
            retry=retry,
        )
    assert_all_present(list(files), target_dir=target, policy=verify)
