"""Transfer backends — the bytes-on-the-wire phase of one download request.

The orchestrator in :mod:`nemar._download` is policy and choreography
(index, version, metadata, manifest, selection, target check); this
module is the single seam where the bytes actually move.

* :class:`PythonBackend` uses a thread pool over a shared
  ``httpx.Client`` with a small connection pool. It is the only HTTPS
  adapter and exercises the Range/206 + 416-recovery + retry contracts
  the production endpoint sometimes needs.

It implements the :class:`TransferBackend` protocol:

.. code-block:: python

    def transfer(
        self,
        files: Sequence[DatasetFile],
        *,
        target_dir: Path,
        options: TransferOptions,
        verify: VerifyPolicy,
        retry: RetryPolicy,
    ) -> None

Selection is policy-driven via :class:`TransferOptions.backend` ("auto" |
"python" | "datalad") and resolved by :func:`select_backend`. The
``"datalad"`` path layers :mod:`~nemar._datalad` over the HTTPS pick
when the dataset index advertises a ``datalad_url``; the layered
backend falls back to HTTPS on any DataLad failure.
"""

from __future__ import annotations

import concurrent.futures
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx
from tqdm.auto import tqdm

from nemar import __version__
from nemar._datalad import DataLadBackend
from nemar._endpoint import DataEndpoint
from nemar._models import DatasetFile
from nemar._retry import RetryPolicy
from nemar._verification import (
    VerifyPolicy,
    VerifyResult,
    _describe_failure,
    assert_all_present,
    partition_pending,
)
from nemar._verification import check as _verify_check
from nemar.errors import DataLadError, S3Error, TransferError, VerificationError
from nemar.s3 import S3Backend


class _RetryableError(Exception):
    """Raised when a per-file request can be retried.

    Private to this module: the python-backend per-file loop catches it.
    The :mod:`nemar._download` orchestrator has its own retry hierarchy
    for JSON fetches; they are intentionally disjoint so each loop owns
    its own error classes.
    """


class _RetryFreshError(_RetryableError):
    """Raised when the retry must abandon any local partial bytes.

    Subclass of :class:`_RetryableError` so the broader ``except``
    branch still treats it as retryable; the per-file driver
    distinguishes it to set ``force_fresh=True`` on the next attempt
    (a one-shot recovery after HTTP 416).
    """


VALID_BACKENDS = frozenset({"auto", "python", "datalad", "s3"})
"""The accepted values for ``TransferOptions.backend``.

Co-resident with :func:`select_backend` (the only consumer that turns
a value here into a concrete backend) and :func:`~nemar._request._validate`
(which imports this constant). Keeping the set and its consumers in
one module means a new backend addition does not drift between
validator and selector.
"""


@dataclass(frozen=True)
class TransferOptions:
    """Transfer-backend knobs that travel with one download request.

    Bundles the three runtime values the orchestrator forwards to the
    transfer phase: which backend to select, how much concurrency to
    allow, and the per-stream timeout for the Python backend. Lives
    next to :func:`select_backend` (the consumer that reads
    ``.backend``) rather than under ``_request`` because every concrete
    backend takes one of these as a parameter — co-locating it with
    the protocol it parameterizes keeps the transfer surface
    self-contained.
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
    :func:`assert_all_present` after this returns to gate on the
    final hash + size sweep.
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

        Raises :class:`TransferError` on any irrecoverable failure. The
        orchestrator does not catch it; it is the user-facing failure
        path.
        """


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
    :class:`~nemar._request.TransferOptions` and the index's
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


class PythonBackend:
    """Adapter that streams files with ``httpx`` + a thread pool.

    Opens one shared ``httpx.Client`` for the whole batch so keepalive
    can amortize the TCP+TLS handshake across the thousands of small
    TSV/JSON files a BIDS dataset typically carries. The connection
    pool is sized at ~2x the concurrency cap so retries do not starve
    in-flight requests.

    Per-file handling lives in
    :func:`_transfer_one_with_python` (the retry driver) and
    :func:`_transfer_one_attempt` (one HTTP attempt). They preserve
    the subtle behaviour the production endpoint sometimes needs:

    * Range/206 resume against partial bytes already on disk.
    * 416 → :class:`_RetryFreshError` → one-shot ``force_fresh=True``
      retry that unlinks the stale partial.
    * The progress-bar overshoot fix when a server returns 200
      instead of the requested 206 for a Range request.
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
        total = sum(file.size or 0 for file in files)
        # One httpx.Client for the whole batch. BIDS datasets have
        # thousands of small TSV/JSON files; opening a fresh client per
        # file means a TCP+TLS handshake per file. Reusing a single
        # client with a connection pool lets keepalive carry the cost
        # across the batch. The pool is sized at ~2x the concurrency
        # cap so retries do not starve in-flight requests.
        #
        # TODO: enterprise users behind custom-CA MITM proxies would
        # benefit from ``truststore.SSLContext()`` here so the OS trust
        # store is honoured without an explicit ``SSL_CERT_FILE``.
        # Adding the truststore dependency complicates the wheel; defer
        # until there is a concrete user request. The ``verify`` knob
        # below is the hook.
        max_concurrent_downloads = options.max_concurrent_downloads
        stream_timeout = options.stream_timeout
        limits = httpx.Limits(
            max_connections=max(1, max_concurrent_downloads * 2),
            max_keepalive_connections=max(1, max_concurrent_downloads),
        )
        with httpx.Client(
            follow_redirects=False,
            timeout=stream_timeout,
            headers={
                "accept": "*/*",
                "user-agent": f"nemar-py/{__version__}",
            },
            limits=limits,
            verify=True,
        ) as client:
            with tqdm(
                total=total,
                desc="Overall",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_concurrent_downloads
                ) as executor:
                    futures = [
                        executor.submit(
                            _transfer_one_with_python,
                            file,
                            target_dir,
                            retry,
                            verify,
                            progress,
                            stream_timeout,
                            client,
                        )
                        for file in files
                    ]
                    errors: list[BaseException] = []
                    for future in concurrent.futures.as_completed(futures):
                        exc = future.exception()
                        if exc is not None:
                            errors.append(exc)
                    if errors:
                        head = "; ".join(str(e) for e in errors[:3])
                        raise TransferError(
                            f"{len(errors)} file(s) failed during transfer: {head}"
                        ) from errors[0]


def _transfer_one_with_python(
    file: DatasetFile,
    target_dir: Path,
    retry: RetryPolicy,
    verify: VerifyPolicy,
    progress: tqdm,
    stream_timeout: float,
    client: httpx.Client | None = None,
) -> None:
    """Drive one file to disk inside the bulk Python transfer.

    Thin wrapper around :func:`_stream_with_retries` for the bulk path.
    The bulk caller wants a raise on verify failure (so the
    ``ThreadPoolExecutor`` future fails and the error aggregator can
    report it); the shared streaming helper returns the
    :class:`VerifyResult`, so we translate here.

    Both ``retry`` and ``verify`` are full policy objects, threaded
    through unchanged so that any non-default knob (custom
    ``base_backoff``, ``retryable_status``, ``error_sentinel_max_bytes``,
    …) the caller put on the policy is honored end-to-end.
    """
    outfile = target_dir / file.path
    outfile.parent.mkdir(parents=True, exist_ok=True)
    result = _stream_with_retries(
        file,
        outfile=outfile,
        client=client,
        retry=retry,
        verify=verify,
        stream_timeout=stream_timeout,
        progress=progress,
    )
    if result is not VerifyResult.OK:
        raise VerificationError(_describe_failure(file, outfile, result))


def download_one(
    file: DatasetFile,
    target_path: Path,
    *,
    client: httpx.Client | None = None,
    retry: RetryPolicy | None = None,
    verify: VerifyPolicy | None = None,
    endpoint: DataEndpoint | None = None,
    stream_timeout: float = 60.0,
) -> VerifyResult:
    """Download one :class:`DatasetFile` to ``target_path``.

    The smallest useful transfer operation. Owns range-resume, HTTP 416
    recovery, jittered retry, and final verify for one URL → one path.
    The bulk :class:`PythonBackend` uses it internally per file; external
    callers can reach it without invoking the full bulk orchestrator.

    Parameters
    ----------
    file
        The manifest entry to download. ``file.url`` is the source;
        ``file.size`` / ``file.sha256`` / ``file.md5`` drive verification.
    target_path
        Full destination file path (not a directory). The parent
        directory is created if it does not exist.
    client
        Optional :class:`httpx.Client` to reuse. When ``None`` (default)
        the function builds and closes its own short-lived client. Pass
        a client to amortize TCP/TLS handshakes across many files.
    retry
        Retry policy. Defaults to :meth:`RetryPolicy.default`.
        ``RetryPolicy.default().with_attempts(0)`` disables retries.
    verify
        Verification policy applied to the file post-transfer. Defaults
        to :class:`VerifyPolicy` (size + hash).
    endpoint
        Optional :class:`~nemar._endpoint.DataEndpoint` to enforce
        origin scoping at the call site. When supplied, ``file.url``
        must share the endpoint's scheme + netloc; otherwise
        :class:`~nemar.errors.EndpointError` is raised before any
        bytes are fetched. Files produced by
        :meth:`~nemar.models.VersionManifest.parse` are already origin-scoped;
        passing ``endpoint`` here is the right move for callers that
        hand-build a :class:`DatasetFile` from untrusted input.
    stream_timeout
        Per-stream HTTP timeout in seconds. Defaults to 60.

    Returns
    -------
    VerifyResult
        The result of verifying the downloaded file against ``file``.
        Returning a non-``OK`` value is not an error -- it surfaces a
        condition (size or hash mismatch, error sentinel, missing file
        after a swallowed failure) the caller may want to branch on.

    Raises
    ------
    EndpointError
        When ``endpoint`` is supplied and ``file.url`` is off-origin.
    TransferError
        On irrecoverable transport failure (retries exhausted, or a
        non-retryable HTTP error such as 4xx that is not 416).

    """
    if endpoint is not None:
        endpoint.assert_within(file.url)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if retry is None:
        retry = RetryPolicy.default()
    if verify is None:
        verify = VerifyPolicy()

    owns_client = client is None
    if client is None:
        client = _build_oneshot_client(stream_timeout)
    progress = tqdm(disable=True, leave=False)
    try:
        return _stream_with_retries(
            file,
            outfile=target_path,
            client=client,
            retry=retry,
            verify=verify,
            stream_timeout=stream_timeout,
            progress=progress,
        )
    finally:
        progress.close()
        if owns_client:
            client.close()


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
    list from a parsed :class:`~nemar.models.VersionManifest`) reach for
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
        Optional :class:`~nemar.models.DataEndpoint` used to enforce
        origin scoping. When supplied, every file's ``url`` must share
        the endpoint's scheme + netloc; otherwise
        :class:`~nemar.errors.EndpointError` is raised before any
        bytes move. When ``None`` (default) no origin check runs —
        matches :func:`download_one`'s default and trusts the caller's
        own scoping (typically inherited from
        :meth:`~nemar.models.VersionManifest.parse`).

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


def _build_oneshot_client(stream_timeout: float) -> httpx.Client:
    """Build a short-lived ``httpx.Client`` for the single-file path.

    Mirrors the headers / redirect / verify posture of the bulk client
    in ``_transfer_with_python`` but omits the multi-connection pool
    caps — a one-shot client only handles one file at a time.
    """
    return httpx.Client(
        follow_redirects=False,
        timeout=stream_timeout,
        headers={
            "accept": "*/*",
            "user-agent": f"nemar-py/{__version__}",
        },
        verify=True,
    )


def _stream_with_retries(
    file: DatasetFile,
    *,
    outfile: Path,
    client: httpx.Client | None,
    retry: RetryPolicy,
    verify: VerifyPolicy,
    stream_timeout: float,
    progress: tqdm,
) -> VerifyResult:
    """Run one file's stream/retry/verify loop and return the verify outcome.

    Mirrors the previous ``_transfer_one_with_python`` loop but threads
    ``RetryPolicy`` / ``VerifyPolicy`` through as values and returns
    the :class:`VerifyResult` instead of raising on verify failure.
    Irrecoverable transport errors still raise :class:`TransferError`
    after retries are exhausted.
    """
    last_attempt = retry.max_attempts - 1
    force_fresh = False
    for attempt in range(retry.max_attempts):
        try:
            _transfer_one_attempt(
                file,
                outfile=outfile,
                progress=progress,
                stream_timeout=stream_timeout,
                policy=retry,
                force_fresh=force_fresh,
                client=client,
            )
            return _verify_check(file, outfile, verify)
        except retry.retryable_exceptions as exc:
            if attempt == last_attempt:
                raise TransferError(f"Failed to download {file.path}: {exc}") from exc
            force_fresh = False
        except _RetryFreshError as exc:
            if attempt == last_attempt:
                raise TransferError(f"Failed to download {file.path}: {exc}") from exc
            force_fresh = True
        except _RetryableError as exc:
            if attempt == last_attempt:
                raise TransferError(f"Failed to download {file.path}: {exc}") from exc
            force_fresh = False

        time.sleep(retry.next_delay(attempt))

    raise TransferError(  # pragma: no cover
        f"Unexpected retry exhaustion when downloading {file.path}"
    )


def _transfer_one_attempt(
    file: DatasetFile,
    *,
    outfile: Path,
    progress: tqdm,
    stream_timeout: float,
    policy: RetryPolicy | None = None,
    force_fresh: bool = False,
    client: httpx.Client | None = None,
) -> None:
    """One HTTP attempt for one file.

    Handles Range/206 resume, the 200-instead-of-206 progress
    overshoot correction (some servers ignore Range), the 416
    fresh-retry signal, and the retryable-status escalation. Raises
    :class:`_RetryFreshError` when the local partial cannot be
    salvaged, :class:`_RetryableError` for transient HTTP statuses,
    and :class:`TransferError` for non-retryable HTTP errors.
    """
    if policy is None:
        policy = RetryPolicy.default()
    request_headers: dict[str, str] = {
        "accept": "*/*",
        "user-agent": f"nemar-py/{__version__}",
    }
    mode = "wb"
    # Defensive against the layered DataLad → HTTPS fallback: a partial
    # DataLad attempt can leave a git-annex symlink (often broken,
    # pointing into ``.git/annex/objects/...``) at the manifest path.
    # Without this unlink, ``open(symlink, "wb")`` below would follow
    # the symlink and write through to the annex object path, corrupting
    # the git-annex working tree. ``Path.exists()`` returns False on a
    # broken symlink, so the size-based branches below would not catch
    # this case on their own. Run unconditionally on any symlink, broken
    # or live — HTTPS authority is the manifest, not the symlink target.
    if outfile.is_symlink():
        outfile.unlink()
    # ``force_fresh`` discards any local partial bytes before sending
    # an unconditional GET. Used when the previous attempt got a 416 and
    # the local partial cannot represent a valid suffix of the current
    # object.
    if force_fresh and outfile.exists():
        outfile.unlink()
    local_size = outfile.stat().st_size if outfile.exists() else 0
    if not force_fresh and file.size is not None and 0 < local_size < file.size:
        request_headers["Range"] = f"bytes={local_size}-"
        # Disable compression only on Range requests. Range + gzip is
        # fragile: the server may stream a compressed full body instead
        # of the requested byte range, defeating the resume. On a fresh
        # GET we let httpx send its default ``gzip, deflate`` so small
        # JSON/TSV BIDS sidecars come back compressed.
        request_headers["accept-encoding"] = ""
        mode = "ab"
        progress.update(local_size)
    elif file.size is not None and local_size > file.size:
        outfile.unlink()
    elif file.size is None and outfile.exists():
        outfile.unlink()

    # Prefer the shared client when one was passed in. The
    # module-level ``httpx.stream`` fallback exists for callers that
    # still drive this function directly (e.g. legacy tests) so we do
    # not break their patched-stream pattern.
    if client is not None:
        stream_cm = client.stream(
            "GET", file.url, headers=request_headers, timeout=stream_timeout
        )
    else:
        stream_cm = httpx.stream(
            "GET", file.url, headers=request_headers, timeout=stream_timeout
        )
    with stream_cm as response:
        # A 416 against a Range request means the server's view of
        # the object has changed (object shorter than ``local_size``, or
        # replaced). The partial cannot be salvaged. Unlink it, refund
        # the pre-credited progress, and ask the outer loop to retry
        # with ``force_fresh=True`` (one-shot, not infinite).
        if mode == "ab" and response.status_code == 416:
            progress.update(-local_size)
            if outfile.exists():
                outfile.unlink()
            raise _RetryFreshError(
                "HTTP 416 -- resetting partial download for a fresh GET"
            )
        if policy.should_retry_status(response.status_code):
            raise _RetryableError(f"HTTP {response.status_code}")
        if response.is_error:
            raise TransferError(
                f"HTTP {response.status_code} while downloading {file.path}"
            )
        if mode == "ab" and response.status_code != 206:
            # Server returned 200 instead of 206 — it does not honour the Range
            # request and will send the full file. Back out the progress we
            # already credited for the local partial bytes so the bar does not
            # overshoot the true file size.
            progress.update(-local_size)
            mode = "wb"

        with outfile.open(mode) as handle:
            previous = response.num_bytes_downloaded
            for chunk in response.iter_bytes():
                handle.write(chunk)
                current = response.num_bytes_downloaded
                progress.update(current - previous)
                previous = current
