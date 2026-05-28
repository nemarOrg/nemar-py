"""The HTTPS streaming engine — one ``TransferBackend`` adapter + primitive.

This module is the bytes-on-the-wire mechanics: a thread-pooled
``httpx`` adapter (:class:`PythonBackend`), its per-file retry driver
(:func:`_stream_with_retries`), one HTTP attempt with Range/206 resume
and 416 recovery (:func:`_transfer_one_attempt`), and the public
single-file primitive (:func:`download_one`).

It depends only on the seam (:mod:`nemar._backend`) and leaf modules
(``_models``, ``_retry``, ``_verification``, ``_endpoint``, ``errors``).
It knows nothing about backend *selection* — that lives in
:mod:`nemar._transfer`, which imports :class:`PythonBackend` from here.

The per-file retry driver raises two control-flow signals,
:class:`~nemar._retry._RetryableError` and
:class:`~nemar._retry._RetryFreshError`, imported from :mod:`nemar._retry`.
These were once a private copy local to this module; they now live in
``_retry`` and are shared with the JSON-fetch loop in
:mod:`nemar._transport` so both retry loops own the same error
vocabulary rather than maintaining disjoint duplicates.
"""

from __future__ import annotations

import concurrent.futures
import time
from collections.abc import Sequence
from pathlib import Path

import httpx
from tqdm.auto import tqdm

from nemar._backend import TransferOptions
from nemar._endpoint import DataEndpoint
from nemar._models import DatasetFile
from nemar._retry import RetryPolicy, _RetryableError, _RetryFreshError
from nemar._verification import (
    VerifyPolicy,
    VerifyResult,
    _describe_failure,
)
from nemar._verification import check as _verify_check
from nemar._version import __version__
from nemar.errors import TransferError, VerificationError


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
        :meth:`~nemar._models.VersionManifest.parse` are already origin-scoped;
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
