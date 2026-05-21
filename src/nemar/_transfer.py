"""Transfer backends — the bytes-on-the-wire phase of one download request.

The orchestrator in :mod:`nemar._download` is policy and choreography
(index, version, metadata, manifest, selection, target check); this
module is the single seam where the bytes actually move. Two adapters
of one seam:

* :class:`Aria2Backend` shells out to the ``aria2c`` subprocess. It is
  preferred for speed (multiplexed connections, native parallelism) and
  is selected automatically when ``aria2c`` is on ``PATH``.
* :class:`PythonBackend` uses a thread pool over a shared
  ``httpx.Client`` with a small connection pool. It is the always-available
  fallback and exercises the same Range/206 + 416-recovery + retry
  contracts the production endpoint sometimes needs.

Both implement the :class:`TransferBackend` protocol:

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
"aria2" | "python") and resolved by :func:`select_backend`. Explicit
``"aria2"`` with no ``aria2c`` on ``PATH`` raises
``RuntimeError('The "aria2" downloader requires aria2c on PATH.')`` --
that exact wording is part of the public contract because the public
error path is what library callers and the CLI surface to users.

Test hook
---------

``_ARIA2_EXTRA_ARGS`` is a module-level list of extra command-line
arguments appended to every ``aria2c`` invocation. It is empty in
production. Integration / e2e fixtures monkey-patch it (typically via
``monkeypatch.setattr(nemar._transfer, "_ARIA2_EXTRA_ARGS",
["--check-certificate=false"])``) when targeting a local HTTPS fixture
signed by a private CA -- aria2c on macOS uses AppleTLS, which silently
ignores ``--ca-certificate``. The hook lives here, next to the only
place that consumes it. Not part of the public API.
"""

from __future__ import annotations

import concurrent.futures
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import httpx
from tqdm.auto import tqdm

from nemar import __version__
from nemar._errors import TransferError, VerificationError
from nemar._models import DatasetFile
from nemar._request import TransferOptions
from nemar._retry import RetryPolicy
from nemar._verification import (
    VerifyPolicy,
    VerifyResult,
    _describe_failure,
)
from nemar._verification import check as _verify_check

# Private hook: extra command-line arguments appended to every ``aria2c``
# invocation. Empty in production. Test fixtures monkey-patch this to
# inject flags such as ``--check-certificate=false`` when targeting the
# local HTTPS fixture server (aria2c on macOS uses AppleTLS, which
# silently ignores ``--ca-certificate`` and rejects the self-signed
# chain otherwise). Not part of the public API.
_ARIA2_EXTRA_ARGS: list[str] = []


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

        Raises ``RuntimeError`` on any irrecoverable failure. The
        orchestrator does not catch it; it is the user-facing failure
        path.
        """


def select_backend(options: TransferOptions) -> TransferBackend:
    """Resolve ``options.backend`` against the runtime environment.

    * ``"aria2"`` and ``aria2c`` on ``PATH`` → :class:`Aria2Backend`.
    * ``"aria2"`` and no ``aria2c`` → ``RuntimeError`` with the
      preserved message ``'The "aria2" downloader requires aria2c on
      PATH.'``.
    * ``"auto"`` → :class:`Aria2Backend` when ``aria2c`` is present,
      otherwise :class:`PythonBackend` (with a one-line notice to
      ``tqdm.write``).
    * ``"python"`` → :class:`PythonBackend`.
    """
    has_aria2 = shutil.which("aria2c") is not None
    requested = options.backend
    if requested == "aria2" and not has_aria2:
        raise TransferError('The "aria2" downloader requires aria2c on PATH.')
    if requested == "aria2" or (requested == "auto" and has_aria2):
        return Aria2Backend()
    if requested == "auto":
        tqdm.write("aria2c was not found on PATH; using the Python downloader.")
    return PythonBackend()


class Aria2Backend:
    """Adapter that drives the ``aria2c`` subprocess.

    Writes the manifest to a temp input file and shells out. The
    aria2c command-line flags are pinned: ``--continue=true`` and
    ``--auto-file-renaming=false`` give resume semantics, the
    ``--max-tries`` / ``--retry-wait`` pair lets aria2c own its own
    retry loop with attempts that mirror the Python loop's count, and
    the ``--split`` cap keeps ``max_concurrent_downloads * split``
    bounded at 32 to avoid connection storms on servers and corporate
    proxies that enforce low per-host connection limits.

    The ``SSL_CERT_FILE`` environment variable is forwarded to
    ``--ca-certificate`` because aria2c does not honour the variable
    the way ``httpx`` / ``certifi`` do. The module-level
    :data:`_ARIA2_EXTRA_ARGS` list is appended last for test
    instrumentation.
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
        for file in files:
            (target_dir / file.path).parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", delete=False
        ) as input_file:
            input_path = Path(input_file.name)
            for file in files:
                input_file.write(f"{file.url}\n")
                input_file.write(f"  out={file.path}\n")
                checksum = _aria2_checksum(file) if verify.verify_hash else None
                if checksum is not None:
                    input_file.write(f"  checksum={checksum}\n")

        max_concurrent_downloads = options.max_concurrent_downloads
        cmd = [
            "aria2c",
            "--continue=true",
            "--auto-file-renaming=false",
            "--allow-overwrite=true",
            "--conditional-get=true",
            "--file-allocation=none",
            "--remote-time=true",
            f"--max-tries={retry.max_attempts}",
            "--retry-wait=2",
            f"--max-concurrent-downloads={max_concurrent_downloads}",
            f"--max-connection-per-server={max_concurrent_downloads}",
            # Cap split so that max_concurrent_downloads * split <= 32,
            # preventing connection storms on servers and corporate
            # proxies that enforce low per-host connection limits
            # (commonly 32-64).
            f"--split={max(1, min(16, 32 // max_concurrent_downloads))}",
            "--min-split-size=1M",
            f"--dir={target_dir}",
            f"--input-file={input_path}",
        ]
        # ``aria2c`` does not honour ``SSL_CERT_FILE`` the way ``httpx`` /
        # ``certifi`` do. When callers pre-configure a custom CA bundle
        # through the environment (e.g. corporate proxies, on-premise
        # mirrors, local test fixtures using a private CA), forward it
        # explicitly so the aria2 backend can verify the chain against
        # the same trust store as the rest of the client. Harmless
        # against the public NEMAR endpoint which uses a publicly-trusted
        # certificate.
        ssl_cert_file = os.environ.get("SSL_CERT_FILE")
        if ssl_cert_file:
            cmd.append(f"--ca-certificate={ssl_cert_file}")
        cmd.extend(_ARIA2_EXTRA_ARGS)
        try:
            subprocess.run(cmd, check=True, timeout=options.aria2_timeout)
        except subprocess.TimeoutExpired as exc:
            raise TransferError(
                f"aria2c timed out after {options.aria2_timeout} seconds."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise TransferError(
                "aria2c failed to download the selected NEMAR files."
            ) from exc
        finally:
            input_path.unlink(missing_ok=True)
        # The orchestrator runs ``assert_all_present`` with the caller's
        # full verify policy (size + hash) after this function returns
        # -- the aria2 path does not need a separate verify sweep.


def _aria2_checksum(file: DatasetFile) -> str | None:
    """Return the strongest available manifest checksum in aria2 syntax."""
    if file.sha256 is not None:
        return f"sha-256={file.sha256}"
    if file.md5 is not None:
        return f"md5={file.md5}"
    return None


class PythonBackend:
    """Adapter that streams files with ``httpx`` + a thread pool.

    Opens one shared ``httpx.Client`` for the whole batch so keepalive
    can amortize the TCP+TLS handshake across the thousands of small
    TSV/JSON files a BIDS dataset typically carries. The connection
    pool caps mirror the per-server connection budget the aria2 path
    already enforces (~2x concurrency cap so retries do not starve
    in-flight requests).

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
        # across the batch. The pool caps mirror the per-server
        # connection budget the aria2 path already enforces (~2x
        # concurrency cap so retries do not starve in-flight requests).
        #
        # TODO: enterprise users behind custom-CA MITM proxies would
        # benefit from ``truststore.SSLContext()`` here so the OS trust
        # store is honoured without an explicit ``SSL_CERT_FILE``.
        # Adding the truststore dependency complicates the wheel; defer
        # until there is a concrete user request. The ``verify`` knob
        # below is the hook.
        max_concurrent_downloads = options.max_concurrent_downloads
        max_retries = retry.max_attempts - 1
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
                            verify.verify_hash,
                            verify.verify_size,
                            max_retries,
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
    verify_hash: bool,
    verify_size: bool,
    max_retries: int,
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
    """
    outfile = target_dir / file.path
    outfile.parent.mkdir(parents=True, exist_ok=True)
    retry = RetryPolicy.default().with_attempts(max_retries)
    verify = VerifyPolicy(verify_size=verify_size, verify_hash=verify_hash)
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
    RuntimeError
        On irrecoverable transport failure (retries exhausted, or a
        non-retryable HTTP error such as 4xx that is not 416).

    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    effective_retry = retry if retry is not None else RetryPolicy.default()
    effective_verify = verify if verify is not None else VerifyPolicy()

    owns_client = client is None
    active_client = (
        client if client is not None else _build_oneshot_client(stream_timeout)
    )
    progress = tqdm(disable=True, leave=False)
    try:
        return _stream_with_retries(
            file,
            outfile=target_path,
            client=active_client,
            retry=effective_retry,
            verify=effective_verify,
            stream_timeout=stream_timeout,
            progress=progress,
        )
    finally:
        progress.close()
        if owns_client:
            active_client.close()


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
    Irrecoverable transport errors still raise ``RuntimeError`` after
    retries are exhausted.
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
    and ``RuntimeError`` for non-retryable HTTP errors.
    """
    if policy is None:
        policy = RetryPolicy.default()
    request_headers: dict[str, str] = {
        "accept": "*/*",
        "user-agent": f"nemar-py/{__version__}",
    }
    mode = "wb"
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
            local_size = 0

        with outfile.open(mode + "") as handle:
            previous = response.num_bytes_downloaded
            for chunk in response.iter_bytes():
                handle.write(chunk)
                current = response.num_bytes_downloaded
                progress.update(current - previous)
                previous = current
