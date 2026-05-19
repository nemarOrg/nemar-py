"""Download NEMAR datasets from ``data.nemar.org``.

The flow is:

download
  _fetch_dataset_index
  _fetch_dataset_metadata
  _fetch_version_manifest
  _select_bids_files
  _assert_target_matches_dataset_version
  _transfer_files
    _transfer_with_aria2 or _transfer_with_python
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urljoin

import httpx
from tqdm.auto import tqdm

from nemar import __version__, _bids
from nemar._endpoint import DataEndpoint
from nemar._models import (
    DatasetFile,
    DatasetIndex,
    DatasetVersion,
    VersionManifest,
    parse_dataset_index,
)
from nemar._retry import RetryPolicy, _next_backoff
from nemar._selection import SelectionPlan
from nemar._verification import (
    VerifyPolicy,
    VerifyResult,
    assert_all_present,
    detect_case_collisions,
    file_hash,
    partition_pending,
)
from nemar._verification import check as _verify_check

DEFAULT_DATA_URL = "https://data.nemar.org/"
DATASET_ID_RE = re.compile(r"^nm\d{6}$")

# Module-level aliases kept for the streaming retry loop in
# ``_transfer_one_attempt`` / ``_transfer_one_with_python``.
_DEFAULT_POLICY = RetryPolicy.default()
RETRY_STATUS_CODES: frozenset[int] = _DEFAULT_POLICY.retryable_status
RETRY_EXCEPTIONS = _DEFAULT_POLICY.retryable_exceptions

TransferBackend = Literal["auto", "aria2", "python"]

# Private hook: extra command-line arguments appended to every ``aria2c``
# invocation. Empty in production. Test fixtures monkey-patch this to inject
# flags such as ``--check-certificate=false`` when targeting the local HTTPS
# fixture server (aria2c on macOS uses AppleTLS, which silently ignores
# ``--ca-certificate`` and rejects the self-signed chain otherwise).
# Not part of the public API.
_ARIA2_EXTRA_ARGS: list[str] = []


class _RetryableError(Exception):
    """Raised when a request can be retried."""


class _RetryFreshError(_RetryableError):
    """Raised when the retry must abandon any local partial bytes.

    Subclass of :class:`_RetryableError` so existing handlers still treat
    it as retryable; the per-file driver distinguishes it to set
    ``force_fresh=True`` on the next attempt (S7).
    """


def download(
    *,
    dataset: str,
    tag: str | None = None,
    target_dir: Path | str | None = None,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    subject: str | Iterable[str] | None = None,
    session: str | Iterable[str] | None = None,
    task: str | Iterable[str] | None = None,
    run: str | Iterable[str] | None = None,
    acquisition: str | Iterable[str] | None = None,
    datatype: str | Iterable[str] | None = None,
    suffix: str | Iterable[str] | None = None,
    extension: str | Iterable[str] | None = None,
    scope: str | Iterable[str] | None = None,
    pipeline: str | Iterable[str] | None = None,
    entity: Mapping[str, Any] | Iterable[str] | None = None,
    downloader: TransferBackend | str = "auto",
    verify_hash: bool = True,
    verify_size: bool = True,
    max_retries: int = 5,
    max_concurrent_downloads: int = 16,
    metadata_timeout: float = 30.0,
    stream_timeout: float = 60.0,
    aria2_timeout: float | None = None,
    data_url: str = DEFAULT_DATA_URL,
) -> None:
    """Download a public NEMAR dataset through ``data.nemar.org``."""
    _validate_download_options(
        dataset=dataset,
        downloader=downloader,
        max_retries=max_retries,
        max_concurrent_downloads=max_concurrent_downloads,
        data_url=data_url,
    )

    endpoint = DataEndpoint.from_url(data_url)
    data_url = endpoint.url
    requested_tag = _normalize_version_tag(tag) if tag is not None else None
    target_path = Path(dataset if target_dir is None else target_dir).expanduser()
    target_path = target_path.resolve()
    include_patterns = _normalize_bids_patterns(include)
    exclude_patterns = _normalize_bids_patterns(exclude)
    bids_query = _bids.BidsQuery.from_filters(
        subject=subject,
        session=session,
        task=task,
        run=run,
        acquisition=acquisition,
        datatype=datatype,
        suffix=suffix,
        extension=extension,
        scope=scope,
        pipeline=pipeline,
        entity=entity,
    )

    tqdm.write(f"This is nemar-py {__version__}.")
    tqdm.write(f"Preparing to download {dataset} from {data_url}")

    with httpx.Client(
        follow_redirects=True,
        headers={
            "accept": "application/json",
            "user-agent": f"nemar-py/{__version__}",
        },
        timeout=metadata_timeout,
    ) as client:
        index = _fetch_dataset_index(
            client,
            dataset=dataset,
            data_url=data_url,
            max_retries=max_retries,
            endpoint=endpoint,
        )
        version = index.resolve_version(requested_tag)
        selected_tag = version.version
        _fetch_dataset_metadata(
            client,
            index=index,
            data_url=data_url,
            max_retries=max_retries,
            endpoint=endpoint,
        )
        manifest_url = endpoint.url_for(version.manifest_url)
        manifest_payload = _fetch_version_manifest(
            client,
            dataset=dataset,
            version=version,
            manifest_url=manifest_url,
            max_retries=max_retries,
            endpoint=endpoint,
        )

    manifest = VersionManifest.parse(
        manifest_payload,
        manifest_url=manifest_url,
        endpoint=endpoint,
    )
    files = list(manifest)
    selected_files = _select_bids_files(
        files,
        query=bids_query,
        include=include_patterns,
        exclude=exclude_patterns,
    )
    _assert_target_matches_dataset_version(
        dataset=dataset,
        tag=selected_tag,
        target_dir=target_path,
    )
    # S4: refuse to start a download whose manifest entries would silently
    # overwrite each other on a case-insensitive target filesystem
    # (HFS+ / APFS in default mode / NTFS). Detected at transfer-prep
    # time rather than at manifest-parse time because the answer
    # depends on the target volume, which the parser does not know.
    _assert_no_case_collisions(selected_files, target_dir=target_path)

    tqdm.write(
        "Retrieving "
        f"{len(selected_files)} of {len(files)} manifest files "
        f"({max_concurrent_downloads} concurrent downloads)."
    )
    _transfer_files(
        selected_files,
        target_dir=target_path,
        downloader=str(downloader),
        verify_hash=verify_hash,
        verify_size=verify_size,
        max_retries=max_retries,
        max_concurrent_downloads=max_concurrent_downloads,
        stream_timeout=stream_timeout,
        aria2_timeout=aria2_timeout,
    )
    tqdm.write(f"Finished downloading {dataset} {selected_tag}.")


def fetch_dataset_index(
    *,
    dataset: str,
    data_url: str = DEFAULT_DATA_URL,
    metadata_timeout: float = 30.0,
    max_retries: int = 5,
) -> DatasetIndex:
    """Return the version index advertised by the NEMAR data endpoint."""
    _validate_endpoint_query_options(
        dataset=dataset,
        data_url=data_url,
        max_retries=max_retries,
    )
    endpoint = DataEndpoint.from_url(data_url)
    with httpx.Client(
        follow_redirects=True,
        headers={
            "accept": "application/json",
            "user-agent": f"nemar-py/{__version__}",
        },
        timeout=metadata_timeout,
    ) as client:
        return _fetch_dataset_index(
            client,
            dataset=dataset,
            data_url=endpoint.url,
            max_retries=max_retries,
            endpoint=endpoint,
        )


def list_dataset_versions(
    *,
    dataset: str,
    data_url: str = DEFAULT_DATA_URL,
    metadata_timeout: float = 30.0,
    max_retries: int = 5,
) -> list[DatasetVersion]:
    """Return versions advertised by the NEMAR data endpoint."""
    return fetch_dataset_index(
        dataset=dataset,
        data_url=data_url,
        metadata_timeout=metadata_timeout,
        max_retries=max_retries,
    ).versions


def _validate_download_options(
    *,
    dataset: str,
    downloader: str,
    max_retries: int,
    max_concurrent_downloads: int,
    data_url: str,
) -> None:
    if not DATASET_ID_RE.fullmatch(dataset):
        raise ValueError('dataset must look like "nm000132".')
    if downloader not in {"auto", "aria2", "python"}:
        raise ValueError('downloader must be one of "auto", "aria2", or "python".')
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative.")
    if max_concurrent_downloads < 1:
        raise ValueError("max_concurrent_downloads must be at least 1.")
    # Validates HTTPS and normalizes the trailing slash; we discard the value
    # here because the caller re-builds the endpoint in ``download``.
    DataEndpoint.from_url(data_url)


def _validate_endpoint_query_options(
    *, dataset: str, data_url: str, max_retries: int
) -> None:
    if not DATASET_ID_RE.fullmatch(dataset):
        raise ValueError('dataset must look like "nm000132".')
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative.")
    DataEndpoint.from_url(data_url)


def _normalize_data_url(data_url: str) -> str:
    return DataEndpoint.from_url(data_url).url


def _normalize_version_tag(tag: str) -> str:
    tag = tag.strip()
    if not tag:
        raise ValueError("tag must not be empty.")
    if tag[0].isdigit():
        return f"v{tag}"
    return tag


def _normalize_bids_patterns(patterns: Iterable[str] | None) -> list[str]:
    if patterns is None:
        return []
    if isinstance(patterns, str):
        return [patterns]
    return list(patterns)


def _fetch_dataset_index(
    client: httpx.Client,
    *,
    dataset: str,
    data_url: str,
    max_retries: int,
    endpoint: DataEndpoint | None = None,
) -> DatasetIndex:
    index_url = (
        endpoint.url_for(f"{dataset}/")
        if endpoint is not None
        else urljoin(data_url, f"{dataset}/")
    )
    payload = _fetch_json_with_retries(
        client,
        url=index_url,
        what=f"retrieving NEMAR index for {dataset}",
        max_retries=max_retries,
        endpoint=endpoint,
    )
    index = parse_dataset_index(payload)
    if index.dataset_id != dataset:
        raise RuntimeError(
            f"Requested {dataset}, but the NEMAR index described {index.dataset_id}."
        )
    return index


def _fetch_dataset_metadata(
    client: httpx.Client,
    *,
    index: DatasetIndex,
    data_url: str,
    max_retries: int,
    endpoint: DataEndpoint | None = None,
) -> dict[str, Any] | None:
    if index.metadata_url is None:
        return None
    payload = _fetch_json_with_retries(
        client,
        url=_resolve_data_url(data_url, index.metadata_url),
        what=f"retrieving NEMAR metadata for {index.dataset_id}",
        max_retries=max_retries,
        endpoint=endpoint,
    )
    if not isinstance(payload, dict):
        raise RuntimeError("The NEMAR metadata payload must be a JSON object.")
    return payload


def _fetch_version_manifest(
    client: httpx.Client,
    *,
    dataset: str,
    version: DatasetVersion,
    manifest_url: str,
    max_retries: int,
    endpoint: DataEndpoint | None = None,
) -> Any:
    try:
        return _fetch_json_with_retries(
            client,
            url=manifest_url,
            what=f"retrieving NEMAR manifest for {dataset} {version.version}",
            max_retries=max_retries,
            endpoint=endpoint,
        )
    except RuntimeError as exc:
        message = str(exc)
        if "Version not published" in message:
            raise RuntimeError(
                f"NEMAR advertises {dataset} {version.version} at "
                f"{manifest_url}, but that version is not published on the "
                "public data endpoint yet. This downloader only uses "
                "data.nemar.org and will not fall back to S3."
            ) from exc
        raise


def _fetch_json_with_retries(
    client: httpx.Client,
    *,
    url: str,
    what: str,
    max_retries: int,
    retry_backoff: float = 0.5,
    endpoint: DataEndpoint | None = None,
) -> Any:
    policy = RetryPolicy.default().with_attempts(max_retries)
    last_attempt = policy.max_attempts - 1
    for attempt in range(policy.max_attempts):
        try:
            response = client.get(url)
            if policy.should_retry_status(response.status_code):
                raise _RetryableError(f"HTTP {response.status_code}")
            if response.is_error:
                detail = _response_detail(response)
                raise RuntimeError(
                    f"Error when {what}: HTTP {response.status_code} {detail}"
                )
            # Validate the final URL after any redirects against the
            # configured data endpoint. ``httpx`` with ``follow_redirects=True``
            # would otherwise let a 302 to another host silently bypass the
            # ``data.nemar.org``-only scope.
            if endpoint is not None:
                endpoint.assert_within(str(response.url))
            return response.json()
        except policy.retryable_exceptions as exc:
            if attempt == last_attempt:
                raise RuntimeError(f"Network error when {what}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON when {what}: {url}") from exc
        except _RetryableError as exc:
            if attempt == last_attempt:
                raise RuntimeError(f"Retryable error when {what}: {exc}") from exc

        remaining = last_attempt - attempt
        tqdm.write(f"Retrying after failure when {what} ({remaining} retries remain).")
        time.sleep(_next_backoff(retry_backoff))
        retry_backoff *= 2

    raise RuntimeError(f"Unexpected retry exhaustion when {what}.")


def _response_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except json.JSONDecodeError:
        text = response.text.strip()
        return text[:300] if text else ""
    if isinstance(payload, dict):
        detail = payload.get("message") or payload.get("error")
        if detail:
            return str(detail)
    return json.dumps(payload)[:300]


def _resolve_data_url(data_url: str, value: str) -> str:
    return DataEndpoint.from_url(data_url).url_for(value)


def _select_bids_files(
    files: Sequence[DatasetFile],
    *,
    query: _bids.BidsQuery,
    include: list[str],
    exclude: list[str],
) -> list[DatasetFile]:
    """Thin shim over :class:`SelectionPlan` that preserves the prior interface."""
    plan = SelectionPlan.build(files, query=query, include=include, exclude=exclude)
    plan.raise_if_unmatched_includes(filenames=[file.path for file in files])
    return list(plan.final)


def _assert_target_matches_dataset_version(
    *, dataset: str, tag: str, target_dir: Path
) -> None:
    if not target_dir.exists():
        return
    if next(target_dir.iterdir(), None) is None:
        return

    dataset_description_path = target_dir / "dataset_description.json"
    if not dataset_description_path.exists():
        tqdm.write(
            "Target directory is not empty and has no dataset_description.json. "
            "Continuing so interrupted downloads can resume."
        )
        return

    try:
        dataset_description = json.loads(
            dataset_description_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Could not parse local {dataset_description_path} as JSON."
        ) from exc

    local_doi = dataset_description.get("DatasetDOI")
    if local_doi is None:
        raise RuntimeError(
            'Local "dataset_description.json" does not contain "DatasetDOI".'
        )
    if not isinstance(local_doi, str):
        raise RuntimeError('Local "DatasetDOI" must be a string.')
    local_doi = local_doi.removeprefix("doi:")

    expected_doi = f"10.82901/nemar.{dataset}"
    if not local_doi.startswith(expected_doi):
        raise RuntimeError(
            "The target directory appears to contain a different NEMAR dataset. "
            f'Local DatasetDOI: "{local_doi}". Requested dataset: {dataset}.'
        )

    local_version = dataset_description.get("Version")
    if local_version is None:
        raise RuntimeError(
            'Local "dataset_description.json" matches the requested dataset DOI but '
            'does not contain a "Version" field. This could lead to overwriting data '
            "from a different version. Use an empty target directory, or remove the "
            "existing files if you intend to re-download."
        )
    if not isinstance(local_version, str):
        raise RuntimeError('Local "Version" must be a string when present.')
    local_tag = _normalize_version_tag(local_version)
    if local_tag != tag:
        raise FileExistsError(
            f"You requested {dataset} {tag}, but {local_tag} exists locally in "
            "the target directory. Use an empty target directory or request the "
            "same version."
        )


def _assert_no_case_collisions(
    files: Sequence[DatasetFile], *, target_dir: Path
) -> None:
    """Refuse to start a download that would silently overwrite itself.

    Two manifest entries that differ only by case (``foo.bin`` and
    ``Foo.bin``) map to the same on-disk file on case-insensitive
    volumes (HFS+, APFS default, NTFS). The second download would
    silently overwrite the first -- data-loss with no warning. We
    raise here so the failure is visible.

    The probe runs against the actual target volume because APFS can
    be either case-sensitive or case-insensitive on the same OS.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    collisions = detect_case_collisions(files, target_dir=target_dir)
    if not collisions:
        return
    examples = "; ".join(f"{a} <-> {b}" for a, b in collisions[:5])
    raise RuntimeError(
        "The NEMAR manifest contains paths that collide on a "
        "case-insensitive filesystem at the target directory. Move the "
        "download to a case-sensitive volume, or contact the dataset "
        f"maintainer. Examples: {examples}"
    )


def _transfer_files(
    files: Sequence[DatasetFile],
    *,
    target_dir: Path,
    downloader: str,
    verify_hash: bool,
    verify_size: bool,
    max_retries: int,
    max_concurrent_downloads: int,
    stream_timeout: float,
    aria2_timeout: float | None = None,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    policy = VerifyPolicy(verify_size=verify_size, verify_hash=verify_hash)
    pending = partition_pending(files, target_dir=target_dir, policy=policy)
    if not pending:
        tqdm.write("All selected files already exist locally.")
        return

    selected_backend = _select_transfer_backend(downloader)
    if selected_backend == "aria2":
        _transfer_with_aria2(
            pending,
            target_dir=target_dir,
            verify_hash=verify_hash,
            verify_size=verify_size,
            max_retries=max_retries,
            max_concurrent_downloads=max_concurrent_downloads,
            aria2_timeout=aria2_timeout,
        )
    else:
        _transfer_with_python(
            pending,
            target_dir=target_dir,
            verify_hash=verify_hash,
            verify_size=verify_size,
            max_retries=max_retries,
            max_concurrent_downloads=max_concurrent_downloads,
            stream_timeout=stream_timeout,
        )

    # Final correctness sweep. Goes through the back-compat shim so that
    # tests which monkeypatch ``_verify_manifest_files`` still observe the
    # hook; production callers see the same behaviour as a direct
    # ``assert_all_present`` call.
    _verify_manifest_files(
        files,
        target_dir=target_dir,
        verify_hash=verify_hash,
        verify_size=verify_size,
    )


def _select_transfer_backend(requested: str) -> Literal["aria2", "python"]:
    has_aria2 = shutil.which("aria2c") is not None
    if requested == "aria2" and not has_aria2:
        raise RuntimeError('The "aria2" downloader requires aria2c on PATH.')
    if requested == "aria2" or (requested == "auto" and has_aria2):
        return "aria2"
    if requested == "auto":
        tqdm.write("aria2c was not found on PATH; using the Python downloader.")
    return "python"


def _files_missing_or_incomplete(
    files: Sequence[DatasetFile],
    *,
    target_dir: Path,
    verify_hash: bool,
    verify_size: bool,
) -> list[DatasetFile]:
    """Back-compat shim. Delegates to :func:`partition_pending`."""
    return partition_pending(
        files,
        target_dir=target_dir,
        policy=VerifyPolicy(verify_size=verify_size, verify_hash=verify_hash),
    )


def _local_file_satisfies_manifest(
    file: DatasetFile,
    *,
    target_dir: Path,
    verify_hash: bool,
    verify_size: bool,
) -> bool:
    """Back-compat shim. Delegates to :func:`nemar._verification.check`."""
    return (
        _verify_check(
            file,
            target_dir / file.path,
            VerifyPolicy(verify_size=verify_size, verify_hash=verify_hash),
        )
        is VerifyResult.OK
    )


def _transfer_with_aria2(
    files: Sequence[DatasetFile],
    *,
    target_dir: Path,
    verify_hash: bool,
    verify_size: bool,
    max_retries: int,
    max_concurrent_downloads: int,
    aria2_timeout: float | None = None,
) -> None:
    for file in files:
        (target_dir / file.path).parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as input_file:
        input_path = Path(input_file.name)
        for file in files:
            input_file.write(f"{file.url}\n")
            input_file.write(f"  out={file.path}\n")
            checksum = _aria2_checksum(file) if verify_hash else None
            if checksum is not None:
                input_file.write(f"  checksum={checksum}\n")

    # aria2c owns its own retry loop. We translate ``max_retries`` into
    # ``--max-tries`` (one initial attempt plus the requested retries) by
    # going through ``RetryPolicy`` so the two retry contracts read the same
    # number. ``--retry-wait`` is the aria2-side fixed wait between retries
    # (seconds), independent of the Python loop's exponential jittered curve.
    policy = RetryPolicy.default().with_attempts(max_retries)
    cmd = [
        "aria2c",
        "--continue=true",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        "--conditional-get=true",
        "--file-allocation=none",
        "--remote-time=true",
        f"--max-tries={policy.max_attempts}",
        "--retry-wait=2",
        f"--max-concurrent-downloads={max_concurrent_downloads}",
        f"--max-connection-per-server={max_concurrent_downloads}",
        # Cap split so that max_concurrent_downloads * split <= 32, preventing
        # connection storms on servers and corporate proxies that enforce low
        # per-host connection limits (commonly 32-64).
        f"--split={max(1, min(16, 32 // max_concurrent_downloads))}",
        "--min-split-size=1M",
        f"--dir={target_dir}",
        f"--input-file={input_path}",
    ]
    # ``aria2c`` does not honour ``SSL_CERT_FILE`` the way ``httpx`` /
    # ``certifi`` do. When callers pre-configure a custom CA bundle through
    # the environment (e.g. corporate proxies, on-premise mirrors, local
    # test fixtures using a private CA), forward it explicitly so the
    # aria2 backend can verify the chain against the same trust store as
    # the rest of the client. Harmless against the public NEMAR endpoint
    # which uses a publicly-trusted certificate.
    ssl_cert_file = os.environ.get("SSL_CERT_FILE")
    if ssl_cert_file:
        cmd.append(f"--ca-certificate={ssl_cert_file}")
    cmd.extend(_ARIA2_EXTRA_ARGS)
    try:
        subprocess.run(cmd, check=True, timeout=aria2_timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"aria2c timed out after {aria2_timeout} seconds.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "aria2c failed to download the selected NEMAR files."
        ) from exc
    finally:
        input_path.unlink(missing_ok=True)

    # S8: the outer ``_transfer_files`` already runs ``_verify_manifest_files``
    # with the caller's full verify policy (size + hash) after this
    # function returns. A second size-only sweep here would just re-stat
    # every file twice on the aria2 happy path. ``verify_size`` is kept
    # in the signature so existing callers and tests do not need to
    # change their kwargs.
    _ = verify_size


def _aria2_checksum(file: DatasetFile) -> str | None:
    if file.sha256 is not None:
        return f"sha-256={file.sha256}"
    if file.md5 is not None:
        return f"md5={file.md5}"
    return None


def _transfer_with_python(
    files: Sequence[DatasetFile],
    *,
    target_dir: Path,
    verify_hash: bool,
    verify_size: bool,
    max_retries: int,
    max_concurrent_downloads: int,
    stream_timeout: float,
) -> None:
    total = sum(file.size or 0 for file in files)
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
                    verify_hash,
                    verify_size,
                    max_retries,
                    progress,
                    stream_timeout,
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
                raise RuntimeError(
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
) -> None:
    outfile = target_dir / file.path
    outfile.parent.mkdir(parents=True, exist_ok=True)

    policy = RetryPolicy.default().with_attempts(max_retries)
    last_attempt = policy.max_attempts - 1
    backoff = policy.base_backoff
    # S7: after a 416 on a Range request the partial file is stale --
    # the server's view of the object has changed. We retry once with
    # ``force_fresh=True`` so the next attempt sends an unconditional
    # GET against a freshly-unlinked target.
    force_fresh = False
    for attempt in range(policy.max_attempts):
        try:
            _transfer_one_attempt(
                file,
                outfile=outfile,
                progress=progress,
                stream_timeout=stream_timeout,
                policy=policy,
                force_fresh=force_fresh,
            )
            _verify_manifest_file(
                file,
                outfile,
                verify_hash=verify_hash,
                verify_size=verify_size,
            )
            return
        except policy.retryable_exceptions as exc:
            if attempt == last_attempt:
                raise RuntimeError(f"Failed to download {file.path}: {exc}") from exc
            force_fresh = False
        except _RetryFreshError as exc:
            if attempt == last_attempt:
                raise RuntimeError(f"Failed to download {file.path}: {exc}") from exc
            force_fresh = True
        except _RetryableError as exc:
            if attempt == last_attempt:
                raise RuntimeError(f"Failed to download {file.path}: {exc}") from exc
            force_fresh = False

        time.sleep(_next_backoff(backoff))
        backoff *= 2


def _transfer_one_attempt(
    file: DatasetFile,
    *,
    outfile: Path,
    progress: tqdm,
    stream_timeout: float,
    policy: RetryPolicy | None = None,
    force_fresh: bool = False,
) -> None:
    if policy is None:
        policy = RetryPolicy.default()
    request_headers = {
        "accept": "*/*",
        "accept-encoding": "",
        "user-agent": f"nemar-py/{__version__}",
    }
    mode = "wb"
    # ``force_fresh`` (S7) discards any local partial bytes before sending
    # an unconditional GET. Used when the previous attempt got a 416 and
    # the local partial cannot represent a valid suffix of the current
    # object.
    if force_fresh and outfile.exists():
        outfile.unlink()
    local_size = outfile.stat().st_size if outfile.exists() else 0
    if not force_fresh and file.size is not None and 0 < local_size < file.size:
        request_headers["Range"] = f"bytes={local_size}-"
        mode = "ab"
        progress.update(local_size)
    elif file.size is not None and local_size > file.size:
        outfile.unlink()
    elif file.size is None and outfile.exists():
        outfile.unlink()

    with httpx.stream(
        "GET", file.url, headers=request_headers, timeout=stream_timeout
    ) as response:
        # S7: a 416 against a Range request means the server's view of
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
            raise RuntimeError(
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


def _verify_manifest_files(
    files: Sequence[DatasetFile],
    *,
    target_dir: Path,
    verify_hash: bool,
    verify_size: bool,
) -> None:
    """Back-compat shim. Delegates to :func:`assert_all_present`."""
    assert_all_present(
        files,
        target_dir=target_dir,
        policy=VerifyPolicy(verify_size=verify_size, verify_hash=verify_hash),
    )


def _verify_manifest_file(
    file: DatasetFile,
    outfile: Path,
    *,
    verify_hash: bool,
    verify_size: bool,
) -> None:
    """Back-compat shim. Verifies one file via :func:`assert_all_present`.

    Existing callers passed ``outfile`` directly rather than reconstructing
    it from ``target_dir / file.path``; preserve that by reverse-engineering
    a fake target dir from ``outfile``.
    """
    target_dir = outfile.parent
    # ``assert_all_present`` joins ``target_dir / file.path``. When the
    # caller already resolved ``outfile`` to an exact path we honour that
    # by passing a synthetic single-file relative path on a directory that
    # is the parent of ``outfile`` joined with the original file.path's
    # parent. The simplest correct shape: stage ``file`` under a
    # one-file dataset whose path equals ``outfile.name`` rooted at
    # ``outfile.parent``.
    from dataclasses import replace as _replace

    fileshim = _replace(file, path=outfile.name)
    assert_all_present(
        [fileshim],
        target_dir=target_dir,
        policy=VerifyPolicy(verify_size=verify_size, verify_hash=verify_hash),
    )


def _local_hash_matches_manifest(file: DatasetFile, outfile: Path) -> bool:
    """Back-compat shim. Delegates to ``_verification``."""
    if file.sha256 is not None:
        return file_hash(outfile, "sha256").lower() == file.sha256.lower()
    if file.md5 is not None:
        return file_hash(outfile, "md5").lower() == file.md5.lower()
    return True


def _file_hash(path: Path, algorithm: Literal["sha256", "md5"]) -> str:
    """Back-compat shim. Delegates to :func:`nemar._verification.file_hash`."""
    return file_hash(path, algorithm)
