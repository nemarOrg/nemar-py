"""Download NEMAR datasets from ``data.nemar.org``.

The flow is:

download
  NEMARClient.fetch_index
  NEMARClient.fetch_metadata
  NEMARClient.fetch_manifest
  select_files → raise_if_unmatched_includes (inlined in _run)
  _check_local_compatibility (inlined in _run)
  detect_case_collisions (inlined in _run)
  select_backend → TransferBackend.transfer  (see :mod:`nemar._transfer`)

Metadata-phase fetches go through :class:`nemar._client.NEMARClient`, which
owns one ``httpx.Client`` (and one TLS session) per ``download()`` call and
delegates the retry loop + redirect-origin check to
:func:`nemar._transport.fetch_json`. Bytes-on-the-wire concerns (the
HTTPS adapter, the per-file retry loop) live in :mod:`nemar._transfer`.
This module is the orchestrator only.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from nemar import __version__
from nemar._client import NEMARClient
from nemar._models import (
    DatasetFile,
    DatasetIndex,
    DatasetVersion,
    VersionManifest,
)
from nemar._request import (
    DEFAULT_DATA_URL,
    DownloadRequest,
    _normalize_version_tag,
)
from nemar._selection import raise_if_unmatched_includes, select_files
from nemar._transfer import select_backend
from nemar._verification import (
    assert_all_present,
    detect_case_collisions,
    partition_pending,
)
from nemar.errors import (
    LocalTargetError,
    LocalVersionMismatchError,
    SelectionError,
)

_DATASET_DESCRIPTION_FILENAME = "dataset_description.json"
_DOI_PREFIX_TEMPLATE = "10.82901/nemar.{dataset}"


def _check_local_compatibility(
    target_path: Path, *, dataset: str, tag: str,
) -> None:
    """Raise if downloading ``dataset`` ``tag`` into ``target_path`` would clobber data.

    Returns silently in the three "no compatibility check needed" cases:

    1. ``target_path`` does not exist.
    2. ``target_path`` exists but is empty.
    3. ``target_path`` is non-empty but has no ``dataset_description.json``
       — the "let interrupted downloads resume" case. The orchestrator
       logs a resume notice for this branch via
       :func:`_target_has_files_without_description`.

    When a ``dataset_description.json`` is present:

    * Raises :class:`LocalTargetError` if the file is malformed (bad JSON,
      missing/non-string ``DatasetDOI``, non-string ``Version``) so silent
      corruption can never sneak through.
    * Raises :class:`LocalTargetError` if the local DOI prefix names a
      *different* dataset.
    * Raises :class:`LocalTargetError` if the DOI matches but no Version
      was recorded — continuing would risk silently mixing two versions
      of the same dataset.
    * Raises :class:`LocalVersionMismatchError` (a subclass of both
      :class:`LocalTargetError` and the builtin :class:`FileExistsError`)
      if the DOI matches but the Version is a different tag than the
      one requested.
    """
    if not target_path.exists():
        return
    if next(target_path.iterdir(), None) is None:
        return

    dataset_description_path = target_path / _DATASET_DESCRIPTION_FILENAME
    if not dataset_description_path.exists():
        return

    # ``utf-8-sig`` strips a UTF-8 BOM if present and is otherwise
    # equivalent to ``utf-8``. Some editors (legacy Windows Notepad,
    # Excel CSV-export pipelines) prepend a BOM that would otherwise
    # turn the first key into ``"﻿DatasetDOI"`` and silently fail
    # the DOI check.
    try:
        payload = json.loads(
            dataset_description_path.read_text(encoding="utf-8-sig")
        )
    except json.JSONDecodeError as exc:
        raise LocalTargetError(
            f"Could not parse local {dataset_description_path} as JSON."
        ) from exc

    doi = payload.get("DatasetDOI")
    if doi is None:
        raise LocalTargetError(
            'Local "dataset_description.json" does not contain "DatasetDOI".'
        )
    if not isinstance(doi, str):
        raise LocalTargetError('Local "DatasetDOI" must be a string.')
    local_doi = doi.removeprefix("doi:")
    local_dataset_id = local_doi.removeprefix("10.82901/nemar.")

    version = payload.get("Version")
    if version is not None and not isinstance(version, str):
        raise LocalTargetError('Local "Version" must be a string when present.')
    local_version_tag = (
        _normalize_version_tag(version) if version is not None else None
    )

    expected_doi = _DOI_PREFIX_TEMPLATE.format(dataset=dataset)
    canonical_local_doi = _DOI_PREFIX_TEMPLATE.format(dataset=local_dataset_id)
    if not canonical_local_doi.startswith(expected_doi):
        raise LocalTargetError(
            "The target directory appears to contain a different NEMAR dataset. "
            f'Local DatasetDOI: "{canonical_local_doi}". '
            f"Requested dataset: {dataset}."
        )

    if local_version_tag is None:
        raise LocalTargetError(
            'Local "dataset_description.json" matches the requested dataset DOI '
            'but does not contain a "Version" field. This could lead to '
            "overwriting data from a different version. Use an empty target "
            "directory, or remove the existing files if you intend to "
            "re-download."
        )

    if local_version_tag != tag:
        raise LocalVersionMismatchError(
            f"You requested {dataset} {tag}, but {local_version_tag} exists "
            "locally in the target directory. Use an empty target directory "
            "or request the same version."
        )


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
    downloader: str = "auto",
    verify_hash: bool = True,
    verify_size: bool = True,
    max_retries: int = 5,
    max_concurrent_downloads: int = 16,
    metadata_timeout: float = 30.0,
    stream_timeout: float = 60.0,
    data_url: str = DEFAULT_DATA_URL,
    no_data: bool = False,
) -> None:
    """Download a public NEMAR dataset through ``data.nemar.org``.

    The explicit kwargs preserve IDE / type-checker introspection. The
    body delegates to :meth:`DownloadRequest.from_kwargs` for the
    normalization sweep, then to :func:`_run` for the algorithmic
    sequence (index → version → metadata → manifest → select →
    target-check → transfer).

    ``no_data=True`` switches to a metadata-only transfer: after BIDS
    selection, only files whose checksum is git-tracked (i.e. the
    small sidecars, JSON descriptors, TSVs, ``README`` etc.) are
    transferred. Files that ride the git-annex backend
    (``sha256`` / ``md5`` checksum) are dropped. Useful for catalog
    inspection, manifest verification, or pre-flight tooling that
    only needs the BIDS metadata without the heavy recording binaries.
    """
    request = DownloadRequest.from_kwargs(
        dataset=dataset,
        tag=tag,
        target_dir=target_dir,
        include=include,
        exclude=exclude,
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
        downloader=downloader,
        verify_hash=verify_hash,
        verify_size=verify_size,
        max_retries=max_retries,
        max_concurrent_downloads=max_concurrent_downloads,
        metadata_timeout=metadata_timeout,
        stream_timeout=stream_timeout,
        data_url=data_url,
        no_data=no_data,
    )
    _run(request)


@dataclass(frozen=True)
class _MetadataResult:
    """The output of the metadata-fetch phase of :func:`_run`.

    Carries only the three values the downstream phases consume: the
    resolved version tag, the optional DataLad sibling URL, and the
    parsed manifest. ``index`` and ``version`` stay local to
    :func:`_fetch_metadata`.
    """

    selected_tag: str
    datalad_url: str | None
    manifest: VersionManifest


def _fetch_metadata(request: DownloadRequest) -> _MetadataResult:
    """Fetch index → resolve version → fetch metadata + manifest.

    The ``NEMARClient`` owns the metadata ``httpx.Client`` for the three
    fetches; the transfer phase runs after the context closes so the
    metadata connection pool is released before the (possibly long)
    file transfer begins.
    """
    with NEMARClient(
        data_url=request.endpoint.url,
        metadata_timeout=request.metadata_timeout,
        max_retries=request.retry.max_attempts - 1,
    ) as client:
        index = client.fetch_index(request.dataset)
        version = index.resolve_version(request.requested_tag)
        client.fetch_metadata(index)
        manifest = client.fetch_manifest(index, version)
    return _MetadataResult(
        selected_tag=version.version,
        datalad_url=index.datalad_url,
        manifest=manifest,
    )


def _select_files_for_request(
    request: DownloadRequest, files: list[DatasetFile]
) -> list[DatasetFile]:
    """Apply the BIDS query + include/exclude + ``no_data`` filter.

    Raises :class:`~nemar.errors.SelectionError` (via
    :func:`~nemar._selection.raise_if_unmatched_includes`) when an
    include pattern matched nothing. ``no_data=True`` keeps only
    git-tracked sidecars (drops annexed binaries).
    """
    result = select_files(
        files,
        query=request.bids_query,
        include=list(request.include_patterns),
        exclude=list(request.exclude_patterns),
    )
    raise_if_unmatched_includes(result, filenames=[file.path for file in files])
    selected_files = list(result.selected)
    if request.no_data:
        selected_files = [
            file for file in selected_files if file.git_sha1 is not None
        ]
    return selected_files


def _transfer_and_verify(
    request: DownloadRequest,
    selected_files: list[DatasetFile],
    *,
    datalad_url: str | None,
    selected_tag: str,
    manifest_count: int,
) -> None:
    """Guard the target, transfer the pending files, then verify all.

    Local-compatibility check + case-collision guard run before any
    bytes move. ``partition_pending`` trusts size only so an idempotent
    re-run does not re-hash the whole dataset; the final
    :func:`~nemar._verification.assert_all_present` is the real hash gate
    over every selected file.
    """
    _check_local_compatibility(
        request.target_path, dataset=request.dataset, tag=selected_tag
    )
    if _target_has_files_without_description(request.target_path):
        tqdm.write(
            "Target directory is not empty and has no dataset_description.json. "
            "Continuing so interrupted downloads can resume."
        )
    request.target_path.mkdir(parents=True, exist_ok=True)
    collisions = detect_case_collisions(
        selected_files, target_dir=request.target_path
    )
    if collisions:
        examples = "; ".join(f"{a} <-> {b}" for a, b in collisions[:5])
        raise SelectionError(
            "The NEMAR manifest contains paths that collide on a "
            "case-insensitive filesystem at the target directory. Move the "
            "download to a case-sensitive volume, or contact the dataset "
            f"maintainer. Examples: {examples}"
        )

    tqdm.write(
        "Retrieving "
        f"{len(selected_files)} of {manifest_count} manifest files "
        f"({request.transfer.max_concurrent_downloads} concurrent downloads)."
    )
    pending = partition_pending(
        selected_files,
        target_dir=request.target_path,
        policy=request.verify,
        pre_transfer=True,
    )
    if pending:
        backend = select_backend(
            request.transfer,
            dataset=request.dataset,
            datalad_url=datalad_url,
            revision=selected_tag,
        )
        backend.transfer(
            pending,
            target_dir=request.target_path,
            options=request.transfer,
            verify=request.verify,
            retry=request.retry,
        )
    else:
        tqdm.write("All selected files already exist locally.")
    assert_all_present(
        selected_files,
        target_dir=request.target_path,
        policy=request.verify,
    )
    tqdm.write(f"Finished downloading {request.dataset} {selected_tag}.")


def _run(request: DownloadRequest) -> None:
    """Execute one normalized :class:`DownloadRequest` in three phases.

    1-4. :func:`_fetch_metadata` — index, version, metadata, manifest.
    5.   :func:`_select_files_for_request` — BIDS selection + ``no_data``.
    6.   :func:`_transfer_and_verify` — target guards, transfer, verify.
    """
    tqdm.write(f"This is nemar-py {__version__}.")
    tqdm.write(
        f"Preparing to download {request.dataset} from {request.endpoint.url}"
    )
    meta = _fetch_metadata(request)
    files = list(meta.manifest)
    selected_files = _select_files_for_request(request, files)
    _transfer_and_verify(
        request,
        selected_files,
        datalad_url=meta.datalad_url,
        selected_tag=meta.selected_tag,
        manifest_count=len(files),
    )


def fetch_dataset_index(
    *,
    dataset: str,
    data_url: str = DEFAULT_DATA_URL,
    metadata_timeout: float = 30.0,
    max_retries: int = 5,
) -> DatasetIndex:
    """Return the version index advertised by the NEMAR data endpoint.

    Thin wrapper that constructs a one-shot :class:`NEMARClient`, fetches
    the index, and closes. Callers that hold a NEMARClient across many
    datasets should call :meth:`NEMARClient.fetch_index` directly to
    amortize the TLS / connection-pool cost.
    """
    with NEMARClient(
        data_url=data_url,
        metadata_timeout=metadata_timeout,
        max_retries=max_retries,
    ) as client:
        return client.fetch_index(dataset)


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


def _target_has_files_without_description(target_dir: Path) -> bool:
    """Report whether ``target_dir`` has files but no ``dataset_description.json``.

    The case where the orchestrator logs the resume-friendly notice
    after :func:`_check_local_compatibility` returns silently. Kept
    separate because it is a fact about the directory's contents, not
    about the dataset identity — the compatibility checker treats
    these three "nothing to compare" cases (missing dir, empty dir,
    no description) as indistinguishable, but the orchestrator wants
    to log only the third.
    """
    if not target_dir.exists():
        return False
    if next(target_dir.iterdir(), None) is None:
        return False
    return not (target_dir / _DATASET_DESCRIPTION_FILENAME).exists()

