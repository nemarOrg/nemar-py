"""Download NEMAR datasets from ``data.nemar.org``.

The flow is:

download
  NEMARClient.fetch_index
  NEMARClient.fetch_metadata
  NEMARClient.fetch_manifest
  SelectionPlan.build (inlined in _run)
  LocalDataset.from_dir(...).assert_compatible_with(...)
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
from nemar._errors import (
    LocalTargetError,
    LocalVersionMismatchError,
    SelectionError,
)
from nemar._models import (
    DatasetIndex,
    DatasetVersion,
)
from nemar._request import (
    DEFAULT_DATA_URL,
    DownloadRequest,
    _normalize_version_tag,
)
from nemar._selection import SelectionPlan
from nemar._transfer import select_backend
from nemar._verification import (
    assert_all_present,
    detect_case_collisions,
    partition_pending,
)

_DATASET_DESCRIPTION_FILENAME = "dataset_description.json"
_DOI_PREFIX_TEMPLATE = "10.82901/nemar.{dataset}"


@dataclass(frozen=True)
class LocalDataset:
    """The dataset identity recovered from a local target directory.

    Previously lived in ``nemar._local_dataset`` as the public-ish surface
    of a one-purpose module. Inlined here because :func:`_run` is its only
    caller, and the disk-I/O / pure-reasoning split survives intact: the
    classmethod is the only thing that touches the filesystem; instance
    methods are pure decisions over already-parsed fields.

    Construction is via :meth:`from_dir`, which returns ``None`` when the
    directory is empty / non-existent / has no ``dataset_description.json``
    — those are the cases where a fresh download has nothing to clash
    with and the orchestrator should simply proceed.

    A non-``None`` instance carries the parsed identity:

    * ``dataset_id`` — the bare dataset id (``"nm000132"``) recovered
      from the DOI prefix.
    * ``version_tag`` — the normalized version tag (``"v1.0.0"``), or
      ``None`` when the local description is silent on Version.

    All compatibility decisions live in :meth:`assert_compatible_with`,
    which is pure: it never re-reads the filesystem.
    """

    dataset_id: str
    version_tag: str | None

    @classmethod
    def from_dir(cls, path: Path) -> LocalDataset | None:
        """Inspect ``path`` and return a :class:`LocalDataset` if one exists.

        Returns ``None`` in three "no compatibility check needed" cases:

        1. ``path`` does not exist.
        2. ``path`` exists but is empty.
        3. ``path`` is non-empty but has no ``dataset_description.json``
           (the "let interrupted downloads resume" case — callers
           still want to log this; see the resume notice in :func:`_run`).

        Returns a populated instance when ``dataset_description.json``
        is present. Raises :class:`LocalTargetError` if the file is present
        but malformed (bad JSON, missing/non-string ``DatasetDOI``,
        non-string ``Version``) so silent corruption can never sneak
        through.
        """
        if not path.exists():
            return None
        if next(path.iterdir(), None) is None:
            return None

        dataset_description_path = path / _DATASET_DESCRIPTION_FILENAME
        if not dataset_description_path.exists():
            return None

        # ``utf-8-sig`` strips a UTF-8 BOM if present and is otherwise
        # equivalent to ``utf-8``. Some editors (legacy Windows
        # Notepad, Excel CSV-export pipelines) prepend a BOM that
        # would otherwise turn the first key into ``"﻿DatasetDOI"``
        # and silently fail the DOI check.
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
        doi = doi.removeprefix("doi:")

        dataset_id = doi.removeprefix("10.82901/nemar.")

        version = payload.get("Version")
        if version is not None and not isinstance(version, str):
            raise LocalTargetError('Local "Version" must be a string when present.')
        version_tag = _normalize_version_tag(version) if version is not None else None

        return cls(dataset_id=dataset_id, version_tag=version_tag)

    def assert_compatible_with(self, *, dataset: str, tag: str) -> None:
        """Raise if downloading ``dataset`` ``tag`` here would clobber data.

        * Raises :class:`LocalTargetError` if the local DOI prefix names a
          *different* dataset.
        * Raises :class:`LocalTargetError` if the DOI matches but no
          Version was recorded — continuing would risk silently
          mixing two versions of the same dataset.
        * Raises :class:`LocalVersionMismatchError` (a subclass of both
          :class:`LocalTargetError` and the builtin :class:`FileExistsError`)
          if the DOI matches but the Version is a different tag than the
          one requested.

        Pure: no filesystem access. ``self`` already carries the
        parsed identity.
        """
        expected_doi = _DOI_PREFIX_TEMPLATE.format(dataset=dataset)
        local_doi = _DOI_PREFIX_TEMPLATE.format(dataset=self.dataset_id)
        if not local_doi.startswith(expected_doi):
            raise LocalTargetError(
                "The target directory appears to contain a different NEMAR dataset. "
                f'Local DatasetDOI: "{local_doi}". Requested dataset: {dataset}.'
            )

        if self.version_tag is None:
            raise LocalTargetError(
                'Local "dataset_description.json" matches the requested dataset DOI '
                'but does not contain a "Version" field. This could lead to '
                "overwriting data from a different version. Use an empty target "
                "directory, or remove the existing files if you intend to "
                "re-download."
            )

        if self.version_tag != tag:
            raise LocalVersionMismatchError(
                f"You requested {dataset} {tag}, but {self.version_tag} exists "
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
) -> None:
    """Download a public NEMAR dataset through ``data.nemar.org``.

    The explicit kwargs preserve IDE / type-checker introspection. The
    body delegates to :meth:`DownloadRequest.from_kwargs` for the
    normalization sweep, then to :func:`_run` for the algorithmic
    sequence (index → version → metadata → manifest → select →
    target-check → transfer).
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
    )
    _run(request)


def _run(request: DownloadRequest) -> None:
    """Execute one normalized :class:`DownloadRequest`.

    The six algorithmic steps in order:

    1. Fetch the dataset index.
    2. Resolve the requested version.
    3. Fetch the dataset metadata document (if advertised).
    4. Fetch the version manifest payload.
    5. Parse the manifest, select files, assert target compatibility,
       and guard against case-insensitive-filesystem collisions.
    6. Transfer the selected files.
    """
    data_url = request.endpoint.url

    tqdm.write(f"This is nemar-py {__version__}.")
    tqdm.write(f"Preparing to download {request.dataset} from {data_url}")

    # The NEMARClient owns the metadata httpx.Client for the three
    # index/metadata/manifest fetches. The transfer phase runs after the
    # context closes so the metadata connection pool is released before
    # the (possibly long) file transfer phase begins.
    with NEMARClient(
        data_url=data_url,
        metadata_timeout=request.metadata_timeout,
        max_retries=request.retry.max_attempts - 1,
    ) as client:
        index = client.fetch_index(request.dataset)
        version = index.resolve_version(request.requested_tag)
        selected_tag = version.version
        datalad_url = index.datalad_url
        client.fetch_metadata(index)
        manifest = client.fetch_manifest(index, version)

    files = list(manifest)
    plan = SelectionPlan.build(
        files,
        query=request.bids_query,
        include=list(request.include_patterns),
        exclude=list(request.exclude_patterns),
    )
    plan.raise_if_unmatched_includes(filenames=[file.path for file in files])
    selected_files = list(plan.final)
    local = LocalDataset.from_dir(request.target_path)
    if local is not None:
        local.assert_compatible_with(dataset=request.dataset, tag=selected_tag)
    elif _target_has_files_without_description(request.target_path):
        # Resume-friendly behavior: a non-empty target without a
        # ``dataset_description.json`` (e.g. a previous interrupted
        # download) is allowed to proceed. Log it so operators can
        # tell at a glance why no compatibility check ran.
        tqdm.write(
            "Target directory is not empty and has no dataset_description.json. "
            "Continuing so interrupted downloads can resume."
        )
    # Refuse to start a download whose manifest entries would silently
    # overwrite each other on a case-insensitive target filesystem
    # (HFS+ / APFS in default mode / NTFS). Detected at transfer-prep
    # time rather than at manifest-parse time because the answer
    # depends on the target volume, which the parser does not know.
    # The probe runs against the actual target volume because APFS can
    # be either case-sensitive or case-insensitive on the same OS.
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
        f"{len(selected_files)} of {len(files)} manifest files "
        f"({request.transfer.max_concurrent_downloads} concurrent downloads)."
    )
    # Trust size pre-transfer; the post-transfer ``assert_all_present`` is
    # the real gate that re-checks the hash on every file. Hashing every
    # already-present file before the network does anything would re-read
    # the entire dataset off disk on every idempotent re-run.
    pending = partition_pending(
        selected_files,
        target_dir=request.target_path,
        policy=request.verify,
        pre_transfer=True,
    )
    if pending:
        backend = select_backend(
            request.transfer,
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
    # Final correctness sweep. Runs over the FULL manifest (not just the
    # files we transferred) because the pre-transfer partition trusts
    # size only. This is the real hash gate that catches a
    # right-size-wrong-content file on disk.
    assert_all_present(
        selected_files,
        target_dir=request.target_path,
        policy=request.verify,
    )
    tqdm.write(f"Finished downloading {request.dataset} {selected_tag}.")


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

    The case where the orchestrator logs the resume-friendly notice and
    proceeds without a compatibility check. Kept separate from
    :class:`LocalDataset` because it is a fact about the directory's
    contents, not about the dataset identity.
    """
    if not target_dir.exists():
        return False
    if next(target_dir.iterdir(), None) is None:
        return False
    return not (target_dir / "dataset_description.json").exists()

