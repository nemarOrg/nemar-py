"""The NEMAR S3 file-delivery contract.

The ``data.nemar.org`` metadata endpoint is a thin catalog + manifest
service; the actual file bytes live on S3. The contract below was
verified against the official DataLad clones of four representative
datasets (nm000132, nm000104, on005505, nm000133) and confirmed across
the full live catalog: every dataset's git-annex ``nemar-s3`` special
remote advertises the same bucket, region, and public host, with a
uniform per-dataset path prefix.

Bucket layout
-------------

Each dataset lives at ``s3://<NEMAR_S3_BUCKET>/<dataset>/`` with up to
four child prefixes. Three of them are addressable by ``(dataset,
version)`` alone and are exposed as helpers below:

``s3://nemar/<dataset>/``
├── ``objects/<annex-key>``           — content blobs (git-annex)
├── ``version/<version>.json``        — canonical compact manifest
├── ``version/<version>-summary.json`` — lightweight catalog summary
├── ``archives/<version>.zip``         — whole-dataset bundle
└── ``qa/`` (some datasets only)       — QA artifacts (not exposed here)

Where:

* ``NEMAR_S3_HOST`` / ``NEMAR_S3_BUCKET`` — single public host and
  bucket across the catalog.
* ``<dataset>`` — the bare dataset id (``nm000132``, ``on005505``).
  Both prefixes share one bucket; ``<dataset>/`` partitions it.
* ``<version>`` — the ``v<X.Y.Z>`` form (``v1.0.2``, ``v2.0.0``). The
  helpers accept ``"1.0.2"`` or ``"v1.0.2"`` interchangeably.
* ``<annex-key>`` — the git-annex content-addressed key,
  e.g. ``SHA256E-s<size>--<hex>.<ext>`` or
  ``MD5E-s<size>--<hex>.<ext>``. The dataset's
  ``.gitattributes annex.backend`` picks the algorithm; the size and
  extension stay encoded in the key for content-type negotiation.

Public-read, private-list
-------------------------

The bucket's *content* is public: unsigned ``GET`` / ``HEAD`` against
every URL the helpers below produce returns 200 OK. The bucket's
*index* is private: ``ListObjects`` requires NEMAR-internal AWS
credentials. Discovery therefore still goes through
``data.nemar.org/`` — you cannot enumerate the bucket yourself. Once
you know ``(dataset, version)`` (from the catalog endpoint, a DataLad
clone, or hard-coded knowledge), the helpers below let you skip the
``data.nemar.org`` round-trip entirely.

Why bypass ``data.nemar.org``?
------------------------------

The catalog endpoint serves pre-signed object URLs with
``X-Amz-Expires=3600`` and transforms the manifest into a flat array
on every request. Direct S3 access through these helpers has neither:
URLs do not expire, and the manifest is the compact dict-keyed-by-path
shape the canonical source uses.
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm.auto import tqdm

from nemar._backend import TransferOptions
from nemar._constants import (
    CHUNK_BYTES,
    MULTIPART_CHUNK_BYTES,
    MULTIPART_THRESHOLD_BYTES,
)
from nemar._models import DatasetFile
from nemar._retry import RetryPolicy
from nemar._staging import discard, staged, staging_path
from nemar._verification import VerifyPolicy
from nemar.errors import S3Error

NEMAR_S3_BUCKET = "nemar"
NEMAR_S3_REGION = "us-east-2"
NEMAR_S3_HOST = "https://nemar.s3.us-east-2.amazonaws.com"


def _get_whole(client, object_key: str, staging: Path, *, progress) -> None:
    """Stream one object into ``staging`` with a single unsigned GetObject.

    Streamed rather than buffered so a multi-GB object never sits in memory,
    and the bar advances per chunk so a large fetch does not look stalled.
    """
    resp = client.get_object(Bucket=NEMAR_S3_BUCKET, Key=object_key)
    with open(staging, "wb") as fh:
        for chunk in resp["Body"].iter_chunks(chunk_size=CHUNK_BYTES):
            fh.write(chunk)
            progress.update(len(chunk))


def _get_ranged(
    client,
    object_key: str,
    staging: Path,
    *,
    size: int,
    workers: int,
    progress,
) -> None:
    """Fetch one object as parallel byte ranges into ``staging``.

    The object length comes from the **manifest**, never from ``HeadObject``:
    this bucket is public-read but private-list, so an anonymous HEAD is denied
    (403). That is also why ``boto3``'s managed transfer (``download_fileobj``
    + ``TransferConfig``) cannot be used here — it probes with a HEAD first.
    Driving the ranges ourselves keeps the single-unsigned-GET contract while
    still saturating more than one connection on a large object.

    Each worker opens its own handle and seeks to its own offset, so the writes
    never share a file position.
    """
    with open(staging, "wb") as fh:
        fh.truncate(size)

    starts = range(0, size, MULTIPART_CHUNK_BYTES)

    def part(start: int) -> None:
        end = min(start + MULTIPART_CHUNK_BYTES, size) - 1
        resp = client.get_object(
            Bucket=NEMAR_S3_BUCKET,
            Key=object_key,
            Range=f"bytes={start}-{end}",
        )
        with open(staging, "r+b") as fh:
            fh.seek(start)
            for chunk in resp["Body"].iter_chunks(chunk_size=CHUNK_BYTES):
                fh.write(chunk)
                progress.update(len(chunk))

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        # list() forces every future, so a failing range raises here rather
        # than being dropped and leaving a silently short file.
        list(executor.map(part, starts))


def _normalize_version(version: str) -> str:
    """Normalize a version tag to the ``v<X.Y.Z>`` form S3 keys use.

    Accepts ``"v1.0.2"`` (passes through), ``"1.0.2"`` (gains a ``v``
    prefix), or surrounding whitespace (stripped). Empty / whitespace-
    only input is a caller error.
    """
    version = version.strip()
    if not version:
        raise ValueError("version must not be empty.")
    if version[0].isdigit():
        return f"v{version}"
    return version


def s3_object_url(dataset: str, annex_key: str) -> str:
    r"""Return the canonical, unsigned S3 URL for one git-annex content key.

    Parameters
    ----------
    dataset
        The bare NEMAR dataset id (e.g. ``"nm000132"`` or ``"on005505"``).
        Not validated here — pass the same id you would hand to
        :func:`nemar.download`.
    annex_key
        A git-annex content key (e.g.
        ``"SHA256E-s12345--abc...xyz.bin"``). Typically obtained from a
        DataLad clone of the dataset (``git annex find --format='${key}\n'``)
        or from the manifest URL's path component.

    Returns
    -------
    str
        ``https://<NEMAR_S3_HOST>/<dataset>/objects/<annex_key>``. The
        bucket is publicly readable, so the URL works without
        credentials.

    """
    return f"{NEMAR_S3_HOST}/{dataset}/objects/{annex_key}"


def version_url(dataset: str, version: str) -> str:
    """Return the canonical S3 URL for one dataset version's manifest.

    The manifest is a compact JSON document — keyed by file path, with
    each entry carrying the git-annex content ``key``, ``size``, and
    ``checksum``. This is the same data ``data.nemar.org`` transforms
    into its per-request pre-signed manifest, served from its
    canonical S3 location without the transform and without the
    1-hour pre-signed URL window.

    Parameters
    ----------
    dataset
        The bare NEMAR dataset id (``"nm000132"``, ``"on005505"``).
    version
        The version tag. Accepts both ``"v1.0.2"`` and ``"1.0.2"``;
        the leading ``v`` is added when missing.

    Returns
    -------
    str
        ``https://<NEMAR_S3_HOST>/<dataset>/version/<v>.json``. The
        URL is unsigned and works for any caller.

    """
    return f"{NEMAR_S3_HOST}/{dataset}/version/{_normalize_version(version)}.json"


def version_summary_url(dataset: str, version: str) -> str:
    """Return the canonical S3 URL for one dataset version's summary.

    The summary is a lightweight JSON document with totals, modalities,
    subjects, and a paths array — useful for catalog browsing without
    parsing the full manifest. Same public-read contract as
    :func:`version_url`.
    """
    return (
        f"{NEMAR_S3_HOST}/{dataset}/version/{_normalize_version(version)}-summary.json"
    )


def archive_url(dataset: str, version: str) -> str:
    """Return the canonical S3 URL for one dataset version's whole-dataset ZIP.

    Sizes range from tens of megabytes to hundreds of gigabytes
    depending on the dataset. Useful for callers that want to mirror
    a dataset in one transfer instead of iterating the manifest. The
    ZIP is content-equivalent to a DataLad clone at the same tag.

    .. note::

       **Best-effort availability.** Unlike :func:`version_url` and
       :func:`version_summary_url` (which we have seen 100 % of the
       time across the catalog), the archive is a build artifact
       published asynchronously after a version is cut. A random sample
       of 40 datasets had archives present for ~92 % of versions;
       freshly-published versions may not have one for hours or days.

       Callers should ``HEAD`` the URL and fall back to iterating the
       manifest (via :func:`nemar.download` or
       :func:`nemar.download_files`) when the archive is not
       yet available.
    """
    return f"{NEMAR_S3_HOST}/{dataset}/archives/{_normalize_version(version)}.zip"


def annex_key_for(file: DatasetFile) -> str | None:
    """Derive the git-annex content-addressed key for one :class:`DatasetFile`.

    Returns the SHA256E or MD5E form the bucket layout uses
    (``SHA256E-s<size>--<hex><suffix>``), or ``None`` when the file's
    checksum is git-tracked (``git_sha1``) or absent. ``None`` is the
    signal to short-circuit the S3 layer for this file — the caller
    raises :class:`~nemar.errors.S3Error` so the layered wrapper sends
    the whole batch to the next layer (DataLad / HTTPS).

    The suffix carries the same role as in git-annex itself: it lets a
    consumer guess the content type from the key. ``path`` is the
    in-manifest BIDS path; the suffix is the last ``.<ext>`` of the
    final path component, empty when the file has no extension.
    """
    if file.size is None:
        return None
    suffix = PurePosixPath(file.path).suffix
    if file.sha256:
        return f"SHA256E-s{file.size}--{file.sha256}{suffix}"
    if file.md5:
        return f"MD5E-s{file.size}--{file.md5}{suffix}"
    return None


class S3Backend:
    """Adapter that fetches files directly from the public NEMAR S3 bucket.

    Anonymous public-read against ``nemar.s3.us-east-2.amazonaws.com``
    via a synchronous ``boto3`` client (unsigned). The pattern matches
    ``eegdash``'s downloader (which has been running this in production
    for years against the same bucket) — one shared client per batch,
    anonymous auth, content-addressed at ``<dataset>/objects/<annex-key>``.

    ``boto3`` (not ``s3fs``) on purpose: ``s3fs`` drags in
    ``aiobotocore``, which hard-pins ``botocore`` and makes any
    environment that also needs ``boto3`` unsolvable. See
    https://github.com/eegdash/EEGDash/issues/397.

    Failure-mode contract: the **first** S3 miss aborts the whole
    batch's S3 transfer with :class:`~nemar.errors.S3Error`. The
    layered wrapper catches it and sends the whole batch to the next
    layer (DataLad → HTTPS). Backends stay batch-atomic; per-file
    fallback granularity is intentionally out of the contract.

    Hash verification is **not** this backend's job — the orchestrator
    runs :func:`~nemar._verification.assert_all_present` after
    ``transfer`` returns. Every other backend honours the same split.
    """

    def __init__(self, dataset: str, *, max_concurrency: int = 16) -> None:
        """Bind the backend to one NEMAR dataset id and a concurrency cap."""
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
        self.dataset = dataset
        self.max_concurrency = max_concurrency

    def transfer(
        self,
        files: Sequence[DatasetFile],
        *,
        target_dir: Path,
        options: TransferOptions,
        verify: VerifyPolicy,
        retry: RetryPolicy,
    ) -> None:
        """Fetch every entry in ``files`` into ``target_dir`` from S3.

        The ``options`` / ``verify`` / ``retry`` policies are accepted
        to honour the :class:`~nemar._backend.TransferBackend`
        protocol; this backend uses ``options.max_concurrent_downloads``
        to size the shared ``boto3`` client's connection pool and
        otherwise leaves verification / retry to the orchestrator and
        the layered wrapper. The HTTPS fallback exercises both policies
        fully.
        """
        del verify, retry  # See docstring — split owned by orchestrator.
        max_concurrency = min(
            self.max_concurrency, max(1, options.max_concurrent_downloads)
        )

        # Resolve every annex key BEFORE fetching anything. The batch-atomic
        # contract says one non-annexed entry sends the whole batch to the next
        # layer; doing that check up front means we hand over a clean slate
        # instead of a directory half-populated by objects fetched before the
        # miss, which the fallback would then have to redo anyway.
        keyed: list[tuple[DatasetFile, str]] = []
        for file in files:
            key = annex_key_for(file)
            if key is None:
                raise S3Error(
                    f"{file.path!r} is not annexed "
                    "(no sha256/md5 checksum on the manifest entry); "
                    "the S3 backend has nothing to fetch."
                )
            keyed.append((file, key))

        # Split the budget between files and parts of a single file. Many small
        # files want breadth; a handful of multi-GB objects want depth, and this
        # bucket serves single objects of 10 GB+. Fixing either dimension alone
        # leaves one of those two shapes running effectively single-threaded.
        file_workers = max(1, min(max_concurrency, len(keyed)))
        part_workers = max(1, max_concurrency // file_workers)

        try:
            client = boto3.client(
                "s3",
                region_name=NEMAR_S3_REGION,
                config=Config(
                    signature_version=UNSIGNED,
                    # Every in-flight part needs its own pooled connection, or
                    # the added parallelism just queues on the pool.
                    max_pool_connections=file_workers * part_workers + 4,
                    retries={"max_attempts": 5, "mode": "standard"},
                ),
            )
        except Exception as exc:  # pragma: no cover — defensive
            raise S3Error(f"failed to construct anonymous S3 client: {exc}") from exc

        total = sum(file.size or 0 for file in files)
        with tqdm(
            total=total,
            desc="S3",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress:

            def fetch(entry: tuple[DatasetFile, str]) -> None:
                file, key = entry
                object_key = f"{self.dataset}/objects/{key}"
                s3_path = f"{NEMAR_S3_BUCKET}/{object_key}"
                local_path = target_dir / file.path
                try:
                    # Writes land in a .part file and are renamed into place
                    # only once the body is complete, so an interrupted fetch
                    # cannot leave a truncated file under the real name.
                    with staged(local_path) as staging:
                        size = file.size
                        if (
                            part_workers > 1
                            and size is not None
                            and size >= MULTIPART_THRESHOLD_BYTES
                        ):
                            _get_ranged(
                                client,
                                object_key,
                                staging,
                                size=size,
                                workers=part_workers,
                                progress=progress,
                            )
                        else:
                            _get_whole(client, object_key, staging, progress=progress)
                except Exception as exc:
                    discard(staging_path(local_path))
                    raise S3Error(
                        f"S3 fetch failed for {file.path!r} (s3://{s3_path}): {exc}"
                    ) from exc

            if file_workers == 1:
                for entry in keyed:
                    fetch(entry)
                return

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=file_workers
            ) as executor:
                futures = [executor.submit(fetch, entry) for entry in keyed]
                errors: list[BaseException] = []
                for future in concurrent.futures.as_completed(futures):
                    exc = future.exception()
                    if exc is not None:
                        errors.append(exc)
            if errors:
                head = "; ".join(str(exc) for exc in errors[:3])
                more = "" if len(errors) <= 3 else f" (+{len(errors) - 3} more)"
                raise S3Error(
                    f"{len(errors)} file(s) failed during S3 transfer: {head}{more}"
                )
