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

from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import s3fs
from tqdm.auto import tqdm

from nemar._errors import S3Error
from nemar._models import DatasetFile

if TYPE_CHECKING:
    # Type-only imports — used purely for annotations on
    # :meth:`S3Backend.transfer`. Pulling :class:`TransferOptions` at
    # runtime would create a ``_transfer → _s3 → _transfer`` cycle.
    # :class:`~nemar._transfer.TransferBackend` is a structural Protocol,
    # so the runtime never inspects these annotations.
    from nemar._retry import RetryPolicy
    from nemar._transfer import TransferOptions
    from nemar._verification import VerifyPolicy

NEMAR_S3_BUCKET = "nemar"
NEMAR_S3_REGION = "us-east-2"
NEMAR_S3_HOST = "https://nemar.s3.us-east-2.amazonaws.com"


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
    return (
        f"{NEMAR_S3_HOST}/{dataset}/version/{_normalize_version(version)}.json"
    )


def version_summary_url(dataset: str, version: str) -> str:
    """Return the canonical S3 URL for one dataset version's summary.

    The summary is a lightweight JSON document with totals, modalities,
    subjects, and a paths array — useful for catalog browsing without
    parsing the full manifest. Same public-read contract as
    :func:`version_url`.
    """
    return (
        f"{NEMAR_S3_HOST}/{dataset}/version/"
        f"{_normalize_version(version)}-summary.json"
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
       :func:`nemar.transfer.download_files`) when the archive is not
       yet available.
    """
    return (
        f"{NEMAR_S3_HOST}/{dataset}/archives/"
        f"{_normalize_version(version)}.zip"
    )


def annex_key_for(file: DatasetFile) -> str | None:
    """Derive the git-annex content-addressed key for one :class:`DatasetFile`.

    Returns the SHA256E or MD5E form the bucket layout uses
    (``SHA256E-s<size>--<hex><suffix>``), or ``None`` when the file's
    checksum is git-tracked (``git_sha1``) or absent. ``None`` is the
    signal to short-circuit the S3 layer for this file — the caller
    raises :class:`~nemar._errors.S3Error` so the layered wrapper sends
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
    via ``s3fs``. The pattern matches ``eegdash``'s downloader (which
    has been running this in production for years against the same
    bucket) — one shared :class:`s3fs.S3FileSystem` per batch, anonymous
    auth, content-addressed at ``<dataset>/objects/<annex-key>``.

    Failure-mode contract: the **first** S3 miss aborts the whole
    batch's S3 transfer with :class:`~nemar._errors.S3Error`. The
    layered wrapper catches it and sends the whole batch to the next
    layer (DataLad → HTTPS). Backends stay batch-atomic; per-file
    fallback granularity is intentionally out of the contract.

    Hash verification is **not** this backend's job — the orchestrator
    runs :func:`~nemar._verification.assert_all_present` after
    ``transfer`` returns. Every other backend honours the same split.
    """

    def __init__(self, dataset: str, *, max_concurrency: int = 16) -> None:
        if max_concurrency < 1:
            raise ValueError(
                f"max_concurrency must be >= 1, got {max_concurrency}"
            )
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
        to honour the :class:`~nemar._transfer.TransferBackend`
        protocol; this backend uses ``options.max_concurrent_downloads``
        for the shared :class:`s3fs.S3FileSystem` and otherwise leaves
        verification / retry to the orchestrator and the layered
        wrapper. The HTTPS fallback exercises both policies fully.
        """
        del verify, retry  # See docstring — split owned by orchestrator.
        max_concurrency = min(
            self.max_concurrency, max(1, options.max_concurrent_downloads)
        )
        try:
            fs = s3fs.S3FileSystem(
                anon=True,
                max_concurrency=max_concurrency,
                client_kwargs={"region_name": NEMAR_S3_REGION},
            )
        except Exception as exc:  # pragma: no cover — defensive
            raise S3Error(
                f"failed to construct anonymous S3 filesystem: {exc}"
            ) from exc

        total = sum(file.size or 0 for file in files)
        with tqdm(
            total=total,
            desc="S3",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress:
            for file in files:
                key = annex_key_for(file)
                if key is None:
                    raise S3Error(
                        f"{file.path!r} is not annexed "
                        "(no sha256/md5 checksum on the manifest entry); "
                        "the S3 backend has nothing to fetch."
                    )
                s3_path = f"{NEMAR_S3_BUCKET}/{self.dataset}/objects/{key}"
                local_path = target_dir / file.path
                local_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    fs.get(s3_path, str(local_path))
                except Exception as exc:
                    raise S3Error(
                        f"S3 fetch failed for {file.path!r} "
                        f"(s3://{s3_path}): {exc}"
                    ) from exc
                progress.update(file.size or 0)
