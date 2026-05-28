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
