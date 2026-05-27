"""The NEMAR S3 file-delivery contract.

The ``data.nemar.org`` metadata endpoint is a thin catalog + manifest
service; the actual file bytes live on S3. The contract below was
verified against the official DataLad clones of four representative
datasets (nm000132, nm000104, on005505, nm000133): every dataset's
git-annex ``nemar-s3`` special remote advertises the same bucket,
region, and public host, with a uniform per-dataset path prefix.

Layout
------

``https://<NEMAR_S3_HOST>/<dataset>/objects/<annex-key>``

* ``NEMAR_S3_HOST`` — single public host across the catalog.
* ``<dataset>`` — the bare dataset id (``nm000132``, ``on005505``).
  Both prefixes share one bucket; ``<dataset>/objects/`` partitions it.
* ``<annex-key>`` — the git-annex content-addressed key,
  e.g. ``SHA256E-s<size>--<hex>.<ext>`` or
  ``MD5E-s<size>--<hex>.<ext>``. The dataset's
  ``.gitattributes annex.backend`` picks the algorithm; the size and
  extension stay encoded in the key for content-type negotiation.

The bucket is publicly readable: an unsigned ``GET`` against the
canonical URL returns 200 OK. The pre-signed URLs the
``data.nemar.org`` manifest serves are a convenience layer (1-hour
``X-Amz-Expires``, ``response-content-disposition`` for nice filenames)
on top of the same canonical objects — they are not required for read
access.

When to use this module
-----------------------

Library callers who already hold a git-annex key from a DataLad clone
(e.g. eegdash composing its own ingestion pipeline) can build URLs
directly without round-tripping through the ``data.nemar.org``
manifest. Callers who do not have an annex key in hand should use
:func:`nemar.download` / :func:`nemar.transfer.download_files` instead
— those route through the manifest and get pre-signed URLs for free.
"""

from __future__ import annotations

NEMAR_S3_BUCKET = "nemar"
NEMAR_S3_REGION = "us-east-2"
NEMAR_S3_HOST = "https://nemar.s3.us-east-2.amazonaws.com"


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
