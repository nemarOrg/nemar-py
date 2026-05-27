"""Tests for the NEMAR S3 contract surface.

The contract was verified against the official DataLad clones of four
representative datasets (nm000132, nm000104, on005505, nm000133): every
git-annex ``nemar-s3`` special remote in those repos points at the
same bucket, datacenter, and public host, with a uniform per-dataset
``fileprefix=<dataset>/objects/``. The bucket is publicly readable
(``curl -I`` returns 200 against unsigned object URLs), so the helper
returns directly-usable URLs.

If NEMAR ever changes the bucket name, region, or per-dataset path
layout, these tests fail loudly and tell us exactly where.
"""

from __future__ import annotations

from nemar.s3 import (
    NEMAR_S3_BUCKET,
    NEMAR_S3_HOST,
    NEMAR_S3_REGION,
    s3_object_url,
)


def test_canonical_constants() -> None:
    """The three constants pin what we observed in every DataLad clone."""
    assert NEMAR_S3_BUCKET == "nemar"
    assert NEMAR_S3_REGION == "us-east-2"
    assert NEMAR_S3_HOST == "https://nemar.s3.us-east-2.amazonaws.com"


def test_s3_object_url_for_sha256e_key() -> None:
    """SHA256E annex keys land at ``<host>/<dataset>/objects/<key>``.

    The key shape ``SHA256E-s<size>--<hex>.<ext>`` is git-annex's
    extension-preserving variant of SHA-256. We verified this against
    a real object in nm000133's manifest: the unsigned URL returns
    200 OK, confirming the bucket is publicly readable along this
    path.
    """
    key = (
        "SHA256E-s261803--"
        "4cf4901ecf736c525ce534c4466030ec8d13c2fc0c8b31ddf86f17c55faadf89.mat"
    )
    assert s3_object_url("nm000133", key) == (
        "https://nemar.s3.us-east-2.amazonaws.com/"
        "nm000133/objects/" + key
    )


def test_s3_object_url_for_md5e_key() -> None:
    """MD5E annex keys (datasets like nm000104) follow the same shape.

    The key shape ``MD5E-s<size>--<hex>.<ext>`` differs only in the
    algorithm prefix; the URL layout is identical.
    """
    key = "MD5E-s197576448--c70cae6e4a043e2124d7e5ee94422d02.bdf"
    assert s3_object_url("nm000104", key) == (
        "https://nemar.s3.us-east-2.amazonaws.com/"
        "nm000104/objects/" + key
    )


def test_s3_object_url_handles_on_prefix() -> None:
    """``on*`` datasets share the same bucket as ``nm*`` datasets.

    Pinned because nothing in the SHAREd-bucket design forces this —
    a future reorg could split prefixes by source-of-truth. The current
    contract: one bucket, ``<dataset>/objects/`` partitions it.
    """
    key = (
        "SHA256E-s45212208--"
        "c6ff3e0ad41bd7b17e7e60329a278af681ce678b878365d0f7bd5ec0825a9fbe.set"
    )
    url = s3_object_url("on005505", key)
    assert url.startswith("https://nemar.s3.us-east-2.amazonaws.com/on005505/objects/")
    assert url.endswith(key)
