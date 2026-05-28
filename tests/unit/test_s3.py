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

import hashlib
import json
import socket

import pytest
import s3fs

from nemar._models import DatasetFile
from nemar._retry import RetryPolicy
from nemar._transfer import TransferOptions
from nemar._verification import VerifyPolicy
from nemar.errors import S3Error
from nemar.s3 import (
    NEMAR_S3_BUCKET,
    NEMAR_S3_HOST,
    NEMAR_S3_REGION,
    S3Backend,
    annex_key_for,
    archive_url,
    s3_object_url,
    version_summary_url,
    version_url,
)

# ``moto`` and ``boto3`` are dev-only deps used by the in-memory S3
# server. CI's minimal install does not pull them, so guard the import
# and skip the S3Backend-transfer tests when missing. ``annex_key_for``
# and the URL-helper tests above need neither — they keep running.
try:
    import boto3
    from moto.server import ThreadedMotoServer

    _moto_available = True
except ImportError:
    _moto_available = False
    boto3 = None  # type: ignore[assignment]
    ThreadedMotoServer = None  # type: ignore[assignment,misc]


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


def test_version_url_with_v_prefix() -> None:
    """``version/<v>.json`` is the canonical compact manifest.

    Confirmed against the live bucket: every dataset has this object at
    the documented path, and an unsigned ``HEAD`` returns 200. The S3
    manifest is in a different shape from ``data.nemar.org``'s
    transformed version (dict keyed by path vs. flat array with
    pre-signed URLs), but it is the same content.
    """
    assert version_url("nm000133", "v1.0.2") == (
        "https://nemar.s3.us-east-2.amazonaws.com/nm000133/version/v1.0.2.json"
    )


def test_version_url_normalizes_bare_version() -> None:
    """``"1.0.0"`` and ``"v1.0.0"`` both round-trip to the same URL.

    Matches the normalization rule used by
    ``DownloadRequest.from_kwargs(tag=...)``: leading digit gets a
    ``v`` prefix automatically. Saves the caller a string-format step
    when they already have a bare semver from elsewhere.
    """
    assert version_url("nm000133", "1.0.2") == version_url("nm000133", "v1.0.2")


def test_version_summary_url() -> None:
    """``version/<v>-summary.json`` is the lightweight catalog summary
    (modalities, subjects, totals, paths array). Same public-read
    contract as the full manifest.
    """
    assert version_summary_url("nm000132", "v1.1.1") == (
        "https://nemar.s3.us-east-2.amazonaws.com/"
        "nm000132/version/v1.1.1-summary.json"
    )


def test_archive_url() -> None:
    """``archives/<v>.zip`` is the whole-dataset bundle (tens of MB to
    hundreds of GB depending on the dataset). Useful for callers that
    want one transfer instead of iterating the manifest.
    """
    assert archive_url("nm000104", "v2.0.0") == (
        "https://nemar.s3.us-east-2.amazonaws.com/nm000104/archives/v2.0.0.zip"
    )


def test_empty_version_rejected() -> None:
    """Empty / whitespace-only version is a caller error, not a 404 hunt."""
    with pytest.raises(ValueError, match="version must not be empty"):
        version_url("nm000132", "")
    with pytest.raises(ValueError, match="version must not be empty"):
        archive_url("nm000132", "   ")


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


# ---------------------------------------------------------------------------
# annex_key_for — derives the bucket key from a DatasetFile
# ---------------------------------------------------------------------------


class TestAnnexKeyFor:
    """``annex_key_for`` builds the git-annex content key per backend rule.

    The bucket layout is content-addressed; this helper is the single
    seam that turns a manifest entry into a bucket key. Each branch
    here pins one of the four observed manifest shapes against a key
    that matches what ``git annex`` itself would have produced for the
    same content.
    """

    def _file(self, **kw) -> DatasetFile:
        defaults = {"path": "data/sample.set", "url": "https://example/x", "size": 4096}
        defaults.update(kw)
        return DatasetFile(**defaults)

    def test_sha256_with_extension_uses_sha256e(self) -> None:
        key = annex_key_for(self._file(sha256="abc" * 16))
        assert key == f"SHA256E-s4096--{'abc' * 16}.set"

    def test_md5_extensionless_path_drops_dot(self) -> None:
        key = annex_key_for(self._file(path="README", md5="d" * 32))
        assert key == f"MD5E-s4096--{'d' * 32}"

    def test_git_sha1_only_returns_none(self) -> None:
        """git_sha1-tracked files are git-objects, not annex objects."""
        key = annex_key_for(self._file(git_sha1="b" * 40))
        assert key is None

    def test_no_checksum_returns_none(self) -> None:
        """A manifest entry without any checksum cannot be addressed in S3."""
        assert annex_key_for(self._file()) is None

    def test_no_size_returns_none(self) -> None:
        """The annex key embeds size; without it we cannot synthesize a key."""
        f = self._file(sha256="a" * 64)
        f = DatasetFile(path=f.path, url=f.url, size=None, sha256=f.sha256)
        assert annex_key_for(f) is None


# ---------------------------------------------------------------------------
# S3Backend — end-to-end transfer against a moto-backed in-memory bucket
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def moto_s3(monkeypatch):
    """Stand up an in-memory S3 service via ``moto.server.ThreadedMotoServer``.

    ``s3fs`` talks HTTP to a real-shaped S3 endpoint; ``moto``'s
    in-process ``mock_aws`` decorator does not catch the underlying
    ``aiobotocore`` traffic. The threaded server does. We point both
    ``boto3`` (for object publishing in test setup) and the production
    ``s3fs`` (for the backend under test) at the same URL via the
    standard ``AWS_ENDPOINT_URL_S3`` env var that ``aiobotocore``
    respects natively.
    """
    # ``s3fs`` keeps a class-level filesystem cache; clear it so a
    # previously-cached anonymous fs (pointing at a closed moto port)
    # does not hijack this test's traffic.
    s3fs.S3FileSystem.clear_instance_cache()
    port = _free_port()
    server = ThreadedMotoServer(port=port)
    server.start()
    endpoint_url = f"http://127.0.0.1:{port}"
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", endpoint_url)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-2")
    try:
        client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name="us-east-2",
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
        )
        # Each test gets its own ``ThreadedMotoServer`` but moto's
        # backend state is process-level, so the bucket may already
        # exist from a previous test. Idempotent create avoids the
        # race.
        try:
            client.create_bucket(
                Bucket="nemar",
                CreateBucketConfiguration={"LocationConstraint": "us-east-2"},
            )
        except client.exceptions.BucketAlreadyOwnedByYou:
            pass
        # Mirror the real NEMAR bucket's public-read object policy so
        # the production ``S3Backend`` (which uses ``anon=True``) can
        # ``GET`` what the test publishes. Without this, moto enforces
        # the default private-by-default behaviour and 403s anonymous
        # GETs — which is correct AWS-side semantics but not what the
        # production bucket exposes.
        client.put_bucket_policy(
            Bucket="nemar",
            Policy=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": "*",
                            "Action": "s3:GetObject",
                            "Resource": "arn:aws:s3:::nemar/*",
                        }
                    ],
                }
            ),
        )
        yield client
    finally:
        server.stop()


def _publish_annex_object(
    client, dataset: str, annex_key: str, content: bytes
) -> None:
    client.put_object(
        Bucket="nemar",
        Key=f"{dataset}/objects/{annex_key}",
        Body=content,
    )


def _dataset_file(path: str, content: bytes) -> DatasetFile:
    return DatasetFile(
        path=path,
        url=f"https://example/{path}",
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _options(*, max_concurrent_downloads: int = 1) -> TransferOptions:
    return TransferOptions(
        backend="s3",
        max_concurrent_downloads=max_concurrent_downloads,
        stream_timeout=60.0,
    )


def _policies() -> tuple[RetryPolicy, VerifyPolicy]:
    return RetryPolicy.default().with_attempts(0), VerifyPolicy()


@pytest.mark.skipif(
    not _moto_available,
    reason="moto / boto3 not installed (dev group); install to run S3 chain tests.",
)
class TestS3BackendTransfer:
    """``S3Backend.transfer`` against a moto-backed bucket.

    Pins three contract halves:

    * Happy path — every annexed file lands on disk with full content.
    * No-checksum file — backend raises :class:`S3Error` before any
      S3 call so the layered wrapper falls through to the next layer.
    * Missing key — :class:`S3Error` is raised with the offending key
      in the message.
    """

    def test_happy_path_writes_file(self, moto_s3, tmp_path) -> None:
        content = b"chunked-tsv-content" * 16
        f = _dataset_file("eeg/sub-001/eeg.set", content)
        _publish_annex_object(moto_s3, "nm000132", annex_key_for(f), content)

        backend = S3Backend(dataset="nm000132")
        retry, verify = _policies()
        backend.transfer(
            [f],
            target_dir=tmp_path,
            options=_options(),
            verify=verify,
            retry=retry,
        )

        out = tmp_path / "eeg" / "sub-001" / "eeg.set"
        assert out.read_bytes() == content

    def test_no_checksum_raises_s3error_before_any_network(
        self, moto_s3, tmp_path
    ) -> None:
        f = DatasetFile(
            path="data/headers.json",
            url="https://example/data/headers.json",
            size=128,
        )

        backend = S3Backend(dataset="nm000132")
        retry, verify = _policies()
        with pytest.raises(S3Error, match="not annexed"):
            backend.transfer(
                [f],
                target_dir=tmp_path,
                options=_options(),
                verify=verify,
                retry=retry,
            )

    def test_missing_key_raises_s3error(self, moto_s3, tmp_path) -> None:
        f = _dataset_file("eeg/missing.set", b"never published")

        backend = S3Backend(dataset="nm000132")
        retry, verify = _policies()
        with pytest.raises(S3Error, match="S3 fetch failed"):
            backend.transfer(
                [f],
                target_dir=tmp_path,
                options=_options(),
                verify=verify,
                retry=retry,
            )
