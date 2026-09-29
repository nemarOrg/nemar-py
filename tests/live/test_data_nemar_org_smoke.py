"""Live smoke test against data.nemar.org.

Skipped unless ``NEMAR_LIVE_TEST=1`` is set. Used to detect upstream schema
drift; not part of the default CI suite.
"""

from __future__ import annotations

import os

import httpx
import pytest

import nemar
from nemar._models import VersionManifest
from nemar._verification import VerifyResult
from nemar.s3 import (
    archive_url,
    s3_object_url,
    version_summary_url,
    version_url,
)

pytestmark = pytest.mark.live

SKIP_REASON = "Set NEMAR_LIVE_TEST=1 to run live smoke tests."

LIVE_DATASET = os.environ.get("NEMAR_LIVE_DATASET", "nm000132")


def _public_head(url: str, *, archive: bool = False) -> httpx.Response:
    response = httpx.head(url, follow_redirects=False, timeout=10.0)
    if response.status_code == 403:
        pytest.skip(f"Anonymous S3 access is unavailable: {url}")
    if archive and response.status_code == 404:
        pytest.skip(f"Archive has not been published: {url}")
    assert response.status_code == 200, f"{url}: HTTP {response.status_code}"
    return response


@pytest.mark.skipif(
    os.environ.get("NEMAR_LIVE_TEST") != "1",
    reason=SKIP_REASON,
)
def test_fetch_dataset_index_against_live_endpoint() -> None:
    idx = nemar.fetch_dataset_index(dataset=LIVE_DATASET)
    assert idx.dataset_id == LIVE_DATASET
    assert idx.latest
    assert idx.versions


@pytest.mark.skipif(
    os.environ.get("NEMAR_LIVE_TEST") != "1",
    reason=SKIP_REASON,
)
def test_download_one_small_file_from_live_endpoint(tmp_path) -> None:
    nemar.download(
        dataset=LIVE_DATASET,
        target_dir=tmp_path,
        include=["dataset_description.json"],
        downloader="python",
        max_concurrent_downloads=1,
    )
    out = tmp_path / "dataset_description.json"
    assert out.exists()
    assert out.stat().st_size > 0


@pytest.mark.skipif(
    os.environ.get("NEMAR_LIVE_TEST") != "1",
    reason=SKIP_REASON,
)
def test_on_prefixed_dataset_index_against_live_endpoint() -> None:
    """``on*`` is the second NEMAR dataset-id prefix (the OpenNeuro-derived
    catalog). Pins that input validation and manifest parsing both
    accept the real ``on005505`` shape so future regex regressions are
    caught immediately by the live smoke suite.
    """
    idx = nemar.fetch_dataset_index(dataset="on005505")
    assert idx.dataset_id == "on005505"
    assert idx.latest
    assert idx.versions


@pytest.mark.skipif(
    os.environ.get("NEMAR_LIVE_TEST") != "1",
    reason=SKIP_REASON,
)
def test_s3_canonical_url_serves_an_object() -> None:
    """An existing object permits unsigned HEAD when public read is granted."""
    # Find one small SHA256E-keyed file from the live manifest.
    with nemar.NEMARClient() as client:
        index = client.fetch_index(LIVE_DATASET)
        version = index.resolve_version("latest")
        manifest = client.fetch_manifest(index, version)
    sha_entries = [f for f in manifest if f.sha256 and f.size and f.size < 200_000]
    assert sha_entries, "expected at least one small sha256-checksummed file"
    file = sha_entries[0]
    # Reconstruct the annex key from the file's metadata. The path
    # component after ``/objects/`` is the canonical key.
    s3_path = file.url.split("/objects/", 1)[1].split("?", 1)[0]
    canonical = s3_object_url(LIVE_DATASET, s3_path)
    _public_head(canonical)


@pytest.mark.skipif(
    os.environ.get("NEMAR_LIVE_TEST") != "1",
    reason=SKIP_REASON,
)
def test_s3_version_url_unsigned_head_returns_200() -> None:
    """``version_url(...)`` resolves the canonical compact manifest.

    Public-read denial is legitimate for a bucket-policy-excluded dataset.
    """
    latest = nemar.fetch_dataset_index(dataset=LIVE_DATASET).latest
    url = version_url(LIVE_DATASET, latest)
    response = _public_head(url)
    assert response.headers.get("content-type", "").startswith("application/json")


@pytest.mark.skipif(
    os.environ.get("NEMAR_LIVE_TEST") != "1",
    reason=SKIP_REASON,
)
def test_s3_version_summary_url_unsigned_head_returns_200() -> None:
    """``version_summary_url(...)`` resolves the lightweight catalog summary."""
    latest = nemar.fetch_dataset_index(dataset=LIVE_DATASET).latest
    url = version_summary_url(LIVE_DATASET, latest)
    _public_head(url)


@pytest.mark.skipif(
    os.environ.get("NEMAR_LIVE_TEST") != "1",
    reason=SKIP_REASON,
)
def test_s3_archive_url_unsigned_head_returns_200() -> None:
    """``archive_url(...)`` resolves the whole-dataset ZIP.

    Check a non-trivial size when the archive exists and is readable.
    ``HEAD`` only — the body would be multi-GB.
    """
    latest = nemar.fetch_dataset_index(dataset=LIVE_DATASET).latest
    url = archive_url(LIVE_DATASET, latest)
    response = _public_head(url, archive=True)
    content_length = int(response.headers.get("content-length", "0"))
    assert content_length > 1_000_000, (
        f"archive looks like a stub: only {content_length} bytes"
    )


@pytest.mark.skipif(
    os.environ.get("NEMAR_LIVE_TEST") != "1",
    reason=SKIP_REASON,
)
def test_manifest_carries_both_git_and_sha256_entries() -> None:
    """The real ``nm000132`` manifest mixes git-tracked small files
    (``checksum_algorithm: "git"`` → ``DatasetFile.git_sha1``) and
    S3 binaries (``checksum_algorithm: "sha256"`` →
    ``DatasetFile.sha256``).

    Pins that the parser dispatches correctly against the live shape.
    A regression here would mean either checksum bucket is empty
    even though the upstream manifest still advertises both.
    """
    with nemar.NEMARClient() as client:
        index = client.fetch_index(LIVE_DATASET)
        version = index.resolve_version("latest")
        manifest = client.fetch_manifest(index, version)
    has_git = any(f.git_sha1 is not None for f in manifest)
    has_sha256 = any(f.sha256 is not None for f in manifest)
    assert has_git, "expected at least one git-tracked entry in manifest"
    assert has_sha256, "expected at least one sha256-tracked entry in manifest"


@pytest.mark.skipif(
    os.environ.get("NEMAR_LIVE_TEST") != "1", reason=SKIP_REASON,
)
def test_compact_manifest_downloads_git_and_annex_bytes(tmp_path) -> None:
    """Compact manifests resolve durable routes, including annex redirects."""
    with nemar.NEMARClient() as api, httpx.Client(timeout=30.0) as raw:
        index = api.fetch_index(LIVE_DATASET)
        url = version_url(LIVE_DATASET, index.latest)
        response = raw.get(url)
        if response.status_code == 403:
            pytest.skip("Compact S3 manifest is not publicly readable")
        response.raise_for_status()
        manifest = VersionManifest.parse(
            response.json(), manifest_url=url, endpoint=api.endpoint,
        )
        for annexed in (False, True):
            candidates = [f for f in manifest if f.size and f.size < 200_000
                          and bool(f.sha256 or f.md5) == annexed]
            assert candidates
            file = min(candidates, key=lambda f: f.size)
            assert file.url == file.bytes_url
            assert nemar.download_one(
                file, tmp_path / file.path, client=raw,
            ) is VerifyResult.OK
