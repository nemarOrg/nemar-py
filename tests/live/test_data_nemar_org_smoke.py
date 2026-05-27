"""Live smoke test against data.nemar.org.

Skipped unless ``NEMAR_LIVE_TEST=1`` is set. Used to detect upstream schema
drift; not part of the default CI suite.
"""

from __future__ import annotations

import os

import pytest

import nemar

pytestmark = pytest.mark.live

SKIP_REASON = "Set NEMAR_LIVE_TEST=1 to run live smoke tests."

LIVE_DATASET = os.environ.get("NEMAR_LIVE_DATASET", "nm000132")


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
