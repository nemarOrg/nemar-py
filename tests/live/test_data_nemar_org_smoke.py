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
