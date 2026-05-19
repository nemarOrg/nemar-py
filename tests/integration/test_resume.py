"""Integration: resume via HTTP Range / 206 Partial Content."""

from __future__ import annotations

import pytest

import nemar
from tests.fixtures.factories import (
    make_blob,
    make_index,
    make_manifest_entry,
    make_manifest_list,
)

pytestmark = pytest.mark.integration


def test_resume_from_partial_file_uses_range_header(nemar_endpoint, target_dir):
    blob = make_blob(seed=11, size_bytes=4096)
    rel = "data/big.bin"
    index = make_index(dataset="nm000132")
    manifest = make_manifest_list(
        [
            make_manifest_entry(path=rel, content=blob.content),
        ]
    )
    nemar_endpoint.publish(
        "nm000132",
        index=index,
        manifest=manifest,
        files={f"v1.0.0/{rel}": blob.content},
        metadata={"Name": "nm000132"},
    )
    nemar_endpoint.serve_with_range(f"/nm000132/v1.0.0/{rel}")

    # Pre-seed a partial download to force the resume branch.
    partial = target_dir / "data" / "big.bin"
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(blob.content[:1024])

    nemar.download(
        dataset="nm000132",
        target_dir=target_dir,
        data_url=nemar_endpoint.base_url,
        downloader="python",
        max_concurrent_downloads=1,
    )

    assert partial.read_bytes() == blob.content
