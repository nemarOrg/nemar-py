"""Integration: filesystem edge cases."""

from __future__ import annotations

import os

import pytest

import nemar
from tests.fixtures.factories import (
    make_blob,
    make_dataset_description,
    make_index,
    make_manifest_entry,
    make_manifest_list,
    write_dataset_description,
)

pytestmark = pytest.mark.integration


def _publish(nemar_endpoint, blob, *, path: str = "f.bin"):
    index = make_index(dataset="nm000132")
    manifest = make_manifest_list(
        [make_manifest_entry(path=path, content=blob.content)]
    )
    nemar_endpoint.publish(
        "nm000132",
        index=index,
        manifest=manifest,
        files={f"v1.0.0/{path}": blob.content},
        metadata={"Name": "nm000132"},
    )


def test_existing_wrong_version_blocks_download(nemar_endpoint, target_dir):
    blob = make_blob(seed=1, size_bytes=64)
    _publish(nemar_endpoint, blob)
    write_dataset_description(
        target_dir,
        make_dataset_description(dataset="nm000132", version="v9.9.9"),
    )

    with pytest.raises(FileExistsError, match="v9.9.9"):
        nemar.download(
            dataset="nm000132",
            target_dir=target_dir,
            data_url=nemar_endpoint.base_url,
            downloader="python",
            max_concurrent_downloads=1,
            max_retries=0,
        )


def test_partial_file_with_wrong_hash_is_retried(nemar_endpoint, target_dir):
    blob = make_blob(seed=2, size_bytes=256)
    _publish(nemar_endpoint, blob)
    bad = target_dir / "f.bin"
    bad.write_bytes(b"\x00" * 256)  # right size, wrong content

    nemar.download(
        dataset="nm000132",
        target_dir=target_dir,
        data_url=nemar_endpoint.base_url,
        downloader="python",
        max_concurrent_downloads=1,
    )

    assert bad.read_bytes() == blob.content


@pytest.mark.skipif(os.name == "nt", reason="POSIX-only chmod semantics")
def test_readonly_target_dir_raises(nemar_endpoint, target_dir):
    blob = make_blob(seed=3, size_bytes=32)
    _publish(nemar_endpoint, blob)
    target_dir.chmod(0o555)  # read+exec only
    try:
        with pytest.raises((PermissionError, OSError, RuntimeError)):
            nemar.download(
                dataset="nm000132",
                target_dir=target_dir,
                data_url=nemar_endpoint.base_url,
                downloader="python",
                max_concurrent_downloads=1,
                max_retries=0,
            )
    finally:
        target_dir.chmod(0o755)
