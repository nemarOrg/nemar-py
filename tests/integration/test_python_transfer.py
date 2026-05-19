"""Integration: real HTTPS for the Python streaming downloader."""

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


def _publish_one_file(nemar_endpoint, blob, *, path: str = "data/sample.bin"):
    index = make_index(dataset="nm000132")
    manifest = make_manifest_list(
        [
            make_manifest_entry(
                path=path,
                content=blob.content,
            )
        ]
    )
    nemar_endpoint.publish(
        "nm000132",
        index=index,
        manifest=manifest,
        files={f"v1.0.0/{path}": blob.content},
        metadata={"Name": "nm000132"},
    )


def test_python_downloader_writes_file_and_verifies_sha256(nemar_endpoint, target_dir):
    blob = make_blob(seed=42, size_bytes=1024)
    _publish_one_file(nemar_endpoint, blob)

    nemar.download(
        dataset="nm000132",
        target_dir=target_dir,
        data_url=nemar_endpoint.base_url,
        downloader="python",
        max_concurrent_downloads=1,
    )

    out = target_dir / "data" / "sample.bin"
    assert out.exists()
    assert out.read_bytes() == blob.content


def test_stream_timeout_is_configurable(nemar_endpoint, target_dir):
    """The per-stream HTTP timeout is exposed as a kwarg.

    This currently fails: ``stream_timeout`` is hard-coded to 60.0 inside
    ``_transfer_one_attempt``. Driver test for bug fix #4.
    """
    blob = make_blob(seed=7, size_bytes=256)
    _publish_one_file(nemar_endpoint, blob)
    nemar_endpoint.slow_response("/nm000132/v1.0.0/data/sample.bin", delay_seconds=2.0)

    import httpx

    with pytest.raises((RuntimeError, httpx.ReadTimeout)):
        nemar.download(
            dataset="nm000132",
            target_dir=target_dir,
            data_url=nemar_endpoint.base_url,
            downloader="python",
            max_concurrent_downloads=1,
            max_retries=0,
            stream_timeout=0.1,
        )
