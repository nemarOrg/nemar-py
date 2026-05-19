"""Integration: real HTTPS for the Python streaming downloader."""

from __future__ import annotations

import hashlib

import httpx
import pytest

import nemar
from nemar import _download
from nemar._models import DatasetFile
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


def test_progress_does_not_overshoot_when_server_ignores_range(
    tmp_path,
) -> None:
    """Progress bar total equals file size when server returns 200 for a Range request.

    The server ignores the Range header and returns HTTP 200 with the full body.
    Before the fix the bar would overshoot by ``local_size`` (the pre-seeded
    partial bytes), because ``progress.update(local_size)`` ran before the
    request and was never subtracted on the 206→200 downgrade.
    """
    from tqdm.auto import tqdm as _tqdm

    data = b"hello nemar progress bar"
    partial_size = 8  # pre-seeded bytes

    target = tmp_path / "ds"
    target.mkdir()
    out = target / "data" / "sample.bin"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Pre-seed a partial file to trigger the resume / Range branch.
    out.write_bytes(data[:partial_size])

    file = DatasetFile(
        path="data/sample.bin",
        url="https://data.nemar.org/nm000132/v1.0.0/data/sample.bin",
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )

    # Always respond 200 regardless of Range header (simulates non-compliant server).
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=data, request=request)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)

    original_stream = httpx.stream

    def patched_stream(method: str, url: str, **kwargs):
        return client.stream(method, url, **kwargs)

    # Track updates via a mock progress object.
    total_updates: list[int] = []

    class TrackingProgress(_tqdm):
        def update(self, n=1):
            total_updates.append(n)
            return super().update(n)

    httpx.stream = patched_stream
    try:
        # Call _transfer_one_attempt directly so we control the progress object.
        with TrackingProgress(
            total=len(data), desc="test", unit="B", unit_scale=True
        ) as progress:
            _download._transfer_one_attempt(
                file,
                outfile=out,
                progress=progress,
                stream_timeout=60.0,
            )
    finally:
        httpx.stream = original_stream
        client.close()

    assert out.read_bytes() == data

    # The fix must emit a negative update of exactly -partial_size when the
    # server downgrades from 206 to 200, reversing the pre-credited bytes.
    # Without the fix the update list would never contain a negative value and
    # the bar would overshoot by ``partial_size``.
    assert any(u == -partial_size for u in total_updates), (
        f"Expected a -{partial_size} correction update; got: {total_updates}"
    )

    # The initial positive update (crediting existing partial bytes) must also
    # appear before the correction so we can confirm the full round-trip.
    assert total_updates[0] == partial_size, (
        f"First update should credit partial bytes ({partial_size}); got: {total_updates}"
    )
