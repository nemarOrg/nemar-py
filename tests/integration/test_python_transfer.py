"""Integration: real HTTPS for the Python streaming downloader."""

from __future__ import annotations

import hashlib

import httpx
import pytest
from tqdm.auto import tqdm

import nemar
from nemar import _transfer
from nemar._models import DatasetFile
from nemar._retry import RetryPolicy
from nemar._verification import VerifyPolicy
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

    class TrackingProgress(tqdm):
        def update(self, n=1):
            total_updates.append(n)
            return super().update(n)

    httpx.stream = patched_stream
    try:
        # Call _transfer_one_attempt directly so we control the progress object.
        with TrackingProgress(
            total=len(data), desc="test", unit="B", unit_scale=True
        ) as progress:
            _transfer._transfer_one_attempt(
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
        f"First update should credit partial bytes ({partial_size}); "
        f"got: {total_updates}"
    )


def test_shared_httpx_client_reused_across_files(
    monkeypatch, nemar_endpoint, target_dir
) -> None:
    """One httpx.Client serves the whole streaming batch (G).

    Counts ``httpx.Client.__init__`` invocations during a batch transfer.
    There are several clients across the full ``download()`` flow (one
    for metadata/index, one for the streaming download). The candidate
    G claim is that the *streaming* phase uses exactly one client even
    when many files are transferred -- so the increment that happens
    during the streaming phase should be exactly 1, regardless of file
    count.
    """
    blobs = [make_blob(seed=i, size_bytes=256) for i in range(5)]
    paths = [f"data/file_{i}.bin" for i in range(5)]
    index = make_index(dataset="nm000132")
    manifest = make_manifest_list(
        [
            make_manifest_entry(path=path, content=blob.content)
            for path, blob in zip(paths, blobs, strict=True)
        ]
    )
    nemar_endpoint.publish(
        "nm000132",
        index=index,
        manifest=manifest,
        files={
            f"v1.0.0/{path}": blob.content
            for path, blob in zip(paths, blobs, strict=True)
        },
        metadata={"Name": "nm000132"},
    )

    original_init = httpx.Client.__init__
    init_calls: list[dict] = []

    def patched_init(self, *args, **kwargs):
        init_calls.append({"limits": kwargs.get("limits")})
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", patched_init)

    nemar.download(
        dataset="nm000132",
        target_dir=target_dir,
        data_url=nemar_endpoint.base_url,
        downloader="python",
        max_concurrent_downloads=2,
    )

    # The streaming client is identifiable by its ``limits`` kwarg --
    # the metadata client does not set one.
    streaming_inits = [c for c in init_calls if c["limits"] is not None]
    assert len(streaming_inits) == 1, (
        f"Expected exactly one streaming-client init for 5 files; "
        f"observed {len(streaming_inits)} (total clients: {len(init_calls)})"
    )

    # Verify every file landed correctly to make sure the shared client
    # actually delivered the bytes.
    for path, blob in zip(paths, blobs, strict=True):
        out = target_dir / path
        assert out.exists(), f"missing {path}"
        assert out.read_bytes() == blob.content, f"corrupt {path}"


def test_http_416_on_resume_retries_fresh(tmp_path) -> None:
    """A 416 against a Range request triggers exactly one fresh retry (S7).

    The first call uses ``Range: bytes={partial}-``; the server replies
    416 (Range Not Satisfiable) because the object has changed or is
    shorter. The driver must unlink the stale partial and issue an
    unconditional GET on the next attempt, succeeding without infinite
    loops.
    """
    data = b"fresh download contents"
    partial_size = 8

    target = tmp_path / "ds"
    target.mkdir()
    out = target / "data" / "sample.bin"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"X" * partial_size)  # stale partial

    file = DatasetFile(
        path="data/sample.bin",
        url="https://data.nemar.org/nm000132/v1.0.0/data/sample.bin",
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )

    call_count = {"n": 0, "ranged": 0, "fresh": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if "Range" in request.headers or "range" in request.headers:
            call_count["ranged"] += 1
            return httpx.Response(416, content=b"", request=request)
        call_count["fresh"] += 1
        return httpx.Response(200, content=data, request=request)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)

    original_stream = httpx.stream

    def patched_stream(method: str, url: str, **kwargs):
        return client.stream(method, url, **kwargs)

    httpx.stream = patched_stream
    try:
        with tqdm(
            total=len(data), desc="test", unit="B", unit_scale=True
        ) as progress:
            _transfer._transfer_one_with_python(
                file,
                target,
                RetryPolicy.default().with_attempts(3),
                VerifyPolicy(verify_size=True, verify_hash=True),
                progress,
                60.0,
            )
    finally:
        httpx.stream = original_stream
        client.close()

    # Round trip: at least one Range attempt that 416'd, then exactly one
    # fresh attempt. We must not loop on 416 forever.
    assert call_count["ranged"] >= 1, f"Expected a Range request; got {call_count}"
    assert call_count["fresh"] == 1, (
        f"Expected exactly one fresh retry; got {call_count}"
    )
    assert out.read_bytes() == data
