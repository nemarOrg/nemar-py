"""Concurrency and cross-module regression matrix.

These tests exercise the integration story by combining seams that
historically lived in different modules:

* NEMARClient (metadata) + download_one (per-file transfer) in concurrent
  threads — proves the python httpx.Client is thread-safe for our usage.
* Multiple bulk download() calls back-to-back against the same fixture —
  proves the orchestrator releases all resources cleanly.
"""

from __future__ import annotations

import concurrent.futures
import threading
from pathlib import Path

import httpx
import pytest

import nemar
from nemar import download_one
from nemar._retry import RetryPolicy
from nemar._verification import VerifyPolicy, VerifyResult
from tests.fixtures.factories import (
    make_blob,
    make_index,
    make_manifest_entry,
    make_manifest_list,
)

pytestmark = pytest.mark.integration


def _publish_files(nemar_endpoint, n: int = 5):
    blobs = [make_blob(seed=i, size_bytes=128 + i) for i in range(n)]
    index = make_index(dataset="nm000132")
    manifest = make_manifest_list(
        [
            make_manifest_entry(path=f"file_{i:03d}.bin", content=blobs[i].content)
            for i in range(n)
        ]
    )
    nemar_endpoint.publish(
        "nm000132",
        index=index,
        manifest=manifest,
        files={f"v1.0.0/file_{i:03d}.bin": blobs[i].content for i in range(n)},
        metadata={"Name": "nm000132"},
    )
    return blobs


def test_concurrent_download_one_calls_share_httpx_client(nemar_endpoint, target_dir):
    """download_one with a shared client should be safe for parallel use."""
    blobs = _publish_files(nemar_endpoint, n=8)

    with nemar.NEMARClient(data_url=nemar_endpoint.base_url) as client:
        index = client.fetch_index("nm000132")
        version = index.resolve_version("latest")
        manifest = client.fetch_manifest(index, version)

    # Use a separate, transfer-class httpx.Client (no follow_redirects).
    with httpx.Client(follow_redirects=False, timeout=30.0) as transfer_client:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = []
            for i in range(8):
                file = manifest.file(f"file_{i:03d}.bin")
                out_path = target_dir / f"out_{i:03d}.bin"
                futures.append(
                    pool.submit(
                        download_one,
                        file,
                        out_path,
                        client=transfer_client,
                        retry=RetryPolicy.default().with_attempts(0),
                    )
                )
            results = [f.result() for f in futures]

    assert all(r is VerifyResult.OK for r in results)
    for i in range(8):
        out = target_dir / f"out_{i:03d}.bin"
        assert out.read_bytes() == blobs[i].content


def test_concurrent_nemar_clients_do_not_interfere(nemar_endpoint, target_dir):
    """Two NEMARClient instances against the same endpoint do not interfere."""
    _publish_files(nemar_endpoint, n=3)
    results: list[str] = []

    def worker(label: str) -> None:
        with nemar.NEMARClient(data_url=nemar_endpoint.base_url) as client:
            index = client.fetch_index("nm000132")
            results.append(f"{label}:{index.dataset_id}")

    threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(results) == [
        "t0:nm000132",
        "t1:nm000132",
        "t2:nm000132",
        "t3:nm000132",
    ]


def test_two_full_downloads_back_to_back_succeed(nemar_endpoint, tmp_path: Path):
    """download() called twice in the same process should both succeed.

    Catches resource leaks: open file descriptors, hung threads, partial
    state in module-level mutables.
    """
    _publish_files(nemar_endpoint, n=4)

    target_a = tmp_path / "run_a"
    target_b = tmp_path / "run_b"

    nemar.download(
        dataset="nm000132",
        target_dir=target_a,
        data_url=nemar_endpoint.base_url,
        downloader="python",
        max_concurrent_downloads=2,
        max_retries=0,
    )
    nemar.download(
        dataset="nm000132",
        target_dir=target_b,
        data_url=nemar_endpoint.base_url,
        downloader="python",
        max_concurrent_downloads=2,
        max_retries=0,
    )

    for i in range(4):
        assert (target_a / f"file_{i:03d}.bin").exists()
        assert (target_b / f"file_{i:03d}.bin").exists()


def test_full_download_then_per_file_download_one_share_endpoint(
    nemar_endpoint, tmp_path: Path
):
    """Bulk download(...) and download_one() can target the same endpoint
    in the same process without interfering."""
    blobs = _publish_files(nemar_endpoint, n=3)

    bulk_target = tmp_path / "bulk"
    nemar.download(
        dataset="nm000132",
        target_dir=bulk_target,
        data_url=nemar_endpoint.base_url,
        downloader="python",
        max_concurrent_downloads=2,
        max_retries=0,
    )

    # Now reach the per-file primitive directly.
    with nemar.NEMARClient(data_url=nemar_endpoint.base_url) as client:
        index = client.fetch_index("nm000132")
        version = index.resolve_version("latest")
        manifest = client.fetch_manifest(index, version)
        file = manifest.file("file_001.bin")

        single_path = tmp_path / "single.bin"
        result = download_one(
            file,
            single_path,
            retry=RetryPolicy.default().with_attempts(0),
        )
        assert result is VerifyResult.OK
        assert single_path.read_bytes() == blobs[1].content


def test_nemar_client_with_verify_policy_round_trip(nemar_endpoint, target_dir):
    """Custom VerifyPolicy is honored by download_one (verify_size only,
    verify_hash disabled).
    """
    blobs = _publish_files(nemar_endpoint, n=1)

    with nemar.NEMARClient(data_url=nemar_endpoint.base_url) as client:
        index = client.fetch_index("nm000132")
        version = index.resolve_version("latest")
        manifest = client.fetch_manifest(index, version)
        file = manifest.file("file_000.bin")

        result = download_one(
            file,
            target_dir / "no_hash.bin",
            verify=VerifyPolicy(verify_size=True, verify_hash=False),
            retry=RetryPolicy.default().with_attempts(0),
        )
        assert result is VerifyResult.OK
        assert (target_dir / "no_hash.bin").read_bytes() == blobs[0].content
