"""End-to-end: real aria2c subprocess."""

from __future__ import annotations

import shutil

import pytest

import nemar
from nemar import _transfer
from tests.fixtures.factories import (
    make_blob,
    make_index,
    make_manifest_entry,
    make_manifest_list,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.integration,
    pytest.mark.aria2,
    pytest.mark.skipif(
        shutil.which("aria2c") is None,
        reason="aria2c is not on PATH",
    ),
]


@pytest.fixture
def aria2_skip_cert_check(monkeypatch):
    """Disable aria2c's certificate check against the local fixture's CA.

    aria2c on macOS uses AppleTLS, which silently ignores
    ``--ca-certificate``. The fixture server is signed by a private trustme
    CA, so we have to disable cert verification for the test to reach the
    HTTPS endpoint at all. Confined to tests via the ``_ARIA2_EXTRA_ARGS``
    private hook on ``nemar._transfer``.
    """
    monkeypatch.setattr(_transfer, "_ARIA2_EXTRA_ARGS", ["--check-certificate=false"])


def test_aria2c_downloads_one_file(nemar_endpoint, target_dir, aria2_skip_cert_check):
    blob = make_blob(seed=10, size_bytes=2048)
    index = make_index(dataset="nm000132")
    manifest = make_manifest_list(
        [
            make_manifest_entry(path="data/sample.bin", content=blob.content),
        ]
    )
    nemar_endpoint.publish(
        "nm000132",
        index=index,
        manifest=manifest,
        files={"v1.0.0/data/sample.bin": blob.content},
        metadata={"Name": "nm000132"},
    )

    nemar.download(
        dataset="nm000132",
        target_dir=target_dir,
        data_url=nemar_endpoint.base_url,
        downloader="aria2",
        max_concurrent_downloads=1,
        verify_hash=False,  # aria2 can't verify against our local CA's sha256
    )

    out = target_dir / "data" / "sample.bin"
    assert out.exists()
    assert out.read_bytes() == blob.content


def test_aria2_timeout_kills_hung_process(
    nemar_endpoint, target_dir, aria2_skip_cert_check
):
    """``aria2_timeout`` must kill a hung aria2c subprocess.

    Drives bug fix #2. Currently fails: ``download`` has no ``aria2_timeout``
    kwarg, and the existing ``subprocess.run`` call has no ``timeout``.
    """
    blob = make_blob(seed=20, size_bytes=128)
    index = make_index(dataset="nm000132")
    manifest = make_manifest_list(
        [
            make_manifest_entry(path="data/sample.bin", content=blob.content),
        ]
    )
    nemar_endpoint.publish(
        "nm000132",
        index=index,
        manifest=manifest,
        files={"v1.0.0/data/sample.bin": blob.content},
        metadata={"Name": "nm000132"},
    )
    # Make the server sleep so long that any reasonable aria2_timeout
    # fires first. 5s is enough wall-time margin against the 1.0s timeout
    # below while keeping test teardown fast (werkzeug blocks on the
    # in-flight handler when cleaning up handler registrations).
    nemar_endpoint.slow_response(
        "/nm000132/v1.0.0/data/sample.bin",
        delay_seconds=5.0,
    )

    with pytest.raises(RuntimeError, match="timed out"):
        nemar.download(
            dataset="nm000132",
            target_dir=target_dir,
            data_url=nemar_endpoint.base_url,
            downloader="aria2",
            max_concurrent_downloads=1,
            verify_hash=False,
            aria2_timeout=1.0,
        )
