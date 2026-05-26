"""Integration test for the EEGDash-facing integration trio.

The architectural deepening exposed three seams an external client (eegdash
in particular) would consume together:

1. :class:`nemar.NEMARClient` — long-lived metadata handle (one TLS session
   for many fetches).
2. :func:`nemar.VersionManifest.file` — random-access per-relpath lookup
   into a parsed manifest.
3. :func:`download_one` — single-file streaming primitive.

This test runs the three of them in sequence against the local HTTPS
fixture server. If this passes, the eegdash integration story holds.
"""

from __future__ import annotations

import hashlib

import httpx
import pytest

import nemar
from nemar._verification import VerifyResult
from nemar.transfer import download_one
from tests.fixtures.factories import (
    make_blob,
    make_index,
    make_manifest_entry,
    make_manifest_list,
)

pytestmark = pytest.mark.integration


def test_client_manifest_file_lookup_download_one_round_trip(
    nemar_endpoint, target_dir
):
    blob = make_blob(seed=42, size_bytes=2048)
    relpath = "sub-001/eeg/sub-001_task-MMN_eeg.set"
    index = make_index(dataset="nm000132")
    manifest = make_manifest_list(
        [make_manifest_entry(path=relpath, content=blob.content)]
    )
    nemar_endpoint.publish(
        "nm000132",
        index=index,
        manifest=manifest,
        files={f"v1.0.0/{relpath}": blob.content},
    )

    target_file = target_dir / "single.set"

    with nemar.NEMARClient(data_url=nemar_endpoint.base_url) as client:
        dataset_index = client.fetch_index("nm000132")
        assert dataset_index.dataset_id == "nm000132"

        version = dataset_index.resolve_version("latest")
        assert version.version == "v1.0.0"

        version_manifest = client.fetch_manifest(dataset_index, version)
        assert len(version_manifest) == 1
        assert relpath in version_manifest

        file = version_manifest.file(relpath)
        assert file.path == relpath
        assert file.sha256 == hashlib.sha256(blob.content).hexdigest()
        assert file.size == len(blob.content)

        result = download_one(file, target_file)
        assert result is VerifyResult.OK

    assert target_file.exists()
    assert target_file.read_bytes() == blob.content


def test_client_pool_reuse_reduces_handshake_cost(nemar_endpoint):
    """One NEMARClient context should reuse its httpx.Client across calls.

    The point of the seam: callers iterating across many datasets pay one
    TLS handshake, not N. We can't measure handshake count directly from
    pytest-httpserver, but we can confirm the same ``httpx.Client`` instance
    services every call by introspecting the client.
    """
    blob = make_blob(seed=7, size_bytes=128)
    for dataset in ("nm000132", "nm000133"):
        index = make_index(dataset=dataset)
        manifest = make_manifest_list(
            [make_manifest_entry(path="dataset_description.json", content=blob.content)]
        )
        nemar_endpoint.publish(
            dataset,
            index=index,
            manifest=manifest,
            files={"v1.0.0/dataset_description.json": blob.content},
        )

    with nemar.NEMARClient(data_url=nemar_endpoint.base_url) as client:
        internal_client = client._client  # type: ignore[attr-defined]
        assert isinstance(internal_client, httpx.Client)
        idx1 = client.fetch_index("nm000132")
        idx2 = client.fetch_index("nm000133")

    assert idx1.dataset_id == "nm000132"
    assert idx2.dataset_id == "nm000133"
    # After the context closed, the client should be closed.
    assert internal_client.is_closed


def test_download_one_with_shared_client_works(nemar_endpoint, target_dir):
    """Passing an httpx.Client to download_one reuses the connection pool."""
    blob = make_blob(seed=11, size_bytes=512)
    relpath = "data.bin"
    index = make_index(dataset="nm000132")
    manifest_payload = make_manifest_list(
        [make_manifest_entry(path=relpath, content=blob.content)]
    )
    nemar_endpoint.publish(
        "nm000132",
        index=index,
        manifest=manifest_payload,
        files={f"v1.0.0/{relpath}": blob.content},
    )

    with nemar.NEMARClient(data_url=nemar_endpoint.base_url) as client:
        index_obj = client.fetch_index("nm000132")
        version = index_obj.resolve_version("latest")
        manifest = client.fetch_manifest(index_obj, version)
        file = manifest.file(relpath)

        # Build a transfer-class httpx.Client that does NOT follow redirects
        # (matches PythonBackend's posture).
        with httpx.Client(follow_redirects=False, timeout=30.0) as transfer_client:
            target_file = target_dir / "out.bin"
            download_one(file, target_file, client=transfer_client)
            assert target_file.read_bytes() == blob.content
            # Caller-supplied client must NOT be closed by download_one.
            assert not transfer_client.is_closed
