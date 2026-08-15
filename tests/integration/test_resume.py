"""Integration: resume via HTTP Range / 206 Partial Content."""

from __future__ import annotations

import hashlib

import pytest
from tqdm.auto import tqdm

import nemar
from nemar import _streaming
from nemar._models import DatasetFile
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


@pytest.mark.parametrize(
    ("legacy_bytes", "description"),
    [
        pytest.param(b"old partial", "a short file from an older version", id="short"),
        pytest.param(b"", "an empty file", id="empty"),
    ],
)
def test_legacy_partial_at_the_final_path_is_overwritten(
    tmp_path, httpserver, legacy_bytes, description
):
    """A partial left at the final path by an older version must not corrupt.

    Before staging, an interrupted transfer left its bytes under the real name.
    Those files are still on disk after an upgrade. They are not resumed from
    (the Range request reads the .part), so the transfer must simply replace
    them rather than append to them and produce a corrupt result.
    """
    data = b"complete and correct payload"
    httpserver.expect_request("/data/sample.bin").respond_with_data(data)
    out = tmp_path / "data" / "sample.bin"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(legacy_bytes)

    file = DatasetFile(
        path="data/sample.bin",
        url=httpserver.url_for("/data/sample.bin"),
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    with tqdm(total=len(data), desc="test", unit="B") as progress:
        _streaming._transfer_one_attempt(
            file, outfile=out, progress=progress, stream_timeout=60.0
        )

    assert out.read_bytes() == data, description
    assert list(tmp_path.rglob("*.part")) == []
