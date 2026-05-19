"""End-to-end: full download + verification."""

from __future__ import annotations

import json

import pytest

import nemar
from tests.fixtures.factories import (
    make_blob,
    make_dataset_description,
    make_index,
    make_manifest_entry,
    make_manifest_list,
)

pytestmark = [pytest.mark.e2e, pytest.mark.integration]


def test_full_download_writes_all_files_with_correct_hashes(nemar_endpoint, target_dir):
    dd_bytes = json.dumps(
        make_dataset_description(dataset="nm000132", version="v1.0.0")
    ).encode("utf-8")
    blobs = [make_blob(seed=i, size_bytes=512) for i in range(3)]

    index = make_index(dataset="nm000132")
    manifest = make_manifest_list(
        [
            make_manifest_entry(path="dataset_description.json", content=dd_bytes),
            *[
                make_manifest_entry(path=f"sub-001/eeg/run-{i}.bin", content=b.content)
                for i, b in enumerate(blobs)
            ],
        ]
    )
    nemar_endpoint.publish(
        "nm000132",
        index=index,
        manifest=manifest,
        files={
            "v1.0.0/dataset_description.json": dd_bytes,
            **{
                f"v1.0.0/sub-001/eeg/run-{i}.bin": b.content
                for i, b in enumerate(blobs)
            },
        },
        metadata={"Name": "nm000132"},
    )

    nemar.download(
        dataset="nm000132",
        target_dir=target_dir,
        data_url=nemar_endpoint.base_url,
        downloader="python",
        max_concurrent_downloads=4,
    )

    assert (target_dir / "dataset_description.json").read_bytes() == dd_bytes
    for i, b in enumerate(blobs):
        out = target_dir / "sub-001" / "eeg" / f"run-{i}.bin"
        assert out.read_bytes() == b.content
