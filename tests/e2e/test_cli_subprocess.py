"""End-to-end: invoke the nemar-py CLI as a subprocess."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from tests.fixtures.factories import (
    make_blob,
    make_index,
    make_manifest_entry,
    make_manifest_list,
)

pytestmark = [pytest.mark.e2e, pytest.mark.integration]


def test_cli_versions_command_against_local_endpoint(nemar_endpoint, target_dir):
    blob = make_blob(seed=1, size_bytes=32)
    index = make_index(dataset="nm000132")
    manifest = make_manifest_list(
        [
            make_manifest_entry(path="dataset_description.json", content=blob.content),
        ]
    )
    nemar_endpoint.publish(
        "nm000132",
        index=index,
        manifest=manifest,
        files={"v1.0.0/dataset_description.json": blob.content},
        metadata={"Name": "nm000132"},
    )

    env = os.environ.copy()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nemar",
            "versions",
            "--dataset",
            "nm000132",
            "--data-url",
            nemar_endpoint.base_url,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "v1.0.0" in result.stdout


def test_cli_download_writes_file(nemar_endpoint, target_dir):
    blob = make_blob(seed=2, size_bytes=128)
    index = make_index(dataset="nm000132")
    manifest = make_manifest_list(
        [
            make_manifest_entry(path="dataset_description.json", content=blob.content),
        ]
    )
    nemar_endpoint.publish(
        "nm000132",
        index=index,
        manifest=manifest,
        files={"v1.0.0/dataset_description.json": blob.content},
        metadata={"Name": "nm000132"},
    )

    env = os.environ.copy()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nemar",
            "download",
            "--dataset",
            "nm000132",
            "--target-dir",
            str(target_dir),
            "--data-url",
            nemar_endpoint.base_url,
            "--downloader",
            "python",
            "--max-concurrent-downloads",
            "1",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    assert (target_dir / "dataset_description.json").read_bytes() == blob.content
