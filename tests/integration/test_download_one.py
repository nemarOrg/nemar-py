"""Integration tests for ``download_one`` against the local HTTPS fixture.

The unit suite covers the wiring with ``httpx.MockTransport``. These
tests exercise the same primitive against the real ``pytest_httpserver``
HTTPS fixture so the Range/206 resume path and the certificate chain
are validated end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nemar import download_one
from nemar._models import DatasetFile
from nemar._verification import VerifyResult
from tests.fixtures.factories import (
    make_blob,
    make_index,
    make_manifest_entry,
    make_manifest_list,
)

pytestmark = pytest.mark.integration


def _publish_single_file(
    nemar_endpoint,
    blob,
    *,
    path: str = "data/sample.bin",
    dataset: str = "nm000132",
) -> DatasetFile:
    """Publish a one-file dataset and return the DatasetFile pointing at it."""
    index = make_index(dataset=dataset)
    manifest = make_manifest_list(
        [make_manifest_entry(path=path, content=blob.content)]
    )
    nemar_endpoint.publish(
        dataset,
        index=index,
        manifest=manifest,
        files={f"v1.0.0/{path}": blob.content},
        metadata={"Name": dataset},
    )
    return DatasetFile(
        path=path,
        url=f"{nemar_endpoint.base_url.rstrip('/')}/{dataset}/v1.0.0/{path}",
        size=blob.size,
        sha256=blob.sha256,
    )


def test_download_one_writes_correct_bytes_over_https(
    nemar_endpoint, tmp_path: Path
) -> None:
    """Smoke: download_one against the HTTPS fixture writes byte-perfect content."""
    blob = make_blob(seed=101, size_bytes=2048)
    file = _publish_single_file(nemar_endpoint, blob)

    target = tmp_path / "ds" / "data" / "sample.bin"
    result = download_one(file, target)

    assert result is VerifyResult.OK
    assert target.read_bytes() == blob.content


def test_download_one_resumes_from_partial_via_range(
    nemar_endpoint, tmp_path: Path
) -> None:
    """A pre-seeded partial target completes via Range/206.

    Pre-seeding bytes before calling ``download_one`` must trigger the
    ``Range: bytes={n}-`` resume path. The fixture's ``serve_with_range``
    handler returns HTTP 206 with a sliced body; the function must
    append the suffix and verify the full file.
    """
    blob = make_blob(seed=202, size_bytes=4096)
    rel = "data/big.bin"
    file = _publish_single_file(nemar_endpoint, blob, path=rel)
    nemar_endpoint.serve_with_range(f"/nm000132/v1.0.0/{rel}")

    target = tmp_path / "ds" / "data" / "big.bin"
    target.parent.mkdir(parents=True, exist_ok=True)
    # Pre-seed the first 1024 bytes so the resume branch fires.
    target.write_bytes(blob.content[:1024])

    result = download_one(file, target)

    assert result is VerifyResult.OK
    assert target.read_bytes() == blob.content
