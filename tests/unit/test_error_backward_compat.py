"""Backward-compatibility tests for the NemarError hierarchy.

The hierarchy is the new typed seam — but legacy callers that catch
``RuntimeError`` or ``FileExistsError`` must keep working without change.
These tests pin that contract module-by-module: each typed error is
catchable by its legacy class, with no behavior change to the caller.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nemar._download import LocalDataset
from nemar._endpoint import DataEndpoint
from nemar._models import VersionManifest, parse_dataset_index
from nemar._selection import SelectionPlan
from nemar._verification import VerifyPolicy, assert_all_present
from nemar.errors import (
    DatasetIndexError,
    EndpointError,
    LocalTargetError,
    LocalVersionMismatchError,
    ManifestError,
    NemarError,
    SelectionError,
    TransportError,
    VerificationError,
)


def test_endpoint_off_origin_is_a_runtime_error():
    """``except RuntimeError`` still catches off-origin URL refusal, with a
    useful message — not just a typed-but-blank exception."""
    endpoint = DataEndpoint.from_url("https://data.nemar.org/")
    with pytest.raises(RuntimeError, match="Refusing to download a file") as info:
        endpoint.assert_within("https://elsewhere.example.com/")
    assert isinstance(info.value, EndpointError)
    assert isinstance(info.value, NemarError)


def test_manifest_shape_error_is_a_runtime_error():
    """``except RuntimeError`` still catches malformed manifests."""
    endpoint = DataEndpoint.from_url("https://data.nemar.org/")
    with pytest.raises(
        RuntimeError, match="must be a JSON object or array"
    ) as info:
        VersionManifest.parse(
            12345,  # not a list or dict
            manifest_url="https://data.nemar.org/nm000132/v1/manifest.json",
            endpoint=endpoint,
        )
    assert isinstance(info.value, ManifestError)


def test_dataset_index_validation_failure_is_a_runtime_error():
    """parse_dataset_index raises a RuntimeError (subclass) on bad payload."""
    with pytest.raises(
        RuntimeError, match="unexpected dataset index payload"
    ) as info:
        parse_dataset_index({"not": "a valid dataset index"})
    assert isinstance(info.value, DatasetIndexError)


def test_dataset_index_resolve_version_runtime_error():
    """index.resolve_version raises a RuntimeError (subclass) for missing tag."""
    index = parse_dataset_index(
        {
            "dataset_id": "nm000132",
            "latest": "v1.0.0",
            "versions": [
                {
                    "version": "v1.0.0",
                    "manifest_url": "v1.0.0/manifest.json",
                }
            ],
        }
    )
    with pytest.raises(RuntimeError, match="does not exist") as info:
        index.resolve_version("v9.9.9")
    assert isinstance(info.value, DatasetIndexError)


def test_selection_zero_match_is_a_runtime_error():
    """SelectionPlan.build raises a RuntimeError (subclass) on zero match."""
    from nemar._bids import BidsQuery
    from nemar._models import DatasetFile

    files = [
        DatasetFile(
            path="sub-001/eeg/sub-001_task-MMN_eeg.set",
            url="https://data.nemar.org/a",
        )
    ]
    with pytest.raises(
        RuntimeError, match="No files matched the BIDS query"
    ) as info:
        SelectionPlan.build(
            files,
            query=BidsQuery.from_filters(subject="002"),
            include=[],
            exclude=[],
        )
    assert isinstance(info.value, SelectionError)


def test_local_version_mismatch_is_a_file_exists_error(tmp_path: Path):
    """LocalDataset.assert_compatible_with raises FileExistsError on version drift.

    Legacy callers MUST be able to catch FileExistsError, not just NemarError —
    and the message MUST name both versions so a user can act on it.
    """
    dataset_description = tmp_path / "dataset_description.json"
    dataset_description.write_text(
        '{"DatasetDOI": "doi:10.82901/nemar.nm000132", "Version": "v1.0.0"}',
        encoding="utf-8",
    )
    local = LocalDataset.from_dir(tmp_path)
    assert local is not None
    with pytest.raises(FileExistsError, match="v2.0.0.*v1.0.0") as info:
        local.assert_compatible_with(dataset="nm000132", tag="v2.0.0")
    # Same exception is ALSO a LocalVersionMismatchError and a NemarError.
    assert isinstance(info.value, LocalVersionMismatchError)
    assert isinstance(info.value, LocalTargetError)
    assert isinstance(info.value, NemarError)


def test_local_doi_mismatch_is_a_runtime_error(tmp_path: Path):
    """A wrong dataset under DatasetDOI raises a RuntimeError (subclass)."""
    dataset_description = tmp_path / "dataset_description.json"
    dataset_description.write_text(
        '{"DatasetDOI": "doi:10.82901/nemar.nm999999", "Version": "v1.0.0"}',
        encoding="utf-8",
    )
    local = LocalDataset.from_dir(tmp_path)
    assert local is not None
    with pytest.raises(
        RuntimeError, match="different NEMAR dataset"
    ) as info:
        local.assert_compatible_with(dataset="nm000132", tag="v1.0.0")
    assert isinstance(info.value, LocalTargetError)
    # DOI mismatch is NOT a FileExistsError — only version mismatch is.
    assert not isinstance(info.value, FileExistsError)


def test_verification_failure_is_a_runtime_error(tmp_path: Path):
    """assert_all_present raises a RuntimeError (subclass) on missing files."""
    from nemar._models import DatasetFile

    file = DatasetFile(
        path="missing.txt",
        url="https://data.nemar.org/missing.txt",
        size=10,
    )
    with pytest.raises(
        RuntimeError, match="Expected downloaded file is missing"
    ) as info:
        assert_all_present(
            [file], target_dir=tmp_path, policy=VerifyPolicy()
        )
    assert isinstance(info.value, VerificationError)


def test_transport_invalid_json_is_a_runtime_error():
    """fetch_json raises a RuntimeError (subclass) on non-JSON response."""
    import httpx

    from nemar._retry import RetryPolicy
    from nemar._transport import fetch_json

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json at all", request=request)

    transport = httpx.MockTransport(handler)
    policy = RetryPolicy.default().with_attempts(0)
    with httpx.Client(transport=transport) as client:
        with pytest.raises(RuntimeError, match="Invalid JSON") as info:
            fetch_json(
                client,
                url="https://data.nemar.org/nm000132/",
                what="retrieving NEMAR index for nm000132",
                policy=policy,
            )
    assert isinstance(info.value, TransportError)
