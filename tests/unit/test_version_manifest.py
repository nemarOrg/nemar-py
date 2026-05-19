"""Tests for the ``VersionManifest`` value type."""

from __future__ import annotations

import pytest

from nemar._endpoint import DataEndpoint
from nemar._models import DatasetFile, VersionManifest

MANIFEST_URL = "https://data.nemar.org/nm000132/v1.0.0/manifest.json"
ENDPOINT = DataEndpoint.from_url("https://data.nemar.org/")


def _parse_payload(payload, *, endpoint=ENDPOINT):
    return VersionManifest.parse(
        payload, manifest_url=MANIFEST_URL, endpoint=endpoint
    )


class TestParse:
    """``VersionManifest.parse`` round-trips supported manifest shapes."""

    def test_round_trips_list_payload(self) -> None:
        manifest = _parse_payload(
            [
                {
                    "path": "dataset_description.json",
                    "url": "dataset_description.json",
                    "size": 12,
                    "sha256": '"abc"',
                }
            ]
        )

        assert isinstance(manifest, VersionManifest)
        assert manifest.manifest_url == MANIFEST_URL
        assert manifest.endpoint is ENDPOINT
        assert len(manifest.files) == 1

        file = manifest.files[0]
        assert isinstance(file, DatasetFile)
        assert file.path == "dataset_description.json"
        assert (
            file.url
            == "https://data.nemar.org/nm000132/v1.0.0/dataset_description.json"
        )
        assert file.size == 12
        assert file.sha256 == "abc"

    def test_round_trips_files_dict_payload(self) -> None:
        manifest = _parse_payload(
            {
                "files": [
                    {"path": "a.json", "size": 1},
                    {"path": "b.json", "size": 2},
                ]
            }
        )
        assert [f.path for f in manifest.files] == ["a.json", "b.json"]


class TestDunderProtocols:
    """``VersionManifest`` behaves like a sequence of files."""

    def test_len_reports_file_count(self) -> None:
        manifest = _parse_payload(
            [{"path": "a.json"}, {"path": "b.json"}, {"path": "c.json"}]
        )
        assert len(manifest) == 3

    def test_iter_yields_files_in_order(self) -> None:
        manifest = _parse_payload(
            [{"path": "a.json"}, {"path": "b.json"}, {"path": "c.json"}]
        )
        assert [f.path for f in manifest] == ["a.json", "b.json", "c.json"]


class TestOriginEnforcement:
    """``VersionManifest`` defers origin checking to the endpoint."""

    def test_mixed_origin_files_raise(self) -> None:
        with pytest.raises(RuntimeError, match="outside the configured NEMAR"):
            _parse_payload(
                [
                    {"path": "ok.json"},
                    {"path": "bad.json", "url": "https://example.org/bad.json"},
                ]
            )


class TestEmptyManifestPreservesExistingBehavior:
    """Empty manifests fail loudly (preserves existing parser contract)."""

    def test_empty_list_raises(self) -> None:
        with pytest.raises(RuntimeError, match="did not contain any downloadable"):
            _parse_payload([])


class TestImmutability:
    """``VersionManifest`` is a frozen value type."""

    def test_is_frozen(self) -> None:
        manifest = _parse_payload([{"path": "a.json"}])
        with pytest.raises((AttributeError, TypeError)):
            manifest.files = ()  # type: ignore[misc]

    def test_files_is_tuple(self) -> None:
        manifest = _parse_payload([{"path": "a.json"}])
        assert isinstance(manifest.files, tuple)
