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
    """``VersionManifest`` trusts manifest-advertised file URLs.

    The production ``data.nemar.org`` manifest mixes
    ``raw.githubusercontent.com`` (git-tracked small files) and
    ``nemar.s3.us-east-2.amazonaws.com`` (annex content) origins
    alongside ``data.nemar.org`` itself. Per-file origin scoping was
    removed once we confirmed that contract; the endpoint still
    validates the index + manifest fetches via
    :func:`nemar._transport.fetch_json`, but file URLs come from the
    trusted manifest payload and are taken as-is.
    """

    def test_off_origin_file_url_is_accepted(self) -> None:
        """Manifest entries with non-``data.nemar.org`` URLs parse cleanly."""
        manifest = _parse_payload(
            [
                {"path": "ok.json"},
                {
                    "path": "big.set",
                    "url": (
                        "https://nemar.s3.us-east-2.amazonaws.com/"
                        "annex/objects/.../big.set"
                    ),
                },
                {
                    "path": "config",
                    "url": (
                        "https://raw.githubusercontent.com/"
                        "nemarDatasets/nm000132/v1.0.0/config"
                    ),
                },
            ]
        )
        urls = {f.path: f.url for f in manifest}
        assert "nemar.s3.us-east-2.amazonaws.com" in urls["big.set"]
        assert "raw.githubusercontent.com" in urls["config"]


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


class TestPathLookup:
    """``VersionManifest`` exposes O(1) lookup by manifest-relative path."""

    def test_file_returns_dataset_file_for_existing_path(self) -> None:
        manifest = _parse_payload(
            [
                {"path": "dataset_description.json", "size": 12},
                {"path": "sub-001/eeg/sub-001_task-MMN_eeg.set", "size": 99},
            ]
        )

        resolved = manifest.file("sub-001/eeg/sub-001_task-MMN_eeg.set")

        assert isinstance(resolved, DatasetFile)
        assert resolved.path == "sub-001/eeg/sub-001_task-MMN_eeg.set"
        assert resolved.size == 99
        # The same DatasetFile instance the iterator would surface — no copy.
        assert resolved is manifest.files[1]

    def test_file_missing_path_raises_runtime_error_with_message(self) -> None:
        manifest = _parse_payload([{"path": "dataset_description.json"}])

        missing = "sub-001/eeg/sub-001_task-MMN_eeg.set"
        with pytest.raises(RuntimeError) as excinfo:
            manifest.file(missing)

        message = str(excinfo.value)
        assert missing in message
        assert MANIFEST_URL in message

    def test_contains_reports_presence(self) -> None:
        manifest = _parse_payload(
            [
                {"path": "a.json"},
                {"path": "sub-001/eeg/sub-001_task-MMN_eeg.set"},
            ]
        )

        assert "a.json" in manifest
        assert "sub-001/eeg/sub-001_task-MMN_eeg.set" in manifest
        assert "missing.json" not in manifest

    def test_lookup_works_for_dict_shaped_payload(self) -> None:
        # The ``files``-keyed mapping shape exercises ``_mapping_entries``.
        manifest = _parse_payload(
            {
                "files": {
                    "dataset_description.json": {"size": 12},
                    "sub-001/eeg/sub-001_task-MMN_eeg.set": {"size": 99},
                }
            }
        )

        resolved = manifest.file("sub-001/eeg/sub-001_task-MMN_eeg.set")
        assert resolved.path == "sub-001/eeg/sub-001_task-MMN_eeg.set"
        assert resolved.size == 99
        assert "dataset_description.json" in manifest
        assert "absent.json" not in manifest
