"""Tests for the local target-directory compatibility check.

The check used to live on a two-method ``LocalDataset`` dataclass; it was
inlined into the single private :func:`_check_local_compatibility`
function because ``_run`` was its only caller. These tests exercise the
function directly; the orchestrator wiring is covered by
``tests/integration/test_filesystem.py::test_existing_wrong_version_blocks_download``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nemar._download import _check_local_compatibility

# ---------------------------------------------------------------------------
# Silent early-return cases: no clobber risk, no exception raised.
# ---------------------------------------------------------------------------


def test_returns_silently_for_nonexistent_dir(tmp_path: Path) -> None:
    """A target directory that doesn't exist yet is a clean slate."""
    missing = tmp_path / "does_not_exist"

    _check_local_compatibility(missing, dataset="nm000132", tag="v1.0.0")


def test_returns_silently_for_empty_dir(tmp_path: Path) -> None:
    """An empty target directory is a clean slate."""
    _check_local_compatibility(tmp_path, dataset="nm000132", tag="v1.0.0")


def test_returns_silently_for_dir_without_dataset_description(
    tmp_path: Path,
) -> None:
    """A target with files but no dataset_description.json is treated as resumable.

    Resume-friendly behavior: previous interrupted downloads can re-run
    without forcing the user to clear the directory. The orchestrator logs
    a notice via ``_target_has_files_without_description``; the
    compatibility check itself just stays silent.
    """
    (tmp_path / "partial.tmp").write_text("x", encoding="utf-8")

    _check_local_compatibility(tmp_path, dataset="nm000132", tag="v1.0.0")


# ---------------------------------------------------------------------------
# A populated dataset_description.json is parsed and matched against the
# requested (dataset, tag).
# ---------------------------------------------------------------------------


def test_passes_when_dataset_and_tag_match(tmp_path: Path) -> None:
    """A matching DOI + matching Version returns without raising."""
    (tmp_path / "dataset_description.json").write_text(
        json.dumps(
            {
                "DatasetDOI": "doi:10.82901/nemar.nm000132",
                "Version": "1.0.0",
            }
        ),
        encoding="utf-8",
    )

    _check_local_compatibility(tmp_path, dataset="nm000132", tag="v1.0.0")


def test_passes_when_doi_lacks_doi_prefix(tmp_path: Path) -> None:
    """A bare DOI without the ``doi:`` prefix is also accepted."""
    (tmp_path / "dataset_description.json").write_text(
        json.dumps(
            {
                "DatasetDOI": "10.82901/nemar.nm000132",
                "Version": "1.0.0",
            }
        ),
        encoding="utf-8",
    )

    _check_local_compatibility(tmp_path, dataset="nm000132", tag="v1.0.0")


def test_handles_utf8_bom(tmp_path: Path) -> None:
    """A UTF-8 BOM-prefixed dataset_description.json is parsed correctly."""
    payload = json.dumps(
        {
            "DatasetDOI": "doi:10.82901/nemar.nm000132",
            "Version": "1.0.0",
        }
    )
    # Some editors (Notepad, legacy Excel CSV exporters) prepend a UTF-8
    # BOM that breaks naive ``json.loads(read_text("utf-8"))``.
    (tmp_path / "dataset_description.json").write_bytes(
        b"\xef\xbb\xbf" + payload.encode("utf-8")
    )

    _check_local_compatibility(tmp_path, dataset="nm000132", tag="v1.0.0")


# ---------------------------------------------------------------------------
# Malformed dataset_description.json must not be silently ignored.
# ---------------------------------------------------------------------------


def test_raises_on_malformed_json(tmp_path: Path) -> None:
    """A corrupt dataset_description.json must not be silently ignored."""
    (tmp_path / "dataset_description.json").write_text("{bad", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Could not parse local"):
        _check_local_compatibility(tmp_path, dataset="nm000132", tag="v1.0.0")


def test_raises_when_doi_missing(tmp_path: Path) -> None:
    """A dataset_description without DatasetDOI cannot be reasoned about."""
    (tmp_path / "dataset_description.json").write_text(
        json.dumps({"Name": "something"}), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match='does not contain "DatasetDOI"'):
        _check_local_compatibility(tmp_path, dataset="nm000132", tag="v1.0.0")


def test_raises_on_non_string_doi(tmp_path: Path) -> None:
    """A non-string DatasetDOI is malformed."""
    (tmp_path / "dataset_description.json").write_text(
        json.dumps({"DatasetDOI": 123}), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="DatasetDOI.*string"):
        _check_local_compatibility(tmp_path, dataset="nm000132", tag="v1.0.0")


def test_raises_on_non_string_version(tmp_path: Path) -> None:
    """A non-string Version is malformed."""
    (tmp_path / "dataset_description.json").write_text(
        json.dumps({"DatasetDOI": "10.82901/nemar.nm000132", "Version": 100}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Version.*string"):
        _check_local_compatibility(tmp_path, dataset="nm000132", tag="v1.0.0")


# ---------------------------------------------------------------------------
# Compatibility-decision cases: the actual semantic gates.
# ---------------------------------------------------------------------------


def test_raises_on_doi_mismatch(tmp_path: Path) -> None:
    """A DOI for a different dataset is a hard stop."""
    (tmp_path / "dataset_description.json").write_text(
        json.dumps({"DatasetDOI": "10.82901/nemar.nm000999"}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="different NEMAR dataset"):
        _check_local_compatibility(tmp_path, dataset="nm000132", tag="v1.1.1")


def test_doi_mismatch_message_names_both(tmp_path: Path) -> None:
    """The DOI mismatch message names both the local DOI and the requested dataset."""
    (tmp_path / "dataset_description.json").write_text(
        json.dumps({"DatasetDOI": "10.82901/nemar.nm000999"}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError) as exc:
        _check_local_compatibility(tmp_path, dataset="nm000132", tag="v1.1.1")

    message = str(exc.value)
    assert "10.82901/nemar.nm000999" in message
    assert "nm000132" in message


def test_raises_on_missing_version_with_matching_doi(tmp_path: Path) -> None:
    """Matching DOI but missing Version is a data-loss guard."""
    (tmp_path / "dataset_description.json").write_text(
        json.dumps({"DatasetDOI": "10.82901/nemar.nm000132"}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="does not contain a.*Version.*field"):
        _check_local_compatibility(tmp_path, dataset="nm000132", tag="v1.1.1")


def test_raises_on_version_mismatch(tmp_path: Path) -> None:
    """A mismatched Version raises FileExistsError, not RuntimeError.

    The distinction matters: callers may catch FileExistsError to
    surface a recovery hint ("use a different target directory") that
    is meaningless for other errors.
    """
    (tmp_path / "dataset_description.json").write_text(
        json.dumps(
            {
                "DatasetDOI": "10.82901/nemar.nm000132",
                "Version": "1.0.0",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError, match="v1.0.0 exists locally"):
        _check_local_compatibility(tmp_path, dataset="nm000132", tag="v1.1.1")


def test_version_mismatch_message_names_both_versions(tmp_path: Path) -> None:
    """The FileExistsError message names both the requested and the local tag."""
    (tmp_path / "dataset_description.json").write_text(
        json.dumps(
            {
                "DatasetDOI": "10.82901/nemar.nm000132",
                "Version": "1.0.0",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError) as exc:
        _check_local_compatibility(tmp_path, dataset="nm000132", tag="v1.1.1")

    message = str(exc.value)
    assert "v1.1.1" in message
    assert "v1.0.0" in message
