"""Tests for the on-disk LocalDataset value type."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from nemar._local_dataset import LocalDataset

# ---------------------------------------------------------------------------
# from_dir: the "no compatibility check needed" cases return None.
# ---------------------------------------------------------------------------


def test_from_dir_returns_none_for_nonexistent_dir(tmp_path: Path) -> None:
    """A target directory that doesn't exist yet is a clean slate."""
    missing = tmp_path / "does_not_exist"

    assert LocalDataset.from_dir(missing) is None


def test_from_dir_returns_none_for_empty_dir(tmp_path: Path) -> None:
    """An empty target directory is a clean slate."""
    assert LocalDataset.from_dir(tmp_path) is None


def test_from_dir_returns_none_for_dir_without_dataset_description(
    tmp_path: Path,
) -> None:
    """A target with files but no dataset_description.json is treated as resumable.

    Resume-friendly behavior: previous interrupted downloads can re-run
    without forcing the user to clear the directory.
    """
    (tmp_path / "partial.tmp").write_text("x", encoding="utf-8")

    assert LocalDataset.from_dir(tmp_path) is None


# ---------------------------------------------------------------------------
# from_dir: a populated dataset_description.json yields a LocalDataset.
# ---------------------------------------------------------------------------


def test_from_dir_parses_dataset_description(tmp_path: Path) -> None:
    """When dataset_description.json exists it produces a LocalDataset."""
    (tmp_path / "dataset_description.json").write_text(
        json.dumps(
            {
                "DatasetDOI": "doi:10.82901/nemar.nm000132",
                "Version": "1.0.0",
            }
        ),
        encoding="utf-8",
    )

    local = LocalDataset.from_dir(tmp_path)

    assert local is not None
    assert local.dataset_id == "nm000132"
    assert local.version_tag == "v1.0.0"


def test_from_dir_strips_doi_prefix(tmp_path: Path) -> None:
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

    local = LocalDataset.from_dir(tmp_path)

    assert local is not None
    assert local.dataset_id == "nm000132"
    assert local.version_tag == "v1.0.0"


def test_from_dir_handles_utf8_bom(tmp_path: Path) -> None:
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

    local = LocalDataset.from_dir(tmp_path)

    assert local is not None
    assert local.dataset_id == "nm000132"
    assert local.version_tag == "v1.0.0"


def test_from_dir_yields_none_version_when_missing(tmp_path: Path) -> None:
    """Missing Version is preserved (no value type lying about its content)."""
    (tmp_path / "dataset_description.json").write_text(
        json.dumps({"DatasetDOI": "doi:10.82901/nemar.nm000132"}),
        encoding="utf-8",
    )

    local = LocalDataset.from_dir(tmp_path)

    assert local is not None
    assert local.dataset_id == "nm000132"
    assert local.version_tag is None


def test_from_dir_raises_on_malformed_json(tmp_path: Path) -> None:
    """A corrupt dataset_description.json must not be silently ignored."""
    (tmp_path / "dataset_description.json").write_text("{bad", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Could not parse local"):
        LocalDataset.from_dir(tmp_path)


def test_from_dir_raises_when_doi_missing(tmp_path: Path) -> None:
    """A dataset_description without DatasetDOI cannot be reasoned about."""
    (tmp_path / "dataset_description.json").write_text(
        json.dumps({"Name": "something"}), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match='does not contain "DatasetDOI"'):
        LocalDataset.from_dir(tmp_path)


def test_from_dir_raises_on_non_string_doi(tmp_path: Path) -> None:
    """A non-string DatasetDOI is malformed."""
    (tmp_path / "dataset_description.json").write_text(
        json.dumps({"DatasetDOI": 123}), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="DatasetDOI.*string"):
        LocalDataset.from_dir(tmp_path)


def test_from_dir_raises_on_non_string_version(tmp_path: Path) -> None:
    """A non-string Version is malformed."""
    (tmp_path / "dataset_description.json").write_text(
        json.dumps({"DatasetDOI": "10.82901/nemar.nm000132", "Version": 100}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Version.*string"):
        LocalDataset.from_dir(tmp_path)


# ---------------------------------------------------------------------------
# assert_compatible_with: the actual compatibility decision.
# ---------------------------------------------------------------------------


def test_assert_compatible_passes_when_dataset_and_tag_match(tmp_path: Path) -> None:
    """A compatible LocalDataset returns without raising."""
    (tmp_path / "dataset_description.json").write_text(
        json.dumps(
            {
                "DatasetDOI": "doi:10.82901/nemar.nm000132",
                "Version": "1.0.0",
            }
        ),
        encoding="utf-8",
    )
    local = LocalDataset.from_dir(tmp_path)
    assert local is not None

    local.assert_compatible_with(dataset="nm000132", tag="v1.0.0")


def test_assert_compatible_raises_on_doi_mismatch(tmp_path: Path) -> None:
    """A DOI for a different dataset is a hard stop."""
    (tmp_path / "dataset_description.json").write_text(
        json.dumps({"DatasetDOI": "10.82901/nemar.nm000999"}),
        encoding="utf-8",
    )
    local = LocalDataset.from_dir(tmp_path)
    assert local is not None

    with pytest.raises(RuntimeError, match="different NEMAR dataset"):
        local.assert_compatible_with(dataset="nm000132", tag="v1.1.1")


def test_assert_compatible_doi_mismatch_names_both(tmp_path: Path) -> None:
    """The DOI mismatch message names both the local DOI and the requested dataset."""
    (tmp_path / "dataset_description.json").write_text(
        json.dumps({"DatasetDOI": "10.82901/nemar.nm000999"}),
        encoding="utf-8",
    )
    local = LocalDataset.from_dir(tmp_path)
    assert local is not None

    with pytest.raises(RuntimeError) as exc:
        local.assert_compatible_with(dataset="nm000132", tag="v1.1.1")

    message = str(exc.value)
    assert "10.82901/nemar.nm000999" in message
    assert "nm000132" in message


def test_assert_compatible_raises_on_missing_version_with_matching_doi(
    tmp_path: Path,
) -> None:
    """Matching DOI but missing Version is a data-loss guard."""
    (tmp_path / "dataset_description.json").write_text(
        json.dumps({"DatasetDOI": "10.82901/nemar.nm000132"}),
        encoding="utf-8",
    )
    local = LocalDataset.from_dir(tmp_path)
    assert local is not None

    with pytest.raises(RuntimeError, match="does not contain a.*Version.*field"):
        local.assert_compatible_with(dataset="nm000132", tag="v1.1.1")


def test_assert_compatible_raises_on_version_mismatch(tmp_path: Path) -> None:
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
    local = LocalDataset.from_dir(tmp_path)
    assert local is not None

    with pytest.raises(FileExistsError, match="v1.0.0 exists locally"):
        local.assert_compatible_with(dataset="nm000132", tag="v1.1.1")


def test_assert_compatible_version_mismatch_names_both_versions(
    tmp_path: Path,
) -> None:
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
    local = LocalDataset.from_dir(tmp_path)
    assert local is not None

    with pytest.raises(FileExistsError) as exc:
        local.assert_compatible_with(dataset="nm000132", tag="v1.1.1")

    message = str(exc.value)
    assert "v1.1.1" in message
    assert "v1.0.0" in message


# ---------------------------------------------------------------------------
# Value-type properties: frozen, no I/O on instance methods, no mutation.
# ---------------------------------------------------------------------------


def test_local_dataset_is_frozen(tmp_path: Path) -> None:
    """LocalDataset is a value type and must reject attribute mutation."""
    (tmp_path / "dataset_description.json").write_text(
        json.dumps(
            {
                "DatasetDOI": "doi:10.82901/nemar.nm000132",
                "Version": "1.0.0",
            }
        ),
        encoding="utf-8",
    )
    local = LocalDataset.from_dir(tmp_path)
    assert local is not None

    with pytest.raises(dataclasses.FrozenInstanceError):
        local.dataset_id = "nm999999"  # type: ignore[misc]


def test_assert_compatible_does_no_disk_io(tmp_path: Path) -> None:
    """Instance methods do not touch the filesystem after construction.

    We construct, delete the dir on disk, then assert compatibility. If
    the instance ever re-reads the file it will explode here.
    """
    (tmp_path / "dataset_description.json").write_text(
        json.dumps(
            {
                "DatasetDOI": "doi:10.82901/nemar.nm000132",
                "Version": "1.0.0",
            }
        ),
        encoding="utf-8",
    )
    local = LocalDataset.from_dir(tmp_path)
    assert local is not None

    # Nuke the source after construction; the instance must not care.
    (tmp_path / "dataset_description.json").unlink()
    tmp_path.rmdir()

    local.assert_compatible_with(dataset="nm000132", tag="v1.0.0")
