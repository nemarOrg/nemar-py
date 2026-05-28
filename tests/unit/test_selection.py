"""Tests for the BIDS file selection function."""

import pytest

from nemar._models import BidsQuery, DatasetFile
from nemar._request import build_bids_query
from nemar._selection import (
    SelectionResult,
    raise_if_unmatched_includes,
    select_files,
)
from tests.fixtures.factories import make_dataset_file


def _files(paths: list[str]) -> list[DatasetFile]:
    return [make_dataset_file(path) for path in paths]


def test_select_returns_all_non_dotfiles_for_trivial_selection() -> None:
    """An empty query with no include/exclude returns all non-dotfiles in order."""
    paths = [
        "dataset_description.json",
        "participants.tsv",
        "sub-001/eeg/sub-001_task-MMN_eeg.set",
    ]
    files = _files(paths)

    result = select_files(
        files,
        query=BidsQuery(),
        include=[],
        exclude=[],
    )

    assert isinstance(result, SelectionResult)
    assert [f.path for f in result.selected] == paths
    assert result.unmatched_includes == ()


def test_essentials_survive_when_present_in_manifest() -> None:
    """Essential BIDS root files survive even when an exclude pattern hits them."""
    files = _files(
        [
            "dataset_description.json",
            "participants.tsv",
            "sub-001/eeg/sub-001_task-MMN_eeg.set",
            "sub-001/eeg/sub-001_task-MMN_eeg.fdt",
        ]
    )

    result = select_files(
        files,
        query=BidsQuery(),
        include=["sub-001/eeg"],
        exclude=["*.fdt"],
    )

    selected_paths = {f.path for f in result.selected}
    assert "dataset_description.json" in selected_paths
    assert "participants.tsv" in selected_paths
    # The recording survives the include; the .fdt does not.
    assert "sub-001/eeg/sub-001_task-MMN_eeg.set" in selected_paths
    assert "sub-001/eeg/sub-001_task-MMN_eeg.fdt" not in selected_paths


def test_absent_essentials_are_not_synthesized() -> None:
    """Essential names that aren't in the manifest are not invented."""
    files = _files(
        [
            "dataset_description.json",
            "sub-001/eeg/sub-001_task-MMN_eeg.set",
        ]
    )

    result = select_files(
        files,
        query=BidsQuery(),
        include=[],
        exclude=[],
    )

    selected_paths = {f.path for f in result.selected}
    assert "participants.tsv" not in selected_paths
    assert "README" not in selected_paths
    assert "dataset_description.json" in selected_paths


def test_root_sidecars_auto_included_when_bids_query_active() -> None:
    """A BIDS query auto-pulls root-level ``*.json`` / ``*.tsv`` sidecars.

    Real NEMAR datasets keep inherited BIDS sidecars at the dataset root
    (``task-typing_events.json``, ``space-*_coordsystem.json``, etc.).
    Without this auto-include, a ``subject=...`` query downloads the
    recording and its in-directory sidecars but leaves the inherited
    root sidecars behind — an analyst then can't decode event columns
    or coordinate systems. Pinning the contract here documents the
    Option-A trade-off: small over-fetch (root JSON/TSV are tiny), no
    BIDS-inheritance walk.
    """
    files = _files(
        [
            "dataset_description.json",
            "participants.tsv",
            "task-typing_events.json",  # inherited sidecar (applies)
            "space-leftForearm_coordsystem.json",  # inherited sidecar (applies)
            "task-other_events.json",  # inherited sidecar (does NOT apply)
            "sub-001/eeg/sub-001_task-MMN_eeg.set",
            "sub-002/eeg/sub-002_task-MMN_eeg.set",
        ]
    )

    result = select_files(
        files,
        query=build_bids_query(subject="001"),
        include=[],
        exclude=[],
    )

    selected_paths = {f.path for f in result.selected}
    # The recording lands.
    assert "sub-001/eeg/sub-001_task-MMN_eeg.set" in selected_paths
    # The unrelated subject does NOT land.
    assert "sub-002/eeg/sub-002_task-MMN_eeg.set" not in selected_paths
    # Root sidecars land — including the over-fetched ``task-other_events.json``
    # that doesn't strictly apply (the Option-A trade-off vs Option-B).
    assert "task-typing_events.json" in selected_paths
    assert "space-leftForearm_coordsystem.json" in selected_paths
    assert "task-other_events.json" in selected_paths
    # Existing essentials still land.
    assert "dataset_description.json" in selected_paths
    assert "participants.tsv" in selected_paths


def test_root_sidecars_not_auto_included_without_bids_query() -> None:
    """No BIDS query → root sidecar sweep stays off.

    Without a BIDS query the user is asking for the whole manifest (or
    using ``include`` patterns for precise selection); auto-including
    root JSON/TSV would either be redundant (everything is selected
    anyway) or surprising (path-include is meant to be precise). The
    existing ``ESSENTIAL_BIDS_FILES`` carve-out still applies.
    """
    files = _files(
        [
            "dataset_description.json",
            "task-typing_events.json",
            "sub-001/eeg/sub-001_task-MMN_eeg.set",
        ]
    )

    result = select_files(
        files,
        query=BidsQuery(),  # empty
        include=["sub-001/**"],
        exclude=[],
    )

    selected_paths = {f.path for f in result.selected}
    # Recording landed via the path include.
    assert "sub-001/eeg/sub-001_task-MMN_eeg.set" in selected_paths
    # Essential still kept.
    assert "dataset_description.json" in selected_paths
    # Root sidecar NOT swept (no BIDS query active).
    assert "task-typing_events.json" not in selected_paths


def test_root_sidecars_sweep_skips_subdirectory_jsons() -> None:
    """The sweep is root-only: deep JSON/TSV files do not enter the
    essential set by virtue of having a sidecar suffix.

    Otherwise a BIDS query would unintentionally pull every JSON in
    every subject's directory across the dataset, defeating the point
    of the query.
    """
    files = _files(
        [
            "dataset_description.json",
            "sub-001/eeg/sub-001_task-MMN_eeg.json",  # not root
            "sub-001/eeg/sub-001_task-MMN_eeg.set",
            "sub-002/eeg/sub-002_task-MMN_eeg.json",  # not root, unrelated subject
            "sub-002/eeg/sub-002_task-MMN_eeg.set",
        ]
    )

    result = select_files(
        files,
        query=build_bids_query(subject="001"),
        include=[],
        exclude=[],
    )

    selected_paths = {f.path for f in result.selected}
    assert "sub-001/eeg/sub-001_task-MMN_eeg.set" in selected_paths
    assert "sub-001/eeg/sub-001_task-MMN_eeg.json" in selected_paths
    # The other subject's sidecar is NOT swept — the rule is root-only.
    assert "sub-002/eeg/sub-002_task-MMN_eeg.json" not in selected_paths
    assert "sub-002/eeg/sub-002_task-MMN_eeg.set" not in selected_paths


def test_subject_query_does_not_match_derivatives_by_default() -> None:
    """A ``subject="02"`` query with no explicit scope stays within raw.

    Regression guard against the nm000133 over-fetch: previously, a
    subject query matched ``sub-02`` anywhere — including
    ``derivatives/<pipeline>/sub-02/...`` — pulling 260+ MB of epoched
    derivative data the caller didn't ask for. The fix defaults
    ``BidsQuery.from_filters(scope=None)`` to ``scope="raw"``.

    Callers who want derivatives still get them by passing
    ``scope="derivatives"`` (or ``["raw","derivatives"]``).
    """
    files = _files(
        [
            "dataset_description.json",
            "sub-02/ses-01/eeg/sub-02_ses-01_task-images_eeg.bdf",
            "sub-02/ses-01/eeg/sub-02_ses-01_task-images_events.tsv",
            "derivatives/epoched/sub-02/ses-01/eeg/sub-02_ses-01_task-images_epo.fif",
        ]
    )

    result = select_files(
        files,
        query=build_bids_query(subject="02"),
        include=[],
        exclude=[],
    )

    selected = {f.path for f in result.selected}
    assert "sub-02/ses-01/eeg/sub-02_ses-01_task-images_eeg.bdf" in selected
    assert "sub-02/ses-01/eeg/sub-02_ses-01_task-images_events.tsv" in selected
    # The over-fetch is gone — derivatives stay opt-in.
    assert (
        "derivatives/epoched/sub-02/ses-01/eeg/sub-02_ses-01_task-images_epo.fif"
        not in selected
    )


def test_explicit_derivatives_scope_still_works() -> None:
    """Callers who opt into derivatives get them.

    Confirms the default-to-raw fix doesn't break the documented
    workflow ``scope="derivatives", pipeline=...``.
    """
    files = _files(
        [
            "sub-02/ses-01/eeg/sub-02_ses-01_task-images_eeg.bdf",
            "derivatives/epoched/sub-02/ses-01/eeg/sub-02_ses-01_task-images_epo.fif",
        ]
    )

    result = select_files(
        files,
        query=build_bids_query(
            subject="02",
            scope="derivatives",
            pipeline="epoched",
        ),
        include=[],
        exclude=[],
    )

    selected = {f.path for f in result.selected}
    # Only the derivative landed — raw stays out because scope was
    # narrowed to derivatives only.
    assert (
        "derivatives/epoched/sub-02/ses-01/eeg/sub-02_ses-01_task-images_epo.fif"
        in selected
    )
    assert "sub-02/ses-01/eeg/sub-02_ses-01_task-images_eeg.bdf" not in selected


def test_exclude_pattern_does_not_strip_essentials() -> None:
    """Excluding essentials by pattern is overridden by the essential carve-out."""
    files = _files(
        [
            "dataset_description.json",
            "participants.tsv",
            "sub-001/eeg/sub-001_task-MMN_eeg.set",
        ]
    )

    result = select_files(
        files,
        query=BidsQuery(),
        include=[],
        exclude=["*.tsv", "*.json"],
    )

    final_paths = {f.path for f in result.selected}
    # The essentials survive the exclude — they're force-kept.
    assert "dataset_description.json" in final_paths
    assert "participants.tsv" in final_paths
    # Non-essential files matched by the exclude are dropped.
    assert "sub-001/eeg/sub-001_task-MMN_eeg.set" in final_paths


def test_unmatched_includes_populated_for_literal_miss() -> None:
    """An include pattern that matches nothing is recorded in unmatched_includes."""
    files = _files(
        [
            "participants.tsv",
            "sub-001/eeg/sub-001_task-MMN_eeg.set",
        ]
    )

    result = select_files(
        files,
        query=BidsQuery(),
        include=["participant.tsv"],
        exclude=[],
    )

    assert "participant.tsv" in result.unmatched_includes


def test_raise_if_unmatched_includes_suggests_close_match() -> None:
    """The error preserves the did-you-mean hint for typos."""
    files = _files(
        [
            "participants.tsv",
            "sub-001/eeg/sub-001_task-MMN_eeg.set",
        ]
    )

    result = select_files(
        files,
        query=BidsQuery(),
        include=["participant.tsv"],
        exclude=[],
    )

    with pytest.raises(RuntimeError, match="Perhaps you mean"):
        raise_if_unmatched_includes(result, filenames=[f.path for f in files])


def test_raise_if_unmatched_includes_no_suggestion_for_unique_pattern() -> None:
    """A literal pattern with no close match falls back to no-suggestion error."""
    files = _files(["sub-001/eeg/sub-001_task-MMN_eeg.set"])

    result = select_files(
        files,
        query=BidsQuery(),
        include=["totally-unrelated-name"],
        exclude=[],
    )

    with pytest.raises(RuntimeError) as info:
        raise_if_unmatched_includes(result, filenames=[f.path for f in files])

    assert "Perhaps you mean" not in str(info.value)
    assert "Could not find path in the NEMAR manifest" in str(info.value)


def test_raise_if_unmatched_includes_is_noop_when_all_matched() -> None:
    """No unmatched includes → no raise. The helper just returns."""
    files = _files(
        [
            "participants.tsv",
            "sub-001/eeg/sub-001_task-MMN_eeg.set",
        ]
    )

    result = select_files(
        files,
        query=BidsQuery(),
        include=["participants.tsv"],
        exclude=[],
    )

    # Should not raise; result.unmatched_includes is empty.
    raise_if_unmatched_includes(result, filenames=[f.path for f in files])
    assert result.unmatched_includes == ()


def test_zero_match_query_without_hintable_entities_omits_available() -> None:
    """When the manifest has no subjects/sessions/tasks, no Available: tail is added."""
    files = _files(["dataset_description.json", "README.md"])

    with pytest.raises(RuntimeError) as info:
        select_files(
            files,
            query=build_bids_query(subject="001"),
            include=[],
            exclude=[],
        )

    message = str(info.value)
    assert "No files matched the BIDS query" in message
    assert "Available:" not in message


def test_zero_match_query_lists_available_entities() -> None:
    """When a non-empty query matches nothing, the error echoes the manifest."""
    files = _files(
        [
            "sub-001/eeg/sub-001_task-MMN_eeg.set",
            "sub-002/eeg/sub-002_task-MMN_eeg.set",
            "sub-003/eeg/sub-003_task-P3_eeg.set",
        ]
    )

    with pytest.raises(RuntimeError) as info:
        select_files(
            files,
            query=build_bids_query(subject="999"),
            include=[],
            exclude=[],
        )

    message = str(info.value)
    assert "Available:" in message
    assert "subjects=" in message
    assert "001" in message
    assert "002" in message
    assert "003" in message
