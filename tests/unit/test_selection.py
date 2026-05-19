"""Tests for the SelectionPlan value type."""

import pytest

from nemar import _bids
from nemar._models import DatasetFile
from nemar._selection import SelectionPlan


def _files(paths: list[str]) -> list[DatasetFile]:
    return [
        DatasetFile(path=path, url=f"https://data.nemar.org/nm000132/v1.0.0/{path}")
        for path in paths
    ]


def test_build_round_trips_trivial_selection() -> None:
    """An empty query with no include/exclude returns all non-dotfiles in order."""
    paths = [
        "dataset_description.json",
        "participants.tsv",
        "sub-001/eeg/sub-001_task-MMN_eeg.set",
    ]
    files = _files(paths)

    plan = SelectionPlan.build(
        files,
        query=_bids.BidsQuery(),
        include=[],
        exclude=[],
    )

    assert [f.path for f in plan.final] == paths
    assert plan.matched_by_query == tuple(paths)
    assert plan.included_by_pattern == {}
    assert plan.excluded_by_pattern == {}
    assert plan.unmatched_includes == ()


def test_essential_kept_contains_present_essentials() -> None:
    """Essential BIDS root files survive when they exist in the manifest."""
    files = _files(
        [
            "dataset_description.json",
            "participants.tsv",
            "sub-001/eeg/sub-001_task-MMN_eeg.set",
            "sub-001/eeg/sub-001_task-MMN_eeg.fdt",
        ]
    )

    plan = SelectionPlan.build(
        files,
        query=_bids.BidsQuery(),
        include=["sub-001/eeg"],
        exclude=["*.fdt"],
    )

    assert "dataset_description.json" in plan.essential_kept
    assert "participants.tsv" in plan.essential_kept


def test_essential_kept_never_contains_absent_essentials() -> None:
    """Essential names that aren't in the manifest are not synthesized."""
    files = _files(
        [
            "dataset_description.json",
            "sub-001/eeg/sub-001_task-MMN_eeg.set",
        ]
    )

    plan = SelectionPlan.build(
        files,
        query=_bids.BidsQuery(),
        include=[],
        exclude=[],
    )

    assert "participants.tsv" not in plan.essential_kept
    assert "README" not in plan.essential_kept
    assert "dataset_description.json" in plan.essential_kept


def test_excluded_by_pattern_records_paths_but_keeps_essentials() -> None:
    """Excluding essentials by pattern is overridden by the essential carve-out."""
    files = _files(
        [
            "dataset_description.json",
            "participants.tsv",
            "sub-001/eeg/sub-001_task-MMN_eeg.set",
        ]
    )

    plan = SelectionPlan.build(
        files,
        query=_bids.BidsQuery(),
        include=[],
        exclude=["*.tsv", "*.json"],
    )

    final_paths = {f.path for f in plan.final}
    assert "dataset_description.json" in final_paths
    assert "participants.tsv" in final_paths
    # The exclude record itself reports the raw match, even if essentials
    # ultimately survive.
    assert "participants.tsv" in plan.excluded_by_pattern["*.tsv"]
    assert "dataset_description.json" in plan.excluded_by_pattern["*.json"]


def test_unmatched_includes_populated_for_literal_miss() -> None:
    """An include pattern that matches nothing is recorded in unmatched_includes."""
    files = _files(
        [
            "participants.tsv",
            "sub-001/eeg/sub-001_task-MMN_eeg.set",
        ]
    )

    plan = SelectionPlan.build(
        files,
        query=_bids.BidsQuery(),
        include=["participant.tsv"],
        exclude=[],
    )

    assert "participant.tsv" in plan.unmatched_includes


def test_raise_if_unmatched_includes_suggests_close_match() -> None:
    """The error preserves the did-you-mean hint for typos."""
    files = _files(
        [
            "participants.tsv",
            "sub-001/eeg/sub-001_task-MMN_eeg.set",
        ]
    )

    plan = SelectionPlan.build(
        files,
        query=_bids.BidsQuery(),
        include=["participant.tsv"],
        exclude=[],
    )

    with pytest.raises(RuntimeError, match="Perhaps you mean"):
        plan.raise_if_unmatched_includes(filenames=[f.path for f in files])


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
        SelectionPlan.build(
            files,
            query=_bids.BidsQuery.from_filters(subject="999"),
            include=[],
            exclude=[],
        )

    message = str(info.value)
    assert "Available:" in message
    assert "subjects=" in message
    assert "001" in message
    assert "002" in message
    assert "003" in message
