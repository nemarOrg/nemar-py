"""Tests for BIDS-aware query parsing.

The value types (:class:`BidsPath`, :class:`BidsQuery`) live in
:mod:`nemar._models`; building a query from raw kwargs
(:func:`build_bids_query`) is request normalization and lives in
:mod:`nemar._request`; matching a parsed path against a query
(:func:`_path_matches`) lives in :mod:`nemar._selection`. Tests import
each from its real home.
"""

import pytest

from nemar._bids import BidsPath, _path_matches, build_bids_query


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        pytest.param(
            "sub-001/ses-01/eeg/sub-001_ses-01_task-MMN_acq-high_run-02_eeg.set",
            {
                "entities": {
                    "sub": "001",
                    "ses": "01",
                    "task": "MMN",
                    "acq": "high",
                    "run": "02",
                },
                "datatype": "eeg",
                "suffix": "eeg",
                "extension": ".set",
                "scope": "raw",
                "pipeline": None,
            },
            id="raw-eeg-set",
        ),
        pytest.param(
            "sub-001/anat/sub-001_T1w.nii.gz",
            {
                "entities": {"sub": "001"},
                "datatype": "anat",
                "suffix": "T1w",
                "extension": ".nii.gz",
                "scope": "raw",
                "pipeline": None,
            },
            id="compound-nifti-extension",
        ),
        pytest.param(
            "derivatives/eeglab/sub-001/eeg/sub-001_task-MMN_desc-clean_eeg.set",
            {
                "entities": {"sub": "001", "task": "MMN", "desc": "clean"},
                "datatype": "eeg",
                "suffix": "eeg",
                "extension": ".set",
                "scope": "derivatives",
                "pipeline": "eeglab",
            },
            id="derivative-pipeline",
        ),
        pytest.param(
            "stimuli/task-MMN/deviant_stereo.wav",
            {
                "entities": {"task": "MMN"},
                "datatype": None,
                "suffix": "stereo",
                "extension": ".wav",
                "scope": "stimuli",
                "pipeline": None,
            },
            id="stimuli-task-folder",
        ),
        pytest.param(
            "sourcedata/sub-001/eeg/sub-001_task-rest_eeg.edf",
            {
                "entities": {"sub": "001", "task": "rest"},
                "datatype": "eeg",
                "suffix": "eeg",
                "extension": ".edf",
                "scope": "sourcedata",
                "pipeline": None,
            },
            id="sourcedata-eeg",
        ),
        pytest.param(
            "participants.tsv",
            {
                "entities": {},
                "datatype": None,
                "suffix": "participants",
                "extension": ".tsv",
                "scope": "raw",
                "pipeline": None,
            },
            id="root-metadata",
        ),
        pytest.param(
            "sub-001/eeg/sub-001_task-MMN_events",
            {
                "entities": {"sub": "001", "task": "MMN"},
                "datatype": "eeg",
                "suffix": "events",
                "extension": None,
                "scope": "raw",
                "pipeline": None,
            },
            id="no-extension",
        ),
        pytest.param(
            "",
            {
                "entities": {},
                "datatype": None,
                "suffix": None,
                "extension": None,
                "scope": "raw",
                "pipeline": None,
            },
            id="empty-path",
        ),
    ],
)
def test_bids_path_parse_semantics(path, expected) -> None:
    """BIDS paths expose semantic entities, scope, suffix, and extension."""
    parsed = BidsPath.parse(path)

    assert parsed.entities == expected["entities"]
    assert parsed.datatype == expected["datatype"]
    assert parsed.suffix == expected["suffix"]
    assert parsed.extension == expected["extension"]
    assert parsed.scope == expected["scope"]
    assert parsed.pipeline == expected["pipeline"]


@pytest.mark.parametrize(
    ("path", "filters", "expected"),
    [
        pytest.param(
            "sub-001/eeg/sub-001_task-MMN_run-01_eeg.set",
            {
                "subject": "sub-001",
                "task": "MMN",
                "run": "run-01",
                "datatype": "eeg",
                "suffix": "eeg",
                "extension": "set",
            },
            True,
            id="prefixed-and-plain-labels",
        ),
        pytest.param(
            "sub-001/eeg/sub-001_task-MMN_desc-clean_eeg.set",
            {"entity": ["subject=001", "desc=clean"]},
            True,
            id="generic-entity-list",
        ),
        pytest.param(
            "sub-001/eeg/sub-001_task-MMN_desc-clean_eeg.set",
            {"entity": {"subject": "sub-001", "desc": ["clean"]}},
            True,
            id="generic-entity-mapping",
        ),
        pytest.param(
            "derivatives/eeglab/sub-001/eeg/sub-001_task-MMN_desc-clean_eeg.set",
            {"scope": "derivatives", "pipeline": "eeglab", "subject": "001"},
            True,
            id="derivative-pipeline",
        ),
        pytest.param(
            "stimuli/task-MMN/deviant_stereo.wav",
            {"scope": "stimuli", "task": "MMN"},
            True,
            id="stimuli-scope",
        ),
        pytest.param(
            "sub-001/eeg/sub-001_task-MMN_run-01_eeg.set",
            {"subject": "002"},
            False,
            id="subject-mismatch",
        ),
        pytest.param(
            "sub-001/eeg/sub-001_task-MMN_run-01_eeg.set",
            {"datatype": "beh"},
            False,
            id="datatype-mismatch",
        ),
        pytest.param(
            "sub-001/eeg/sub-001_task-MMN_run-01_eeg.set",
            {"suffix": "events"},
            False,
            id="suffix-mismatch",
        ),
        pytest.param(
            "sub-001/eeg/sub-001_task-MMN_run-01_eeg.set",
            {"extension": ".fdt"},
            False,
            id="extension-mismatch",
        ),
        pytest.param(
            "sub-001/eeg/sub-001_task-MMN_run-01_eeg.set",
            {"scope": "derivatives"},
            False,
            id="scope-mismatch",
        ),
        pytest.param(
            "derivatives/eeglab/sub-001/eeg/sub-001_task-MMN_desc-clean_eeg.set",
            {"pipeline": "mne"},
            False,
            id="pipeline-mismatch",
        ),
    ],
)
def test_bids_query_matches_semantic_filters(path, filters, expected) -> None:
    """BIDS queries apply every provided semantic constraint."""
    assert (
        _path_matches(BidsPath.parse(path), build_bids_query(**filters)) is expected
    )


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        # ``from_filters(**{})`` defaults to ``scope=raw`` per the
        # post-nm000133 audit: subject/session queries previously
        # matched derivatives because no scope filter was applied.
        pytest.param({}, "scope=raw", id="default-scope-raw"),
        pytest.param(
            {"scope": ["raw", "derivatives"]},
            "scope=raw,derivatives",
            id="explicit-multi-scope",
        ),
        pytest.param(
            {
                "subject": ["sub-001", "002"],
                "datatype": "eeg",
                "suffix": "eeg",
                "extension": "set",
                "scope": "derivatives",
                "pipeline": "eeglab",
            },
            (
                "sub=001,002; datatype=eeg; suffix=eeg; extension=.set; "
                "scope=derivatives; pipeline=eeglab"
            ),
            id="all-filter-families",
        ),
    ],
)
def test_bids_query_describe(filters, expected) -> None:
    """Query descriptions remain compact and deterministic."""
    assert build_bids_query(**filters).describe() == expected


@pytest.mark.parametrize(
    ("filters", "expected_entities"),
    [
        pytest.param(
            {"run": 1},
            {"run": ("1",)},
            id="non-string-filter-value",
        ),
        pytest.param(
            {"entity": {"subject": "sub-001", "acquisition": "acq-high"}},
            {"sub": ("001",), "acq": ("high",)},
            id="entity-aliases",
        ),
    ],
)
def test_bids_query_normalizes_entity_values(filters, expected_entities) -> None:
    """BIDS entity values are normalized without losing aliases."""
    assert build_bids_query(**filters).entities == expected_entities


@pytest.mark.parametrize(
    "entity",
    [
        pytest.param(["task-MMN"], id="missing-equals"),
        pytest.param(["=MMN"], id="missing-key"),
        pytest.param(["task="], id="missing-value"),
    ],
)
def test_bids_query_rejects_bad_generic_entity(entity) -> None:
    """Generic entity filters must be explicit key=value pairs."""
    with pytest.raises(ValueError, match="key=value"):
        build_bids_query(entity=entity)


@pytest.mark.parametrize(
    "scope",
    [
        pytest.param("invalid", id="single-invalid"),
        pytest.param(["raw", "invalid"], id="mixed-invalid"),
    ],
)
def test_bids_query_rejects_unknown_scope(scope) -> None:
    """Scope filters must be one of the advertised BIDS dataset scopes."""
    with pytest.raises(ValueError, match="Unknown BIDS dataset scope"):
        build_bids_query(scope=scope)
