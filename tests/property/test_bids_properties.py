"""Property tests for BIDS path parsing and query matching."""

from __future__ import annotations

import string

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nemar._models import DATASET_SCOPES, BidsPath
from nemar._request import build_bids_query

label_chars = string.ascii_lowercase + string.digits
labels = st.text(alphabet=label_chars, min_size=1, max_size=8)
entities = st.sampled_from(["sub", "ses", "task", "run", "acq"])


@st.composite
def bids_filenames(draw) -> str:
    sub = draw(labels)
    has_ses = draw(st.booleans())
    has_task = draw(st.booleans())
    tokens = [f"sub-{sub}"]
    if has_ses:
        tokens.append(f"ses-{draw(labels)}")
    if has_task:
        tokens.append(f"task-{draw(labels)}")
    suffix = draw(st.sampled_from(["eeg", "events", "channels"]))
    ext = draw(st.sampled_from([".tsv", ".json", ".set", ".vhdr"]))
    return "_".join(tokens + [suffix]) + ext


@settings(suppress_health_check=[HealthCheck.too_slow])
@given(sub=labels, filename=bids_filenames())
def test_bids_path_parse_recovers_subject_entity(sub: str, filename: str) -> None:
    path = f"sub-{sub}/eeg/sub-{sub}_" + filename.split("_", 1)[1]
    parsed = BidsPath.parse(path)
    assert parsed.entities.get("sub") == sub


@given(scope=st.sampled_from(sorted(DATASET_SCOPES)))
def test_bids_query_scope_round_trip(scope: str) -> None:
    query = build_bids_query(scope=scope)
    assert query.scopes == (scope,)


@given(
    extensions=st.lists(
        st.sampled_from(["tsv", ".tsv", "TSV", ".TSV"]), min_size=1, max_size=4
    )
)
def test_bids_query_extension_normalization(extensions: list[str]) -> None:
    query = build_bids_query(extension=extensions)
    for ext in query.extensions:
        assert ext.startswith(".")
        assert ext == ext.lower()


@given(subject=labels)
def test_bids_query_describe_is_deterministic(subject: str) -> None:
    a = build_bids_query(subject=subject).describe()
    b = build_bids_query(subject=subject).describe()
    assert a == b
