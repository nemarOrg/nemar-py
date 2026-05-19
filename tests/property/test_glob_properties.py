"""Property tests for glob_filter semantics."""

from __future__ import annotations

import string

from hypothesis import given, strategies as st

from nemar._glob import glob_filter, is_dotfile

name_chars = string.ascii_lowercase + string.digits + "-_"
filenames = st.lists(
    st.text(alphabet=name_chars + "/", min_size=1, max_size=40),
    min_size=1,
    max_size=20,
)


@given(filenames=filenames)
def test_glob_returns_dict_for_each_pattern(filenames: list[str]) -> None:
    patterns = ["*", "sub-*", "*.json"]
    result = glob_filter(filenames, patterns)
    assert set(result.keys()) == set(patterns)


@given(filenames=filenames)
def test_bare_pattern_matches_basename_anywhere(filenames: list[str]) -> None:
    safe = [name for name in filenames if name and not name.startswith("/")]
    candidates = safe + ["a/b/c/match.tsv", "deep/nested/match.tsv"]
    matches = glob_filter(candidates, ["match.tsv"])["match.tsv"]
    if any(name.endswith("match.tsv") for name in candidates):
        assert matches


@given(
    name=st.text(alphabet=string.ascii_lowercase + ".", min_size=1, max_size=8),
)
def test_is_dotfile_consistent_with_split(name: str) -> None:
    segments = name.split("/")
    expected = any(seg.startswith(".") for seg in segments)
    assert is_dotfile(name) == expected
