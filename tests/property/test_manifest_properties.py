"""Property tests for parse_version_manifest."""

from __future__ import annotations

import string
from pathlib import PurePosixPath

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from nemar._models import parse_version_manifest

path_chars = string.ascii_lowercase + string.digits + "-_/"
relative_paths = st.text(alphabet=path_chars, min_size=1, max_size=60).filter(
    lambda p: ".." not in p.split("/")
    and not p.startswith("/")
    and "\x00" not in p
    and p.strip("/")
)


@given(paths=st.lists(relative_paths, min_size=1, max_size=8, unique=True))
def test_parse_list_manifest_round_trip_up_to_canonicalization(
    paths: list[str],
) -> None:
    """Parsing a list-shaped manifest canonicalizes paths via PurePosixPath.

    The contract: ``parse_version_manifest`` returns paths in the same order
    as the input, but normalized (trailing slashes stripped, redundant
    separators collapsed). Two raw inputs that canonicalize to the same path
    collide on the parser's duplicate check; ``assume`` filters those cases
    so we only exercise inputs whose canonical forms remain distinct.
    """
    canonical = [PurePosixPath(path).as_posix() for path in paths]
    assume(len(set(canonical)) == len(canonical))

    payload = [{"path": path, "size": idx} for idx, path in enumerate(paths)]
    files = parse_version_manifest(
        payload,
        manifest_url="https://localhost/nm000132/v1/manifest.json",
        data_url="https://localhost/",
    )

    assert [f.path for f in files] == canonical


def test_parse_list_manifest_canonicalizes_trailing_slash() -> None:
    """Concrete witness of the canonicalization contract.

    Locks in the case ``'a/' -> 'a'`` that the property strategy originally
    exposed, so a future change to ``_validate_relative_path`` that drops
    canonicalization fails loudly.
    """
    files = parse_version_manifest(
        [{"path": "a/", "size": 1}],
        manifest_url="https://localhost/nm000132/v1/manifest.json",
        data_url="https://localhost/",
    )
    assert [f.path for f in files] == ["a"]


def test_parse_list_manifest_canonicalizes_redundant_separators() -> None:
    """PurePosixPath collapses ``'a//b'`` to ``'a/b'``."""
    files = parse_version_manifest(
        [{"path": "a//b", "size": 1}],
        manifest_url="https://localhost/nm000132/v1/manifest.json",
        data_url="https://localhost/",
    )
    assert [f.path for f in files] == ["a/b"]


@given(
    path=relative_paths,
    raw_hash=st.text(alphabet=string.hexdigits.lower(), min_size=8, max_size=64),
)
def test_parse_manifest_coerces_quoted_hash(path: str, raw_hash: str) -> None:
    quoted = '"' + raw_hash + '"'
    files = parse_version_manifest(
        [{"path": path, "sha256": quoted}],
        manifest_url="https://localhost/nm000132/v1/manifest.json",
        data_url="https://localhost/",
    )
    assert files[0].sha256 == raw_hash


def test_parse_manifest_rejects_duplicate_paths_property() -> None:
    payload = [{"path": "a.txt"}, {"path": "a.txt"}]
    with pytest.raises(RuntimeError, match="duplicate"):
        parse_version_manifest(
            payload,
            manifest_url="https://localhost/nm000132/v1/manifest.json",
            data_url="https://localhost/",
        )
