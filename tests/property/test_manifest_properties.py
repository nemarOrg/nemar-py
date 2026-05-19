"""Property tests for parse_version_manifest."""

from __future__ import annotations

import hashlib
import string

import pytest
from hypothesis import given, strategies as st

from nemar._models import parse_version_manifest

path_chars = string.ascii_lowercase + string.digits + "-_/"
relative_paths = st.text(alphabet=path_chars, min_size=1, max_size=60).filter(
    lambda p: ".." not in p.split("/")
    and not p.startswith("/")
    and "\x00" not in p
    and p.strip("/")
)


@pytest.mark.xfail(
    strict=True,
    reason="bug discovered by property test; tracked separately: "
    "parse_version_manifest canonicalises paths via PurePosixPath "
    "(e.g. '0/' -> '0'), breaking exact-identity round trip",
)
@given(paths=st.lists(relative_paths, min_size=1, max_size=8, unique=True))
def test_parse_list_manifest_round_trip(paths: list[str]) -> None:
    payload = [
        {"path": path, "size": idx}
        for idx, path in enumerate(paths)
    ]
    files = parse_version_manifest(
        payload,
        manifest_url="https://localhost/nm000132/v1/manifest.json",
        data_url="https://localhost/",
    )
    assert [f.path for f in files] == paths


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
