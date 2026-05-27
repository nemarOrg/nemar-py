"""Tests for the consolidated ``_verification`` module."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from nemar._models import DatasetFile
from nemar._verification import (
    VerifyPolicy,
    VerifyResult,
    assert_all_present,
    check,
    detect_case_collisions,
    file_hash,
    git_blob_sha1,
    partition_pending,
)
from tests.fixtures.factories import make_dataset_file as _make_file


class TestCheck:
    """One row per :class:`VerifyResult` outcome."""

    def test_missing_when_file_absent(self, tmp_path: Path) -> None:
        file = _make_file("absent.bin", size=4)
        assert (
            check(file, tmp_path / file.path, VerifyPolicy()) is VerifyResult.MISSING
        )

    def test_ok_when_size_and_hash_match(self, tmp_path: Path) -> None:
        data = b"ok-blob"
        file = _make_file(
            "x.bin",
            size=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )
        (tmp_path / file.path).write_bytes(data)
        assert check(file, tmp_path / file.path, VerifyPolicy()) is VerifyResult.OK

    def test_size_mismatch(self, tmp_path: Path) -> None:
        file = _make_file("x.bin", size=10)
        (tmp_path / file.path).write_bytes(b"short")
        assert (
            check(file, tmp_path / file.path, VerifyPolicy())
            is VerifyResult.SIZE_MISMATCH
        )

    def test_hash_mismatch(self, tmp_path: Path) -> None:
        file = _make_file("x.bin", sha256="0" * 64)
        (tmp_path / file.path).write_bytes(b"different")
        assert (
            check(file, tmp_path / file.path, VerifyPolicy())
            is VerifyResult.HASH_MISMATCH
        )

    def test_error_sentinel_when_small_json_with_error_key(
        self, tmp_path: Path
    ) -> None:
        """S9: tiny ``{"error": ...}`` body with undeclared size is a sentinel."""
        file = _make_file("x.bin")  # size is None on purpose
        (tmp_path / file.path).write_bytes(
            json.dumps({"error": "not found"}).encode("utf-8")
        )
        assert (
            check(file, tmp_path / file.path, VerifyPolicy())
            is VerifyResult.ERROR_SENTINEL
        )

    def test_error_sentinel_skipped_when_manifest_declares_size(
        self, tmp_path: Path
    ) -> None:
        """A small file with a declared matching size is NOT a sentinel."""
        body = b'{"error": "small but legitimate"}'
        file = _make_file("x.bin", size=len(body))
        (tmp_path / file.path).write_bytes(body)
        # Size matches; not a sentinel. Hash check skipped via policy.
        assert (
            check(file, tmp_path / file.path, VerifyPolicy(verify_hash=False))
            is VerifyResult.OK
        )

    def test_verify_size_disabled(self, tmp_path: Path) -> None:
        """``verify_size=False`` skips size matching."""
        file = _make_file("x.bin", size=100)
        (tmp_path / file.path).write_bytes(b"tiny")
        assert (
            check(
                file,
                tmp_path / file.path,
                VerifyPolicy(verify_size=False, verify_hash=False),
            )
            is VerifyResult.OK
        )

    def test_verify_hash_disabled(self, tmp_path: Path) -> None:
        """``verify_hash=False`` skips hash matching."""
        file = _make_file("x.bin", sha256="0" * 64)
        (tmp_path / file.path).write_bytes(b"anything")
        assert (
            check(
                file,
                tmp_path / file.path,
                VerifyPolicy(verify_hash=False, verify_size=False),
            )
            is VerifyResult.OK
        )

    def test_git_sha1_hash_match_returns_ok(self, tmp_path: Path) -> None:
        """A :class:`DatasetFile` carrying ``git_sha1`` verifies against
        the local file's git blob SHA-1.

        Pins that the check() dispatcher picks the git path when
        :attr:`DatasetFile.git_sha1` is set and no stronger hash is
        present — the situation for every
        ``raw.githubusercontent.com``-served file in NEMAR's manifest.
        """
        content = b"hello\n"
        file = _make_file(
            "x.txt",
            size=len(content),
            git_sha1="ce013625030ba8dba906f756967f9e9ca394464a",
        )
        (tmp_path / file.path).write_bytes(content)
        assert check(file, tmp_path / file.path, VerifyPolicy()) is VerifyResult.OK

    def test_git_sha1_hash_mismatch_is_detected(self, tmp_path: Path) -> None:
        """Wrong git blob hash → :attr:`VerifyResult.HASH_MISMATCH`."""
        content = b"hello\n"
        file = _make_file(
            "x.txt",
            size=len(content),
            git_sha1="0" * 40,  # plausible shape, wrong value
        )
        (tmp_path / file.path).write_bytes(content)
        assert (
            check(file, tmp_path / file.path, VerifyPolicy())
            is VerifyResult.HASH_MISMATCH
        )


class TestS1UppercaseHash:
    """S1: an uppercase hex hash in the manifest must still match."""

    def test_uppercase_sha256_matches_lowercase_digest(self, tmp_path: Path) -> None:
        data = b"uppercase hash"
        upper = hashlib.sha256(data).hexdigest().upper()
        # Constructed directly because ``_coerce_hash`` would normalize.
        file = DatasetFile(
            path="x.bin",
            url="https://data.nemar.org/nm000132/v1.0.0/x.bin",
            sha256=upper,
        )
        (tmp_path / file.path).write_bytes(data)
        assert check(file, tmp_path / file.path, VerifyPolicy()) is VerifyResult.OK

    def test_uppercase_md5_matches_lowercase_digest(self, tmp_path: Path) -> None:
        data = b"md5 case test"
        upper = hashlib.md5(data).hexdigest().upper()
        file = DatasetFile(
            path="x.bin",
            url="https://data.nemar.org/nm000132/v1.0.0/x.bin",
            md5=upper,
        )
        (tmp_path / file.path).write_bytes(data)
        assert check(file, tmp_path / file.path, VerifyPolicy()) is VerifyResult.OK


class TestPartitionPending:
    def test_returns_only_non_ok_files(self, tmp_path: Path) -> None:
        complete = _make_file("complete.bin", size=2)
        (tmp_path / complete.path).write_bytes(b"ok")
        missing = _make_file("missing.bin", size=4)
        broken = _make_file("broken.bin", size=4)
        (tmp_path / broken.path).write_bytes(b"x")

        pending = partition_pending(
            [complete, missing, broken],
            target_dir=tmp_path,
            policy=VerifyPolicy(verify_hash=False),
        )

        assert [file.path for file in pending] == ["missing.bin", "broken.bin"]

    def test_pre_transfer_trusts_size_and_skips_hash(self, tmp_path: Path) -> None:
        """A right-size, wrong-content file is partitioned as complete pre-transfer.

        Contract: ``pre_transfer=True`` trusts the local size (and the
        error-sentinel probe) but does not re-hash already-present files.
        The post-transfer ``assert_all_present`` is the real gate that
        catches mismatched content.
        """
        data = b"actual content"
        # Manifest advertises a sha256 that does NOT match the local bytes.
        wrong_hash = "0" * 64
        file = _make_file("complete.bin", size=len(data), sha256=wrong_hash)
        (tmp_path / file.path).write_bytes(data)

        pending = partition_pending(
            [file],
            target_dir=tmp_path,
            policy=VerifyPolicy(),
            pre_transfer=True,
        )
        # Skipped from the pending list: the size matches.
        assert pending == []

        # The same call WITHOUT ``pre_transfer`` re-hashes and surfaces it.
        pending_full = partition_pending(
            [file],
            target_dir=tmp_path,
            policy=VerifyPolicy(),
        )
        assert pending_full == [file]


class TestAssertAllPresent:
    """One assertion per non-OK result variant."""

    def test_passes_when_all_ok(self, tmp_path: Path) -> None:
        file = _make_file("x.bin", size=2)
        (tmp_path / file.path).write_bytes(b"ok")
        # Returns None without raising.
        assert_all_present(
            [file],
            target_dir=tmp_path,
            policy=VerifyPolicy(verify_hash=False),
        )

    def test_raises_for_missing(self, tmp_path: Path) -> None:
        file = _make_file("missing.bin", size=4)
        with pytest.raises(RuntimeError, match="missing"):
            assert_all_present(
                [file], target_dir=tmp_path, policy=VerifyPolicy()
            )

    def test_raises_for_size_mismatch(self, tmp_path: Path) -> None:
        file = _make_file("x.bin", size=10)
        (tmp_path / file.path).write_bytes(b"short")
        with pytest.raises(RuntimeError, match="Size mismatch"):
            assert_all_present(
                [file], target_dir=tmp_path, policy=VerifyPolicy(verify_hash=False)
            )

    def test_raises_for_hash_mismatch(self, tmp_path: Path) -> None:
        file = _make_file("x.bin", sha256="0" * 64)
        (tmp_path / file.path).write_bytes(b"different")
        with pytest.raises(RuntimeError, match="Checksum mismatch"):
            assert_all_present(
                [file], target_dir=tmp_path, policy=VerifyPolicy(verify_size=False)
            )

    def test_raises_for_error_sentinel(self, tmp_path: Path) -> None:
        file = _make_file("x.bin")
        (tmp_path / file.path).write_bytes(
            json.dumps({"error": "object not available"}).encode("utf-8")
        )
        with pytest.raises(RuntimeError, match="error payload"):
            assert_all_present(
                [file], target_dir=tmp_path, policy=VerifyPolicy()
            )


class TestDetectCaseCollisions:
    """S4: case-insensitive filesystem collision detection."""

    def test_returns_empty_on_case_sensitive_fs(self, tmp_path: Path) -> None:
        # Probe first: skip on case-insensitive FS (Darwin default, Windows).
        # We treat the actual FS as truth rather than the platform.
        probe_a = tmp_path / "probe_a"
        probe_b = tmp_path / "PROBE_A"
        probe_a.write_text("a")
        try:
            if probe_b.exists() and os.path.samefile(probe_a, probe_b):
                pytest.skip("Filesystem is case-insensitive; covered by other test.")
        finally:
            probe_a.unlink(missing_ok=True)

        files = [
            _make_file("data/Sample.bin"),
            _make_file("data/sample.bin"),
        ]
        collisions = detect_case_collisions(files, target_dir=tmp_path)
        assert collisions == []

    def test_returns_pairs_on_case_insensitive_fs(self, tmp_path: Path) -> None:
        probe_a = tmp_path / "probe_a"
        probe_b = tmp_path / "PROBE_A"
        probe_a.write_text("a")
        try:
            is_ci = probe_b.exists() and os.path.samefile(probe_a, probe_b)
        finally:
            probe_a.unlink(missing_ok=True)
        if not is_ci:
            pytest.skip("Filesystem is case-sensitive; covered by the other test.")

        files = [
            _make_file("data/Sample.bin"),
            _make_file("data/sample.bin"),
        ]
        collisions = detect_case_collisions(files, target_dir=tmp_path)
        assert collisions == [("data/Sample.bin", "data/sample.bin")]

    def test_no_collision_when_paths_truly_unique(self, tmp_path: Path) -> None:
        files = [
            _make_file("a.bin"),
            _make_file("b.bin"),
            _make_file("c/d.bin"),
        ]
        assert detect_case_collisions(files, target_dir=tmp_path) == []


class TestFileHash:
    """The exported file-hash primitive."""

    def test_sha256(self, tmp_path: Path) -> None:
        data = b"some content"
        path = tmp_path / "x.bin"
        path.write_bytes(data)
        assert file_hash(path, "sha256") == hashlib.sha256(data).hexdigest()

    def test_md5(self, tmp_path: Path) -> None:
        data = b"some content"
        path = tmp_path / "x.bin"
        path.write_bytes(data)
        assert file_hash(path, "md5") == hashlib.md5(data).hexdigest()


class TestGitBlobSha1:
    """The git-blob SHA-1 helper used for ``raw.githubusercontent.com``
    files in NEMAR's manifest.
    """

    def test_known_blob_matches_git_format(self, tmp_path: Path) -> None:
        """Empty file → ``e69de29bb2d1d6434b8b29ae775ad8c2e48c5391``.

        This is the canonical empty-blob SHA-1 in git
        (``git hash-object /dev/null``). Hard-coded so any drift in the
        helper is loud.
        """
        path = tmp_path / "empty.bin"
        path.write_bytes(b"")
        assert git_blob_sha1(path) == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"

    def test_known_short_blob(self, tmp_path: Path) -> None:
        """``b"hello\\n"`` → ``ce013625030ba8dba906f756967f9e9ca394464a``.

        Another canonical case: ``git hash-object`` of a tiny file. Pins
        the prefix format ``"blob <size>\\0<content>"`` so silently
        switching to plain ``sha1(content)`` (a common bug) would
        break this test immediately.
        """
        path = tmp_path / "hello.txt"
        path.write_bytes(b"hello\n")
        assert git_blob_sha1(path) == "ce013625030ba8dba906f756967f9e9ca394464a"

    def test_differs_from_plain_sha1(self, tmp_path: Path) -> None:
        """Plain SHA-1 over the content alone must NOT match the git blob
        hash — that's the whole point of the helper.
        """
        data = b"x" * 64
        path = tmp_path / "x.bin"
        path.write_bytes(data)
        assert git_blob_sha1(path) != hashlib.sha1(data).hexdigest()  # noqa: S324
