"""Tests for the staging-file helpers.

The point of staging is that an interrupted transfer can never leave a
truncated file under the final name, so these pin both halves: the final name
does not appear until the bytes are complete, and a failure leaves neither a
final file nor a stray ``.part`` behind.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from nemar._constants import PARTIAL_SUFFIX
from nemar._staging import _commit, direct, staged, staging_path


def test_staging_path_appends_the_partial_suffix(tmp_path: Path) -> None:
    """The staging name is derived, not invented, so callers can find it."""
    final = tmp_path / "sub-01_eeg.bdf"

    assert staging_path(final) == tmp_path / f"sub-01_eeg.bdf{PARTIAL_SUFFIX}"


def test_staging_path_keeps_the_file_in_its_own_directory(tmp_path: Path) -> None:
    """Staging beside the target keeps the rename on one filesystem.

    ``os.replace`` is only atomic within a filesystem, so a staging file in a
    global temp dir would silently become a copy across a mount boundary.
    """
    final = tmp_path / "nested" / "deep" / "file.set"

    assert staging_path(final).parent == final.parent


def test_commit_moves_bytes_onto_the_final_path(tmp_path: Path) -> None:
    final = tmp_path / "out.bin"
    staging = staging_path(final)
    staging.write_bytes(b"payload")

    _commit(staging, final)

    assert final.read_bytes() == b"payload"
    assert not staging.exists()


def test_commit_replaces_an_existing_file(tmp_path: Path) -> None:
    """A re-download must overwrite, not fail on an existing target."""
    final = tmp_path / "out.bin"
    final.write_bytes(b"stale")
    staging = staging_path(final)
    staging.write_bytes(b"fresh")

    _commit(staging, final)

    assert final.read_bytes() == b"fresh"


def test_staged_creates_parent_directories(tmp_path: Path) -> None:
    """Callers should not have to mkdir before writing a nested path."""
    final = tmp_path / "sub-01" / "eeg" / "out.bin"

    with staged(final) as staging:
        staging.write_bytes(b"data")

    assert final.read_bytes() == b"data"


def test_staged_leaves_no_partial_behind_on_success(tmp_path: Path) -> None:
    final = tmp_path / "out.bin"

    with staged(final) as staging:
        staging.write_bytes(b"data")

    assert list(tmp_path.glob(f"*{PARTIAL_SUFFIX}")) == []


def test_staged_never_exposes_the_final_name_mid_write(tmp_path: Path) -> None:
    """The whole point: partial bytes must not sit under the real name."""
    final = tmp_path / "out.bin"

    with staged(final) as staging:
        staging.write_bytes(b"half")
        assert not final.exists()

    assert final.exists()


def test_staged_cleans_up_when_the_body_raises(tmp_path: Path) -> None:
    """A failure leaves neither a final file nor a stray .part behind.

    Nothing resumes from a ``.part`` today, so keeping one would just be dead
    weight the next run overwrites.
    """
    final = tmp_path / "out.bin"

    with pytest.raises(RuntimeError, match="boom"):
        with staged(final) as staging:
            staging.write_bytes(b"partial")
            raise RuntimeError("boom")

    assert not final.exists()
    assert list(tmp_path.glob(f"*{PARTIAL_SUFFIX}")) == []


def test_staged_cleanup_tolerates_a_body_that_never_wrote(tmp_path: Path) -> None:
    """Failing before the first byte must not mask the original error."""
    final = tmp_path / "out.bin"

    with pytest.raises(RuntimeError, match="early"):
        with staged(final):
            raise RuntimeError("early")

    assert not final.exists()


def test_commit_fsyncs_before_renaming(tmp_path: Path, monkeypatch) -> None:
    """The durability half matters as much as the atomic half.

    ``os.replace`` orders the directory entry, not the file's data, so without
    an fsync a power loss can leave a correctly-named file holding unwritten
    blocks.
    """
    order: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(
        os, "fsync", lambda fd: order.append("fsync") or real_fsync(fd)
    )
    monkeypatch.setattr(
        os, "replace", lambda a, b: order.append("replace") or real_replace(a, b)
    )
    final = tmp_path / "out.bin"
    staging = staging_path(final)
    staging.write_bytes(b"payload")

    _commit(staging, final)

    assert order == ["fsync", "replace"]


def test_direct_yields_the_final_path_itself(tmp_path: Path) -> None:
    """The no-staging counterpart writes where the caller asked, no rename."""
    final = tmp_path / "sub-01" / "small.tsv"

    with direct(final) as destination:
        assert destination == final
        destination.write_bytes(b"data")

    assert final.read_bytes() == b"data"
    assert list(tmp_path.rglob(f"*{PARTIAL_SUFFIX}")) == []


def test_direct_creates_parent_directories(tmp_path: Path) -> None:
    final = tmp_path / "deep" / "nested" / "small.tsv"

    with direct(final) as destination:
        destination.write_bytes(b"x")

    assert final.exists()


def test_staged_removes_the_partial_when_commit_itself_fails(
    tmp_path: Path, monkeypatch
) -> None:
    """A failing fsync/rename must not orphan the .part.

    ``_commit`` used to run outside the guard, so an EIO/ENOSPC fsync or a
    rename onto a directory left the staging file behind -- the one outcome
    this module promises cannot happen.
    """
    final = tmp_path / "out.bin"
    monkeypatch.setattr(
        os, "fsync", lambda fd: (_ for _ in ()).throw(OSError(5, "I/O error"))
    )

    with pytest.raises(OSError, match="I/O error"):
        with staged(final) as staging:
            staging.write_bytes(b"payload")

    assert not final.exists()
    assert list(tmp_path.glob(f"*{PARTIAL_SUFFIX}")) == []


def test_commit_fsync_uses_a_writable_handle(tmp_path: Path) -> None:
    """Windows' fsync maps to FlushFileBuffers, which needs write access.

    An O_RDONLY descriptor fails there, which would break every staged (large)
    fetch on Windows while passing on the Linux-only CI matrix.
    """
    import inspect

    from nemar import _staging

    source = inspect.getsource(_staging._commit)
    assert "O_RDONLY" not in source
    # And it still works, without truncating what was written.
    final = tmp_path / "out.bin"
    staging = staging_path(final)
    staging.write_bytes(b"payload")
    _commit(staging, final)
    assert final.read_bytes() == b"payload"
