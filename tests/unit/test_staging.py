"""Tests for the staging-file helpers.

The point of staging is that an interrupted transfer can never leave a
truncated file under the final name, so these pin both halves: the final name
does not appear until the bytes are complete, and a failure leaves no file
under the real name.

The module offers three disciplines and they differ on purpose — ``staged``
discards a failed partial, ``staging_path`` + ``commit`` keeps it for a
resuming backend, and ``direct`` skips staging entirely — so the tests are
parametrized over them wherever the expected behaviour is shared.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest

from nemar import _staging
from nemar._constants import PARTIAL_SUFFIX
from nemar._staging import commit, direct, staged, staging_path

#: Both write-through disciplines, for the behaviour they share.
DESTINATIONS = [
    pytest.param(staged, id="staged"),
    pytest.param(direct, id="direct"),
]


@pytest.mark.parametrize(
    ("final_name", "expected_name"),
    [
        pytest.param("sub-01_eeg.bdf", f"sub-01_eeg.bdf{PARTIAL_SUFFIX}", id="suffix"),
        pytest.param("archive.tar.gz", f"archive.tar.gz{PARTIAL_SUFFIX}", id="double"),
        pytest.param("README", f"README{PARTIAL_SUFFIX}", id="no-extension"),
    ],
)
def test_staging_path_appends_the_partial_suffix(
    tmp_path: Path, final_name: str, expected_name: str
) -> None:
    """The staging name is derived, not invented, so callers can find it.

    Appended rather than replacing the extension, so two files differing only
    by extension cannot collide on one staging path.
    """
    assert staging_path(tmp_path / final_name) == tmp_path / expected_name


def test_staging_path_keeps_the_file_in_its_own_directory(tmp_path: Path) -> None:
    """Staging beside the target keeps the rename on one filesystem.

    ``os.replace`` is only atomic within a filesystem, so a staging file in a
    global temp dir would silently degrade into a copy across a mount boundary.
    """
    final = tmp_path / "nested" / "deep" / "file.set"

    assert staging_path(final).parent == final.parent


@pytest.mark.parametrize(
    "preexisting",
    [pytest.param(None, id="new-file"), pytest.param(b"stale", id="overwrite")],
)
def test_commit_moves_bytes_onto_the_final_path(
    tmp_path: Path, preexisting: bytes | None
) -> None:
    """A re-download must overwrite rather than fail on an existing target."""
    final = tmp_path / "out.bin"
    if preexisting is not None:
        final.write_bytes(preexisting)
    staging = staging_path(final)
    staging.write_bytes(b"payload")

    commit(staging, final)

    assert final.read_bytes() == b"payload"
    assert not staging.exists()


def test_commit_fsyncs_before_renaming(tmp_path: Path, monkeypatch) -> None:
    """The durability half matters as much as the atomic half.

    ``os.replace`` orders the directory entry, not the file's data, so without
    an fsync a power loss can leave a correctly-named file holding unwritten
    blocks.
    """
    order: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: order.append("fsync") or real_fsync(fd))
    monkeypatch.setattr(
        os, "replace", lambda a, b: order.append("replace") or real_replace(a, b)
    )
    final = tmp_path / "out.bin"
    staging = staging_path(final)
    staging.write_bytes(b"payload")

    commit(staging, final)

    assert order == ["fsync", "replace"]


def test_commit_fsync_uses_a_writable_handle() -> None:
    """Windows' fsync maps to FlushFileBuffers, which needs write access.

    An O_RDONLY descriptor fails there, which would break every staged (large)
    fetch on Windows while passing on the Linux-only CI matrix.
    """
    assert "O_RDONLY" not in inspect.getsource(_staging.commit)


@pytest.mark.parametrize("destination", DESTINATIONS)
def test_destination_creates_parent_directories(tmp_path: Path, destination) -> None:
    """Callers should not have to mkdir before writing a nested path."""
    final = tmp_path / "sub-01" / "eeg" / "out.bin"

    with destination(final) as target:
        target.write_bytes(b"data")

    assert final.read_bytes() == b"data"


@pytest.mark.parametrize("destination", DESTINATIONS)
def test_destination_leaves_no_partial_behind_on_success(
    tmp_path: Path, destination
) -> None:
    """Either discipline ends with the bytes at the real name and nothing else."""
    final = tmp_path / "out.bin"

    with destination(final) as target:
        target.write_bytes(b"data")

    assert final.read_bytes() == b"data"
    assert list(tmp_path.rglob(f"*{PARTIAL_SUFFIX}")) == []


def test_direct_yields_the_final_path_itself(tmp_path: Path) -> None:
    """``direct`` is the no-rename discipline: it writes where asked."""
    final = tmp_path / "small.tsv"

    with direct(final) as target:
        assert target == final


def test_staged_never_exposes_the_final_name_mid_write(tmp_path: Path) -> None:
    """The whole point: partial bytes must not sit under the real name."""
    final = tmp_path / "out.bin"

    with staged(final) as staging:
        staging.write_bytes(b"half")
        assert not final.exists()

    assert final.exists()


@pytest.mark.parametrize(
    "wrote_bytes",
    [pytest.param(True, id="after-first-byte"), pytest.param(False, id="before-write")],
)
def test_staged_cleans_up_when_the_body_raises(
    tmp_path: Path, wrote_bytes: bool
) -> None:
    """A failure leaves neither a final file nor a stray .part behind.

    Nothing resumes from an S3 ``.part`` -- the fetch reopens ``"wb"`` or
    re-truncates -- so keeping one would be dead weight the next run overwrites.
    The ``before-write`` case also pins that cleanup cannot mask the original
    error by raising over it.
    """
    final = tmp_path / "out.bin"

    with pytest.raises(RuntimeError, match="boom"):
        with staged(final) as staging:
            if wrote_bytes:
                staging.write_bytes(b"partial")
            raise RuntimeError("boom")

    assert not final.exists()
    assert list(tmp_path.glob(f"*{PARTIAL_SUFFIX}")) == []


def test_staged_removes_the_partial_when_commit_itself_fails(
    tmp_path: Path, monkeypatch
) -> None:
    """A failing fsync/rename must not orphan the .part.

    ``commit`` used to run outside the guard, so an EIO/ENOSPC fsync or a
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
