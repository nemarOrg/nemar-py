"""Write-to-staging-then-rename, so a partial file is never mistaken for a whole one.

A leaf module — no ``nemar`` imports beyond :mod:`nemar._constants`.

Backends used to stream straight into the final path. That is fine until a
transfer is interrupted: what remains on disk is a short file sitting at the
real name, and the only thing standing between it and downstream code is a
later size or hash check. Anything that reads the file before that check runs
sees truncated data with no signal that it is truncated.

Staging removes the ambiguity structurally. Bytes land in ``<path>.part``;
:func:`commit` fsyncs and then :func:`os.replace` moves it into place, which is
atomic within a filesystem. A crash therefore leaves either the previous file
or a ``.part`` file, never a half-written final one.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from nemar._constants import PARTIAL_SUFFIX

__all__ = ["direct", "staged", "staging_path"]


def staging_path(final: Path) -> Path:
    """Return the staging path used while ``final`` is being written.

    Deliberately beside the target rather than in a global temp directory:
    :func:`os.replace` is only atomic within a filesystem, so staging elsewhere
    would silently degrade into a copy across a mount boundary.
    """
    return final.with_name(final.name + PARTIAL_SUFFIX)


def _commit(staging: Path, final: Path) -> None:
    """Flush ``staging`` durably and move it onto ``final`` atomically.

    The fsync matters as much as the rename: ``os.replace`` orders the
    directory entry, not the file's data, so without it a power loss can leave
    a correctly-named file holding unwritten blocks.
    """
    fd = os.open(staging, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(staging, final)


@contextmanager
def direct(final: Path) -> Iterator[Path]:
    """Yield ``final`` itself, creating its parent — no staging, no rename.

    The counterpart to :func:`staged`, for files small enough that a rename per
    file costs more than the crash-safety is worth. On a networked filesystem a
    rename is a metadata round-trip that does not parallelise: measured ~14x
    fewer small files per second on Lustre. A truncated small file is still
    caught by the post-transfer size and hash gate; a truncated multi-GB object
    is the case worth paying for.
    """
    final.parent.mkdir(parents=True, exist_ok=True)
    yield final


@contextmanager
def staged(final: Path) -> Iterator[Path]:
    """Yield a staging path that is renamed onto ``final`` on clean exit.

    On an exception the staging file is removed. Keeping it would only pay off
    for a backend that resumes from those bytes, and none does today: the S3
    fetch reopens ``"wb"`` or re-truncates, so a leftover ``.part`` would be
    dead weight that the next run overwrites anyway. Cleaning up here keeps the
    invariant in one place rather than splitting it between this module and
    each caller's error handler.
    """
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = staging_path(final)
    try:
        yield staging
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    _commit(staging, final)
