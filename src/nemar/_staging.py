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

__all__ = ["commit", "discard", "staged", "staging_path"]


def staging_path(final: Path) -> Path:
    """Return the staging path used while ``final`` is being written."""
    return final.with_name(final.name + PARTIAL_SUFFIX)


def commit(staging: Path, final: Path) -> None:
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


def discard(staging: Path) -> None:
    """Remove a staging file, ignoring the case where it never appeared."""
    staging.unlink(missing_ok=True)


@contextmanager
def staged(final: Path) -> Iterator[Path]:
    """Yield a staging path that is renamed onto ``final`` on clean exit.

    On an exception the staging file is left in place deliberately: a resumable
    backend can pick up from those bytes on the next run, and it cannot be
    confused for finished output because it does not carry the final name.
    """
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = staging_path(final)
    yield staging
    commit(staging, final)
