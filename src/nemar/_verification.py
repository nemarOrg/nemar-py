"""Local-file verification against a NEMAR manifest entry.

The downloader has one repeated question: "does this local file satisfy
its manifest entry?" The historical answer lived at four call sites with
slightly different semantics (predicate vs. raise, pre-transfer filter vs.
post-transfer assertion). This module consolidates that question into one
small value-typed surface:

* :class:`VerifyResult` enumerates every outcome the caller may need to
  branch on (OK, MISSING, SIZE_MISMATCH, HASH_MISMATCH, ERROR_SENTINEL).
* :class:`VerifyPolicy` carries the (verify-size, verify-hash,
  error-sentinel size) knobs without forcing every caller to thread
  three keyword arguments through their signature.
* :func:`check` is the pure predicate.
* :func:`partition_pending` returns the files that still need transfer.
* :func:`assert_all_present` is the post-transfer assertion that turns a
  non-``OK`` result into a clear ``RuntimeError``.
* :func:`detect_case_collisions` is the pre-transfer guard against
  case-insensitive-filesystem clashes that would silently overwrite data.

The "predicate, then assert" split exists because the predicate is used
both before transfer (skip already-complete files) and inside the
post-transfer hot path (single-source-of-truth verifier). Raising from
the predicate would force every pre-transfer caller to wrap a try/except
just to detect a planned-for outcome.
"""

from __future__ import annotations

import enum
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from nemar._errors import VerificationError
from nemar._models import DatasetFile


class VerifyResult(enum.Enum):
    """Outcome of verifying one local file against its manifest entry."""

    OK = "ok"
    MISSING = "missing"
    SIZE_MISMATCH = "size_mismatch"
    HASH_MISMATCH = "hash_mismatch"
    ERROR_SENTINEL = "error_sentinel"


@dataclass(frozen=True)
class VerifyPolicy:
    """How a local file is verified against its manifest entry.

    ``error_sentinel_max_bytes`` bounds how many bytes the verifier will
    read off disk when probing for a JSON error payload. Files
    larger than this are never considered sentinels.
    """

    verify_size: bool = True
    verify_hash: bool = True
    error_sentinel_max_bytes: int = 200


def check(file: DatasetFile, path: Path, policy: VerifyPolicy) -> VerifyResult:
    """Return the verification result for one local file.

    The function is total: it returns a :class:`VerifyResult` for every
    input rather than raising. Callers that want to fail loudly should
    use :func:`assert_all_present` (after transfer) or branch on the
    result themselves.
    """
    if not path.exists() or not path.is_file():
        return VerifyResult.MISSING

    local_size = path.stat().st_size

    # A tiny, undeclared-size file that parses as ``{"error": ...}``
    # is the NEMAR endpoint's way of saying "I do not have this object".
    # We only probe when the manifest itself did not declare a size, so
    # legitimate small files (empty sidecars, tiny TSVs) are not
    # mis-classified.
    if (
        file.size is None
        and 0 < local_size <= policy.error_sentinel_max_bytes
        and _looks_like_error_sentinel(path)
    ):
        return VerifyResult.ERROR_SENTINEL

    if policy.verify_size and file.size is not None and local_size != file.size:
        return VerifyResult.SIZE_MISMATCH

    if policy.verify_hash and not _hash_matches(file, path):
        return VerifyResult.HASH_MISMATCH

    return VerifyResult.OK


def partition_pending(
    files: Sequence[DatasetFile],
    *,
    target_dir: Path,
    policy: VerifyPolicy,
    pre_transfer: bool = False,
) -> list[DatasetFile]:
    """Return the subset of ``files`` whose local result is not ``OK``.

    This is the pre-transfer filter that decides what still needs to be
    fetched. Anything other than :attr:`VerifyResult.OK` (including a
    detected error-sentinel from a previous failed run) is included so
    the transfer backend gets a chance to replace it.

    When ``pre_transfer=True`` the hash check is suppressed and only the
    size (and error-sentinel) checks decide membership. This is the
    intentional contract for the *pre-transfer* call site: trusting size
    on already-present files lets a 100%-complete re-run skip an O(dataset)
    rehash of every local file before the network does anything. The
    post-transfer :func:`assert_all_present` re-checks the hash and is the
    real gate -- a file with the right size but wrong content is caught
    there, not here.
    """
    effective_policy = (
        replace(policy, verify_hash=False) if pre_transfer else policy
    )
    return [
        file
        for file in files
        if check(file, target_dir / file.path, effective_policy) is not VerifyResult.OK
    ]


def assert_all_present(
    files: Sequence[DatasetFile],
    *,
    target_dir: Path,
    policy: VerifyPolicy,
) -> None:
    """Raise ``RuntimeError`` if any local file is not ``OK``.

    The message names the offending file and the reason in plain
    language so the failure mode is obvious from the traceback.
    """
    for file in files:
        path = target_dir / file.path
        result = check(file, path, policy)
        if result is VerifyResult.OK:
            continue
        raise VerificationError(_describe_failure(file, path, result))


def detect_case_collisions(
    files: Sequence[DatasetFile],
    *,
    target_dir: Path,
) -> list[tuple[str, str]]:
    """Return manifest path pairs that would collide on a case-insensitive FS.

    The check is a no-op on case-sensitive filesystems (Linux ext4, NFS
    in default mode). On HFS+, APFS in case-insensitive mode, and NTFS,
    two manifest entries that differ only by case map to the same target
    file -- the second download silently overwrites the first.

    We probe the actual target directory rather than guessing from the
    platform: APFS can be either case-sensitive or case-insensitive, and
    network mounts can disagree with their host.
    """
    if not _filesystem_is_case_insensitive(target_dir):
        return []

    seen: dict[str, str] = {}
    collisions: list[tuple[str, str]] = []
    for file in files:
        key = file.path.lower()
        if key in seen and seen[key] != file.path:
            collisions.append((seen[key], file.path))
        else:
            seen.setdefault(key, file.path)
    return collisions


def file_hash(path: Path, algorithm: Literal["sha256", "md5"]) -> str:
    """Return the hex digest of ``path`` under ``algorithm``.

    Exposed at module level so the aria2 path (which delegates checksum
    enforcement to aria2c) can still share the same primitive when it
    needs a Python-side digest.
    """
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_matches(file: DatasetFile, path: Path) -> bool:
    """Return ``True`` when the strongest available manifest hash matches.

    Hashes are compared case-insensitively because manifests sometimes
    advertise uppercase hex. ``DatasetFile.sha256`` / ``.md5`` are
    already normalized by :func:`_models._coerce_hash`, so this guard is
    a belt-and-braces against historical manifests that bypass the
    parser.
    """
    if file.sha256 is not None:
        return file_hash(path, "sha256").lower() == file.sha256.lower()
    if file.md5 is not None:
        return file_hash(path, "md5").lower() == file.md5.lower()
    return True


def _looks_like_error_sentinel(path: Path) -> bool:
    """Return ``True`` if ``path`` is a tiny JSON object with an ``error`` key."""
    try:
        with path.open("rb") as handle:
            head = handle.read(512)
    except OSError:
        return False
    try:
        payload = json.loads(head)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    return isinstance(payload, dict) and "error" in payload


def _describe_failure(file: DatasetFile, path: Path, result: VerifyResult) -> str:
    if result is VerifyResult.MISSING:
        return f"Expected downloaded file is missing: {file.path}"
    if result is VerifyResult.SIZE_MISMATCH:
        actual = path.stat().st_size if path.exists() else 0
        return (
            f"Size mismatch for {file.path}: expected {file.size} bytes, "
            f"got {actual}."
        )
    if result is VerifyResult.HASH_MISMATCH:
        return f"Checksum mismatch for {file.path}."
    if result is VerifyResult.ERROR_SENTINEL:
        snippet = _read_error_snippet(path)
        return (
            f"NEMAR returned an error payload instead of {file.path}: {snippet}"
        )
    # Unreachable: VerifyResult.OK is filtered out by the caller. This
    # exists only to satisfy the type checker on enum exhaustiveness.
    return (  # pragma: no cover
        f"Unknown verification result for {file.path}: {result!r}"
    )


def _read_error_snippet(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            head = handle.read(512)
        payload = json.loads(head)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return "<unreadable>"
    if isinstance(payload, dict):
        message = payload.get("error") or payload.get("message") or ""
        return str(message)[:200]
    return "<unreadable>"


def _filesystem_is_case_insensitive(target_dir: Path) -> bool:
    """Probe whether ``target_dir`` is on a case-insensitive filesystem.

    Tries two strategies, in order:

    1. If ``target_dir`` already exists, create a temp file and check
       whether an uppercase variant of its name resolves to the same
       inode via :func:`os.path.samefile`.
    2. Otherwise, fall back to ``os.path.normcase`` -- on Windows this
       lowercases the path; on POSIX it returns the input unchanged.
       The probe via ``samefile`` is preferred because APFS can be
       either case-sensitive or case-insensitive on the same OS.
    """
    if not target_dir.exists():
        # If the target does not exist yet, fall back to a heuristic
        # based on ``os.name``. ``nt`` (Windows) and ``Darwin`` (HFS+
        # default) are case-insensitive in practice. We treat anything
        # we cannot probe as case-sensitive so we never falsely flag
        # collisions on POSIX -- a false positive here would block
        # legitimate downloads.
        return os.name == "nt"

    try:
        with tempfile.NamedTemporaryFile(
            "w", prefix="nemar_case_", suffix=".probe", dir=target_dir, delete=False
        ) as fh:
            probe = Path(fh.name)
        try:
            upper = probe.with_name(probe.name.upper())
            if upper == probe:
                # Already uppercase suffix -- compare the lowercase variant instead.
                upper = probe.with_name(probe.name.lower())
            if upper == probe:
                return False
            return upper.exists() and os.path.samefile(probe, upper)
        finally:
            probe.unlink(missing_ok=True)
    except OSError:
        return False
