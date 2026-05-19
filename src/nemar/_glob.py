"""Glob-style matching for BIDS paths."""

from collections.abc import Iterable

from wcmatch import glob


def is_dotfile(path: str) -> bool:
    """Return whether any path segment is dot-prefixed."""
    return any(part.startswith(".") for part in path.split("/"))


def glob_filter(
    filenames: Iterable[str],
    patterns: Iterable[str],
) -> dict[str, set[str]]:
    """Return filenames matched by each pattern.

    Bare patterns match basenames at any depth. Every pattern is also tried as a
    directory prefix so ``sub-001`` matches all files below ``sub-001/``.
    """
    names = list(filenames)
    results: dict[str, set[str]] = {}

    for original in patterns:
        anchored = original.startswith("/")
        pattern = original.removeprefix("/")
        stripped = pattern.rstrip("/")
        bare = "/" not in stripped

        flags = glob.GLOBSTAR
        if bare and not anchored:
            flags |= glob.MATCHBASE

        matches = {str(path) for path in glob.globfilter(names, pattern, flags=flags)}
        if stripped:
            matches |= {
                str(path)
                for path in glob.globfilter(
                    names, stripped + "/**", flags=glob.GLOBSTAR
                )
            }
        results[original] = matches

    return results
