"""NEMAR BIDS file selection: a function over a manifest.

This module owns the composition of three operations that previously lived
in ``_download._select_bids_files``:

1. Apply a :class:`~nemar._models.BidsQuery` to manifest paths.
2. Narrow the result with include glob patterns.
3. Subtract exclude glob patterns and force-keep essential BIDS root files.

The public surface is the function :func:`select_files`, which returns
a :class:`SelectionResult` carrying the selected files plus the small
piece of diagnostic state the orchestrator actually consumes
(unmatched include patterns, surfaced through
:func:`raise_if_unmatched_includes`). The previous ``SelectionPlan``
class dressed this linear flow as an entity with behaviour; the
function + frozen result type is more honest about what is happening.

The BIDS *value types* (:class:`~nemar._models.BidsPath`,
:class:`~nemar._models.BidsQuery`) live in :mod:`nemar._models` with the
other dumb value types. *Matching* a parsed path against a query
(:func:`_path_matches`) lives here, next to :func:`select_files` which
applies it. *Building* a query from raw kwargs is request normalization,
so it lives in :func:`nemar._request.build_bids_query` next to its sole
caller.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from difflib import get_close_matches

from wcmatch import glob

from nemar._models import BidsPath, BidsQuery, DatasetFile
from nemar.errors import SelectionError

ESSENTIAL_BIDS_FILES = frozenset(
    {
        "dataset_description.json",
        "participants.tsv",
        "participants.json",
        "README",
        "README.md",
        "CHANGES",
        "LICENSE",
    }
)

# Entities surfaced in zero-match hints. Keys are BidsQuery entity keys, values
# are the user-facing label used in the error message.
_HINT_ENTITIES: tuple[tuple[str, str], ...] = (
    ("sub", "subjects"),
    ("ses", "sessions"),
    ("task", "tasks"),
)

# Cap each list of available entity values in the zero-match hint so the
# message stays compact for large datasets.
_MAX_HINT_VALUES = 10


@dataclass(frozen=True)
class SelectionResult:
    """The output of :func:`select_files`.

    Carries the selected manifest entries plus the small piece of
    diagnostic state the orchestrator needs to surface include-pattern
    typos. Everything else the selection touches (intermediate match
    sets, per-pattern matches, the essential carve-out) is internal
    bookkeeping and does not leave the function.
    """

    selected: tuple[DatasetFile, ...]
    unmatched_includes: tuple[str, ...]


def select_files(
    files: Sequence[DatasetFile],
    *,
    query: BidsQuery,
    include: Sequence[str],
    exclude: Sequence[str],
) -> SelectionResult:
    """Compute the selection for a manifest under the given filters.

    Three phases, in order:

    1. Apply ``query`` to manifest paths (or pass everything non-dotfile
       through when the query is empty).
    2. Narrow the result with ``include`` glob patterns (no-op when
       ``include`` is empty).
    3. Subtract ``exclude`` glob patterns, then re-add essential BIDS
       root files plus — when the query is non-empty — the root-level
       ``*.json`` / ``*.tsv`` sidecar sweep (Option A).

    Raises :class:`~nemar.errors.SelectionError` when a non-empty BIDS
    query matches zero files; the message echoes available
    subjects/sessions/tasks to help users correct typos. Unmatched
    include patterns are surfaced through :attr:`SelectionResult.unmatched_includes`
    and the caller is expected to validate them via
    :func:`raise_if_unmatched_includes`.
    """
    filenames = [file.path for file in files]

    if query.is_empty():
        matched = tuple(
            filename
            for filename in filenames
            if not is_dotfile(filename)
        )
    else:
        parsed = [
            (file.path, BidsPath.parse(file.path))
            for file in files
            if not is_dotfile(file.path)
        ]
        matched = tuple(
            path for path, bids_path in parsed if _path_matches(bids_path, query)
        )
        if not matched:
            raise SelectionError(_zero_match_message(query, parsed))

    matched_set = set(matched)

    if include:
        include_matches = glob_filter(filenames, include)
        included_paths = {
            filename
            for matches in include_matches.values()
            for filename in matches
        }
        matched_set &= included_paths
    else:
        include_matches = {}

    if exclude:
        exclude_matches = glob_filter(filenames, exclude)
        excluded_paths = {
            filename
            for matches in exclude_matches.values()
            for filename in matches
        }
    else:
        excluded_paths = set()

    essential = ESSENTIAL_BIDS_FILES & set(filenames)
    # Root-level sidecar sweep ("Option A"): when a BIDS query is
    # active, any ``*.json`` / ``*.tsv`` at the dataset root joins
    # the essential set. Real NEMAR datasets keep BIDS-inherited
    # sidecars at the root (``task-<label>_events.json``,
    # ``space-*_coordsystem.json``, …) — without this sweep a
    # ``subject=...`` query would pull the recording and its
    # in-directory sidecars but miss the metadata that lets a
    # downstream tool decode event columns or coordinate systems.
    # The sweep over-fetches the small handful of root sidecars
    # that don't strictly apply (e.g. ``task-other_events.json``
    # for a recording we did not select); the alternative — a full
    # BIDS-inheritance walk that matches by entity overlap — is an
    # order of magnitude more code without a meaningful payoff on
    # the dataset shapes NEMAR actually publishes.
    if not query.is_empty():
        essential = essential | {
            filename
            for filename in filenames
            if "/" not in filename
            and (filename.endswith(".json") or filename.endswith(".tsv"))
        }
    selected_paths = (matched_set - excluded_paths) | essential

    unmatched_includes = tuple(
        pattern for pattern, matches in include_matches.items() if not matches
    )

    selected = tuple(file for file in files if file.path in selected_paths)

    return SelectionResult(
        selected=selected,
        unmatched_includes=unmatched_includes,
    )


def raise_if_unmatched_includes(
    result: SelectionResult, *, filenames: Sequence[str]
) -> None:
    """Raise ``SelectionError`` with did-you-mean hints for unmatched includes.

    ``filenames`` is the full manifest path list, used to suggest close
    matches via :func:`difflib.get_close_matches` for literal (non-glob)
    patterns. No-op when ``result.unmatched_includes`` is empty.
    """
    for pattern in result.unmatched_includes:
        has_glob = any(char in pattern for char in "*?[")
        candidates = (
            [] if has_glob else get_close_matches(pattern, list(filenames))
        )
        if candidates:
            extra = " Perhaps you mean: " + ", ".join(candidates[:5])
        else:
            extra = ""
        raise SelectionError(
            f"Could not find path in the NEMAR manifest: {pattern}.{extra}"
        )


# Glob-style matching helpers. Previously lived in ``nemar._glob`` as a
# standalone module; inlined here because :func:`select_files` is the
# only consumer of glob_filter, and ``is_dotfile`` is also a
# selection-time predicate (skip ``.DS_Store``, ``.git`` etc. before BIDS
# parsing). Keeping them in this module keeps the BIDS-selection logic
# in one file. Names are unchanged so the property test
# (``tests/property/test_glob_properties.py``) imports them directly.


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


def _zero_match_message(
    query: BidsQuery,
    parsed: list[tuple[str, BidsPath]],
) -> str:
    """Build the zero-match error message, with an Available: hint when useful.

    Only called from :func:`select_files` after a non-empty query
    matched zero files, so the query is always non-empty here.
    """
    base = f"No files matched the BIDS query: {query.describe()}."

    hint_parts: list[str] = []
    for entity_key, label in _HINT_ENTITIES:
        values: set[str] = set()
        for _, bids_path in parsed:
            value = bids_path.entities.get(entity_key)
            if value is not None:
                values.add(value)
        if values:
            sorted_values = sorted(values)[:_MAX_HINT_VALUES]
            hint_parts.append(f"{label}=[{','.join(sorted_values)}]")

    if not hint_parts:
        return base
    return f"{base} Available: {', '.join(hint_parts)}"


def _path_matches(path: BidsPath, query: BidsQuery) -> bool:
    """Return whether ``path`` satisfies every constraint in ``query``.

    Previously ``BidsPath.matches(query)``. Hoisted as a top-level
    predicate so the semantics live with the selection composition that
    invokes it. Behaviour is unchanged; the tests pin every branch.
    """
    for key, values in query.entities.items():
        if path.entities.get(key) not in values:
            return False
    if query.datatypes and path.datatype not in query.datatypes:
        return False
    if query.suffixes and path.suffix not in query.suffixes:
        return False
    if query.extensions and path.extension not in query.extensions:
        return False
    if query.scopes and path.scope not in query.scopes:
        return False
    if query.pipelines and path.pipeline not in query.pipelines:
        return False
    return True
