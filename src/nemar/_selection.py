"""SelectionPlan: a transparent record of NEMAR BIDS file selection.

This module owns the composition of three operations that previously lived in
``_download._select_bids_files``:

1. Apply a :class:`~nemar._bids.BidsQuery` to manifest paths.
2. Narrow the result with include glob patterns.
3. Subtract exclude glob patterns and force-keep essential BIDS root files.

Instead of returning only the final list of files, :class:`SelectionPlan`
preserves the *plan* of how the selection happened, which makes debugging,
diagnostics, and future callers far easier.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import get_close_matches

from nemar import _bids, _glob
from nemar._errors import SelectionError
from nemar._models import DatasetFile

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
class SelectionPlan:
    """A frozen record of how a NEMAR manifest was filtered.

    Use :meth:`build` to construct an instance; the dataclass fields capture
    everything that happened along the way so callers can inspect, log, or
    surface diagnostics about the selection.
    """

    matched_by_query: tuple[str, ...]
    included_by_pattern: Mapping[str, tuple[str, ...]]
    excluded_by_pattern: Mapping[str, tuple[str, ...]]
    essential_kept: tuple[str, ...]
    unmatched_includes: tuple[str, ...]
    final: tuple[DatasetFile, ...]

    @classmethod
    def build(
        cls,
        files: Sequence[DatasetFile],
        *,
        query: _bids.BidsQuery,
        include: Sequence[str],
        exclude: Sequence[str],
    ) -> SelectionPlan:
        """Compute the selection plan for a manifest under the given filters.

        Raises ``RuntimeError`` when a non-empty BIDS query matches zero files;
        the message echoes the available subjects/sessions/tasks to help users
        correct typos.
        """
        filenames = [file.path for file in files]

        if query.is_empty():
            matched = tuple(
                filename
                for filename in filenames
                if not _glob.is_dotfile(filename)
            )
        else:
            parsed = [
                (file.path, _bids.BidsPath.parse(file.path))
                for file in files
                if not _glob.is_dotfile(file.path)
            ]
            matched = tuple(
                path for path, bids_path in parsed if bids_path.matches(query)
            )
            if not matched:
                raise SelectionError(_zero_match_message(query, parsed))

        matched_set = set(matched)

        if include:
            include_matches = _glob.glob_filter(filenames, include)
            included_by_pattern = {
                pattern: tuple(sorted(include_matches.get(pattern, ())))
                for pattern in include
            }
            included_paths = {
                filename
                for matches in include_matches.values()
                for filename in matches
            }
            matched_set &= included_paths
        else:
            include_matches = {}
            included_by_pattern = {}

        if exclude:
            exclude_matches = _glob.glob_filter(filenames, exclude)
            excluded_by_pattern = {
                pattern: tuple(sorted(exclude_matches.get(pattern, ())))
                for pattern in exclude
            }
            excluded_paths = {
                filename
                for matches in exclude_matches.values()
                for filename in matches
            }
        else:
            excluded_by_pattern = {}
            excluded_paths = set()

        essential = ESSENTIAL_BIDS_FILES & set(filenames)
        selected = (matched_set - excluded_paths) | essential
        essential_kept = tuple(sorted(essential))

        unmatched_includes = tuple(
            pattern for pattern, matches in include_matches.items() if not matches
        )

        final = tuple(file for file in files if file.path in selected)

        return cls(
            matched_by_query=matched,
            included_by_pattern=included_by_pattern,
            excluded_by_pattern=excluded_by_pattern,
            essential_kept=essential_kept,
            unmatched_includes=unmatched_includes,
            final=final,
        )

    def raise_if_unmatched_includes(self, *, filenames: list[str]) -> None:
        """Raise ``RuntimeError`` with did-you-mean hints for unmatched includes.

        ``filenames`` is the full manifest path list, used to suggest close
        matches via :func:`difflib.get_close_matches` for literal (non-glob)
        patterns.
        """
        for pattern in self.unmatched_includes:
            has_glob = any(char in pattern for char in "*?[")
            candidates = [] if has_glob else get_close_matches(pattern, filenames)
            if candidates:
                extra = " Perhaps you mean: " + ", ".join(candidates[:5])
            else:
                extra = ""
            raise SelectionError(
                f"Could not find path in the NEMAR manifest: {pattern}.{extra}"
            )


def _zero_match_message(
    query: _bids.BidsQuery,
    parsed: list[tuple[str, _bids.BidsPath]],
) -> str:
    """Build the zero-match error message, with an Available: hint when useful.

    Only called from :meth:`SelectionPlan.build` after a non-empty query
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
