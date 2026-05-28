"""The BIDS concept: path parsing, the semantic query, building it, and matching.

A leaf module — no ``nemar`` imports. Everything about BIDS lives here so
the concept can be held in one mental unit: the value types
(:class:`BidsPath`, :class:`BidsQuery`), the path parser, the kwargs →
query builder (:func:`build_bids_query`), and the path↔query matcher
(:func:`_path_matches`). Consolidated from ``_models`` (types + parse),
``_request`` (build), and ``_selection`` (match) because BIDS is one
model-coupled concept; splitting it across three modules at low distance
only fragmented it (balanced-coupling: high strength wants low distance).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

COMPOUND_EXTENSIONS = (".nii.gz", ".tsv.gz")
DATASET_SCOPES = {"raw", "derivatives", "stimuli", "sourcedata", "code"}
ENTITY_ALIASES = {
    "subject": "sub",
    "session": "ses",
    "acquisition": "acq",
}


@dataclass(frozen=True)
class BidsPath:
    """Parsed BIDS semantics for one relative path.

    Pure value type: fields plus a :meth:`parse` constructor. The
    matching predicate (does this path satisfy a query?) is
    :func:`_path_matches`, also in this module.
    """

    path: str
    entities: Mapping[str, str] = field(default_factory=dict)
    datatype: str | None = None
    suffix: str | None = None
    extension: str | None = None
    scope: str = "raw"
    pipeline: str | None = None

    @classmethod
    def parse(cls, path: str) -> BidsPath:
        """Parse a relative path into BIDS entities, datatype, suffix, extension."""
        parts = PurePosixPath(path).parts
        scope, pipeline, semantic_parts = _split_dataset_scope(parts)
        filename = (
            semantic_parts[-1] if semantic_parts else parts[-1] if parts else path
        )
        stem, extension = _split_extension(filename)
        entities: dict[str, str] = {}

        for dirname in semantic_parts[:-1]:
            _add_entity_from_token(entities, dirname)

        filename_tokens = stem.split("_")
        for token in filename_tokens[:-1]:
            _add_entity_from_token(entities, token)

        suffix = (
            filename_tokens[-1] if filename_tokens and filename_tokens[-1] else None
        )
        datatype = _parse_datatype(semantic_parts)
        return cls(
            path=path,
            entities=entities,
            datatype=datatype,
            suffix=suffix,
            extension=extension,
            scope=scope,
            pipeline=pipeline,
        )


@dataclass(frozen=True)
class BidsQuery:
    """A semantic BIDS query over manifest paths.

    Pure value type: a frozen bundle of normalized filter tuples plus
    two read-only helpers (:meth:`is_empty`, :meth:`describe`). Built
    from raw kwargs by :func:`build_bids_query` (also here); applied to
    a parsed :class:`BidsPath` by :func:`_path_matches` (also here).
    """

    entities: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    datatypes: tuple[str, ...] = ()
    suffixes: tuple[str, ...] = ()
    extensions: tuple[str, ...] = ()
    scopes: tuple[str, ...] = ()
    pipelines: tuple[str, ...] = ()

    def is_empty(self) -> bool:
        """Return whether no semantic BIDS constraints were provided."""
        return not (
            self.entities
            or self.datatypes
            or self.suffixes
            or self.extensions
            or self.scopes
            or self.pipelines
        )

    def describe(self) -> str:
        """Return a compact user-facing representation of the query."""
        parts: list[str] = []
        for key, values in sorted(self.entities.items()):
            parts.append(f"{key}={','.join(values)}")
        if self.datatypes:
            parts.append(f"datatype={','.join(self.datatypes)}")
        if self.suffixes:
            parts.append(f"suffix={','.join(self.suffixes)}")
        if self.extensions:
            parts.append(f"extension={','.join(self.extensions)}")
        if self.scopes:
            parts.append(f"scope={','.join(self.scopes)}")
        if self.pipelines:
            parts.append(f"pipeline={','.join(self.pipelines)}")
        return "; ".join(parts) or "<empty>"


def normalize_entity_key(key: str) -> str:
    """Normalize a BIDS entity key (e.g. ``"subject"`` → ``"sub"``).

    Single source of truth for the alias table, shared by
    :meth:`BidsPath.parse` and :func:`build_bids_query` so the parse
    side and the query-construction side cannot drift.
    """
    return ENTITY_ALIASES.get(key, key)


def _split_dataset_scope(
    parts: tuple[str, ...],
) -> tuple[str, str | None, tuple[str, ...]]:
    if not parts:
        return "raw", None, parts
    if parts[0] == "derivatives":
        pipeline = parts[1] if len(parts) > 2 else None
        semantic_parts = parts[2:] if pipeline is not None else parts[1:]
        return "derivatives", pipeline, semantic_parts
    if parts[0] in DATASET_SCOPES - {"raw", "derivatives"}:
        return parts[0], None, parts[1:]
    return "raw", None, parts


def _split_extension(filename: str) -> tuple[str, str | None]:
    lower = filename.lower()
    for extension in COMPOUND_EXTENSIONS:
        if lower.endswith(extension):
            return filename[: -len(extension)], extension
    path = PurePosixPath(filename)
    if not path.suffix:
        return filename, None
    return filename[: -len(path.suffix)], path.suffix.lower()


def _add_entity_from_token(entities: dict[str, str], token: str) -> None:
    if "-" not in token:
        return
    key, value = token.split("-", 1)
    if key and value:
        entities[normalize_entity_key(key)] = value


def _parse_datatype(parts: tuple[str, ...]) -> str | None:
    directories = parts[:-1]
    for index, dirname in enumerate(directories):
        if not dirname.startswith("sub-"):
            continue
        datatype_index = index + 1
        if datatype_index < len(directories) and directories[datatype_index].startswith(
            "ses-"
        ):
            datatype_index += 1
        if datatype_index < len(directories):
            return directories[datatype_index]
    return None


def build_bids_query(
    *,
    subject: str | Iterable[str] | None = None,
    session: str | Iterable[str] | None = None,
    task: str | Iterable[str] | None = None,
    run: str | Iterable[str] | None = None,
    acquisition: str | Iterable[str] | None = None,
    datatype: str | Iterable[str] | None = None,
    suffix: str | Iterable[str] | None = None,
    extension: str | Iterable[str] | None = None,
    scope: str | Iterable[str] | None = None,
    pipeline: str | Iterable[str] | None = None,
    entity: Mapping[str, Any] | Iterable[str] | None = None,
) -> BidsQuery:
    """Build a :class:`BidsQuery` from common BIDS filters and generic entities.

    Called by :meth:`nemar._request.DownloadRequest.from_kwargs`. Kept
    here, in the BIDS module, alongside the query type it constructs and
    the matcher that consumes it.
    """
    entities = _normalize_entity_mapping(entity)
    _add_entity_filter(entities, "sub", subject)
    _add_entity_filter(entities, "ses", session)
    _add_entity_filter(entities, "task", task)
    _add_entity_filter(entities, "run", run)
    _add_entity_filter(entities, "acq", acquisition)

    # Default to ``raw`` scope when none is supplied. Without this,
    # a subject/session query would also match files under
    # ``derivatives/<pipeline>/sub-X/...`` because the BIDS subject
    # entity appears in both the raw tree and any subject-scoped
    # derivative — surprising over-fetch (we've seen 260 MB of
    # epoched derivatives sneak in on a sub-02 query against
    # nm000133). The explicit choices are unchanged: callers who
    # want a derivative pipeline, stimuli, sourcedata, or code pass
    # ``scope=`` directly (e.g. ``scope="derivatives", pipeline=...``).
    if scope is None:
        scope = "raw"
    return BidsQuery(
        entities=entities,
        datatypes=_normalize_values(datatype),
        suffixes=_normalize_values(suffix),
        extensions=tuple(
            _normalize_extension(v) for v in _normalize_values(extension)
        ),
        scopes=_normalize_scopes(scope),
        pipelines=_normalize_values(pipeline),
    )


def _normalize_entity_mapping(
    entity: Mapping[str, Any] | Iterable[str] | None,
) -> dict[str, tuple[str, ...]]:
    if entity is None:
        return {}
    if isinstance(entity, Mapping):
        return {
            normalize_entity_key(key): _normalize_entity_values(key, value)
            for key, value in entity.items()
        }

    out: dict[str, tuple[str, ...]] = {}
    for item in entity:
        if "=" not in item:
            raise ValueError(
                f'BIDS entity filters must use "key=value" syntax, got {item!r}.'
            )
        key, value = item.split("=", 1)
        if not key or not value:
            raise ValueError(
                f'BIDS entity filters must use "key=value" syntax, got {item!r}.'
            )
        out[normalize_entity_key(key)] = _normalize_entity_values(key, value)
    return out


def _add_entity_filter(
    entities: dict[str, tuple[str, ...]],
    key: str,
    value: str | Iterable[str] | None,
) -> None:
    values = _normalize_entity_values(key, value)
    if values:
        entities[key] = values


def _normalize_entity_values(key: str, values: Any) -> tuple[str, ...]:
    normalized_key = normalize_entity_key(key)
    out = []
    for value in _normalize_values(values):
        prefix = f"{normalized_key}-"
        out.append(value.removeprefix(prefix))
    return tuple(out)


def _normalize_values(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    if not isinstance(values, Iterable):
        return (str(values),)
    return tuple(str(value) for value in values)


def _normalize_extension(value: str) -> str:
    value = value.lower()
    if not value.startswith("."):
        value = "." + value
    return value


def _normalize_scopes(values: Any) -> tuple[str, ...]:
    scopes = tuple(value.lower() for value in _normalize_values(values))
    invalid = sorted(set(scopes) - DATASET_SCOPES)
    if invalid:
        expected = ", ".join(sorted(DATASET_SCOPES))
        raise ValueError(
            "Unknown BIDS dataset scope(s): "
            f"{', '.join(invalid)}. Expected one of: {expected}."
        )
    return scopes


def _path_matches(path: BidsPath, query: BidsQuery) -> bool:
    """Return whether ``path`` satisfies every constraint in ``query``.

    Applied by :func:`nemar._selection.select_files`. Behaviour is
    unchanged; the tests pin every branch.
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
