"""Value type composing a complete NEMAR download request.

The orchestrator in :mod:`nemar._download` used to accept ~25 keyword
arguments and spend its first 45 lines shuffling them into normalized
shape. That choreography is what :class:`DownloadRequest` owns now: a
frozen value that bundles the validated dataset identity, endpoint,
normalized version tag, resolved target path, BIDS query, and the
transfer / verify / retry policies. The orchestrator becomes a thin
algorithm of six step calls in order against a single ``request`` value.

This module also owns BIDS-query *construction* (:func:`build_bids_query`
and its kwargs-normalization helpers): turning the raw ``subject=`` /
``task=`` / ``entity=`` kwargs into a frozen :class:`~nemar._models.BidsQuery`
is part of the same normalization pass as the rest of ``from_kwargs``.
The query *type* lives in :mod:`nemar._models` and its *matching* against
paths lives in :mod:`nemar._selection`; only the build step is here, next
to its sole caller.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemar._backend import VALID_BACKENDS, TransferOptions
from nemar._constants import DATASET_ID_RE, DEFAULT_DATA_URL
from nemar._endpoint import DataEndpoint
from nemar._models import DATASET_SCOPES, BidsQuery, normalize_entity_key
from nemar._retry import RetryPolicy
from nemar._verification import VerifyPolicy


@dataclass(frozen=True)
class DownloadRequest:
    """A fully-normalized NEMAR download request.

    Construct via :meth:`from_kwargs` (library callers and the CLI). The
    orchestrator ``_download._run`` consumes one of these and executes the
    six algorithmic steps (index, version, metadata, manifest, select,
    transfer) against it.

    All inputs that the orchestrator previously normalized inline are
    pre-resolved here: ``target_path`` is absolute and home-expanded;
    ``endpoint`` is validated HTTPS with a normalized trailing slash;
    ``requested_tag`` carries its ``v`` prefix when the user wrote a
    bare digit version; ``bids_query`` is the already-built query;
    ``include_patterns`` / ``exclude_patterns`` are tuples of strings
    (never ``None``, never a bare string).
    """

    dataset: str
    endpoint: DataEndpoint
    requested_tag: str | None
    target_path: Path
    include_patterns: tuple[str, ...]
    exclude_patterns: tuple[str, ...]
    bids_query: BidsQuery
    transfer: TransferOptions
    retry: RetryPolicy
    verify: VerifyPolicy
    metadata_timeout: float
    no_data: bool
    trust_existing: bool = False

    @classmethod
    def from_kwargs(
        cls,
        *,
        dataset: str,
        tag: str | None = None,
        target_dir: Path | str | None = None,
        include: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
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
        downloader: str = "auto",
        verify_hash: bool = True,
        verify_size: bool = True,
        max_retries: int = 5,
        max_concurrent_downloads: int = 16,
        metadata_timeout: float = 30.0,
        stream_timeout: float = 60.0,
        data_url: str = DEFAULT_DATA_URL,
        no_data: bool = False,
        trust_existing: bool = False,
    ) -> DownloadRequest:
        """Build a request from the public ``download()`` kwargs.

        This is the single normalization point: every check and conversion
        that ``download()`` previously did in its first 45 lines lives
        here so the orchestrator stays algorithmic.
        """
        _validate(
            dataset=dataset,
            downloader=downloader,
            max_retries=max_retries,
            max_concurrent_downloads=max_concurrent_downloads,
        )
        endpoint = DataEndpoint.from_url(data_url)
        requested_tag = _normalize_version_tag(tag) if tag is not None else None
        target_path = Path(
            dataset if target_dir is None else target_dir
        ).expanduser().resolve()
        include_patterns = _normalize_patterns(include)
        exclude_patterns = _normalize_patterns(exclude)
        bids_query = build_bids_query(
            subject=subject,
            session=session,
            task=task,
            run=run,
            acquisition=acquisition,
            datatype=datatype,
            suffix=suffix,
            extension=extension,
            scope=scope,
            pipeline=pipeline,
            entity=entity,
        )
        transfer = TransferOptions(
            backend=str(downloader),
            max_concurrent_downloads=max_concurrent_downloads,
            stream_timeout=stream_timeout,
        )
        retry = RetryPolicy.default().with_attempts(max_retries)
        verify = VerifyPolicy(verify_size=verify_size, verify_hash=verify_hash)

        return cls(
            dataset=dataset,
            endpoint=endpoint,
            requested_tag=requested_tag,
            target_path=target_path,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            bids_query=bids_query,
            transfer=transfer,
            retry=retry,
            verify=verify,
            metadata_timeout=metadata_timeout,
            no_data=no_data,
            trust_existing=trust_existing,
        )


def _validate(
    *,
    dataset: str,
    downloader: str,
    max_retries: int,
    max_concurrent_downloads: int,
) -> None:
    if not DATASET_ID_RE.fullmatch(dataset):
        raise ValueError(
            'dataset must look like "nm000132" or "on005505".'
        )
    if downloader not in VALID_BACKENDS:
        raise ValueError(
            'downloader must be one of "auto", "python", "datalad", or "s3".'
        )
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative.")
    if max_concurrent_downloads < 1:
        raise ValueError("max_concurrent_downloads must be at least 1.")


def _normalize_version_tag(tag: str) -> str:
    tag = tag.strip()
    if not tag:
        raise ValueError("tag must not be empty.")
    if tag[0].isdigit():
        return f"v{tag}"
    return tag


def _normalize_patterns(patterns: Iterable[str] | None) -> tuple[str, ...]:
    if patterns is None:
        return ()
    if isinstance(patterns, str):
        return (patterns,)
    return tuple(patterns)


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

    The sole caller is :meth:`DownloadRequest.from_kwargs`, so the
    raw-kwargs → normalized-tuples transformation lives here next to it.
    The query *type* lives in :mod:`nemar._models`; *matching* a parsed
    path against the query lives in :mod:`nemar._selection`.
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
