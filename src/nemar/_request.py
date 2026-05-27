"""Value type composing a complete NEMAR download request.

The orchestrator in :mod:`nemar._download` used to accept ~25 keyword
arguments and spend its first 45 lines shuffling them into normalized
shape. That choreography is what :class:`DownloadRequest` owns now: a
frozen value that bundles the validated dataset identity, endpoint,
normalized version tag, resolved target path, BIDS query, and the
transfer / verify / retry policies. The orchestrator becomes a thin
algorithm of six step calls in order against a single ``request`` value.

:class:`TransferOptions` is the small bundle of transfer-knob defaults
(backend, concurrency, stream timeout, aria2 timeout) that travels with
the request. It lives next to :class:`DownloadRequest` rather than in a
separate ``_transfer_options.py`` because the two are constructed and
consumed together; splitting them would invite drift between the
normalization (request) and the runtime knob (options).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemar._bids import BidsQuery
from nemar._endpoint import DataEndpoint
from nemar._retry import RetryPolicy
from nemar._verification import VerifyPolicy

DEFAULT_DATA_URL = "https://data.nemar.org/"
DATASET_ID_RE = re.compile(r"^(nm|on)\d{6}$")
_VALID_BACKENDS = frozenset({"auto", "aria2", "python", "datalad"})


@dataclass(frozen=True)
class TransferOptions:
    """Transfer-backend knobs that travel with one download request.

    Bundles the four runtime values the orchestrator forwards to
    :func:`nemar._download._transfer_files`: which backend to select,
    how much concurrency to allow, the per-stream timeout for the
    Python backend, and the optional wall-clock timeout for the
    ``aria2c`` subprocess.
    """

    backend: str
    max_concurrent_downloads: int
    stream_timeout: float
    aria2_timeout: float | None


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
        aria2_timeout: float | None = None,
        data_url: str = DEFAULT_DATA_URL,
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
        bids_query = BidsQuery.from_filters(
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
            aria2_timeout=aria2_timeout,
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
    if downloader not in _VALID_BACKENDS:
        raise ValueError(
            'downloader must be one of "auto", "aria2", "python", or "datalad".'
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
