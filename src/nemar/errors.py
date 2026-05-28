"""Typed error hierarchy for nemar-py.

Every library-raised failure subclasses :class:`NemarError`. Programmatic
callers branch on the subclass; the CLI catches the base class at the boundary.
Each subclass documents its own failure mode below.

:class:`NemarError` subclasses :class:`RuntimeError`, so existing callers and
tests that catch ``RuntimeError`` keep working.

``ValueError`` is intentionally **not** part of this hierarchy. It remains the
right type for input-validation failures at the public boundary
(``download(...)``, ``NEMARClient(...)``, CLI option parsing); the CLI catches
both ``NemarError`` and ``ValueError`` at the boundary.
"""

from __future__ import annotations


class NemarError(RuntimeError):
    """Base class for every library-raised nemar-py failure.

    Subclasses :class:`RuntimeError` so existing callers and tests that
    catch ``RuntimeError`` keep working — programmatic callers can now
    branch on the typed subclasses without churn at the boundary.

    Do not raise :class:`NemarError` directly — always use a subclass.
    """


class EndpointError(NemarError):
    """Off-origin URL or bad endpoint configuration."""


class DatasetIndexError(NemarError):
    """Dataset-index payload validation or version resolution failure."""


class ManifestError(NemarError):
    """Version-manifest payload shape or content failure."""


class SelectionError(NemarError):
    """BIDS selection failure (zero match, unmatched include, case collision)."""


class TransportError(NemarError):
    """JSON-fetch retry / HTTP / JSON-decode failure during metadata reads."""


class TransferError(NemarError):
    """Bytes-on-the-wire failure during file streaming."""


class DataLadError(TransferError):
    """DataLad / git-annex transfer failure.

    Covers the optional-dependency import error, ``datalad install`` /
    ``datalad clone`` failures, and ``datalad get`` failures. The layered
    backend catches this specifically so it can fall back to HTTPS;
    callers who only catch :class:`TransferError` still see the original
    diagnostic when DataLad fails fatally.
    """


class S3Error(TransferError):
    """Direct-from-S3 transfer failure.

    Raised by :class:`~nemar.s3.S3Backend` when ``s3fs`` cannot resolve
    or fetch an object (missing key, unauthenticated, network error,
    non-annexed manifest entry, …). Subclasses :class:`TransferError`
    so callers who catch ``TransferError`` still see the diagnostic;
    the layered wrapper catches :class:`S3Error` specifically so it can
    fall back to the next layer (DataLad / HTTPS) for the whole batch.
    """


class VerificationError(NemarError):
    """Local file does not satisfy its manifest entry."""


class LocalTargetError(NemarError):
    """Pre-existing target directory is incompatible with the requested dataset."""


class LocalVersionMismatchError(LocalTargetError, FileExistsError):
    """Pre-existing target carries the same dataset but a different version.

    Inherits from both :class:`LocalTargetError` (so programmatic callers
    can branch on the typed hierarchy) and the builtin ``FileExistsError``
    (so legacy callers and tests that catch ``FileExistsError`` keep
    working without change).
    """
