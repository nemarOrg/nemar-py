"""Typed error hierarchy for nemar-py.

Every library-raised failure is a subclass of :class:`NemarError`. Programmatic
callers can branch on the subclass; the CLI catches the base class at the
boundary. Subclasses align with the modules that own each failure mode so
``isinstance(exc, ManifestError)`` answers "this came from manifest parsing"
without grepping a message string.

Subclasses:

* :class:`EndpointError` — origin-scoping violations (off-origin redirect,
  bad HTTPS prefix). Raised by :mod:`nemar._endpoint`.
* :class:`DatasetIndexError` — dataset-index validation and version
  resolution. Raised by :mod:`nemar._models` (``parse_dataset_index``,
  ``DatasetIndex.resolve_version``) and :class:`nemar._client.NEMARClient`
  on payload identity mismatch.
* :class:`ManifestError` — version-manifest shape / content errors and
  "version not published". Raised by
  :func:`nemar._models.VersionManifest.parse` and
  :class:`nemar._client.NEMARClient` on manifest payload issues.
* :class:`SelectionError` — BIDS selection (zero match, unmatched includes,
  case collisions). Raised by :class:`nemar._selection.SelectionPlan` and
  the case-collision guard in :func:`nemar._download._run`.
* :class:`TransportError` — JSON-fetch retries exhausted, HTTP errors,
  JSON-decode failures. Raised by :func:`nemar._transport.fetch_json`.
  (Off-origin redirects raise :class:`EndpointError`, not ``TransportError``.)
* :class:`TransferError` — bytes-on-the-wire failures from the transfer
  backends. Raised by :mod:`nemar._transfer` adapters and
  :func:`nemar.download_one`.
* :class:`VerificationError` — local file does not satisfy its manifest
  entry. Raised by :func:`nemar._verification.assert_all_present` and the
  bulk-path per-file verify in :mod:`nemar._transfer`.
* :class:`LocalTargetError` — pre-existing target directory carries a
  different NEMAR dataset (DOI mismatch, missing Version). Raised by
  :class:`nemar._local_dataset.LocalDataset`.
* :class:`LocalVersionMismatchError` — pre-existing target carries the
  same dataset but a different version. Subclass of both
  :class:`LocalTargetError` and the builtin ``FileExistsError`` so legacy
  callers that catch ``FileExistsError`` keep working.

``ValueError`` is **not** part of this hierarchy. It remains the right type
for input-validation failures at the public boundary (``download(...)``,
``NEMARClient(...)``, the CLI option parsing). The CLI catches both
``NemarError`` and ``ValueError`` at the boundary.
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
