"""A lightweight client for accessing NEMAR datasets.

The top-level surface is intentionally narrow: it carries the eight
public names that the recommended use cases need, plus ``__version__``.
Anything else lives under a clearly-named submodule:

* :mod:`nemar.errors` — the full typed error hierarchy. Everything is a
  subclass of :class:`NemarError` (which is also re-exported here), so
  the catch-all idiom ``except nemar.NemarError`` still works.
* :mod:`nemar.models` — :class:`~nemar.models.VersionManifest` and
  :class:`~nemar.models.DataEndpoint` for callers that parse or
  construct these value types directly.
* :mod:`nemar.config` — :class:`~nemar.config.DownloadRequest`,
  :class:`~nemar.config.TransferOptions`,
  :class:`~nemar.config.VerifyResult` — request/policy escape hatches.
* :mod:`nemar.transfer` — :func:`~nemar.transfer.download_one`, the
  single-file transfer primitive.
"""

from importlib import metadata as _metadata

__version__: str
try:
    __version__ = _metadata.version("nemar-py")
except _metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from nemar._bids import BidsQuery as BidsQuery
from nemar._client import NEMARClient as NEMARClient
from nemar._download import download as download
from nemar._download import fetch_dataset_index as fetch_dataset_index
from nemar._download import list_dataset_versions as list_dataset_versions
from nemar._errors import DataLadError as DataLadError
from nemar._errors import NemarError as NemarError
from nemar._verification import VerifyPolicy as VerifyPolicy
