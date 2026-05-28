"""A lightweight client for accessing NEMAR datasets.

The top-level surface is intentionally narrow: it carries the eight
public names that the recommended use cases need, plus ``__version__``.
Anything else lives under a clearly-named submodule:

* :mod:`nemar.errors` — the full typed error hierarchy. Everything is a
  subclass of :class:`NemarError` (which is also re-exported here), so
  the catch-all idiom ``except nemar.NemarError`` still works.
* :mod:`nemar.transfer` — :func:`~nemar.transfer.download_one` /
  :func:`~nemar.transfer.download_files`, the per-file and bulk
  transfer primitives that external clients (eegdash, custom
  selectors) compose on top of a parsed manifest.
* :mod:`nemar.s3` — the documented NEMAR S3 contract: bucket constants
  plus URL helpers (:func:`~nemar.s3.s3_object_url`,
  :func:`~nemar.s3.version_url`, :func:`~nemar.s3.archive_url`) for
  callers that want to bypass the catalog endpoint.
"""

from importlib import metadata as _metadata

__version__: str
try:
    __version__ = _metadata.version("nemar-py")
except _metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from nemar._client import NEMARClient as NEMARClient
from nemar._download import download as download
from nemar._download import fetch_dataset_index as fetch_dataset_index
from nemar._download import list_dataset_versions as list_dataset_versions
from nemar._models import BidsQuery as BidsQuery
from nemar._verification import VerifyPolicy as VerifyPolicy
from nemar.errors import DataLadError as DataLadError
from nemar.errors import NemarError as NemarError
