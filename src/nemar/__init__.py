"""A lightweight client for accessing NEMAR datasets."""

from importlib import metadata

__version__: str
try:
    __version__ = metadata.version("nemar-py")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from nemar._bids import BidsQuery as BidsQuery
from nemar._download import download as download
from nemar._download import fetch_dataset_index as fetch_dataset_index
from nemar._download import list_dataset_versions as list_dataset_versions
from nemar._endpoint import DataEndpoint as DataEndpoint
from nemar._models import VersionManifest as VersionManifest
from nemar._request import DownloadRequest as DownloadRequest
from nemar._request import TransferOptions as TransferOptions
