"""A lightweight client for accessing NEMAR datasets."""

from importlib import metadata

__version__: str
try:
    __version__ = metadata.version("nemar-py")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from nemar._bids import BidsQuery as BidsQuery
from nemar._client import NEMARClient as NEMARClient
from nemar._download import download as download
from nemar._download import fetch_dataset_index as fetch_dataset_index
from nemar._download import list_dataset_versions as list_dataset_versions
from nemar._endpoint import DataEndpoint as DataEndpoint
from nemar._errors import DatasetIndexError as DatasetIndexError
from nemar._errors import EndpointError as EndpointError
from nemar._errors import LocalTargetError as LocalTargetError
from nemar._errors import LocalVersionMismatchError as LocalVersionMismatchError
from nemar._errors import ManifestError as ManifestError
from nemar._errors import NemarError as NemarError
from nemar._errors import SelectionError as SelectionError
from nemar._errors import TransferError as TransferError
from nemar._errors import TransportError as TransportError
from nemar._errors import VerificationError as VerificationError
from nemar._models import VersionManifest as VersionManifest
from nemar._request import DownloadRequest as DownloadRequest
from nemar._request import TransferOptions as TransferOptions
from nemar._transfer import download_one as download_one
from nemar._verification import VerifyPolicy as VerifyPolicy
from nemar._verification import VerifyResult as VerifyResult
