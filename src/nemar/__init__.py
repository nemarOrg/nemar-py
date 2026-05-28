"""A lightweight client for accessing NEMAR datasets."""

__version__ = "0.1.0"

from nemar._client import NEMARClient
from nemar._download import download, fetch_dataset_index, list_dataset_versions
from nemar._models import BidsQuery
from nemar._streaming import download_one
from nemar._transfer import download_files
from nemar._verification import VerifyPolicy
from nemar.errors import DataLadError, NemarError

__all__ = [
    "BidsQuery",
    "DataLadError",
    "NEMARClient",
    "NemarError",
    "VerifyPolicy",
    "__version__",
    "download",
    "fetch_dataset_index",
    "list_dataset_versions",
    "download_files",
    "download_one",
]
