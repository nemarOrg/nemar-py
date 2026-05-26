"""Typed error hierarchy — public submodule.

Re-exports of the error classes defined in :mod:`nemar._errors`. The
inheritance graph and semantics are unchanged; only the import path is
public. ``NemarError`` and ``DataLadError`` are also reachable from the
top-level ``nemar`` package because they are the two classes users
branch on most often (catch-all and DataLad-fallback signal); the rest
live here.

The recommended catch-all idiom is still:

.. code-block:: python

    import nemar
    try:
        nemar.download(dataset="nm000132", target_dir="data")
    except nemar.NemarError as exc:
        ...

Every class below is a subclass of :class:`nemar.NemarError`, so the
above pattern catches all nemar-raised exceptions even after this split.
"""

from __future__ import annotations

from nemar._errors import DataLadError as DataLadError
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
