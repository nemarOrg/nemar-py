"""Tests for the NemarError hierarchy.

The hierarchy is the public seam for programmatic error handling. These
tests pin two contracts:

1. Every typed subclass is reachable from the public ``nemar`` namespace.
2. The inheritance graph holds: each subclass is a :class:`NemarError`,
   :class:`NemarError` is a :class:`RuntimeError` (backward-compat),
   :class:`LocalVersionMismatchError` is also a :class:`FileExistsError`.
"""

from __future__ import annotations

import pytest

import nemar
from nemar._endpoint import DataEndpoint
from nemar.errors import (
    DataLadError,
    DatasetIndexError,
    EndpointError,
    LocalTargetError,
    LocalVersionMismatchError,
    ManifestError,
    NemarError,
    SelectionError,
    TransferError,
    TransportError,
    VerificationError,
)


@pytest.mark.parametrize(
    "cls",
    [
        EndpointError,
        DatasetIndexError,
        ManifestError,
        SelectionError,
        TransportError,
        TransferError,
        VerificationError,
        LocalTargetError,
        LocalVersionMismatchError,
        DataLadError,
    ],
)
def test_subclass_is_a_nemar_error(cls):
    assert issubclass(cls, NemarError)


def test_datalad_error_is_a_transfer_error():
    """DataLad failures live under TransferError so the layered backend can
    catch them with a narrow ``except`` while leaving every other
    bytes-on-the-wire failure to propagate.
    """
    assert issubclass(DataLadError, TransferError)


def test_nemar_error_is_a_runtime_error():
    """Legacy callers that catch RuntimeError continue to catch NemarError."""
    assert issubclass(NemarError, RuntimeError)


def test_local_version_mismatch_is_also_file_exists_error():
    """The version-mismatch case keeps FileExistsError semantics for legacy catches."""
    assert issubclass(LocalVersionMismatchError, FileExistsError)
    assert issubclass(LocalVersionMismatchError, LocalTargetError)


def test_endpoint_assert_within_raises_endpoint_error():
    """Off-origin URLs raise EndpointError, not bare RuntimeError."""
    endpoint = DataEndpoint.from_url("https://data.nemar.org/")
    with pytest.raises(EndpointError, match="Refusing to download a file"):
        endpoint.assert_within("https://elsewhere.example.com/file")


def test_top_level_keeps_root_and_fallback_errors():
    """The two errors users branch on most often stay at top level.

    The API diet kept ``NemarError`` (the recommended catch-all) and
    ``DataLadError`` (the layered-fallback signal) reachable as
    ``nemar.NemarError`` / ``nemar.DataLadError``. The rest moved under
    :mod:`nemar.errors`.
    """
    for name in ("NemarError", "DataLadError"):
        assert hasattr(nemar, name), f"nemar.{name} is not exported"
        assert isinstance(getattr(nemar, name), type), f"nemar.{name} is not a class"


def test_all_typed_errors_are_exported_from_nemar_errors():
    """Every typed error class is reachable from the :mod:`nemar.errors` submodule.

    After the API diet, the typed error classes live under
    ``nemar.errors``. ``NemarError`` and ``DataLadError`` are
    re-exported there too so a single import block covers every
    branchable error class.
    """
    import nemar.errors as nemar_errors

    expected = {
        "NemarError",
        "EndpointError",
        "DatasetIndexError",
        "ManifestError",
        "SelectionError",
        "TransportError",
        "TransferError",
        "VerificationError",
        "LocalTargetError",
        "LocalVersionMismatchError",
        "DataLadError",
    }
    for name in expected:
        assert hasattr(nemar_errors, name), f"nemar.errors.{name} is not exported"
        assert isinstance(getattr(nemar_errors, name), type), (
            f"nemar.errors.{name} is not a class"
        )
