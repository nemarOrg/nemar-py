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
from nemar import (
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
from nemar._endpoint import DataEndpoint


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
    ],
)
def test_subclass_is_a_nemar_error(cls):
    assert issubclass(cls, NemarError)


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


def test_all_typed_errors_are_exported_from_nemar():
    """Every typed error class is reachable from the top-level ``nemar`` module."""
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
    }
    for name in expected:
        assert hasattr(nemar, name), f"nemar.{name} is not exported"
        assert isinstance(getattr(nemar, name), type), f"nemar.{name} is not a class"
