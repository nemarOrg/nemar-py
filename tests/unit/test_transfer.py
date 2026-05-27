"""Unit tests for the transfer backend module.

Covers the seam between the orchestrator and the bytes-on-the-wire
phase:

* :func:`select_backend` — policy- and ``datalad_url``-driven backend
  resolution.
* :class:`PythonBackend` — minimal end-to-end pipe over a
  ``MockTransport`` so the shared-client path is exercised.

These tests are the contract that lets future work move the backend
into an adapter subpackage without touching the orchestrator.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx

from nemar._datalad import DataLadBackend, LayeredBackend
from nemar._models import DatasetFile
from nemar._request import TransferOptions
from nemar._retry import RetryPolicy
from nemar._transfer import (
    PythonBackend,
    select_backend,
)
from nemar._verification import VerifyPolicy


def _make_options(
    *,
    backend: str = "python",
    max_concurrent_downloads: int = 1,
    stream_timeout: float = 60.0,
) -> TransferOptions:
    return TransferOptions(
        backend=backend,
        max_concurrent_downloads=max_concurrent_downloads,
        stream_timeout=stream_timeout,
    )


class TestSelectBackend:
    """Backend resolution by policy and advertised ``datalad_url``."""

    def test_auto_returns_python_backend(self) -> None:
        backend = select_backend(_make_options(backend="auto"))
        assert isinstance(backend, PythonBackend)

    def test_explicit_python_returns_python_backend(self) -> None:
        backend = select_backend(_make_options(backend="python"))
        assert isinstance(backend, PythonBackend)

    def test_auto_with_datalad_url_returns_layered_backend(self) -> None:
        """When the index advertises a datalad_url, ``auto`` layers
        DataLad over HTTPS.
        """
        backend = select_backend(
            _make_options(backend="auto"),
            datalad_url="https://github.com/OpenNeuroDatasets/ds000132.git",
            revision="v1.0.0",
        )
        assert isinstance(backend, LayeredBackend)
        assert isinstance(backend.primary, DataLadBackend)
        assert backend.primary.datalad_url.endswith("ds000132.git")
        assert backend.primary.revision == "v1.0.0"
        assert isinstance(backend.fallback, PythonBackend)

    def test_explicit_datalad_with_url_returns_layered_backend(self) -> None:
        """Explicit ``datalad`` keeps the HTTPS fallback ('always fall back')."""
        backend = select_backend(
            _make_options(backend="datalad"),
            datalad_url="https://github.com/OpenNeuroDatasets/ds000132.git",
        )
        assert isinstance(backend, LayeredBackend)
        assert isinstance(backend.primary, DataLadBackend)
        assert isinstance(backend.fallback, PythonBackend)

    def test_explicit_datalad_without_url_falls_back_to_https(self) -> None:
        """``datalad`` with no advertised URL degrades to plain HTTPS with a notice."""
        backend = select_backend(_make_options(backend="datalad"), datalad_url=None)
        assert isinstance(backend, PythonBackend)

    def test_explicit_python_ignores_datalad_url(self) -> None:
        """Explicit ``python`` is an opt-out from the DataLad layer."""
        backend = select_backend(
            _make_options(backend="python"),
            datalad_url="https://github.com/OpenNeuroDatasets/ds000132.git",
        )
        assert isinstance(backend, PythonBackend)


class TestPythonBackendTransfer:
    """End-to-end shape of the Python adapter using a mocked transport."""

    def test_python_backend_writes_file_and_verifies(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        data = b"hello nemar"
        file = DatasetFile(
            path="dataset_description.json",
            url="https://data.nemar.org/nm000132/v1.0.0/dataset_description.json",
            size=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.host == "data.nemar.org"
            return httpx.Response(200, content=data, request=request)

        transport = httpx.MockTransport(handler)
        original_client = httpx.Client

        class PatchedClient(httpx.Client):
            def __init__(self, *args, **kwargs):
                kwargs["transport"] = transport
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(httpx, "Client", PatchedClient)
        try:
            backend = PythonBackend()
            backend.transfer(
                [file],
                target_dir=tmp_path,
                options=_make_options(
                    backend="python",
                    max_concurrent_downloads=1,
                    stream_timeout=60.0,
                ),
                verify=VerifyPolicy(verify_size=True, verify_hash=True),
                retry=RetryPolicy.default().with_attempts(0),
            )
        finally:
            monkeypatch.setattr(httpx, "Client", original_client)

        assert (tmp_path / "dataset_description.json").read_bytes() == data


class TestBackendProtocolShape:
    """The HTTPS adapter honours the documented interface."""

    def test_python_backend_exposes_transfer_method(self) -> None:
        assert hasattr(PythonBackend, "transfer")
        assert callable(PythonBackend.transfer)
