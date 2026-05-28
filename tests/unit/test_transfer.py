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

from nemar._datalad import DataLadBackend
from nemar._models import DatasetFile
from nemar._retry import RetryPolicy
from nemar._transfer import (
    LayeredBackend,
    PythonBackend,
    TransferOptions,
    select_backend,
)
from nemar._verification import VerifyPolicy
from nemar.errors import DataLadError, S3Error
from nemar.s3 import S3Backend


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


# ---------------------------------------------------------------------------
# LayeredBackend — generalized fallback contract
# ---------------------------------------------------------------------------


class _RecordingBackend:
    """Tiny in-memory backend stub that records calls and can raise."""

    def __init__(self, *, raises: BaseException | None = None) -> None:
        self.raises = raises
        self.calls: list[tuple] = []

    def transfer(self, files, *, target_dir, options, verify, retry) -> None:
        self.calls.append((tuple(files), target_dir, options, verify, retry))
        if self.raises is not None:
            raise self.raises


def _opts() -> TransferOptions:
    return TransferOptions(
        backend="auto", max_concurrent_downloads=1, stream_timeout=60.0
    )


class TestLayeredBackendGeneralized:
    """``LayeredBackend.fallback_on`` is the seam.

    The pre-S3 contract caught :class:`DataLadError` only; the new
    contract takes a tuple of exception classes so the same wrapper
    composes the S3 layer (catching :class:`S3Error`) and the DataLad
    layer (catching :class:`DataLadError`) with no per-layer special
    case.
    """

    def test_fallback_on_custom_class_catches_and_proceeds(
        self, tmp_path: Path
    ) -> None:
        from nemar.errors import S3Error

        primary = _RecordingBackend(raises=S3Error("boom"))
        fallback = _RecordingBackend()
        wrapper = LayeredBackend(primary, fallback, fallback_on=(S3Error,))

        wrapper.transfer(
            [],
            target_dir=tmp_path,
            options=_opts(),
            verify=VerifyPolicy(),
            retry=RetryPolicy.default().with_attempts(0),
        )
        assert len(primary.calls) == 1
        assert len(fallback.calls) == 1

    def test_non_matching_exception_propagates(self, tmp_path: Path) -> None:
        from nemar.errors import DataLadError, S3Error

        # Wrap a primary that raises DataLadError, but only catch S3Error.
        primary = _RecordingBackend(raises=DataLadError("not caught"))
        fallback = _RecordingBackend()
        wrapper = LayeredBackend(primary, fallback, fallback_on=(S3Error,))

        import pytest

        with pytest.raises(DataLadError):
            wrapper.transfer(
                [],
                target_dir=tmp_path,
                options=_opts(),
                verify=VerifyPolicy(),
                retry=RetryPolicy.default().with_attempts(0),
            )
        assert len(fallback.calls) == 0

    def test_default_fallback_on_is_dataladerror(self, tmp_path: Path) -> None:
        """Default preserves the pre-S3 call sites that did not pass the kwarg."""
        from nemar.errors import DataLadError

        primary = _RecordingBackend(raises=DataLadError("boom"))
        fallback = _RecordingBackend()
        wrapper = LayeredBackend(primary, fallback)

        wrapper.transfer(
            [],
            target_dir=tmp_path,
            options=_opts(),
            verify=VerifyPolicy(),
            retry=RetryPolicy.default().with_attempts(0),
        )
        assert len(fallback.calls) == 1


# ---------------------------------------------------------------------------
# select_backend — chain shape per (downloader, datalad_url)
# ---------------------------------------------------------------------------


class TestSelectBackendChainShape:
    """The chain is a list-builder, not a branching pyramid.

    Each case below pins one (downloader, datalad_url) → backend
    shape. Adding a new layer in the future should only add new
    rows; existing rows must not drift.
    """

    def _select(self, *, backend: str, datalad_url: str | None):
        return select_backend(
            TransferOptions(
                backend=backend, max_concurrent_downloads=1, stream_timeout=60.0
            ),
            dataset="nm000132",
            datalad_url=datalad_url,
        )

    def test_python_returns_bare_python_backend(self) -> None:
        result = self._select(backend="python", datalad_url=None)
        assert isinstance(result, PythonBackend)

    def test_s3_returns_bare_s3_backend(self) -> None:
        result = self._select(backend="s3", datalad_url=None)
        assert isinstance(result, S3Backend)

    def test_auto_with_datalad_url_returns_three_layer_chain(self) -> None:
        # auto + datalad_url → S3 → (DataLad → HTTPS)
        chain = self._select(backend="auto", datalad_url="https://x/datalad")
        assert isinstance(chain, LayeredBackend)
        assert isinstance(chain.primary, S3Backend)
        assert chain.fallback_on == (S3Error,)

        inner = chain.fallback
        assert isinstance(inner, LayeredBackend)
        assert isinstance(inner.primary, DataLadBackend)
        assert inner.fallback_on == (DataLadError,)
        assert isinstance(inner.fallback, PythonBackend)

    def test_auto_without_datalad_url_returns_two_layer_chain(self) -> None:
        chain = self._select(backend="auto", datalad_url=None)
        assert isinstance(chain, LayeredBackend)
        assert isinstance(chain.primary, S3Backend)
        assert chain.fallback_on == (S3Error,)
        assert isinstance(chain.fallback, PythonBackend)

    def test_datalad_with_url_returns_datalad_over_https(self) -> None:
        chain = self._select(backend="datalad", datalad_url="https://x/datalad")
        assert isinstance(chain, LayeredBackend)
        assert isinstance(chain.primary, DataLadBackend)
        assert chain.fallback_on == (DataLadError,)
        assert isinstance(chain.fallback, PythonBackend)

    def test_datalad_without_url_degrades_to_python(self) -> None:
        result = self._select(backend="datalad", datalad_url=None)
        assert isinstance(result, PythonBackend)
