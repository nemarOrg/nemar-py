"""Unit tests for the transfer backend module.

Covers the seam between the orchestrator and the bytes-on-the-wire phase:

* :func:`select_backend` — policy- and PATH-driven backend resolution.
* :class:`Aria2Backend` — input-file format, split×concurrency cap,
  error normalization, SSL_CERT_FILE passthrough.
* :class:`PythonBackend` — minimal end-to-end pipe over a
  ``MockTransport`` so the shared-client (G) path is exercised.

These tests are the contract that lets future work move the backends
into adapter subpackages without touching the orchestrator.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import httpx
import pytest

from nemar import _transfer
from nemar._models import DatasetFile
from nemar._request import TransferOptions
from nemar._retry import RetryPolicy
from nemar._transfer import (
    Aria2Backend,
    PythonBackend,
    select_backend,
)
from nemar._verification import VerifyPolicy


def _make_options(
    *,
    backend: str = "python",
    max_concurrent_downloads: int = 1,
    stream_timeout: float = 60.0,
    aria2_timeout: float | None = None,
) -> TransferOptions:
    return TransferOptions(
        backend=backend,
        max_concurrent_downloads=max_concurrent_downloads,
        stream_timeout=stream_timeout,
        aria2_timeout=aria2_timeout,
    )


class TestSelectBackend:
    """Backend resolution by request × aria2c-on-PATH."""

    def test_auto_picks_aria2_when_present(self, monkeypatch) -> None:
        monkeypatch.setattr(
            _transfer.shutil, "which", lambda name: "/usr/bin/aria2c"
        )
        backend = select_backend(_make_options(backend="auto"))
        assert isinstance(backend, Aria2Backend)

    def test_auto_falls_back_to_python_when_aria2_missing(self, monkeypatch) -> None:
        monkeypatch.setattr(_transfer.shutil, "which", lambda name: None)
        backend = select_backend(_make_options(backend="auto"))
        assert isinstance(backend, PythonBackend)

    def test_explicit_aria2_without_aria2c_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(_transfer.shutil, "which", lambda name: None)
        with pytest.raises(RuntimeError, match="requires aria2c on PATH"):
            select_backend(_make_options(backend="aria2"))

    def test_explicit_aria2_with_aria2c_returns_aria2_backend(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            _transfer.shutil, "which", lambda name: "/usr/bin/aria2c"
        )
        backend = select_backend(_make_options(backend="aria2"))
        assert isinstance(backend, Aria2Backend)

    def test_explicit_python_returns_python_backend(self, monkeypatch) -> None:
        # Even when aria2c is present, explicit "python" must return PythonBackend.
        monkeypatch.setattr(
            _transfer.shutil, "which", lambda name: "/usr/bin/aria2c"
        )
        backend = select_backend(_make_options(backend="python"))
        assert isinstance(backend, PythonBackend)


class TestAria2Checksum:
    """The static checksum helper preserves manifest-hash preference."""

    @pytest.mark.parametrize(
        ("file", "expected"),
        [
            pytest.param(
                DatasetFile(
                    path="x",
                    url="https://data.nemar.org/x",
                    sha256="abc",
                    md5="def",
                ),
                "sha-256=abc",
                id="sha256-preferred",
            ),
            pytest.param(
                DatasetFile(path="x", url="https://data.nemar.org/x", md5="def"),
                "md5=def",
                id="md5",
            ),
            pytest.param(
                DatasetFile(path="x", url="https://data.nemar.org/x"),
                None,
                id="no-checksum",
            ),
        ],
    )
    def test_aria2_checksum_returns_expected(self, file, expected) -> None:
        assert _transfer._aria2_checksum(file) == expected


class TestAria2BackendInputFile:
    """The aria2 adapter writes manifest-driven input files in the expected shape."""

    def test_input_file_lines_match_aria2_grammar(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        seen: dict[str, object] = {}

        def run(cmd, check, **kwargs):
            seen["cmd"] = cmd
            seen["kwargs"] = kwargs
            input_arg = next(arg for arg in cmd if arg.startswith("--input-file="))
            input_path = Path(input_arg.split("=", 1)[1])
            seen["input"] = input_path.read_text(encoding="utf-8")
            for line in seen["input"].splitlines():
                if line.startswith("  out="):
                    (tmp_path / line.split("=", 1)[1]).write_text(
                        "ok", encoding="utf-8"
                    )

        monkeypatch.setattr(_transfer.subprocess, "run", run)

        backend = Aria2Backend()
        backend.transfer(
            [
                DatasetFile(
                    path="participants.tsv",
                    url="https://data.nemar.org/participants.tsv",
                    size=2,
                    sha256=hashlib.sha256(b"ok").hexdigest(),
                )
            ],
            target_dir=tmp_path,
            options=_make_options(
                backend="aria2",
                max_concurrent_downloads=4,
            ),
            verify=VerifyPolicy(verify_size=True, verify_hash=True),
            retry=RetryPolicy.default().with_attempts(2),
        )

        assert "https://data.nemar.org/participants.tsv" in seen["input"]
        assert "  out=participants.tsv" in seen["input"]
        assert "  checksum=sha-256=" in seen["input"]
        assert "--max-tries=3" in seen["cmd"]


class TestAria2BackendSplitCap:
    """The split×concurrency budget is preserved across all configurations."""

    @pytest.mark.parametrize(
        ("max_concurrent_downloads", "expected_split"),
        [
            pytest.param(1, 16, id="1-download-max-split"),
            pytest.param(2, 16, id="2-downloads-split-16"),
            pytest.param(4, 8, id="4-downloads-split-8"),
            pytest.param(8, 4, id="8-downloads-split-4"),
            pytest.param(16, 2, id="16-downloads-split-2"),
            pytest.param(32, 1, id="32-downloads-split-1"),
            pytest.param(64, 1, id="64-downloads-clamped-to-1"),
        ],
    )
    def test_split_cap(
        self,
        monkeypatch,
        tmp_path: Path,
        max_concurrent_downloads,
        expected_split,
    ) -> None:
        seen: dict[str, object] = {}

        def run(cmd, check, **kwargs):
            seen["cmd"] = cmd
            input_arg = next(arg for arg in cmd if arg.startswith("--input-file="))
            input_path = Path(input_arg.split("=", 1)[1])
            for line in input_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("  out="):
                    (tmp_path / line.split("=", 1)[1]).write_text(
                        "ok", encoding="utf-8"
                    )

        monkeypatch.setattr(_transfer.subprocess, "run", run)

        backend = Aria2Backend()
        backend.transfer(
            [
                DatasetFile(
                    path="participants.tsv",
                    url="https://data.nemar.org/participants.tsv",
                )
            ],
            target_dir=tmp_path,
            options=_make_options(
                backend="aria2",
                max_concurrent_downloads=max_concurrent_downloads,
            ),
            verify=VerifyPolicy(verify_size=False, verify_hash=False),
            retry=RetryPolicy.default().with_attempts(0),
        )

        split_arg = next(
            (arg for arg in seen["cmd"] if arg.startswith("--split=")), None
        )
        assert split_arg == f"--split={expected_split}", (
            f"Expected --split={expected_split}, got {split_arg}"
        )


class TestAria2BackendErrorPaths:
    """Subprocess failures surface as RuntimeError with the historical wording."""

    def test_subprocess_failure_becomes_runtime_error(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        def run(cmd, check, **kwargs):
            raise subprocess.CalledProcessError(1, cmd)

        monkeypatch.setattr(_transfer.subprocess, "run", run)

        backend = Aria2Backend()
        with pytest.raises(RuntimeError, match="aria2c failed"):
            backend.transfer(
                [DatasetFile(path="x", url="https://data.nemar.org/x")],
                target_dir=tmp_path,
                options=_make_options(
                    backend="aria2", max_concurrent_downloads=1
                ),
                verify=VerifyPolicy(verify_size=False, verify_hash=False),
                retry=RetryPolicy.default().with_attempts(0),
            )

    def test_subprocess_timeout_reports_seconds(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        def run(cmd, check, **kwargs):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=1.5)

        monkeypatch.setattr(_transfer.subprocess, "run", run)

        backend = Aria2Backend()
        with pytest.raises(RuntimeError, match="aria2c timed out after"):
            backend.transfer(
                [DatasetFile(path="x", url="https://data.nemar.org/x")],
                target_dir=tmp_path,
                options=_make_options(
                    backend="aria2",
                    max_concurrent_downloads=1,
                    aria2_timeout=1.5,
                ),
                verify=VerifyPolicy(verify_size=False, verify_hash=False),
                retry=RetryPolicy.default().with_attempts(0),
            )


class TestAria2ExtraArgsHook:
    """Test fixtures inject extra aria2 flags via the module-level hook."""

    def test_extra_args_are_appended_to_command(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        seen: dict[str, object] = {}

        def run(cmd, check, **kwargs):
            seen["cmd"] = cmd
            input_arg = next(arg for arg in cmd if arg.startswith("--input-file="))
            input_path = Path(input_arg.split("=", 1)[1])
            for line in input_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("  out="):
                    (tmp_path / line.split("=", 1)[1]).write_text(
                        "ok", encoding="utf-8"
                    )

        monkeypatch.setattr(_transfer.subprocess, "run", run)
        monkeypatch.setattr(
            _transfer, "_ARIA2_EXTRA_ARGS", ["--check-certificate=false"]
        )

        backend = Aria2Backend()
        backend.transfer(
            [DatasetFile(path="x", url="https://data.nemar.org/x")],
            target_dir=tmp_path,
            options=_make_options(
                backend="aria2", max_concurrent_downloads=1
            ),
            verify=VerifyPolicy(verify_size=False, verify_hash=False),
            retry=RetryPolicy.default().with_attempts(0),
        )

        assert "--check-certificate=false" in seen["cmd"]


class TestAria2BackendSslCertPassthrough:
    """SSL_CERT_FILE env is forwarded as --ca-certificate when set."""

    def test_ssl_cert_file_becomes_ca_certificate(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        seen: dict[str, object] = {}

        def run(cmd, check, **kwargs):
            seen["cmd"] = cmd
            input_arg = next(arg for arg in cmd if arg.startswith("--input-file="))
            input_path = Path(input_arg.split("=", 1)[1])
            for line in input_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("  out="):
                    (tmp_path / line.split("=", 1)[1]).write_text(
                        "ok", encoding="utf-8"
                    )

        monkeypatch.setattr(_transfer.subprocess, "run", run)
        monkeypatch.setenv("SSL_CERT_FILE", "/tmp/test-ca.pem")

        backend = Aria2Backend()
        backend.transfer(
            [DatasetFile(path="x", url="https://data.nemar.org/x")],
            target_dir=tmp_path,
            options=_make_options(
                backend="aria2", max_concurrent_downloads=1
            ),
            verify=VerifyPolicy(verify_size=False, verify_hash=False),
            retry=RetryPolicy.default().with_attempts(0),
        )

        assert "--ca-certificate=/tmp/test-ca.pem" in seen["cmd"]

    def test_ssl_cert_file_unset_omits_ca_certificate(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        seen: dict[str, object] = {}

        def run(cmd, check, **kwargs):
            seen["cmd"] = cmd
            input_arg = next(arg for arg in cmd if arg.startswith("--input-file="))
            input_path = Path(input_arg.split("=", 1)[1])
            for line in input_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("  out="):
                    (tmp_path / line.split("=", 1)[1]).write_text(
                        "ok", encoding="utf-8"
                    )

        monkeypatch.setattr(_transfer.subprocess, "run", run)
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)

        backend = Aria2Backend()
        backend.transfer(
            [DatasetFile(path="x", url="https://data.nemar.org/x")],
            target_dir=tmp_path,
            options=_make_options(
                backend="aria2", max_concurrent_downloads=1
            ),
            verify=VerifyPolicy(verify_size=False, verify_hash=False),
            retry=RetryPolicy.default().with_attempts(0),
        )

        ca_flags = [
            arg for arg in seen["cmd"] if str(arg).startswith("--ca-certificate=")
        ]
        assert ca_flags == []


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
    """Both adapters honour the same public interface."""

    def test_adapters_expose_transfer_method(self) -> None:
        for cls in (Aria2Backend, PythonBackend):
            assert hasattr(cls, "transfer")
            assert callable(cls.transfer)
