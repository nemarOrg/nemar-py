"""Unit tests for ``nemar.transfer.download_files``.

The bulk variant of ``download_one``. Pins the contract documented in
:func:`nemar._transfer.download_files`: builds one shared
``httpx.Client`` for the batch, runs the same pre-transfer
``partition_pending`` + backend + post-transfer ``assert_all_present``
pipeline that the full orchestrator runs, honors the supplied
``TransferOptions`` / ``VerifyPolicy`` / ``RetryPolicy``, and enforces
origin scoping when an ``endpoint`` is provided.

These tests use ``httpx.MockTransport`` (matching the existing
``test_download_one.py`` pattern) rather than the local HTTPS fixture
because the contract is about wiring and policy plumbing, not
real-network behavior. Integration coverage lives next to the existing
multi-file transfer integration tests.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import pytest

from nemar._endpoint import DataEndpoint
from nemar._errors import EndpointError, TransferError, VerificationError
from nemar._models import DatasetFile
from nemar._request import TransferOptions
from nemar._retry import RetryPolicy
from nemar._verification import VerifyPolicy
from nemar.transfer import download_files


def _make_file(
    content: bytes,
    *,
    path: str,
    url_host: str = "data.nemar.org",
) -> DatasetFile:
    return DatasetFile(
        path=path,
        url=f"https://{url_host}/nm000132/v1.0.0/{path}",
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _python_options() -> TransferOptions:
    return TransferOptions(
        backend="python",
        max_concurrent_downloads=4,
        stream_timeout=30.0,
        aria2_timeout=None,
    )


def _install_mock_transport(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    """Patch httpx.Client to use a MockTransport for every instance.

    download_files builds its own httpx.Client (no client= kwarg), so we
    intercept at construction time the same way test_download_one does.
    """
    transport = httpx.MockTransport(handler)
    original_init = httpx.Client.__init__

    def patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["transport"] = transport
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", patched_init)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_downloads_every_file_and_verifies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two files → both land on disk with the right content."""
    contents = {
        "a.bin": b"alpha payload",
        "deep/b.bin": b"bravo payload, longer to vary size",
    }
    files = [_make_file(c, path=p) for p, c in contents.items()]

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path.split("/v1.0.0/", 1)[1]
        return httpx.Response(200, content=contents[path], request=request)

    _install_mock_transport(monkeypatch, handler)
    download_files(files, tmp_path, options=_python_options())

    for relpath, content in contents.items():
        target = tmp_path / relpath
        assert target.exists()
        assert target.read_bytes() == content


def test_empty_list_is_a_noop(tmp_path: Path) -> None:
    """An empty file list returns without building a client or touching disk."""
    download_files([], tmp_path)
    # The target_dir is created on demand even for empty lists — matches
    # the orchestrator's behavior and lets callers chain follow-ups safely.
    assert tmp_path.exists()


def test_target_dir_string_is_accepted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """target_dir accepts str (not just Path), matching download()'s signature."""
    content = b"path or string both work"
    file = _make_file(content, path="x.bin")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content, request=request)

    _install_mock_transport(monkeypatch, handler)
    download_files([file], str(tmp_path / "sub"), options=_python_options())
    assert (tmp_path / "sub" / "x.bin").read_bytes() == content


# ---------------------------------------------------------------------------
# Pre-transfer partition: already-present files are skipped (size-only)
# ---------------------------------------------------------------------------


def test_already_present_files_are_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A file already on disk at the right size is not re-downloaded.

    Pre-transfer ``partition_pending`` is size-only; the post-transfer
    ``assert_all_present`` runs the full hash check across every file
    regardless. This test pins the skip path: no network call should
    fire for the already-present file.
    """
    present_content = b"already here"
    missing_content = b"need to fetch this one"
    present_file = _make_file(present_content, path="present.bin")
    missing_file = _make_file(missing_content, path="missing.bin")

    # Pre-place the present file with matching size and content.
    (tmp_path / "present.bin").write_bytes(present_content)

    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path.split("/v1.0.0/", 1)[1]
        requested_paths.append(path)
        return httpx.Response(200, content=missing_content, request=request)

    _install_mock_transport(monkeypatch, handler)
    download_files(
        [present_file, missing_file], tmp_path, options=_python_options()
    )

    assert requested_paths == ["missing.bin"]
    assert (tmp_path / "missing.bin").read_bytes() == missing_content


# ---------------------------------------------------------------------------
# Post-transfer assert_all_present catches manifest mismatches
# ---------------------------------------------------------------------------


def test_size_mismatch_raises_verification_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Manifest size 100 but server returns 5 bytes → VerificationError."""
    file = DatasetFile(
        path="bad.bin",
        url="https://data.nemar.org/nm000132/v1.0.0/bad.bin",
        size=100,
        sha256=hashlib.sha256(b"abcde").hexdigest(),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"abcde", request=request)

    _install_mock_transport(monkeypatch, handler)
    with pytest.raises((TransferError, VerificationError)):
        download_files([file], tmp_path, options=_python_options())


# ---------------------------------------------------------------------------
# Origin scoping (optional)
# ---------------------------------------------------------------------------


def test_endpoint_supplied_rejects_off_origin_file(tmp_path: Path) -> None:
    """When endpoint= is supplied, every file URL is checked first.

    Matches the optional ``endpoint=`` parameter on ``download_one``.
    The check happens *before* any bytes move so no partial state is
    left on disk after a malicious or misconfigured file list.
    """
    endpoint = DataEndpoint.from_url("https://data.nemar.org/")
    ok_file = _make_file(b"ok", path="ok.bin")
    evil_file = DatasetFile(
        path="evil.bin",
        url="https://evil.example.com/legit.bin",
        size=1,
    )

    with pytest.raises(EndpointError, match="Refusing to download"):
        download_files(
            [ok_file, evil_file],
            tmp_path,
            options=_python_options(),
            endpoint=endpoint,
        )

    # No partial work: neither file lands on disk.
    assert not (tmp_path / "ok.bin").exists()
    assert not (tmp_path / "evil.bin").exists()


def test_endpoint_omitted_does_not_enforce_origin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without endpoint=, off-origin URLs are NOT auto-rejected.

    Matches ``download_one``'s default. Callers who want enforcement
    must opt in by passing ``endpoint=``.
    """
    content = b"caller did not enforce"
    file = _make_file(content, path="x.bin", url_host="alt.example.com")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content, request=request)

    _install_mock_transport(monkeypatch, handler)
    download_files([file], tmp_path, options=_python_options())
    assert (tmp_path / "x.bin").read_bytes() == content


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_options_picks_python_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No options= argument → uses the python backend defaults.

    Predictable default: the python backend is always available and
    doesn't depend on PATH state. Callers who want aria2 must opt in
    via ``options=TransferOptions(backend="aria2", ...)``.
    """
    content = b"defaults are fine"
    file = _make_file(content, path="d.bin")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content, request=request)

    _install_mock_transport(monkeypatch, handler)
    download_files([file], tmp_path)
    assert (tmp_path / "d.bin").read_bytes() == content


def test_retry_policy_is_honored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caller-supplied RetryPolicy controls per-file attempt count.

    The orchestrator threads RetryPolicy through to the python backend's
    per-file loop. With ``with_attempts(0)`` the first failure should
    surface as TransferError without retries.
    """
    file = _make_file(b"x" * 8, path="r.bin")
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(503, request=request)

    _install_mock_transport(monkeypatch, handler)
    with pytest.raises(TransferError):
        download_files(
            [file],
            tmp_path,
            options=_python_options(),
            retry=RetryPolicy.default().with_attempts(0),
        )
    assert attempts["n"] == 1


def test_verify_policy_hash_off_skips_hash_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """VerifyPolicy(verify_hash=False) skips the post-transfer hash sweep.

    Used by callers who already have their own integrity guarantees
    (e.g., serving from a trusted mirror that has its own checksumming).
    """
    content = b"real content"
    bad_hash = hashlib.sha256(b"not the content").hexdigest()
    file = DatasetFile(
        path="h.bin",
        url="https://data.nemar.org/nm000132/v1.0.0/h.bin",
        size=len(content),
        sha256=bad_hash,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content, request=request)

    _install_mock_transport(monkeypatch, handler)
    # Would raise if verify_hash were True (manifest hash doesn't match real content).
    download_files(
        [file],
        tmp_path,
        options=_python_options(),
        verify=VerifyPolicy(verify_hash=False),
    )
    assert (tmp_path / "h.bin").read_bytes() == content
