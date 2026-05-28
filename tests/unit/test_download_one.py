"""Unit tests for ``download_one``.

The public per-file streaming primitive. These tests pin the contract
documented for ``download_one``: it builds its own short-lived client by
default, reuses a caller-supplied client when provided, returns the
post-transfer ``VerifyResult``, raises only on irrecoverable transport
failure, retries per the supplied ``RetryPolicy``, and creates the
target path's parent directory.

The tests use ``httpx.MockTransport`` rather than the local HTTPS
fixture server because the contract is about wiring, not real-network
behavior; the integration suite covers the HTTPS smoke and resume
paths separately.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import pytest

from nemar import download_one
from nemar._models import DatasetFile
from nemar._retry import RetryPolicy
from nemar._verification import VerifyPolicy, VerifyResult


def _make_file(
    content: bytes,
    *,
    path: str = "data/sample.bin",
    size: int | None = None,
    sha256: str | None = None,
) -> DatasetFile:
    return DatasetFile(
        path=path,
        url=f"https://data.nemar.org/nm000132/v1.0.0/{path}",
        size=len(content) if size is None else size,
        sha256=hashlib.sha256(content).hexdigest() if sha256 is None else sha256,
    )


def test_happy_path_writes_file_and_returns_ok(monkeypatch, tmp_path: Path) -> None:
    """200 OK + matching sha256 → file written, VerifyResult.OK returned.

    No client is supplied. ``download_one`` must build its own short-lived
    client and tear it down by the time it returns. We probe the wiring
    by patching ``httpx.Client.__init__`` to attach a MockTransport.
    """
    content = b"hello nemar download_one"
    file = _make_file(content)
    target = tmp_path / "subdir" / "sample.bin"

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=content, request=request)

    transport = httpx.MockTransport(handler)
    original_init = httpx.Client.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = transport
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", patched_init)

    result = download_one(file, target)

    assert result is VerifyResult.OK
    assert target.exists()
    assert target.read_bytes() == content
    # Parent directory was created.
    assert target.parent.is_dir()
    # Exactly one network round-trip.
    assert len(seen) == 1


def test_caller_supplied_client_is_reused(tmp_path: Path) -> None:
    """When ``client`` is provided, ``download_one`` MUST use that client.

    We pass a client whose only transport is a MockTransport. If the
    function ignored the caller's client and built a new one (or fell
    back to ``httpx.stream``), the request would never reach the mock
    handler. Counting calls into the mock proves the caller's client
    carried the request.
    """
    content = b"shared client carries the bytes"
    file = _make_file(content, path="dir/shared.bin")
    target = tmp_path / "shared.bin"

    request_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        request_count["n"] += 1
        return httpx.Response(200, content=content, request=request)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        result = download_one(file, target, client=client)
        # The caller's client must still be usable after download_one
        # returns (one-shot does not close caller-owned clients).
        assert not client.is_closed

    assert result is VerifyResult.OK
    assert target.read_bytes() == content
    assert request_count["n"] == 1


def test_default_client_is_built_and_torn_down(monkeypatch, tmp_path: Path) -> None:
    """``client=None`` (default) must build a short-lived client.

    The function MUST NOT require the caller to supply a client for the
    one-shot case. We assert the function constructed at least one
    ``httpx.Client`` during the call (covered by patching
    ``httpx.Client.__init__``).
    """
    content = b"default client works"
    file = _make_file(content, path="default.bin")
    target = tmp_path / "default.bin"

    init_calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content, request=request)

    transport = httpx.MockTransport(handler)
    original_init = httpx.Client.__init__

    def patched_init(self, *args, **kwargs):
        init_calls.append(1)
        kwargs["transport"] = transport
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", patched_init)

    result = download_one(file, target)

    assert result is VerifyResult.OK
    assert target.read_bytes() == content
    # The function created at least one Client because we did not pass one.
    assert len(init_calls) >= 1


def test_size_mismatch_returns_verify_result(monkeypatch, tmp_path: Path) -> None:
    """A right-bytes-wrong-size response must surface as a VerifyResult.

    The manifest declares ``size = len(content) + 100`` (a deliberate
    mismatch). The server delivers the real content. After transfer the
    file is on disk but its size does not match the manifest; verify
    must report SIZE_MISMATCH and ``download_one`` MUST return that
    result rather than raise.
    """
    content = b"short content"
    file = _make_file(
        content,
        path="mismatch.bin",
        size=len(content) + 100,
        sha256=hashlib.sha256(content).hexdigest(),
    )
    target = tmp_path / "mismatch.bin"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content, request=request)

    transport = httpx.MockTransport(handler)
    original_init = httpx.Client.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = transport
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", patched_init)

    result = download_one(file, target)

    assert result is VerifyResult.SIZE_MISMATCH
    # The bytes the server delivered still land on disk; the verify
    # outcome is a report, not a roll-back.
    assert target.read_bytes() == content


def test_http_500_retried_then_succeeds(tmp_path: Path) -> None:
    """A 5xx response is retried per the supplied ``RetryPolicy``.

    ``RetryPolicy.default().with_attempts(2)`` permits two retries (three
    attempts total). Server returns 500 on the first call and 200 on the
    second. The result must be VerifyResult.OK.
    """
    content = b"second time succeeds"
    file = _make_file(content, path="flaky.bin")
    target = tmp_path / "flaky.bin"

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, content=b"oops", request=request)
        return httpx.Response(200, content=content, request=request)

    transport = httpx.MockTransport(handler)
    # Build the policy with a very small backoff so the test stays fast.
    base_policy = RetryPolicy.default()
    fast_policy = RetryPolicy(
        max_attempts=3,
        base_backoff=0.0,
        max_backoff=0.0,
        retryable_status=base_policy.retryable_status,
        retryable_exceptions=base_policy.retryable_exceptions,
    )
    with httpx.Client(transport=transport) as client:
        result = download_one(
            file,
            target,
            client=client,
            retry=fast_policy,
        )

    assert result is VerifyResult.OK
    assert calls["n"] == 2
    assert target.read_bytes() == content


def test_no_retries_raises_runtime_error_on_irrecoverable_failure(
    tmp_path: Path,
) -> None:
    """``retry=RetryPolicy.default().with_attempts(0)`` means zero retries.

    With ``with_attempts(0)`` -> ``max_attempts=1``, the very first 500
    must raise ``RuntimeError`` rather than retry silently.
    """
    content = b"never delivered"
    file = _make_file(content, path="hard-fail.bin")
    target = tmp_path / "hard-fail.bin"

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, content=b"down", request=request)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        with pytest.raises(RuntimeError, match="hard-fail.bin"):
            download_one(
                file,
                target,
                client=client,
                retry=RetryPolicy.default().with_attempts(0),
            )

    # Exactly one attempt -- no silent retries.
    assert calls["n"] == 1


def test_verify_policy_is_honored(tmp_path: Path) -> None:
    """A custom ``VerifyPolicy`` disabling hash check skips hash work.

    When ``verify_hash=False`` the manifest hash is ignored. We prove
    this by providing a deliberately bogus sha256 that would fail hash
    verification under the default policy.
    """
    content = b"hash is not checked here"
    file = _make_file(
        content,
        path="no-hash.bin",
        sha256="0" * 64,  # bogus; would fail under default verify
    )
    target = tmp_path / "no-hash.bin"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content, request=request)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        result = download_one(
            file,
            target,
            client=client,
            verify=VerifyPolicy(verify_size=True, verify_hash=False),
        )

    assert result is VerifyResult.OK
    assert target.read_bytes() == content


def test_parent_directory_is_created_on_demand(tmp_path: Path) -> None:
    """``target_path``'s parent need not exist; download_one mkdirs it."""
    content = b"deep tree mkdir"
    file = _make_file(content, path="deeply/nested/tree.bin")
    target = tmp_path / "a" / "b" / "c" / "tree.bin"
    assert not target.parent.exists()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content, request=request)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        result = download_one(file, target, client=client)

    assert result is VerifyResult.OK
    assert target.parent.is_dir()
    assert target.read_bytes() == content


def test_existing_symlink_at_target_is_replaced_not_followed(
    tmp_path: Path,
) -> None:
    """A pre-existing symlink at ``target_path`` is unlinked before write.

    Regression guard against the DataLad-then-HTTPS interaction: when
    the layered backend's DataLad attempt partially succeeds it can
    leave git-annex symlinks at the manifest paths (broken symlinks
    pointing into ``.git/annex/objects/...``). Without an unlink-first
    guard, ``open(symlink, "wb")`` would follow the symlink and write
    the new content at the annex object path instead of the manifest
    path — leaving the on-disk dataset and the git-annex tree in
    inconsistent states.

    We assert two invariants: the target ends as a regular file with
    the downloaded content, and the original symlink target is NOT
    written to.
    """
    content = b"https path must overwrite the symlink, not follow it"
    file = _make_file(content)
    fake_annex_target = tmp_path / ".not-an-annex" / "object.bin"
    fake_annex_target.parent.mkdir(parents=True)
    # Pre-existing broken symlink at the manifest path. Matches the shape
    # of a git-annex placeholder when the annex object has not yet been
    # materialized.
    target = tmp_path / file.path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(fake_annex_target)
    assert target.is_symlink() and not target.exists()  # broken symlink

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content, request=request)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        result = download_one(file, target, client=client)

    assert result is VerifyResult.OK
    assert not target.is_symlink()  # regular file now
    assert target.read_bytes() == content
    # The annex-shaped target was not written through: nothing materialized
    # at the symlink destination, no accidental annex-tree mutation.
    assert not fake_annex_target.exists()
