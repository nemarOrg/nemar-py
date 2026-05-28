"""Network-chaos integration tests.

Each test interposes a :mod:`toxiproxy` proxy between the nemar-py
client and the local HTTPS fixture server, then exercises the client
under one specific failure mode. The point is to confirm that the
retry / Range-resume / timeout machinery handles real-world TCP
nastiness — not just the HTTP-status-code mocks the rest of the
integration suite drives.

Gated by ``NEMAR_CHAOS=1``; requires ``toxiproxy-server`` on PATH.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import httpx
import pytest

from nemar._models import DatasetFile
from nemar._retry import RetryPolicy
from nemar._verification import VerifyPolicy, VerifyResult, check
from nemar.transfer import download_one
from tests.fixtures.toxiproxy import ChaosHandle

pytestmark = pytest.mark.integration


def _publish_single_file(
    nemar_endpoint, *, path: str, content: bytes
) -> DatasetFile:
    """Publish one file via the HTTPS fixture and return its DatasetFile.

    Uses the same factory pattern as the rest of the integration suite
    so the chaos tests share fixture infrastructure.
    """
    from tests.fixtures.factories import (
        make_index,
        make_manifest_entry,
        make_manifest_list,
    )

    index = make_index(dataset="nm000132")
    entry = make_manifest_entry(path=path, content=content)
    manifest = make_manifest_list([entry])
    nemar_endpoint.publish(
        "nm000132",
        index=index,
        manifest=manifest,
        files={f"v1.0.0/{path}": content},
    )
    return DatasetFile(
        path=path,
        url=f"{nemar_endpoint.base_url}nm000132/v1.0.0/{path}",
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _file_via_chaos(file: DatasetFile, chaos: ChaosHandle) -> DatasetFile:
    """Rewrite a DatasetFile so its ``url`` traverses the chaos proxy.

    The fixture publishes at ``https://localhost:<fixture_port>/...``;
    we swap the port for the chaos listener so every byte the client
    fetches goes through toxiproxy first.
    """
    parsed = httpx.URL(file.url)
    rewritten = parsed.copy_with(port=chaos.listen_port)
    return DatasetFile(
        path=file.path,
        url=str(rewritten),
        size=file.size,
        sha256=file.sha256,
    )


# ---------------------------------------------------------------------------
# Scenario 1 — Latency tolerance
# ---------------------------------------------------------------------------


def test_download_succeeds_with_high_latency(
    chaos_proxy: ChaosHandle, nemar_endpoint, tmp_path: Path
) -> None:
    """200 ms of artificial latency on every byte must not break a small fetch.

    Pins that ``download_one`` does not have an implicit
    sub-second timeout sneaking in below the configured stream
    timeout. Real-world satellite / overseas connections regularly
    show this kind of RTT.
    """
    content = b"latency tolerance payload" * 100
    file = _publish_single_file(
        nemar_endpoint, path="data/latency.bin", content=content
    )
    chaos_proxy.add_toxic(
        name="rtt-200ms",
        type="latency",
        attributes={"latency": 200},
    )

    via_chaos = _file_via_chaos(file, chaos_proxy)
    target = tmp_path / file.path
    result = download_one(via_chaos, target)

    assert result is VerifyResult.OK
    assert target.read_bytes() == content


# ---------------------------------------------------------------------------
# Scenario 2 — Throughput throttle
# ---------------------------------------------------------------------------


def test_download_completes_under_bandwidth_throttle(
    chaos_proxy: ChaosHandle, nemar_endpoint, tmp_path: Path
) -> None:
    """A 16 KB/s downstream throttle still completes a 16 KiB file cleanly.

    Pins that the stream timeout governs per-chunk, not whole-body,
    so a long-running slow transfer does not get killed
    prematurely. The wall-clock budget here is ~1 s plus jitter;
    we give it 30 s.
    """
    content = b"slow-but-steady" * 1024  # ~15 KiB, small enough to ride 16 KB/s
    file = _publish_single_file(
        nemar_endpoint, path="data/throttle.bin", content=content
    )
    chaos_proxy.add_toxic(
        name="throttle",
        type="bandwidth",
        attributes={"rate": 16},  # KB/s
    )

    via_chaos = _file_via_chaos(file, chaos_proxy)
    target = tmp_path / file.path
    started = time.monotonic()
    result = download_one(via_chaos, target)
    elapsed = time.monotonic() - started

    assert result is VerifyResult.OK
    assert target.read_bytes() == content
    # Sanity: the throttle was actually active. A fast connection would
    # have finished in milliseconds, not seconds.
    assert elapsed >= 0.3, f"transfer too fast ({elapsed:.2f}s) for throttle"


# ---------------------------------------------------------------------------
# Scenario 3 — Mid-stream drop, then retry succeeds
# ---------------------------------------------------------------------------


def test_partial_bytes_survive_drop_and_resume_completes(
    chaos_proxy: ChaosHandle, nemar_endpoint, tmp_path: Path
) -> None:
    """A drop mid-stream + clean retry round-trip reconstitutes the file.

    Two-phase contract:

    1. With a ``limit_data`` toxic cutting every connection at
       1024 bytes, the first ``download_one`` (zero retries)
       surfaces a :class:`TransferError`. Partial bytes are on
       disk because the per-attempt writer flushes as it goes.
    2. After we clear the toxic, a second ``download_one`` call
       sees the partial, issues ``Range: bytes=<N>-``, and the
       fixture server serves the rest. The final file matches the
       manifest sha256.

    This pins two separate guarantees: (a) drops surface as a
    typed ``TransferError`` rather than corrupting silently, and
    (b) the resume contract works end-to-end against a real TCP
    cut, not just an HTTP-status-code mock.
    """
    # 100 KiB body, cut at ~32 KiB of TLS-encrypted transmit. That's
    # comfortably past the handshake + headers so we are guaranteed to
    # have written *some* body bytes before the cut.
    content = b"R" * (100 * 1024)
    file = _publish_single_file(
        nemar_endpoint, path="data/drop.bin", content=content
    )
    nemar_endpoint.serve_with_range(f"/nm000132/v1.0.0/{file.path}")
    chaos_proxy.add_toxic(
        name="cut-32k",
        type="limit_data",
        attributes={"bytes": 32768},
    )
    via_chaos = _file_via_chaos(file, chaos_proxy)
    target = tmp_path / file.path

    # Phase 1: drop fails fast (no retries) → TransferError, partial on disk.
    from nemar._errors import TransferError

    with pytest.raises(TransferError):
        download_one(
            via_chaos,
            target,
            retry=RetryPolicy.default().with_attempts(0),
        )
    on_disk = target.stat().st_size
    assert 0 < on_disk < file.size, (
        f"expected a partial file 0 < N < {file.size}, got {on_disk} B"
    )

    # Phase 2: clear the toxic and resume.
    chaos_proxy.proxy.destroy_toxic("cut-32k")
    result = download_one(via_chaos, target)
    assert result is VerifyResult.OK
    assert target.read_bytes() == content


# ---------------------------------------------------------------------------
# Scenario 4 — Verifier guards bad bytes even through a happy network
# ---------------------------------------------------------------------------


def test_post_transfer_verifier_catches_corruption(
    chaos_proxy: ChaosHandle, nemar_endpoint, tmp_path: Path
) -> None:
    """A toxic that flips bytes en route must trigger a hash mismatch.

    Pins the contract that the post-transfer verify is the final
    word: even if a buggy transport / bad CDN smuggled in wrong
    bytes (which network toxics can simulate), ``check(...)`` will
    flag it.

    Toxiproxy does not have a built-in byte-flip toxic, so we
    cheat: write content A to the fixture, claim its size+sha256 as
    if it were content B (random), and confirm the verifier
    rejects with HASH_MISMATCH.
    """
    real_content = b"genuine bytes" * 64
    fake_sha = hashlib.sha256(b"impostor").hexdigest()
    file = _publish_single_file(
        nemar_endpoint, path="data/tamper.bin", content=real_content
    )
    tampered = DatasetFile(
        path=file.path,
        url=file.url.replace(
            str(httpx.URL(file.url).port), str(chaos_proxy.listen_port)
        ),
        size=file.size,
        sha256=fake_sha,
    )

    target = tmp_path / file.path
    download_one(tampered, target)  # writes bytes, returns
    result = check(tampered, target, VerifyPolicy())
    assert result is VerifyResult.HASH_MISMATCH
