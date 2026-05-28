"""Network-chaos integration tests.

Each test interposes a :mod:`toxiproxy` proxy between the nemar-py
client and the local HTTPS fixture server, then exercises the client
under one specific failure mode. The point is to confirm that the
retry / Range-resume / timeout machinery handles real-world TCP
nastiness — not just the HTTP-status-code mocks the rest of the
integration suite drives.

Toxics are added through ``chaostoolkit-toxiproxy``'s functional
helpers (``create_latency_toxic``, ``create_bandwith_degradation_toxic``,
``create_limiter_toxic``), all routed through the per-test
:class:`~tests.fixtures.toxiproxy.ChaosHandle`'s ``proxy_name`` and
shared toxiproxy ``config``.

Gated by ``NEMAR_CHAOS=1``; requires ``toxiproxy-server`` on PATH.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import httpx
import pytest

from nemar._errors import TransferError
from nemar._models import DatasetFile
from nemar._retry import RetryPolicy
from nemar._verification import VerifyPolicy, VerifyResult, check
from nemar.transfer import download_one
from tests.fixtures.factories import (
    make_index,
    make_manifest_entry,
    make_manifest_list,
)
from tests.fixtures.toxiproxy import ChaosHandle

# ``chaostoolkit-toxiproxy`` is only in the dev dependency group; CI's
# minimal install (pytest + httpserver + trustme + hypothesis) does not
# pull it. ``pytest.importorskip`` raises ``pytest.skip`` at module
# load when the package is missing, so the whole module's tests are
# skipped rather than crashing pytest collection on CI.
_chaostoxi_actions = pytest.importorskip(
    "chaostoxi.toxic.actions",
    reason=(
        "chaostoolkit-toxiproxy not installed; install the dev dependency "
        "group to enable chaos tests."
    ),
)
create_bandwith_degradation_toxic = (
    _chaostoxi_actions.create_bandwith_degradation_toxic
)
create_latency_toxic = _chaostoxi_actions.create_latency_toxic
create_limiter_toxic = _chaostoxi_actions.create_limiter_toxic
delete_toxic = _chaostoxi_actions.delete_toxic

pytestmark = pytest.mark.integration


def _publish_single_file(
    nemar_endpoint, *, path: str, content: bytes
) -> DatasetFile:
    """Publish one file via the HTTPS fixture and return its DatasetFile."""
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
    """Rewrite a DatasetFile so its ``url`` traverses the chaos proxy."""
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
    create_latency_toxic(
        for_proxy=chaos_proxy.proxy_name,
        toxic_name="rtt-200ms",
        latency=200,
        configuration=chaos_proxy.config,
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
    prematurely.
    """
    content = b"slow-but-steady" * 1024  # ~15 KiB
    file = _publish_single_file(
        nemar_endpoint, path="data/throttle.bin", content=content
    )
    create_bandwith_degradation_toxic(
        for_proxy=chaos_proxy.proxy_name,
        toxic_name="throttle",
        rate=16,  # KB/s
        configuration=chaos_proxy.config,
    )

    via_chaos = _file_via_chaos(file, chaos_proxy)
    target = tmp_path / file.path
    started = time.monotonic()
    result = download_one(via_chaos, target)
    elapsed = time.monotonic() - started

    assert result is VerifyResult.OK
    assert target.read_bytes() == content
    # Sanity: the throttle was actually active.
    assert elapsed >= 0.3, f"transfer too fast ({elapsed:.2f}s) for throttle"


# ---------------------------------------------------------------------------
# Scenario 3 — Mid-stream drop, then clean retry resumes via Range/206
# ---------------------------------------------------------------------------


def test_partial_bytes_survive_drop_and_resume_completes(
    chaos_proxy: ChaosHandle, nemar_endpoint, tmp_path: Path
) -> None:
    """A drop mid-stream + clean retry round-trip reconstitutes the file.

    Two-phase contract: first attempt fails with a partial on disk,
    second attempt resumes via Range/206 and verifies clean.
    """
    content = b"R" * (100 * 1024)
    file = _publish_single_file(
        nemar_endpoint, path="data/drop.bin", content=content
    )
    nemar_endpoint.serve_with_range(f"/nm000132/v1.0.0/{file.path}")
    create_limiter_toxic(
        for_proxy=chaos_proxy.proxy_name,
        toxic_name="cut-32k",
        bytes_limit=32768,
        configuration=chaos_proxy.config,
    )
    via_chaos = _file_via_chaos(file, chaos_proxy)
    target = tmp_path / file.path

    # Phase 1: drop fails fast → TransferError, partial on disk.
    with pytest.raises(TransferError):
        download_one(
            via_chaos,
            target,
            retry=RetryPolicy.default().with_attempts(0),
        )
    on_disk = target.stat().st_size
    assert 0 < on_disk < file.size, (
        f"expected partial 0 < N < {file.size}, got {on_disk} B"
    )

    # Phase 2: clear the toxic and resume.
    delete_toxic(
        for_proxy=chaos_proxy.proxy_name,
        toxic_name="cut-32k",
        configuration=chaos_proxy.config,
    )
    result = download_one(via_chaos, target)
    assert result is VerifyResult.OK
    assert target.read_bytes() == content


# ---------------------------------------------------------------------------
# Scenario 4 — Verifier guards bad bytes
# ---------------------------------------------------------------------------


def test_post_transfer_verifier_catches_corruption(
    chaos_proxy: ChaosHandle, nemar_endpoint, tmp_path: Path
) -> None:
    """A claimed-hash mismatch must surface as ``VerifyResult.HASH_MISMATCH``.

    Toxiproxy does not have a built-in byte-flip toxic, so we cheat:
    write content A to the fixture, claim its size+sha256 as if it were
    content B, and confirm the verifier flags the mismatch.
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
    download_one(tampered, target)
    result = check(tampered, target, VerifyPolicy())
    assert result is VerifyResult.HASH_MISMATCH
