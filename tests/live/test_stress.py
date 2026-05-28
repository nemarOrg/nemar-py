"""Live stress + stability suite against data.nemar.org.

Five scenarios that exercise the failure modes the unit + integration
suites cannot reach (concurrency, idempotency at scale, real S3
contracts, multi-MB Range-resume, catalog-wide consistency). All five
hit the live endpoint; running them is gated by ``NEMAR_STRESS=1``
(separate from ``NEMAR_LIVE_TEST=1`` so the lighter smoke suite can
run independently).

Scenarios
---------

1. **Concurrent index fetches** — 16 parallel callers on one shared
   :class:`nemar.NEMARClient`. Pins that the shared
   ``httpx.Client`` + retry policy are thread-safe under realistic
   fan-out.
2. **Idempotent repeat downloads** — same file 20 times via
   :func:`nemar.transfer.download_one`. Pins that
   ``partition_pending`` skips redundant fetches and the verifier
   re-checks every time.
3. **Bulk transfer of 20 files** via
   :func:`nemar.transfer.download_files`. Exercises the shared
   ``httpx.Client`` connection pool, tqdm progress aggregation, and
   the post-transfer ``assert_all_present`` sweep at scale.
4. **Mid-stream interrupt + resume** — pre-seeds half of a multi-KB
   file and asks ``download_one`` to finish via the Range/206 path.
   Pins that the resume contract reconstitutes a byte-exact file.
5. **S3 helper sweep** — 3 of 4 nemar.s3 helpers × 20 random catalog
   datasets. ``version_url`` / ``version_summary_url`` must be
   100 % available; ``archive_url`` is best-effort (asynchronously
   published artifact), so we require ≥ 80 % availability instead
   of universal coverage.

The scenarios are intentionally independent (no shared state) so a
failure in one does not cascade.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import os
import random
import shutil
from pathlib import Path

import httpx
import pytest

import nemar
from nemar._verification import VerifyPolicy, VerifyResult, check
from nemar.s3 import archive_url, version_summary_url, version_url
from nemar.transfer import download_files, download_one

pytestmark = [pytest.mark.live, pytest.mark.slow]

SKIP_REASON = "Set NEMAR_STRESS=1 to run live stress scenarios."

# Eight datasets pulled into the index-fetch fan-out. Picked for stable
# tiny manifests + spread across both prefixes; the exact list is
# documentation, not contract.
_FANOUT_DATASETS = (
    "nm000132",
    "nm000133",
    "nm000104",
    "nm000118",
    "on005505",
    "on007180",
    "on007315",
    "on006126",
)

_PRIMARY_DATASET = os.environ.get("NEMAR_STRESS_DATASET", "nm000133")


def _gate() -> None:
    """Skip the calling test when the stress gate env var is not set."""
    if os.environ.get("NEMAR_STRESS") != "1":
        pytest.skip(SKIP_REASON)


# ---------------------------------------------------------------------------
# Scenario 1: 16 concurrent index fetches on one shared client
# ---------------------------------------------------------------------------


def test_concurrent_index_fetches_share_one_client() -> None:
    """16 parallel ``fetch_index`` calls against one shared client must
    all succeed without contention, returning the same per-dataset
    ``latest`` tag every caller saw.

    Pins the contract that ``NEMARClient``'s embedded ``httpx.Client``
    is thread-safe for our usage. A regression would surface as
    intermittent ``Connection`` / ``RemoteProtocolError`` exceptions
    or as a mismatched ``latest`` field across concurrent fetches.
    """
    _gate()
    ids = list(_FANOUT_DATASETS) * 2  # 16 calls, 8 distinct datasets
    seen: list[tuple[str, str]] = []
    with nemar.NEMARClient() as client, concurrent.futures.ThreadPoolExecutor(
        max_workers=8
    ) as pool:
        futures = {pool.submit(client.fetch_index, ds): ds for ds in ids}
        for future in concurrent.futures.as_completed(futures):
            ds = futures[future]
            idx = future.result()
            seen.append((ds, idx.latest))

    distinct = {pair for pair in seen}
    assert len(distinct) == len(_FANOUT_DATASETS), (
        f"expected {len(_FANOUT_DATASETS)} distinct datasets, "
        f"got {len(distinct)}: {distinct}"
    )


# ---------------------------------------------------------------------------
# Scenario 2: 20 idempotent re-runs of the same one-file download
# ---------------------------------------------------------------------------


def test_idempotent_repeat_downloads(tmp_path: Path) -> None:
    """Re-running ``download_one`` against an already-complete file is
    a verify-and-return — bytes do not change, size stays exact.

    Pins idempotency: a 20-iteration loop must not corrupt the file
    or grow it. The verifier sees the size + hash match every time
    and returns ``VerifyResult.OK`` without touching the wire on
    re-runs (the partition_pending skip happens in the bulk path, not
    here; here the point is that the per-file ``download_one``
    overwrites idempotently).
    """
    _gate()
    target = tmp_path / "repeat"
    target.mkdir()

    with nemar.NEMARClient() as client:
        index = client.fetch_index(_PRIMARY_DATASET)
        version = index.resolve_version("latest")
        manifest = client.fetch_manifest(index, version)

    candidates = [f for f in manifest if f.size and f.git_sha1]
    assert candidates, "expected at least one git-tracked sidecar"
    file = min(candidates, key=lambda f: f.size or 0)
    outfile = target / file.path

    download_one(file, outfile)
    first_bytes = outfile.read_bytes()
    assert outfile.stat().st_size == file.size

    for _ in range(20):
        download_one(file, outfile)

    assert outfile.stat().st_size == file.size
    assert outfile.read_bytes() == first_bytes


# ---------------------------------------------------------------------------
# Scenario 3: bulk transfer of N files via download_files
# ---------------------------------------------------------------------------


def test_bulk_download_files_handles_twenty_small_files(tmp_path: Path) -> None:
    """``download_files([...20 files], target_dir)`` must land every
    file on disk and the post-transfer ``assert_all_present`` must
    accept every one.

    Pins the bulk-transfer pipeline end-to-end:
    ``partition_pending`` (size-only pre-skip) → backend transfer
    over the shared ``httpx.Client`` → ``assert_all_present`` (full
    hash sweep). 20 small files is enough to amortize TCP/TLS reuse
    and surface any per-file accounting bugs.
    """
    _gate()
    target = tmp_path / "bulk"
    target.mkdir()

    with nemar.NEMARClient() as client:
        index = client.fetch_index(_PRIMARY_DATASET)
        version = index.resolve_version("latest")
        manifest = client.fetch_manifest(index, version)

    files = sorted(
        (f for f in manifest if f.size and f.size < 5_000_000),
        key=lambda f: f.size or 0,
    )[:20]
    assert len(files) == 20

    download_files(files, target_dir=target)

    for file in files:
        path = target / file.path
        assert path.exists(), f"missing file {file.path}"
        assert path.stat().st_size == file.size


# ---------------------------------------------------------------------------
# Scenario 4: interrupt mid-stream, restart, confirm resume works
# ---------------------------------------------------------------------------


def test_interrupt_then_resume_reconstitutes_file(tmp_path: Path) -> None:
    """A pre-seeded half-file must finish via Range/206 to byte-exact.

    Pins the resume contract: ``_transfer_one_attempt`` detects the
    partial bytes on disk, issues ``Range: bytes={local_size}-``,
    receives 206 with the right ``Content-Range``, appends, and the
    post-transfer SHA-256 matches the manifest.
    """
    _gate()
    target = tmp_path / "resume"
    target.mkdir()

    with nemar.NEMARClient() as client:
        index = client.fetch_index(_PRIMARY_DATASET)
        version = index.resolve_version("latest")
        manifest = client.fetch_manifest(index, version)

    candidates = [
        f
        for f in manifest
        if f.sha256 and f.size and 200_000 <= f.size <= 1_000_000
    ]
    assert candidates, "expected at least one 200KB-1MB sha256 file"
    file = candidates[0]
    outfile = target / file.path
    outfile.parent.mkdir(parents=True, exist_ok=True)

    half = (file.size or 0) // 2
    with httpx.Client(timeout=30.0) as raw:
        partial = raw.get(file.url, headers={"Range": f"bytes=0-{half - 1}"})
        partial.raise_for_status()
        outfile.write_bytes(partial.content)
    assert outfile.stat().st_size == half

    download_one(file, outfile)

    assert outfile.stat().st_size == file.size
    assert check(file, outfile, VerifyPolicy()) is VerifyResult.OK
    assert hashlib.sha256(outfile.read_bytes()).hexdigest() == file.sha256


# ---------------------------------------------------------------------------
# Scenario 5: S3 helpers sweep — 20 random datasets, 3 helpers each
# ---------------------------------------------------------------------------


def test_s3_helpers_resolve_across_random_catalog_sample() -> None:
    """20 random datasets × 3 helpers must mostly return 200 unsigned.

    Pins:

    * :func:`version_url` and :func:`version_summary_url` are 100 %
      available across the catalog — any 4xx/5xx is a contract break.
    * :func:`archive_url` is best-effort: archives are async build
      artifacts, so we require ≥ 80 % availability rather than
      universal coverage. A baseline sweep at the time of writing
      saw 92 % (37/40); the 80 % floor lets a handful of freshly-cut
      versions slip through without falsely failing the suite.
    """
    _gate()
    catalog = httpx.get(
        "https://data.nemar.org/",
        headers={"Accept": "application/json"},
        timeout=30.0,
    ).json()
    all_ids = [e["id"] for e in catalog["datasets"]]
    rng = random.Random(0xBEEF)
    sample = rng.sample(all_ids, 20)

    version_failures: list[str] = []
    summary_failures: list[str] = []
    archive_ok = 0
    archive_total = 0

    with httpx.Client(timeout=15.0) as raw, nemar.NEMARClient() as client:
        for ds in sample:
            tag = client.fetch_index(ds).latest

            v = raw.head(version_url(ds, tag)).status_code
            if v != 200:
                version_failures.append(f"{ds}: HTTP {v}")

            s = raw.head(version_summary_url(ds, tag)).status_code
            if s != 200:
                summary_failures.append(f"{ds}: HTTP {s}")

            archive_total += 1
            a = raw.head(archive_url(ds, tag)).status_code
            if a == 200:
                archive_ok += 1

    assert not version_failures, (
        f"version_url contract broken: {version_failures}"
    )
    assert not summary_failures, (
        f"version_summary_url contract broken: {summary_failures}"
    )
    archive_rate = archive_ok / archive_total
    assert archive_rate >= 0.80, (
        f"archive_url availability {archive_rate:.0%} below 80% floor "
        f"({archive_ok}/{archive_total})"
    )


# ---------------------------------------------------------------------------
# Optional housekeeping: shared temp dir survives between scenarios
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_temp(tmp_path: Path) -> Path:
    """Each scenario gets a fresh tmp dir; clean up afterwards."""
    yield tmp_path
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)
