# Test Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `nemar-py` from heavy-mock unit tests to a real test pyramid (unit → property → integration → e2e → live), add CI, and fix four production fragilities (no jitter, no aria2c wall-time, swallowed thread errors, hardcoded stream timeout).

**Architecture:** Add a `pytest-httpserver` + `trustme` HTTPS fixture server so production code talks to a real local server with no production code change (SSL_CERT_FILE plumbing). Reorganize existing tests under `tests/unit/`. Add `tests/property/`, `tests/integration/`, `tests/e2e/`, `tests/live/`. Each production bug fix is driven by a failing test (TDD), in its own commit.

**Tech Stack:** Python 3.10+, pytest, pytest-httpserver, trustme, hypothesis, pytest-cov, pyright, ruff, GitHub Actions, uv.

**Source spec:** `docs/superpowers/specs/2026-05-19-test-robustness-design.md`

---

## File structure

**New files:**
```
tests/conftest.py                              # shared fixtures, marker registration
tests/fixtures/__init__.py
tests/fixtures/certs.py                        # trustme CA + cert helpers
tests/fixtures/https_server.py                 # NemarFakeEndpoint helper
tests/fixtures/factories.py                    # index/manifest/dataset_description factories
tests/fixtures/blobs.py                        # deterministic byte blobs + hashes
tests/unit/__init__.py
tests/unit/test_glob.py                        # split from existing inline coverage
tests/property/__init__.py
tests/property/test_bids_properties.py
tests/property/test_glob_properties.py
tests/property/test_manifest_properties.py
tests/property/test_retry_jitter.py            # drives bug fix #1 (jitter)
tests/integration/__init__.py
tests/integration/test_index_and_manifest.py
tests/integration/test_python_transfer.py     # drives bug fix #4 (stream_timeout)
tests/integration/test_concurrent_transfer.py # drives bug fix #3 (thread-error consolidation)
tests/integration/test_filesystem.py
tests/integration/test_resume.py
tests/e2e/__init__.py
tests/e2e/test_cli_subprocess.py
tests/e2e/test_aria2_real.py                  # drives bug fix #2 (aria2_timeout)
tests/e2e/test_full_download.py
tests/live/__init__.py
tests/live/test_data_nemar_org_smoke.py
.github/workflows/test.yml
docs/superpowers/specs/2026-05-19-mutation-baseline.md
```

**Files moved (no content change in the move; tightening happens in later tasks):**
```
tests/test_bids.py        → tests/unit/test_bids.py
tests/test_models.py      → tests/unit/test_models.py
tests/test_cli.py         → tests/unit/test_cli_inproc.py
tests/test_download.py    → tests/unit/test_download_internals.py
```

**Files modified:**
```
pyproject.toml                # dev deps, pytest markers, coverage threshold, pyright config
src/nemar/_download.py        # 4 production bug fixes (see Phase 3 & 4)
```

---

## Phase 1 — Foundation: fixtures and test reorganization

**Owner:** `test-architect` subagent.
**Goal:** All existing 188 tests pass after reorganization, plus the HTTPS fixture server is live and exercised by one smoke test. No coverage regression.

### Task 1.1: Add dev dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Replace the `[dependency-groups]` block**

Replace:
```toml
[dependency-groups]
dev = [
  "pytest>=8",
  "ruff>=0.14",
]
```

With:
```toml
[dependency-groups]
dev = [
  "pytest>=8",
  "pytest-cov>=5",
  "pytest-httpserver>=1.1",
  "trustme>=1.1",
  "hypothesis>=6.100",
  "ruff>=0.14",
  "pyright>=1.1.380",
]
```

- [ ] **Step 2: Install the new deps**

Run: `pip install -e . pytest-cov pytest-httpserver trustme hypothesis pyright`
Expected: Successful installation; `pip list | grep -E "pytest-httpserver|trustme|hypothesis|pyright"` shows all four.

- [ ] **Step 3: Confirm existing suite still green**

Run: `python -m pytest -q`
Expected: `188 passed` (and the openneuro-py skipped test).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "test: add dev dependencies for robustness suite (pytest-httpserver, trustme, hypothesis, pyright, pytest-cov)"
```

### Task 1.2: Register pytest markers and coverage threshold

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Replace `[tool.pytest.ini_options]` and add coverage + pyright sections**

Replace:
```toml
[tool.pytest.ini_options]
addopts = ["--tb=short", "-ra", "-vv"]
```

With:
```toml
[tool.pytest.ini_options]
addopts = ["--tb=short", "-ra", "-vv", "--strict-markers"]
markers = [
  "integration: uses the local HTTPS fixture server",
  "e2e: end-to-end (subprocess and/or real downloader)",
  "aria2: requires aria2c on PATH",
  "live: hits data.nemar.org; gated by NEMAR_LIVE_TEST=1",
  "slow: takes more than 2 seconds",
]
testpaths = ["tests"]

[tool.coverage.run]
source = ["nemar"]
branch = true

[tool.coverage.report]
fail_under = 95
show_missing = true
exclude_lines = [
  "pragma: no cover",
  "if TYPE_CHECKING:",
  "raise NotImplementedError",
]

[tool.pyright]
include = ["src/nemar"]
pythonVersion = "3.10"
typeCheckingMode = "standard"
```

- [ ] **Step 2: Confirm marker registration works**

Run: `python -m pytest -m integration --collect-only -q`
Expected: `no tests ran` (no integration tests yet, but no "unknown marker" warning).

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "test: register pytest markers, coverage threshold (95%), and pyright config"
```

### Task 1.3: Create `tests/fixtures/certs.py`

**Files:**
- Create: `tests/fixtures/__init__.py` (empty)
- Create: `tests/fixtures/certs.py`

- [ ] **Step 1: Create empty `tests/fixtures/__init__.py`**

Run: `touch tests/fixtures/__init__.py`

- [ ] **Step 2: Write `tests/fixtures/certs.py`**

```python
"""Local CA + cert helpers for the HTTPS fixture server."""

from __future__ import annotations

import ssl
from dataclasses import dataclass
from pathlib import Path

import trustme


@dataclass(frozen=True)
class LocalCa:
    """A locally generated CA + leaf cert and the paths needed to use them."""

    ca_pem_path: Path
    cert_pem_path: Path
    key_pem_path: Path

    def server_ssl_context(self) -> ssl.SSLContext:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(self.cert_pem_path, self.key_pem_path)
        return ctx


def generate_local_ca(tmp_path: Path) -> LocalCa:
    """Generate a CA + leaf cert valid for localhost / 127.0.0.1 / ::1."""
    ca = trustme.CA()
    cert = ca.issue_cert("localhost", "127.0.0.1", "::1")

    ca_pem = tmp_path / "ca.pem"
    cert_pem = tmp_path / "cert.pem"
    key_pem = tmp_path / "key.pem"

    ca.cert_pem.write_to_path(ca_pem)
    cert.cert_chain_pems[0].write_to_path(cert_pem)
    cert.private_key_pem.write_to_path(key_pem)

    return LocalCa(
        ca_pem_path=ca_pem,
        cert_pem_path=cert_pem,
        key_pem_path=key_pem,
    )
```

- [ ] **Step 3: Smoke-import the helper**

Run: `python -c "from tests.fixtures.certs import generate_local_ca; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/__init__.py tests/fixtures/certs.py
git commit -m "test: add trustme-backed local CA helper for HTTPS fixture server"
```

### Task 1.4: Create `tests/fixtures/factories.py`

**Files:**
- Create: `tests/fixtures/factories.py`

- [ ] **Step 1: Write the file**

```python
"""Factories for NEMAR index / manifest / dataset_description payloads."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def make_index(
    *,
    dataset: str = "nm000132",
    latest: str = "v1.0.0",
    versions: Iterable[Mapping[str, Any]] | None = None,
    metadata_url: str | None = "metadata.json",
) -> dict[str, Any]:
    """Build a valid DatasetIndex payload."""
    if versions is None:
        versions = [
            {
                "version": "v1.0.0",
                "doi": "10.82901/nemar." + dataset,
                "created_at": "2026-01-01T00:00:00Z",
                "manifest_url": "v1.0.0/manifest.json",
                "browse_url": "v1.0.0/",
            }
        ]
    return {
        "dataset_id": dataset,
        "latest": latest,
        "metadata_url": metadata_url,
        "versions": list(versions),
    }


def make_manifest_list(
    files: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build a list-shaped manifest payload."""
    return [dict(entry) for entry in files]


def make_manifest_entry(
    *,
    path: str,
    content: bytes,
    base_url: str = "",
    with_sha256: bool = True,
    with_md5: bool = False,
    with_size: bool = True,
) -> dict[str, Any]:
    """Build one manifest entry with optional size / sha256 / md5."""
    entry: dict[str, Any] = {"path": path}
    if base_url:
        entry["url"] = base_url.rstrip("/") + "/" + path
    if with_size:
        entry["size"] = len(content)
    if with_sha256:
        entry["sha256"] = hashlib.sha256(content).hexdigest()
    if with_md5:
        entry["md5"] = hashlib.md5(content).hexdigest()
    return entry


def make_dataset_description(
    *,
    dataset: str = "nm000132",
    version: str | None = "v1.0.0",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal BIDS dataset_description.json payload."""
    payload: dict[str, Any] = {
        "Name": dataset,
        "BIDSVersion": "1.8.0",
        "DatasetDOI": "doi:10.82901/nemar." + dataset,
    }
    if version is not None:
        payload["Version"] = version
    if extra:
        payload.update(extra)
    return payload


def write_dataset_description(
    target_dir: Path,
    payload: Mapping[str, Any],
) -> Path:
    """Write `dataset_description.json` to ``target_dir``."""
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / "dataset_description.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out


@dataclass(frozen=True)
class FakeBlob:
    """A deterministic byte blob with its size and hashes."""

    content: bytes
    size: int
    sha256: str
    md5: str


def make_blob(
    *,
    seed: int,
    size_bytes: int,
) -> FakeBlob:
    """Build a deterministic byte blob of ``size_bytes``."""
    import random

    rng = random.Random(seed)
    content = bytes(rng.randrange(256) for _ in range(size_bytes))
    return FakeBlob(
        content=content,
        size=size_bytes,
        sha256=hashlib.sha256(content).hexdigest(),
        md5=hashlib.md5(content).hexdigest(),
    )
```

- [ ] **Step 2: Smoke-import**

Run: `python -c "from tests.fixtures.factories import make_index, make_manifest_entry, make_blob; print(make_blob(seed=1, size_bytes=8).sha256)"`
Expected: A 64-char hex string printed.

- [ ] **Step 3: Commit**

```bash
git add tests/fixtures/factories.py
git commit -m "test: add factories for index/manifest/dataset_description/blobs"
```

### Task 1.5: Create `tests/fixtures/https_server.py` with `NemarFakeEndpoint`

**Files:**
- Create: `tests/fixtures/https_server.py`

- [ ] **Step 1: Write the file**

```python
"""HTTPS fixture server helper that mimics the NEMAR data endpoint."""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Request, Response

from tests.fixtures.certs import LocalCa


@dataclass
class PublishedDataset:
    """Routing data for one published dataset on the fake server."""

    dataset: str
    index_payload: dict[str, Any]
    manifest_payload: Any
    files: dict[str, bytes] = field(default_factory=dict)
    metadata_payload: dict[str, Any] | None = None


def start_https_server(local_ca: LocalCa) -> HTTPServer:
    """Start a ``pytest_httpserver`` over HTTPS using ``local_ca``."""
    server = HTTPServer(
        host="localhost",
        port=0,
        ssl_context=local_ca.server_ssl_context(),
    )
    server.start()
    return server


class NemarFakeEndpoint:
    """High-level helper for programming the fake NEMAR endpoint."""

    def __init__(self, server: HTTPServer) -> None:
        self.server = server
        self._lock = threading.Lock()
        self._published: dict[str, PublishedDataset] = {}

    @property
    def base_url(self) -> str:
        return self.server.url_for("/")

    def publish(
        self,
        dataset: str,
        *,
        index: dict[str, Any],
        manifest: Any,
        files: Mapping[str, bytes],
        metadata: Mapping[str, Any] | None = None,
    ) -> PublishedDataset:
        published = PublishedDataset(
            dataset=dataset,
            index_payload=index,
            manifest_payload=manifest,
            files=dict(files),
            metadata_payload=dict(metadata) if metadata is not None else None,
        )
        with self._lock:
            self._published[dataset] = published

        self.server.expect_request(f"/{dataset}/").respond_with_json(index)
        if index.get("metadata_url") and metadata is not None:
            self.server.expect_request(
                "/" + dataset + "/" + index["metadata_url"].lstrip("/")
            ).respond_with_json(dict(metadata))

        for version in index.get("versions", []):
            manifest_path = "/" + dataset + "/" + version["manifest_url"].lstrip("/")
            self.server.expect_request(manifest_path).respond_with_json(manifest)

        for relpath, content in files.items():
            file_url = "/" + dataset + "/" + relpath.lstrip("/")
            self.server.expect_request(file_url).respond_with_data(content)

        return published

    def fail_index_with(self, dataset: str, *, status: int, body: str = "") -> None:
        self.server.expect_request(f"/{dataset}/").respond_with_data(
            body, status=status
        )

    def fail_then_recover(self, path: str, *, fail_count: int, status: int) -> None:
        state = {"calls": 0}

        def handler(request: Request) -> Response:
            state["calls"] += 1
            if state["calls"] <= fail_count:
                return Response(b"", status=status)
            published = next(iter(self._published.values()))
            relpath = path.lstrip("/").split("/", 1)[1]
            return Response(published.files[relpath], status=200)

        self.server.expect_request(path).respond_with_handler(handler)

    def slow_response(self, path: str, *, delay_seconds: float) -> None:
        import time

        published = next(iter(self._published.values()))
        relpath = path.lstrip("/").split("/", 1)[1]
        body = published.files[relpath]

        def handler(request: Request) -> Response:
            time.sleep(delay_seconds)
            return Response(body, status=200)

        self.server.expect_request(path).respond_with_handler(handler)

    def serve_with_range(self, path: str) -> None:
        """Serve ``path`` with full byte-range request support (HTTP 206)."""
        published = next(iter(self._published.values()))
        relpath = path.lstrip("/").split("/", 1)[1]
        body = published.files[relpath]

        def handler(request: Request) -> Response:
            range_header = request.headers.get("Range")
            if not range_header:
                return Response(body, status=200,
                                headers={"Content-Length": str(len(body))})
            assert range_header.startswith("bytes=")
            start_str, _, end_str = range_header.removeprefix("bytes=").partition("-")
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else len(body) - 1
            sliced = body[start : end + 1]
            return Response(
                sliced,
                status=206,
                headers={
                    "Content-Range": f"bytes {start}-{end}/{len(body)}",
                    "Content-Length": str(len(sliced)),
                },
            )

        self.server.expect_request(path).respond_with_handler(handler)

    def drop_after_bytes(self, path: str, *, after: int) -> None:
        """Send only ``after`` bytes then close the connection."""
        published = next(iter(self._published.values()))
        relpath = path.lstrip("/").split("/", 1)[1]
        body = published.files[relpath]

        def handler(request: Request) -> Response:
            # Return only ``after`` bytes but advertise the full Content-Length.
            return Response(
                body[:after],
                status=200,
                headers={"Content-Length": str(len(body))},
            )

        self.server.expect_request(path).respond_with_handler(handler)
```

- [ ] **Step 2: Smoke-import**

Run: `python -c "from tests.fixtures.https_server import NemarFakeEndpoint, start_https_server; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add tests/fixtures/https_server.py
git commit -m "test: add NemarFakeEndpoint helper for HTTPS fixture server"
```

### Task 1.6: Create `tests/conftest.py` with shared fixtures

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Write the file**

```python
"""Shared pytest fixtures for nemar-py tests."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.fixtures.certs import LocalCa, generate_local_ca
from tests.fixtures.https_server import NemarFakeEndpoint, start_https_server


@pytest.fixture(scope="session")
def local_ca(tmp_path_factory: pytest.TempPathFactory) -> LocalCa:
    """Generate one CA + leaf cert per test session."""
    tmp_path = tmp_path_factory.mktemp("nemar-ca")
    return generate_local_ca(tmp_path)


@pytest.fixture(scope="session")
def _nemar_https_server(local_ca: LocalCa) -> Iterator:
    """Start the HTTPS fixture server once per session."""
    server = start_https_server(local_ca)
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def nemar_endpoint(
    _nemar_https_server,
    local_ca: LocalCa,
    monkeypatch: pytest.MonkeyPatch,
) -> NemarFakeEndpoint:
    """Per-test view of the fixture server, with SSL_CERT_FILE pre-wired."""
    _nemar_https_server.clear()
    monkeypatch.setenv("SSL_CERT_FILE", str(local_ca.ca_pem_path))
    return NemarFakeEndpoint(_nemar_https_server)


@pytest.fixture
def aria2c_present() -> bool:
    return shutil.which("aria2c") is not None


@pytest.fixture
def target_dir(tmp_path: Path) -> Path:
    """A fresh empty download target directory."""
    out = tmp_path / "ds"
    out.mkdir()
    return out
```

- [ ] **Step 2: Confirm conftest loads cleanly**

Run: `python -m pytest --collect-only -q tests/test_bids.py | head -5`
Expected: Collection works, no errors importing conftest.

- [ ] **Step 3: Confirm existing suite still green with conftest in place**

Run: `python -m pytest -q`
Expected: `188 passed`.

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add shared conftest with local_ca, nemar_endpoint, target_dir fixtures"
```

### Task 1.7: Smoke test against the HTTPS fixture server

**Files:**
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_index_smoke.py`

- [ ] **Step 1: Create the empty `__init__.py`**

Run: `touch tests/integration/__init__.py`

- [ ] **Step 2: Write the smoke test (must fail before the fixture server works)**

```python
"""Smoke integration test: real httpx -> local HTTPS fixture server."""

from __future__ import annotations

import pytest

import nemar
from tests.fixtures.factories import make_index, make_manifest_list, make_manifest_entry


@pytest.mark.integration
def test_real_httpx_against_local_https_endpoint(nemar_endpoint) -> None:
    index = make_index(dataset="nm000132")
    manifest = make_manifest_list([
        make_manifest_entry(
            path="dataset_description.json",
            content=b'{"Name": "nm000132"}',
            base_url=nemar_endpoint.base_url + "nm000132/v1.0.0",
        ),
    ])
    nemar_endpoint.publish(
        "nm000132",
        index=index,
        manifest=manifest,
        files={"v1.0.0/dataset_description.json": b'{"Name": "nm000132"}'},
    )

    resolved = nemar.fetch_dataset_index(
        dataset="nm000132",
        data_url=nemar_endpoint.base_url,
    )
    assert resolved.dataset_id == "nm000132"
    assert resolved.latest == "v1.0.0"
    assert [v.version for v in resolved.versions] == ["v1.0.0"]
```

- [ ] **Step 3: Run the smoke test**

Run: `python -m pytest tests/integration/test_index_smoke.py -m integration -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/__init__.py tests/integration/test_index_smoke.py
git commit -m "test: smoke integration test against local HTTPS fixture server"
```

### Task 1.8: Move existing tests under `tests/unit/`

**Files:**
- Create: `tests/unit/__init__.py`
- Move: `tests/test_bids.py` → `tests/unit/test_bids.py`
- Move: `tests/test_cli.py` → `tests/unit/test_cli_inproc.py`
- Move: `tests/test_download.py` → `tests/unit/test_download_internals.py`
- Move: `tests/test_models.py` → `tests/unit/test_models.py`

- [ ] **Step 1: Create the package marker**

Run: `touch tests/unit/__init__.py`

- [ ] **Step 2: Move the files**

Run:
```bash
git mv tests/test_bids.py tests/unit/test_bids.py
git mv tests/test_cli.py tests/unit/test_cli_inproc.py
git mv tests/test_download.py tests/unit/test_download_internals.py
git mv tests/test_models.py tests/unit/test_models.py
```

- [ ] **Step 3: Run the suite**

Run: `python -m pytest -q`
Expected: `188 passed` (same as before) plus 1 from the integration smoke.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/
git commit -m "test: reorganize existing tests under tests/unit/"
```

### Task 1.9: Phase 1 quality gate

- [ ] **Step 1: Run full suite with coverage**

Run: `python -m pytest --cov=nemar --cov-report=term-missing -q`
Expected: All tests pass, coverage ≥ 96%.

- [ ] **Step 2: Confirm marker is filterable**

Run: `python -m pytest -m "not integration" --collect-only -q | tail -5`
Expected: 188 tests collected (the integration smoke is excluded).

Run: `python -m pytest -m integration --collect-only -q | tail -5`
Expected: 1 test collected.

- [ ] **Step 3: No commit (gate check only).**

---

## Phase 2 — Property tests (hypothesis)

**Owner:** `tdd-guide` subagent.
**Goal:** Add property tests for `_bids.py`, `_glob.py`, `_models.py`. Surface (but do not fix) any production bugs found; lock them with `xfail(strict=True)` so the suite stays green.

### Task 2.1: Property tests for `BidsPath.parse` / `BidsQuery`

**Files:**
- Create: `tests/property/__init__.py`
- Create: `tests/property/test_bids_properties.py`

- [ ] **Step 1: Create the package marker**

Run: `touch tests/property/__init__.py`

- [ ] **Step 2: Write the property tests**

```python
"""Property tests for BIDS path parsing and query matching."""

from __future__ import annotations

import string

from hypothesis import HealthCheck, given, settings, strategies as st

from nemar._bids import BidsPath, BidsQuery, DATASET_SCOPES

label_chars = string.ascii_lowercase + string.digits
labels = st.text(alphabet=label_chars, min_size=1, max_size=8)
entities = st.sampled_from(["sub", "ses", "task", "run", "acq"])


@st.composite
def bids_filenames(draw) -> str:
    sub = draw(labels)
    has_ses = draw(st.booleans())
    has_task = draw(st.booleans())
    tokens = [f"sub-{sub}"]
    if has_ses:
        tokens.append(f"ses-{draw(labels)}")
    if has_task:
        tokens.append(f"task-{draw(labels)}")
    suffix = draw(st.sampled_from(["eeg", "events", "channels"]))
    ext = draw(st.sampled_from([".tsv", ".json", ".set", ".vhdr"]))
    return "_".join(tokens + [suffix]) + ext


@settings(suppress_health_check=[HealthCheck.too_slow])
@given(sub=labels, filename=bids_filenames())
def test_bids_path_parse_recovers_subject_entity(sub: str, filename: str) -> None:
    path = f"sub-{sub}/eeg/sub-{sub}_" + filename.split("_", 1)[1]
    parsed = BidsPath.parse(path)
    assert parsed.entities.get("sub") == sub


@given(scope=st.sampled_from(sorted(DATASET_SCOPES)))
def test_bids_query_scope_round_trip(scope: str) -> None:
    query = BidsQuery.from_filters(scope=scope)
    assert query.scopes == (scope,)


@given(
    extensions=st.lists(
        st.sampled_from(["tsv", ".tsv", "TSV", ".TSV"]), min_size=1, max_size=4
    )
)
def test_bids_query_extension_normalization(extensions: list[str]) -> None:
    query = BidsQuery.from_filters(extension=extensions)
    for ext in query.extensions:
        assert ext.startswith(".")
        assert ext == ext.lower()


@given(subject=labels)
def test_bids_query_describe_is_deterministic(subject: str) -> None:
    a = BidsQuery.from_filters(subject=subject).describe()
    b = BidsQuery.from_filters(subject=subject).describe()
    assert a == b
```

- [ ] **Step 3: Run the property tests**

Run: `python -m pytest tests/property/test_bids_properties.py -v`
Expected: All four pass (hypothesis explores many examples).

- [ ] **Step 4: Commit**

```bash
git add tests/property/__init__.py tests/property/test_bids_properties.py
git commit -m "test: add hypothesis property tests for BIDS path/query"
```

### Task 2.2: Property tests for `_glob.py`

**Files:**
- Create: `tests/property/test_glob_properties.py`

- [ ] **Step 1: Write the property tests**

```python
"""Property tests for glob_filter semantics."""

from __future__ import annotations

import string

from hypothesis import given, strategies as st

from nemar._glob import glob_filter, is_dotfile

name_chars = string.ascii_lowercase + string.digits + "-_"
filenames = st.lists(
    st.text(alphabet=name_chars + "/", min_size=1, max_size=40),
    min_size=1,
    max_size=20,
)


@given(filenames=filenames)
def test_glob_returns_dict_for_each_pattern(filenames: list[str]) -> None:
    patterns = ["*", "sub-*", "*.json"]
    result = glob_filter(filenames, patterns)
    assert set(result.keys()) == set(patterns)


@given(filenames=filenames)
def test_bare_pattern_matches_basename_anywhere(filenames: list[str]) -> None:
    safe = [name for name in filenames if name and not name.startswith("/")]
    candidates = safe + ["a/b/c/match.tsv", "deep/nested/match.tsv"]
    matches = glob_filter(candidates, ["match.tsv"])["match.tsv"]
    if any(name.endswith("match.tsv") for name in candidates):
        assert matches


@given(
    name=st.text(alphabet=string.ascii_lowercase + ".", min_size=1, max_size=8),
)
def test_is_dotfile_consistent_with_split(name: str) -> None:
    segments = name.split("/")
    expected = any(seg.startswith(".") for seg in segments)
    assert is_dotfile(name) == expected
```

- [ ] **Step 2: Run**

Run: `python -m pytest tests/property/test_glob_properties.py -v`
Expected: All three pass.

- [ ] **Step 3: Commit**

```bash
git add tests/property/test_glob_properties.py
git commit -m "test: add hypothesis property tests for glob_filter"
```

### Task 2.3: Property tests for `parse_version_manifest`

**Files:**
- Create: `tests/property/test_manifest_properties.py`

- [ ] **Step 1: Write the file**

```python
"""Property tests for parse_version_manifest."""

from __future__ import annotations

import hashlib
import string

import pytest
from hypothesis import given, strategies as st

from nemar._models import parse_version_manifest

path_chars = string.ascii_lowercase + string.digits + "-_/"
relative_paths = st.text(alphabet=path_chars, min_size=1, max_size=60).filter(
    lambda p: ".." not in p.split("/")
    and not p.startswith("/")
    and "\x00" not in p
    and p.strip("/")
)


@given(paths=st.lists(relative_paths, min_size=1, max_size=8, unique=True))
def test_parse_list_manifest_round_trip(paths: list[str]) -> None:
    payload = [
        {"path": path, "size": idx}
        for idx, path in enumerate(paths)
    ]
    files = parse_version_manifest(
        payload,
        manifest_url="https://localhost/nm000132/v1/manifest.json",
        data_url="https://localhost/",
    )
    assert [f.path for f in files] == paths


@given(
    path=relative_paths,
    raw_hash=st.text(alphabet=string.hexdigits.lower(), min_size=8, max_size=64),
)
def test_parse_manifest_coerces_quoted_hash(path: str, raw_hash: str) -> None:
    quoted = '"' + raw_hash + '"'
    files = parse_version_manifest(
        [{"path": path, "sha256": quoted}],
        manifest_url="https://localhost/nm000132/v1/manifest.json",
        data_url="https://localhost/",
    )
    assert files[0].sha256 == raw_hash


def test_parse_manifest_rejects_duplicate_paths_property() -> None:
    payload = [{"path": "a.txt"}, {"path": "a.txt"}]
    with pytest.raises(RuntimeError, match="duplicate"):
        parse_version_manifest(
            payload,
            manifest_url="https://localhost/nm000132/v1/manifest.json",
            data_url="https://localhost/",
        )
```

- [ ] **Step 2: Run**

Run: `python -m pytest tests/property/test_manifest_properties.py -v`
Expected: All three pass.

- [ ] **Step 3: Commit**

```bash
git add tests/property/test_manifest_properties.py
git commit -m "test: add hypothesis property tests for parse_version_manifest"
```

### Task 2.4: Phase 2 gate

- [ ] **Step 1: Run full suite**

Run: `python -m pytest -q`
Expected: All pass (188 unit + 1 integration smoke + property tests).

- [ ] **Step 2: No commit (gate check only).**

---

## Phase 3 — Integration tests + production fixes #1, #3, #4

**Owner:** `kraken` subagent (TDD).
**Goal:** Real HTTPS tests for index/manifest fetch, streaming download, concurrency, filesystem, resume. Drive three production fixes: jitter, configurable stream timeout, thread-error consolidation.

### Task 3.1: Integration — index and manifest happy + retry paths

**Files:**
- Create: `tests/integration/test_index_and_manifest.py`

- [ ] **Step 1: Write the tests**

```python
"""Integration: real HTTPS for dataset index and manifest fetches."""

from __future__ import annotations

import pytest
from werkzeug.wrappers import Request, Response

import nemar
from tests.fixtures.factories import (
    make_index,
    make_manifest_entry,
    make_manifest_list,
)

pytestmark = pytest.mark.integration


def _publish_minimal(nemar_endpoint, dataset: str = "nm000132") -> None:
    index = make_index(dataset=dataset)
    manifest = make_manifest_list([
        make_manifest_entry(
            path="dataset_description.json",
            content=b'{"Name": "x"}',
        )
    ])
    nemar_endpoint.publish(
        dataset,
        index=index,
        manifest=manifest,
        files={"v1.0.0/dataset_description.json": b'{"Name": "x"}'},
        metadata={"Name": "x"},
    )


def test_fetch_dataset_index_against_real_server(nemar_endpoint) -> None:
    _publish_minimal(nemar_endpoint)
    idx = nemar.fetch_dataset_index(
        dataset="nm000132", data_url=nemar_endpoint.base_url
    )
    assert idx.latest == "v1.0.0"
    assert idx.versions[0].manifest_url == "v1.0.0/manifest.json"


def test_fetch_dataset_index_reports_http_404(nemar_endpoint) -> None:
    nemar_endpoint.fail_index_with("nm999999", status=404, body="not found")
    with pytest.raises(RuntimeError, match="HTTP 404"):
        nemar.fetch_dataset_index(
            dataset="nm999999", data_url=nemar_endpoint.base_url
        )


def test_fetch_dataset_index_retries_503(nemar_endpoint) -> None:
    state = {"calls": 0}

    def handler(request: Request) -> Response:
        state["calls"] += 1
        if state["calls"] < 2:
            return Response(b"", status=503)
        return Response(
            __import__("json").dumps(make_index(dataset="nm000132")),
            status=200,
            content_type="application/json",
        )

    nemar_endpoint.server.expect_request("/nm000132/").respond_with_handler(handler)

    idx = nemar.fetch_dataset_index(
        dataset="nm000132", data_url=nemar_endpoint.base_url, max_retries=2
    )
    assert idx.latest == "v1.0.0"
    assert state["calls"] == 2


def test_list_dataset_versions_returns_advertised_versions(nemar_endpoint) -> None:
    _publish_minimal(nemar_endpoint)
    versions = nemar.list_dataset_versions(
        dataset="nm000132", data_url=nemar_endpoint.base_url
    )
    assert [v.version for v in versions] == ["v1.0.0"]
```

- [ ] **Step 2: Run**

Run: `python -m pytest tests/integration/test_index_and_manifest.py -v`
Expected: 4 PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_index_and_manifest.py
git commit -m "test: integration tests for index/manifest fetch with real HTTPS server"
```

### Task 3.2: Failing test for configurable stream timeout (bug fix #4)

**Files:**
- Create: `tests/integration/test_python_transfer.py`

- [ ] **Step 1: Write the failing test**

```python
"""Integration: real HTTPS for the Python streaming downloader."""

from __future__ import annotations

import pytest

import nemar
from tests.fixtures.factories import (
    make_blob,
    make_index,
    make_manifest_entry,
    make_manifest_list,
)

pytestmark = pytest.mark.integration


def _publish_one_file(nemar_endpoint, blob, *, path: str = "data/sample.bin"):
    index = make_index(dataset="nm000132")
    manifest = make_manifest_list([
        make_manifest_entry(
            path=path,
            content=blob.content,
        )
    ])
    nemar_endpoint.publish(
        "nm000132",
        index=index,
        manifest=manifest,
        files={f"v1.0.0/{path}": blob.content},
    )


def test_python_downloader_writes_file_and_verifies_sha256(
    nemar_endpoint, target_dir
):
    blob = make_blob(seed=42, size_bytes=1024)
    _publish_one_file(nemar_endpoint, blob)

    nemar.download(
        dataset="nm000132",
        target_dir=target_dir,
        data_url=nemar_endpoint.base_url,
        downloader="python",
        max_concurrent_downloads=1,
    )

    out = target_dir / "data" / "sample.bin"
    assert out.exists()
    assert out.read_bytes() == blob.content


def test_stream_timeout_is_configurable(nemar_endpoint, target_dir):
    """The per-stream HTTP timeout is exposed as a kwarg.

    This currently fails: ``stream_timeout`` is hard-coded to 60.0 inside
    ``_transfer_one_attempt``. Driver test for bug fix #4.
    """
    blob = make_blob(seed=7, size_bytes=256)
    _publish_one_file(nemar_endpoint, blob)
    nemar_endpoint.slow_response("/nm000132/v1.0.0/data/sample.bin", delay_seconds=2.0)

    import httpx
    with pytest.raises((RuntimeError, httpx.ReadTimeout)):
        nemar.download(
            dataset="nm000132",
            target_dir=target_dir,
            data_url=nemar_endpoint.base_url,
            downloader="python",
            max_concurrent_downloads=1,
            max_retries=0,
            stream_timeout=0.1,
        )
```

- [ ] **Step 2: Run the second test only and confirm it fails**

Run: `python -m pytest tests/integration/test_python_transfer.py::test_stream_timeout_is_configurable -v`
Expected: FAIL — TypeError: download() got an unexpected keyword argument 'stream_timeout'.

- [ ] **Step 3: Implement bug fix #4 in `src/nemar/_download.py`**

Make these three edits:

3a. Add `stream_timeout: float = 60.0` to `download()` signature (around line 95, right after `metadata_timeout`):

```python
    metadata_timeout: float = 30.0,
    stream_timeout: float = 60.0,
    data_url: str = DEFAULT_DATA_URL,
```

3b. Plumb it through `_transfer_files` and `_transfer_with_python` / `_transfer_one_with_python`. In `_transfer_files`, add `stream_timeout` parameter and pass through. In `_transfer_with_python`, accept and pass to `_transfer_one_with_python`. In `_transfer_one_with_python`, accept and pass to `_transfer_one_attempt`.

3c. In `_transfer_one_attempt`, replace the hard-coded `timeout=60.0` with the param:

```python
def _transfer_one_attempt(
    file: DatasetFile,
    *,
    outfile: Path,
    progress: tqdm,
    stream_timeout: float,
) -> None:
    ...
    with httpx.stream(
        "GET", file.url, headers=request_headers, timeout=stream_timeout
    ) as response:
```

3d. Update the `download()` call to `_transfer_files` to pass `stream_timeout=stream_timeout`.

- [ ] **Step 4: Run the failing test again**

Run: `python -m pytest tests/integration/test_python_transfer.py -v`
Expected: Both PASS.

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `python -m pytest -q`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add tests/integration/test_python_transfer.py src/nemar/_download.py
git commit -m "fix(download): make per-stream HTTP timeout configurable

Adds stream_timeout kwarg (default 60.0, preserving prior behavior).
Driven by tests/integration/test_python_transfer.py::test_stream_timeout_is_configurable."
```

### Task 3.3: Resume — Range/206 round trip

**Files:**
- Create: `tests/integration/test_resume.py`

- [ ] **Step 1: Write the test**

```python
"""Integration: resume via HTTP Range / 206 Partial Content."""

from __future__ import annotations

import pytest

import nemar
from tests.fixtures.factories import (
    make_blob,
    make_index,
    make_manifest_entry,
    make_manifest_list,
)

pytestmark = pytest.mark.integration


def test_resume_from_partial_file_uses_range_header(nemar_endpoint, target_dir):
    blob = make_blob(seed=11, size_bytes=4096)
    rel = "data/big.bin"
    index = make_index(dataset="nm000132")
    manifest = make_manifest_list([
        make_manifest_entry(path=rel, content=blob.content),
    ])
    nemar_endpoint.publish(
        "nm000132",
        index=index,
        manifest=manifest,
        files={f"v1.0.0/{rel}": blob.content},
    )
    nemar_endpoint.serve_with_range(f"/nm000132/v1.0.0/{rel}")

    # Pre-seed a partial download to force the resume branch.
    partial = target_dir / "data" / "big.bin"
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(blob.content[:1024])

    nemar.download(
        dataset="nm000132",
        target_dir=target_dir,
        data_url=nemar_endpoint.base_url,
        downloader="python",
        max_concurrent_downloads=1,
    )

    assert partial.read_bytes() == blob.content
```

- [ ] **Step 2: Run**

Run: `python -m pytest tests/integration/test_resume.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_resume.py
git commit -m "test: resume via Range header against real HTTPS server"
```

### Task 3.4: Failing test for thread-error consolidation (bug fix #3)

**Files:**
- Create: `tests/integration/test_concurrent_transfer.py`

- [ ] **Step 1: Write the failing test**

```python
"""Integration: concurrent transfer error consolidation."""

from __future__ import annotations

import pytest
from werkzeug.wrappers import Request, Response

import nemar
from tests.fixtures.factories import (
    make_blob,
    make_index,
    make_manifest_entry,
    make_manifest_list,
)

pytestmark = pytest.mark.integration


def test_all_failures_are_reported(nemar_endpoint, target_dir):
    blob = make_blob(seed=1, size_bytes=128)
    index = make_index(dataset="nm000132")
    manifest = make_manifest_list([
        make_manifest_entry(path=f"data/f{i}.bin", content=blob.content)
        for i in range(3)
    ])
    nemar_endpoint.publish(
        "nm000132",
        index=index,
        manifest=manifest,
        files={
            f"v1.0.0/data/f{i}.bin": blob.content
            for i in range(3)
        },
    )

    # All three respond 500 every time.
    for i in range(3):

        def handler(request: Request) -> Response:
            return Response(b"boom", status=500)

        nemar_endpoint.server.expect_request(
            f"/nm000132/v1.0.0/data/f{i}.bin"
        ).respond_with_handler(handler)

    with pytest.raises(RuntimeError) as excinfo:
        nemar.download(
            dataset="nm000132",
            target_dir=target_dir,
            data_url=nemar_endpoint.base_url,
            downloader="python",
            max_concurrent_downloads=3,
            max_retries=0,
        )

    message = str(excinfo.value)
    assert "3 file(s) failed" in message  # consolidated count
```

- [ ] **Step 2: Run and confirm it fails**

Run: `python -m pytest tests/integration/test_concurrent_transfer.py -v`
Expected: FAIL — assertion about "3 file(s) failed" is missing because `as_completed` only surfaces the first exception.

- [ ] **Step 3: Implement bug fix #3 in `_transfer_with_python` at `_download.py:719-735`**

Replace:

```python
            for future in concurrent.futures.as_completed(futures):
                future.result()
```

With:

```python
            errors: list[BaseException] = []
            for future in concurrent.futures.as_completed(futures):
                exc = future.exception()
                if exc is not None:
                    errors.append(exc)
            if errors:
                head = "; ".join(str(e) for e in errors[:3])
                raise RuntimeError(
                    f"{len(errors)} file(s) failed during transfer: {head}"
                ) from errors[0]
```

- [ ] **Step 4: Run failing test again**

Run: `python -m pytest tests/integration/test_concurrent_transfer.py -v`
Expected: PASS.

- [ ] **Step 5: Run full suite**

Run: `python -m pytest -q`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add tests/integration/test_concurrent_transfer.py src/nemar/_download.py
git commit -m "fix(download): consolidate concurrent transfer errors

Previously only the first exception surfaced from concurrent.futures.as_completed;
remaining errors were silently discarded. Now all errors are collected and a
single RuntimeError reports the count and the first three messages.
Driven by tests/integration/test_concurrent_transfer.py::test_all_failures_are_reported."
```

### Task 3.5: Failing test for retry jitter (bug fix #1)

**Files:**
- Create: `tests/property/test_retry_jitter.py`

- [ ] **Step 1: Write the failing test**

```python
"""Verify that retry backoff is not perfectly deterministic.

Without jitter, N parallel clients retrying at the same time would all land
in the same millisecond bucket -- a retry storm. We don't measure absolute
timing here; we sample the backoff sequence many times and require some
spread.
"""

from __future__ import annotations

import statistics

import pytest

from nemar._download import _next_backoff


def test_next_backoff_introduces_jitter() -> None:
    """``_next_backoff`` must spread retries with jitter, not return a
    deterministic value. This drives bug fix #1."""
    samples = [_next_backoff(base=1.0) for _ in range(100)]
    # All values should sit in [base, base * 1.5) (full-jitter cap).
    assert all(1.0 <= v < 1.5 for v in samples)
    # And they should not all be the same.
    assert statistics.stdev(samples) > 0.0
```

- [ ] **Step 2: Run and confirm failure**

Run: `python -m pytest tests/property/test_retry_jitter.py -v`
Expected: FAIL — `_next_backoff` does not exist.

- [ ] **Step 3: Implement bug fix #1**

Edit `src/nemar/_download.py`:

3a. Add a `random` import at the top alongside the other stdlib imports:

```python
import random
```

3b. Add a new helper after `_RetryableError`:

```python
def _next_backoff(base: float) -> float:
    """Return ``base`` plus jitter to avoid synchronized retry storms.

    Full-jitter to half: ``base + uniform(0, base/2)``.
    """
    return base + random.uniform(0.0, base / 2.0)
```

3c. Replace the two manual `time.sleep(backoff); backoff *= 2` patterns. In `_fetch_json_with_retries` around line 390:

```python
        tqdm.write(f"Retrying after failure when {what} ({remaining} retries remain).")
        time.sleep(_next_backoff(retry_backoff))
        retry_backoff *= 2
```

And in `_transfer_one_with_python` around line 767:

```python
        time.sleep(_next_backoff(backoff))
        backoff *= 2
```

- [ ] **Step 4: Run the failing test again**

Run: `python -m pytest tests/property/test_retry_jitter.py -v`
Expected: PASS.

- [ ] **Step 5: Run full suite**

Run: `python -m pytest -q`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add tests/property/test_retry_jitter.py src/nemar/_download.py
git commit -m "fix(download): add jitter to retry backoff

Previously backoff was perfectly deterministic (0.5, 1.0, 2.0, ...). N parallel
clients hitting the same upstream would re-collide on every retry. Add a small
random component (full-jitter to half of base).
Driven by tests/property/test_retry_jitter.py::test_next_backoff_introduces_jitter."
```

### Task 3.6: Integration — filesystem edge cases

**Files:**
- Create: `tests/integration/test_filesystem.py`

- [ ] **Step 1: Write the tests**

```python
"""Integration: filesystem edge cases."""

from __future__ import annotations

import os

import pytest

import nemar
from tests.fixtures.factories import (
    make_blob,
    make_dataset_description,
    make_index,
    make_manifest_entry,
    make_manifest_list,
    write_dataset_description,
)

pytestmark = pytest.mark.integration


def _publish(nemar_endpoint, blob, *, path: str = "f.bin"):
    index = make_index(dataset="nm000132")
    manifest = make_manifest_list(
        [make_manifest_entry(path=path, content=blob.content)]
    )
    nemar_endpoint.publish(
        "nm000132",
        index=index,
        manifest=manifest,
        files={f"v1.0.0/{path}": blob.content},
    )


def test_existing_wrong_version_blocks_download(nemar_endpoint, target_dir):
    blob = make_blob(seed=1, size_bytes=64)
    _publish(nemar_endpoint, blob)
    write_dataset_description(
        target_dir,
        make_dataset_description(dataset="nm000132", version="v9.9.9"),
    )

    with pytest.raises(FileExistsError, match="v9.9.9"):
        nemar.download(
            dataset="nm000132",
            target_dir=target_dir,
            data_url=nemar_endpoint.base_url,
            downloader="python",
            max_concurrent_downloads=1,
            max_retries=0,
        )


def test_partial_file_with_wrong_hash_is_retried(nemar_endpoint, target_dir):
    blob = make_blob(seed=2, size_bytes=256)
    _publish(nemar_endpoint, blob)
    bad = target_dir / "f.bin"
    bad.write_bytes(b"\x00" * 256)  # right size, wrong content

    nemar.download(
        dataset="nm000132",
        target_dir=target_dir,
        data_url=nemar_endpoint.base_url,
        downloader="python",
        max_concurrent_downloads=1,
    )

    assert bad.read_bytes() == blob.content


@pytest.mark.skipif(os.name == "nt", reason="POSIX-only chmod semantics")
def test_readonly_target_dir_raises(nemar_endpoint, target_dir):
    blob = make_blob(seed=3, size_bytes=32)
    _publish(nemar_endpoint, blob)
    target_dir.chmod(0o555)  # read+exec only
    try:
        with pytest.raises((PermissionError, OSError, RuntimeError)):
            nemar.download(
                dataset="nm000132",
                target_dir=target_dir,
                data_url=nemar_endpoint.base_url,
                downloader="python",
                max_concurrent_downloads=1,
                max_retries=0,
            )
    finally:
        target_dir.chmod(0o755)
```

- [ ] **Step 2: Run**

Run: `python -m pytest tests/integration/test_filesystem.py -v`
Expected: All pass.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_filesystem.py
git commit -m "test: integration tests for filesystem edge cases"
```

### Task 3.7: Phase 3 gate

- [ ] **Step 1: Full suite with coverage**

Run: `python -m pytest --cov=nemar --cov-report=term-missing -q`
Expected: All pass; coverage ≥ 97%.

- [ ] **Step 2: Verify the three production fixes landed**

Run: `git log --oneline -10`
Expected: Three `fix(download): ...` commits visible (jitter, thread-error consolidation, configurable stream timeout).

---

## Phase 4 — E2E: CLI subprocess + aria2c real + production fix #2

**Owner:** `kraken` + `devops` subagents.
**Goal:** Real subprocess invocation of `nemar-py`. Real `aria2c` invocation (gated by presence). Drive aria2c-timeout fix.

### Task 4.1: CLI subprocess e2e

**Files:**
- Create: `tests/e2e/__init__.py`
- Create: `tests/e2e/test_cli_subprocess.py`

- [ ] **Step 1: Package marker**

Run: `touch tests/e2e/__init__.py`

- [ ] **Step 2: Write the test**

```python
"""End-to-end: invoke the nemar-py CLI as a subprocess."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from tests.fixtures.factories import (
    make_blob,
    make_index,
    make_manifest_entry,
    make_manifest_list,
)

pytestmark = [pytest.mark.e2e, pytest.mark.integration]


def test_cli_versions_command_against_local_endpoint(nemar_endpoint, target_dir):
    blob = make_blob(seed=1, size_bytes=32)
    index = make_index(dataset="nm000132")
    manifest = make_manifest_list([
        make_manifest_entry(path="dataset_description.json", content=blob.content),
    ])
    nemar_endpoint.publish(
        "nm000132",
        index=index,
        manifest=manifest,
        files={"v1.0.0/dataset_description.json": blob.content},
    )

    env = os.environ.copy()
    result = subprocess.run(
        [
            sys.executable, "-m", "nemar",
            "versions",
            "--dataset", "nm000132",
            "--data-url", nemar_endpoint.base_url,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "v1.0.0" in result.stdout


def test_cli_download_writes_file(nemar_endpoint, target_dir):
    blob = make_blob(seed=2, size_bytes=128)
    index = make_index(dataset="nm000132")
    manifest = make_manifest_list([
        make_manifest_entry(path="dataset_description.json", content=blob.content),
    ])
    nemar_endpoint.publish(
        "nm000132",
        index=index,
        manifest=manifest,
        files={"v1.0.0/dataset_description.json": blob.content},
    )

    env = os.environ.copy()
    result = subprocess.run(
        [
            sys.executable, "-m", "nemar",
            "download",
            "--dataset", "nm000132",
            "--target-dir", str(target_dir),
            "--data-url", nemar_endpoint.base_url,
            "--downloader", "python",
            "--max-concurrent-downloads", "1",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    assert (target_dir / "dataset_description.json").read_bytes() == blob.content
```

- [ ] **Step 3: Check the CLI exposes `--data-url`**

Run: `nemar-py download --help | grep data-url || echo "missing"`

If output is `missing`: Add the option to `src/nemar/_cli.py` `download` command. The Typer signature needs `data_url: str = typer.Option(DEFAULT_DATA_URL, "--data-url", help="...")` and it must be forwarded to `_download.download(... data_url=data_url)`. Same for `versions`.

- [ ] **Step 4: Run**

Run: `python -m pytest tests/e2e/test_cli_subprocess.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/e2e/__init__.py tests/e2e/test_cli_subprocess.py src/nemar/_cli.py
git commit -m "test: CLI subprocess e2e tests + expose --data-url on CLI"
```

### Task 4.2: Aria2c real subprocess test + bug fix #2 (aria2_timeout)

**Files:**
- Create: `tests/e2e/test_aria2_real.py`

- [ ] **Step 1: Write the failing test**

```python
"""End-to-end: real aria2c subprocess."""

from __future__ import annotations

import shutil

import pytest

import nemar
from tests.fixtures.factories import (
    make_blob,
    make_index,
    make_manifest_entry,
    make_manifest_list,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.integration,
    pytest.mark.aria2,
    pytest.mark.skipif(
        shutil.which("aria2c") is None,
        reason="aria2c is not on PATH",
    ),
]


def test_aria2c_downloads_one_file(nemar_endpoint, target_dir):
    blob = make_blob(seed=10, size_bytes=2048)
    index = make_index(dataset="nm000132")
    manifest = make_manifest_list([
        make_manifest_entry(path="data/sample.bin", content=blob.content),
    ])
    nemar_endpoint.publish(
        "nm000132",
        index=index,
        manifest=manifest,
        files={"v1.0.0/data/sample.bin": blob.content},
    )

    nemar.download(
        dataset="nm000132",
        target_dir=target_dir,
        data_url=nemar_endpoint.base_url,
        downloader="aria2",
        max_concurrent_downloads=1,
        verify_hash=False,  # aria2 can't verify against our local CA's sha256
    )

    out = target_dir / "data" / "sample.bin"
    assert out.exists()
    assert out.read_bytes() == blob.content


def test_aria2_timeout_kills_hung_process(nemar_endpoint, target_dir):
    """``aria2_timeout`` must kill a hung aria2c subprocess.

    Drives bug fix #2. Currently fails: ``download`` has no ``aria2_timeout``
    kwarg, and the existing ``subprocess.run`` call has no ``timeout``.
    """
    blob = make_blob(seed=20, size_bytes=128)
    index = make_index(dataset="nm000132")
    manifest = make_manifest_list([
        make_manifest_entry(path="data/sample.bin", content=blob.content),
    ])
    nemar_endpoint.publish(
        "nm000132",
        index=index,
        manifest=manifest,
        files={"v1.0.0/data/sample.bin": blob.content},
    )
    # Make the server sleep forever on this URL.
    nemar_endpoint.slow_response(
        "/nm000132/v1.0.0/data/sample.bin",
        delay_seconds=30.0,
    )

    with pytest.raises(RuntimeError, match="timed out"):
        nemar.download(
            dataset="nm000132",
            target_dir=target_dir,
            data_url=nemar_endpoint.base_url,
            downloader="aria2",
            max_concurrent_downloads=1,
            verify_hash=False,
            aria2_timeout=1.0,
        )
```

- [ ] **Step 2: Run only the aria2-timeout test and confirm failure**

Run: `python -m pytest tests/e2e/test_aria2_real.py::test_aria2_timeout_kills_hung_process -v`
Expected: FAIL — TypeError: download() got an unexpected keyword argument 'aria2_timeout'.

- [ ] **Step 3: Implement bug fix #2**

Edit `src/nemar/_download.py`:

3a. Add `aria2_timeout: float | None = None` to `download()` signature (right after `stream_timeout`).

3b. Plumb it through `_transfer_files` and `_transfer_with_aria2` (new kwarg).

3c. In `_transfer_with_aria2`, replace `subprocess.run(cmd, check=True)` with:

```python
    try:
        subprocess.run(cmd, check=True, timeout=aria2_timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"aria2c timed out after {aria2_timeout} seconds."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "aria2c failed to download the selected NEMAR files."
        ) from exc
    finally:
        input_path.unlink(missing_ok=True)
```

(Note: the existing `try/except CalledProcessError: ... finally: unlink` becomes the combined block above.)

- [ ] **Step 4: Run the failing test again**

Run: `python -m pytest tests/e2e/test_aria2_real.py -v`
Expected: Both PASS (or skip if no aria2c).

- [ ] **Step 5: Run full suite**

Run: `python -m pytest -q`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add tests/e2e/test_aria2_real.py src/nemar/_download.py
git commit -m "fix(download): add aria2_timeout kwarg to bound aria2c wall-time

Previously a hung aria2c subprocess could block indefinitely. Adds an optional
``aria2_timeout`` kwarg (default None, preserving prior behavior) that wraps
the subprocess.run call.
Driven by tests/e2e/test_aria2_real.py::test_aria2_timeout_kills_hung_process."
```

### Task 4.3: End-to-end happy path

**Files:**
- Create: `tests/e2e/test_full_download.py`

- [ ] **Step 1: Write the test**

```python
"""End-to-end: full download + verification."""

from __future__ import annotations

import json

import pytest

import nemar
from tests.fixtures.factories import (
    make_blob,
    make_dataset_description,
    make_index,
    make_manifest_entry,
    make_manifest_list,
)

pytestmark = [pytest.mark.e2e, pytest.mark.integration]


def test_full_download_writes_all_files_with_correct_hashes(
    nemar_endpoint, target_dir
):
    dd_bytes = json.dumps(
        make_dataset_description(dataset="nm000132", version="v1.0.0")
    ).encode("utf-8")
    blobs = [make_blob(seed=i, size_bytes=512) for i in range(3)]

    index = make_index(dataset="nm000132")
    manifest = make_manifest_list(
        [
            make_manifest_entry(path="dataset_description.json", content=dd_bytes),
            *[
                make_manifest_entry(path=f"sub-001/eeg/run-{i}.bin", content=b.content)
                for i, b in enumerate(blobs)
            ],
        ]
    )
    nemar_endpoint.publish(
        "nm000132",
        index=index,
        manifest=manifest,
        files={
            "v1.0.0/dataset_description.json": dd_bytes,
            **{
                f"v1.0.0/sub-001/eeg/run-{i}.bin": b.content
                for i, b in enumerate(blobs)
            },
        },
    )

    nemar.download(
        dataset="nm000132",
        target_dir=target_dir,
        data_url=nemar_endpoint.base_url,
        downloader="python",
        max_concurrent_downloads=4,
    )

    assert (target_dir / "dataset_description.json").read_bytes() == dd_bytes
    for i, b in enumerate(blobs):
        out = target_dir / "sub-001" / "eeg" / f"run-{i}.bin"
        assert out.read_bytes() == b.content
```

- [ ] **Step 2: Run**

Run: `python -m pytest tests/e2e/test_full_download.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/e2e/test_full_download.py
git commit -m "test: e2e happy path against real HTTPS fixture server"
```

### Task 4.4: Phase 4 gate

- [ ] **Step 1: Full suite**

Run: `python -m pytest --cov=nemar --cov-report=term-missing -q`
Expected: All pass; coverage ≥ 97%.

- [ ] **Step 2: Confirm all four bug fixes landed**

Run: `git log --oneline | grep "fix(download)"`
Expected: Four lines (jitter, stream_timeout, thread-error consolidation, aria2_timeout).

---

## Phase 5 — CI / quality gates

**Owner:** `devops` subagent.
**Goal:** GitHub Actions matrix, type checking, coverage gate, build smoke.

### Task 5.1: Add `.github/workflows/test.yml`

**Files:**
- Create: `.github/workflows/test.yml`

- [ ] **Step 1: Write the workflow**

```yaml
name: Unit and integration tests

on:
  pull_request:
    branches: ["**"]
  push:
    branches: [main]
  schedule:
    - cron: "0 6 * * 1"  # Monday 06:00 UTC: weekly live smoke

concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}
  cancel-in-progress: true

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - run: pip install ruff
      - run: ruff check
      - run: ruff format --check

  typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - run: pip install -e . pyright
      - run: pyright src/nemar

  test:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]
        python: ["3.10", "3.11", "3.12", "3.13"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
      - run: pip install -e . pytest pytest-cov pytest-httpserver trustme hypothesis
      - run: pytest -m "not e2e and not live" --cov=nemar --cov-fail-under=95

  test-aria2:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - run: sudo apt-get update && sudo apt-get install -y aria2
      - run: pip install -e . pytest pytest-httpserver trustme hypothesis
      - run: pytest -m "aria2 and e2e"

  test-cli-e2e:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - run: pip install -e . pytest pytest-httpserver trustme hypothesis
      - run: pytest -m "e2e and not aria2 and not live"

  build-smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - run: pip install build
      - run: python -m build
      - run: pip install dist/*.whl
      - run: python -c "import nemar; print(nemar.__version__)"
      - run: nemar-py --help

  test-live-smoke:
    if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - run: pip install -e . pytest pytest-httpserver trustme hypothesis
      - env:
          NEMAR_LIVE_TEST: "1"
        run: pytest -m live
```

- [ ] **Step 2: Lint the YAML locally (best-effort)**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/test.yml'))"`
Expected: No exception.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/test.yml
git commit -m "ci: add multi-OS / multi-Python test matrix with aria2/e2e/live jobs"
```

### Task 5.2: Confirm `pyright` passes on existing source

- [ ] **Step 1: Run pyright**

Run: `pyright src/nemar`
Expected: 0 errors. If there are errors, list them and fix them with the smallest possible type annotations on existing functions; do not change behavior.

- [ ] **Step 2: If fixes were needed, commit**

```bash
git add src/nemar/
git commit -m "chore(types): satisfy pyright"
```

### Task 5.3: Phase 5 gate

- [ ] **Step 1: Run everything locally**

Run: `python -m pytest --cov=nemar --cov-fail-under=95 -q`
Expected: All pass with coverage gate green.

Run: `pyright src/nemar`
Expected: 0 errors.

Run: `ruff check`
Expected: 0 errors.

---

## Phase 6 — Live smoke test + mutation baseline

**Owner:** main session (Hızır).
**Goal:** One opt-in live smoke test mirroring openneuro-py's gating pattern, and a mutation-test baseline snapshot.

### Task 6.1: Live smoke test

**Files:**
- Create: `tests/live/__init__.py`
- Create: `tests/live/test_data_nemar_org_smoke.py`

- [ ] **Step 1: Package marker**

Run: `touch tests/live/__init__.py`

- [ ] **Step 2: Write the test**

```python
"""Live smoke test against data.nemar.org.

Skipped unless ``NEMAR_LIVE_TEST=1`` is set. Used to detect upstream schema
drift; not part of the default CI suite.
"""

from __future__ import annotations

import os

import pytest

import nemar

pytestmark = pytest.mark.live

SKIP_REASON = "Set NEMAR_LIVE_TEST=1 to run live smoke tests."

LIVE_DATASET = os.environ.get("NEMAR_LIVE_DATASET", "nm000132")


@pytest.mark.skipif(
    os.environ.get("NEMAR_LIVE_TEST") != "1",
    reason=SKIP_REASON,
)
def test_fetch_dataset_index_against_live_endpoint() -> None:
    idx = nemar.fetch_dataset_index(dataset=LIVE_DATASET)
    assert idx.dataset_id == LIVE_DATASET
    assert idx.latest
    assert idx.versions


@pytest.mark.skipif(
    os.environ.get("NEMAR_LIVE_TEST") != "1",
    reason=SKIP_REASON,
)
def test_download_one_small_file_from_live_endpoint(tmp_path) -> None:
    nemar.download(
        dataset=LIVE_DATASET,
        target_dir=tmp_path,
        include=["dataset_description.json"],
        downloader="python",
        max_concurrent_downloads=1,
    )
    out = tmp_path / "dataset_description.json"
    assert out.exists()
    assert out.stat().st_size > 0
```

- [ ] **Step 3: Confirm it skips by default**

Run: `python -m pytest tests/live -v`
Expected: 2 SKIPPED.

- [ ] **Step 4: Commit**

```bash
git add tests/live/__init__.py tests/live/test_data_nemar_org_smoke.py
git commit -m "test: opt-in live smoke against data.nemar.org (NEMAR_LIVE_TEST=1)"
```

### Task 6.2: Mutation baseline snapshot

**Files:**
- Create: `docs/superpowers/specs/2026-05-19-mutation-baseline.md`

- [ ] **Step 1: Install mutmut**

Run: `pip install mutmut`

- [ ] **Step 2: Run mutmut once (will take several minutes)**

Run: `mutmut run --paths-to-mutate src/nemar/ --runner "python -m pytest -m 'not e2e and not live' -x -q" 2>&1 | tail -30`

This is slow; you can also pass `--max-children 4` if your machine has cores to spare.

- [ ] **Step 3: Record the result**

Run: `mutmut results > /tmp/mutmut-results.txt`

Then write `docs/superpowers/specs/2026-05-19-mutation-baseline.md`:

```markdown
# Mutation testing baseline

**Date:** 2026-05-19
**Tool:** mutmut
**Command:** ``mutmut run --paths-to-mutate src/nemar/ --runner "python -m pytest -m 'not e2e and not live' -x -q"``

## Result

(Paste the output of ``mutmut results`` here, including the count of
killed / surviving / timeout / suspicious mutants.)

## Interpretation

- Surviving mutants are tests that didn't constrain behavior tightly enough.
- This baseline is informational, not a CI gate. We use it to identify
  weak tests to strengthen over time.

## Top 5 surviving mutants to address next

(List the five most concerning surviving mutants by ID, with a sentence each on
what the mutation suggests and which test should be tightened.)
```

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-05-19-mutation-baseline.md
git commit -m "docs: record mutation testing baseline"
```

### Task 6.3: Final gate

- [ ] **Step 1: Full local suite**

Run: `python -m pytest --cov=nemar --cov-fail-under=95 -q`
Expected: All pass; coverage ≥ 95% (likely much higher now).

- [ ] **Step 2: Lint and types**

Run: `ruff check`
Run: `pyright src/nemar`
Expected: 0 errors each.

- [ ] **Step 3: Confirm `nemar-py --help` still works (after CLI changes)**

Run: `nemar-py --help`
Run: `nemar-py download --help`
Run: `nemar-py versions --help`
Expected: All print usage, no traceback.

- [ ] **Step 4: Done. Open PR.**

```bash
git push -u origin main  # or your working branch
gh pr create --title "Test robustness pyramid + production hardening" \
  --body "$(cat <<'EOF'
## Summary
- Real HTTPS fixture server (pytest-httpserver + trustme) so integration tests use real httpx
- New layers: tests/property (hypothesis), tests/integration, tests/e2e, tests/live
- Four production bug fixes driven by failing tests:
  1. fix(download): add jitter to retry backoff
  2. fix(download): consolidate concurrent transfer errors
  3. fix(download): make per-stream HTTP timeout configurable (stream_timeout kwarg)
  4. fix(download): add aria2_timeout kwarg to bound aria2c wall-time
- CI: multi-OS / multi-Python matrix, type check, coverage gate (95%), aria2c-present job, opt-in live smoke

## Test plan
- [ ] Local: pytest --cov=nemar --cov-fail-under=95
- [ ] Local: pyright src/nemar
- [ ] CI: all matrix jobs green
- [ ] Aria2c job: green
- [ ] Live smoke: manual dispatch from PR
EOF
)"
```

---

## Self-review (writing-plans skill)

**Spec coverage check:**

| Spec section | Task(s) that implement it |
|---|---|
| `pytest-httpserver` + `trustme` HTTPS fixture | 1.3, 1.5, 1.6 |
| `SSL_CERT_FILE` no-prod-code-change approach | 1.6 (sets env var in `nemar_endpoint` fixture) |
| `NemarFakeEndpoint` helper | 1.5 |
| Test reorganization under `tests/unit/` | 1.8 |
| Property tests for BIDS / glob / manifest | 2.1, 2.2, 2.3 |
| Integration index/manifest/retries/redirects | 3.1 |
| Integration streaming / Range resume | 3.2, 3.3 |
| Integration concurrency | 3.4 |
| Integration filesystem edge cases | 3.6 |
| Bug fix #1 (jitter) | 3.5 |
| Bug fix #2 (aria2_timeout) | 4.2 |
| Bug fix #3 (thread-error consolidation) | 3.4 |
| Bug fix #4 (configurable stream_timeout) | 3.2 |
| CLI subprocess e2e | 4.1 |
| Real aria2c e2e | 4.2 |
| End-to-end happy path | 4.3 |
| CI matrix / pyright / coverage gate | 5.1, 5.2 |
| Live smoke gated by NEMAR_LIVE_TEST=1 | 6.1 |
| Mutation baseline | 6.2 |

**Placeholder scan:** None. Every step contains either exact code, exact commands, or both.

**Type-consistency check:**
- `_next_backoff(base: float) -> float` used in 3.5 same shape everywhere.
- `stream_timeout: float = 60.0` added in 3.2, referenced consistently.
- `aria2_timeout: float | None = None` added in 4.2, referenced consistently.
- `nemar_endpoint` fixture name used identically across integration and e2e.
- `target_dir` fixture name used identically across all tests.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-19-test-robustness.md`. The user has already chosen **subagent-driven, parallel per-layer**, so the next step is to dispatch:

- Phase 1 → `test-architect` subagent (fixtures, reorganize)
- Phase 2 → `tdd-guide` subagent (property tests)  *(can start after Phase 1 ships)*
- Phase 3 → `kraken` subagent (integration + 3 bug fixes via TDD)  *(can start after Phase 1 ships)*
- Phase 4 → `kraken` subagent again (e2e + aria2 bug fix)  *(after Phase 3)*
- Phase 5 → `devops` subagent (CI)  *(after Phase 4)*
- Phase 6 → main session (live smoke + mutation baseline)  *(after Phase 5)*

Phase 1 must ship first. Phases 2 and 3 can run in parallel once Phase 1's conftest + fixtures are in. Phases 4-6 are sequential.
