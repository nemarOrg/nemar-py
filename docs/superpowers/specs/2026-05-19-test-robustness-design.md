# Test Robustness & Production Hardening Design

**Status:** Approved scope, awaiting spec review
**Date:** 2026-05-19
**Author:** Bru (with Claude)
**Scope:** C — Full test pyramid + production bug fixes
**Estimated effort:** 10–14 days, parallelizable across subagents

## Problem

`nemar-py` reports 96% line coverage from 188 passing tests, but coverage is misleading. Nearly every test mocks `httpx.Client`, `httpx.stream`, or `subprocess.run`. The library never actually:

- Talks to a real HTTP server (mocked at the transport layer)
- Streams a real body, handles a real Range request, or recovers from a real mid-stream drop
- Invokes a real `aria2c` subprocess
- Runs the CLI as a real subprocess
- Exercises >1 worker thread (concurrency is plumbed but untested)
- Encounters real filesystem edge cases (readonly target, disk-full, unicode paths, symlinks)

We also have no CI, no type checking, no Python version matrix, no coverage threshold, and no mutation testing baseline. Several real-code fragilities are already visible by reading the source:

- Retry backoff has no jitter — synchronized retry storm risk
- `aria2c` subprocess has no wall-time timeout — can hang forever
- `concurrent.futures.as_completed` drops all non-first exceptions silently
- Resume path (`Range: bytes=N-`) doesn't re-verify the bytes already on disk before appending
- `httpx.stream(... timeout=60.0)` is hard-coded; not user-configurable

This spec proposes a layered test pyramid, a CI quality gate, and targeted bug fixes for the production fragilities above.

## Goals & non-goals

**Goals**
1. Move from "executes the lines" coverage to "verifies the behavior" coverage.
2. Establish a real-HTTP integration layer so tests exercise actual `httpx` transport, redirect handling, streaming, and Range semantics.
3. Cover the highest-risk untested surfaces: concurrency, `aria2c` parity, CLI subprocess, filesystem edge cases.
4. Fix the four production fragilities listed above (jitter, aria2c timeout, thread-error consolidation, configurable stream timeout).
5. Add CI with multi-OS / multi-Python matrix, type checking, coverage threshold, and an aria2c-present matrix entry.
6. Add an opt-in live smoke test against `data.nemar.org` (mirrors the openneuro-py `OPENNEURO_TEST_TOKEN` pattern).

**Non-goals**
- Refactoring `_download.py` beyond the four targeted fixes. (It's 858 lines and has scope to be split; that's a separate spec.)
- Changing the public API surface.
- Authentication / S3 fallback (the project explicitly scopes itself to the public data endpoint).
- Replacing pydantic, httpx, tqdm, or wcmatch.

## Architecture

### Test pyramid

```
                    /\
                   /  \    e2e/ (4-6 tests)
                  /____\     CLI subprocess, aria2c real, live smoke
                 /      \
                /        \   integration/ (~25 tests)
               /__________\    real httpx → pytest-httpserver (HTTPS w/ local cert)
              /            \
             /              \   property/ (~10 tests)
            /________________\    hypothesis: BIDS, glob, manifest fuzz
           /                  \
          /                    \   unit/ (existing 188, reorganized)
         /______________________\    tightened assertions, shared conftest fixtures
```

### Directory layout

```
tests/
  conftest.py                      # shared fixtures (server, factories, target_dir)
  fixtures/
    sample_manifests/              # canonical + edge-case JSON shapes
    sample_bids_tree/              # small tree used by e2e
  unit/
    test_bids.py                   # moved from tests/test_bids.py
    test_glob.py                   # NEW — split out from inline _glob coverage
    test_models.py                 # moved
    test_download_internals.py     # moved, tightened
    test_cli_inproc.py             # moved from tests/test_cli.py
  property/
    test_bids_properties.py        # parse(path).path == path (roundtrip)
    test_glob_properties.py        # bare/anchored semantics invariants
    test_manifest_properties.py    # mutate valid payloads → parser never crashes
  integration/                     # marker: @pytest.mark.integration
    test_index_and_manifest.py     # index fetch, retries, malformed JSON, redirects
    test_python_transfer.py        # stream, Range/206, drop-mid-stream, chunked, no-CL
    test_concurrent_transfer.py    # >1 worker, thread-error consolidation
    test_filesystem.py             # readonly target, partial w/ wrong hash, unicode
    test_resume.py                 # full resume cycle: partial → interrupt → resume → verify
  e2e/                             # marker: @pytest.mark.e2e
    test_cli_subprocess.py         # subprocess.run(["nemar-py", "download", ...])
    test_aria2_real.py             # @pytest.mark.aria2 — skipif aria2c missing
    test_full_download.py          # end-to-end happy path against fixture server
  live/                            # marker: @pytest.mark.live (env-gated)
    test_data_nemar_org_smoke.py   # NEMAR_LIVE_TEST=1 to run
```

### Shared fixtures (`conftest.py`)

| Fixture | Scope | Purpose |
|---|---|---|
| `nemar_https_server` | session | `pytest_httpserver.HTTPServer(host="localhost", port=0, ssl_context=ctx)` where `ctx` loads a `trustme`-generated cert issued for SANs `("localhost", "127.0.0.1", "::1")`. Returns base URL `https://localhost:<port>/`. CA PEM path exposed via `server.ca_pem_path`. |
| `trust_local_ca` | function | `monkeypatch.setenv("SSL_CERT_FILE", str(server.ca_pem_path))`. `httpx` (and the system `ssl` module) reads `SSL_CERT_FILE` for the default CA bundle, so production code paths that build their own `httpx.Client()` without a `verify=` kwarg automatically trust the local server. **No production code change required.** Verified end-to-end in spike (2026-05-19): real `nemar.fetch_dataset_index(data_url="https://localhost:<port>/")` succeeded against the local fixture server. |
| `dataset_index_factory` | function | `make_index(dataset="nm000132", latest="v1.0.0", versions=[…])` → JSON dict matching `DatasetIndex`. |
| `manifest_factory` | function | `make_manifest(paths, *, with_sha256=True, base_url=…)` → JSON list shape, with helpers `make_mapping_manifest`, `make_aliased_manifest` to exercise non-canonical shapes. |
| `dataset_description_factory` | function | Writes `dataset_description.json` with configurable `DatasetDOI` + `Version`. |
| `target_dir` | function | `tmp_path / "ds"` shortcut, with optional `pre_seed=` keyword to drop partial files / a previous dataset_description. |
| `bytes_blob_factory` | function | Deterministic `bytes` of N MB with computed sha256/md5, registered with the server. |
| `aria2c_present` | session | `bool(shutil.which("aria2c"))`; used by `pytest.mark.skipif`. |

### Server programming model

`pytest-httpserver` exposes `server.expect_request(uri).respond_with_json(payload)` and `respond_with_data(bytes)`. We wrap it in a small `NemarFakeEndpoint` helper:

```python
class NemarFakeEndpoint:
    def __init__(self, server: HTTPServer): ...
    def publish(self, dataset: str, *, index: dict, manifest: dict, files: dict[str, bytes]) -> None: ...
    def fail_index_with(self, dataset: str, *, status: int, body: str = "") -> None: ...
    def fail_then_recover(self, path: str, *, fail_count: int, status: int) -> None: ...
    def slow_response(self, path: str, *, delay_seconds: float) -> None: ...
    def serve_with_range(self, path: str, *, support_range: bool = True) -> None: ...
    def drop_after_bytes(self, path: str, *, after: int) -> None: ...
```

The helper centralizes the gnarly handler-callback writing so individual tests stay readable.

### Production code changes (Scope C deltas)

Four targeted fixes, each in its own commit, each driven by a failing test written first (TDD):

| # | Change | File | Test that drives it |
|---|---|---|---|
| 1 | Add jitter to `_fetch_json_with_retries` and `_transfer_one_with_python` backoff (`backoff * (1 + random.random() * 0.25)` or full-jitter; pick one in implementation) | `_download.py:360, _download.py:749` | `property/test_retry_jitter.py` — statistical: 100 retries don't all land in the same 10ms window |
| 2 | Add wall-time timeout to `aria2c` subprocess: `subprocess.run(cmd, check=True, timeout=aria2_timeout)`; new kwarg `aria2_timeout: float \| None = None` on `download()` | `_download.py:677` | `e2e/test_aria2_real.py::test_aria2_timeout_kills_hung_process` (uses a fixture server that never closes the connection) |
| 3 | Consolidate `concurrent.futures` errors: collect all exceptions, raise a single `RuntimeError` with the count and first 3 messages | `_download.py:734` | `integration/test_concurrent_transfer.py::test_all_failures_are_reported` |
| 4 | Make stream timeout configurable: new kwarg `stream_timeout: float = 60.0` on `download()`, plumbed to `_transfer_one_attempt` | `_download.py:71, _download.py:789` | `integration/test_python_transfer.py::test_stream_timeout_is_configurable` |

All four changes preserve the existing positional-arg-free public API (only new kwargs with safe defaults). No breaking changes.

### CI (`.github/workflows/test.yml`)

| Job | Runs on | What it does |
|---|---|---|
| `lint` | ubuntu-latest, Py 3.13 | `ruff check`, `ruff format --check` |
| `typecheck` | ubuntu-latest, Py 3.13 | `pyright src/nemar` |
| `test-matrix` | (ubuntu-latest, macos-latest) × (3.10, 3.11, 3.12, 3.13) | `pytest -m "not e2e and not live" --cov=nemar --cov-fail-under=95` |
| `test-aria2` | ubuntu-latest, Py 3.13 | `apt-get install -y aria2`, then `pytest -m aria2 -m e2e` |
| `test-cli-e2e` | ubuntu-latest, Py 3.13 | `pytest -m e2e -k "not aria2 and not live"` |
| `test-live-smoke` | ubuntu-latest, Py 3.13 | `pytest -m live` — only runs when `NEMAR_LIVE_TEST=1` is set (manual dispatch + scheduled weekly) |
| `build-smoke` | ubuntu-latest | `uv build` and `uv run --isolated --with . --no-project -- python -c "import nemar"` and `nemar-py --help` |

`pyproject.toml` additions:

```toml
[tool.pytest.ini_options]
addopts = ["--tb=short", "-ra", "-vv", "--strict-markers"]
markers = [
  "integration: needs the local HTTPS fixture server",
  "e2e: needs subprocess and/or real downloader",
  "aria2: needs aria2c on PATH",
  "live: hits data.nemar.org; gated by NEMAR_LIVE_TEST=1",
  "slow: takes >2 s",
]

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
strict = ["src/nemar"]
pythonVersion = "3.10"

[dependency-groups]
dev = [
  "pytest>=8",
  "pytest-cov>=5",
  "pytest-httpserver>=1.1",
  "trustme>=1.1",            # self-signed cert generation
  "hypothesis>=6.100",
  "ruff>=0.14",
  "pyright>=1.1.380",
]
```

### Live smoke test (gated)

`tests/live/test_data_nemar_org_smoke.py`:

- Skipped unless `NEMAR_LIVE_TEST=1`.
- Fetches `fetch_dataset_index(dataset="nm000132")` (or whichever tiny known-good dataset NEMAR exposes), asserts shape only.
- Downloads exactly one small file via `download(... include=["dataset_description.json"])` into `tmp_path`, asserts file exists, sha256 matches, byte count > 0.
- Does NOT assert specific version numbers (would be flaky as upstream evolves).
- Annotated `@pytest.mark.flaky(reruns=3, reruns_delay=10)` for network blips.

## Data flow / control flow

### Integration test interaction

```
pytest
  └── nemar_https_server (session fixture, ~50 ms startup)
       └── trustme.CA → trustme.Issued(cert + key) → SSLContext
  └── test function
       └── endpoint.publish(dataset="nm000132", index=…, manifest=…, files=…)
       └── monkeypatch SSL_CERT_FILE → httpx trusts local cert
       └── nemar.download(dataset="nm000132", data_url="https://localhost:PORT/", …)
            └── real httpx.Client → real TLS → local server → real bytes
            └── real file writes to tmp_path
       └── assertions on tmp_path contents and server.log_requests()
```

### Resume test interaction (the hardest case)

```
1. server.serve_with_range("file.bin", support_range=True)
2. server.drop_after_bytes("file.bin", after=512)
3. tmp_path / "file.bin" is pre-seeded with 512 bytes of the expected content
4. nemar.download(... max_retries=2)
5. Expectation:
   - First attempt: server drops at 512 bytes → ReadError → retry
   - Retry sends Range: bytes=512-, receives 206, completes
   - sha256 verification passes
6. Test asserts:
   - File on disk is byte-identical to original
   - server received exactly 2 GETs for that path
   - Second GET had Range header
```

## Error handling

- Test infrastructure errors (server startup, cert generation) raise during fixture teardown — clear failure, not a flake.
- Each integration test resets `nemar_https_server` request expectations in a function-scoped wrapper.
- Tests that intentionally trigger production errors use `pytest.raises(RuntimeError, match=…)` with a `match` pattern strict enough to catch wording regressions but loose enough to survive copy edits.
- Property tests use `hypothesis.example` to lock in any minimization found during development.

## Testing the tests (mutation baseline)

After the new suite is green, run `mutmut run` once and record the surviving-mutant count in `docs/superpowers/specs/2026-05-19-mutation-baseline.md`. We don't gate CI on it (mutmut is slow and noisy); we use it to identify tests that don't actually constrain behavior.

## Rollout plan (phases)

Each phase ends with all tests green and CI passing. Phases are sized so they can ship independently as PRs.

| Phase | Owner (proposed subagent) | Deliverable | Gate |
|---|---|---|---|
| 0 | (this session) | This design doc, user review, implementation plan via `writing-plans` | User approval |
| 1 | `test-architect` | `conftest.py` + fixtures + reorganize existing tests into `tests/unit/` + shared factories. Existing 188 tests still pass. | Coverage stays ≥96% |
| 2 | `tdd-guide` | `tests/property/` — 10 hypothesis tests. May surface bugs in `_bids.py` / `_glob.py` / `_models.py`; document but don't fix in this phase. | New tests green |
| 3 | `kraken` (TDD) | `tests/integration/` — 25 tests against real HTTPS fixture server. Drives implementation of jitter, configurable stream timeout, thread-error consolidation. | Coverage ≥97%, 3/4 production fixes landed |
| 4 | `kraken` + `devops` | `tests/e2e/` — CLI subprocess + aria2c real. Drives aria2c-timeout fix. | aria2c-timeout fix landed |
| 5 | `devops` | `.github/workflows/test.yml`, `pyproject.toml` updates, pyright config | CI green on PR |
| 6 | (this session) | `tests/live/` smoke + `docs/.../mutation-baseline.md` | Manual dispatch live run green |

## Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Self-signed cert / TLS plumbing is fragile across OSes | Medium | Use `trustme` (battle-tested), generate once per session, inject by monkeypatching `SSL_CERT_FILE` env var (httpx + stdlib `ssl` both honor it). If a CI runner ignores it, fall back to a private `_NEMAR_ALLOW_HTTP_FOR_TESTS` guard in `_validate_*_options` |
| `pytest-httpserver` lacks Range support out of the box | Medium | We write our own `respond_with_handler` callback that reads `Range` and slices the response; we know exactly what we need |
| `aria2c` real tests are flaky in CI | Low | Use a single small file and a local server; mark as `slow` so they're easy to quarantine |
| Property tests find existing bugs we can't quickly fix | Medium | Use `@hypothesis.example` + `xfail(strict=True, reason="known issue, see ticket")` to lock the known bug without blocking the suite |
| Bug-fix changes break someone's downstream usage | Low | All four production fixes are additive kwargs with safe defaults; no API removal |
| Live smoke test flakes block PRs | Low | Gated by env var; only runs on manual dispatch and weekly cron; failures notify but don't block |

## Open questions resolved by user

| Question | Answer |
|---|---|
| Scope (A/B/C)? | **C** — pyramid + bug fixes |
| HTTPS approach? | **Self-signed cert** with httpx verify override |
| Live smoke gated? | **Yes**, `NEMAR_LIVE_TEST=1` |
| Delegation strategy? | **Per-layer subagents in parallel** |

## Out of scope (parking lot for follow-up specs)

- Splitting `_download.py` (858 lines) into `_index.py`, `_manifest.py`, `_transfer.py`, `_verify.py`.
- Async download path (current is sync + threads).
- S3 fallback / authenticated downloads.
- Performance benchmarks (`pytest-benchmark`) — add when we have a baseline regression to chase.
- Replacing tqdm with a structured-logging-friendly progress reporter.
- Hardening dotfile filtering into a configurable option.

---

**Next step:** User reviews this spec. On approval, invoke the `writing-plans` skill to turn this into a concrete, ordered implementation plan with per-phase tasks, then dispatch the per-layer subagents in parallel.
