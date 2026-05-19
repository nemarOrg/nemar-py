# Architectural Refactor & Surgical Fixes — Implementation Plan

**Goal:** Land all 8 architectural deepening candidates (A–H) and the 12 surgical fixes (S1–S12) surfaced by the four-agent architectural review.

**Source:** `docs/superpowers/specs/2026-05-19-test-robustness-design.md` (test pyramid spec) + the architect/pathfinder/oracle/sleuth synthesis posted in the chat on 2026-05-19.

**Approach:** Maximum safe parallelism. 4 batches, parallel within a batch, sequential between batches when there's a real dependency.

## Dependency graph

```
R1 (parallel x4)
  F+S5  ──┐       LocalDatasetIdentity + BOM fix (new file: _local.py)
  B     ──┤       RetryPolicy value type (_download.py retry helpers)
  C     ──┤       SelectionPlan (_download.py:_select_bids_files)
  tail  ──┘       S3, S6, S10, S11, S12 (small UX fixes)
              │
              ▼
R2 (1 agent, sequential within)
  D + E           DataEndpoint, then VersionManifest (depends on D)
              │
              ▼
R3 (parallel x2)
  A + Sx  ─┐     Verification module + S1, S2, S4, S7, S8, S9
  G       ─┘     Shared httpx.Client + connection pooling cap
              │
              ▼
R4 (1 agent)
  H               DownloadRequest value type composing everything
                  CONTEXT.md updates + final verifier sweep
```

## What gets folded where (surgical → architectural)

| Surgical fix | Lands in batch | Reason |
|---|---|---|
| S1 (uppercase hex) | R3 / A | Hash normalization is part of `manifest_verification` |
| S2 (weak ETag prefix) | R3 / A | Same as S1 |
| S3 (missing local Version) | R1 / tail | Independent one-liner |
| S4 (case-insensitive FS dedup) | R3 / A | Verification predicate concern |
| S5 (UTF-8 BOM) | R1 / F | Lands with the `_local.py` extraction |
| S6 (progress bar overshoot) | R1 / tail | Independent fix in `_transfer_one_attempt` |
| S7 (HTTP 416 recovery) | R3 / A | Recovery policy lives with verification |
| S8 (double aria2 verify) | R3 / A | Consolidation absorbs this |
| S9 (JSON sentinel) | R3 / A | Verification predicate concern |
| S10 (`__main__.py` guard) | R1 / tail | Trivial |
| S11 (zero-match suggestions) | R1 / C | Lands with SelectionPlan |
| S12 (split×concurrency cap) | R1 / tail | Independent fix in aria2 args |

## Conventions

- Commit per logical unit, with `feat(...)`, `refactor(...)`, or `fix(...)` prefix.
- No `Co-Authored-By` trailers. No push without user consent.
- Each agent reads CONTEXT.md before writing names. New domain terms get added to CONTEXT.md inline.
- TDD: new tests first, then implementation.
- All existing tests must remain green. Coverage gate stays at 94%.

## Acceptance for each batch

- All tests green: `pytest --cov=nemar --cov-fail-under=94 -q`
- Lint clean: `ruff check`
- Types clean: `pyright src/nemar`
- CLI smoke: `nemar-py --help`, `nemar-py download --help`, `nemar-py versions --help`
- No public API removal (additive only — new kwargs default to current behavior)
