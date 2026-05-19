# Mutation testing baseline

**Date:** 2026-05-19
**Tool:** mutmut 3.5.0
**Scope:** `src/nemar/_bids.py`, `src/nemar/_glob.py`, `src/nemar/_models.py`
**Runner:** unit tests (`tests/unit/`) + property tests (`tests/property/`), excluding the CLI/download-integration tests that depend on the full `nemar` package being mutated (they are skipped because mutmut only copies the three target modules into its sandbox).

## Why this scope and not all of `src/nemar/`

Full mutmut on `_download.py` (358 statements with branch coverage spanning httpx streaming, retries, concurrency, and the aria2c subprocess) would multiply the run time by 20-30× because the integration tests against the local HTTPS fixture server take ~10 s wall-time per invocation. The baseline measures the small, pure modules where the test suite is fastest and the signal is cleanest. A future change can extend the baseline to `_download.py` against the unit-only subset (`tests/unit/test_download_internals.py` plus `tests/property/test_retry_jitter.py`).

## How to reproduce

```bash
pip install mutmut
mutmut run
mutmut export-cicd-stats
cat mutants/mutmut-cicd-stats.json
```

Configuration lives in `pyproject.toml` under `[tool.mutmut]`. The relevant test-selection knobs are encoded there.

## Result

```json
{
    "killed": 379,
    "survived": 100,
    "total": 479,
    "no_tests": 0,
    "skipped": 0,
    "suspicious": 0,
    "timeout": 0,
    "segfault": 0
}
```

**Kill ratio: 79.1%** (`killed / total`).

| Module | Mutants (approx) | Comments |
|---|---|---|
| `_bids.py` | ~210 | BIDS path parsing and query construction; well covered by unit + property tests |
| `_glob.py` | ~50 | Small surface; couple of survivors around the prefix `+ "/**"` branch |
| `_models.py` | ~220 | Largest survivor population — most are error-message string-literal mutations that we deliberately do not pin in tests |

## Interpretation

This baseline is **informational, not a CI gate.** A 79.1% kill ratio is acceptable for an MVP suite. We use it as a worklist of where tests can be strengthened.

### Why the 100 survivors break into two groups

**Group A — Noise (≈85 of 100):** String-literal mutations inside error messages and human-readable descriptions. Examples:

- `"The NEMAR data endpoint returned an unexpected dataset index"` → `"XXThe NEMAR... XX"` (survives because tests use `pytest.raises(RuntimeError)` without `match=` on every word, by design — pinning prose is brittle).
- `tqdm.write(...)` content mutations.
- `f"...{value}..."` prefix/suffix mutations on log lines.

These survivors are **not addressable** without making tests brittle to copy edits. They are accepted noise.

**Group B — Real gaps (≈15 of 100):** Mutations that change actual behavior and live through the suite. The five most worth attention are:

| # | Mutant | What the mutation does | Why it survives | Suggested fix |
|---|---|---|---|---|
| 1 | `nemar._bids.x__split_dataset_scope__mutmut_10` | `len(parts) > 2` → `len(parts) >= 2` in the `derivatives/{pipeline}/...` branch | No test exercises a path of the form `derivatives/foo` (exactly two segments). Both old and new code agree on every other input length. | Add a unit test for `BidsPath.parse("derivatives/eeglab")` and assert `pipeline == "eeglab"` (or `None`, depending on the intended contract — clarify first). |
| 2 | `nemar._models.x__entry_to_file__mutmut_10` | `_first_value(entry, "path", "filename", "name", ...)` → `..., None, ...` (drops `"name"` from the alias list) | No manifest test uses a `{"name": "..."}` entry; the parser silently falls through to `"relative_path"`. | Add `tests/unit/test_models.py::test_parse_manifest_uses_name_field` with payload `[{"name": "data.bin"}]` and assert `files[0].path == "data.bin"`. |
| 3 | `nemar._glob.x_glob_filter__mutmut_39` | Drops the `flags=glob.GLOBSTAR` argument from the second `globfilter` call (the prefix `pattern/**` retry) | No test exercises a multi-level prefix match where GLOBSTAR is required. | Add a unit test for `glob_filter(["a/b/c/d.txt"], ["a"])` and assert `"a/b/c/d.txt"` is matched (depends on GLOBSTAR for the `a/**` retry). |
| 4 | `nemar._bids.x__normalize_entity_mapping__mutmut_15/16` | Off-by-one in the `for item in entity:` loop that parses `key=value` strings | The property test covers most well-formed inputs, but doesn't probe the boundary between `Iterable[str]` and `Mapping`. | Add a parametrized unit test that covers entity-iterable inputs at boundary lengths 0, 1, many. |
| 5 | `nemar._models.x__coerce_size__mutmut_2/_coerce_hash__mutmut_2` | Inverts the `value in (None, "")` short-circuit to `value not in (None, "")` | Both branches happen to produce `None` or pass-through for many inputs, so the test for "missing field returns None" still passes. | Strengthen `test_parse_version_manifest_accepts_supported_shapes[entries-mapping]` to assert that an entry with `"size": ""` (empty string) produces `file.size is None`. |

The remaining ≈10 of Group B are minor variations on the same patterns above (off-by-one, alias drops, short-circuit inversions).

## Recommendations

1. **Do not gate CI on mutmut.** It's noisy and slow. Use it quarterly or before major releases.
2. **Address the five Group B findings above** in a follow-up commit. Each is a single unit test, low cost.
3. **Extend the scope to `_download.py`** in a follow-up baseline (separate doc), running mutmut against `tests/unit/test_download_internals.py` only to keep the wall-time tractable.
4. **Accept Group A noise.** Pinning exact error wording would make tests brittle to harmless copy edits.

## Files

- Config: `pyproject.toml` `[tool.mutmut]` section
- Raw stats: `mutants/mutmut-cicd-stats.json` (gitignored by default)
- Per-mutant details: `mutmut results` (lists survivors); `mutmut show <mutant-name>` for a unified diff.
