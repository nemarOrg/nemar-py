# Reference implementations

Snapshots of related projects kept here for design reference and
pattern-copying. **Not built. Not tested. Not imported by `nemar-py`.**

## `openneuro-py/`

A snapshot of [github.com/openneuro-py/openneuro-py][upstream] — an OpenNeuro
client that solves a parallel problem (the OpenNeuro public dataset endpoint).
We keep it here because some of its choices (BIDS selection vocabulary,
manifest-shape tolerance, retry policy) informed `nemar-py`'s design.

What it is:

- A frozen copy of the upstream at the time of import. The inner `.git/`
  directory has been removed; this is a snapshot, not a parallel repo.
- A pure reference. The upstream is the authoritative source.

What it is **not**:

- Not on `nemar-py`'s CI pipeline.
- Not on `nemar-py`'s dependency list (`pyproject.toml`).
- Not imported by anything in `src/nemar/`.

If you want a working copy to run or contribute to, clone the upstream
separately:

```shell
git clone https://github.com/openneuro-py/openneuro-py
```

[upstream]: https://github.com/openneuro-py/openneuro-py
