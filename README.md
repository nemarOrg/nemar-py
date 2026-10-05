# nemar-py

Python + CLI client for downloading public **NEMAR** datasets (BIDS / EEG / MEG / iEEG) from `data.nemar.org`.

## Install

```shell
pip install nemar-py            # S3 + HTTPS backends
pip install nemar-py[datalad]   # + the optional DataLad layer
```

The default install uses the direct-S3 and HTTPS backends. The optional
`[datalad]` extra adds the DataLad layer; it bundles the `git-annex` binary
via the `psychoinformatics-de` PyPI wheel (Linux, macOS, Windows), so no
system package step is needed. Without the extra, `--downloader datalad` and
the `auto` chain's DataLad layer fall through to S3 / HTTPS automatically.

## Quick start

```shell
nemar-py download nm000132 -o data/nm000132
```

```python
import nemar
nemar.download(dataset="nm000132", target_dir="data/nm000132")
```

## CLI

| Command                        | Purpose                              |
| ------------------------------ | ------------------------------------ |
| `nemar-py download <DATASET>`  | Download a dataset                   |
| `nemar-py versions <DATASET>`  | List versions advertised by NEMAR    |

### Common flags

| Flag                      | Effect                                              |
| ------------------------- | --------------------------------------------------- |
| `-o, --output DIR`        | Target directory (default `./<DATASET>`)            |
| `-j, --jobs N`            | Parallel downloads (default 16)                     |
| `--tag TAG`               | Pin a version; default `latest`                     |
| `--downloader BACKEND`    | `auto` (default) \| `s3` \| `python` \| `datalad`   |
| `--no-data`               | Sidecars only — skip annexed binaries               |
| `--stimuli`               | Add `stimuli/` scope                                |
| `--derivatives`           | Add `derivatives/` scope                            |
| `--sourcedata`            | Add `sourcedata/` scope — the original pre-BIDS distribution |
| `--metadata-timeout S`    | Override 30 s metadata timeout                      |
| `--verbose`               | Echo resolved parameters before transfer            |

### BIDS filters (repeatable)

| Flag                                                    | Accepts                                |
| ------------------------------------------------------- | -------------------------------------- |
| `--subject`, `--session`, `--task`, `--run`, `--acq`    | `001` or `sub-001`                     |
| `--datatype`, `--suffix`, `--extension`                 | `eeg`, `T1w`, `.set`, …                |
| `--scope`                                               | `raw`, `derivatives`, `stimuli`, …     |
| `--pipeline`                                            | Subfolder under `derivatives/`         |
| `--entity key=value`                                    | Generic BIDS entity                    |
| `--include PATTERN`, `--exclude PATTERN`                | Path globs                             |

Labels accept either bare (`001`) or BIDS-prefixed (`sub-001`) form.

## Examples

```shell
# One subject, one task, EEG .set + sidecars
nemar-py download nm000132 \
  --subject 001 --task MMN --datatype eeg --suffix eeg --extension .set

# Derivatives from one pipeline
nemar-py download nm000132 --derivatives --pipeline eeglab --subject 001

# Metadata sweep — sidecars only, no big binaries
nemar-py download nm000132 --no-data -o nm000132-metadata

# The original pre-BIDS distribution, exactly as the authors published it
nemar-py download nm000341 --scope sourcedata -o nm000341-original
```

### `sourcedata/` — the original upstream distribution

Many NEMAR deposits carry a `sourcedata/` tree holding the **original,
pre-BIDS files** as distributed by the dataset authors, alongside a
`sourcedata_provenance.json` recording each file's upstream name, size and
SHA-256. That makes NEMAR usable as a mirror of the original source when the
upstream host is slow, gated, or gone.

`sourcedata` is **not** fetched by default — `--scope sourcedata` gets only
that tree, while `--sourcedata` adds it alongside the default `raw` scope.

```python
import nemar
nemar.download(
    dataset="nm000132",
    subject=["001", "002"],
    task="MMN", datatype="eeg", suffix="eeg", extension=".set",
)
```

## Behaviour

- Catalog (`index`, `version`, `manifest`) is fetched from `https://data.nemar.org/{dataset}/`. No auth.
- File bytes use a layered chain: **S3 → (DataLad) → HTTPS**.
  - **S3** is tried first — anonymous public-read against `nemar.s3.us-east-2.amazonaws.com`, content-addressed at `<dataset>/objects/<git-annex-key>` (`eegdash`-style direct fetch).
  - **DataLad** is an optional middle layer, active only when the `[datalad]` extra is installed *and* the dataset index advertises a `datalad_url`. Absent the extra, this layer is skipped (a missing import is caught and falls through).
  - **HTTPS** uses the manifest's file URLs, with Range/206 resume. Public annex URLs are unsigned; bucket-policy-excluded datasets may still have expiring signed URLs. A 403 tries the durable `bytes_url` once, following redirects to fresh bytes. If access is still denied, the download fails.
- BIDS root files (`dataset_description.json`, `participants.tsv`/`json`, `README*`, `CHANGES`, `LICENSE`) are always kept — even with `--include` / `--exclude`.

### Coming from openneuro-py

`dataset`, `tag`, `target_dir`, `include`, and `exclude` are shared keyword
names, not a promise of interchangeable dataset IDs or snapshots. Use NEMAR
`nm…` / `on…` IDs and NEMAR version tags; `ds…` IDs are rejected. For example,
NEMAR `on000117` v1.0.0 derives from OpenNeuro `ds000117` v1.1.0: never copy
one archive's tag to the other without checking provenance.

Without BIDS filters, explicit hidden-file includes work like openneuro-py:
`include=[".bidsignore"]` or `include=["**/.*"]`. Hidden files remain skipped
by default and by ordinary `*` / `**` patterns; excludes still apply.
Unlike current openneuro-py, `.bidsignore` is not automatically kept.
BIDS filters continue to select non-hidden BIDS paths, so do not combine
those filters with a hidden-file-only include.

```python
import nemar

# If .bidsignore is advertised by this NEMAR snapshot; otherwise raises.
# Root BIDS essentials are also retained. This example downloads files.
nemar.download(
    dataset="on000117", tag="v1.0.0", target_dir="data/on000117-v1.0.0",
    include=[".bidsignore"],
)
```

Use a separate destination from existing OpenNeuro downloads: local DOI/version
checks are NEMAR-specific. This is not a drop-in replacement for openneuro-py's
`source` or string-valued `verify_hash` options.
