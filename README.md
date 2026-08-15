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
  - **HTTPS** through `data.nemar.org` is the always-available fallback, with Range/206 resume.
- BIDS root files (`dataset_description.json`, `participants.tsv`/`json`, `README*`, `CHANGES`, `LICENSE`) are always kept — even with `--include` / `--exclude`.

