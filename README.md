# nemar-py

Python + CLI client for downloading public **NEMAR** datasets (BIDS / EEG / MEG / iEEG) from `data.nemar.org`.

## Install

```shell
pip install nemar-py
```

DataLad and the `git-annex` binary both ship in the default install — no system package manager step required (the `git-annex` PyPI wheel from `psychoinformatics-de` bundles the Haskell binary for Linux, macOS, and Windows).

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
| `--downloader BACKEND`    | `auto` (default) \| `python` \| `datalad`           |
| `--no-data`               | Sidecars only — skip annexed binaries               |
| `--stimuli`               | Add `stimuli/` scope                                |
| `--derivatives`           | Add `derivatives/` scope                            |
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
```

```python
import nemar
nemar.download(
    dataset="nm000132",
    subject=["001", "002"],
    task="MMN", datatype="eeg", suffix="eeg", extension=".set",
)
```

## Behaviour

- Starts from `https://data.nemar.org/{dataset}/`. No auth, no S3 fallback.
- **DataLad is the default first layer**; HTTPS resumes via Range/206 on any DataLad failure (missing sibling, network error, …).
- BIDS root files (`dataset_description.json`, `participants.tsv`/`json`, `README*`, `CHANGES`, `LICENSE`) are always kept — even with `--include` / `--exclude`.

