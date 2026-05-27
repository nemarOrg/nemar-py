# nemar-py

A Python client for accessing public NEMAR datasets through the NEMAR data
endpoint.

## Usage

```shell
nemar-py download --dataset nm000132 --target-dir data/nm000132
```

The downloader starts from:

```text
https://data.nemar.org/{dataset}/
```

It uses the advertised dataset index, metadata URL, and version manifest. It
does not use authentication and does not fall back to S3.

To inspect the versions already advertised by NEMAR:

```shell
nemar-py versions --dataset nm000132
```

`--tag latest` resolves through the endpoint's `latest` field, and explicit
tags are validated against the endpoint's advertised `versions`.

By default, `nemar-py` uses a built-in HTTPS downloader. When the NEMAR
dataset index advertises a DataLad sibling and the optional `datalad`
extra is installed, `nemar-py` clones it first and runs `datalad get`
for the requested files; HTTPS is the automatic fallback when anything
in the DataLad path fails.

```shell
pip install nemar-py[datalad]
```

Pick the backend explicitly if you need to:

```shell
nemar-py download --dataset nm000132 --downloader datalad   # DataLad, HTTPS fallback
nemar-py download --dataset nm000132 --downloader python    # HTTPS only (built-in)
```

`--downloader datalad` still falls back to HTTPS when the dataset index does
not advertise a DataLad URL or when DataLad fails for any reason — the
two-layer model is the steady-state contract.

```shell
nemar-py download --dataset nm000132 \
                  --tag v1.1.1 \
                  --subject 001 \
                  --task MMN \
                  --datatype eeg \
                  --suffix eeg \
                  --extension .set
```

Semantic BIDS filters can be repeated. They accept either BIDS-prefixed labels
(`sub-001`, `task-MMN`, `run-01`) or plain labels (`001`, `MMN`, `01`).
Less common BIDS entities can be selected with `--entity key=value`, for example
`--entity desc=clean`.

Dataset-level BIDS folders are selected with `--scope`:

```shell
# Download stimuli for one task
nemar-py download --dataset nm000132 --scope stimuli --task MMN

# Download derivatives from one pipeline
nemar-py download --dataset nm000132 \
                  --scope derivatives \
                  --pipeline eeglab \
                  --subject 001
```

`--include` and `--exclude` remain available as path-level refinements. When
combined with semantic BIDS filters, `--include` narrows the semantic result and
`--exclude` removes paths from it.

Essential BIDS root files such as `dataset_description.json`,
`participants.tsv`, `participants.json`, `README`, `README.md`, `CHANGES`, and
`LICENSE` are kept when present, even when include or exclude patterns are used.

## Python

```python
import nemar

index = nemar.fetch_dataset_index(dataset="nm000132")
versions = nemar.list_dataset_versions(dataset="nm000132")

nemar.download(
    dataset="nm000132",
    target_dir="data/nm000132",
    subject=["001", "002"],
    task="MMN",
    datatype="eeg",
    suffix="eeg",
    extension=".set",
)

nemar.download(
    dataset="nm000132",
    scope="stimuli",
    task="MMN",
)
```

## Reference implementations

A snapshot of [openneuro-py](https://github.com/openneuro-py/openneuro-py) is
kept under [`docs/reference-implementations/`](docs/reference-implementations/)
as a design reference. It is not built, tested, or imported by `nemar-py`.
