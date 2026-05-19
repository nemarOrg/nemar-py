# Project Context

## Domain Language

- **NEMAR data endpoint**: The public HTTPS origin at `https://data.nemar.org/`.
  This is the canonical entry point for downloads in this client. Modeled
  internally as the `DataEndpoint` value type (`src/nemar/_endpoint.py`),
  which owns HTTPS validation, trailing-slash normalization, and the
  origin-scoping rule applied to every dataset-index, metadata, manifest,
  and per-file URL (including after HTTP redirects).
- **Dataset index**: The JSON document at `/{dataset}/` that advertises the
  latest version, metadata URL, and version manifest URLs.
- **NEMAR metadata**: The dataset-level metadata document advertised by the
  dataset index. It describes scientific metadata, not the file inventory.
- **Version manifest**: The JSON file advertised for a specific dataset version.
  It is the file inventory used for BIDS selection and download planning.
  Modeled internally as the `VersionManifest` value type
  (`src/nemar/_models.py`), which wraps the parsed `DatasetFile` tuple, the
  source manifest URL, and the `DataEndpoint` whose origin every file URL
  must respect.
- **BIDS file selection**: Include/exclude pattern handling over manifest paths,
  with essential root-level BIDS files kept when present.
- **BIDS query**: Semantic selection over parsed BIDS entities such as subject,
  session, task, run, acquisition, datatype, suffix, extension, and generic
  `key=value` entities.
- **Dataset scope**: The top-level BIDS-adjacent area selected by `raw`,
  `derivatives`, `stimuli`, `sourcedata`, or `code`. Derivatives may additionally
  be selected by pipeline name from `derivatives/<pipeline>/`.
- **Transfer backend**: The concrete download implementation used after file
  selection. `aria2c` is preferred for speed; the built-in Python downloader is
  the fallback.

## Architectural Notes

- The client is intentionally scoped to the NEMAR data endpoint. Manifest file
  URLs outside the configured data origin are refused.
- Dataset version control uses the dataset index already advertised by
  `data.nemar.org`: `latest`, `versions`, DOI, created date, and manifest URL.
- No authentication module exists because the public NEMAR data endpoint is the
  supported interface for this entry point.
