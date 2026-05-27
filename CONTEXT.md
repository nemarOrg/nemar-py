# Project Context

## Domain Language

- **NEMAR data endpoint**: The public HTTPS origin at `https://data.nemar.org/`.
  This is the canonical entry point for downloads in this client. Modeled
  internally as the `DataEndpoint` value type (`src/nemar/_endpoint.py`),
  which owns HTTPS validation, trailing-slash normalization, and the
  origin-scoping rule applied to every dataset-index, metadata, manifest,
  and per-file URL (including after HTTP redirects).
- **NEMAR client**: A long-lived handle to the NEMAR data endpoint, modeled
  as the `NEMARClient` context manager (`src/nemar/_client.py`). Owns one
  `httpx.Client` and exposes `fetch_index`, `fetch_metadata`,
  `fetch_manifest` against the configured `DataEndpoint`. The orchestrator
  and the public metadata helpers (`fetch_dataset_index`,
  `list_dataset_versions`) construct one internally; library callers can
  hold one open across many datasets to amortize TLS and connection-pool
  cost.
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
  selection. Modeled as adapters of one seam, all exposing the
  `TransferBackend` protocol in `src/nemar/_transfer.py`. One HTTPS
  adapter and one optional first-layer adapter exist:
  - `PythonBackend` — the HTTPS adapter. Thread pool over a shared
    `httpx.Client` with a per-batch connection pool; owns the Range/206
    resume, HTTP 416 recovery, and per-file retry semantics.
  - `DataLadBackend` (`src/nemar/_datalad.py`) — optional first layer that
    clones the dataset's DataLad sibling advertised by the NEMAR index
    (`DatasetIndex.datalad_url`) and runs `datalad get` against the
    BIDS-selected paths. Requires the `datalad` extra
    (`pip install nemar-py[datalad]`). A missing optional dep, clone
    failure, checkout failure, or `get` failure becomes a
    `DataLadError` (`src/nemar/_errors.py`) — a subclass of `TransferError`
    the layered wrapper catches narrowly.
  - `LayeredBackend` (`src/nemar/_datalad.py`) — wraps `DataLadBackend`
    as a primary over `PythonBackend`. On `DataLadError` it writes a
    tqdm notice and re-runs the same file set through the HTTPS adapter.
    Every other transfer failure propagates. This is the steady-state
    contract whenever a `datalad_url` is known: DataLad first, HTTPS
    second.

  Selection is policy-driven via `TransferOptions.backend`
  ("auto" | "python" | "datalad") and resolved by `select_backend` in
  `src/nemar/_transfer.py`:
  - `auto` + `datalad_url` advertised → `LayeredBackend` over
    `PythonBackend`.
  - `auto` + no `datalad_url` → `PythonBackend`.
  - `datalad` + `datalad_url` advertised → `LayeredBackend` over
    `PythonBackend`. The fallback to HTTPS happens **even when
    `datalad` was requested explicitly** — the two-layer contract is
    steady-state.
  - `datalad` + no `datalad_url` → `PythonBackend`, with a tqdm notice.
  - `python` → opt-out from the DataLad layer; HTTPS only.

  The backend choice and its runtime knobs (concurrency, stream
  timeout) are bundled in the `TransferOptions` value type
  (`src/nemar/_request.py`), which travels inside a `DownloadRequest`.
  The `datalad_url` and resolved version tag are threaded by `_run`
  (`src/nemar/_download.py`) into `select_backend` *after* the index
  fetch, so they do not appear on `TransferOptions` itself.
- **Download request**: A `DownloadRequest` value type
  (`src/nemar/_request.py`) bundling the normalized dataset identity,
  `DataEndpoint`, requested tag, target path, BIDS query, include/exclude
  patterns, and the `TransferOptions` / `RetryPolicy` / `VerifyPolicy`
  policies. Library and CLI callers construct one via
  `DownloadRequest.from_kwargs(...)`; the orchestrator's six algorithmic
  steps (`_run`) execute against it.
- **Transport**: The JSON-fetch-with-retries primitive used by every metadata
  read against the NEMAR data endpoint. Modeled internally as `fetch_json` in
  `src/nemar/_transport.py`. Owns the redirect-origin check, retryable-status
  classification (delegated to `RetryPolicy`), and exhausted-retry error
  reshaping.
- **Verification**: The single predicate "does this local file satisfy its
  manifest entry?", modeled in `src/nemar/_verification.py` as
  `check(file, path, policy) -> VerifyResult`. Used as a filter before transfer
  (skip already-complete files) and as the assertion after transfer
  (`assert_all_present`). Outcomes are `OK`, `MISSING`, `SIZE_MISMATCH`,
  `HASH_MISMATCH`, and `ERROR_SENTINEL` (a tiny `{"error": ...}` JSON file
  the NEMAR endpoint writes when an object is unavailable). The same module
  exposes `detect_case_collisions`, the pre-transfer guard against
  case-insensitive filesystem clashes that would silently overwrite data.
- **Local dataset**: The on-disk view of a NEMAR dataset, modeled as the
  `LocalDataset` value type (lives in `src/nemar/_download.py`). Inspects a
  target directory and asserts that any pre-existing
  `dataset_description.json` identifies the same dataset and the requested
  version, refusing to overwrite a different version's files. Construction
  (`LocalDataset.from_dir`) is the only disk-touching step; the
  `assert_compatible_with` instance method is pure reasoning over the
  parsed identity.
- **Single-file download**: The smallest useful transfer operation, modeled
  as the public function `nemar.download_one(file, target_path, *,
  client=None, retry=None, verify=None)` in `src/nemar/_transfer.py`. Owns
  range-resume, HTTP 416 recovery, jittered retry, and final verify for one
  URL → one path. The `PythonBackend` uses it internally per file (via a
  shared `_stream_with_retries` helper); external callers can reach it
  without invoking the full bulk orchestrator.
- **Error hierarchy**: All library-raised failures inherit from
  `nemar.NemarError` (which itself subclasses `RuntimeError` for backward
  compatibility with legacy callers). Subclasses align with the modules
  that own each failure mode: `EndpointError` (off-origin),
  `DatasetIndexError` (index payload / version resolution), `ManifestError`
  (manifest shape / content), `SelectionError` (BIDS selection, case
  collisions), `TransportError` (JSON-fetch retries exhausted, HTTP,
  decode), `TransferError` (bytes-on-the-wire), `DataLadError` (DataLad /
  git-annex transfer failure — subclass of `TransferError` so the
  layered backend catches it narrowly for fallback), `VerificationError`
  (local file fails its manifest entry), `LocalTargetError` (target dir
  holds a different dataset), and `LocalVersionMismatchError` which
  additionally subclasses `FileExistsError` so legacy callers that catch
  the builtin keep working. Defined in `src/nemar/_errors.py`. `ValueError` stays for
  input validation at the public boundary.

## Architectural Notes

- The client is intentionally scoped to the NEMAR data endpoint. Manifest file
  URLs outside the configured data origin are refused.
- Dataset version control uses the dataset index already advertised by
  `data.nemar.org`: `latest`, `versions`, DOI, created date, and manifest URL.
- No authentication module exists because the public NEMAR data endpoint is the
  supported interface for this entry point.
