"""Shared primitive constants with no ``nemar`` dependencies.

A leaf module so the metadata client (:mod:`nemar._client`), the
request normalizer (:mod:`nemar._request`), the orchestrator
(:mod:`nemar._download`), and the CLI (:mod:`nemar.__main__`) all read
the same dataset-id pattern and default origin without importing one
another. These previously lived in :mod:`nemar._request`, which forced
the lower-level ``_client`` to import the higher-level request module
just for two constants.
"""

from __future__ import annotations

import re

DEFAULT_DATA_URL = "https://data.nemar.org/"
DATASET_ID_RE = re.compile(r"^(nm|on)\d{6}$")

#: Suffix for the staging file a transfer writes into before it is renamed
#: into place. Downloading straight to the final path means an interrupted
#: transfer leaves a short file that is indistinguishable from a complete one
#: until something re-hashes it; staging + :func:`os.replace` makes a partial
#: file impossible to mistake for a finished one.
PARTIAL_SUFFIX = ".part"

#: Read/write chunk for streamed bodies and hashing (1 MiB).
CHUNK_BYTES = 1024 * 1024

#: Objects at or above this size are fetched in parallel ranged parts rather
#: than as one stream. Below it the per-part overhead outweighs the gain.
MULTIPART_THRESHOLD_BYTES = 64 * 1024 * 1024

#: Size of each ranged part of a multipart download.
MULTIPART_CHUNK_BYTES = 16 * 1024 * 1024
