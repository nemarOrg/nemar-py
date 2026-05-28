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
