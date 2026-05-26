"""Transfer primitives — public submodule.

Re-exports of the two byte-on-the-wire primitives:

* :func:`download_one` — fetch a single :class:`DatasetFile` to a
  given target path. The smallest useful transfer operation.
* :func:`download_files` — bulk variant for a sequence of
  :class:`DatasetFile` entries. Skips the full
  :func:`nemar.download` orchestrator (no dataset index, no BIDS
  query) so external clients (eegdash, custom selectors) can compose
  their own selection logic on top of a parsed manifest.

Most end users want :func:`nemar.download` instead, which owns the
full index → manifest → BIDS-selection → transfer flow.
"""

from __future__ import annotations

from nemar._transfer import download_files as download_files
from nemar._transfer import download_one as download_one
