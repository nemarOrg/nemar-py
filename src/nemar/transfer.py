"""Single-file transfer primitive — public submodule.

Re-export of :func:`nemar._transfer.download_one`, the smallest useful
transfer operation. Callers reach for it when they want to fetch a
single :class:`~nemar.models.VersionManifest` entry without the
metadata orchestrator (e.g., resuming one file after an interrupted
batch). Most users want :func:`nemar.download` instead, which owns the
full index → manifest → BIDS-selection → transfer flow.
"""

from __future__ import annotations

from nemar._transfer import download_one as download_one
