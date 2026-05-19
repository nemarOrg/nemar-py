"""Factories for NEMAR index / manifest / dataset_description payloads."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def make_index(
    *,
    dataset: str = "nm000132",
    latest: str = "v1.0.0",
    versions: Iterable[Mapping[str, Any]] | None = None,
    metadata_url: str | None = "metadata.json",
) -> dict[str, Any]:
    """Build a valid DatasetIndex payload."""
    if versions is None:
        versions = [
            {
                "version": "v1.0.0",
                "doi": "10.82901/nemar." + dataset,
                "created_at": "2026-01-01T00:00:00Z",
                "manifest_url": "v1.0.0/manifest.json",
                "browse_url": "v1.0.0/",
            }
        ]
    return {
        "dataset_id": dataset,
        "latest": latest,
        "metadata_url": metadata_url,
        "versions": list(versions),
    }


def make_manifest_list(
    files: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build a list-shaped manifest payload."""
    return [dict(entry) for entry in files]


def make_manifest_entry(
    *,
    path: str,
    content: bytes,
    base_url: str = "",
    with_sha256: bool = True,
    with_md5: bool = False,
    with_size: bool = True,
) -> dict[str, Any]:
    """Build one manifest entry with optional size / sha256 / md5."""
    entry: dict[str, Any] = {"path": path}
    if base_url:
        entry["url"] = base_url.rstrip("/") + "/" + path
    if with_size:
        entry["size"] = len(content)
    if with_sha256:
        entry["sha256"] = hashlib.sha256(content).hexdigest()
    if with_md5:
        entry["md5"] = hashlib.md5(content).hexdigest()
    return entry


def make_dataset_description(
    *,
    dataset: str = "nm000132",
    version: str | None = "v1.0.0",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal BIDS dataset_description.json payload."""
    payload: dict[str, Any] = {
        "Name": dataset,
        "BIDSVersion": "1.8.0",
        "DatasetDOI": "doi:10.82901/nemar." + dataset,
    }
    if version is not None:
        payload["Version"] = version
    if extra:
        payload.update(extra)
    return payload


def write_dataset_description(
    target_dir: Path,
    payload: Mapping[str, Any],
) -> Path:
    """Write `dataset_description.json` to ``target_dir``."""
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / "dataset_description.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out


@dataclass(frozen=True)
class FakeBlob:
    """A deterministic byte blob with its size and hashes."""

    content: bytes
    size: int
    sha256: str
    md5: str


def make_blob(
    *,
    seed: int,
    size_bytes: int,
) -> FakeBlob:
    """Build a deterministic byte blob of ``size_bytes``."""
    import random

    rng = random.Random(seed)
    content = bytes(rng.randrange(256) for _ in range(size_bytes))
    return FakeBlob(
        content=content,
        size=size_bytes,
        sha256=hashlib.sha256(content).hexdigest(),
        md5=hashlib.md5(content).hexdigest(),
    )
