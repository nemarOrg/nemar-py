"""Models for NEMAR dataset metadata and manifests."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urljoin

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nemar._endpoint import DataEndpoint


class DatasetVersion(BaseModel):
    """A version advertised by ``https://data.nemar.org/{dataset}/``."""

    model_config = ConfigDict(extra="allow")

    version: str
    doi: str | None = None
    created_at: str | None = None
    manifest_url: str
    browse_url: str | None = None


class DatasetIndex(BaseModel):
    """Dataset index advertised by the NEMAR data endpoint."""

    model_config = ConfigDict(extra="allow")

    dataset_id: str
    latest: str
    metadata_url: str | None = None
    versions: list[DatasetVersion] = Field(default_factory=list)

    def resolve_version(self, tag: str | None = None) -> DatasetVersion:
        """Resolve ``None``, ``latest``, or an explicit advertised version."""
        requested = self.latest if tag in (None, "latest") else tag
        for version in self.versions:
            if version.version == requested:
                return version

        available = ", ".join(v.version for v in self.versions) or "none"
        raise RuntimeError(
            f'The requested NEMAR version "{requested}" does not exist for '
            f"{self.dataset_id}. Available versions: {available}"
        )


@dataclass(frozen=True)
class DatasetFile:
    """A file resolved from a NEMAR manifest."""

    path: str
    url: str
    size: int | None = None
    sha256: str | None = None
    md5: str | None = None


def parse_dataset_index(payload: Any) -> DatasetIndex:
    """Validate a NEMAR dataset index payload."""
    try:
        return DatasetIndex.model_validate(payload)
    except ValidationError as exc:
        raise RuntimeError(
            "The NEMAR data endpoint returned an unexpected dataset index "
            f"payload: {exc.errors(include_input=False)}"
        ) from exc


@dataclass(frozen=True)
class VersionManifest:
    """A parsed NEMAR version manifest.

    Wraps the inventory of :class:`DatasetFile` entries together with the
    manifest URL they were resolved against and the :class:`DataEndpoint`
    whose origin they must respect. Duplicate-path detection and the
    origin-scoping rule live on the value type itself so the invariants
    survive past the parser.
    """

    files: tuple[DatasetFile, ...]
    manifest_url: str
    endpoint: DataEndpoint

    @classmethod
    def parse(
        cls,
        payload: Any,
        *,
        manifest_url: str,
        endpoint: DataEndpoint,
    ) -> VersionManifest:
        """Parse common manifest shapes into a ``VersionManifest``.

        The public manifest schema is intentionally treated as a small,
        tolerant seam because NEMAR is still exposing the versioned manifest
        links. The downloader remains strict about the resolved URL origin
        (every file's URL must be ``endpoint.assert_within``-valid) and
        about duplicate paths within a manifest.
        """
        entries = list(_iter_manifest_entries(payload))
        files = tuple(
            _entry_to_file(entry, manifest_url=manifest_url, endpoint=endpoint)
            for entry in entries
        )
        if not files:
            raise RuntimeError(
                "The NEMAR manifest did not contain any downloadable files."
            )

        seen: set[str] = set()
        duplicates: set[str] = set()
        for file in files:
            if file.path in seen:
                duplicates.add(file.path)
            seen.add(file.path)
        if duplicates:
            examples = ", ".join(sorted(duplicates)[:5])
            raise RuntimeError(
                f"The NEMAR manifest contains duplicate paths: {examples}"
            )

        return cls(files=files, manifest_url=manifest_url, endpoint=endpoint)

    def __iter__(self) -> Iterator[DatasetFile]:
        return iter(self.files)

    def __len__(self) -> int:
        return len(self.files)


def _iter_manifest_entries(payload: Any) -> Iterable[Any]:
    if isinstance(payload, list):
        yield from payload
        return

    if not isinstance(payload, Mapping):
        raise RuntimeError("The NEMAR manifest must be a JSON object or array.")

    for key in ("files", "entries", "manifest", "objects"):
        value = payload.get(key)
        if isinstance(value, list):
            yield from value
            return
        if isinstance(value, Mapping):
            yield from _mapping_entries(value)
            return

    if payload and all(isinstance(key, str) for key in payload):
        yield from _mapping_entries(payload)
        return

    raise RuntimeError(
        "The NEMAR manifest JSON shape is not recognized. Expected a list of "
        "files or an object with a files/entries/manifest key."
    )


def _mapping_entries(value: Mapping[str, Any]) -> Iterable[Any]:
    for path, metadata in value.items():
        if isinstance(metadata, Mapping):
            entry = dict(metadata)
            entry.setdefault("path", path)
            yield entry
        elif isinstance(metadata, str):
            yield {"path": path, "url": metadata}
        else:
            yield {"path": path}


def _entry_to_file(
    entry: Any, *, manifest_url: str, endpoint: DataEndpoint
) -> DatasetFile:
    if isinstance(entry, str):
        raw_path = entry
        raw_url: Any = None
        size = None
        sha256 = None
        md5 = None
    elif isinstance(entry, Mapping):
        raw_path = _first_value(
            entry,
            "path",
            "filename",
            "name",
            "relative_path",
            "relativePath",
            "key",
        )
        raw_url = _first_value(entry, "url", "download_url", "downloadUrl", "href")
        if raw_url is None and isinstance(entry.get("urls"), list) and entry["urls"]:
            raw_url = entry["urls"][0]
        size = _coerce_size(_first_value(entry, "size", "bytes", "size_bytes"))
        sha256 = _coerce_hash(_first_value(entry, "sha256", "sha256sum"))
        md5 = _coerce_hash(_first_value(entry, "md5", "md5sum", "etag"))
    else:
        raise RuntimeError(f"Unsupported manifest entry type: {type(entry).__name__}")

    path = _validate_relative_path(raw_path)
    url = _resolve_file_url(raw_url, path=path, manifest_url=manifest_url)
    endpoint.assert_within(url)

    return DatasetFile(path=path, url=url, size=size, sha256=sha256, md5=md5)


def _first_value(entry: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = entry.get(key)
        if value not in (None, ""):
            return value
    return None


def _validate_relative_path(raw_path: Any) -> str:
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError("Manifest entry is missing a non-empty relative path.")
    if "\x00" in raw_path:
        raise RuntimeError(f"Manifest path contains a NUL byte: {raw_path!r}")

    path = PurePosixPath(raw_path)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"Unsafe manifest path: {raw_path!r}")
    return path.as_posix()


def _resolve_file_url(raw_url: Any, *, path: str, manifest_url: str) -> str:
    if raw_url is None:
        version_root = manifest_url.rsplit("/", 1)[0] + "/"
        return urljoin(version_root, path)
    if not isinstance(raw_url, str) or not raw_url:
        raise RuntimeError(f"Manifest URL for {path} is not a non-empty string.")
    return urljoin(manifest_url, raw_url)


def _coerce_size(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise RuntimeError(f"Manifest file size is not an integer: {value!r}")


def _coerce_hash(value: Any) -> str | None:
    """Normalize a manifest hash value to a comparable hex string.

    The NEMAR manifest sometimes inherits hashes from upstream ETag
    headers, which may carry the ``W/`` weak-ETag marker and outer
    quotes (e.g. ``W/"abcdef"``). We strip both so the value compares
    equal to the lowercase hex produced by ``hashlib``. Uppercase hex
    is left as-is here; the verifier lowercases both sides at compare
    time.
    """
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise RuntimeError(f"Manifest checksum is not a string: {value!r}")
    value = value.strip()
    # ``W/"abc"`` -> strip ``W/`` -> ``"abc"`` -> strip outer quotes -> ``abc``.
    # We strip ``W/`` both before and after the quote strip so we cover
    # the ``"W/abc"`` ordering some servers emit.
    if value.startswith("W/"):
        value = value[2:]
    value = value.strip('"')
    if value.startswith("W/"):
        value = value[2:]
    return value or None
