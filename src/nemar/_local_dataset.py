"""On-disk view of a NEMAR dataset.

The orchestrator in :mod:`nemar._download` is concerned with HTTP and
transfer scheduling; reasoning about *what is already on disk* in the
target directory is a separate concern modeled here. A
:class:`LocalDataset` is a small immutable value type that owns:

* the classmethod that inspects the target dir (the only disk I/O), and
* the assertion that decides whether a fresh download is compatible
  with the local state (no I/O -- pure reasoning over already-parsed
  fields).

The split exists so that ``_run()`` does not have to know about JSON
parsing, BOM-tolerant decoding, or DOI/version semantics, and so that
the rules themselves can be exercised in isolation without standing up
a transfer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from nemar._errors import LocalTargetError, LocalVersionMismatchError
from nemar._request import _normalize_version_tag

_DATASET_DESCRIPTION_FILENAME = "dataset_description.json"
_DOI_PREFIX_TEMPLATE = "10.82901/nemar.{dataset}"


@dataclass(frozen=True)
class LocalDataset:
    """The dataset identity recovered from a local target directory.

    Construction is via :meth:`from_dir`, which returns ``None`` when
    the directory is empty / non-existent / has no
    ``dataset_description.json`` -- those are the cases where a fresh
    download has nothing to clash with and the orchestrator should
    simply proceed.

    A non-``None`` instance carries the parsed identity:

    * ``dataset_id`` -- the bare dataset id (``"nm000132"``) recovered
      from the DOI prefix.
    * ``version_tag`` -- the normalized version tag (``"v1.0.0"``), or
      ``None`` when the local description is silent on Version.

    All compatibility decisions live in :meth:`assert_compatible_with`,
    which is pure: it never re-reads the filesystem.
    """

    dataset_id: str
    version_tag: str | None

    @classmethod
    def from_dir(cls, path: Path) -> LocalDataset | None:
        """Inspect ``path`` and return a :class:`LocalDataset` if one exists.

        Returns ``None`` in three "no compatibility check needed" cases:

        1. ``path`` does not exist.
        2. ``path`` exists but is empty.
        3. ``path`` is non-empty but has no ``dataset_description.json``
           (the "let interrupted downloads resume" case -- callers
           still want to log this; see the note in :mod:`nemar._download`).

        Returns a populated instance when ``dataset_description.json``
        is present. Raises :class:`RuntimeError` if the file is present
        but malformed (bad JSON, missing/non-string ``DatasetDOI``,
        non-string ``Version``) so silent corruption can never sneak
        through.
        """
        if not path.exists():
            return None
        if next(path.iterdir(), None) is None:
            return None

        dataset_description_path = path / _DATASET_DESCRIPTION_FILENAME
        if not dataset_description_path.exists():
            return None

        # ``utf-8-sig`` strips a UTF-8 BOM if present and is otherwise
        # equivalent to ``utf-8``. Some editors (legacy Windows
        # Notepad, Excel CSV-export pipelines) prepend a BOM that
        # would otherwise turn the first key into ``"﻿DatasetDOI"``
        # and silently fail the DOI check.
        try:
            payload = json.loads(
                dataset_description_path.read_text(encoding="utf-8-sig")
            )
        except json.JSONDecodeError as exc:
            raise LocalTargetError(
                f"Could not parse local {dataset_description_path} as JSON."
            ) from exc

        doi = payload.get("DatasetDOI")
        if doi is None:
            raise LocalTargetError(
                'Local "dataset_description.json" does not contain "DatasetDOI".'
            )
        if not isinstance(doi, str):
            raise LocalTargetError('Local "DatasetDOI" must be a string.')
        doi = doi.removeprefix("doi:")

        dataset_id = doi.removeprefix("10.82901/nemar.")

        version = payload.get("Version")
        if version is not None and not isinstance(version, str):
            raise LocalTargetError('Local "Version" must be a string when present.')
        version_tag = _normalize_version_tag(version) if version is not None else None

        return cls(dataset_id=dataset_id, version_tag=version_tag)

    def assert_compatible_with(self, *, dataset: str, tag: str) -> None:
        """Raise if downloading ``dataset`` ``tag`` here would clobber data.

        * Raises :class:`RuntimeError` if the local DOI prefix names a
          *different* dataset.
        * Raises :class:`RuntimeError` if the DOI matches but no
          Version was recorded -- continuing would risk silently
          mixing two versions of the same dataset.
        * Raises :class:`FileExistsError` if the DOI matches but the
          Version is a different tag than the one requested.

        Pure: no filesystem access. ``self`` already carries the
        parsed identity.
        """
        expected_doi = _DOI_PREFIX_TEMPLATE.format(dataset=dataset)
        local_doi = _DOI_PREFIX_TEMPLATE.format(dataset=self.dataset_id)
        if not local_doi.startswith(expected_doi):
            raise LocalTargetError(
                "The target directory appears to contain a different NEMAR dataset. "
                f'Local DatasetDOI: "{local_doi}". Requested dataset: {dataset}.'
            )

        if self.version_tag is None:
            raise LocalTargetError(
                'Local "dataset_description.json" matches the requested dataset DOI '
                'but does not contain a "Version" field. This could lead to '
                "overwriting data from a different version. Use an empty target "
                "directory, or remove the existing files if you intend to "
                "re-download."
            )

        if self.version_tag != tag:
            raise LocalVersionMismatchError(
                f"You requested {dataset} {tag}, but {self.version_tag} exists "
                "locally in the target directory. Use an empty target directory "
                "or request the same version."
            )
