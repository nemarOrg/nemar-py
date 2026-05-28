"""Unit tests for the DataLad transfer layer.

The DataLad backend is the first layer of the two-layer transfer model
described in :mod:`nemar._datalad`. These tests pin four contracts:

1. The DataLad import (one bundle: api + Dataset + exception classes)
   is the single seam tests substitute to drive the DataLad code path
   without exercising the real ``datalad`` package.
2. Every documented DataLad failure surface (install error, ``Dataset``
   open error, checkout error, ``get`` error) becomes
   :class:`DataLadError`. Programming bugs (``TypeError`` etc.)
   propagate unchanged so a real defect surfaces immediately.
3. Existing on-disk clones are reused: the second run of a request
   into a populated ``target_dir`` opens via :class:`Dataset` instead
   of re-installing.
4. :class:`LayeredBackend` catches :class:`DataLadError` specifically
   and runs the fallback over the same file set; every other transfer
   failure propagates.

The tests use a tiny fake DataLad module bundle that records the calls
the backend made, so the contract is verified at the public seam rather
than against the real ``datalad`` package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from nemar import _datalad
from nemar._backend import TransferOptions
from nemar._datalad import DataLadBackend, _DataLadModules
from nemar._models import DatasetFile
from nemar._retry import RetryPolicy
from nemar._transfer import LayeredBackend
from nemar._verification import VerifyPolicy
from nemar.errors import DataLadError, TransferError

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeCommandError(Exception):
    """Stand-in for ``datalad.support.exceptions.CommandError``."""


class _FakeIncompleteResultsError(Exception):
    """Stand-in for ``datalad.support.exceptions.IncompleteResultsError``."""


_FAKE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    _FakeCommandError,
    _FakeIncompleteResultsError,
)


@dataclass
class _FakeRepo:
    checked_out: list[str] = field(default_factory=list)
    checkout_raises: Exception | None = None

    def checkout(self, revision: str) -> None:
        if self.checkout_raises is not None:
            raise self.checkout_raises
        self.checked_out.append(revision)


@dataclass
class _FakeDataset:
    repo: _FakeRepo = field(default_factory=_FakeRepo)
    got: list[tuple[list[str], int]] = field(default_factory=list)
    get_raises: Exception | None = None

    def get(self, *, path: list[str], jobs: int) -> None:
        if self.get_raises is not None:
            raise self.get_raises
        self.got.append((list(path), jobs))


@dataclass
class _FakeApi:
    """Minimal stand-in for ``datalad.api``."""

    dataset: _FakeDataset
    installed: list[tuple[str, str]] = field(default_factory=list)
    install_raises: Exception | None = None

    def install(self, *, source: str, path: str) -> _FakeDataset:
        if self.install_raises is not None:
            raise self.install_raises
        self.installed.append((source, path))
        return self.dataset


@dataclass
class _FakeBundle:
    """All three DataLad surfaces the backend touches.

    Exposed as a flat dataclass so individual tests can override
    behavior (``install_raises`` etc.) without rebuilding the bundle.
    """

    dataset: _FakeDataset = field(default_factory=_FakeDataset)
    open_calls: list[str] = field(default_factory=list)
    open_raises: Exception | None = None
    api: _FakeApi = field(init=False)

    def __post_init__(self) -> None:
        self.api = _FakeApi(dataset=self.dataset)

    def open_existing(self, path: str) -> _FakeDataset:
        if self.open_raises is not None:
            raise self.open_raises
        self.open_calls.append(path)
        return self.dataset

    def as_modules(self) -> _DataLadModules:
        return _DataLadModules(
            api=self.api,
            Dataset=self.open_existing,
            exceptions=_FAKE_EXCEPTIONS,
        )


@dataclass
class _RecordingBackend:
    """Captures ``transfer`` calls for assertions in LayeredBackend tests."""

    calls: list[tuple[tuple[str, ...], Path]] = field(default_factory=list)
    raises: Exception | None = None

    def transfer(self, files, *, target_dir, options, verify, retry) -> None:
        if self.raises is not None:
            raise self.raises
        self.calls.append((tuple(f.path for f in files), target_dir))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _options(*, concurrency: int = 4) -> TransferOptions:
    return TransferOptions(
        backend="datalad",
        max_concurrent_downloads=concurrency,
        stream_timeout=60.0,
    )


def _files() -> list[DatasetFile]:
    return [
        DatasetFile(path="dataset_description.json", url="https://x/a"),
        DatasetFile(path="sub-001/eeg/sub-001_task-rest_eeg.set", url="https://x/b"),
    ]


@pytest.fixture
def fake_bundle(monkeypatch: pytest.MonkeyPatch) -> _FakeBundle:
    bundle = _FakeBundle()
    monkeypatch.setattr(_datalad, "_import_datalad", bundle.as_modules)
    return bundle


# ---------------------------------------------------------------------------
# DataLadBackend
# ---------------------------------------------------------------------------


class TestDataLadBackendHappyPath:
    def test_install_checkout_get_in_order(
        self, fake_bundle: _FakeBundle, tmp_path: Path
    ) -> None:
        """install → checkout(revision) → get(paths) is the contract."""
        backend = DataLadBackend(
            datalad_url="https://github.com/OpenNeuroDatasets/ds000132.git",
            revision="v1.0.0",
        )
        files = _files()

        backend.transfer(
            files,
            target_dir=tmp_path,
            options=_options(),
            verify=VerifyPolicy(),
            retry=RetryPolicy.default(),
        )

        assert fake_bundle.api.installed == [
            (
                "https://github.com/OpenNeuroDatasets/ds000132.git",
                str(tmp_path),
            )
        ]
        assert fake_bundle.dataset.repo.checked_out == ["v1.0.0"]
        assert fake_bundle.dataset.got == [
            (
                [
                    "dataset_description.json",
                    "sub-001/eeg/sub-001_task-rest_eeg.set",
                ],
                4,
            )
        ]

    def test_no_revision_skips_checkout(
        self, fake_bundle: _FakeBundle, tmp_path: Path
    ) -> None:
        """Omitted revision leaves the clone at HEAD."""
        DataLadBackend(datalad_url="https://x/repo.git").transfer(
            _files(),
            target_dir=tmp_path,
            options=_options(),
            verify=VerifyPolicy(),
            retry=RetryPolicy.default(),
        )
        assert fake_bundle.dataset.repo.checked_out == []

    def test_empty_files_is_a_noop_after_install(
        self, fake_bundle: _FakeBundle, tmp_path: Path
    ) -> None:
        """An empty pending set still clones (cheap) but does not call get."""
        DataLadBackend(datalad_url="https://x/repo.git").transfer(
            [],
            target_dir=tmp_path,
            options=_options(),
            verify=VerifyPolicy(),
            retry=RetryPolicy.default(),
        )
        assert fake_bundle.api.installed != []  # clone happened
        assert fake_bundle.dataset.got == []  # but no get

    def test_existing_clone_is_reused_not_reinstalled(
        self, fake_bundle: _FakeBundle, tmp_path: Path
    ) -> None:
        """A pre-existing ``.datalad`` marker opens the clone in-place.

        Pins the idempotency contract: re-running a request against a
        populated target reuses the existing clone via the
        :class:`Dataset` constructor instead of calling
        :func:`datalad.api.install` a second time. Without this,
        ``install`` against a pre-existing path can raise and the
        layered backend would silently fall through to HTTPS on every
        re-run.
        """
        marker = tmp_path / ".datalad"
        marker.mkdir()

        DataLadBackend(datalad_url="https://x/repo.git").transfer(
            _files(),
            target_dir=tmp_path,
            options=_options(),
            verify=VerifyPolicy(),
            retry=RetryPolicy.default(),
        )

        assert fake_bundle.api.installed == []  # no re-clone
        assert fake_bundle.open_calls == [str(tmp_path)]
        assert fake_bundle.dataset.got != []  # but get still happened


class TestDataLadBackendFailureModes:
    def test_install_failure_raises_datalad_error(
        self, fake_bundle: _FakeBundle, tmp_path: Path
    ) -> None:
        fake_bundle.api.install_raises = _FakeCommandError("clone refused: 404")
        with pytest.raises(DataLadError, match="install failed"):
            DataLadBackend(datalad_url="https://x/repo.git").transfer(
                _files(),
                target_dir=tmp_path,
                options=_options(),
                verify=VerifyPolicy(),
                retry=RetryPolicy.default(),
            )

    def test_open_existing_clone_failure_raises_datalad_error(
        self, fake_bundle: _FakeBundle, tmp_path: Path
    ) -> None:
        """A broken pre-existing clone surfaces as DataLadError, not a crash."""
        (tmp_path / ".datalad").mkdir()
        fake_bundle.open_raises = _FakeCommandError(
            "not a valid DataLad dataset"
        )
        with pytest.raises(DataLadError, match="failed to open existing"):
            DataLadBackend(datalad_url="https://x/repo.git").transfer(
                _files(),
                target_dir=tmp_path,
                options=_options(),
                verify=VerifyPolicy(),
                retry=RetryPolicy.default(),
            )

    def test_checkout_failure_raises_datalad_error(
        self, fake_bundle: _FakeBundle, tmp_path: Path
    ) -> None:
        fake_bundle.dataset.repo.checkout_raises = _FakeCommandError(
            "unknown ref"
        )
        with pytest.raises(DataLadError, match="check out revision 'v9.9.9'"):
            DataLadBackend(
                datalad_url="https://x/repo.git", revision="v9.9.9"
            ).transfer(
                _files(),
                target_dir=tmp_path,
                options=_options(),
                verify=VerifyPolicy(),
                retry=RetryPolicy.default(),
            )

    def test_get_failure_raises_datalad_error(
        self, fake_bundle: _FakeBundle, tmp_path: Path
    ) -> None:
        fake_bundle.dataset.get_raises = _FakeIncompleteResultsError(
            "annex out of disk"
        )
        with pytest.raises(DataLadError, match="get failed for 2 file"):
            DataLadBackend(datalad_url="https://x/repo.git").transfer(
                _files(),
                target_dir=tmp_path,
                options=_options(),
                verify=VerifyPolicy(),
                retry=RetryPolicy.default(),
            )

    def test_programming_bug_propagates_unchanged(
        self, fake_bundle: _FakeBundle, tmp_path: Path
    ) -> None:
        """``TypeError`` / ``AttributeError`` are NOT swallowed.

        Pinning this protects against a regression where a broad
        ``except Exception`` masks a real defect (wrong kwarg, missing
        attribute) as a DataLad failure that routes to the fallback —
        the bug would never surface during testing.
        """
        fake_bundle.api.install_raises = TypeError(
            "install() got unexpected keyword argument 'whatever'"
        )
        with pytest.raises(TypeError, match="install"):
            DataLadBackend(datalad_url="https://x/repo.git").transfer(
                _files(),
                target_dir=tmp_path,
                options=_options(),
                verify=VerifyPolicy(),
                retry=RetryPolicy.default(),
            )

    def test_datalad_error_is_a_transfer_error(self) -> None:
        """LayeredBackend catches a narrow subclass; library callers can catch
        either ``DataLadError`` or the parent ``TransferError`` and still see
        the DataLad failure path. Pinned here so an accidental refactor that
        drops the inheritance breaks a unit test rather than the fallback.
        """
        assert issubclass(DataLadError, TransferError)


# ---------------------------------------------------------------------------
# LayeredBackend
# ---------------------------------------------------------------------------


class TestLayeredBackend:
    def test_primary_success_skips_fallback(self, tmp_path: Path) -> None:
        primary = _RecordingBackend()
        fallback = _RecordingBackend()
        files = _files()

        LayeredBackend(primary, fallback).transfer(
            files,
            target_dir=tmp_path,
            options=_options(),
            verify=VerifyPolicy(),
            retry=RetryPolicy.default(),
        )

        assert len(primary.calls) == 1
        assert fallback.calls == []

    def test_datalad_error_triggers_fallback(self, tmp_path: Path) -> None:
        primary = _RecordingBackend(raises=DataLadError("clone refused"))
        fallback = _RecordingBackend()
        files = _files()

        LayeredBackend(primary, fallback).transfer(
            files,
            target_dir=tmp_path,
            options=_options(),
            verify=VerifyPolicy(),
            retry=RetryPolicy.default(),
        )

        assert len(fallback.calls) == 1
        # Fallback received the exact same file set.
        recorded_paths, recorded_target = fallback.calls[0]
        assert recorded_paths == tuple(f.path for f in files)
        assert recorded_target == tmp_path

    def test_non_datalad_error_propagates(self, tmp_path: Path) -> None:
        """A plain TransferError must not silently retry on the fallback
        — only the DataLad-specific subclass triggers fallback. Otherwise
        a real HTTPS failure on the *primary* (when the primary itself
        is an HTTPS backend) would silently retry on a second HTTPS
        backend, defeating the point of an explicit choice.
        """
        primary = _RecordingBackend(raises=TransferError("upstream timed out"))
        fallback = _RecordingBackend()

        with pytest.raises(TransferError, match="timed out"):
            LayeredBackend(primary, fallback).transfer(
                _files(),
                target_dir=tmp_path,
                options=_options(),
                verify=VerifyPolicy(),
                retry=RetryPolicy.default(),
            )

        assert fallback.calls == []


# ---------------------------------------------------------------------------
# Optional-dependency import seam
# ---------------------------------------------------------------------------


class TestImportSeam:
    def test_real_import_returns_bundle_when_datalad_is_available(self) -> None:
        """``_import_datalad`` returns a populated bundle from the real package.

        ``datalad`` is a hard dependency, so it is always installed in the
        dev/test environment. Calling the seam directly pins the wiring:
        api + Dataset + a 2-tuple of exception classes all land on the
        bundle in the expected slots.
        """
        bundle = _datalad._import_datalad()
        assert isinstance(bundle, _DataLadModules)
        assert bundle.api is not None
        assert bundle.Dataset is not None
        assert isinstance(bundle.exceptions, tuple)
        assert len(bundle.exceptions) == 2
        assert all(
            isinstance(exc, type) and issubclass(exc, BaseException)
            for exc in bundle.exceptions
        )
