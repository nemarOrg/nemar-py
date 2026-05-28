"""Tests for NEMAR download orchestration."""

import hashlib
from pathlib import Path

import httpx
import pytest

from nemar import _bids, _download, fetch_dataset_index
from nemar._download import download
from nemar._models import DatasetFile
from nemar._request import DownloadRequest
from nemar._selection import SelectionPlan
from nemar._verification import (
    VerifyPolicy,
    VerifyResult,
    assert_all_present,
    check,
)
from tests.fixtures.factories import make_dataset_file, make_index


def dataset_files(paths: list[str]) -> list[DatasetFile]:
    """Create manifest files with stable data.nemar.org URLs."""
    return [make_dataset_file(path) for path in paths]


@pytest.mark.parametrize(
    ("paths", "query", "include", "exclude", "expected"),
    [
        pytest.param(
            [
                "dataset_description.json",
                "participants.tsv",
                "sub-001/eeg/file.set",
                "sub-001/eeg/file.fdt",
                "sub-002/eeg/file.set",
            ],
            _bids.BidsQuery(),
            ["sub-001/eeg"],
            ["*.fdt"],
            [
                "dataset_description.json",
                "participants.tsv",
                "sub-001/eeg/file.set",
            ],
            id="essentials-survive-glob-filtering",
        ),
        pytest.param(
            [
                "dataset_description.json",
                "sub-001/eeg/sub-001_task-MMN_run-01_eeg.set",
                "sub-001/eeg/sub-001_task-MMN_run-01_eeg.fdt",
                "sub-001/eeg/sub-001_task-P3_run-01_eeg.set",
                "sub-002/eeg/sub-002_task-MMN_run-01_eeg.set",
                "sub-001/beh/sub-001_task-MMN_events.tsv",
            ],
            _bids.BidsQuery.from_filters(
                subject="sub-001",
                task="task-MMN",
                datatype="eeg",
                suffix="eeg",
                extension=".set",
            ),
            [],
            [],
            [
                "dataset_description.json",
                "sub-001/eeg/sub-001_task-MMN_run-01_eeg.set",
            ],
            id="semantic-bids-query",
        ),
        pytest.param(
            [
                "dataset_description.json",
                "sub-001/eeg/sub-001_task-MMN_run-01_eeg.set",
                "sub-001/eeg/sub-001_task-MMN_run-02_eeg.set",
            ],
            _bids.BidsQuery.from_filters(subject="001", task="MMN"),
            ["*run-02*"],
            [],
            [
                "dataset_description.json",
                "sub-001/eeg/sub-001_task-MMN_run-02_eeg.set",
            ],
            id="include-narrows-semantic-query",
        ),
        pytest.param(
            [
                "dataset_description.json",
                "sub-001/eeg/sub-001_task-MMN_eeg.set",
                "derivatives/eeglab/sub-001/eeg/sub-001_task-MMN_desc-clean_eeg.set",
                "derivatives/other/sub-001/eeg/sub-001_task-MMN_desc-clean_eeg.set",
            ],
            _bids.BidsQuery.from_filters(
                scope="derivatives",
                pipeline="eeglab",
                subject="001",
                task="MMN",
            ),
            [],
            [],
            [
                "dataset_description.json",
                "derivatives/eeglab/sub-001/eeg/sub-001_task-MMN_desc-clean_eeg.set",
            ],
            id="derivative-pipeline",
        ),
        pytest.param(
            [
                "dataset_description.json",
                "stimuli/task-MMN/deviant_stereo.wav",
                "stimuli/task-P3/target.bmp",
            ],
            _bids.BidsQuery.from_filters(scope="stimuli", task="MMN"),
            [],
            [],
            [
                "dataset_description.json",
                "stimuli/task-MMN/deviant_stereo.wav",
            ],
            id="stimuli-task",
        ),
    ],
)
def test_select_bids_files_applies_semantic_and_path_filters(
    paths, query, include, exclude, expected
) -> None:
    """BIDS file selection composes semantic filters, globs, and essentials."""
    files = dataset_files(paths)
    plan = SelectionPlan.build(
        files,
        query=query,
        include=include,
        exclude=exclude,
    )
    plan.raise_if_unmatched_includes(filenames=[file.path for file in files])

    assert [file.path for file in plan.final] == expected


def test_select_bids_files_errors_when_bids_query_has_no_matches() -> None:
    """Empty semantic selections fail loudly."""
    files = [
        DatasetFile(
            path="sub-001/eeg/sub-001_task-MMN_eeg.set",
            url="https://data.nemar.org/a",
        )
    ]

    with pytest.raises(RuntimeError, match="No files matched the BIDS query"):
        SelectionPlan.build(
            files,
            query=_bids.BidsQuery.from_filters(subject="002"),
            include=[],
            exclude=[],
        )


def test_select_bids_files_suggests_close_include() -> None:
    """Unmatched path includes report close manifest paths."""
    files = [
        DatasetFile(path="participants.tsv", url="https://data.nemar.org/a"),
        DatasetFile(
            path="sub-001/eeg/sub-001_task-MMN_eeg.set",
            url="https://data.nemar.org/b",
        ),
    ]
    plan = SelectionPlan.build(
        files,
        query=_bids.BidsQuery(),
        include=["participant.tsv"],
        exclude=[],
    )

    with pytest.raises(RuntimeError, match="Perhaps you mean"):
        plan.raise_if_unmatched_includes(filenames=[file.path for file in files])


def test_fetch_dataset_index_uses_advertised_versions(monkeypatch) -> None:
    """Dataset version control comes from the NEMAR data index."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://data.nemar.org/nm000132/"
        return httpx.Response(
            200,
            json=make_index(
                dataset="nm000132",
                latest="v1.1.1",
                versions=[
                    {
                        "version": "v1.1.1",
                        "doi": "10.82901/nemar.nm000132.v1.1.1",
                        "created_at": "2026-04-04 06:05:15",
                        "manifest_url": "/nm000132/v1.1.1/manifest.json",
                    },
                    {
                        "version": "v1.0.0",
                        "doi": "10.82901/nemar.nm000132.v1.0.0",
                        "created_at": "2026-03-14 12:20:43",
                        "manifest_url": "/nm000132/v1.0.0/manifest.json",
                    },
                ],
            ),
            request=request,
        )

    transport = httpx.MockTransport(handler)

    class Client(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", Client)

    index = _download.fetch_dataset_index(dataset="nm000132")

    assert index.latest == "v1.1.1"
    assert [version.version for version in index.versions] == ["v1.1.1", "v1.0.0"]
    assert index.resolve_version("v1.0.0").manifest_url.endswith("manifest.json")
    assert index.resolve_version("latest").version == "v1.1.1"


def test_public_list_dataset_versions_uses_endpoint_index(monkeypatch) -> None:
    """The public version list is derived from the dataset index."""

    def fetch_dataset_index(**kwargs):
        return _download.DatasetIndex.model_validate(
            make_index(
                dataset="nm000132",
                latest="v1.0.0",
                metadata_url=None,
                versions=[{"version": "v1.0.0", "manifest_url": "/manifest.json"}],
            )
        )

    monkeypatch.setattr(_download, "fetch_dataset_index", fetch_dataset_index)

    versions = _download.list_dataset_versions(dataset="nm000132")

    assert [version.version for version in versions] == ["v1.0.0"]


def test_download_uses_data_endpoint_until_manifest(
    monkeypatch, tmp_path: Path
) -> None:
    """The public entry point starts with https://data.nemar.org/{dataset}/."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path == "/nm000132/":
            return httpx.Response(
                200,
                json=make_index(
                    dataset="nm000132",
                    metadata_url="/nm000132/metadata.json",
                    versions=[
                        {
                            "version": "v1.0.0",
                            "manifest_url": "/nm000132/v1.0.0/manifest.json",
                        }
                    ],
                ),
                request=request,
            )
        if request.url.path == "/nm000132/metadata.json":
            return httpx.Response(200, json={"dataset_id": "nm000132"}, request=request)
        return httpx.Response(
            404,
            json={"error": "Version not published"},
            request=request,
        )

    transport = httpx.MockTransport(handler)
    original_client = httpx.Client

    class Client(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", Client)

    with pytest.raises(RuntimeError, match="will not fall back to S3"):
        download(dataset="nm000132", tag="latest", target_dir=tmp_path)

    assert calls[:3] == [
        "https://data.nemar.org/nm000132/",
        "https://data.nemar.org/nm000132/metadata.json",
        "https://data.nemar.org/nm000132/v1.0.0/manifest.json",
    ]
    monkeypatch.setattr(httpx, "Client", original_client)


def test_download_orchestrates_manifest_selection_and_transfer(
    monkeypatch, tmp_path: Path
):
    """The public entry point can complete against mocked data.nemar.org payloads."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path == "/nm000132/":
            return httpx.Response(
                200,
                json=make_index(
                    dataset="nm000132",
                    metadata_url="/nm000132/metadata.json",
                    versions=[
                        {
                            "version": "v1.0.0",
                            "manifest_url": "/nm000132/v1.0.0/manifest.json",
                        }
                    ],
                ),
                request=request,
            )
        if request.url.path == "/nm000132/metadata.json":
            return httpx.Response(200, json={"dataset_id": "nm000132"}, request=request)
        if request.url.path == "/nm000132/v1.0.0/manifest.json":
            return httpx.Response(
                200,
                json={
                    "files": [
                        {"path": "dataset_description.json", "size": 2},
                        {
                            "path": "stimuli/task-MMN/deviant.wav",
                            "url": "stimuli/task-MMN/deviant.wav",
                            "size": 4,
                        },
                        {
                            "path": "stimuli/task-P3/target.wav",
                            "url": "stimuli/task-P3/target.wav",
                            "size": 4,
                        },
                    ]
                },
                request=request,
            )
        raise AssertionError(f"unexpected URL: {request.url}")

    transport = httpx.MockTransport(handler)

    class Client(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    transferred = {}

    class StubBackend:
        def transfer(self, files, **kwargs):
            transferred["paths"] = [file.path for file in files]
            transferred["target_dir"] = kwargs["target_dir"]
            # Materialize the would-be transferred files so the
            # orchestrator's post-transfer ``assert_all_present`` sweep
            # finds them on disk with the right sizes from the manifest.
            for file in files:
                outfile = kwargs["target_dir"] / file.path
                outfile.parent.mkdir(parents=True, exist_ok=True)
                outfile.write_bytes(b"x" * (file.size or 0))

    monkeypatch.setattr(httpx, "Client", Client)
    monkeypatch.setattr(
        _download, "select_backend", lambda options, **_kw: StubBackend()
    )

    download(
        dataset="nm000132",
        target_dir=tmp_path,
        scope="stimuli",
        task="MMN",
        downloader="python",
        verify_hash=False,
    )

    assert calls == [
        "https://data.nemar.org/nm000132/",
        "https://data.nemar.org/nm000132/metadata.json",
        "https://data.nemar.org/nm000132/v1.0.0/manifest.json",
    ]
    assert transferred["paths"] == [
        "dataset_description.json",
        "stimuli/task-MMN/deviant.wav",
    ]
    assert transferred["target_dir"] == tmp_path.resolve()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dataset": "bad"}, "dataset must look"),
        ({"downloader": "wget"}, "downloader must"),
        ({"max_retries": -1}, "max_retries"),
        ({"max_concurrent_downloads": 0}, "max_concurrent"),
        ({"data_url": "http://data.nemar.org/"}, "HTTPS"),
    ],
)
def test_download_request_validates_options(kwargs, message) -> None:
    """``DownloadRequest.from_kwargs`` rejects unsafe or unsupported values."""
    valid = {
        "dataset": "nm000132",
        "downloader": "python",
        "max_retries": 1,
        "max_concurrent_downloads": 1,
        "data_url": "https://data.nemar.org/",
    }
    valid.update(kwargs)

    with pytest.raises(ValueError, match=message):
        DownloadRequest.from_kwargs(**valid)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param({"dataset": "bad"}, "dataset must look", id="dataset-id"),
        pytest.param({"max_retries": -1}, "max_retries", id="negative-retries"),
        pytest.param({"data_url": "http://data.nemar.org/"}, "HTTPS", id="http-url"),
    ],
)
def test_fetch_dataset_index_validates_options(kwargs, message) -> None:
    """``fetch_dataset_index`` rejects unsafe or unsupported values."""
    valid = {
        "dataset": "nm000132",
        "max_retries": 1,
        "data_url": "https://data.nemar.org/",
    }
    valid.update(kwargs)

    with pytest.raises(ValueError, match=message):
        fetch_dataset_index(**valid)


def test_fetch_dataset_index_rejects_mismatched_payload(monkeypatch) -> None:
    """Dataset index payloads must describe the requested dataset."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=make_index(
                dataset="nm000999",
                metadata_url=None,
                versions=[{"version": "v1.0.0", "manifest_url": "/manifest.json"}],
            ),
            request=request,
        )

    transport = httpx.MockTransport(handler)

    class Client(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", Client)

    with pytest.raises(RuntimeError, match="described nm000999"):
        _download.fetch_dataset_index(dataset="nm000132")


def test_download_accepts_index_without_metadata_url(
    monkeypatch, tmp_path: Path
) -> None:
    """When the index advertises no metadata URL, the orchestrator skips that fetch.

    Replaces the legacy ``test_fetch_dataset_metadata_accepts_missing_url`` unit
    test now that the metadata fetch lives inline in ``_run()``.
    """
    calls: list[str] = []

    # The factory always emits a ``metadata_url`` value; clear it so the
    # orchestrator takes the "no metadata advertised" branch and skips that
    # fetch.
    index_payload = make_index(
        dataset="nm000132",
        versions=[
            {
                "version": "v1.0.0",
                "manifest_url": "/nm000132/v1.0.0/manifest.json",
            }
        ],
    )
    index_payload["metadata_url"] = None

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path == "/nm000132/":
            return httpx.Response(200, json=index_payload, request=request)
        if request.url.path == "/nm000132/v1.0.0/manifest.json":
            return httpx.Response(200, json={"files": []}, request=request)
        raise AssertionError(f"unexpected URL: {request.url}")

    transport = httpx.MockTransport(handler)

    class Client(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", Client)

    class _NoopBackend:
        def transfer(self, files, **kwargs) -> None:
            return None

    monkeypatch.setattr(
        _download, "select_backend", lambda options, **_kw: _NoopBackend()
    )

    # An empty manifest is rejected downstream of the metadata branch, which
    # is what we actually want to exercise here: the orchestrator must NOT
    # have reached for metadata.json when the index advertises none.
    with pytest.raises(RuntimeError, match="did not contain any downloadable files"):
        download(
            dataset="nm000132", target_dir=tmp_path, downloader="python"
        )

    # The orchestrator never asked for metadata.json because the index did
    # not advertise one.
    assert "https://data.nemar.org/nm000132/metadata.json" not in calls
    # It did, however, fetch index then manifest in that order.
    assert calls == [
        "https://data.nemar.org/nm000132/",
        "https://data.nemar.org/nm000132/v1.0.0/manifest.json",
    ]


def test_download_rejects_non_object_metadata_payload(
    monkeypatch, tmp_path: Path
) -> None:
    """The metadata payload must be a JSON object.

    Replaces the legacy ``test_fetch_dataset_metadata_rejects_non_object`` unit
    test now that the type check lives inline in ``_run()``.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/nm000132/":
            return httpx.Response(
                200,
                json=make_index(
                    dataset="nm000132",
                    metadata_url="/nm000132/metadata.json",
                    versions=[
                        {
                            "version": "v1.0.0",
                            "manifest_url": "/nm000132/v1.0.0/manifest.json",
                        }
                    ],
                ),
                request=request,
            )
        if request.url.path == "/nm000132/metadata.json":
            # Non-object metadata payload: the orchestrator must reject it
            # before it ever gets to the manifest fetch.
            return httpx.Response(200, json=[], request=request)
        raise AssertionError(f"unexpected URL: {request.url}")

    transport = httpx.MockTransport(handler)

    class Client(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", Client)

    with pytest.raises(RuntimeError, match="metadata payload"):
        download(
            dataset="nm000132", target_dir=tmp_path, downloader="python"
        )


# Target-directory compatibility checks live in the private
# :func:`nemar._download._check_local_compatibility` function. The
# exhaustive behavioural coverage lives in
# ``tests/unit/test_local_dataset.py``; the end-to-end orchestrator
# wiring is exercised by
# ``tests/integration/test_filesystem.py::test_existing_wrong_version_blocks_download``.


@pytest.mark.parametrize(
    ("file", "content", "verify_hash", "verify_size", "expected"),
    [
        pytest.param(
            DatasetFile(path="x", url="https://data.nemar.org/x"),
            None,
            False,
            False,
            False,
            id="missing-file",
        ),
        pytest.param(
            DatasetFile(path="x", url="https://data.nemar.org/x", size=2),
            b"ok",
            False,
            True,
            True,
            id="matching-size",
        ),
        pytest.param(
            DatasetFile(path="x", url="https://data.nemar.org/x", size=2),
            b"wrong",
            False,
            True,
            False,
            id="size-mismatch",
        ),
        pytest.param(
            DatasetFile(
                path="x",
                url="https://data.nemar.org/x",
                md5=hashlib.md5(b"ok").hexdigest(),
            ),
            b"ok",
            True,
            False,
            True,
            id="matching-md5",
        ),
        pytest.param(
            DatasetFile(path="x", url="https://data.nemar.org/x", sha256="bad"),
            b"ok",
            True,
            False,
            False,
            id="hash-mismatch",
        ),
    ],
)
def test_local_file_satisfies_manifest(
    tmp_path: Path, file, content, verify_hash, verify_size, expected
) -> None:
    """Local files are reused only when selected manifest checks pass."""
    if content is not None:
        (tmp_path / file.path).write_bytes(content)

    result = check(
        file,
        tmp_path / file.path,
        VerifyPolicy(verify_size=verify_size, verify_hash=verify_hash),
    )
    assert (result is VerifyResult.OK) is expected


@pytest.mark.parametrize(
    ("file", "content", "verify_hash", "verify_size", "message"),
    [
        pytest.param(
            DatasetFile(path="x", url="https://data.nemar.org/x", size=2),
            None,
            False,
            True,
            "missing",
            id="missing",
        ),
        pytest.param(
            DatasetFile(path="x", url="https://data.nemar.org/x", size=2),
            b"wrong",
            False,
            True,
            "Size mismatch",
            id="size-mismatch",
        ),
        pytest.param(
            DatasetFile(path="x", url="https://data.nemar.org/x", md5="bad"),
            b"wrong",
            True,
            False,
            "Checksum mismatch",
            id="md5-mismatch",
        ),
        pytest.param(
            DatasetFile(path="x", url="https://data.nemar.org/x", sha256="bad"),
            b"wrong",
            True,
            False,
            "Checksum mismatch",
            id="sha256-mismatch",
        ),
    ],
)
def test_verify_manifest_file_errors(
    tmp_path: Path, file, content, verify_hash, verify_size, message
) -> None:
    """Verifier reports missing, size, and checksum mismatches."""
    outfile = tmp_path / file.path
    if content is not None:
        outfile.write_bytes(content)

    with pytest.raises(RuntimeError, match=message):
        assert_all_present(
            [file],
            target_dir=tmp_path,
            policy=VerifyPolicy(verify_size=verify_size, verify_hash=verify_hash),
        )
