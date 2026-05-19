"""Tests for NEMAR download orchestration."""

import hashlib
import json
import os
import subprocess
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import httpx
import pytest

from nemar import _bids, _download
from nemar._download import download
from nemar._models import DatasetFile


def dataset_files(paths: list[str]) -> list[DatasetFile]:
    """Create manifest files with stable data.nemar.org URLs."""
    return [
        DatasetFile(path=path, url=f"https://data.nemar.org/nm000132/v1.0.0/{path}")
        for path in paths
    ]


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
    selected = _download._select_bids_files(
        dataset_files(paths),
        query=query,
        include=include,
        exclude=exclude,
    )

    assert [file.path for file in selected] == expected


def test_select_bids_files_errors_when_bids_query_has_no_matches() -> None:
    """Empty semantic selections fail loudly."""
    files = [
        DatasetFile(
            path="sub-001/eeg/sub-001_task-MMN_eeg.set",
            url="https://data.nemar.org/a",
        )
    ]

    with pytest.raises(RuntimeError, match="No files matched the BIDS query"):
        _download._select_bids_files(
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

    with pytest.raises(RuntimeError, match="Perhaps you mean"):
        _download._select_bids_files(
            files,
            query=_bids.BidsQuery(),
            include=["participant.tsv"],
            exclude=[],
        )


def test_fetch_version_manifest_reports_unpublished_data_endpoint() -> None:
    """Missing public manifests are not replaced by a hidden storage fallback."""
    response = httpx.Response(
        404,
        json={"error": "Version not published"},
        request=httpx.Request(
            "GET", "https://data.nemar.org/nm000132/v1/manifest.json"
        ),
    )
    client = MagicMock()
    client.get.return_value = response
    version = MagicMock(version="v1.1.1")

    with pytest.raises(RuntimeError, match="will not fall back to S3"):
        _download._fetch_version_manifest(
            client,
            dataset="nm000132",
            version=version,
            manifest_url="https://data.nemar.org/nm000132/v1.1.1/manifest.json",
            max_retries=0,
        )


def test_python_downloader_fetches_from_data_endpoint(
    monkeypatch, tmp_path: Path
) -> None:
    """The Python adapter downloads, verifies size, and verifies checksum.

    Patches the ``httpx.Client`` constructor used by ``_transfer_with_python``
    so the shared-client (G) path goes through a ``MockTransport`` instead
    of a real network. Existing behaviour: one file in, one file on disk.
    """
    data = b"hello nemar"
    file = DatasetFile(
        path="dataset_description.json",
        url="https://data.nemar.org/nm000132/v1.0.0/dataset_description.json",
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "data.nemar.org"
        return httpx.Response(200, content=data, request=request)

    transport = httpx.MockTransport(handler)
    original_client = httpx.Client

    class PatchedClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", PatchedClient)
    try:
        _download._transfer_with_python(
            [file],
            target_dir=tmp_path,
            verify_hash=True,
            verify_size=True,
            max_retries=0,
            max_concurrent_downloads=1,
            stream_timeout=60.0,
        )
    finally:
        monkeypatch.setattr(httpx, "Client", original_client)

    assert (tmp_path / "dataset_description.json").read_bytes() == data


def test_fetch_dataset_index_uses_advertised_versions(monkeypatch) -> None:
    """Dataset version control comes from the NEMAR data index."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://data.nemar.org/nm000132/"
        return httpx.Response(
            200,
            json={
                "dataset_id": "nm000132",
                "latest": "v1.1.1",
                "metadata_url": "/nm000132/metadata.json",
                "versions": [
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
            },
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
            {
                "dataset_id": "nm000132",
                "latest": "v1.0.0",
                "versions": [
                    {"version": "v1.0.0", "manifest_url": "/manifest.json"},
                ],
            }
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
                json={
                    "dataset_id": "nm000132",
                    "latest": "v1.0.0",
                    "metadata_url": "/nm000132/metadata.json",
                    "versions": [
                        {
                            "version": "v1.0.0",
                            "manifest_url": "/nm000132/v1.0.0/manifest.json",
                        }
                    ],
                },
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
                json={
                    "dataset_id": "nm000132",
                    "latest": "v1.0.0",
                    "metadata_url": "/nm000132/metadata.json",
                    "versions": [
                        {
                            "version": "v1.0.0",
                            "manifest_url": "/nm000132/v1.0.0/manifest.json",
                        }
                    ],
                },
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

    def transfer_files(files, **kwargs):
        transferred["paths"] = [file.path for file in files]
        transferred["target_dir"] = kwargs["target_dir"]

    monkeypatch.setattr(httpx, "Client", Client)
    monkeypatch.setattr(_download, "_transfer_files", transfer_files)

    download(
        dataset="nm000132",
        target_dir=tmp_path,
        scope="stimuli",
        task="MMN",
        downloader="python",
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
    from nemar._request import DownloadRequest

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
    from nemar import fetch_dataset_index

    valid = {
        "dataset": "nm000132",
        "max_retries": 1,
        "data_url": "https://data.nemar.org/",
    }
    valid.update(kwargs)

    with pytest.raises(ValueError, match=message):
        fetch_dataset_index(**valid)


@pytest.mark.parametrize(
    ("data_url", "expected"),
    [
        pytest.param("https://data.nemar.org", "https://data.nemar.org/", id="bare"),
        pytest.param(
            "https://data.nemar.org/",
            "https://data.nemar.org/",
            id="already-normalized",
        ),
        pytest.param(
            "https://data.nemar.org///",
            "https://data.nemar.org/",
            id="extra-slashes",
        ),
    ],
)
def test_normalize_data_url(data_url, expected) -> None:
    """Configured NEMAR origins are normalized to one trailing slash."""
    assert _download._normalize_data_url(data_url) == expected


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        pytest.param("1.0.0", "v1.0.0", id="numeric-version"),
        pytest.param("v1.0.0", "v1.0.0", id="prefixed-version"),
        pytest.param(" latest ", "latest", id="trimmed-alias"),
    ],
)
def test_normalize_version_tag(tag, expected) -> None:
    """Version tags accept NEMAR endpoint labels and numeric aliases."""
    assert _download._normalize_version_tag(tag) == expected


@pytest.mark.parametrize(
    "tag",
    [
        pytest.param("", id="empty"),
        pytest.param("   ", id="blank"),
    ],
)
def test_normalize_version_tag_rejects_blank(tag) -> None:
    """Blank version tags fail before endpoint lookup."""
    with pytest.raises(ValueError, match="must not be empty"):
        _download._normalize_version_tag(tag)


@pytest.mark.parametrize(
    ("patterns", "expected"),
    [
        pytest.param(None, [], id="none"),
        pytest.param("sub-001", ["sub-001"], id="single-string"),
        pytest.param(["sub-001", "*.set"], ["sub-001", "*.set"], id="list"),
    ],
)
def test_normalize_bids_patterns(patterns, expected) -> None:
    """Include and exclude patterns normalize without splitting strings."""
    assert _download._normalize_bids_patterns(patterns) == expected


def test_fetch_dataset_index_rejects_mismatched_payload() -> None:
    """Dataset index payloads must describe the requested dataset."""
    client = MagicMock()
    client.get.return_value = httpx.Response(
        200,
        json={
            "dataset_id": "nm000999",
            "latest": "v1.0.0",
            "versions": [{"version": "v1.0.0", "manifest_url": "/manifest.json"}],
        },
        request=httpx.Request("GET", "https://data.nemar.org/nm000132/"),
    )

    with pytest.raises(RuntimeError, match="described nm000999"):
        _download._fetch_dataset_index(
            client,
            dataset="nm000132",
            data_url="https://data.nemar.org/",
            max_retries=0,
        )


def test_fetch_dataset_metadata_accepts_missing_url() -> None:
    """Dataset metadata is optional in the endpoint index."""
    index = _download.DatasetIndex.model_validate(
        {
            "dataset_id": "nm000132",
            "latest": "v1.0.0",
            "versions": [{"version": "v1.0.0", "manifest_url": "/manifest.json"}],
        }
    )

    assert (
        _download._fetch_dataset_metadata(
            MagicMock(),
            index=index,
            data_url="https://data.nemar.org/",
            max_retries=0,
        )
        is None
    )


def test_fetch_dataset_metadata_rejects_non_object() -> None:
    """Dataset metadata payloads must be JSON objects."""
    client = MagicMock()
    client.get.return_value = httpx.Response(
        200,
        json=[],
        request=httpx.Request("GET", "https://data.nemar.org/nm000132/metadata.json"),
    )
    index = _download.DatasetIndex.model_validate(
        {
            "dataset_id": "nm000132",
            "latest": "v1.0.0",
            "metadata_url": "/nm000132/metadata.json",
            "versions": [{"version": "v1.0.0", "manifest_url": "/manifest.json"}],
        }
    )

    with pytest.raises(RuntimeError, match="metadata payload"):
        _download._fetch_dataset_metadata(
            client,
            index=index,
            data_url="https://data.nemar.org/",
            max_retries=0,
        )


def test_fetch_json_with_retries_recovers_after_retryable_status() -> None:
    """Retryable HTTP statuses are retried before failing."""
    client = MagicMock()
    client.get.side_effect = [
        httpx.Response(503, request=httpx.Request("GET", "https://example.org")),
        httpx.Response(
            200,
            json={"ok": True},
            request=httpx.Request("GET", "https://example.org"),
        ),
    ]

    with mock.patch("nemar._download.time.sleep"):
        result = _download._fetch_json_with_retries(
            client,
            url="https://example.org",
            what="testing",
            max_retries=1,
        )

    assert result == {"ok": True}
    assert client.get.call_count == 2


@pytest.mark.parametrize(
    ("response_or_error", "message"),
    [
        pytest.param(
            httpx.ConnectError("connection failed"),
            "Network error",
            id="network-exception",
        ),
        pytest.param(
            httpx.Response(
                503,
                request=httpx.Request("GET", "https://example.org"),
            ),
            "Retryable error",
            id="retryable-status-exhausted",
        ),
        pytest.param(
            httpx.Response(
                404,
                json={"message": "gone"},
                request=httpx.Request("GET", "https://example.org"),
            ),
            "HTTP 404 gone",
            id="non-retryable-status",
        ),
    ],
)
def test_fetch_json_with_retries_reports_failures(response_or_error, message) -> None:
    """Fetch failures are normalized by retry class."""
    client = MagicMock()
    if isinstance(response_or_error, BaseException):
        client.get.side_effect = response_or_error
    else:
        client.get.return_value = response_or_error

    with pytest.raises(RuntimeError, match=message):
        _download._fetch_json_with_retries(
            client,
            url="https://example.org",
            what="testing",
            max_retries=0,
        )


def test_fetch_json_with_retries_reports_invalid_json() -> None:
    """Invalid JSON responses include the requested URL."""
    client = MagicMock()
    client.get.return_value = httpx.Response(
        200,
        text="not-json",
        request=httpx.Request("GET", "https://example.org"),
    )

    with pytest.raises(RuntimeError, match="Invalid JSON"):
        _download._fetch_json_with_retries(
            client,
            url="https://example.org",
            what="testing",
            max_retries=0,
        )


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        pytest.param(
            httpx.Response(
                500,
                json={"message": "server exploded"},
                request=httpx.Request("GET", "https://example.org"),
            ),
            "server exploded",
            id="json-message",
        ),
        pytest.param(
            httpx.Response(
                500,
                json={"error": "version missing"},
                request=httpx.Request("GET", "https://example.org"),
            ),
            "version missing",
            id="json-error",
        ),
        pytest.param(
            httpx.Response(
                500,
                json=["bad"],
                request=httpx.Request("GET", "https://example.org"),
            ),
            '["bad"]',
            id="json-non-object",
        ),
        pytest.param(
            httpx.Response(
                500,
                text="server exploded",
                request=httpx.Request("GET", "https://example.org"),
            ),
            "server exploded",
            id="text",
        ),
        pytest.param(
            httpx.Response(
                500,
                text="",
                request=httpx.Request("GET", "https://example.org"),
            ),
            "",
            id="empty-text",
        ),
    ],
)
def test_response_detail_formats_endpoint_errors(response, expected) -> None:
    """Endpoint error details stay concise across JSON and text payloads."""
    assert _download._response_detail(response) == expected


@pytest.mark.parametrize(
    "description",
    [
        pytest.param(None, id="non-empty-without-description"),
        pytest.param(
            {"DatasetDOI": "doi:10.82901/nemar.nm000132", "Version": "1.0.0"},
            id="matching-dataset-and-version",
        ),
    ],
)
def test_assert_target_matches_dataset_version_allows_resume_states(
    tmp_path: Path, description
) -> None:
    """Compatible local targets can be reused or resumed."""
    if description is None:
        (tmp_path / "partial.tmp").write_text("x", encoding="utf-8")
    else:
        (tmp_path / "dataset_description.json").write_text(
            json.dumps(description),
            encoding="utf-8",
        )

    _download._assert_target_matches_dataset_version(
        dataset="nm000132",
        tag="v1.0.0",
        target_dir=tmp_path,
    )


@pytest.mark.parametrize(
    ("content", "error_type", "message"),
    [
        pytest.param("{bad", RuntimeError, "Could not parse local", id="bad-json"),
        pytest.param(
            json.dumps({}),
            RuntimeError,
            "does not contain",
            id="missing-dataset-doi",
        ),
        pytest.param(
            json.dumps({"DatasetDOI": 123}),
            RuntimeError,
            "DatasetDOI.*string",
            id="non-string-doi",
        ),
        pytest.param(
            json.dumps({"DatasetDOI": "10.82901/nemar.nm000999"}),
            RuntimeError,
            "different NEMAR dataset",
            id="other-dataset",
        ),
        pytest.param(
            json.dumps({"DatasetDOI": "10.82901/nemar.nm000132", "Version": 100}),
            RuntimeError,
            "Version.*string",
            id="non-string-version",
        ),
        pytest.param(
            json.dumps({"DatasetDOI": "10.82901/nemar.nm000132"}),
            RuntimeError,
            "does not contain a.*Version.*field",
            id="missing-version-with-files",
        ),
        pytest.param(
            json.dumps(
                {
                    "DatasetDOI": "10.82901/nemar.nm000132",
                    "Version": "1.0.0",
                }
            ),
            FileExistsError,
            "v1.0.0 exists locally",
            id="version-mismatch",
        ),
    ],
)
def test_assert_target_matches_dataset_version_rejects_invalid_local_description(
    tmp_path: Path, content, error_type, message
) -> None:
    """Existing NEMAR datasets cannot be silently mixed across versions."""
    (tmp_path / "dataset_description.json").write_text(content, encoding="utf-8")

    with pytest.raises(error_type, match=message):
        _download._assert_target_matches_dataset_version(
            dataset="nm000132",
            tag="v1.1.1",
            target_dir=tmp_path,
        )


def test_transfer_files_skips_complete_files(tmp_path: Path) -> None:
    """Complete local files avoid invoking a transfer backend."""
    outfile = tmp_path / "participants.tsv"
    outfile.write_text("ok", encoding="utf-8")
    file = DatasetFile(
        path="participants.tsv",
        url="https://data.nemar.org/participants.tsv",
        size=2,
    )

    _download._transfer_files(
        [file],
        target_dir=tmp_path,
        downloader="python",
        verify_hash=False,
        verify_size=True,
        max_retries=0,
        max_concurrent_downloads=1,
        stream_timeout=60.0,
    )


@pytest.mark.parametrize(
    ("backend", "transfer_name"),
    [
        pytest.param("aria2", "_transfer_with_aria2", id="aria2"),
        pytest.param("python", "_transfer_with_python", id="python"),
    ],
)
def test_transfer_files_dispatches_to_selected_backend(
    monkeypatch, tmp_path: Path, backend, transfer_name
) -> None:
    """Pending files are sent to the selected transfer backend."""
    called = {}

    def transfer(files, **kwargs):
        called["paths"] = [file.path for file in files]
        called["target_dir"] = kwargs["target_dir"]

    monkeypatch.setattr(
        _download, "_select_transfer_backend", lambda requested: backend
    )
    monkeypatch.setattr(_download, transfer_name, transfer)
    monkeypatch.setattr(
        _download, "_verify_manifest_files", lambda *args, **kwargs: None
    )

    _download._transfer_files(
        [DatasetFile(path="participants.tsv", url="https://data.nemar.org/p.tsv")],
        target_dir=tmp_path,
        downloader="auto",
        verify_hash=False,
        verify_size=True,
        max_retries=0,
        max_concurrent_downloads=1,
        stream_timeout=60.0,
    )

    assert called == {"paths": ["participants.tsv"], "target_dir": tmp_path}


@pytest.mark.parametrize(
    ("which_result", "requested", "expected", "error"),
    [
        pytest.param(None, "auto", "python", None, id="auto-without-aria2"),
        pytest.param(
            None,
            "aria2",
            None,
            "requires aria2c",
            id="aria2-requested-but-missing",
        ),
        pytest.param("/usr/bin/aria2c", "auto", "aria2", None, id="auto-with-aria2"),
        pytest.param(
            "/usr/bin/aria2c",
            "aria2",
            "aria2",
            None,
            id="explicit-aria2-with-aria2",
        ),
        pytest.param(
            "/usr/bin/aria2c",
            "python",
            "python",
            None,
            id="explicit-python-with-aria2",
        ),
    ],
)
def test_select_transfer_backend(
    monkeypatch, which_result, requested, expected, error
) -> None:
    """Transfer backend selection depends on request and aria2 availability."""
    monkeypatch.setattr(_download.shutil, "which", lambda name: which_result)

    if error is not None:
        with pytest.raises(RuntimeError, match=error):
            _download._select_transfer_backend(requested)
    else:
        assert _download._select_transfer_backend(requested) == expected


@pytest.mark.parametrize(
    ("file", "expected"),
    [
        pytest.param(
            DatasetFile(
                path="x",
                url="https://data.nemar.org/x",
                sha256="abc",
                md5="def",
            ),
            "sha-256=abc",
            id="sha256-preferred",
        ),
        pytest.param(
            DatasetFile(path="x", url="https://data.nemar.org/x", md5="def"),
            "md5=def",
            id="md5",
        ),
        pytest.param(
            DatasetFile(path="x", url="https://data.nemar.org/x"),
            None,
            id="no-checksum",
        ),
    ],
)
def test_aria2_checksum(file, expected) -> None:
    """aria2 checksum syntax follows the strongest manifest hash."""
    assert _download._aria2_checksum(file) == expected


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

    assert (
        _download._local_file_satisfies_manifest(
            file,
            target_dir=tmp_path,
            verify_hash=verify_hash,
            verify_size=verify_size,
        )
        is expected
    )


def test_transfer_with_aria2_builds_input_file(monkeypatch, tmp_path: Path) -> None:
    """The aria2 adapter writes relative outputs and checksum directives."""
    seen = {}

    def run(cmd, check, **kwargs):
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs
        input_arg = next(arg for arg in cmd if arg.startswith("--input-file="))
        input_path = Path(input_arg.split("=", 1)[1])
        seen["input"] = input_path.read_text(encoding="utf-8")
        for line in seen["input"].splitlines():
            if line.startswith("  out="):
                (tmp_path / line.split("=", 1)[1]).write_text("ok", encoding="utf-8")

    monkeypatch.setattr(_download.subprocess, "run", run)

    _download._transfer_with_aria2(
        [
            DatasetFile(
                path="participants.tsv",
                url="https://data.nemar.org/participants.tsv",
                size=2,
                sha256=hashlib.sha256(b"ok").hexdigest(),
            )
        ],
        target_dir=tmp_path,
        verify_hash=True,
        verify_size=False,
        max_retries=2,
        max_concurrent_downloads=4,
    )

    assert "https://data.nemar.org/participants.tsv" in seen["input"]
    assert "  out=participants.tsv" in seen["input"]
    assert "  checksum=sha-256=" in seen["input"]
    assert "--max-tries=3" in seen["cmd"]


@pytest.mark.parametrize(
    ("max_concurrent_downloads", "expected_split"),
    [
        pytest.param(1, 16, id="1-download-max-split"),
        pytest.param(2, 16, id="2-downloads-split-16"),
        pytest.param(4, 8, id="4-downloads-split-8"),
        pytest.param(8, 4, id="8-downloads-split-4"),
        pytest.param(16, 2, id="16-downloads-split-2"),
        pytest.param(32, 1, id="32-downloads-split-1"),
        pytest.param(64, 1, id="64-downloads-clamped-to-1"),
    ],
)
def test_transfer_with_aria2_split_caps_total_connections(
    monkeypatch, tmp_path: Path, max_concurrent_downloads, expected_split
) -> None:
    """aria2 --split is capped so max_concurrent_downloads * split <= 32."""
    seen: dict[str, object] = {}

    def run(cmd, check, **kwargs):
        seen["cmd"] = cmd
        input_arg = next(arg for arg in cmd if arg.startswith("--input-file="))
        input_path = Path(input_arg.split("=", 1)[1])
        for line in input_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("  out="):
                (tmp_path / line.split("=", 1)[1]).write_text("ok", encoding="utf-8")

    monkeypatch.setattr(_download.subprocess, "run", run)

    _download._transfer_with_aria2(
        [
            DatasetFile(
                path="participants.tsv",
                url="https://data.nemar.org/participants.tsv",
            )
        ],
        target_dir=tmp_path,
        verify_hash=False,
        verify_size=False,
        max_retries=0,
        max_concurrent_downloads=max_concurrent_downloads,
    )

    split_arg = next(
        (arg for arg in seen["cmd"] if arg.startswith("--split=")), None
    )
    assert split_arg == f"--split={expected_split}", (
        f"Expected --split={expected_split}, got {split_arg}"
    )


def test_transfer_with_aria2_reports_failure(monkeypatch, tmp_path: Path) -> None:
    """aria2 failures are normalized to RuntimeError."""

    def run(cmd, check, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(_download.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="aria2c failed"):
        _download._transfer_with_aria2(
            [DatasetFile(path="x", url="https://data.nemar.org/x")],
            target_dir=tmp_path,
            verify_hash=False,
            verify_size=False,
            max_retries=0,
            max_concurrent_downloads=1,
        )


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
        _download._verify_manifest_file(
            file,
            outfile,
            verify_hash=verify_hash,
            verify_size=verify_size,
        )


def _is_case_insensitive_fs(tmp_path: Path) -> bool:
    probe = tmp_path / "ci_probe"
    probe.write_text("x")
    try:
        upper = tmp_path / "CI_PROBE"
        return upper.exists() and os.path.samefile(probe, upper)
    finally:
        probe.unlink(missing_ok=True)


def test_assert_no_case_collisions_passes_when_paths_unique(tmp_path: Path) -> None:
    """Non-colliding paths are accepted on any filesystem."""
    files = [
        DatasetFile(path="a.bin", url="https://data.nemar.org/a.bin"),
        DatasetFile(path="b.bin", url="https://data.nemar.org/b.bin"),
    ]
    _download._assert_no_case_collisions(files, target_dir=tmp_path)


def test_assert_no_case_collisions_raises_on_case_insensitive_fs(
    tmp_path: Path,
) -> None:
    """Manifest paths that differ only by case fail before transfer (S4)."""
    if not _is_case_insensitive_fs(tmp_path):
        pytest.skip("Filesystem is case-sensitive; collision is impossible here.")

    files = [
        DatasetFile(path="data/Sample.bin", url="https://data.nemar.org/x"),
        DatasetFile(path="data/sample.bin", url="https://data.nemar.org/y"),
    ]
    with pytest.raises(RuntimeError, match="case-insensitive filesystem"):
        _download._assert_no_case_collisions(files, target_dir=tmp_path)


def test_assert_no_case_collisions_is_noop_on_case_sensitive_fs(
    tmp_path: Path,
) -> None:
    """On case-sensitive filesystems, sibling paths are not flagged."""
    if _is_case_insensitive_fs(tmp_path):
        pytest.skip("Filesystem is case-insensitive; collision is the right outcome.")

    files = [
        DatasetFile(path="data/Sample.bin", url="https://data.nemar.org/x"),
        DatasetFile(path="data/sample.bin", url="https://data.nemar.org/y"),
    ]
    # Should not raise.
    _download._assert_no_case_collisions(files, target_dir=tmp_path)
