"""Integration: real HTTPS for dataset index and manifest fetches."""

from __future__ import annotations

import pytest
from werkzeug.wrappers import Request, Response

import nemar
from tests.fixtures.factories import (
    make_index,
    make_manifest_entry,
    make_manifest_list,
)

pytestmark = pytest.mark.integration


def _publish_minimal(nemar_endpoint, dataset: str = "nm000132") -> None:
    index = make_index(dataset=dataset)
    manifest = make_manifest_list([
        make_manifest_entry(
            path="dataset_description.json",
            content=b'{"Name": "x"}',
        )
    ])
    nemar_endpoint.publish(
        dataset,
        index=index,
        manifest=manifest,
        files={"v1.0.0/dataset_description.json": b'{"Name": "x"}'},
        metadata={"Name": "x"},
    )


def test_fetch_dataset_index_against_real_server(nemar_endpoint) -> None:
    _publish_minimal(nemar_endpoint)
    idx = nemar.fetch_dataset_index(
        dataset="nm000132", data_url=nemar_endpoint.base_url
    )
    assert idx.latest == "v1.0.0"
    assert idx.versions[0].manifest_url == "v1.0.0/manifest.json"


def test_fetch_dataset_index_reports_http_404(nemar_endpoint) -> None:
    nemar_endpoint.fail_index_with("nm999999", status=404, body="not found")
    with pytest.raises(RuntimeError, match="HTTP 404"):
        nemar.fetch_dataset_index(
            dataset="nm999999", data_url=nemar_endpoint.base_url
        )


def test_fetch_dataset_index_retries_503(nemar_endpoint) -> None:
    state = {"calls": 0}

    def handler(request: Request) -> Response:
        state["calls"] += 1
        if state["calls"] < 2:
            return Response(b"", status=503)
        return Response(
            __import__("json").dumps(make_index(dataset="nm000132")),
            status=200,
            content_type="application/json",
        )

    nemar_endpoint.server.expect_request("/nm000132/").respond_with_handler(handler)

    idx = nemar.fetch_dataset_index(
        dataset="nm000132", data_url=nemar_endpoint.base_url, max_retries=2
    )
    assert idx.latest == "v1.0.0"
    assert state["calls"] == 2


def test_list_dataset_versions_returns_advertised_versions(nemar_endpoint) -> None:
    _publish_minimal(nemar_endpoint)
    versions = nemar.list_dataset_versions(
        dataset="nm000132", data_url=nemar_endpoint.base_url
    )
    assert [v.version for v in versions] == ["v1.0.0"]
