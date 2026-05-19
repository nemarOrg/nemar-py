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
    manifest = make_manifest_list(
        [
            make_manifest_entry(
                path="dataset_description.json",
                content=b'{"Name": "x"}',
            )
        ]
    )
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
    assert idx.versions[0].manifest_url == "nm000132/v1.0.0/manifest.json"


def test_fetch_dataset_index_reports_http_404(nemar_endpoint) -> None:
    nemar_endpoint.fail_index_with("nm999999", status=404, body="not found")
    with pytest.raises(RuntimeError, match="HTTP 404"):
        nemar.fetch_dataset_index(dataset="nm999999", data_url=nemar_endpoint.base_url)


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


def test_fetch_dataset_index_rejects_off_origin_redirect(nemar_endpoint) -> None:
    """A 302 to another netloc must not silently bypass the origin scope.

    The configured endpoint is ``https://localhost:<port>/``. The fixture
    server redirects the first ``/nm000132/`` request to the same server
    bound by IP literal (``https://127.0.0.1:<port>/...``). TLS validates
    against both names (see ``generate_local_ca``), so httpx will follow
    the redirect and a naive client would return the (off-origin) payload.
    The downloader's origin check must catch the netloc divergence and
    raise instead.
    """
    import json as _json

    parsed_base = nemar_endpoint.base_url.rstrip("/")
    # Re-target the same fixture server through its IP literal so TLS
    # handshakes succeed but the final URL's netloc differs from the
    # configured endpoint's netloc.
    redirect_target = parsed_base.replace("//localhost:", "//127.0.0.1:")
    assert redirect_target != parsed_base, (
        "Fixture base_url is expected to use localhost"
    )

    def handler(request: Request) -> Response:
        # Respond once with a redirect, then with the index JSON. The
        # second response stands in for the off-origin server returning
        # a valid-looking payload.
        if request.host.startswith("localhost"):
            return Response(
                b"",
                status=302,
                headers={"Location": f"{redirect_target}/nm000132/"},
            )
        return Response(
            _json.dumps(make_index(dataset="nm000132")),
            status=200,
            content_type="application/json",
        )

    nemar_endpoint.server.expect_request("/nm000132/").respond_with_handler(handler)

    with pytest.raises(RuntimeError, match="outside the configured NEMAR"):
        nemar.fetch_dataset_index(
            dataset="nm000132", data_url=nemar_endpoint.base_url
        )
