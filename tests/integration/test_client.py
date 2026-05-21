"""Integration: ``NEMARClient`` against the local HTTPS fixture server."""

from __future__ import annotations

import pytest

from nemar import NEMARClient
from tests.fixtures.factories import (
    make_index,
    make_manifest_entry,
    make_manifest_list,
)

pytestmark = pytest.mark.integration


def _publish_minimal(nemar_endpoint, dataset: str = "nm000132") -> None:
    """Publish a single-file dataset matching the standard test layout."""
    base_url = nemar_endpoint.base_url + f"{dataset}/v1.0.0"
    index = make_index(dataset=dataset)
    manifest = make_manifest_list(
        [
            make_manifest_entry(
                path="dataset_description.json",
                content=b'{"Name": "x"}',
                base_url=base_url,
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


def test_client_walks_index_metadata_and_manifest(nemar_endpoint) -> None:
    """A single ``NEMARClient`` resolves all three metadata fetches end-to-end."""
    _publish_minimal(nemar_endpoint)

    with NEMARClient(data_url=nemar_endpoint.base_url) as client:
        index = client.fetch_index("nm000132")
        metadata = client.fetch_metadata(index)
        version = index.resolve_version("latest")
        manifest = client.fetch_manifest(index, version)

    assert index.dataset_id == "nm000132"
    assert index.latest == "v1.0.0"
    assert metadata == {"Name": "x"}
    assert len(manifest) == 1
    assert next(iter(manifest)).path == "dataset_description.json"


def test_client_endpoint_property_round_trips(nemar_endpoint) -> None:
    """The endpoint is the configured base URL, normalized with one slash."""
    with NEMARClient(data_url=nemar_endpoint.base_url) as client:
        assert client.endpoint.url == nemar_endpoint.base_url
