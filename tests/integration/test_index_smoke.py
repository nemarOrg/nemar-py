"""Smoke integration test: real httpx -> local HTTPS fixture server."""

from __future__ import annotations

import pytest

import nemar
from tests.fixtures.factories import make_index, make_manifest_entry, make_manifest_list


@pytest.mark.integration
def test_real_httpx_against_local_https_endpoint(nemar_endpoint) -> None:
    index = make_index(dataset="nm000132")
    manifest = make_manifest_list(
        [
            make_manifest_entry(
                path="dataset_description.json",
                content=b'{"Name": "nm000132"}',
                base_url=nemar_endpoint.base_url + "nm000132/v1.0.0",
            ),
        ]
    )
    nemar_endpoint.publish(
        "nm000132",
        index=index,
        manifest=manifest,
        files={"v1.0.0/dataset_description.json": b'{"Name": "nm000132"}'},
    )

    resolved = nemar.fetch_dataset_index(
        dataset="nm000132",
        data_url=nemar_endpoint.base_url,
    )
    assert resolved.dataset_id == "nm000132"
    assert resolved.latest == "v1.0.0"
    assert [v.version for v in resolved.versions] == ["v1.0.0"]
