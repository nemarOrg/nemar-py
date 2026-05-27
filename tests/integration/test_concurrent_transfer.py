"""Integration: concurrent transfer error consolidation."""

from __future__ import annotations

import pytest
from pytest_httpserver.httpserver import HandlerType
from werkzeug.wrappers import Request, Response

import nemar
from tests.fixtures.factories import (
    make_blob,
    make_index,
    make_manifest_entry,
    make_manifest_list,
)

pytestmark = pytest.mark.integration


def test_all_failures_are_reported(nemar_endpoint, target_dir):
    blob = make_blob(seed=1, size_bytes=128)
    index = make_index(dataset="nm000132")
    manifest = make_manifest_list(
        [
            make_manifest_entry(path=f"data/f{i}.bin", content=blob.content)
            for i in range(3)
        ]
    )
    nemar_endpoint.publish(
        "nm000132",
        index=index,
        manifest=manifest,
        files={f"v1.0.0/data/f{i}.bin": blob.content for i in range(3)},
        metadata={"Name": "nm000132"},
    )

    # All three respond 500 every time.
    for i in range(3):

        def handler(request: Request) -> Response:
            return Response(b"boom", status=500)

        # ONESHOT to override the permanent file handler installed by publish().
        nemar_endpoint.server.expect_request(
            f"/nm000132/v1.0.0/data/f{i}.bin",
            handler_type=HandlerType.ONESHOT,
        ).respond_with_handler(handler)

    with pytest.raises(RuntimeError) as excinfo:
        nemar.download(
            dataset="nm000132",
            target_dir=target_dir,
            data_url=nemar_endpoint.base_url,
            downloader="python",
            max_concurrent_downloads=3,
            max_retries=0,
        )

    message = str(excinfo.value)
    assert "3 file(s) failed" in message  # consolidated count
