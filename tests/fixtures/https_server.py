"""HTTPS fixture server helper that mimics the NEMAR data endpoint."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Request, Response

from tests.fixtures.certs import LocalCa


@dataclass
class PublishedDataset:
    """Routing data for one published dataset on the fake server."""

    dataset: str
    index_payload: dict[str, Any]
    manifest_payload: Any
    files: dict[str, bytes] = field(default_factory=dict)
    metadata_payload: dict[str, Any] | None = None


def start_https_server(local_ca: LocalCa) -> HTTPServer:
    """Start a ``pytest_httpserver`` over HTTPS using ``local_ca``."""
    server = HTTPServer(
        host="localhost",
        port=0,
        ssl_context=local_ca.server_ssl_context(),
    )
    server.start()
    return server


class NemarFakeEndpoint:
    """High-level helper for programming the fake NEMAR endpoint."""

    def __init__(self, server: HTTPServer) -> None:
        self.server = server
        self._lock = threading.Lock()
        self._published: dict[str, PublishedDataset] = {}

    @property
    def base_url(self) -> str:
        return self.server.url_for("/")

    def publish(
        self,
        dataset: str,
        *,
        index: dict[str, Any],
        manifest: Any,
        files: Mapping[str, bytes],
        metadata: Mapping[str, Any] | None = None,
    ) -> PublishedDataset:
        published = PublishedDataset(
            dataset=dataset,
            index_payload=index,
            manifest_payload=manifest,
            files=dict(files),
            metadata_payload=dict(metadata) if metadata is not None else None,
        )
        with self._lock:
            self._published[dataset] = published

        self.server.expect_request(f"/{dataset}/").respond_with_json(index)
        if index.get("metadata_url") and metadata is not None:
            metadata_path = "/" + index["metadata_url"].lstrip("/")
            self.server.expect_request(metadata_path).respond_with_json(dict(metadata))

        for version in index.get("versions", []):
            manifest_path = "/" + version["manifest_url"].lstrip("/")
            self.server.expect_request(manifest_path).respond_with_json(manifest)

        for relpath, content in files.items():
            file_url = "/" + dataset + "/" + relpath.lstrip("/")
            self.server.expect_request(file_url).respond_with_data(content)

        return published

    def fail_index_with(self, dataset: str, *, status: int, body: str = "") -> None:
        self.server.expect_request(f"/{dataset}/").respond_with_data(
            body, status=status
        )

    def fail_then_recover(self, path: str, *, fail_count: int, status: int) -> None:
        state = {"calls": 0}

        def handler(request: Request) -> Response:
            state["calls"] += 1
            if state["calls"] <= fail_count:
                return Response(b"", status=status)
            published = next(iter(self._published.values()))
            relpath = path.lstrip("/").split("/", 1)[1]
            return Response(published.files[relpath], status=200)

        self.server.expect_request(path).respond_with_handler(handler)

    def slow_response(self, path: str, *, delay_seconds: float) -> None:
        import time

        from pytest_httpserver.httpserver import HandlerType

        published = next(iter(self._published.values()))
        relpath = path.lstrip("/").split("/", 1)[1]
        body = published.files[relpath]

        def handler(request: Request) -> Response:
            time.sleep(delay_seconds)
            return Response(body, status=200)

        # ONESHOT takes precedence over the permanent handler already
        # registered by ``publish`` for the same path, and only intercepts
        # the next matching request.
        self.server.expect_request(
            path, handler_type=HandlerType.ONESHOT
        ).respond_with_handler(handler)

    def serve_with_range(self, path: str) -> None:
        """Serve ``path`` with full byte-range request support (HTTP 206)."""
        from pytest_httpserver.httpserver import HandlerType

        published = next(iter(self._published.values()))
        relpath = path.lstrip("/").split("/", 1)[1]
        body = published.files[relpath]

        def handler(request: Request) -> Response:
            range_header = request.headers.get("Range")
            if not range_header:
                return Response(
                    body, status=200, headers={"Content-Length": str(len(body))}
                )
            assert range_header.startswith("bytes=")
            start_str, _, end_str = range_header.removeprefix("bytes=").partition("-")
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else len(body) - 1
            sliced = body[start : end + 1]
            return Response(
                sliced,
                status=206,
                headers={
                    "Content-Range": f"bytes {start}-{end}/{len(body)}",
                    "Content-Length": str(len(sliced)),
                },
            )

        # ONESHOT so it takes precedence over the permanent file handler
        # for the next matching request.
        self.server.expect_request(
            path, handler_type=HandlerType.ONESHOT
        ).respond_with_handler(handler)

    def drop_after_bytes(self, path: str, *, after: int) -> None:
        """Send only ``after`` bytes then close the connection."""
        from pytest_httpserver.httpserver import HandlerType

        published = next(iter(self._published.values()))
        relpath = path.lstrip("/").split("/", 1)[1]
        body = published.files[relpath]

        def handler(request: Request) -> Response:
            # Return only ``after`` bytes but advertise the full Content-Length.
            return Response(
                body[:after],
                status=200,
                headers={"Content-Length": str(len(body))},
            )

        # ONESHOT so it takes precedence over the permanent file handler
        # for the next matching request.
        self.server.expect_request(
            path, handler_type=HandlerType.ONESHOT
        ).respond_with_handler(handler)
