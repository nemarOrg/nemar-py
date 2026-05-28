"""Toxiproxy harness for network-chaos integration tests.

Sits between the existing local HTTPS fixture server (``nemar_endpoint``)
and the nemar-py client under test. Toxiproxy is a TCP-layer proxy:
TLS handshakes and HTTP semantics pass through unchanged, but every
byte can have latency, bandwidth, or transient-failure toxics applied.

Implementation
--------------

The chaos primitives come from the ``chaostoolkit-toxiproxy``
extension (PyPI: ``chaostoolkit-toxiproxy``). It exposes functional
helpers — ``create_proxy``, ``delete_proxy``, ``create_latency_toxic``,
``create_bandwith_degradation_toxic``, ``create_limiter_toxic``,
``delete_toxic`` — that each take a ``configuration`` dict pointing
at the toxiproxy API endpoint. We boot ``toxiproxy-server`` as a
session-scoped subprocess and yield a per-test :class:`ChaosHandle`
that knows its proxy name + listen port + the shared configuration.

Usage
-----

A test asks for the ``chaos_proxy`` fixture, which yields a
:class:`ChaosHandle`. The handle exposes :meth:`upstream_url` (a
nemar-py-compatible HTTPS URL that goes through the chaos hop) plus
the ``proxy_name`` / ``config`` pair that every ``chaostoxi.*``
helper needs.

Gating
------

The tests that use this fixture are gated by ``NEMAR_CHAOS=1`` because
they require the ``toxiproxy-server`` binary on PATH. Run with::

    NEMAR_CHAOS=1 uv run pytest tests/integration/test_chaos.py
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import pytest
from pytest_httpserver import HTTPServer

# ``chaostoolkit-toxiproxy`` is in the dev dependency group but is not
# part of the default CI install (which only pulls pytest + a handful
# of helpers). Importing it at module top would crash conftest.py on
# every CI run. Wrap it in a try/except so the rest of the test suite
# loads cleanly and the chaos fixtures skip when the package is
# missing.
try:
    from chaostoxi.proxy.actions import (
        create_proxy as _create_proxy,
    )
    from chaostoxi.proxy.actions import (
        delete_proxy as _delete_proxy,
    )
    from chaostoxi.toxic.actions import delete_toxic as _delete_toxic

    _import_error: ImportError | None = None
except ImportError as exc:
    _create_proxy = None  # type: ignore[assignment]
    _delete_proxy = None  # type: ignore[assignment]
    _delete_toxic = None  # type: ignore[assignment]
    _import_error = exc

CHAOS_GATE = "NEMAR_CHAOS"
SKIP_REASON = (
    "Set NEMAR_CHAOS=1 and ensure ``toxiproxy-server`` is on PATH to run "
    "network-chaos tests."
)


@dataclass
class ChaosHandle:
    """Per-test view of a toxiproxy proxy fronting the local HTTPS fixture.

    Carries everything the chaostoolkit-toxiproxy helpers need to
    operate against this specific proxy: the ``proxy_name`` and the
    ``config`` (a dict containing ``toxiproxy_url`` pointing at the
    session-scoped server).

    Example::

        from chaostoxi.toxic.actions import create_latency_toxic
        create_latency_toxic(
            for_proxy=handle.proxy_name,
            toxic_name="lat",
            latency=200,
            configuration=handle.config,
        )
    """

    proxy_name: str
    listen_port: int
    upstream_host: str
    # ``Any`` because ``Configuration`` is a typing alias only
    # importable when ``chaostoolkit-lib`` is installed.
    config: Any

    def upstream_url(self, path: str = "/") -> str:
        """Return ``https://localhost:<chaos_port>/<path>`` for the chaos hop."""
        if not path.startswith("/"):
            path = "/" + path
        return f"https://localhost:{self.listen_port}{path}"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    last_err: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError as exc:  # noqa: PERF203 — short-poll loop
            last_err = exc
            time.sleep(0.05)
    raise RuntimeError(
        f"toxiproxy-server did not come up on port {port} within {timeout}s"
    ) from last_err


def _binary_available() -> bool:
    return shutil.which("toxiproxy-server") is not None


@pytest.fixture(scope="session")
def _toxiproxy_server() -> Iterator[dict[str, Any]]:
    """Boot ``toxiproxy-server`` once per session.

    Yields the shared ``configuration`` dict every chaostoxi helper
    needs (it contains ``toxiproxy_url``). Per-test proxies are
    created in :func:`chaos_proxy` on top of this same config.
    """
    if os.environ.get(CHAOS_GATE) != "1" or not _binary_available():
        pytest.skip(SKIP_REASON)
    if _import_error is not None:
        pytest.skip(
            f"chaostoolkit-toxiproxy is not installed ({_import_error}); "
            "install the dev dependency group to enable chaos tests."
        )
    api_port = _free_port()
    proc = subprocess.Popen(  # noqa: S603 — fixed-arg subprocess
        ["toxiproxy-server", "--host=127.0.0.1", f"--port={api_port}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_port(api_port)
        config: dict[str, Any] = {
            "toxiproxy_url": f"http://127.0.0.1:{api_port}",
        }
        yield config
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


@pytest.fixture
def chaos_proxy(
    _toxiproxy_server: dict[str, Any],
    _nemar_https_server: HTTPServer,
) -> Iterator[ChaosHandle]:
    """Per-test toxiproxy proxy in front of the local HTTPS fixture.

    The fixture creates a fresh proxy named after the test thread,
    forwards it at the local fixture server, yields a
    :class:`ChaosHandle`, then tears the proxy down. The session-scoped
    server keeps running for other tests.
    """
    config = _toxiproxy_server
    upstream_url = _nemar_https_server.url_for("/")
    parsed = urlparse(upstream_url)
    upstream_host = parsed.hostname or "127.0.0.1"
    upstream_port = parsed.port
    assert upstream_port is not None, "expected explicit fixture port"

    listen_port = _free_port()
    name = f"nemar-chaos-{threading.get_ident()}-{listen_port}"
    result = _create_proxy(
        proxy_name=name,
        upstream_host="127.0.0.1",
        upstream_port=upstream_port,
        listen_host="127.0.0.1",
        listen_port=listen_port,
        enabled=True,
        configuration=config,
    )
    assert result is not None, "toxiproxy create_proxy returned None"
    try:
        yield ChaosHandle(
            proxy_name=name,
            listen_port=listen_port,
            upstream_host=upstream_host,
            config=config,
        )
    finally:
        _delete_proxy(proxy_name=name, configuration=config)


# Re-exports kept narrow so tests do not import directly from chaostoxi
# at module top — the optional-dep ``try`` block at the top of this
# module is the single seam.
delete_toxic = _delete_toxic
