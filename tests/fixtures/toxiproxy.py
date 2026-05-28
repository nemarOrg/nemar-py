"""Toxiproxy harness for network-chaos integration tests.

Sits between the existing local HTTPS fixture server (``nemar_endpoint``)
and the nemar-py client under test. Toxiproxy is a TCP-layer proxy:
TLS handshakes and HTTP semantics pass through unchanged, but every
byte can have latency, bandwidth, or transient-failure toxics applied.

Usage
-----

A test asks for the ``chaos_proxy`` fixture, which yields a
:class:`ChaosHandle`. The handle exposes :meth:`upstream_url` (a
nemar-py-compatible HTTPS URL that goes through the chaos hop) and
:meth:`add_toxic` / :meth:`clear_toxics` for in-test scenario control.

The toxiproxy server is launched once per session as a child process
on an ephemeral port. The Python client (:pypi:`toxiproxy-python`)
talks to it over HTTP on that port.

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
from urllib.parse import urlparse

import pytest
import toxiproxy
from pytest_httpserver import HTTPServer
from toxiproxy.api import APIConsumer

CHAOS_GATE = "NEMAR_CHAOS"
SKIP_REASON = (
    "Set NEMAR_CHAOS=1 and ensure ``toxiproxy-server`` is on PATH to run "
    "network-chaos tests."
)


@dataclass
class ChaosHandle:
    """Per-test view of a toxiproxy proxy fronting the local HTTPS fixture.

    Each test gets a fresh proxy. Toxics added during the test are
    cleared on teardown so scenarios cannot leak.
    """

    proxy: toxiproxy.Proxy
    listen_port: int
    upstream_host: str

    def upstream_url(self, path: str = "/") -> str:
        """Return an ``https://localhost:<chaos_port>/<path>`` URL.

        The path is prefixed with ``/`` if missing, matching how
        nemar-py joins dataset id against ``data_url``.
        """
        if not path.startswith("/"):
            path = "/" + path
        return f"https://localhost:{self.listen_port}{path}"

    def add_toxic(self, **kwargs: object) -> object:
        """Attach a toxic to the proxy. See toxiproxy-python docs for shapes.

        Common patterns used by our scenarios:

        * ``add_toxic(name="lat", type="latency", attributes={"latency": 500})``
          — adds 500 ms latency on every downstream byte.
        * ``add_toxic(name="bw", type="bandwidth",
          attributes={"rate": 16})`` — throttles to 16 KB/s.
        * ``add_toxic(name="cut", type="limit_data",
          attributes={"bytes": 4096})`` — closes the connection after
          4 KiB of payload (simulates a mid-stream drop).
        """
        return self.proxy.add_toxic(**kwargs)

    def clear_toxics(self) -> None:
        """Remove every toxic attached to this proxy."""
        for toxic in list(self.proxy.toxics().values()):
            toxic.destroy()


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
def _toxiproxy_server() -> Iterator[toxiproxy.Toxiproxy]:
    """Boot ``toxiproxy-server`` once per session."""
    if os.environ.get(CHAOS_GATE) != "1" or not _binary_available():
        pytest.skip(SKIP_REASON)
    api_port = _free_port()
    proc = subprocess.Popen(  # noqa: S603 — fixed-arg subprocess
        ["toxiproxy-server", "--host=127.0.0.1", f"--port={api_port}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_port(api_port)
        # toxiproxy-python configures the server endpoint at class level
        # rather than via constructor args. Mutating the class is OK
        # here because the fixture is session-scoped (one server, one
        # API port for the whole pytest run).
        APIConsumer.host = "127.0.0.1"
        APIConsumer.port = api_port
        client = toxiproxy.Toxiproxy()
        yield client
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


@pytest.fixture
def chaos_proxy(
    _toxiproxy_server: toxiproxy.Toxiproxy,
    _nemar_https_server: HTTPServer,
) -> Iterator[ChaosHandle]:
    """Per-test toxiproxy proxy in front of the local HTTPS fixture.

    The fixture creates a fresh proxy named after the test thread,
    forwards it at the local fixture server, yields a
    :class:`ChaosHandle`, then tears the proxy down. The session-scoped
    server keeps running for other tests.
    """
    upstream_url = _nemar_https_server.url_for("/")
    parsed = urlparse(upstream_url)
    upstream_host = parsed.hostname or "127.0.0.1"
    upstream_port = parsed.port
    assert upstream_port is not None, "expected explicit fixture port"

    listen_port = _free_port()
    name = f"nemar-chaos-{threading.get_ident()}-{listen_port}"
    proxy = _toxiproxy_server.create(
        upstream=f"127.0.0.1:{upstream_port}",
        name=name,
        listen=f"127.0.0.1:{listen_port}",
        enabled=True,
    )
    try:
        yield ChaosHandle(
            proxy=proxy,
            listen_port=listen_port,
            upstream_host=upstream_host,
        )
    finally:
        # Best-effort teardown: a failing test may leave toxics behind.
        try:
            for toxic in list(proxy.toxics().values()):
                toxic.destroy()
        except Exception:  # noqa: BLE001 — server may already be gone
            pass
        proxy.destroy()
