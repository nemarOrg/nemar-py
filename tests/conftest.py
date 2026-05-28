"""Shared pytest fixtures for nemar-py tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.fixtures.certs import LocalCa, generate_local_ca
from tests.fixtures.https_server import NemarFakeEndpoint, start_https_server

# Re-export the toxiproxy fixtures so tests opt in by name without
# importing the module directly. ``chaos_proxy`` (per-test) depends on
# ``_toxiproxy_server`` (session-scoped) which is the actual subprocess.
from tests.fixtures.toxiproxy import (  # noqa: F401 — re-export for pytest
    _toxiproxy_server,
    chaos_proxy,
)


@pytest.fixture(scope="session")
def local_ca(tmp_path_factory: pytest.TempPathFactory) -> LocalCa:
    """Generate one CA + leaf cert per test session."""
    tmp_path = tmp_path_factory.mktemp("nemar-ca")
    return generate_local_ca(tmp_path)


@pytest.fixture(scope="session")
def _nemar_https_server(local_ca: LocalCa) -> Iterator:
    """Start the HTTPS fixture server once per session."""
    server = start_https_server(local_ca)
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def nemar_endpoint(
    _nemar_https_server,
    local_ca: LocalCa,
    monkeypatch: pytest.MonkeyPatch,
) -> NemarFakeEndpoint:
    """Per-test view of the fixture server, with SSL_CERT_FILE pre-wired."""
    _nemar_https_server.clear()
    monkeypatch.setenv("SSL_CERT_FILE", str(local_ca.ca_pem_path))
    return NemarFakeEndpoint(_nemar_https_server)


@pytest.fixture
def target_dir(tmp_path: Path) -> Path:
    """A fresh empty download target directory."""
    out = tmp_path / "ds"
    out.mkdir()
    return out
