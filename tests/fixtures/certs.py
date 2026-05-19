"""Local CA + cert helpers for the HTTPS fixture server."""

from __future__ import annotations

import ssl
from dataclasses import dataclass
from pathlib import Path

import trustme


@dataclass(frozen=True)
class LocalCa:
    """A locally generated CA + leaf cert and the paths needed to use them."""

    ca_pem_path: Path
    cert_pem_path: Path
    key_pem_path: Path

    def server_ssl_context(self) -> ssl.SSLContext:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(self.cert_pem_path, self.key_pem_path)
        return ctx


def generate_local_ca(tmp_path: Path) -> LocalCa:
    """Generate a CA + leaf cert valid for localhost / 127.0.0.1 / ::1."""
    ca = trustme.CA()
    cert = ca.issue_cert("localhost", "127.0.0.1", "::1")

    ca_pem = tmp_path / "ca.pem"
    cert_pem = tmp_path / "cert.pem"
    key_pem = tmp_path / "key.pem"

    ca.cert_pem.write_to_path(ca_pem)
    cert.cert_chain_pems[0].write_to_path(cert_pem)
    cert.private_key_pem.write_to_path(key_pem)

    return LocalCa(
        ca_pem_path=ca_pem,
        cert_pem_path=cert_pem,
        key_pem_path=key_pem,
    )
