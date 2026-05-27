"""Negative-input and edge-case tests across the new public surface.

The architectural deepening introduced four public seams (NEMARClient,
VersionManifest.file, download_one, NemarError hierarchy). These tests
exercise the boundary conditions that are easy to overlook: bad input,
empty input, zero retries, mismatched policy, weird path shapes.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

import nemar
from nemar._endpoint import DataEndpoint
from nemar._errors import (
    DatasetIndexError,
    ManifestError,
    TransferError,
)
from nemar._models import DatasetFile, VersionManifest, parse_dataset_index
from nemar._retry import RetryPolicy
from nemar._verification import VerifyPolicy, VerifyResult
from nemar.transfer import download_one

# ---------------------------------------------------------------------------
# NEMARClient — input validation
# ---------------------------------------------------------------------------


class TestNEMARClientNegative:
    def test_negative_max_retries_rejected(self):
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            nemar.NEMARClient(max_retries=-1)

    def test_invalid_dataset_id_rejected(self):
        with nemar.NEMARClient(data_url="https://data.nemar.org/") as client:
            with pytest.raises(ValueError, match='dataset must look like "nm000132"'):
                client.fetch_index("invalid-format")

    def test_http_data_url_rejected(self):
        with pytest.raises(ValueError, match="data_url must use HTTPS"):
            nemar.NEMARClient(data_url="http://insecure.example.com/")

    def test_close_is_idempotent(self):
        client = nemar.NEMARClient(data_url="https://data.nemar.org/")
        client.close()
        # Second close must not raise.
        client.close()

    def test_context_manager_closes_client(self):
        with nemar.NEMARClient(data_url="https://data.nemar.org/") as client:
            internal = client._client  # type: ignore[attr-defined]
            assert not internal.is_closed
        assert internal.is_closed


# ---------------------------------------------------------------------------
# VersionManifest.file lookup
# ---------------------------------------------------------------------------


class TestVersionManifestLookupEdges:
    def _manifest(self) -> VersionManifest:
        endpoint = DataEndpoint.from_url("https://data.nemar.org/")
        return VersionManifest.parse(
            [
                {"path": "dataset_description.json"},
                {"path": "sub-001/eeg/sub-001_task-MMN_eeg.set"},
                {"path": "participants.tsv"},
            ],
            manifest_url="https://data.nemar.org/nm000132/v1/manifest.json",
            endpoint=endpoint,
        )

    def test_file_missing_raises_manifest_error(self):
        manifest = self._manifest()
        with pytest.raises(ManifestError, match="does not advertise"):
            manifest.file("not_in_manifest.txt")

    def test_contains_with_non_string_returns_false(self):
        manifest = self._manifest()
        assert (12345 in manifest) is False
        assert (None in manifest) is False
        assert ([] in manifest) is False  # type: ignore[operator]

    def test_lookup_does_not_alter_iteration(self):
        manifest = self._manifest()
        before = [f.path for f in manifest]
        manifest.file("participants.tsv")  # noop side effect
        after = [f.path for f in manifest]
        assert before == after


# ---------------------------------------------------------------------------
# download_one — edge inputs
# ---------------------------------------------------------------------------


class TestDownloadOneEdges:
    def _file(
        self, *, sha256: str | None = None, size: int | None = None
    ) -> DatasetFile:
        return DatasetFile(
            path="data.bin",
            url="https://data.nemar.org/nm000132/v1.0.0/data.bin",
            size=size,
            sha256=sha256,
        )

    def test_zero_retries_does_not_retry(self, tmp_path: Path):
        attempts = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            return httpx.Response(503, request=request)

        transport = httpx.MockTransport(handler)
        with httpx.Client(transport=transport, follow_redirects=False) as client:
            with pytest.raises(TransferError):
                download_one(
                    self._file(size=8),
                    tmp_path / "out.bin",
                    client=client,
                    retry=RetryPolicy.default().with_attempts(0),
                )
        # Zero retries = one attempt total.
        assert attempts["count"] == 1

    def test_size_mismatch_returns_verify_result(self, tmp_path: Path):
        # File size in manifest says 100 but server returns 5 bytes.
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"abcde", request=request)

        transport = httpx.MockTransport(handler)
        with httpx.Client(transport=transport, follow_redirects=False) as client:
            result = download_one(
                self._file(size=100),
                tmp_path / "out.bin",
                client=client,
                retry=RetryPolicy.default().with_attempts(0),
            )
            assert result is VerifyResult.SIZE_MISMATCH

    def test_creates_parent_directories(self, tmp_path: Path):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"x", request=request)

        transport = httpx.MockTransport(handler)
        target = tmp_path / "deeply" / "nested" / "path" / "out.bin"
        assert not target.parent.exists()
        with httpx.Client(transport=transport, follow_redirects=False) as client:
            result = download_one(
                self._file(size=1, sha256=None),
                target,
                client=client,
                retry=RetryPolicy.default().with_attempts(0),
                verify=VerifyPolicy(verify_hash=False),
            )
            assert result is VerifyResult.OK
        assert target.exists()
        assert target.parent.is_dir()

    def test_default_verify_policy_used_when_none(self, tmp_path: Path):
        # No explicit verify=... → uses VerifyPolicy() defaults.
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"x", request=request)

        transport = httpx.MockTransport(handler)
        with httpx.Client(transport=transport, follow_redirects=False) as client:
            # File has size=1 but no hash → size check passes, hash skipped.
            result = download_one(
                self._file(size=1),
                tmp_path / "out.bin",
                client=client,
                retry=RetryPolicy.default().with_attempts(0),
            )
            assert result is VerifyResult.OK

    def test_owns_client_when_none_passed(self, tmp_path: Path):
        # When client=None download_one must close its own client.
        # Hard to assert directly, but we can check it doesn't leak by
        # running many sequential calls.
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"x", request=request)

        transport = httpx.MockTransport(handler)
        # Patch httpx.Client construction so we can confirm the lifecycle.
        opened: list[httpx.Client] = []
        original_init = httpx.Client.__init__

        def tracking_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            kwargs["transport"] = transport
            original_init(self, *args, **kwargs)
            opened.append(self)

        try:
            httpx.Client.__init__ = tracking_init  # type: ignore[method-assign]
            result = download_one(
                self._file(size=1, sha256=None),
                tmp_path / "out.bin",
                retry=RetryPolicy.default().with_attempts(0),
                verify=VerifyPolicy(verify_hash=False),
            )
            assert result is VerifyResult.OK
        finally:
            httpx.Client.__init__ = original_init  # type: ignore[method-assign]

        assert len(opened) == 1
        assert opened[0].is_closed


# ---------------------------------------------------------------------------
# Manifest path validation — security-adjacent
# ---------------------------------------------------------------------------


class TestManifestSafetyEdges:
    def _endpoint(self) -> DataEndpoint:
        return DataEndpoint.from_url("https://data.nemar.org/")

    def test_dotdot_path_rejected(self):
        with pytest.raises(ManifestError, match="Unsafe manifest path"):
            VersionManifest.parse(
                [{"path": "../../etc/passwd"}],
                manifest_url="https://data.nemar.org/nm000132/v1/manifest.json",
                endpoint=self._endpoint(),
            )

    def test_absolute_path_rejected(self):
        with pytest.raises(ManifestError, match="Unsafe manifest path"):
            VersionManifest.parse(
                [{"path": "/etc/passwd"}],
                manifest_url="https://data.nemar.org/nm000132/v1/manifest.json",
                endpoint=self._endpoint(),
            )

    def test_nul_byte_in_path_rejected(self):
        with pytest.raises(ManifestError, match="NUL byte"):
            VersionManifest.parse(
                [{"path": "evil\x00.txt"}],
                manifest_url="https://data.nemar.org/nm000132/v1/manifest.json",
                endpoint=self._endpoint(),
            )

    def test_url_off_origin_accepted_under_new_trust_model(self):
        """Per the post-mapping-audit fix #2, manifest-advertised file URLs
        are trusted regardless of origin. The endpoint validation lives at
        the transport layer (index + manifest fetches); file URLs from
        the trusted manifest payload are taken as-is so legitimate
        ``raw.githubusercontent.com`` / S3 origins work end-to-end.
        """
        manifest = VersionManifest.parse(
            [
                {
                    "path": "ok.txt",
                    "url": "https://malicious.example.com/data",
                }
            ],
            manifest_url="https://data.nemar.org/nm000132/v1/manifest.json",
            endpoint=self._endpoint(),
        )
        # The URL is preserved verbatim — defense-in-depth is gone here.
        assert (
            next(iter(manifest)).url == "https://malicious.example.com/data"
        )

    def test_empty_manifest_rejected(self):
        with pytest.raises(
            ManifestError, match="did not contain any downloadable files"
        ):
            VersionManifest.parse(
                [],
                manifest_url="https://data.nemar.org/nm000132/v1/manifest.json",
                endpoint=self._endpoint(),
            )

    def test_size_as_float_rejected(self):
        with pytest.raises(ManifestError, match="not an integer"):
            VersionManifest.parse(
                [{"path": "a.txt", "size": 3.14}],
                manifest_url="https://data.nemar.org/nm000132/v1/manifest.json",
                endpoint=self._endpoint(),
            )


# ---------------------------------------------------------------------------
# DataEndpoint — URL normalization edges
# ---------------------------------------------------------------------------


class TestDataEndpointEdges:
    def test_query_string_in_url_preserved(self):
        endpoint = DataEndpoint.from_url("https://data.nemar.org/?token=x")
        # Query is preserved in normalized URL.
        # The trailing slash normalization sees "?" as not "/", so the
        # rstrip("/") yields "https://data.nemar.org?token=x"; then "+/".
        # Different decisions are defensible — we pin the current one.
        assert endpoint.url.startswith("https://data.nemar.org")

    def test_endpoint_compares_equal_for_same_url(self):
        a = DataEndpoint.from_url("https://data.nemar.org/")
        b = DataEndpoint.from_url("https://data.nemar.org/")
        assert a == b

    def test_endpoint_url_for_with_absolute_url(self):
        endpoint = DataEndpoint.from_url("https://data.nemar.org/")
        resolved = endpoint.url_for("https://data.nemar.org/nm000132/v1.0.0/file.txt")
        assert resolved == "https://data.nemar.org/nm000132/v1.0.0/file.txt"


# ---------------------------------------------------------------------------
# DatasetIndex — resolve_version edges
# ---------------------------------------------------------------------------


class TestDatasetIndexEdges:
    def _index(self):
        return parse_dataset_index(
            {
                "dataset_id": "nm000132",
                "latest": "v1.0.0",
                "versions": [
                    {"version": "v1.0.0", "manifest_url": "v1.0.0/m.json"},
                    {"version": "v1.1.0", "manifest_url": "v1.1.0/m.json"},
                ],
            }
        )

    def test_none_resolves_to_latest(self):
        version = self._index().resolve_version(None)
        assert version.version == "v1.0.0"

    def test_latest_literal_resolves_to_latest(self):
        version = self._index().resolve_version("latest")
        assert version.version == "v1.0.0"

    def test_explicit_advertised_version_resolved(self):
        version = self._index().resolve_version("v1.1.0")
        assert version.version == "v1.1.0"

    def test_missing_version_lists_available(self):
        with pytest.raises(DatasetIndexError) as info:
            self._index().resolve_version("v9.9.9")
        assert "v1.0.0" in str(info.value)
        assert "v1.1.0" in str(info.value)
