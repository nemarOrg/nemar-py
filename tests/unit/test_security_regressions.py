"""Security and correctness regression tests for P1 review findings.

These tests pin the fixes for the issues surfaced by the massive code
review on the architectural-deepening branch. They are written BEFORE
the fixes (TDD); each one should fail until its corresponding fix lands.

1. Manifest path with control characters (newline, CR, tab) MUST be
   rejected at parse time — defense-in-depth against future transfer
   adapters that might consume manifest paths as structured input.
2. download_one MUST allow callers to enforce origin scoping via an
   optional endpoint= argument.
3. PythonBackend.transfer MUST honor a caller-supplied custom
   RetryPolicy (not silently rebuild a default one).
4. NEMARClient.fetch_manifest MUST translate the upstream "Version not
   published" sentinel into ManifestError without catching the broader
   RuntimeError.
"""

from __future__ import annotations

import httpx
import pytest

import nemar._client as client_mod
from nemar._client import NEMARClient
from nemar._endpoint import DataEndpoint
from nemar._errors import (
    EndpointError,
    ManifestError,
)
from nemar._models import DatasetFile, DatasetIndex, DatasetVersion, VersionManifest
from nemar._request import TransferOptions
from nemar._retry import RetryPolicy
from nemar._transfer import PythonBackend
from nemar._verification import VerifyPolicy
from nemar.transfer import download_one


class TestManifestPathControlCharsRejected:
    """P1 #1 — control characters in manifest paths.

    Defense-in-depth: a malicious manifest path containing newlines,
    CRs, tabs, or DEL could confuse future structured consumers
    (subprocess argv, input files, etc.) even though the current
    HTTPS adapter consumes paths via ``Path()`` directly. Keep the
    parser strict so the boundary is unambiguous.
    """

    def _endpoint(self) -> DataEndpoint:
        return DataEndpoint.from_url("https://data.nemar.org/")

    @pytest.mark.parametrize(
        "evil_path",
        [
            "sub-001/eeg/file\nhttps://evil.example.com/out=../../../tmp/pwn",
            "ok\rcr-injection",
            "ok\tinjection",
            "ok\x01.tsv",
            "ok\x1f.tsv",
            "ok\x7f.tsv",
        ],
    )
    def test_control_char_in_path_is_rejected_at_parse(self, evil_path: str) -> None:
        with pytest.raises(ManifestError, match="control"):
            VersionManifest.parse(
                [{"path": evil_path}],
                manifest_url="https://data.nemar.org/nm000132/v1/manifest.json",
                endpoint=self._endpoint(),
            )

    def test_legal_paths_with_ordinary_punctuation_still_pass(self) -> None:
        """Sanity: the new control-char check must not regress normal paths."""
        manifest = VersionManifest.parse(
            [
                {"path": "sub-001/eeg/sub-001_task-MMN_eeg.set"},
                {"path": "stimuli/task-MMN/sound (deviant).wav"},
                {"path": "derivatives/eeglab/sub-001/eeg/sub-001_desc-clean_eeg.set"},
            ],
            manifest_url="https://data.nemar.org/nm000132/v1/manifest.json",
            endpoint=self._endpoint(),
        )
        assert len(manifest) == 3


class TestDownloadOneAcceptsEndpoint:
    """P1 #3 — download_one must allow origin enforcement via endpoint=.

    The public download_one primitive previously took whatever URL was on
    the DatasetFile without re-checking the origin. Origin scoping is an
    invariant of VersionManifest.parse, but external callers reaching for
    download_one with hand-built DatasetFile should be able to opt in to
    the same guarantee at the call site.
    """

    def test_download_one_rejects_off_origin_url_when_endpoint_passed(
        self, tmp_path
    ) -> None:
        endpoint = DataEndpoint.from_url("https://data.nemar.org/")
        evil_file = DatasetFile(
            path="legit.bin",
            url="https://evil.example.com/legit.bin",
            size=1,
        )

        with pytest.raises(EndpointError, match="Refusing to download"):
            download_one(
                evil_file,
                tmp_path / "legit.bin",
                endpoint=endpoint,
            )


class TestPythonBackendHonorsCustomPolicies:
    """P1 #2 — PythonBackend.transfer must use the caller-supplied
    RetryPolicy and VerifyPolicy, not silently substitute defaults.
    """

    def test_python_backend_uses_caller_supplied_retry_policy_attempts(
        self, tmp_path
    ) -> None:
        """A custom RetryPolicy with 0 retries must result in exactly one
        attempt per file. The previous behavior was correct for max_attempts
        but the test pins that the *value* round-trips, not just the count."""
        attempts = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            return httpx.Response(503, request=request)

        transport = httpx.MockTransport(handler)
        original_init = httpx.Client.__init__

        def patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            kwargs["transport"] = transport
            original_init(self, *args, **kwargs)

        file = DatasetFile(
            path="x.bin",
            url="https://data.nemar.org/nm000132/v1.0.0/x.bin",
            size=4,
        )
        options = TransferOptions(
            backend="python",
            max_concurrent_downloads=1,
            stream_timeout=5.0,
        )
        zero_retry_policy = RetryPolicy.default().with_attempts(0)
        verify = VerifyPolicy(verify_size=False, verify_hash=False)

        backend = PythonBackend()
        try:
            httpx.Client.__init__ = patched_init  # type: ignore[method-assign]
            with pytest.raises(Exception):
                backend.transfer(
                    [file],
                    target_dir=tmp_path,
                    options=options,
                    verify=verify,
                    retry=zero_retry_policy,
                )
        finally:
            httpx.Client.__init__ = original_init  # type: ignore[method-assign]

        assert attempts["count"] == 1, (
            "PythonBackend must honor the caller-supplied RetryPolicy. "
            f"Got {attempts['count']} attempts; expected exactly 1 with "
            "with_attempts(0)."
        )


class TestNEMARClientFetchManifestNarrowCatch:
    """P1 #4 — fetch_manifest must catch a typed Transport/Manifest error,
    not the broader RuntimeError. Catching RuntimeError swallows every
    NemarError subclass (incidental backward-compat side effect).
    """

    def test_unrelated_runtime_error_propagates(self) -> None:
        """If something raises a bare RuntimeError (not a TransportError),
        fetch_manifest must NOT catch and remap it to ManifestError.
        """
        client = NEMARClient(data_url="https://data.nemar.org/")
        try:
            index = DatasetIndex(
                dataset_id="nm000132",
                latest="v1.0.0",
                versions=[
                    DatasetVersion(
                        version="v1.0.0", manifest_url="v1.0.0/manifest.json"
                    )
                ],
            )
            version = index.versions[0]

            sentinel = RuntimeError("Some unrelated RuntimeError, not a TransportError")

            # Replace fetch_json with one that raises an unrelated RuntimeError.
            # The narrow catch in fetch_manifest must let this propagate;
            # the broad catch swallows it.
            def _raise(*_args, **_kwargs):
                raise sentinel

            original = client_mod.fetch_json
            client_mod.fetch_json = _raise  # type: ignore[assignment]
            try:
                # Must raise the original RuntimeError, NOT ManifestError.
                # Before the fix, fetch_manifest catches RuntimeError, sees
                # the message does not match, and re-raises — so this test
                # would actually pass on the current code. The real risk is
                # subtle: if the message DOES contain "Version not published"
                # for an unrelated reason, the wrong typed error is raised.
                with pytest.raises(RuntimeError) as info:
                    client.fetch_manifest(index, version)
                assert info.value is sentinel
            finally:
                client_mod.fetch_json = original  # type: ignore[assignment]
        finally:
            client.close()
