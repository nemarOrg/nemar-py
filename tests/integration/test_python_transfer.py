"""Integration: real HTTPS for the Python streaming downloader."""

from __future__ import annotations

import hashlib
import socket
from pathlib import Path

import httpx
import pytest
import s3fs
from tqdm.auto import tqdm

# ``moto`` / ``boto3`` are dev-only deps for the in-memory S3 server
# used by the chain integration tests. CI's minimal install does not
# pull them, so guard and skip the chain tests when missing. The rest
# of this module (HTTPS streaming tests) does not need either.
try:
    import boto3
    from moto.server import ThreadedMotoServer

    _moto_available = True
except ImportError:
    _moto_available = False
    boto3 = None  # type: ignore[assignment]
    ThreadedMotoServer = None  # type: ignore[assignment,misc]

import nemar
from nemar import _streaming
from nemar._backend import TransferOptions
from nemar._models import DatasetFile
from nemar._retry import RetryPolicy
from nemar._transfer import select_backend
from nemar._verification import VerifyPolicy
from tests.fixtures.factories import (
    make_blob,
    make_index,
    make_manifest_entry,
    make_manifest_list,
)

pytestmark = pytest.mark.integration


def _publish_one_file(nemar_endpoint, blob, *, path: str = "data/sample.bin"):
    index = make_index(dataset="nm000132")
    manifest = make_manifest_list(
        [
            make_manifest_entry(
                path=path,
                content=blob.content,
            )
        ]
    )
    nemar_endpoint.publish(
        "nm000132",
        index=index,
        manifest=manifest,
        files={f"v1.0.0/{path}": blob.content},
        metadata={"Name": "nm000132"},
    )


def test_python_downloader_writes_file_and_verifies_sha256(nemar_endpoint, target_dir):
    blob = make_blob(seed=42, size_bytes=1024)
    _publish_one_file(nemar_endpoint, blob)

    nemar.download(
        dataset="nm000132",
        target_dir=target_dir,
        data_url=nemar_endpoint.base_url,
        downloader="python",
        max_concurrent_downloads=1,
    )

    out = target_dir / "data" / "sample.bin"
    assert out.exists()
    assert out.read_bytes() == blob.content


def test_stream_timeout_is_configurable(nemar_endpoint, target_dir):
    """The per-stream HTTP timeout is exposed as a kwarg.

    This currently fails: ``stream_timeout`` is hard-coded to 60.0 inside
    ``_transfer_one_attempt``. Driver test for bug fix #4.
    """
    blob = make_blob(seed=7, size_bytes=256)
    _publish_one_file(nemar_endpoint, blob)
    nemar_endpoint.slow_response("/nm000132/v1.0.0/data/sample.bin", delay_seconds=2.0)

    with pytest.raises((RuntimeError, httpx.ReadTimeout)):
        nemar.download(
            dataset="nm000132",
            target_dir=target_dir,
            data_url=nemar_endpoint.base_url,
            downloader="python",
            max_concurrent_downloads=1,
            max_retries=0,
            stream_timeout=0.1,
        )


def test_progress_does_not_overshoot_when_server_ignores_range(
    tmp_path,
) -> None:
    """Progress bar total equals file size when server returns 200 for a Range request.

    The server ignores the Range header and returns HTTP 200 with the full body.
    Before the fix the bar would overshoot by ``local_size`` (the pre-seeded
    partial bytes), because ``progress.update(local_size)`` ran before the
    request and was never subtracted on the 206→200 downgrade.
    """
    data = b"hello nemar progress bar"
    partial_size = 8  # pre-seeded bytes

    target = tmp_path / "ds"
    target.mkdir()
    out = target / "data" / "sample.bin"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Pre-seed a partial file to trigger the resume / Range branch.
    out.write_bytes(data[:partial_size])

    file = DatasetFile(
        path="data/sample.bin",
        url="https://data.nemar.org/nm000132/v1.0.0/data/sample.bin",
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )

    # Always respond 200 regardless of Range header (simulates non-compliant server).
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=data, request=request)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)

    original_stream = httpx.stream

    def patched_stream(method: str, url: str, **kwargs):
        return client.stream(method, url, **kwargs)

    # Track updates via a mock progress object.
    total_updates: list[int] = []

    class TrackingProgress(tqdm):
        def update(self, n=1):
            total_updates.append(n)
            return super().update(n)

    httpx.stream = patched_stream
    try:
        # Call _transfer_one_attempt directly so we control the progress object.
        with TrackingProgress(
            total=len(data), desc="test", unit="B", unit_scale=True
        ) as progress:
            _streaming._transfer_one_attempt(
                file,
                outfile=out,
                progress=progress,
                stream_timeout=60.0,
            )
    finally:
        httpx.stream = original_stream
        client.close()

    assert out.read_bytes() == data

    # The fix must emit a negative update of exactly -partial_size when the
    # server downgrades from 206 to 200, reversing the pre-credited bytes.
    # Without the fix the update list would never contain a negative value and
    # the bar would overshoot by ``partial_size``.
    assert any(u == -partial_size for u in total_updates), (
        f"Expected a -{partial_size} correction update; got: {total_updates}"
    )

    # The initial positive update (crediting existing partial bytes) must also
    # appear before the correction so we can confirm the full round-trip.
    assert total_updates[0] == partial_size, (
        f"First update should credit partial bytes ({partial_size}); "
        f"got: {total_updates}"
    )


def test_shared_httpx_client_reused_across_files(
    monkeypatch, nemar_endpoint, target_dir
) -> None:
    """One httpx.Client serves the whole streaming batch (G).

    Counts ``httpx.Client.__init__`` invocations during a batch transfer.
    There are several clients across the full ``download()`` flow (one
    for metadata/index, one for the streaming download). The candidate
    G claim is that the *streaming* phase uses exactly one client even
    when many files are transferred -- so the increment that happens
    during the streaming phase should be exactly 1, regardless of file
    count.
    """
    blobs = [make_blob(seed=i, size_bytes=256) for i in range(5)]
    paths = [f"data/file_{i}.bin" for i in range(5)]
    index = make_index(dataset="nm000132")
    manifest = make_manifest_list(
        [
            make_manifest_entry(path=path, content=blob.content)
            for path, blob in zip(paths, blobs, strict=True)
        ]
    )
    nemar_endpoint.publish(
        "nm000132",
        index=index,
        manifest=manifest,
        files={
            f"v1.0.0/{path}": blob.content
            for path, blob in zip(paths, blobs, strict=True)
        },
        metadata={"Name": "nm000132"},
    )

    original_init = httpx.Client.__init__
    init_calls: list[dict] = []

    def patched_init(self, *args, **kwargs):
        init_calls.append({"limits": kwargs.get("limits")})
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", patched_init)

    nemar.download(
        dataset="nm000132",
        target_dir=target_dir,
        data_url=nemar_endpoint.base_url,
        downloader="python",
        max_concurrent_downloads=2,
    )

    # The streaming client is identifiable by its ``limits`` kwarg --
    # the metadata client does not set one.
    streaming_inits = [c for c in init_calls if c["limits"] is not None]
    assert len(streaming_inits) == 1, (
        f"Expected exactly one streaming-client init for 5 files; "
        f"observed {len(streaming_inits)} (total clients: {len(init_calls)})"
    )

    # Verify every file landed correctly to make sure the shared client
    # actually delivered the bytes.
    for path, blob in zip(paths, blobs, strict=True):
        out = target_dir / path
        assert out.exists(), f"missing {path}"
        assert out.read_bytes() == blob.content, f"corrupt {path}"


def test_http_416_on_resume_retries_fresh(tmp_path) -> None:
    """A 416 against a Range request triggers exactly one fresh retry (S7).

    The first call uses ``Range: bytes={partial}-``; the server replies
    416 (Range Not Satisfiable) because the object has changed or is
    shorter. The driver must unlink the stale partial and issue an
    unconditional GET on the next attempt, succeeding without infinite
    loops.
    """
    data = b"fresh download contents"
    partial_size = 8

    target = tmp_path / "ds"
    target.mkdir()
    out = target / "data" / "sample.bin"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"X" * partial_size)  # stale partial

    file = DatasetFile(
        path="data/sample.bin",
        url="https://data.nemar.org/nm000132/v1.0.0/data/sample.bin",
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )

    call_count = {"n": 0, "ranged": 0, "fresh": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if "Range" in request.headers or "range" in request.headers:
            call_count["ranged"] += 1
            return httpx.Response(416, content=b"", request=request)
        call_count["fresh"] += 1
        return httpx.Response(200, content=data, request=request)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)

    original_stream = httpx.stream

    def patched_stream(method: str, url: str, **kwargs):
        return client.stream(method, url, **kwargs)

    httpx.stream = patched_stream
    try:
        with tqdm(
            total=len(data), desc="test", unit="B", unit_scale=True
        ) as progress:
            _streaming._transfer_one_with_python(
                file,
                target,
                RetryPolicy.default().with_attempts(3),
                VerifyPolicy(verify_size=True, verify_hash=True),
                progress,
                60.0,
            )
    finally:
        httpx.stream = original_stream
        client.close()

    # Round trip: at least one Range attempt that 416'd, then exactly one
    # fresh attempt. We must not loop on 416 forever.
    assert call_count["ranged"] >= 1, f"Expected a Range request; got {call_count}"
    assert call_count["fresh"] == 1, (
        f"Expected exactly one fresh retry; got {call_count}"
    )


def _git_blob_sha1(content: bytes) -> str:
    """Compute git's content-addressed blob SHA1 for a payload."""
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()


def _publish_mixed_manifest(nemar_endpoint) -> tuple[bytes, bytes]:
    """Publish one git-tracked sidecar + one annexed binary.

    Returns the two payloads so the test can assert by content. The
    git-tracked file carries ``checksum_algorithm: "git"`` (so it
    materializes with ``git_sha1`` set on the parsed
    :class:`DatasetFile`); the annexed binary carries a bare ``sha256``
    (so its ``git_sha1`` is ``None``). ``--no-data`` is the seam that
    must split those two populations.
    """
    sidecar = b'{"Name": "nm000132", "BIDSVersion": "1.8.0"}'
    binary = b"\x00" * 256
    index = make_index(dataset="nm000132")
    manifest = make_manifest_list(
        [
            {
                "path": "dataset_description.json",
                "size": len(sidecar),
                "checksum_algorithm": "git",
                "checksum": _git_blob_sha1(sidecar),
            },
            make_manifest_entry(path="eeg/sub-001_eeg.set", content=binary),
        ]
    )
    nemar_endpoint.publish(
        "nm000132",
        index=index,
        manifest=manifest,
        files={
            "v1.0.0/dataset_description.json": sidecar,
            "v1.0.0/eeg/sub-001_eeg.set": binary,
        },
        metadata={"Name": "nm000132"},
    )
    return sidecar, binary


def test_no_data_keeps_git_tracked_sidecars_and_skips_annexed_binaries(
    nemar_endpoint, target_dir
) -> None:
    """``no_data=True`` materializes git-tracked sidecars, skips annexed binaries.

    The filter sits at the manifest-selection seam: anything whose
    parsed :class:`DatasetFile.git_sha1` is ``None`` (i.e. the
    ``sha256`` / ``md5`` annexed objects) is dropped before transfer.
    """
    sidecar, _ = _publish_mixed_manifest(nemar_endpoint)

    nemar.download(
        dataset="nm000132",
        target_dir=target_dir,
        data_url=nemar_endpoint.base_url,
        downloader="python",
        max_concurrent_downloads=1,
        no_data=True,
    )

    assert (target_dir / "dataset_description.json").read_bytes() == sidecar
    assert not (target_dir / "eeg" / "sub-001_eeg.set").exists()


def test_no_data_false_still_fetches_annexed_binaries(
    nemar_endpoint, target_dir
) -> None:
    """Default behavior is unchanged: both populations land on disk.

    Pins the negative half of the contract — ``no_data=False`` (the
    default) does NOT silently skip the annexed binaries.
    """
    sidecar, binary = _publish_mixed_manifest(nemar_endpoint)

    nemar.download(
        dataset="nm000132",
        target_dir=target_dir,
        data_url=nemar_endpoint.base_url,
        downloader="python",
        max_concurrent_downloads=1,
    )

    assert (target_dir / "dataset_description.json").read_bytes() == sidecar
    assert (target_dir / "eeg" / "sub-001_eeg.set").read_bytes() == binary


# ---------------------------------------------------------------------------
# S3 chain integration — moto-backed S3 over real HTTPS fallback
# ---------------------------------------------------------------------------


def _free_port_for_chain() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def moto_s3_chain(monkeypatch):
    """Mirror of the unit-test moto fixture, for end-to-end chain tests."""
    s3fs.S3FileSystem.clear_instance_cache()
    port = _free_port_for_chain()
    server = ThreadedMotoServer(port=port)
    server.start()
    endpoint_url = f"http://127.0.0.1:{port}"
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", endpoint_url)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-2")
    try:
        client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name="us-east-2",
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
        )
        try:
            client.create_bucket(
                Bucket="nemar",
                CreateBucketConfiguration={"LocationConstraint": "us-east-2"},
            )
        except client.exceptions.BucketAlreadyOwnedByYou:
            pass
        import json as _json

        client.put_bucket_policy(
            Bucket="nemar",
            Policy=_json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": "*",
                            "Action": "s3:GetObject",
                            "Resource": "arn:aws:s3:::nemar/*",
                        }
                    ],
                }
            ),
        )
        yield client
    finally:
        server.stop()


@pytest.mark.skipif(
    not _moto_available,
    reason="moto / boto3 not installed (dev group); install to run S3 chain tests.",
)
def test_chain_auto_fetches_from_s3_when_available(
    moto_s3_chain, tmp_path: Path
) -> None:
    """``select_backend('auto')`` resolves to a chain whose primary is S3,
    and a published S3 object lands on disk via S3 (no HTTPS hit).
    """
    from nemar.s3 import annex_key_for

    content = b"chain via S3" * 32
    f = DatasetFile(
        path="eeg/run-01.set",
        url="https://does.not.exist/should-never-be-hit",
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )
    moto_s3_chain.put_object(
        Bucket="nemar",
        Key=f"nm000132/objects/{annex_key_for(f)}",
        Body=content,
    )

    backend = select_backend(
        TransferOptions(
            backend="auto", max_concurrent_downloads=1, stream_timeout=60.0
        ),
        dataset="nm000132",
        datalad_url=None,
    )
    backend.transfer(
        [f],
        target_dir=tmp_path,
        options=TransferOptions(
            backend="auto", max_concurrent_downloads=1, stream_timeout=60.0
        ),
        verify=VerifyPolicy(),
        retry=RetryPolicy.default().with_attempts(0),
    )
    assert (tmp_path / "eeg" / "run-01.set").read_bytes() == content


@pytest.mark.skipif(
    not _moto_available,
    reason="moto / boto3 not installed (dev group); install to run S3 chain tests.",
)
def test_chain_falls_back_to_https_when_s3_misses(
    moto_s3_chain, nemar_endpoint, tmp_path: Path
) -> None:
    """When the S3 object is missing, the chain transparently routes
    the whole batch through the HTTPS fallback served by the local
    pytest-httpserver fixture.
    """
    content = b"chain via HTTPS fallback" * 32
    file = _publish_single_file_via_fixture(
        nemar_endpoint, path="eeg/sub-001/eeg.set", content=content
    )
    # Note: deliberately do NOT publish to moto — first S3 GET 404s.

    backend = select_backend(
        TransferOptions(
            backend="auto", max_concurrent_downloads=1, stream_timeout=60.0
        ),
        dataset="nm000132",
        datalad_url=None,
    )
    backend.transfer(
        [file],
        target_dir=tmp_path,
        options=TransferOptions(
            backend="auto", max_concurrent_downloads=1, stream_timeout=60.0
        ),
        verify=VerifyPolicy(),
        retry=RetryPolicy.default().with_attempts(0),
    )
    assert (tmp_path / file.path).read_bytes() == content


def _publish_single_file_via_fixture(nemar_endpoint, *, path: str, content: bytes):
    """Publish one file via the HTTPS fixture and return its DatasetFile."""
    index = make_index(dataset="nm000132")
    entry = make_manifest_entry(path=path, content=content)
    manifest = make_manifest_list([entry])
    nemar_endpoint.publish(
        "nm000132",
        index=index,
        manifest=manifest,
        files={f"v1.0.0/{path}": content},
    )
    return DatasetFile(
        path=path,
        url=f"{nemar_endpoint.base_url}nm000132/v1.0.0/{path}",
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )
