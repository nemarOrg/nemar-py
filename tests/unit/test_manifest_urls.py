"""Manifest URL contracts: unsigned, signed, durable and compact forms."""

from dataclasses import replace
from hashlib import sha256

import httpx
import pytest

from nemar import download_files, download_one
from nemar._endpoint import DataEndpoint
from nemar._models import VersionManifest
from nemar._retry import RetryPolicy
from nemar._staging import staging_path
from nemar._verification import VerifyResult
from nemar.errors import EndpointError, ManifestError, TransferError

ENDPOINT = DataEndpoint.from_url("https://data.nemar.org/")
MANIFEST_URL = ENDPOINT.url_for("nm000132/v1.0.0/manifest.json")
PUBLIC = "https://nemar.s3.us-east-2.amazonaws.com/nm000132/objects/key.bin"
SIGNED = PUBLIC + "?X-Amz-Signature=fake%2Fsignature&X-Amz-Expires=3600"
DURABLE = ENDPOINT.url_for("nm000132/v1.0.0/data.bin")
CONTENT = b"manifest URL regression bytes"


def _file(url=PUBLIC):
    return VersionManifest.parse(
        [
            {
                "path": "data.bin",
                "url": url,
                "bytes_url": DURABLE,
                "size": len(CONTENT),
                "checksum_algorithm": "sha256",
                "checksum": sha256(CONTENT).hexdigest(),
            }
        ],
        manifest_url=MANIFEST_URL,
        endpoint=ENDPOINT,
    ).files[0]


@pytest.mark.parametrize("url", [PUBLIC, SIGNED, None])
@pytest.mark.parametrize("bulk", [False, True], ids=["single", "bulk"])
def test_manifest_urls_download_and_resume(monkeypatch, tmp_path, url, bulk):
    file = _file(url)
    assert file.url == (url or DURABLE)
    assert file.bytes_url == DURABLE
    target = tmp_path / file.path
    staging_path(target).write_bytes(CONTENT[:5])
    seen = []

    def handler(request):
        seen.append(str(request.url))
        assert request.headers["range"] == "bytes=5-"
        if str(request.url) == SIGNED:
            return httpx.Response(403)
        if str(request.url) == DURABLE:
            return httpx.Response(302, headers={"Location": PUBLIC})
        assert str(request.url) == PUBLIC
        return httpx.Response(
            206,
            content=CONTENT[5:],
            headers={"Content-Range": f"bytes 5-{len(CONTENT) - 1}/{len(CONTENT)}"},
        )

    original = httpx.Client.__init__

    def patched(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        original(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", patched)
    if bulk:
        download_files([file], tmp_path)
    else:
        # The supplied client's redirect preference must not bypass our scope.
        with httpx.Client(follow_redirects=True) as client:
            assert download_one(file, target, client=client) is VerifyResult.OK
    assert target.read_bytes() == CONTENT
    expected = (
        [PUBLIC]
        if url == PUBLIC
        else ([SIGNED, DURABLE, PUBLIC] if url == SIGNED else [DURABLE, PUBLIC])
    )
    assert seen == expected


@pytest.mark.parametrize("status", [302, 403, 404])
def test_unusable_response_keeps_existing_and_partial_bytes(tmp_path, status):
    target = tmp_path / "data.bin"
    target.write_bytes(b"previous version")
    staging_path(target).write_bytes(CONTENT[:5])
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(status, content=b"error or redirect body")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(TransferError, match=f"HTTP {status}"):
            download_one(_file(SIGNED), target, client=client)
    assert seen == ([SIGNED, DURABLE] if status == 403 else [SIGNED])
    assert target.read_bytes() == b"previous version"
    assert staging_path(target).read_bytes() == CONTENT[:5]


def test_redirect_loop_is_bounded(tmp_path):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(302, headers={"Location": DURABLE})

    with httpx.Client(
        transport=httpx.MockTransport(handler),
        max_redirects=2,
    ) as client:
        with pytest.raises(TransferError, match="Too many redirects"):
            download_one(_file(None), tmp_path / "data.bin", client=client)
    assert len(seen) == 3
    assert not (tmp_path / "data.bin").exists()


@pytest.mark.parametrize("bulk", [False, True])
@pytest.mark.parametrize("off_origin", ["redirect", "fallback"])
def test_explicit_scope_blocks_off_origin_before_request(
    monkeypatch,
    tmp_path,
    bulk,
    off_origin,
):
    file = _file(None)
    if off_origin == "fallback":
        file = replace(file, bytes_url=PUBLIC)
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(302, headers={"Location": PUBLIC})

    original = httpx.Client.__init__

    def patched(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        original(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", patched)
    # Bulk errors are aggregated into TransferError; single errors keep type.
    with pytest.raises((EndpointError, TransferError), match="outside"):
        if bulk:
            download_files([file], tmp_path, endpoint=ENDPOINT)
        else:
            download_one(file, tmp_path / file.path, endpoint=ENDPOINT)
    assert seen == ([DURABLE] if off_origin == "redirect" else [])
    assert not (tmp_path / file.path).exists()


@pytest.mark.parametrize("recovery", [False, True])
def test_cross_origin_requests_do_not_forward_client_auth(tmp_path, recovery):
    file = replace(_file(None), bytes_url=PUBLIC if recovery else DURABLE)

    def handler(request):
        if str(request.url) == DURABLE:
            assert request.headers.get("authorization", "").startswith("Basic ")
            assert request.headers["cookie"] == "session=private"
            return httpx.Response(
                403 if recovery else 302, headers={"Location": PUBLIC}
            )
        assert str(request.url) == PUBLIC
        assert "authorization" not in request.headers
        assert "cookie" not in request.headers
        return httpx.Response(200, content=CONTENT)

    with httpx.Client(
        transport=httpx.MockTransport(handler),
        auth=("user", "password"),
        headers={"Cookie": "session=private"},
    ) as client:
        assert (
            download_one(file, tmp_path / file.path, client=client) is VerifyResult.OK
        )


@pytest.mark.parametrize("version", ["1.0.0", "v1.0.0"])
def test_compact_manifest_downloads_git_and_annex_files(tmp_path, version):
    payload = {
        "dataset_id": "nm000132",
        "version": version,
        "files": {
            "data #1.bin": {
                "key": "SHA256E-s29--unused.bin",
                "size": len(CONTENT),
                "checksum": "sha256:" + sha256(CONTENT).hexdigest(),
            },
            "README": {"key": "git:unused", "size": len(CONTENT)},
        },
    }
    manifest = VersionManifest.parse(
        payload,
        manifest_url="https://nemar.s3.us-east-2.amazonaws.com/"
        "nm000132/version/v1.0.0.json",
        endpoint=ENDPOINT,
    )
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, content=CONTENT)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        for file in manifest:
            assert file.url == file.bytes_url
            assert (
                download_one(
                    file,
                    tmp_path / file.path,
                    client=client,
                    retry=RetryPolicy.default().with_attempts(0),
                )
                is VerifyResult.OK
            )
    assert seen == [
        ENDPOINT.url_for("nm000132/v1.0.0/data%20%231.bin"),
        ENDPOINT.url_for("nm000132/v1.0.0/README"),
    ]
    assert manifest.manifest_url.endswith("/version/v1.0.0.json")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_id", "../elsewhere"),
        ("version", "../v2"),
        ("version", "."),
        ("version", "v1/extra"),
    ],
)
def test_compact_manifest_rejects_unsafe_identity(field, value):
    payload = {
        "dataset_id": "nm000132",
        "version": "v1.0.0",
        "files": {"data.bin": {"size": 1}},
    }
    payload[field] = value
    with pytest.raises(ManifestError):
        VersionManifest.parse(payload, manifest_url=MANIFEST_URL, endpoint=ENDPOINT)
