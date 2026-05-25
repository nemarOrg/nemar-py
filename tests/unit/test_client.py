"""Tests for ``NEMARClient`` -- the long-lived metadata seam.

The client owns one ``httpx.Client`` against a configured ``DataEndpoint``
and exposes ``fetch_index`` / ``fetch_metadata`` / ``fetch_manifest`` as the
three metadata operations the orchestrator and the public helpers share.
These tests anchor the construction validation, the context-manager
lifecycle (one client owned, then closed), and the per-method error
translations -- in particular the "Requested X, but the NEMAR index
described Y" and "Version not published" wordings that other callers
already match against.
"""

from __future__ import annotations

from unittest import mock
from unittest.mock import MagicMock

import httpx
import pytest

from nemar._client import NEMARClient
from nemar._endpoint import DataEndpoint
from nemar._models import DatasetIndex, VersionManifest
from tests.fixtures.factories import make_index


def _make_index_obj(
    *,
    dataset: str = "nm000132",
    metadata_url: str | None = "nm000132/metadata.json",
    versions: list[dict] | None = None,
) -> DatasetIndex:
    """Build a parsed DatasetIndex with control over ``metadata_url=None``.

    ``make_index`` substitutes a default when ``metadata_url=None``, so we
    drop the key from the payload after construction when the test wants
    the index to advertise no metadata document.
    """
    if versions is None:
        versions = [
            {"version": "v1.0.0", "manifest_url": "nm000132/v1.0.0/manifest.json"}
        ]
    payload = make_index(
        dataset=dataset, metadata_url=metadata_url, versions=versions
    )
    if metadata_url is None:
        payload.pop("metadata_url", None)
    return DatasetIndex.model_validate(payload)


def _set_client_get(client: NEMARClient, responses_or_errors) -> MagicMock:
    """Wire a mock onto the client's internal httpx client.

    Returns the mock so the test can assert call counts. ``responses_or_errors``
    follows ``MagicMock.side_effect`` semantics.
    """
    mock_httpx = MagicMock()
    if isinstance(responses_or_errors, list):
        mock_httpx.get.side_effect = responses_or_errors
    else:
        mock_httpx.get.return_value = responses_or_errors
    # NEMARClient exposes its internal ``httpx.Client`` through a private
    # ``_client`` attribute so tests can substitute a mock without touching
    # the real network.
    client._client = mock_httpx
    return mock_httpx


class TestConstruction:
    def test_default_endpoint_url_is_data_nemar_org(self) -> None:
        client = NEMARClient()
        assert isinstance(client.endpoint, DataEndpoint)
        assert client.endpoint.url == "https://data.nemar.org/"

    def test_custom_data_url_is_normalized(self) -> None:
        client = NEMARClient(data_url="https://mirror.example.org")
        assert client.endpoint.url == "https://mirror.example.org/"

    def test_non_https_data_url_raises(self) -> None:
        with pytest.raises(ValueError, match="HTTPS"):
            NEMARClient(data_url="http://data.nemar.org/")

    def test_negative_max_retries_raises(self) -> None:
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            NEMARClient(max_retries=-1)

    def test_zero_max_retries_is_allowed(self) -> None:
        """``max_retries=0`` means "one attempt, no retries"."""
        client = NEMARClient(max_retries=0)
        assert client.max_retries == 0

    def test_metadata_timeout_default_matches_public_api(self) -> None:
        """The default metadata timeout matches the ``download()`` default."""
        client = NEMARClient()
        assert client.metadata_timeout == 30.0


class TestContextManager:
    def test_enter_returns_self(self) -> None:
        client = NEMARClient()
        with client as ctx:
            assert ctx is client

    def test_exit_closes_underlying_httpx_client(self) -> None:
        """``__exit__`` must release the owned ``httpx.Client``."""
        with mock.patch("nemar._client.httpx.Client") as fake_client_cls:
            fake_instance = MagicMock()
            fake_client_cls.return_value = fake_instance
            with NEMARClient() as client:
                # Constructing the client lazily owns one ``httpx.Client``.
                _ = client.endpoint  # touch to ensure no error on entry
            assert fake_instance.close.called

    def test_constructor_passes_timeout_and_headers_to_httpx(self) -> None:
        """The internal httpx.Client carries the user-agent and accept headers."""
        with mock.patch("nemar._client.httpx.Client") as fake_client_cls:
            fake_client_cls.return_value = MagicMock()
            with NEMARClient(metadata_timeout=12.5):
                pass
            # Inspect how httpx.Client was constructed.
            assert fake_client_cls.called
            kwargs = fake_client_cls.call_args.kwargs
            assert kwargs.get("follow_redirects") is True
            assert kwargs.get("timeout") == 12.5
            headers = kwargs.get("headers") or {}
            assert headers.get("accept") == "application/json"
            assert "nemar-py/" in headers.get("user-agent", "")


class TestFetchIndex:
    def test_validates_dataset_id_pattern(self) -> None:
        with NEMARClient() as client:
            with pytest.raises(ValueError, match="dataset must look like"):
                client.fetch_index("bad-id")

    def test_returns_dataset_index_with_matching_id(self) -> None:
        with NEMARClient() as client:
            _set_client_get(
                client,
                httpx.Response(
                    200,
                    json=make_index(dataset="nm000132"),
                    request=httpx.Request(
                        "GET", "https://data.nemar.org/nm000132/"
                    ),
                ),
            )
            index = client.fetch_index("nm000132")
        assert isinstance(index, DatasetIndex)
        assert index.dataset_id == "nm000132"

    def test_raises_when_payload_describes_other_dataset(self) -> None:
        with NEMARClient() as client:
            _set_client_get(
                client,
                httpx.Response(
                    200,
                    json=make_index(
                        dataset="nm000999",
                        metadata_url=None,
                        versions=[
                            {"version": "v1.0.0", "manifest_url": "/manifest.json"}
                        ],
                    ),
                    request=httpx.Request(
                        "GET", "https://data.nemar.org/nm000132/"
                    ),
                ),
            )
            with pytest.raises(RuntimeError, match="described nm000999"):
                client.fetch_index("nm000132")


class TestFetchMetadata:
    def test_returns_none_when_metadata_url_is_missing(self) -> None:
        """The index can advertise no metadata document."""
        with NEMARClient() as client:
            # No mock wiring needed -- the helper short-circuits before any GET.
            index = _make_index_obj(metadata_url=None)
            assert client.fetch_metadata(index) is None

    def test_returns_dict_when_metadata_url_is_advertised(self) -> None:
        with NEMARClient() as client:
            _set_client_get(
                client,
                httpx.Response(
                    200,
                    json={"Name": "nm000132", "BIDSVersion": "1.8.0"},
                    request=httpx.Request(
                        "GET", "https://data.nemar.org/nm000132/metadata.json"
                    ),
                ),
            )
            index = _make_index_obj()
            result = client.fetch_metadata(index)
        assert result == {"Name": "nm000132", "BIDSVersion": "1.8.0"}

    def test_raises_when_metadata_payload_is_not_an_object(self) -> None:
        with NEMARClient() as client:
            _set_client_get(
                client,
                httpx.Response(
                    200,
                    json=["not", "an", "object"],
                    request=httpx.Request(
                        "GET", "https://data.nemar.org/nm000132/metadata.json"
                    ),
                ),
            )
            index = _make_index_obj()
            with pytest.raises(RuntimeError, match="must be a JSON object"):
                client.fetch_metadata(index)


class TestFetchManifest:
    def test_returns_version_manifest(self) -> None:
        manifest_payload = [
            {
                "path": "dataset_description.json",
                "url": "https://data.nemar.org/nm000132/v1.0.0/dataset_description.json",
                "size": 2,
            }
        ]
        with NEMARClient() as client:
            _set_client_get(
                client,
                httpx.Response(
                    200,
                    json=manifest_payload,
                    request=httpx.Request(
                        "GET", "https://data.nemar.org/nm000132/v1.0.0/manifest.json"
                    ),
                ),
            )
            index = _make_index_obj()
            version = index.versions[0]
            manifest = client.fetch_manifest(index, version)
        assert isinstance(manifest, VersionManifest)
        assert len(manifest) == 1

    def test_translates_version_not_published_error(self) -> None:
        """A 404 with the magic "Version not published" body becomes a
        human-readable RuntimeError that explicitly mentions the data
        endpoint scope.
        """
        with NEMARClient() as client:
            _set_client_get(
                client,
                httpx.Response(
                    404,
                    json={"error": "Version not published"},
                    request=httpx.Request(
                        "GET", "https://data.nemar.org/nm000132/v1.0.0/manifest.json"
                    ),
                ),
            )
            index = _make_index_obj()
            version = index.versions[0]
            with pytest.raises(RuntimeError, match="will not fall back to S3"):
                client.fetch_manifest(index, version)

    def test_resolves_manifest_url_through_endpoint(self) -> None:
        """The manifest URL is composed against the client's endpoint."""
        captured_urls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_urls.append(str(request.url))
            return httpx.Response(
                200,
                json=[
                    {
                        "path": "dataset_description.json",
                        "url": (
                            "https://data.nemar.org/nm000132/v1.0.0/"
                            "dataset_description.json"
                        ),
                        "size": 2,
                    }
                ],
                request=request,
            )

        transport = httpx.MockTransport(handler)
        with NEMARClient() as client:
            client._client = httpx.Client(transport=transport)
            index = _make_index_obj()
            version = index.versions[0]
            client.fetch_manifest(index, version)
        assert captured_urls == [
            "https://data.nemar.org/nm000132/v1.0.0/manifest.json"
        ]


class TestEndpointProperty:
    def test_endpoint_property_is_accessible(self) -> None:
        client = NEMARClient(data_url="https://mirror.example.org/")
        assert client.endpoint.url == "https://mirror.example.org/"


class TestRetriesPassedThrough:
    def test_fetch_index_retries_retryable_statuses(self) -> None:
        """The client uses its configured ``max_retries`` value."""
        # First call returns 503 (retryable), second returns the index.
        with NEMARClient(max_retries=1) as client:
            _set_client_get(
                client,
                [
                    httpx.Response(
                        503,
                        request=httpx.Request(
                            "GET", "https://data.nemar.org/nm000132/"
                        ),
                    ),
                    httpx.Response(
                        200,
                        json=make_index(dataset="nm000132"),
                        request=httpx.Request(
                            "GET", "https://data.nemar.org/nm000132/"
                        ),
                    ),
                ],
            )
            with mock.patch("nemar._transport.time.sleep"):
                index = client.fetch_index("nm000132")
        assert index.dataset_id == "nm000132"
