"""Tests for the ``DataEndpoint`` value type."""

from __future__ import annotations

import pytest

from nemar._endpoint import DataEndpoint


class TestFromUrl:
    """``DataEndpoint.from_url`` normalizes and validates the data origin."""

    def test_round_trips_normalized_https_url(self) -> None:
        endpoint = DataEndpoint.from_url("https://data.nemar.org/")
        assert endpoint.url == "https://data.nemar.org/"
        assert endpoint.scheme == "https"
        assert endpoint.netloc == "data.nemar.org"

    def test_adds_trailing_slash(self) -> None:
        endpoint = DataEndpoint.from_url("https://data.nemar.org")
        assert endpoint.url == "https://data.nemar.org/"

    def test_collapses_extra_trailing_slashes(self) -> None:
        endpoint = DataEndpoint.from_url("https://data.nemar.org///")
        assert endpoint.url == "https://data.nemar.org/"

    def test_rejects_http(self) -> None:
        with pytest.raises(ValueError, match="HTTPS"):
            DataEndpoint.from_url("http://data.nemar.org/")

    def test_rejects_non_http_scheme(self) -> None:
        with pytest.raises(ValueError, match="HTTPS"):
            DataEndpoint.from_url("ftp://data.nemar.org/")

    def test_accepts_https_localhost_with_port(self) -> None:
        """Fixture servers use ``https://localhost:<port>/``."""
        endpoint = DataEndpoint.from_url("https://localhost:8443/")
        assert endpoint.scheme == "https"
        assert endpoint.netloc == "localhost:8443"


class TestAssertWithin:
    """``DataEndpoint.assert_within`` enforces same-origin file URLs."""

    def test_accepts_same_origin(self) -> None:
        endpoint = DataEndpoint.from_url("https://data.nemar.org/")
        endpoint.assert_within("https://data.nemar.org/nm000132/v1/file.json")

    def test_accepts_root(self) -> None:
        endpoint = DataEndpoint.from_url("https://data.nemar.org/")
        endpoint.assert_within("https://data.nemar.org/")

    def test_rejects_cross_origin(self) -> None:
        endpoint = DataEndpoint.from_url("https://data.nemar.org/")
        with pytest.raises(RuntimeError, match="outside the configured NEMAR"):
            endpoint.assert_within("https://example.org/nm000132/file.json")

    def test_rejects_scheme_mismatch(self) -> None:
        endpoint = DataEndpoint.from_url("https://data.nemar.org/")
        with pytest.raises(RuntimeError, match="outside the configured NEMAR"):
            endpoint.assert_within("http://data.nemar.org/nm000132/file.json")

    def test_rejects_port_mismatch(self) -> None:
        endpoint = DataEndpoint.from_url("https://localhost:8443/")
        with pytest.raises(RuntimeError, match="outside the configured NEMAR"):
            endpoint.assert_within("https://localhost:9999/nm000132/file.json")


class TestUrlFor:
    """``DataEndpoint.url_for`` resolves relative and absolute references."""

    def test_joins_relative_path(self) -> None:
        endpoint = DataEndpoint.from_url("https://data.nemar.org/")
        assert (
            endpoint.url_for("nm000132/manifest.json")
            == "https://data.nemar.org/nm000132/manifest.json"
        )

    def test_joins_absolute_path(self) -> None:
        endpoint = DataEndpoint.from_url("https://data.nemar.org/")
        assert (
            endpoint.url_for("/nm000132/v1/manifest.json")
            == "https://data.nemar.org/nm000132/v1/manifest.json"
        )

    def test_keeps_absolute_url_unchanged(self) -> None:
        endpoint = DataEndpoint.from_url("https://data.nemar.org/")
        assert (
            endpoint.url_for("https://data.nemar.org/nm000132/file.json")
            == "https://data.nemar.org/nm000132/file.json"
        )


class TestImmutability:
    """``DataEndpoint`` is a frozen value type."""

    def test_is_frozen(self) -> None:
        endpoint = DataEndpoint.from_url("https://data.nemar.org/")
        with pytest.raises((AttributeError, TypeError)):
            endpoint.url = "https://example.org/"  # type: ignore[misc]
