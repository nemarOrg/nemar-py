"""Value type for the NEMAR HTTPS data endpoint.

The "scoped to ``data.nemar.org``" invariant is real architectural depth in
this client: dataset-index, metadata, and version-manifest URLs all resolve
against the configured data origin, and per-file URLs are refused outside it.
``DataEndpoint`` centralizes that invariant so it lives in one place instead
of being repeated as ad-hoc HTTPS-prefix checks and ``urlparse`` comparisons
across the parser and the downloader.

Public surface (kept small on purpose):

- :meth:`DataEndpoint.from_url` parses, validates HTTPS, and normalizes the
  trailing slash.
- :meth:`DataEndpoint.assert_within` raises ``RuntimeError`` when a URL does
  not share the endpoint's scheme + netloc (including after a redirect).
- :meth:`DataEndpoint.url_for` is an :func:`urllib.parse.urljoin` shim
  against the normalized endpoint URL.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urljoin, urlparse


@dataclass(frozen=True)
class DataEndpoint:
    """The configured NEMAR data origin.

    ``url`` is normalized to a single trailing slash. ``scheme`` and ``netloc``
    are cached at construction so origin checks stay cheap on the hot path
    (per-file manifest validation).
    """

    url: str
    scheme: str
    netloc: str

    @classmethod
    def from_url(cls, raw: str) -> DataEndpoint:
        """Parse, validate, and normalize a data endpoint URL.

        Validates that the URL uses HTTPS, normalizes any number of trailing
        slashes to exactly one, and caches the scheme + netloc.
        """
        # Preserve the historic wording so existing tests keep matching.
        if not raw.startswith("https://"):
            raise ValueError("data_url must use HTTPS.")

        normalized = raw.rstrip("/") + "/"
        parsed = urlparse(normalized)
        return cls(url=normalized, scheme=parsed.scheme, netloc=parsed.netloc)

    def assert_within(self, url: str) -> None:
        """Raise ``RuntimeError`` if ``url`` is not on this endpoint's origin.

        The message wording matches ``_models._validate_data_origin`` so the
        existing parser tests keep matching after delegation.
        """
        parsed = urlparse(url)
        if parsed.scheme != self.scheme or parsed.netloc != self.netloc:
            raise RuntimeError(
                "Refusing to download a file outside the configured NEMAR "
                f"data origin: {url}. This downloader is intentionally "
                f"scoped to {self.url}."
            )

    def url_for(self, relative_or_absolute: str) -> str:
        """Resolve ``relative_or_absolute`` against the endpoint URL.

        Thin shim over :func:`urllib.parse.urljoin` so the orchestrator does
        not need to know how the endpoint URL is normalized.
        """
        return urljoin(self.url, relative_or_absolute)
