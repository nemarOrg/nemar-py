"""Value type for the NEMAR HTTPS data endpoint.

The same-origin invariant this enforces is **control-plane only**:
dataset-index, metadata, and version-manifest JSON are fetched from the
configured data origin, and :func:`nemar._transport.fetch_json` calls
:meth:`assert_within` on the *final* URL so a redirect cannot silently move
a JSON fetch to another host.

It is intentionally **not** applied to file-byte URLs. Real NEMAR manifests
advertise off-origin byte sources by design — ``nemar.s3.us-east-2.amazonaws.com``
for annexed content and ``raw.githubusercontent.com`` for git-tracked
sidecars — so a per-file origin check would reject the normal download. The
trust model is: the control plane is origin-scoped, and file URLs are taken
from the manifest payload that the (origin-scoped) control plane returned.
The public :func:`nemar.transfer.download_one` / ``download_files`` primitives
accept an optional ``endpoint`` for callers that want to opt into per-file
scoping against hand-built inputs; the default ``download()`` flow does not.

Public surface (kept small on purpose):

- :meth:`DataEndpoint.from_url` parses, validates HTTPS, and normalizes the
  trailing slash.
- :meth:`DataEndpoint.assert_within` raises :class:`~nemar.errors.EndpointError`
  when a URL does not share the endpoint's scheme + netloc (used on the JSON
  control plane's post-redirect URL).
- :meth:`DataEndpoint.url_for` is an :func:`urllib.parse.urljoin` shim
  against the normalized endpoint URL.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

from nemar.errors import EndpointError


@dataclass(frozen=True)
class DataEndpoint:
    """The configured NEMAR data origin.

    ``url`` is normalized to a single trailing slash. ``scheme`` and ``netloc``
    are cached at construction so the control-plane redirect-origin check
    (:meth:`assert_within`, run once per JSON fetch) stays cheap.
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
        """Raise :class:`EndpointError` if ``url`` is not on this endpoint's origin.

        The message wording matches ``_models._validate_data_origin`` so the
        existing parser tests keep matching after delegation.
        """
        parsed = urlparse(url)
        if parsed.scheme != self.scheme or parsed.netloc != self.netloc:
            raise EndpointError(
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
