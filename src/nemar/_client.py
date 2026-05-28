"""Long-lived handle to the NEMAR data endpoint.

The metadata phase of ``download()`` (index, dataset metadata, version manifest)
and the public ``fetch_dataset_index()`` / ``list_dataset_versions()`` helpers
used to each build their own short-lived :class:`httpx.Client`. Every caller
that touched the endpoint paid a fresh TLS handshake even when fetches were
back-to-back. :class:`NEMARClient` bundles ``(DataEndpoint, httpx.Client)`` as
one thing so library callers iterating across many datasets can amortize the
connection-pool and TLS cost.

The orchestrator's three metadata steps move onto methods of this class so the
retry/error-translation contract for each step lives in one place. The class
delegates the raw HTTP retry primitive to :func:`nemar._transport.fetch_json`;
it owns only the per-step domain shaping (the "Requested X, but described Y"
mismatch check, the "metadata must be a JSON object" assertion, the
"Version not published" translation).
"""

from __future__ import annotations

from typing import Any

import httpx

from nemar import __version__
from nemar._endpoint import DataEndpoint
from nemar._models import (
    DatasetIndex,
    DatasetVersion,
    VersionManifest,
    parse_dataset_index,
)
from nemar._request import DATASET_ID_RE, DEFAULT_DATA_URL
from nemar._retry import RetryPolicy
from nemar._transport import fetch_json
from nemar.errors import DatasetIndexError, ManifestError


class NEMARClient:
    """A reusable handle to one configured NEMAR data endpoint.

    ``NEMARClient`` owns exactly one :class:`httpx.Client` for the lifetime
    of the context manager. The three methods (:meth:`fetch_index`,
    :meth:`fetch_metadata`, :meth:`fetch_manifest`) reuse that one client
    and resolve every URL against the bundled :class:`DataEndpoint`, so the
    origin-scoping rule (no off-origin redirects) is enforced uniformly.

    The class is callable both as a context manager and as a long-lived
    object. The httpx client is created at construction time and closed on
    ``__exit__``; calling :meth:`close` explicitly is also supported.
    """

    def __init__(
        self,
        *,
        data_url: str = DEFAULT_DATA_URL,
        metadata_timeout: float = 30.0,
        max_retries: int = 5,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative.")
        self._endpoint = DataEndpoint.from_url(data_url)
        self._metadata_timeout = metadata_timeout
        self._max_retries = max_retries
        self._policy = RetryPolicy.default().with_attempts(max_retries)
        self._client: httpx.Client = httpx.Client(
            follow_redirects=True,
            headers={
                "accept": "application/json",
                "user-agent": f"nemar-py/{__version__}",
            },
            timeout=metadata_timeout,
        )

    @property
    def endpoint(self) -> DataEndpoint:
        """Return the configured data endpoint.

        Exposed so callers that want per-file URL resolution against the
        same origin can do so without re-parsing the data URL.
        """
        return self._endpoint

    @property
    def metadata_timeout(self) -> float:
        return self._metadata_timeout

    @property
    def max_retries(self) -> int:
        return self._max_retries

    def __enter__(self) -> NEMARClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying httpx.Client. Idempotent."""
        self._client.close()

    def fetch_index(self, dataset: str) -> DatasetIndex:
        """Return the dataset index advertised at ``{endpoint}/{dataset}/``.

        Validates ``dataset`` against the NEMAR identifier pattern, then
        delegates to :func:`nemar._transport.fetch_json` with the client's
        retry policy. Raises when the returned payload describes a different
        dataset than the one requested.
        """
        if not DATASET_ID_RE.fullmatch(dataset):
            raise ValueError(
                'dataset must look like "nm000132" or "on005505".'
            )
        payload = fetch_json(
            self._client,
            url=self._endpoint.url_for(f"{dataset}/"),
            what=f"retrieving NEMAR index for {dataset}",
            policy=self._policy,
            endpoint=self._endpoint,
        )
        index = parse_dataset_index(payload)
        if index.dataset_id != dataset:
            raise DatasetIndexError(
                f"Requested {dataset}, but the NEMAR index described "
                f"{index.dataset_id}."
            )
        return index

    def fetch_metadata(self, dataset_index: DatasetIndex) -> dict[str, Any] | None:
        """Return the dataset metadata document advertised by ``dataset_index``.

        Returns ``None`` when ``dataset_index.metadata_url`` is ``None`` (the
        dataset advertises no metadata document). Raises when the payload is
        not a JSON object.
        """
        if dataset_index.metadata_url is None:
            return None
        payload = fetch_json(
            self._client,
            url=self._endpoint.url_for(dataset_index.metadata_url),
            what=f"retrieving NEMAR metadata for {dataset_index.dataset_id}",
            policy=self._policy,
            endpoint=self._endpoint,
        )
        if not isinstance(payload, dict):
            raise ManifestError("The NEMAR metadata payload must be a JSON object.")
        return payload

    def fetch_manifest(
        self,
        dataset_index: DatasetIndex,
        version: DatasetVersion,
    ) -> VersionManifest:
        """Return the parsed :class:`VersionManifest` for ``version``.

        Composes the manifest URL against the client's endpoint, fetches the
        payload with the configured retry policy, translates the upstream
        "Version not published" sentinel into a human-readable error, and
        parses the result into a :class:`VersionManifest`.
        """
        manifest_url = self._endpoint.url_for(version.manifest_url)
        try:
            payload = fetch_json(
                self._client,
                url=manifest_url,
                what=(
                    f"retrieving NEMAR manifest for {dataset_index.dataset_id} "
                    f"{version.version}"
                ),
                policy=self._policy,
                endpoint=self._endpoint,
            )
        except RuntimeError as exc:
            if "Version not published" in str(exc):
                raise ManifestError(
                    f"NEMAR advertises {dataset_index.dataset_id} {version.version} "
                    f"at {manifest_url}, but that version is not published on the "
                    "public data endpoint yet. This downloader only uses "
                    "data.nemar.org and will not fall back to S3."
                ) from exc
            raise
        return VersionManifest.parse(
            payload,
            manifest_url=manifest_url,
            endpoint=self._endpoint,
        )
