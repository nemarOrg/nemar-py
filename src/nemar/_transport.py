"""JSON-fetch-with-retries against the NEMAR data endpoint.

This module owns the HTTP transport primitive every metadata read shares:
issue a JSON ``GET``, classify the response against the configured
:class:`~nemar._retry.RetryPolicy`, validate the final URL against the
configured :class:`~nemar._endpoint.DataEndpoint` after any redirects,
and reshape exhausted retries into a :class:`TransportError` whose
message names what the caller was trying to do.

Two control-flow exception types live here so the retry loop and the
per-file transfer's retry loop can share the same vocabulary:

* :class:`_RetryableError` -- raised internally when a single attempt
  failed in a way the policy permits retrying. Callers should not catch
  it; the loop translates it into a final :class:`TransportError` once
  retries exhaust.
* :class:`_RetryFreshError` -- a subclass of :class:`_RetryableError`
  used by the per-file transfer (``_download._transfer_one_with_python``)
  to ask its outer loop to abandon any local partial bytes and retry
  with an unconditional ``GET``. It lives here because both retry loops
  share the type.

Both names stay underscored: they are internal to the transport +
transfer collaboration. Users of the public API never see them.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
from tqdm.auto import tqdm

from nemar._endpoint import DataEndpoint
from nemar._retry import RetryPolicy
from nemar.errors import TransportError


class _RetryableError(Exception):
    """Raised when a request can be retried."""


class _RetryFreshError(_RetryableError):
    """Raised when the retry must abandon any local partial bytes.

    Subclass of :class:`_RetryableError` so existing handlers still treat
    it as retryable; the per-file driver distinguishes it to set
    ``force_fresh=True`` on the next attempt.
    """


def fetch_json(
    client: httpx.Client,
    *,
    url: str,
    what: str,
    policy: RetryPolicy,
    endpoint: DataEndpoint | None = None,
) -> Any:
    """Fetch ``url`` as JSON, retrying per ``policy``.

    The loop classifies each attempt:

    * Retryable HTTP status (per ``policy.retryable_status``): retry,
      sleeping for ``policy.next_delay`` between attempts. After the
      final attempt, reshape into a :class:`TransportError` whose message
      starts with ``"Retryable error when {what}"``.
    * Non-retryable HTTP error: reshape immediately into a
      :class:`TransportError` whose message starts with ``"Error when
      {what}: HTTP {status} {detail}"``.
    * Retryable transport exception (per ``policy.retryable_exceptions``):
      retry; on exhaustion reshape into ``"Network error when {what}: ..."``.
    * ``json.JSONDecodeError``: reshape immediately into
      ``"Invalid JSON when {what}: {url}"``.

    On success the final URL (after any redirects) is validated against
    ``endpoint`` when one is supplied. This is the redirect-origin
    invariant the rest of the client relies on: ``httpx`` with
    ``follow_redirects=True`` would otherwise let a 302 to another host
    silently bypass the ``data.nemar.org``-only scope.

    Callers always pass a configured ``policy``; there is no implicit
    default here. The orchestrator builds one from
    :meth:`RetryPolicy.default` and
    :meth:`RetryPolicy.with_attempts` once per request.
    """
    last_attempt = policy.max_attempts - 1
    for attempt in range(policy.max_attempts):
        try:
            response = client.get(url)
            if policy.should_retry_status(response.status_code):
                raise _RetryableError(f"HTTP {response.status_code}")
            if response.is_error:
                detail = _response_detail(response)
                raise TransportError(
                    f"Error when {what}: HTTP {response.status_code} {detail}"
                )
            # Validate the final URL after any redirects against the
            # configured data endpoint. ``httpx`` with ``follow_redirects=True``
            # would otherwise let a 302 to another host silently bypass the
            # ``data.nemar.org``-only scope.
            if endpoint is not None:
                endpoint.assert_within(str(response.url))
            return response.json()
        except policy.retryable_exceptions as exc:
            if attempt == last_attempt:
                raise TransportError(
                    f"Network error when {what}: {exc}"
                ) from exc
        except json.JSONDecodeError as exc:
            raise TransportError(f"Invalid JSON when {what}: {url}") from exc
        except _RetryableError as exc:
            if attempt == last_attempt:
                raise TransportError(
                    f"Retryable error when {what}: {exc}"
                ) from exc

        remaining = last_attempt - attempt
        tqdm.write(f"Retrying after failure when {what} ({remaining} retries remain).")
        time.sleep(policy.next_delay(attempt))

    raise TransportError(f"Unexpected retry exhaustion when {what}.")


def _response_detail(response: httpx.Response) -> str:
    """Return a concise human-readable detail string for an error response.

    Tries ``payload["message"]`` / ``payload["error"]`` for JSON payloads,
    falls back to the JSON text (truncated), then to the plain text body
    (truncated). Used by :func:`fetch_json` when an HTTP status indicates
    an error and a single-line detail is wanted in the raised
    :class:`TransportError`.
    """
    try:
        payload = response.json()
    except json.JSONDecodeError:
        text = response.text.strip()
        return text[:300] if text else ""
    if isinstance(payload, dict):
        detail = payload.get("message") or payload.get("error")
        if detail:
            return str(detail)
    return json.dumps(payload)[:300]
