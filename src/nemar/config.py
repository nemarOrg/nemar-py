"""Request and policy value types — public submodule.

Re-exports of :class:`DownloadRequest`, :class:`TransferOptions`, and
:class:`VerifyResult`. These are advanced/escape-hatch surfaces — most
callers just pass kwargs to :func:`nemar.download` and never touch
:class:`DownloadRequest` directly. The common runtime knob
(:class:`VerifyPolicy`) stays at the top level because it shows up in
the recommended public usage examples.
"""

from __future__ import annotations

from nemar._request import DownloadRequest as DownloadRequest
from nemar._request import TransferOptions as TransferOptions
from nemar._verification import VerifyResult as VerifyResult
