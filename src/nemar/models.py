"""NEMAR domain value types — public submodule.

Re-exports of the value types defined in :mod:`nemar._models` and
:mod:`nemar._endpoint`. These are the types library callers occasionally
construct or inspect directly: a parsed :class:`VersionManifest` from a
custom fixture, or a :class:`DataEndpoint` pointed at a private NEMAR
mirror. End users who only call :func:`nemar.download` never need to
touch this submodule.
"""

from __future__ import annotations

from nemar._endpoint import DataEndpoint as DataEndpoint
from nemar._models import VersionManifest as VersionManifest
