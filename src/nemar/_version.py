"""Single source of truth for the package version.

A leaf module with no ``nemar`` imports, so every consumer — the
public ``nemar.__version__`` re-export, the outbound User-Agent header
(``_client`` / ``_streaming``), the CLI ``--version`` banner, and
setuptools' build-time ``attr`` reader — reads the same literal without
a package-initialization back-edge. Bump this one line per release.
"""

__version__ = "0.3.0"
