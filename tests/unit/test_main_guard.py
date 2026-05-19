"""Tests that ``nemar.__main__`` does not invoke the CLI on plain import."""

import importlib
import sys


def test_import_main_does_not_invoke_cli(capsys) -> None:
    """Importing ``nemar.__main__`` must not call ``app()`` or emit output."""
    # Remove any cached module so the import runs fresh.
    sys.modules.pop("nemar.__main__", None)

    importlib.import_module("nemar.__main__")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
