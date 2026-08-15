"""Tests for the command line interface."""

import pytest
from typer.testing import CliRunner

import nemar.__main__ as _cli
from nemar._models import DatasetIndex

runner = CliRunner()


def make_index() -> DatasetIndex:
    """Create a representative endpoint index for CLI rendering tests."""
    return DatasetIndex.model_validate(
        {
            "dataset_id": "nm000132",
            "latest": "v1.1.1",
            "versions": [
                {
                    "version": "v1.1.1",
                    "doi": "10.82901/nemar.nm000132.v1.1.1",
                    "created_at": "2026-04-04 06:05:15",
                    "manifest_url": "/nm000132/v1.1.1/manifest.json",
                },
                {
                    "version": "v1.0.0",
                    "manifest_url": "/nm000132/v1.0.0/manifest.json",
                },
            ],
        }
    )


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [
        pytest.param(
            [
                "--subject",
                "001",
                "--subject",
                "002",
                "--session",
                "01",
                "--task",
                "MMN",
                "--run",
                "01",
                "--acq",
                "high",
                "--datatype",
                "eeg",
                "--suffix",
                "eeg",
                "--extension",
                "set",
            ],
            {
                "subject": ["001", "002"],
                "session": ["01"],
                "task": ["MMN"],
                "run": ["01"],
                "acquisition": ["high"],
                "datatype": ["eeg"],
                "suffix": ["eeg"],
                "extension": ["set"],
            },
            id="raw-bids-filters",
        ),
        pytest.param(
            [
                "--scope",
                "derivatives",
                "--pipeline",
                "eeglab",
                "--entity",
                "desc=clean",
                "--include",
                "sub-001",
                "--exclude",
                "*.fdt",
            ],
            {
                "scope": ["derivatives"],
                "pipeline": ["eeglab"],
                "entity": ["desc=clean"],
                "include": ["sub-001"],
                "exclude": ["*.fdt"],
            },
            id="scope-pipeline-and-globs",
        ),
    ],
)
def test_cli_download_passes_bids_filters(monkeypatch, extra_args, expected) -> None:
    """CLI BIDS options are forwarded to the Python entry point."""
    seen = {}

    def download(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(_cli, "download", download)

    result = runner.invoke(
        _cli.app,
        [
            "download",
            "nm000132",
            "--tag",
            "latest",
            "--target-dir",
            "data",
            "--downloader",
            "python",
            "--max-concurrent-downloads",
            "2",
            *extra_args,
        ],
    )

    assert result.exit_code == 0
    assert seen["dataset"] == "nm000132"
    assert seen["tag"] == "latest"
    assert seen["target_dir"].as_posix() == "data"
    assert seen["downloader"] == "python"
    assert seen["max_concurrent_downloads"] == 2
    for key, value in expected.items():
        assert seen[key] == value


@pytest.mark.parametrize(
    ("command", "patch_name", "message"),
    [
        pytest.param(
            ["download", "nm000132"],
            "download",
            "manifest missing",
            id="download-error",
        ),
        pytest.param(
            ["versions", "bad"],
            "fetch_dataset_index",
            "bad dataset",
            id="versions-error",
        ),
    ],
)
def test_cli_reports_errors(monkeypatch, command, patch_name, message) -> None:
    """Runtime errors become concise CLI errors."""

    def fail(**kwargs):
        raise RuntimeError(message)

    monkeypatch.setattr(_cli, patch_name, fail)

    result = runner.invoke(_cli.app, command)

    assert result.exit_code == 1
    assert f"Error: {message}" in result.stderr


@pytest.mark.parametrize(
    ("args", "expected_stdout"),
    [
        pytest.param(
            ["versions", "nm000132"],
            ["nm000132 latest: v1.1.1", "v1.1.1 (latest)", "v1.0.0"],
            id="table",
        ),
        pytest.param(
            ["versions", "nm000132", "--json"],
            ['"dataset_id": "nm000132"', '"latest": "v1.1.1"'],
            id="json",
        ),
    ],
)
def test_cli_versions_prints_endpoint_index(monkeypatch, args, expected_stdout) -> None:
    """The versions command renders the endpoint index in supported formats."""
    monkeypatch.setattr(_cli, "fetch_dataset_index", lambda **kwargs: make_index())

    result = runner.invoke(_cli.app, args)

    assert result.exit_code == 0
    for expected in expected_stdout:
        assert expected in result.stdout


def test_cli_version_callback() -> None:
    """The global version flag exits after printing package version."""
    result = runner.invoke(_cli.app, ["--version"])

    assert result.exit_code == 0
    assert "nemar-py" in result.stdout


def test_cli_no_data_forwards_flag(monkeypatch) -> None:
    """``--no-data`` reaches ``download(...)`` as ``no_data=True``."""
    seen: dict = {}
    monkeypatch.setattr(_cli, "download", lambda **kwargs: seen.update(kwargs))

    result = runner.invoke(_cli.app, ["download", "nm000132", "--no-data"])

    assert result.exit_code == 0, result.stderr
    assert seen["no_data"] is True


def test_cli_stimuli_and_derivatives_merge_with_default_scope(monkeypatch) -> None:
    """Convenience flags expand to ``scope=['raw', 'stimuli', 'derivatives']``."""
    seen: dict = {}
    monkeypatch.setattr(_cli, "download", lambda **kwargs: seen.update(kwargs))

    result = runner.invoke(
        _cli.app, ["download", "nm000132", "--stimuli", "--derivatives"]
    )

    assert result.exit_code == 0, result.stderr
    assert seen["scope"] == ["raw", "stimuli", "derivatives"]


def test_cli_sourcedata_merges_with_default_scope(monkeypatch) -> None:
    """``--sourcedata`` expands to ``scope=['raw', 'sourcedata']``."""
    seen: dict = {}
    monkeypatch.setattr(_cli, "download", lambda **kwargs: seen.update(kwargs))

    result = runner.invoke(_cli.app, ["download", "nm000132", "--sourcedata"])

    assert result.exit_code == 0, result.stderr
    assert seen["scope"] == ["raw", "sourcedata"]


def test_cli_all_scope_flags_merge(monkeypatch) -> None:
    """The three scope flags compose in a stable order."""
    seen: dict = {}
    monkeypatch.setattr(_cli, "download", lambda **kwargs: seen.update(kwargs))

    result = runner.invoke(
        _cli.app,
        ["download", "nm000132", "--stimuli", "--derivatives", "--sourcedata"],
    )

    assert result.exit_code == 0, result.stderr
    assert seen["scope"] == ["raw", "stimuli", "derivatives", "sourcedata"]


def test_cli_explicit_scope_wins_over_sourcedata_flag(monkeypatch) -> None:
    """An explicit ``--scope`` also overrides ``--sourcedata``."""
    seen: dict = {}
    monkeypatch.setattr(_cli, "download", lambda **kwargs: seen.update(kwargs))

    result = runner.invoke(
        _cli.app,
        ["download", "nm000132", "--scope", "raw", "--sourcedata"],
    )

    assert result.exit_code == 0, result.stderr
    assert seen["scope"] == ["raw"]


def test_cli_explicit_scope_wins_over_convenience_flags(monkeypatch) -> None:
    """An explicit ``--scope`` is the contract; convenience flags step aside."""
    seen: dict = {}
    monkeypatch.setattr(_cli, "download", lambda **kwargs: seen.update(kwargs))

    result = runner.invoke(
        _cli.app,
        [
            "download",
            "nm000132",
            "--scope",
            "derivatives",
            "--stimuli",
        ],
    )

    assert result.exit_code == 0, result.stderr
    assert seen["scope"] == ["derivatives"]


def test_cli_metadata_timeout_forwards(monkeypatch) -> None:
    """``--metadata-timeout`` reaches ``download(...)``."""
    seen: dict = {}
    monkeypatch.setattr(_cli, "download", lambda **kwargs: seen.update(kwargs))

    result = runner.invoke(
        _cli.app,
        ["download", "nm000132", "--metadata-timeout", "12.5"],
    )

    assert result.exit_code == 0, result.stderr
    assert seen["metadata_timeout"] == 12.5


def test_cli_output_alias_for_target_dir(monkeypatch) -> None:
    """``-o`` is a shorthand for ``--output`` / ``--target-dir``."""
    seen: dict = {}
    monkeypatch.setattr(_cli, "download", lambda **kwargs: seen.update(kwargs))

    result = runner.invoke(_cli.app, ["download", "nm000132", "-o", "data"])

    assert result.exit_code == 0, result.stderr
    assert seen["target_dir"].as_posix() == "data"


def test_cli_jobs_alias_for_max_concurrent_downloads(monkeypatch) -> None:
    """``-j`` is a shorthand for ``--max-concurrent-downloads``."""
    seen: dict = {}
    monkeypatch.setattr(_cli, "download", lambda **kwargs: seen.update(kwargs))

    result = runner.invoke(_cli.app, ["download", "nm000132", "-j", "4"])

    assert result.exit_code == 0, result.stderr
    assert seen["max_concurrent_downloads"] == 4


def test_cli_downloader_s3_forwards(monkeypatch) -> None:
    """``--downloader s3`` reaches ``download(...)`` as ``downloader='s3'``.

    Pins the new value is wired through the validator, the CLI parser,
    and the kwargs passthrough; ``select_backend`` is what turns it
    into an :class:`S3Backend`, which the transfer-suite covers.
    """
    seen: dict = {}
    monkeypatch.setattr(_cli, "download", lambda **kwargs: seen.update(kwargs))

    result = runner.invoke(_cli.app, ["download", "nm000132", "--downloader", "s3"])

    assert result.exit_code == 0, result.stderr
    assert seen["downloader"] == "s3"


def test_cli_verbose_prints_resolved_params(monkeypatch) -> None:
    """``--verbose`` echoes a parameter summary to stderr before starting."""
    monkeypatch.setattr(_cli, "download", lambda **kwargs: None)

    result = runner.invoke(_cli.app, ["download", "nm000132", "--verbose"])

    assert result.exit_code == 0, result.stderr
    assert "nemar-py download parameters" in result.stderr
    assert "dataset           = nm000132" in result.stderr
