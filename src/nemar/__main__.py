"""Command line interface for NEMAR downloads and ``python -m nemar`` entry point."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer

import nemar
from nemar._constants import DEFAULT_DATA_URL
from nemar._download import download, fetch_dataset_index

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@dataclass(frozen=True)
class _BidsFilterArgs:
    """Bundle of the 11 BIDS-filter options the CLI repeats per subcommand.

    Held as a dataclass so future subcommands (``dry-run``, ``list-files``,
    ``verify``) can accept the same shape without duplicating the option
    declarations. ``to_download_kwargs`` returns the dict shape that the
    public ``download()`` function (and any future filter-aware command)
    expects.
    """

    include: list[str] | None
    exclude: list[str] | None
    subject: list[str] | None
    session: list[str] | None
    task: list[str] | None
    run: list[str] | None
    acquisition: list[str] | None
    datatype: list[str] | None
    suffix: list[str] | None
    extension: list[str] | None
    scope: list[str] | None
    pipeline: list[str] | None
    entity: list[str] | None

    def to_download_kwargs(self) -> dict[str, Any]:
        """Return the kwargs slice that maps onto ``download(...)``."""
        return {
            "include": self.include,
            "exclude": self.exclude,
            "subject": self.subject,
            "session": self.session,
            "task": self.task,
            "run": self.run,
            "acquisition": self.acquisition,
            "datatype": self.datatype,
            "suffix": self.suffix,
            "extension": self.extension,
            "scope": self.scope,
            "pipeline": self.pipeline,
            "entity": self.entity,
        }


def _merge_scopes(
    explicit: list[str] | None,
    *,
    stimuli: bool,
    derivatives: bool,
    sourcedata: bool = False,
) -> list[str] | None:
    """Combine explicit ``--scope`` with the scope convenience flags.

    If the caller passed ``--scope`` explicitly we honor it untouched
    (their list is the contract). Otherwise we keep the orchestrator's
    default of ``raw`` and additionally include ``stimuli``,
    ``derivatives`` and/or ``sourcedata`` when their flags are set.
    """
    if explicit:
        return explicit
    if not (stimuli or derivatives or sourcedata):
        return None
    scopes = ["raw"]
    if stimuli:
        scopes.append("stimuli")
    if derivatives:
        scopes.append("derivatives")
    if sourcedata:
        scopes.append("sourcedata")
    return scopes


def _print_resolved_params(
    *,
    dataset: str,
    tag: str | None,
    target_dir: Path | None,
    bids_filters: _BidsFilterArgs,
    downloader: str,
    max_concurrent_downloads: int,
    metadata_timeout: float,
    no_data: bool,
    data_url: str,
) -> None:
    """Echo the resolved download knobs to stderr before starting work."""
    typer.secho("nemar-py download parameters:", err=True, bold=True)
    typer.secho(f"  dataset           = {dataset}", err=True)
    typer.secho(f"  tag               = {tag or 'latest'}", err=True)
    typer.secho(f"  target_dir        = {target_dir or dataset}", err=True)
    typer.secho(f"  scope             = {bids_filters.scope or '[raw]'}", err=True)
    typer.secho(f"  subject           = {bids_filters.subject}", err=True)
    typer.secho(f"  task              = {bids_filters.task}", err=True)
    typer.secho(f"  include           = {bids_filters.include}", err=True)
    typer.secho(f"  exclude           = {bids_filters.exclude}", err=True)
    typer.secho(f"  downloader        = {downloader}", err=True)
    typer.secho(f"  jobs              = {max_concurrent_downloads}", err=True)
    typer.secho(f"  metadata_timeout  = {metadata_timeout}s", err=True)
    typer.secho(f"  no_data           = {no_data}", err=True)
    typer.secho(f"  data_url          = {data_url}", err=True)


@app.command(name="download")
def download_cli(
    dataset: Annotated[
        str,
        typer.Argument(
            metavar="DATASET",
            help="The NEMAR dataset identifier, for example nm000132.",
        ),
    ],
    tag: Annotated[
        str | None,
        typer.Option(
            "--tag",
            "-t",
            help="The NEMAR dataset version, for example v1.1.1 or latest.",
            show_default=False,
        ),
    ] = None,
    target_dir: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            "--target-dir",
            help=(
                "The directory to download to. Defaults to the dataset id "
                "under the current working directory."
            ),
            show_default=False,
        ),
    ] = None,
    include: Annotated[
        list[str] | None,
        typer.Option(
            "--include",
            "-i",
            help="Only include matching files or directories. Can be repeated.",
            show_default=False,
        ),
    ] = None,
    exclude: Annotated[
        list[str] | None,
        typer.Option(
            "--exclude",
            "-x",
            help="Exclude matching files or directories. Can be repeated.",
            show_default=False,
        ),
    ] = None,
    subject: Annotated[
        list[str] | None,
        typer.Option(
            "--subject",
            help="BIDS subject label. Accepts 001 or sub-001. Can be repeated.",
            show_default=False,
        ),
    ] = None,
    session: Annotated[
        list[str] | None,
        typer.Option(
            "--session",
            help="BIDS session label. Accepts 01 or ses-01. Can be repeated.",
            show_default=False,
        ),
    ] = None,
    task: Annotated[
        list[str] | None,
        typer.Option(
            "--task",
            help="BIDS task label. Accepts MMN or task-MMN. Can be repeated.",
            show_default=False,
        ),
    ] = None,
    run: Annotated[
        list[str] | None,
        typer.Option(
            "--run",
            help="BIDS run label. Accepts 01 or run-01. Can be repeated.",
            show_default=False,
        ),
    ] = None,
    acquisition: Annotated[
        list[str] | None,
        typer.Option(
            "--acquisition",
            "--acq",
            help="BIDS acquisition label. Accepts high or acq-high. Can be repeated.",
            show_default=False,
        ),
    ] = None,
    datatype: Annotated[
        list[str] | None,
        typer.Option(
            "--datatype",
            help="BIDS datatype directory, for example eeg or anat. Can be repeated.",
            show_default=False,
        ),
    ] = None,
    suffix: Annotated[
        list[str] | None,
        typer.Option(
            "--suffix",
            help="BIDS suffix, for example eeg, events, or T1w. Can be repeated.",
            show_default=False,
        ),
    ] = None,
    extension: Annotated[
        list[str] | None,
        typer.Option(
            "--extension",
            "--ext",
            help="File extension, for example .set, .fdt, or .tsv. Can be repeated.",
            show_default=False,
        ),
    ] = None,
    scope: Annotated[
        list[str] | None,
        typer.Option(
            "--scope",
            help=(
                "Dataset scope: raw, derivatives, stimuli, sourcedata, or code. "
                "Can be repeated."
            ),
            show_default=False,
        ),
    ] = None,
    pipeline: Annotated[
        list[str] | None,
        typer.Option(
            "--pipeline",
            help="Derivative pipeline under derivatives/<pipeline>/. Can be repeated.",
            show_default=False,
        ),
    ] = None,
    entity: Annotated[
        list[str] | None,
        typer.Option(
            "--entity",
            help='Generic BIDS entity filter as "key=value". Can be repeated.',
            show_default=False,
        ),
    ] = None,
    downloader: Annotated[
        str,
        typer.Option(
            "--downloader",
            help=(
                "Transfer backend: auto, python, datalad, or s3. "
                "auto composes S3 → DataLad (when advertised) → HTTPS. "
                "python opts out of every layer and uses HTTPS only. "
                "s3 fetches from the public NEMAR S3 bucket with no fallback. "
                "datalad layers DataLad over HTTPS when a datalad_url is "
                "advertised."
            ),
        ),
    ] = "auto",
    verify_hash: Annotated[
        bool,
        typer.Option(help="Verify checksums when the NEMAR manifest provides them."),
    ] = True,
    verify_size: Annotated[
        bool,
        typer.Option(help="Verify downloaded file sizes when known."),
    ] = True,
    max_retries: Annotated[
        int,
        typer.Option(help="Retry count for metadata and file downloads."),
    ] = 5,
    max_concurrent_downloads: Annotated[
        int,
        typer.Option(
            "--jobs",
            "-j",
            "--max-concurrent-downloads",
            min=1,
            help="Maximum parallel downloads.",
        ),
    ] = 16,
    metadata_timeout: Annotated[
        float,
        typer.Option(
            "--metadata-timeout",
            min=0.0,
            help=(
                "Timeout (seconds) for the small metadata fetches "
                "(index, version, manifest)."
            ),
        ),
    ] = 30.0,
    stimuli: Annotated[
        bool,
        typer.Option(
            "--stimuli",
            help=(
                "Include the ``stimuli`` scope alongside the default "
                "``raw`` scope."
            ),
        ),
    ] = False,
    derivatives: Annotated[
        bool,
        typer.Option(
            "--derivatives",
            help=(
                "Include the ``derivatives`` scope alongside the default "
                "``raw`` scope."
            ),
        ),
    ] = False,
    sourcedata: Annotated[
        bool,
        typer.Option(
            "--sourcedata",
            help=(
                "Include the ``sourcedata`` scope alongside the default "
                "``raw`` scope. ``sourcedata/`` holds the original "
                "pre-BIDS distribution as published upstream."
            ),
        ),
    ] = False,
    no_data: Annotated[
        bool,
        typer.Option(
            "--no-data",
            help=(
                "Skip annexed binaries. Only fetch git-tracked sidecars "
                "(JSON, TSV, README, etc.); useful for inspection."
            ),
        ),
    ] = False,
    trust_existing: Annotated[
        bool,
        typer.Option(
            "--trust-existing",
            help=(
                "Faster idempotent re-runs: trust files already on disk "
                "with the right size and skip re-hashing them. Freshly "
                "downloaded files are still hash-verified."
            ),
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Print the resolved download parameters before starting.",
        ),
    ] = False,
    data_url: Annotated[
        str,
        typer.Option(
            "--data-url",
            help="NEMAR data origin. Must stay on data.nemar.org by default.",
        ),
    ] = DEFAULT_DATA_URL,
) -> None:
    """Download datasets from the public NEMAR data endpoint."""
    bids_filters = _BidsFilterArgs(
        include=include,
        exclude=exclude,
        subject=subject,
        session=session,
        task=task,
        run=run,
        acquisition=acquisition,
        datatype=datatype,
        suffix=suffix,
        extension=extension,
        scope=_merge_scopes(
            scope, stimuli=stimuli, derivatives=derivatives, sourcedata=sourcedata
        ),
        pipeline=pipeline,
        entity=entity,
    )
    if verbose:
        _print_resolved_params(
            dataset=dataset,
            tag=tag,
            target_dir=target_dir,
            bids_filters=bids_filters,
            downloader=downloader,
            max_concurrent_downloads=max_concurrent_downloads,
            metadata_timeout=metadata_timeout,
            no_data=no_data,
            data_url=data_url,
        )
    try:
        download(
            dataset=dataset,
            tag=tag,
            target_dir=target_dir,
            **bids_filters.to_download_kwargs(),
            downloader=downloader,
            verify_hash=verify_hash,
            verify_size=verify_size,
            max_retries=max_retries,
            max_concurrent_downloads=max_concurrent_downloads,
            metadata_timeout=metadata_timeout,
            no_data=no_data,
            trust_existing=trust_existing,
            data_url=data_url,
        )
    except (RuntimeError, ValueError, FileExistsError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc


@app.command(name="versions")
def versions_cli(
    dataset: Annotated[
        str,
        typer.Argument(
            metavar="DATASET",
            help="The NEMAR dataset identifier, for example nm000132.",
        ),
    ],
    data_url: Annotated[
        str,
        typer.Option(
            "--data-url",
            help="NEMAR data origin. Must stay on data.nemar.org by default.",
        ),
    ] = DEFAULT_DATA_URL,
    max_retries: Annotated[
        int,
        typer.Option(help="Retry count for the dataset index request."),
    ] = 5,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw dataset index as JSON."),
    ] = False,
) -> None:
    """Show versions advertised by the public NEMAR data endpoint."""
    try:
        index = fetch_dataset_index(
            dataset=dataset,
            data_url=data_url,
            max_retries=max_retries,
        )
    except (RuntimeError, ValueError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc

    if json_output:
        typer.echo(index.model_dump_json(indent=2))
        return

    typer.echo(f"{index.dataset_id} latest: {index.latest}")
    typer.echo("version\tcreated_at\tdoi\tmanifest_url")
    for version in index.versions:
        label = version.version
        if version.version == index.latest:
            label += " (latest)"
        typer.echo(
            "\t".join(
                [
                    label,
                    version.created_at or "",
                    version.doi or "",
                    version.manifest_url,
                ]
            )
        )


def show_version_callback(show_version: bool) -> None:
    """Print package version and exit."""
    if show_version:
        typer.echo(f"This is nemar-py {nemar.__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            help="Show the version of nemar-py.",
            callback=show_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Access NEMAR datasets."""


if __name__ == "__main__":
    app()
