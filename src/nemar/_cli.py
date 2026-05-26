"""Command line interface for NEMAR downloads."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer

import nemar
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


@app.command(name="download")
def download_cli(
    dataset: Annotated[
        str,
        typer.Option(
            "--dataset",
            "-d",
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
            "--target-dir",
            help="The directory to download to.",
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
                "Transfer backend: auto, aria2, python, or datalad. "
                "auto and datalad layer DataLad over HTTPS when the dataset "
                "index advertises a datalad_url; aria2 and python opt out of "
                "the DataLad layer."
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
        typer.Option(min=1, help="Maximum parallel downloads."),
    ] = 16,
    data_url: Annotated[
        str,
        typer.Option(
            "--data-url",
            help="NEMAR data origin. Must stay on data.nemar.org by default.",
        ),
    ] = "https://data.nemar.org/",
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
        scope=scope,
        pipeline=pipeline,
        entity=entity,
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
            data_url=data_url,
        )
    except (RuntimeError, ValueError, FileExistsError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc


@app.command(name="versions")
def versions_cli(
    dataset: Annotated[
        str,
        typer.Option(
            "--dataset",
            "-d",
            help="The NEMAR dataset identifier, for example nm000132.",
        ),
    ],
    data_url: Annotated[
        str,
        typer.Option(
            "--data-url",
            help="NEMAR data origin. Must stay on data.nemar.org by default.",
        ),
    ] = "https://data.nemar.org/",
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
