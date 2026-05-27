"""Tests for NEMAR metadata and manifest models."""

import pytest

from nemar._endpoint import DataEndpoint
from nemar._models import VersionManifest, parse_dataset_index

MANIFEST_URL = "https://data.nemar.org/nm000132/v1.0.0/manifest.json"
DATA_URL = "https://data.nemar.org/"
ENDPOINT = DataEndpoint.from_url(DATA_URL)


def make_dataset_index():
    """Create a representative public data endpoint index."""
    return parse_dataset_index(
        {
            "dataset_id": "nm000132",
            "latest": "v1.1.1",
            "metadata_url": "/nm000132/metadata.json",
            "versions": [
                {
                    "version": "v1.1.1",
                    "doi": "10.82901/nemar.nm000132.v1.1.1",
                    "created_at": "2026-04-04 06:05:15",
                    "manifest_url": "/nm000132/v1.1.1/manifest.json",
                    "browse_url": "/nm000132/v1.1.1/",
                },
                {
                    "version": "v1.0.0",
                    "doi": "10.82901/nemar.nm000132.v1.0.0",
                    "created_at": "2026-03-14 12:20:43",
                    "manifest_url": "/nm000132/v1.0.0/manifest.json",
                },
            ],
        }
    )


def parse_manifest(payload):
    """Parse a manifest payload using the canonical NEMAR endpoint origin."""
    manifest = VersionManifest.parse(
        payload,
        manifest_url=MANIFEST_URL,
        endpoint=ENDPOINT,
    )
    return list(manifest.files)


def test_parse_dataset_index() -> None:
    """Parse the public data endpoint index shape."""
    index = make_dataset_index()

    assert index.dataset_id == "nm000132"
    assert index.latest == "v1.1.1"
    assert index.resolve_version("v1.1.1").manifest_url.endswith("manifest.json")


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        pytest.param(None, "v1.1.1", id="none-selects-latest"),
        pytest.param("latest", "v1.1.1", id="latest-alias"),
        pytest.param("v1.0.0", "v1.0.0", id="explicit-version"),
    ],
)
def test_dataset_index_resolves_version_aliases(tag, expected) -> None:
    """Version control is resolved from versions advertised by data.nemar.org."""
    assert make_dataset_index().resolve_version(tag).version == expected


@pytest.mark.parametrize(
    "tag",
    [
        pytest.param("v9.9.9", id="unknown-version"),
        pytest.param("", id="empty-version"),
    ],
)
def test_dataset_index_rejects_unknown_versions(tag) -> None:
    """Unknown version tags fail with available endpoint versions."""
    with pytest.raises(RuntimeError, match="Available versions: v1.1.1, v1.0.0"):
        make_dataset_index().resolve_version(tag)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({}, id="missing-required-fields"),
        pytest.param(
            {"dataset_id": "nm000132", "latest": "v1.0.0", "versions": [{}]},
            id="invalid-version-entry",
        ),
    ],
)
def test_parse_dataset_index_rejects_invalid_payloads(payload) -> None:
    """Unexpected endpoint index payloads are normalized to RuntimeError."""
    with pytest.raises(RuntimeError, match="unexpected dataset index"):
        parse_dataset_index(payload)


def test_parse_dataset_index_accepts_datalad_url() -> None:
    """The optional ``datalad_url`` field travels through unchanged.

    The DataLad transfer layer reads this field after index resolution to
    decide whether a git-annex clone is reachable. Missing → None →
    DataLad path is skipped and the HTTPS backend handles the dataset
    on its own.
    """
    index = parse_dataset_index(
        {
            "dataset_id": "nm000132",
            "latest": "v1.0.0",
            "datalad_url": "https://github.com/OpenNeuroDatasets/ds000132.git",
            "versions": [
                {
                    "version": "v1.0.0",
                    "manifest_url": "/nm000132/v1.0.0/manifest.json",
                },
            ],
        }
    )
    assert index.datalad_url == "https://github.com/OpenNeuroDatasets/ds000132.git"


def test_parse_dataset_index_datalad_url_defaults_to_none() -> None:
    """Existing NEMAR index payloads without ``datalad_url`` resolve to None."""
    index = make_dataset_index()
    assert index.datalad_url is None


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        pytest.param(
            [
                {
                    "path": "dataset_description.json",
                    "url": "dataset_description.json",
                    "size": "12",
                    "sha256": '"abc"',
                }
            ],
            {
                "path": "dataset_description.json",
                "url": (
                    "https://data.nemar.org/nm000132/v1.0.0/dataset_description.json"
                ),
                "size": 12,
                "sha256": "abc",
                "md5": None,
            },
            id="list-relative-url-size-string",
        ),
        pytest.param(
            {
                "files": [
                    {
                        "filename": "participants.tsv",
                        "download_url": "/nm000132/v1.0.0/participants.tsv",
                        "bytes": 5,
                        "md5sum": "def",
                    }
                ]
            },
            {
                "path": "participants.tsv",
                "url": "https://data.nemar.org/nm000132/v1.0.0/participants.tsv",
                "size": 5,
                "sha256": None,
                "md5": "def",
            },
            id="files-list-field-aliases",
        ),
        pytest.param(
            {
                "entries": {
                    "sub-001/eeg/sub-001_task-rest_eeg.set": {
                        "href": "/nm000132/v1.0.0/sub-001/eeg/rest.set",
                        "size_bytes": 99,
                        "etag": '"etag-md5"',
                    }
                }
            },
            {
                "path": "sub-001/eeg/sub-001_task-rest_eeg.set",
                "url": "https://data.nemar.org/nm000132/v1.0.0/sub-001/eeg/rest.set",
                "size": 99,
                "sha256": None,
                "md5": "etag-md5",
            },
            id="entries-mapping",
        ),
        pytest.param(
            {"manifest": {"participants.json": "/nm000132/v1.0.0/participants.json"}},
            {
                "path": "participants.json",
                "url": "https://data.nemar.org/nm000132/v1.0.0/participants.json",
                "size": None,
                "sha256": None,
                "md5": None,
            },
            id="manifest-mapping-string-url",
        ),
        pytest.param(
            {"objects": ["code/make_manifest.py"]},
            {
                "path": "code/make_manifest.py",
                "url": "https://data.nemar.org/nm000132/v1.0.0/code/make_manifest.py",
                "size": None,
                "sha256": None,
                "md5": None,
            },
            id="objects-list-string-entry",
        ),
        pytest.param(
            {"README": {}},
            {
                "path": "README",
                "url": "https://data.nemar.org/nm000132/v1.0.0/README",
                "size": None,
                "sha256": None,
                "md5": None,
            },
            id="root-path-mapping",
        ),
        pytest.param(
            [
                {
                    "relativePath": "stimuli/task-MMN/tone.wav",
                    "urls": ["/nm000132/v1.0.0/stimuli/task-MMN/tone.wav"],
                    "size": 3,
                }
            ],
            {
                "path": "stimuli/task-MMN/tone.wav",
                "url": "https://data.nemar.org/nm000132/v1.0.0/stimuli/task-MMN/tone.wav",
                "size": 3,
                "sha256": None,
                "md5": None,
            },
            id="relative-path-and-urls-list",
        ),
    ],
)
def test_parse_version_manifest_accepts_supported_shapes(payload, expected) -> None:
    """Manifest parser accepts the endpoint shapes needed by robust downloads."""
    file = parse_manifest(payload)[0]

    assert file.path == expected["path"]
    assert file.url == expected["url"]
    assert file.size == expected["size"]
    assert file.sha256 == expected["sha256"]
    assert file.md5 == expected["md5"]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        pytest.param([], "did not contain any downloadable files", id="empty-list"),
        pytest.param({}, "JSON shape is not recognized", id="empty-object"),
        pytest.param(123, "JSON object or array", id="scalar-payload"),
        pytest.param([123], "Unsupported manifest entry type", id="scalar-entry"),
        pytest.param(
            [{1: {"path": "x"}}],
            "missing a non-empty relative path",
            id="entry-without-path-alias",
        ),
        pytest.param(
            [{"path": "../dataset_description.json"}],
            "Unsafe manifest path",
            id="parent-directory",
        ),
        pytest.param([{"path": "/absolute"}], "Unsafe manifest path", id="absolute"),
        pytest.param([{"path": "bad\x00name"}], "NUL byte", id="nul-byte"),
        pytest.param(
            [{"path": "dataset_description.json", "url": 123}],
            "not a non-empty string",
            id="non-string-url",
        ),
        pytest.param(
            [{"path": "participants.tsv", "size": "abc"}],
            "file size is not an integer",
            id="invalid-size",
        ),
        pytest.param(
            [{"path": "participants.tsv", "sha256": 123}],
            "checksum is not a string",
            id="invalid-checksum",
        ),
        pytest.param(
            [{"path": "participants.tsv"}, {"path": "participants.tsv"}],
            "duplicate paths",
            id="duplicate-path",
        ),
        pytest.param({1: {}}, "JSON shape is not recognized", id="non-string-key"),
    ],
)
def test_parse_version_manifest_rejects_invalid_shapes(payload, message) -> None:
    """Manifest parser rejects unsafe, ambiguous, or malformed endpoint payloads."""
    with pytest.raises(RuntimeError, match=message):
        parse_manifest(payload)


def test_manifest_parser_dispatches_checksum_algorithm_sha256() -> None:
    """Production manifest schema: ``checksum_algorithm: "sha256"`` lands
    on :attr:`DatasetFile.sha256`.

    Pins the parser's handling of the real ``data.nemar.org`` shape
    where each entry carries one algorithm tag plus an opaque
    ``checksum`` hex string. Confirmed against the live endpoint
    response for large content-addressed binaries served from S3.
    """
    files = parse_manifest(
        [
            {
                "path": "data/big.set",
                "size": 12345,
                "url": "https://nemar.s3.us-east-2.amazonaws.com/abc/big.set",
                "checksum_algorithm": "sha256",
                "checksum": "a" * 64,
            }
        ]
    )
    assert len(files) == 1
    assert files[0].sha256 == "a" * 64
    assert files[0].md5 is None
    assert files[0].git_sha1 is None


def test_manifest_parser_dispatches_checksum_algorithm_git() -> None:
    """Production manifest schema: ``checksum_algorithm: "git"`` lands
    on :attr:`DatasetFile.git_sha1`.

    Confirmed against the live endpoint response for git-tracked small
    files served from ``raw.githubusercontent.com``. The git blob hash
    is content-addressed exactly the way git stores objects
    (``sha1(b"blob <size>\\0" + content)``), so the verifier picks the
    matching helper.
    """
    files = parse_manifest(
        [
            {
                "path": ".gitattributes",
                "size": 427,
                "url": (
                    "https://raw.githubusercontent.com/"
                    "nemarDatasets/nm000132/v1.0.0/.gitattributes"
                ),
                "checksum_algorithm": "git",
                "checksum": "a8599a8e5fdb959624a4c8684c1aa55fb69f5522",
            }
        ]
    )
    assert len(files) == 1
    assert files[0].git_sha1 == "a8599a8e5fdb959624a4c8684c1aa55fb69f5522"
    assert files[0].sha256 is None
    assert files[0].md5 is None


def test_manifest_parser_rejects_unknown_checksum_algorithm() -> None:
    """An unrecognized algorithm tag is a manifest-shape error.

    Defends against silent misverification if NEMAR ever introduces a
    new algorithm (e.g. ``"sha1"``, ``"crc32"``) the verifier does not
    understand.
    """
    with pytest.raises(RuntimeError, match="Unsupported checksum_algorithm"):
        parse_manifest(
            [
                {
                    "path": "x",
                    "checksum_algorithm": "blake2",
                    "checksum": "deadbeef",
                }
            ]
        )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param('"abc"', "abc", id="strong-etag-quotes"),
        pytest.param('W/"abc"', "abc", id="weak-etag-prefix"),
        pytest.param('"W/abc"', "abc", id="quoted-weak-prefix"),
        pytest.param("W/abc", "abc", id="unquoted-weak-prefix"),
        pytest.param("  abc  ", "abc", id="whitespace-trimmed"),
        pytest.param("abc", "abc", id="plain-hex-unchanged"),
        pytest.param("ABCDEF", "ABCDEF", id="uppercase-preserved-here"),
    ],
)
def test_coerce_hash_strips_etag_weak_prefix(raw, expected) -> None:
    """ETag-derived hashes drop W/ and surrounding quotes (S2).

    Uppercase hex is left as-is; the verifier compares case-insensitively
    so callers do not need to worry about which case the manifest used.
    """
    payload = [
        {
            "path": "participants.tsv",
            "url": "participants.tsv",
            "sha256": raw,
        }
    ]
    assert parse_manifest(payload)[0].sha256 == expected
