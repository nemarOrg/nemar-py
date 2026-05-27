"""Tests for ``DownloadRequest`` and ``TransferOptions`` value types.

The orchestrator's 25-kwarg shuffle moved into a frozen value type. These
tests fix the normalization contract -- target path resolution, data URL
normalization, version-tag prefixing, include/exclude defaulting,
BIDS-query construction, and the retry/verify/transfer policy defaults --
so the refactor of ``download()`` is anchored to behavior rather than
implementation.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from nemar._bids import BidsQuery
from nemar._endpoint import DataEndpoint
from nemar._request import DownloadRequest, TransferOptions
from nemar._retry import RetryPolicy
from nemar._verification import VerifyPolicy


def _minimum_kwargs(**overrides: object) -> dict[str, object]:
    """Return the smallest kwargs dict ``download()`` accepts.

    Mirrors the public signature; tests override one field at a time.
    """
    base: dict[str, object] = {
        "dataset": "nm000132",
        "tag": None,
        "target_dir": None,
        "include": None,
        "exclude": None,
        "subject": None,
        "session": None,
        "task": None,
        "run": None,
        "acquisition": None,
        "datatype": None,
        "suffix": None,
        "extension": None,
        "scope": None,
        "pipeline": None,
        "entity": None,
        "downloader": "auto",
        "verify_hash": True,
        "verify_size": True,
        "max_retries": 5,
        "max_concurrent_downloads": 16,
        "metadata_timeout": 30.0,
        "stream_timeout": 60.0,
        "data_url": "https://data.nemar.org/",
    }
    base.update(overrides)
    return base


class TestFromKwargsBuildsValidRequest:
    def test_minimum_inputs_succeeds(self) -> None:
        req = DownloadRequest.from_kwargs(**_minimum_kwargs())
        assert req.dataset == "nm000132"
        assert req.requested_tag is None
        assert isinstance(req.endpoint, DataEndpoint)
        assert isinstance(req.bids_query, BidsQuery)
        assert isinstance(req.transfer, TransferOptions)
        assert isinstance(req.retry, RetryPolicy)
        assert isinstance(req.verify, VerifyPolicy)

    def test_request_is_frozen(self) -> None:
        req = DownloadRequest.from_kwargs(**_minimum_kwargs())
        with pytest.raises(dataclasses.FrozenInstanceError):
            req.dataset = "nm999999"  # type: ignore[misc]

    def test_transfer_options_is_frozen(self) -> None:
        opts = TransferOptions(
            backend="auto",
            max_concurrent_downloads=16,
            stream_timeout=60.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            opts.backend = "python"  # type: ignore[misc]


class TestNormalization:
    def test_target_dir_defaults_to_dataset_name(self, tmp_path: Path) -> None:
        # When target_dir is None, target_path is dataset id, then resolved.
        req = DownloadRequest.from_kwargs(**_minimum_kwargs(target_dir=None))
        assert req.target_path.name == "nm000132"
        assert req.target_path.is_absolute()

    def test_target_dir_user_home_expanded(self) -> None:
        req = DownloadRequest.from_kwargs(**_minimum_kwargs(target_dir="~/nm-data"))
        # Resolved + expanded. The leading ~ must be gone.
        assert "~" not in str(req.target_path)
        assert req.target_path.is_absolute()

    def test_target_dir_relative_resolved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        req = DownloadRequest.from_kwargs(**_minimum_kwargs(target_dir="subdir"))
        assert req.target_path == (tmp_path / "subdir").resolve()

    def test_data_url_trailing_slash_normalized(self) -> None:
        req = DownloadRequest.from_kwargs(
            **_minimum_kwargs(data_url="https://data.nemar.org")
        )
        assert req.endpoint.url == "https://data.nemar.org/"

    def test_tag_with_digit_prefix_gets_v(self) -> None:
        req = DownloadRequest.from_kwargs(**_minimum_kwargs(tag="1.0.0"))
        assert req.requested_tag == "v1.0.0"

    def test_tag_with_v_prefix_kept(self) -> None:
        req = DownloadRequest.from_kwargs(**_minimum_kwargs(tag="v2.1.0"))
        assert req.requested_tag == "v2.1.0"

    def test_tag_latest_kept(self) -> None:
        req = DownloadRequest.from_kwargs(**_minimum_kwargs(tag="latest"))
        assert req.requested_tag == "latest"

    def test_include_string_becomes_tuple(self) -> None:
        req = DownloadRequest.from_kwargs(**_minimum_kwargs(include="*.tsv"))
        assert req.include_patterns == ("*.tsv",)

    def test_include_none_becomes_empty(self) -> None:
        req = DownloadRequest.from_kwargs(**_minimum_kwargs(include=None))
        assert req.include_patterns == ()

    def test_include_list_preserved(self) -> None:
        req = DownloadRequest.from_kwargs(
            **_minimum_kwargs(include=["*.tsv", "*.json"])
        )
        assert req.include_patterns == ("*.tsv", "*.json")

    def test_exclude_list_preserved(self) -> None:
        req = DownloadRequest.from_kwargs(
            **_minimum_kwargs(exclude=["sourcedata/*"])
        )
        assert req.exclude_patterns == ("sourcedata/*",)

    def test_bids_query_built_from_filters(self) -> None:
        req = DownloadRequest.from_kwargs(
            **_minimum_kwargs(subject="001", task="rest")
        )
        # Subject normalization strips the optional ``sub-`` prefix.
        assert req.bids_query.entities.get("sub") == ("001",)
        assert req.bids_query.entities.get("task") == ("rest",)


class TestInvalidInputsRaise:
    def test_bad_dataset_pattern_raises(self) -> None:
        with pytest.raises(ValueError, match="dataset must look like"):
            DownloadRequest.from_kwargs(**_minimum_kwargs(dataset="not-a-dataset"))

    def test_on_prefixed_dataset_id_accepted(self) -> None:
        """``on*`` is the second valid NEMAR dataset ID prefix.

        The real ``data.nemar.org`` catalog mixes ``nm*`` and ``on*``
        identifiers (the latter being datasets imported from OpenNeuro
        without renumbering). Both must round-trip through input
        validation.
        """
        req = DownloadRequest.from_kwargs(**_minimum_kwargs(dataset="on005505"))
        assert req.dataset == "on005505"

    def test_nm_prefixed_dataset_id_still_accepted(self) -> None:
        """Backward compatibility: the original ``nm*`` form keeps working."""
        req = DownloadRequest.from_kwargs(**_minimum_kwargs(dataset="nm000132"))
        assert req.dataset == "nm000132"

    def test_unknown_prefix_still_rejected(self) -> None:
        """Only ``nm*`` and ``on*`` are advertised by NEMAR; other prefixes raise."""
        with pytest.raises(ValueError, match="dataset must look like"):
            DownloadRequest.from_kwargs(**_minimum_kwargs(dataset="zz000132"))

    def test_negative_retries_raises(self) -> None:
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            DownloadRequest.from_kwargs(**_minimum_kwargs(max_retries=-1))

    def test_zero_concurrency_raises(self) -> None:
        with pytest.raises(
            ValueError, match="max_concurrent_downloads must be at least 1"
        ):
            DownloadRequest.from_kwargs(**_minimum_kwargs(max_concurrent_downloads=0))

    def test_non_https_data_url_raises(self) -> None:
        with pytest.raises(ValueError, match="data_url must use HTTPS"):
            DownloadRequest.from_kwargs(
                **_minimum_kwargs(data_url="http://data.nemar.org/")
            )

    def test_bad_downloader_raises(self) -> None:
        with pytest.raises(ValueError, match="downloader must be one of"):
            DownloadRequest.from_kwargs(**_minimum_kwargs(downloader="curl"))

    def test_datalad_downloader_is_accepted(self) -> None:
        """``datalad`` is the third valid backend choice.

        The actual DataLad work is gated on whether the dataset index
        advertises a ``datalad_url``; the validation here only checks
        that the policy string is recognized.
        """
        req = DownloadRequest.from_kwargs(**_minimum_kwargs(downloader="datalad"))
        assert req.transfer.backend == "datalad"

    def test_empty_tag_raises(self) -> None:
        with pytest.raises(ValueError, match="tag must not be empty"):
            DownloadRequest.from_kwargs(**_minimum_kwargs(tag="  "))


class TestTransferOptionsDefaults:
    def test_default_transfer_options(self) -> None:
        req = DownloadRequest.from_kwargs(**_minimum_kwargs())
        assert req.transfer.backend == "auto"
        assert req.transfer.max_concurrent_downloads == 16
        assert req.transfer.stream_timeout == 60.0

    def test_transfer_options_override(self) -> None:
        req = DownloadRequest.from_kwargs(
            **_minimum_kwargs(
                downloader="python",
                max_concurrent_downloads=4,
                stream_timeout=15.0,
            )
        )
        assert req.transfer.backend == "python"
        assert req.transfer.max_concurrent_downloads == 4
        assert req.transfer.stream_timeout == 15.0


class TestVerifyAndRetryPolicies:
    def test_verify_policy_reflects_kwargs(self) -> None:
        req = DownloadRequest.from_kwargs(
            **_minimum_kwargs(verify_hash=False, verify_size=False)
        )
        assert req.verify.verify_hash is False
        assert req.verify.verify_size is False

    def test_verify_policy_defaults(self) -> None:
        req = DownloadRequest.from_kwargs(**_minimum_kwargs())
        assert req.verify.verify_hash is True
        assert req.verify.verify_size is True

    def test_retry_policy_attempts_match_max_retries(self) -> None:
        req = DownloadRequest.from_kwargs(**_minimum_kwargs(max_retries=2))
        # max_attempts is max_retries + 1.
        assert req.retry.max_attempts == 3
