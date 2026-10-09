import hashlib
from unittest import mock

import pandas as pd
import pytest
import requests

import helical.utils.downloader as downloader_module
import helical.utils.mapping as mapping_module
from helical.utils.downloader import Downloader

CONTENT = b"gene,id\nA,1\n"
NAME = "dir/file.csv"


def _response(status=200, content=CONTENT):
    response = requests.Response()
    response.status_code = status
    response.headers["ETag"] = f'"{hashlib.md5(content).hexdigest()}"'
    response.headers["Content-Length"] = str(len(content))
    return response


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(downloader_module, "CACHE_DIR_HELICAL", str(tmp_path))
    monkeypatch.setattr(downloader_module, "_VALIDATED_FILES", set())
    return tmp_path


def _downloader(head):
    downloader = Downloader()
    downloader.session = mock.Mock(head=head)
    return downloader


def _write(cache, content=CONTENT):
    path = cache / NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_valid_file_is_checked_once_per_process(cache):
    _write(cache)
    head = mock.Mock(return_value=_response())
    downloader = _downloader(head)

    with mock.patch.object(
        Downloader, "_md5_checksum", wraps=downloader._md5_checksum
    ) as md5:
        downloader.download_via_name(NAME)
        _downloader(head).download_via_name(NAME)  # another instance, same process

    assert head.call_count == 1
    assert md5.call_count == 1


def test_validated_file_deleted_later_is_downloaded_again(cache):
    path = _write(cache)
    downloader = _downloader(mock.Mock(return_value=_response()))
    downloader.download_via_name(NAME)
    path.unlink()

    with mock.patch.object(
        Downloader,
        "download_via_link",
        side_effect=lambda output, link: output.write_bytes(CONTENT),
    ) as download:
        downloader.download_via_name(NAME)

    download.assert_called_once()
    assert path.read_bytes() == CONTENT


@pytest.mark.parametrize(
    "head",
    [
        mock.Mock(side_effect=requests.ConnectionError("offline")),
        # S3 error bodies carry no ETag and their own length.
        mock.Mock(return_value=_response(status=403, content=b"<Error/>")),
    ],
)
def test_unreachable_remote_keeps_cached_file(cache, head):
    path = _write(cache)
    downloader = _downloader(head)

    with mock.patch.object(Downloader, "download_via_link") as download:
        downloader.download_via_name(NAME)

    download.assert_not_called()
    assert path.read_bytes() == CONTENT


def test_mismatched_file_is_replaced(cache):
    path = _write(cache, b"stale")
    downloader = _downloader(mock.Mock(return_value=_response()))

    def fetch(output, link):
        assert not output.exists()  # the stale copy was deleted first
        output.write_bytes(CONTENT)

    with mock.patch.object(
        Downloader, "download_via_link", side_effect=fetch
    ) as download:
        downloader.download_via_name(NAME)

    download.assert_called_once()
    assert path.read_bytes() == CONTENT


def test_missing_file_is_downloaded_even_if_remote_is_unreachable(cache):
    downloader = _downloader(mock.Mock(side_effect=requests.ConnectionError("offline")))

    with mock.patch.object(
        Downloader, "download_via_link", side_effect=RuntimeError("no network")
    ) as download:
        with pytest.raises(RuntimeError, match="no network"):
            downloader.download_via_name(NAME)

    download.assert_called_once()


def test_check_file_valid_is_false_when_remote_is_unreachable(cache):
    path = _write(cache)
    downloader = _downloader(mock.Mock(side_effect=requests.ConnectionError("offline")))
    assert downloader.check_file_valid("https://example.invalid/x", path) is False


def test_static_ensembl_table_is_parsed_once(tmp_path, monkeypatch):
    pd.DataFrame({"ensembl_id": ["ENSG1"], "gene_name": ["A"]}).to_csv(
        tmp_path / "hsapiens_pybiomart.csv"
    )
    monkeypatch.setattr(mapping_module, "CACHE_DIR_HELICAL", str(tmp_path))
    mapping_module._load_static_ensembl_df.cache_clear()
    try:
        with (
            mock.patch.object(mapping_module, "Downloader") as downloader_cls,
            mock.patch.object(
                mapping_module.pd, "read_csv", wraps=pd.read_csv
            ) as read_csv,
        ):
            first = mapping_module.convert_list_gene_symbols_to_ensembl_ids(["A"])
            second = mapping_module.convert_list_ensembl_ids_to_gene_symbols(["ENSG1"])

        assert first == ["ENSG1"] and second == ["A"]
        assert read_csv.call_count == 1
        assert downloader_cls.call_count == 1
    finally:
        mapping_module._load_static_ensembl_df.cache_clear()
