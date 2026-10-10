# pybiomart uses requests-cache: disbale this
import requests_cache
requests_cache.uninstall_cache()

import requests
from helical.utils.logger import Logger
from helical.constants.enums import LoggingType, LoggingLevel
import logging
import sys
from pathlib import Path
from tqdm import tqdm
from helical.constants.paths import CACHE_DIR_HELICAL
import hashlib
from typing import Optional

LOGGER = logging.getLogger(__name__)
INTERVAL = 1000
CHUNK_SIZE = 1024 * 1024 * 10
LOADING_BAR_LENGTH = 50

# Files already checked against S3 in this process. Checking costs a HEAD
# request and, for a single-part upload, an MD5 of the whole file.
_VALIDATED_FILES: set = set()


class Downloader(Logger):
    def __init__(
        self, loging_type=LoggingType.CONSOLE, level=LoggingLevel.INFO
    ) -> None:
        super().__init__(loging_type, level)
        self.display = True
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_maxsize=100, pool_connections=100)
        self.session.mount("https://", adapter)

    def download_via_link(self, output: Path, link: str) -> None:
        if output.is_file():
            LOGGER.debug(
                f"File: '{output}' exists already. File is not overwritten and nothing is downloaded."
            )
            return

        LOGGER.info(f"Starting to download: '{link}'")
        response = self.session.get(link, stream=True)

        if response.status_code != 200:
            message = f"Failed downloading file from '{link}' with status code: {response.status_code}"
            LOGGER.error(message)
            raise Exception(message)

        total_length = response.headers.get("content-length")
        self.data_length = 0
        self.total_length = int(total_length) if total_length else None

        output.parent.mkdir(parents=True, exist_ok=True)

        if not self.total_length:
            # If total length unknown, just write content directly without progress bar
            with open(output, "wb") as f:
                f.write(response.content)
        else:
            with (
                open(output, "wb") as f,
                tqdm(
                    total=self.total_length, unit="B", unit_scale=True, desc=output.name
                ) as pbar,
            ):
                for data in response.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(data)
                    pbar.update(len(data))

    def download_via_name(self, name: str) -> None:
        bucket_name = "helicalpackage"
        region = "eu-west-2"
        s3_key = name
        output = Path(CACHE_DIR_HELICAL) / name
        url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{s3_key}"

        if output in _VALIDATED_FILES and output.is_file():
            return

        output.parent.mkdir(parents=True, exist_ok=True)

        valid = self._validate(url, output)
        if valid is None and output.is_file():
            # S3 could not be asked. The cached copy may well be fine, and
            # deleting it would leave nothing to fall back on.
            LOGGER.warning(
                f"Could not check '{output}' against the remote copy. Using the cached file."
            )
        elif not valid:
            if output.is_file():
                LOGGER.warning(
                    f"File '{output}' is corrupted or invalid. Deleting and re-downloading."
                )
                output.unlink()  # delete corrupted file

            LOGGER.info(f"Downloading '{name}'")
            self.download_via_link(output, url)

            # Validate again after download
            if not self.check_file_valid(url, output):
                raise ValueError(f"File '{output}' failed validation after download.")
            else:
                LOGGER.info(f"File saved to: '{output}'")
        else:
            LOGGER.debug(f"File '{output}' already exists and is valid.")
        _VALIDATED_FILES.add(output)

    def check_file_valid(self, url: str, file_path: Path) -> bool:
        """
        Validates a local file against the remote object's ETag or Content-Length.
        Returns True if file exists and is considered valid.
        """
        return self._validate(url, file_path) is True

    def _validate(self, url: str, file_path: Path) -> Optional[bool]:
        """Like ``check_file_valid``, but returns None when the remote object
        could not be checked: the request failed or returned an error status.
        """
        if not file_path.is_file():
            return False
        try:
            head = self.session.head(url)
            head.raise_for_status()
        except Exception as e:
            LOGGER.warning(f"Could not reach '{url}' to validate {file_path}: {e}")
            return None

        try:
            etag = head.headers.get("ETag", "").strip('"')
            remote_size = int(head.headers.get("Content-Length", 0))
            local_size = file_path.stat().st_size

            if local_size != remote_size:
                LOGGER.warning(
                    f"File size mismatch: local {local_size}, remote {remote_size}"
                )
                return False

            # Handle single-part ETag (true MD5)
            if "-" not in etag:
                local_md5 = self._md5_checksum(file_path)
                if etag != local_md5:
                    LOGGER.warning(
                        f"MD5 checksum mismatch: local {local_md5}, remote {etag}"
                    )
                    return False

            # For multi-part uploads (ETag has "-"), assume size check is sufficient
            return True

        except Exception as e:
            LOGGER.warning(f"Validation failed for {file_path}: {e}")
            return False

    def _md5_checksum(self, filename: Path, chunk_size: int = 1024 * 1024) -> str:
        m = hashlib.md5()
        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                m.update(chunk)
        return m.hexdigest()
