import requests
from helical.utils.logger import Logger
from helical.constants.enums import LoggingType, LoggingLevel
import logging
import os
import sys
from pathlib import Path
from tqdm import tqdm
from azure.storage.blob import BlobClient
from helical.constants.paths import CACHE_DIR_HELICAL
import hashlib
import boto3
from botocore import UNSIGNED
from botocore.config import Config

LOGGER = logging.getLogger(__name__)
INTERVAL = 1000  # interval to get gene mappings
CHUNK_SIZE = 1024 * 1024 * 10  # 8192 # size of individual chunks to download
LOADING_BAR_LENGTH = 50  # size of the download progression bar in console


class Downloader(Logger):
    def __init__(
        self, loging_type=LoggingType.CONSOLE, level=LoggingLevel.INFO
    ) -> None:
        super().__init__(loging_type, level)
        self.display = True

        # manually create a requests session
        self.session = requests.Session()
        # set an adapter with the required pool size
        adapter = requests.adapters.HTTPAdapter(pool_maxsize=100, pool_connections=100)
        # mount the adapter to the session
        self.session.mount("https://", adapter)

    def download_via_link(self, output: Path, link: str) -> None:
        """
        Download a file via a link.

        Args:
            output: Path to the output file.
            link: URL to download the file from.

        Raises:
            Exception: If the download fails.
        """

        if output.is_file():
            LOGGER.debug(
                f"File: '{output}' exists already. File is not overwritten and nothing is downloaded."
            )

        else:
            LOGGER.info(f"Starting to download: '{link}'")
            response = requests.get(link, stream=True)

            if response.status_code != 200:
                message = f"Failed downloading file from '{link}' with status code: {response.status_code}"
                LOGGER.error(message)
                raise Exception(message)

            total_length = response.headers.get("content-length")

            # Resetting for visualization
            self.data_length = 0
            self.total_length = int(total_length)

            with open(output, "wb") as f:
                if total_length is None:  # no content length header
                    f.write(response.content)
                else:
                    try:
                        for data in response.iter_content(chunk_size=CHUNK_SIZE):
                            if self.display:
                                self._display_download_progress(len(data))
                            f.write(data)
                    except:
                        LOGGER.error(f"Failed downloading file from '{link}'")
            LOGGER.info(f"File saved to: '{output}'")

    def _display_download_progress(self, data_chunk_size: int) -> None:
        """
        Display the download progress in console.

        Args:
            data_chunk_size: Integer of size of the newly downloaded data chunk.
        """
        self.data_length += data_chunk_size
        done = int(LOADING_BAR_LENGTH * self.data_length / self.total_length)
        sys.stdout.write("\r[%s%s]" % ("=" * done, " " * (LOADING_BAR_LENGTH - done)))
        sys.stdout.flush()

    def s3_download_with_progress(self, s3client, bucket, key, output):
        with open(output, "wb") as f:
            s3client.download_fileobj(
                Bucket=bucket,
                Key=key,
                Fileobj=f,
                Callback=S3ProgressPercentage(bucket, key, s3client),
            )

    def download_via_name(self, name: str) -> None:
        """
        Download a file via a link.

        Args:
            name (str): The name of the file to be downloaded.

        Returns:
            None
        """
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        # Example usage
        bucket_name = "helicalpackage"
        s3_key = name
        output = os.path.join(CACHE_DIR_HELICAL, name)

        if not os.path.exists(os.path.dirname(output)):
            os.makedirs(os.path.dirname(output), exist_ok=True)
            LOGGER.info(f"Creating Folder {os.path.dirname(output)}")

        if not Path(output).is_file() or not self.etag_compare(s3_key, output, s3):
            LOGGER.info(
                f"File does not exist or has incorrect hash. Starting to download: '{s3_key}'"
            )
            # temporarily disable INFO logging from Azure package
            original_level = logging.getLogger().getEffectiveLevel()
            logging.getLogger().setLevel(logging.WARNING)
            self.s3_download_with_progress(s3, bucket_name, s3_key, output)
            # restore original logging level
            logging.getLogger().setLevel(original_level)
            assert self.etag_compare(
                s3_key, output, s3
            ), f"Hash of downloaded file '{output}' does not match the expected hash."
            LOGGER.info(f"File saved to: '{output}'")
        else:
            LOGGER.debug(
                f"File: '{output}' exists already. File is not overwritten and nothing is downloaded."
            )

    def etag_compare(self, s3_key, filename, s3client, bucket_name="helicalpackage"):

        # obj = s3client.get_object(Bucket=bucket_name, Key=filename)
        # Get the object's metadata
        response = s3client.head_object(Bucket=bucket_name, Key=s3_key)

        # Extract the ETag
        etag = response["ETag"].strip('"')  # Remove quotes if needed
        # etag = obj.e_tag

        def md5_checksum(filename):
            m = hashlib.md5()
            with open(filename, "rb") as f:
                for data in iter(lambda: f.read(1024 * 1024), b""):
                    m.update(data)
            return m.hexdigest()

        def etag_checksum(filename, chunk_size=8 * 1024 * 1024):
            md5s = []
            with open(filename, "rb") as f:
                for data in iter(lambda: f.read(chunk_size), b""):
                    md5s.append(hashlib.md5(data).digest())
            m = hashlib.md5(b"".join(md5s))
            return "{}-{}".format(m.hexdigest(), len(md5s))

        et = etag  # [1:-1]  # strip quotes
        if "-" in et and et == etag_checksum(filename):
            return True
        if "-" not in et and et == md5_checksum(filename):
            return True
        return False


class S3ProgressPercentage:
    def __init__(self, bucket, key, s3_client):
        self._key = key
        self._size = s3_client.head_object(Bucket=bucket, Key=key)["ContentLength"]
        self._seen_so_far = 0
        self._tqdm = tqdm(total=self._size, unit="B", unit_scale=True, desc=key)

    def __call__(self, bytes_amount):
        self._seen_so_far += bytes_amount
        self._tqdm.update(bytes_amount)


if __name__ == "__main__":
    # Example usage
    downloader = Downloader()
    downloader.download_via_name("10k_pbmcs_proc.h5ad")
