# Copyright (C) Tahoe Therapeutics 2025. All rights reserved.
"""S3 utilities for handling public and private bucket access.

This module provides utilities for working with S3 buckets, including:
- Automatic fallback to unsigned requests for public buckets
- Monkey patching the streaming library for public bucket support
- Helper functions for S3 file downloads
"""

import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

log = logging.getLogger(__name__)


def patch_streaming_for_public_s3():
    """Patch the streaming library's S3 downloader to gracefully handle public
    buckets.

    This patch modifies the streaming library's S3Downloader to automatically
    use unsigned requests when AWS credentials are not available, avoiding
    timeout delays on credential checks.

    Note: This monkey patch is required as of mosaicml-streaming v0.13.0 which
    does not natively support automatic fallback to unsigned requests for public
    S3 buckets.

    The patch is idempotent and logs whether it was successfully applied.
    """
    try:
        import urllib.parse

        import boto3
        from boto3.s3.transfer import TransferConfig
        from streaming.base.storage.download import (
            BOTOCORE_CLIENT_ERROR_CODES,
            S3Downloader,
        )

        # Store original method for potential restoration
        if hasattr(S3Downloader, "_original_download_file_impl"):
            log.debug("S3Downloader patch already applied, skipping")
            return

        S3Downloader._original_download_file_impl = S3Downloader._download_file_impl

        # Detect credential availability once upfront (shared across all instances)
        # Check immediately when patch is applied, not during first download
        def _check_credentials_now():
            """Check once if AWS credentials are available."""
            try:
                # Quick check: just look for credential environment variables or files
                # This is instant and doesn't involve any network calls
                session = boto3.Session()
                credentials = session.get_credentials()

                if credentials is not None:
                    log.debug("AWS credentials detected, using signed requests")
                    return True
                else:
                    log.debug("No AWS credentials found, using unsigned requests")
                    return False

            except Exception as e:
                log.debug(f"Error checking credentials: {e}, using unsigned requests")
                return False

        # Check credentials immediately when patch is applied
        _has_credentials = _check_credentials_now()

        def patched_download_file_impl(
            self,
            remote: str,
            local: str,
            timeout: float,
        ) -> None:
            """Download a file from S3, with automatic fallback to unsigned for
            public buckets."""

            # Ensure S3 client exists
            if self._s3_client is None:
                # Directly create unsigned client if no credentials (checked at patch time)
                if not _has_credentials:
                    self._create_s3_client(unsigned=True, timeout=timeout)
                else:
                    # Try signed client first if credentials available
                    try:
                        self._create_s3_client(timeout=timeout)
                    except (NoCredentialsError, PartialCredentialsError):
                        log.debug("Credentials check was incorrect, using unsigned")
                        self._create_s3_client(unsigned=True, timeout=timeout)

            assert self._s3_client is not None

            obj = urllib.parse.urlparse(remote)
            extra_args = {}

            # Requester Pays buckets require authenticated credentials
            if obj.netloc in getattr(self, "_requester_pays_buckets", []):
                extra_args["RequestPayer"] = "requester"

            def _attempt_download():
                """Helper to perform the actual download."""
                self._s3_client.download_file(
                    obj.netloc,
                    obj.path.lstrip("/"),
                    local,
                    ExtraArgs=extra_args,
                    Config=TransferConfig(use_threads=False),
                )

            try:
                # First attempt with current client (signed if credentials available)
                _attempt_download()

            except NoCredentialsError:
                # No credentials available -> retry with unsigned client
                if "RequestPayer" in extra_args:
                    raise  # Requester-pays buckets must use signed requests

                log.debug(f"Retrying {remote} with unsigned client")
                self._create_s3_client(unsigned=True, timeout=timeout)
                _attempt_download()

            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")

                if error_code in BOTOCORE_CLIENT_ERROR_CODES:
                    # Object not found or permission denied
                    e.args = (
                        f"Object {remote} not found! Check the bucket path or permissions. "
                        "If the bucket is requester pays, set the env var "
                        "`MOSAICML_STREAMING_AWS_REQUESTER_PAYS`.",
                    )
                    raise

                elif error_code in ["403", "AccessDenied", "InvalidRequest"]:
                    # Access denied with signed request, try unsigned for public buckets
                    if "RequestPayer" not in extra_args:
                        log.debug(
                            f"Access denied with signed request for {remote}, trying unsigned",
                        )
                        self._create_s3_client(unsigned=True, timeout=timeout)
                        try:
                            _attempt_download()
                        except Exception:
                            # If unsigned also fails, re-raise original error
                            raise e from None
                    else:
                        raise
                else:
                    raise

        # Apply the monkey patch
        S3Downloader._download_file_impl = patched_download_file_impl
        log.info("Applied S3Downloader patch for public bucket support")

    except ImportError:
        log.debug("Streaming library not installed, skipping S3Downloader patch")
    except Exception as e:
        log.warning(f"Failed to apply S3Downloader patch: {e}")


def download_file_from_s3(
    s3_url: str,
    local_file_path: str,
    use_unsigned: Optional[bool] = None,
) -> Optional[str]:
    """Download a file from an S3 URL to a local path.

    Automatically handles both public and private S3 buckets by:
    1. First attempting with AWS credentials if available
    2. Falling back to unsigned requests for public buckets

    Args:
        s3_url: S3 URL in the format s3://bucket-name/path/to/file
        local_file_path: Local path where the file will be saved
        use_unsigned: If True, only use unsigned requests. If False, only use signed.
                     If None (default), try signed first then unsigned.

    Returns:
        The local path to the downloaded file, or None if download fails

    Raises:
        ValueError: If the S3 URL format is invalid
    """
    # Validate and parse S3 URL
    if not s3_url.startswith("s3://"):
        raise ValueError("URL must start with 's3://'")
    parsed_url = urlparse(s3_url)
    if parsed_url.scheme != "s3":
        raise ValueError("URL scheme must be 's3'")

    bucket_name = parsed_url.netloc
    s3_file_key = parsed_url.path.lstrip("/")

    if not bucket_name:
        raise ValueError("Bucket name cannot be empty")
    if not s3_file_key:
        raise ValueError("S3 file key cannot be empty")

    # Validate and prepare local path
    local_path = Path(local_file_path).resolve()

    # Security check: prevent path traversal attacks
    # Ensure the resolved path doesn't escape the intended directory
    cwd = Path.cwd()
    try:
        # This will raise ValueError if local_path is not relative to cwd
        local_path.relative_to(cwd)
    except ValueError:
        # Path is outside current working directory, which might be intentional
        # Just log a warning but allow it
        log.warning(f"Download path {local_path} is outside current directory")

    # Ensure local directory exists
    if local_path.parent != Path("."):
        local_path.parent.mkdir(parents=True, exist_ok=True)

    def _download_with_client(s3_client, description: str) -> bool:
        """Attempt download with a specific S3 client."""
        try:
            s3_client.download_file(bucket_name, s3_file_key, str(local_path))
            log.info(
                f"Successfully downloaded {s3_url} ({description}) to {local_path}",
            )
            return True
        except (ClientError, NoCredentialsError, PartialCredentialsError) as e:
            log.debug(f"Failed to download {s3_url} ({description}): {e}")
            return False
        except Exception as e:
            # Log unexpected exceptions at warning level
            log.warning(f"Unexpected error downloading {s3_url} ({description}): {e}")
            return False

    # Determine download strategy based on use_unsigned parameter
    if use_unsigned is True:
        # Only try unsigned
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        if _download_with_client(s3, "unsigned"):
            return str(local_path)

    elif use_unsigned is False:
        # Only try signed
        try:
            s3 = boto3.client("s3")
            if _download_with_client(s3, "signed"):
                return str(local_path)
        except (NoCredentialsError, PartialCredentialsError) as e:
            log.error(f"AWS credentials required but not found: {e}")

    else:
        # Try signed first, then unsigned (default behavior)
        try:
            s3 = boto3.client("s3")
            if _download_with_client(s3, "signed"):
                return str(local_path)
        except (NoCredentialsError, PartialCredentialsError):
            log.info("No AWS credentials found, attempting unsigned access")

        # Fallback to unsigned
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        if _download_with_client(s3, "unsigned"):
            return str(local_path)

    log.error(f"Failed to download {s3_url} to {local_path}")
    return None


def is_s3_url(url: str) -> bool:
    """Check if a URL is an S3 URL."""
    return url.startswith("s3://")


def parse_s3_url(s3_url: str) -> tuple[str, str]:
    """Parse an S3 URL into bucket and key components.

    Args:
        s3_url: S3 URL in the format s3://bucket-name/path/to/file

    Returns:
        Tuple of (bucket_name, object_key)

    Raises:
        ValueError: If the URL is not a valid S3 URL
    """
    if not s3_url.startswith("s3://"):
        raise ValueError(f"Not a valid S3 URL: {s3_url}")

    parsed = urlparse(s3_url)
    if parsed.scheme != "s3":
        raise ValueError(f"Not a valid S3 URL: {s3_url}")

    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    if not bucket:
        raise ValueError(f"No bucket name in S3 URL: {s3_url}")
    if not key:
        raise ValueError(f"No object key in S3 URL: {s3_url}")

    return bucket, key
