"""
version saving before safe name change edit.
Safe file-handling services for the telecom anomaly detection backend.

This module validates input files, checks file restrictions, creates safe
filenames, copies files into pipeline-run directories, and reads or writes
CSV and JSON files.

The module does not decide where a run directory should be located.
The run-directory service will supply those destination paths.
"""

import hashlib
import json
import logging
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar
from uuid import uuid4

import pandas as pd


logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Windows can temporarily deny rename/replace/delete operations when another
# process briefly opens a file or directory without delete-sharing enabled.
# Keep retries deliberately narrow and bounded: only these known transient
# Windows sharing/access conditions are retried.
_TRANSIENT_WINDOWS_FILESYSTEM_ERRORS = frozenset({5, 32, 33})
_FILESYSTEM_RETRY_DELAYS_SECONDS = (0.05, 0.10, 0.20, 0.40, 0.80)

from app.core.config import settings
from app.core.exceptions import (
    FileProcessingError,
    InputValidationError,
)

# # alternate import of app.core dependecies
# from app.core import (
#     FileProcessingError,
#     InputValidationError,
#     settings,
# )



def _is_transient_windows_filesystem_error(error: OSError) -> bool:
    """Return whether an OSError is a retryable Windows sharing/access error."""
    return (
        os.name == "nt"
        and getattr(error, "winerror", None)
        in _TRANSIENT_WINDOWS_FILESYSTEM_ERRORS
    )


def _run_filesystem_operation_with_retry(
    operation: Callable[[], _T],
    *,
    operation_name: str,
    source_path: Path | None = None,
    destination_path: Path | None = None,
) -> _T:
    """
    Execute one filesystem mutation with bounded retry for transient Windows locks.

    Non-Windows errors and Windows errors outside the small retryable set are
    raised immediately. If all retries are exhausted, the original final
    OSError is re-raised so existing caller-level exception contracts remain
    unchanged.
    """
    retry_number = 0

    while True:
        try:
            return operation()
        except OSError as error:
            if not _is_transient_windows_filesystem_error(error):
                raise

            if retry_number >= len(_FILESYSTEM_RETRY_DELAYS_SECONDS):
                raise

            delay_seconds = _FILESYSTEM_RETRY_DELAYS_SECONDS[retry_number]
            retry_number += 1

            logger.warning(
                "Transient Windows filesystem error during %s; retry %d/%d "
                "in %.2fs (winerror=%s, source=%s, destination=%s): %s",
                operation_name,
                retry_number,
                len(_FILESYSTEM_RETRY_DELAYS_SECONDS),
                delay_seconds,
                getattr(error, "winerror", None),
                source_path,
                destination_path,
                error,
            )

            time.sleep(delay_seconds)


def rename_path_with_retry(
    source_path: str | Path,
    destination_path: str | Path,
) -> Path:
    """
    Rename a file or directory while tolerating short-lived Windows locks.

    This preserves Path.rename() semantics; it does not silently overwrite a
    destination that the underlying platform would reject.
    """
    source = Path(source_path).expanduser()
    destination = Path(destination_path).expanduser()

    return _run_filesystem_operation_with_retry(
        lambda: source.rename(destination),
        operation_name="rename_path",
        source_path=source,
        destination_path=destination,
    )


def replace_path_with_retry(
    source_path: str | Path,
    destination_path: str | Path,
) -> Path:
    """
    Replace/move a file or directory while tolerating short-lived Windows locks.

    This preserves Path.replace() semantics and re-raises the final OSError when
    a transient lock does not clear within the bounded retry window.
    """
    source = Path(source_path).expanduser()
    destination = Path(destination_path).expanduser()

    return _run_filesystem_operation_with_retry(
        lambda: source.replace(destination),
        operation_name="replace_path",
        source_path=source,
        destination_path=destination,
    )


def remove_tree_with_retry(
    directory_path: str | Path,
) -> None:
    """Remove a directory tree with bounded retry for transient Windows locks."""
    directory = Path(directory_path).expanduser()

    if not directory.exists():
        return

    def _remove() -> None:
        if not directory.exists():
            return
        shutil.rmtree(directory)

    _run_filesystem_operation_with_retry(
        _remove,
        operation_name="remove_tree",
        source_path=directory,
    )


def ensure_directory(directory_path: str | Path) -> Path:
    """
    Create a directory when it does not already exist.

    Args:
        directory_path: Directory that must exist.

    Returns:
        The resolved directory path.

    Raises:
        FileProcessingError: If the directory cannot be created.
    """
    resolved_directory = Path(
        directory_path
    ).expanduser().resolve()

    try:
        resolved_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    except OSError as error:
        raise FileProcessingError(
            "Unable to create the required directory.",
            error_code="DIRECTORY_CREATION_FAILED",
            details={
                "directory": str(resolved_directory),
                "original_error": str(error),
            },
        ) from error

    return resolved_directory


def sanitise_filename(filename: str) -> str:
    """
    Convert a supplied filename into a safe standalone filename.

    Directory information is removed, unsupported filename characters are
    replaced, and empty or unsafe names are rejected.

    Args:
        filename: Original filename supplied by a user or another service.

    Returns:
        A safe filename without directory components.

    Raises:
        InputValidationError: If a valid filename cannot be produced.
    """
    original_name = Path(filename).name.strip()

    if not original_name:
        raise InputValidationError(
            "The filename cannot be empty.",
            error_code="EMPTY_FILENAME",
        )

    # Replace characters that are invalid or unsafe in common file systems.
    safe_name = re.sub(
        r'[<>:"/\\|?*\x00-\x1f]',
        "_",
        original_name,
    )

    safe_name = safe_name.strip(" .")

    if not safe_name or safe_name in {".", ".."}:
        raise InputValidationError(
            "The supplied filename is invalid.",
            error_code="INVALID_FILENAME",
            details={
                "received_filename": filename,
            },
        )

    return safe_name


def validate_file_exists(
    file_path: str | Path,
) -> Path:
    """
    Validate that a supplied path points to an existing file.

    Args:
        file_path: Path of the file that must exist.

    Returns:
        The resolved file path.

    Raises:
        InputValidationError: If the path does not exist or is not a file.
    """
    resolved_path = Path(
        file_path
    ).expanduser().resolve()

    if not resolved_path.exists():
        raise InputValidationError(
            "The requested file does not exist.",
            error_code="FILE_NOT_FOUND",
            details={
                "file_path": str(resolved_path),
            },
        )

    if not resolved_path.is_file():
        raise InputValidationError(
            "The supplied path does not point to a file.",
            error_code="PATH_IS_NOT_FILE",
            details={
                "file_path": str(resolved_path),
            },
        )

    return resolved_path


def validate_file_extension(
    file_path: str | Path,
    allowed_extensions: tuple[str, ...] | None = None,
) -> Path:
    """
    Validate that a file uses one of the permitted extensions.

    Args:
        file_path: File whose extension must be checked.
        allowed_extensions: Optional permitted extensions. Global settings are
            used when this argument is omitted.

    Returns:
        The resolved file path.

    Raises:
        InputValidationError: If the file extension is unsupported.
    """
    resolved_path = Path(
        file_path
    ).expanduser().resolve()

    configured_extensions = (
        allowed_extensions
        if allowed_extensions is not None
        else settings.allowed_input_extensions
    )

    normalised_extensions = tuple(
        extension.lower()
        if extension.startswith(".")
        else f".{extension.lower()}"
        for extension in configured_extensions
    )

    received_extension = resolved_path.suffix.lower()

    if received_extension not in normalised_extensions:
        raise InputValidationError(
            "The supplied file type is not supported.",
            error_code="INVALID_FILE_EXTENSION",
            details={
                "file_path": str(resolved_path),
                "received_extension": received_extension,
                "allowed_extensions": list(
                    normalised_extensions
                ),
            },
        )

    return resolved_path


def validate_file_size(
    file_path: str | Path,
    max_size_bytes: int | None = None,
) -> Path:
    """
    Validate that a file is not empty and does not exceed the size limit.

    Args:
        file_path: File whose size must be checked.
        max_size_bytes: Optional size limit in bytes. The global 5 GB setting
            is used when this argument is omitted.

    Returns:
        The resolved file path.

    Raises:
        InputValidationError: If the file is empty or exceeds the limit.
        FileProcessingError: If the file size cannot be inspected.
    """
    resolved_path = Path(
        file_path
    ).expanduser().resolve()

    permitted_size = (
        max_size_bytes
        if max_size_bytes is not None
        else settings.max_input_file_size_bytes
    )

    if permitted_size <= 0:
        raise InputValidationError(
            "The maximum file-size limit must be greater than zero.",
            error_code="INVALID_FILE_SIZE_LIMIT",
            details={
                "max_size_bytes": permitted_size,
            },
        )

    try:
        file_size = resolved_path.stat().st_size

    except OSError as error:
        raise FileProcessingError(
            "Unable to inspect the file size.",
            error_code="FILE_SIZE_READ_FAILED",
            details={
                "file_path": str(resolved_path),
                "original_error": str(error),
            },
        ) from error

    if file_size == 0:
        raise InputValidationError(
            "The supplied file is empty.",
            error_code="EMPTY_FILE",
            details={
                "file_path": str(resolved_path),
            },
        )

    if file_size > permitted_size:
        raise InputValidationError(
            "The supplied file exceeds the permitted size.",
            error_code="FILE_TOO_LARGE",
            details={
                "file_path": str(resolved_path),
                "file_size_bytes": file_size,
                "maximum_size_bytes": permitted_size,
                "maximum_size_gb": (
                    permitted_size / 1024**3
                ),
            },
        )

    return resolved_path


def validate_input_file(
    file_path: str | Path,
    *,
    allowed_extensions: tuple[str, ...] | None = None,
    max_size_bytes: int | None = None,
) -> Path:
    """
    Perform all general validation checks for an input file.

    This function checks that the file exists, is a regular file, has a
    permitted extension, is not empty, and does not exceed the size limit.

    Args:
        file_path: Input file that must be validated.
        allowed_extensions: Optional permitted file extensions.
        max_size_bytes: Optional maximum permitted file size in bytes.

    Returns:
        The validated and resolved input-file path.
    """
    validated_path = validate_file_exists(
        file_path
    )

    validate_file_extension(
        validated_path,
        allowed_extensions=allowed_extensions,
    )

    validate_file_size(
        validated_path,
        max_size_bytes=max_size_bytes,
    )

    return validated_path


def calculate_file_sha256(
    file_path: str | Path,
    *,
    chunk_size_bytes: int = 1024 * 1024,
) -> str:
    """
    Calculate the SHA-256 checksum of a file using memory-safe chunks.

    The checksum can be stored in the run manifest to identify whether two
    files contain exactly the same content.

    Args:
        file_path: File whose checksum must be calculated.
        chunk_size_bytes: Number of bytes read during each iteration.

    Returns:
        SHA-256 checksum represented as a hexadecimal string.

    Raises:
        InputValidationError: If the chunk size is invalid.
        FileProcessingError: If the file cannot be read.
    """
    validated_path = validate_file_exists(
        file_path
    )

    if chunk_size_bytes <= 0:
        raise InputValidationError(
            "Checksum chunk size must be greater than zero.",
            error_code="INVALID_CHECKSUM_CHUNK_SIZE",
            details={
                "chunk_size_bytes": chunk_size_bytes,
            },
        )

    checksum = hashlib.sha256()

    try:
        with validated_path.open("rb") as file:
            while chunk := file.read(
                chunk_size_bytes
            ):
                checksum.update(chunk)

    except OSError as error:
        raise FileProcessingError(
            "Unable to calculate the file checksum.",
            error_code="CHECKSUM_CALCULATION_FAILED",
            details={
                "file_path": str(validated_path),
                "original_error": str(error),
            },
        ) from error

    return checksum.hexdigest()


def get_file_metadata(
    file_path: str | Path,
    *,
    include_checksum: bool = False,
) -> dict[str, Any]:
    """
    Generate metadata describing a file.

    Args:
        file_path: File whose metadata must be generated.
        include_checksum: Whether to calculate and include a SHA-256 checksum.

    Returns:
        Dictionary containing file name, path, extension, size, modification
        time, and optionally the checksum.

    Raises:
        FileProcessingError: If file metadata cannot be inspected.
    """
    validated_path = validate_file_exists(
        file_path
    )

    try:
        file_statistics = validated_path.stat()

    except OSError as error:
        raise FileProcessingError(
            "Unable to read file metadata.",
            error_code="FILE_METADATA_READ_FAILED",
            details={
                "file_path": str(validated_path),
                "original_error": str(error),
            },
        ) from error

    modified_time = datetime.fromtimestamp(
        file_statistics.st_mtime,
        tz=timezone.utc,
    )

    metadata: dict[str, Any] = {
        "file_name": validated_path.name,
        "file_path": str(validated_path),
        "extension": validated_path.suffix.lower(),
        "size_bytes": file_statistics.st_size,
        "size_mb": round(
            file_statistics.st_size / 1024**2,
            3,
        ),
        "modified_at_utc": modified_time.isoformat(),
    }

    if include_checksum:
        metadata["sha256"] = calculate_file_sha256(
            validated_path
        )

    return metadata


def copy_input_file(
    source_path: str | Path,
    destination_directory: str | Path,
    *,
    destination_filename: str | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Validate and copy an input file into a destination directory.

    Args:
        source_path: Existing input file that must be copied.
        destination_directory: Directory receiving the copied file.
        destination_filename: Optional new filename for the copied file.
        overwrite: Whether an existing destination file may be replaced.

    Returns:
        Path of the copied destination file.

    Raises:
        InputValidationError: If the input file or destination name is invalid.
        FileProcessingError: If the file cannot be copied.
    """
    validated_source = validate_input_file(
        source_path
    )

    resolved_destination_directory = ensure_directory(
        destination_directory
    )

    final_filename = sanitise_filename(
        destination_filename
        if destination_filename is not None
        else validated_source.name
    )

    destination_path = (
        resolved_destination_directory
        / final_filename
    )

    if destination_path.exists() and not overwrite:
        raise FileProcessingError(
            "A file with the same name already exists in the destination.",
            error_code="DESTINATION_FILE_EXISTS",
            details={
                "source_path": str(validated_source),
                "destination_path": str(destination_path),
            },
        )

    try:
        shutil.copy2(
            src=validated_source,
            dst=destination_path,
        )

    except OSError as error:
        raise FileProcessingError(
            "Unable to copy the input file.",
            error_code="FILE_COPY_FAILED",
            details={
                "source_path": str(validated_source),
                "destination_path": str(destination_path),
                "original_error": str(error),
            },
        ) from error

    return destination_path.resolve()


def read_csv_file(
    file_path: str | Path,
    **read_csv_options: Any,
) -> pd.DataFrame:
    """
    Read a CSV file into a pandas DataFrame.

    Args:
        file_path: CSV file that must be loaded.
        **read_csv_options: Additional options passed directly to
            pandas.read_csv.

    Returns:
        DataFrame containing the CSV data.

    Raises:
        InputValidationError: If the CSV file is invalid.
        FileProcessingError: If pandas cannot read the CSV file.
    """
    validated_path = validate_input_file(
        file_path,
        allowed_extensions=(".csv",),
    )

    options = {
        "encoding": settings.default_file_encoding,
        **read_csv_options,
    }

    try:
        dataframe = pd.read_csv(
            validated_path,
            **options,
        )

    except UnicodeDecodeError as error:
        raise FileProcessingError(
            "The CSV file could not be decoded using the selected encoding.",
            error_code="CSV_ENCODING_ERROR",
            details={
                "file_path": str(validated_path),
                "encoding": options.get("encoding"),
                "original_error": str(error),
            },
        ) from error

    except pd.errors.EmptyDataError as error:
        raise InputValidationError(
            "The CSV file does not contain any readable columns.",
            error_code="EMPTY_CSV",
            details={
                "file_path": str(validated_path),
            },
        ) from error

    except pd.errors.ParserError as error:
        raise FileProcessingError(
            "The CSV file could not be parsed.",
            error_code="CSV_PARSE_FAILED",
            details={
                "file_path": str(validated_path),
                "original_error": str(error),
            },
        ) from error

    except (OSError, ValueError) as error:
        raise FileProcessingError(
            "Unable to read the CSV file.",
            error_code="CSV_READ_FAILED",
            details={
                "file_path": str(validated_path),
                "original_error": str(error),
            },
        ) from error

    return dataframe


def write_csv_file(
    dataframe: pd.DataFrame,
    file_path: str | Path,
    *,
    index: bool = False,
    overwrite: bool = True,
    **to_csv_options: Any,
) -> Path:
    """
    Write a pandas DataFrame to a CSV file.

    Args:
        dataframe: DataFrame that must be written.
        file_path: Destination CSV path.
        index: Whether the DataFrame index should be included.
        overwrite: Whether an existing CSV file may be replaced.
        **to_csv_options: Additional options passed to DataFrame.to_csv.

    Returns:
        Resolved path of the generated CSV file.

    Raises:
        InputValidationError: If the destination does not use `.csv`.
        FileProcessingError: If the CSV file cannot be written.
    """
    destination_path = Path(
        file_path
    ).expanduser().resolve()

    if destination_path.suffix.lower() != ".csv":
        raise InputValidationError(
            "CSV output files must use the .csv extension.",
            error_code="INVALID_CSV_OUTPUT_EXTENSION",
            details={
                "file_path": str(destination_path),
            },
        )

    if destination_path.exists() and not overwrite:
        raise FileProcessingError(
            "The destination CSV file already exists.",
            error_code="DESTINATION_FILE_EXISTS",
            details={
                "file_path": str(destination_path),
            },
        )

    ensure_directory(
        destination_path.parent
    )

    options = {
        "encoding": settings.default_file_encoding,
        **to_csv_options,
    }

    try:
        dataframe.to_csv(
            destination_path,
            index=index,
            **options,
        )

    except (OSError, ValueError, TypeError) as error:
        raise FileProcessingError(
            "Unable to write the CSV file.",
            error_code="CSV_WRITE_FAILED",
            details={
                "file_path": str(destination_path),
                "original_error": str(error),
            },
        ) from error

    return destination_path


def read_json_file(
    file_path: str | Path,
) -> Any:
    """
    Read and decode a JSON file.

    Args:
        file_path: JSON file that must be read.

    Returns:
        Python object decoded from the JSON content.

    Raises:
        InputValidationError: If the file does not use the `.json` extension.
        FileProcessingError: If the JSON file cannot be read or decoded.
    """
    validated_path = validate_file_exists(
        file_path
    )

    if validated_path.suffix.lower() != ".json":
        raise InputValidationError(
            "The requested file must use the .json extension.",
            error_code="INVALID_JSON_EXTENSION",
            details={
                "file_path": str(validated_path),
            },
        )

    try:
        with validated_path.open(
            mode="r",
            encoding=settings.default_file_encoding,
        ) as file:
            return json.load(file)

    except json.JSONDecodeError as error:
        raise FileProcessingError(
            "The JSON file contains invalid JSON.",
            error_code="JSON_DECODE_FAILED",
            details={
                "file_path": str(validated_path),
                "line_number": error.lineno,
                "column_number": error.colno,
                "original_error": str(error),
            },
        ) from error

    except OSError as error:
        raise FileProcessingError(
            "Unable to read the JSON file.",
            error_code="JSON_READ_FAILED",
            details={
                "file_path": str(validated_path),
                "original_error": str(error),
            },
        ) from error


def write_json_file(
    data: Any,
    file_path: str | Path,
    *,
    overwrite: bool = True,
    indent: int = 4,
) -> Path:
    """
    Write Python data to a JSON file using an atomic replacement.

    Data is first written to a temporary file in the destination directory.
    The temporary file replaces the final file only after writing succeeds,
    reducing the chance of leaving a partially written manifest.

    Args:
        data: JSON-serialisable Python data.
        file_path: Destination JSON path.
        overwrite: Whether an existing JSON file may be replaced.
        indent: Number of spaces used for JSON indentation.

    Returns:
        Resolved path of the generated JSON file.

    Raises:
        InputValidationError: If the destination extension is invalid.
        FileProcessingError: If the JSON file cannot be written.
    """
    destination_path = Path(
        file_path
    ).expanduser().resolve()

    if destination_path.suffix.lower() != ".json":
        raise InputValidationError(
            "JSON output files must use the .json extension.",
            error_code="INVALID_JSON_OUTPUT_EXTENSION",
            details={
                "file_path": str(destination_path),
            },
        )

    if destination_path.exists() and not overwrite:
        raise FileProcessingError(
            "The destination JSON file already exists.",
            error_code="DESTINATION_FILE_EXISTS",
            details={
                "file_path": str(destination_path),
            },
        )

    ensure_directory(
        destination_path.parent
    )

    temporary_path = destination_path.with_name(
        f".{destination_path.name}.{uuid4().hex}.tmp"
    )

    try:
        with temporary_path.open(
            mode="w",
            encoding=settings.default_file_encoding,
        ) as file:
            json.dump(
                data,
                file,
                indent=indent,
                ensure_ascii=False,
                default=str,
            )

        replace_path_with_retry(
            temporary_path,
            destination_path,
        )

    except (OSError, TypeError, ValueError) as error:
        raise FileProcessingError(
            "Unable to write the JSON file.",
            error_code="JSON_WRITE_FAILED",
            details={
                "file_path": str(destination_path),
                "original_error": str(error),
            },
        ) from error

    finally:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass

    return destination_path
