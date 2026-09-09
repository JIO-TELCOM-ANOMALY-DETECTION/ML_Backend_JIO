"""
Pipeline-run directory management for the telecom anomaly backend.

This module generates run identifiers, builds the standard top-level
directory paths for each pipeline run, creates the required run folders,
validates existing run directories, and lists previously created runs.

The service manages run-level directory structure only. It does not create
attempt/stage/execution folders, copy input files, write manifests, resolve
artifacts, execute workflows, or configure log handlers. Those higher-level
responsibilities belong to the run-management and workflow layers.
"""

import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.core.config import settings
from app.core.exceptions import RunDirectoryError


# ---------------------------------------------------------------------------
# STANDARD RUN-LEVEL DIRECTORY STRUCTURE
# ---------------------------------------------------------------------------
# Every run receives these stable top-level subdirectories under:
#
#     runs/<run_id>/
#
# Attempt-specific training data and stage outputs will later live below
# "attempts/". Part-E saved state and invocation records will live below
# "inference/". Their deeper structures are intentionally not owned here.
RUN_SUBDIRECTORIES: dict[str, str] = {
    "attempts": "attempts",
    "inference": "inference",
    "reports": "reports",
    "agents": "agents",
    "temporary": "temporary",
}


# Run IDs may contain letters, numbers, underscores, hyphens, and full stops.
# The first character must be a letter or number.
RUN_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
)


def generate_run_id(
    prefix: str = "run",
) -> str:
    """
    Generate a unique identifier for a new backend run.

    The identifier contains a readable UTC timestamp and a short random UUID
    section to reduce the chance of two runs receiving the same ID.

    Args:
        prefix: Text placed at the beginning of the generated run ID.

    Returns:
        Run identifier such as:
        run_20260723T041530123456Z_a4f918c2

    Raises:
        RunDirectoryError: If the supplied prefix is invalid.
    """
    cleaned_prefix = prefix.strip()

    if not cleaned_prefix:
        raise RunDirectoryError(
            "The run-ID prefix cannot be empty.",
            error_code="EMPTY_RUN_ID_PREFIX",
        )

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}",
        cleaned_prefix,
    ):
        raise RunDirectoryError(
            "The run-ID prefix contains unsupported characters.",
            error_code="INVALID_RUN_ID_PREFIX",
            details={
                "received_prefix": prefix,
            },
        )

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    unique_section = uuid4().hex[:8]

    return (
        f"{cleaned_prefix}_"
        f"{timestamp}_"
        f"{unique_section}"
    )


def validate_run_id(
    run_id: str,
) -> str:
    """
    Validate and normalise a backend run identifier.

    Validation prevents empty values, directory traversal, path separators,
    and unsupported filename characters.

    Args:
        run_id: Run identifier that must be validated.

    Returns:
        Validated run identifier.

    Raises:
        RunDirectoryError: If the run identifier is unsafe or invalid.
    """
    cleaned_run_id = run_id.strip()

    if not cleaned_run_id:
        raise RunDirectoryError(
            "The run ID cannot be empty.",
            error_code="EMPTY_RUN_ID",
        )

    if cleaned_run_id in {
        ".",
        "..",
    }:
        raise RunDirectoryError(
            "The run ID cannot be '.' or '..'.",
            error_code="INVALID_RUN_ID",
            details={
                "run_id": cleaned_run_id,
            },
        )

    if not RUN_ID_PATTERN.fullmatch(
        cleaned_run_id
    ):
        raise RunDirectoryError(
            "The run ID contains unsupported characters.",
            error_code="INVALID_RUN_ID",
            details={
                "run_id": cleaned_run_id,
                "allowed_characters": (
                    "Letters, numbers, underscores, "
                    "hyphens, and full stops."
                ),
                "maximum_length": 128,
            },
        )

    return cleaned_run_id


def get_run_directory(
    run_id: str,
) -> Path:
    """
    Build the root directory path for a backend run.

    Args:
        run_id: Unique identifier assigned to the run.

    Returns:
        Path in the form runs/<run_id>.
    """
    validated_run_id = validate_run_id(
        run_id
    )

    return settings.get_run_directory(
        validated_run_id
    )


def get_run_log_directory(
    run_id: str,
) -> Path:
    """
    Build the separate logging directory path for a backend run.

    Args:
        run_id: Unique identifier assigned to the run.

    Returns:
        Path in the form logs/<run_id>.
    """
    validated_run_id = validate_run_id(
        run_id
    )

    return settings.get_run_log_directory(
        validated_run_id
    )


def get_run_paths(
    run_id: str,
) -> dict[str, Path]:
    """
    Build all standard top-level paths associated with a backend run.

    This function only calculates paths. It does not create any directories.

    The returned mapping contains:

        run
        logs
        manifest
        attempts
        inference
        reports
        agents
        temporary

    Args:
        run_id: Unique identifier assigned to the run.

    Returns:
        Dictionary containing the run root, log directory, manifest path,
        and every configured top-level run subdirectory.
    """
    validated_run_id = validate_run_id(
        run_id
    )

    run_directory = get_run_directory(
        validated_run_id
    )

    run_paths: dict[str, Path] = {
        "run": run_directory,
        "logs": get_run_log_directory(
            validated_run_id
        ),
        "manifest": (
            run_directory
            / "manifest.json"
        ),
    }

    for key, directory_name in (
        RUN_SUBDIRECTORIES.items()
    ):
        run_paths[key] = (
            run_directory
            / directory_name
        )

    return run_paths


def get_run_subdirectory(
    run_id: str,
    directory_key: str,
) -> Path:
    """
    Return one configured top-level subdirectory path for a backend run.

    Args:
        run_id: Unique identifier assigned to the run.
        directory_key: Configured directory key such as attempts, inference,
            reports, agents, or temporary.

    Returns:
        Requested run-specific directory path.

    Raises:
        RunDirectoryError: If the directory key is unsupported.
    """
    normalised_key = (
        directory_key.strip().lower()
    )

    if normalised_key not in (
        RUN_SUBDIRECTORIES
    ):
        raise RunDirectoryError(
            "The requested run subdirectory is not configured.",
            error_code="UNKNOWN_RUN_SUBDIRECTORY",
            details={
                "received_key": directory_key,
                "available_keys": sorted(
                    RUN_SUBDIRECTORIES
                ),
            },
        )

    return get_run_paths(
        run_id
    )[normalised_key]


def create_run_directories(
    run_id: str | None = None,
    *,
    exist_ok: bool = False,
) -> dict[str, str | Path]:
    """
    Create the standard top-level directory structure for a backend run.

    When no run ID is provided, a new unique run ID is generated. The stable
    run-level folders and the separate run logging folder are then created.

    This service intentionally does not create attempt, stage, execution, or
    inference-invocation child folders. Those will be created later by the
    run-management layer according to the active lifecycle.

    Args:
        run_id: Optional externally supplied run identifier.
        exist_ok: Whether an already existing run directory may be reused.

    Returns:
        Dictionary containing the run ID and every standard top-level path.

    Raises:
        RunDirectoryError: If the directories already exist unexpectedly or
            cannot be created.
    """
    final_run_id = (
        validate_run_id(run_id)
        if run_id is not None
        else generate_run_id()
    )

    run_paths = get_run_paths(
        final_run_id
    )

    run_directory = run_paths["run"]
    log_directory = run_paths["logs"]

    if run_directory.exists() and not exist_ok:
        raise RunDirectoryError(
            "A directory already exists for this backend run.",
            error_code="RUN_DIRECTORY_ALREADY_EXISTS",
            details={
                "run_id": final_run_id,
                "run_directory": str(
                    run_directory
                ),
            },
        )

    try:
        settings.create_base_directories()

        run_directory.mkdir(
            parents=True,
            exist_ok=exist_ok,
        )

        log_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for directory_key in (
            RUN_SUBDIRECTORIES
        ):
            run_paths[
                directory_key
            ].mkdir(
                parents=True,
                exist_ok=True,
            )

    except OSError as error:
        raise RunDirectoryError(
            "Unable to create the backend run directories.",
            error_code="RUN_DIRECTORY_CREATION_FAILED",
            details={
                "run_id": final_run_id,
                "run_directory": str(
                    run_directory
                ),
                "log_directory": str(
                    log_directory
                ),
                "original_error": str(error),
            },
        ) from error

    return {
        "run_id": final_run_id,
        **run_paths,
    }


def validate_run_directory(
    run_id: str,
) -> Path:
    """
    Validate that a backend run directory already exists.

    Args:
        run_id: Unique identifier assigned to the run.

    Returns:
        Existing run-directory path.

    Raises:
        RunDirectoryError: If the run directory does not exist or is not a
            directory.
    """
    run_directory = get_run_directory(
        run_id
    )

    if not run_directory.exists():
        raise RunDirectoryError(
            "The requested backend run directory does not exist.",
            error_code="RUN_DIRECTORY_NOT_FOUND",
            details={
                "run_id": run_id,
                "run_directory": str(
                    run_directory
                ),
            },
        )

    if not run_directory.is_dir():
        raise RunDirectoryError(
            "The backend run path is not a directory.",
            error_code="RUN_PATH_IS_NOT_DIRECTORY",
            details={
                "run_id": run_id,
                "run_directory": str(
                    run_directory
                ),
            },
        )

    return run_directory


def run_directory_exists(
    run_id: str,
) -> bool:
    """
    Check whether a valid backend run directory exists.

    Args:
        run_id: Unique identifier assigned to the run.

    Returns:
        True when the run path exists as a directory; otherwise False.
    """
    run_directory = get_run_directory(
        run_id
    )

    return (
        run_directory.exists()
        and run_directory.is_dir()
    )


def list_run_ids() -> list[str]:
    """
    List the identifiers of all existing backend run directories.

    This filesystem listing is a low-level service operation. Once the run
    registry is introduced, user-facing run lookup should prefer the registry
    rather than treating directory discovery as the primary source of truth.

    Returns:
        Alphabetically sorted list of valid run-directory names.

    Raises:
        RunDirectoryError: If the shared runs directory cannot be inspected.
    """
    runs_directory = (
        settings.runs_directory
    )

    if not runs_directory.exists():
        return []

    try:
        run_ids = [
            path.name
            for path in runs_directory.iterdir()
            if (
                path.is_dir()
                and RUN_ID_PATTERN.fullmatch(
                    path.name
                )
            )
        ]

    except OSError as error:
        raise RunDirectoryError(
            "Unable to inspect the backend runs directory.",
            error_code="RUN_DIRECTORY_LIST_FAILED",
            details={
                "runs_directory": str(
                    runs_directory
                ),
                "original_error": str(error),
            },
        ) from error

    return sorted(
        run_ids
    )
