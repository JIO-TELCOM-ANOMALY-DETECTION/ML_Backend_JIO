"""
Pipeline-run manifest management for the telecom anomaly backend.

Each pipeline run receives a manifest.json file containing:

- run identification
- run status
- input-file metadata
- requested models and options
- preprocessing and model stage statuses
- generated output files
- errors
- summary information
- timestamps and event history

The manifest allows the API, UI, MCP layer, and pipeline orchestrator to inspect
the current state and final result of a run.
"""

from collections.abc import Mapping # mapping is a abstact base class, it is used to represent, check, or hint any read-only dictionary-like object
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.exceptions import (
    FileProcessingError,
    InputValidationError,
    RunManifestError,
    TelecomBackendError,
)
from app.services.file_service import (
    get_file_metadata,
    read_json_file,
    write_json_file,
)

from app.services.run_directory_service import (
    get_run_directory,
    validate_run_directory,
    validate_run_id,
)


MANIFEST_FILENAME = "manifest.json"
MANIFEST_VERSION = "1.0"

VALID_RUN_STATUSES = {
    "created",
    "queued",
    "running",
    "completed",
    "failed",
    "cancelled",
}

VALID_STAGE_STATUSES = {
    "pending",
    "running",
    "completed",
    "failed",
    "skipped",
}


def get_current_utc_timestamp() -> str:
    """
    Generate the current timezone-aware UTC timestamp.

    Returns:
        Current UTC timestamp in ISO 8601 format.
    """
    return datetime.now(
        timezone.utc
    ).isoformat()


def get_manifest_path(
    run_directory: str | Path,
) -> Path:
    """
    Build the manifest-file path for a pipeline run.

    Args:
        run_directory: Root directory of the pipeline run.

    Returns:
        Path in the form runs/<run_id>/manifest.json.
    """
    resolved_run_directory = Path(
        run_directory
    ).expanduser().resolve()

    return (
        resolved_run_directory
        / MANIFEST_FILENAME
    )


def validate_run_status(status: str) -> str:
    """
    Validate and normalise a pipeline-run status.

    Args:
        status: Proposed pipeline-run status.

    Returns:
        Normalised lowercase status.

    Raises:
        RunManifestError: If the status is unsupported.
    """
    normalised_status = status.strip().lower()

    if normalised_status not in VALID_RUN_STATUSES:
        raise RunManifestError(
            f"Unsupported run status '{status}'.",
            error_code="INVALID_RUN_STATUS",
            details={
                "received_status": status,
                "allowed_statuses": sorted(
                    VALID_RUN_STATUSES
                ),
            },
        )

    return normalised_status


def validate_stage_status(status: str) -> str:
    """
    Validate and normalise a pipeline-stage status.

    Args:
        status: Proposed pipeline-stage status.

    Returns:
        Normalised lowercase status.

    Raises:
        RunManifestError: If the status is unsupported.
    """
    normalised_status = status.strip().lower()

    if normalised_status not in VALID_STAGE_STATUSES:
        raise RunManifestError(
            f"Unsupported stage status '{status}'.",
            error_code="INVALID_STAGE_STATUS",
            details={
                "received_status": status,
                "allowed_statuses": sorted(
                    VALID_STAGE_STATUSES
                ),
            },
        )

    return normalised_status


def validate_manifest_structure(
    manifest: Mapping[str, Any],
) -> None:
    """
    Validate the minimum required structure of a run manifest.

    Args:
        manifest: Manifest data that must be validated.

    Raises:
        RunManifestError: If required fields are missing or invalid.
    """
    required_fields = {
        "manifest_version",
        "run_id",
        "status",
        "created_at_utc",
        "updated_at_utc",
        "stages",
        "outputs",
        "errors",
        "events",
    }

    missing_fields = sorted(
        required_fields - set(manifest)
    )

    if missing_fields:
        raise RunManifestError(
            "The run manifest is missing required fields.",
            error_code="INVALID_MANIFEST_STRUCTURE",
            details={
                "missing_fields": missing_fields,
            },
        )

    if not isinstance(manifest["run_id"], str):
        raise RunManifestError(
            "The manifest run ID must be a string.",
            error_code="INVALID_MANIFEST_RUN_ID",
        )

    validate_run_status(
        str(manifest["status"])
    )

    if not isinstance(manifest["stages"], dict):
        raise RunManifestError(
            "The manifest stages field must be a dictionary.",
            error_code="INVALID_MANIFEST_STAGES",
        )

    if not isinstance(manifest["outputs"], list):
        raise RunManifestError(
            "The manifest outputs field must be a list.",
            error_code="INVALID_MANIFEST_OUTPUTS",
        )

    if not isinstance(manifest["errors"], list):
        raise RunManifestError(
            "The manifest errors field must be a list.",
            error_code="INVALID_MANIFEST_ERRORS",
        )

    if not isinstance(manifest["events"], list):
        raise RunManifestError(
            "The manifest events field must be a list.",
            error_code="INVALID_MANIFEST_EVENTS",
        )


def append_manifest_event(
    manifest: dict[str, Any],
    event_type: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> None:
    """
    Add a timestamped event to a run manifest.

    This function modifies the supplied manifest dictionary in place.

    Args:
        manifest: Manifest receiving the event.
        event_type: Machine-readable category of the event.
        message: Human-readable description of the event.
        details: Optional structured event information.
    """
    event = {
        "timestamp_utc": get_current_utc_timestamp(),
        "event_type": event_type,
        "message": message,
        "details": dict(details or {}),
    }

    manifest.setdefault(
        "events",
        [],
    ).append(event)


def create_run_manifest(
    run_directory: str | Path,
    run_id: str,
    *,
    input_file: str | Path | None = None,
    requested_models: list[str] | None = None,
    request_options: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Create the initial manifest for a new pipeline run.

    Args:
        run_directory: Root directory assigned to the pipeline run.
        run_id: Unique identifier assigned to the pipeline run.
        input_file: Optional validated input file associated with the run.
        requested_models: Optional models requested for execution.
        request_options: Optional pipeline settings supplied by the caller.
        overwrite: Whether an existing manifest may be replaced.

    Returns:
        Newly created run-manifest dictionary.

    Raises:
        RunManifestError: If the run ID is empty, the manifest already exists,
            or the manifest cannot be created.
    """
    cleaned_run_id = validate_run_id(
        run_id
    )

    expected_run_directory = validate_run_directory(
        cleaned_run_id
    )

    supplied_run_directory = Path(
        run_directory
    ).expanduser().resolve()

    if supplied_run_directory != expected_run_directory:
        raise RunManifestError(
            "The supplied run directory does not match the run ID.",
            error_code="RUN_DIRECTORY_MISMATCH",
            details={
                "run_id": cleaned_run_id,
                "supplied_directory": str(
                    supplied_run_directory
                ),
                "expected_directory": str(
                    expected_run_directory
                ),
            },
        )

    resolved_run_directory = expected_run_directory

    manifest_path = get_manifest_path(
        resolved_run_directory
    )

    if manifest_path.exists() and not overwrite:
        raise RunManifestError(
            "A manifest already exists for this pipeline run.",
            error_code="MANIFEST_ALREADY_EXISTS",
            details={
                "run_id": cleaned_run_id,
                "manifest_path": str(manifest_path),
            },
        )

    input_metadata: dict[str, Any] | None = None

    if input_file is not None:
        input_metadata = get_file_metadata(
            input_file,
            include_checksum=True,
        )

    current_timestamp = get_current_utc_timestamp()

    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "run_id": cleaned_run_id,
        "run_directory": str(
            resolved_run_directory
        ),
        "status": "created",
        "created_at_utc": current_timestamp,
        "updated_at_utc": current_timestamp,
        "started_at_utc": None,
        "completed_at_utc": None,
        "input": input_metadata,
        "request": {
            "requested_models": list(
                requested_models or []
            ),
            "options": dict(
                request_options or {}
            ),
        },
        "stages": {},
        "outputs": [],
        "errors": [],
        "summary": {},
        "events": [],
    }

    append_manifest_event(
        manifest,
        event_type="run_created",
        message="Pipeline run manifest created.",
        details={
            "run_id": cleaned_run_id,
        },
    )

    save_run_manifest(
        resolved_run_directory,
        manifest,
    )

    return manifest


def load_run_manifest(
    run_directory: str | Path,
) -> dict[str, Any]:
    """
    Load and validate a pipeline-run manifest.

    Args:
        run_directory: Root directory containing manifest.json.

    Returns:
        Validated manifest dictionary.

    Raises:
        RunManifestError: If the manifest is missing, unreadable, or invalid.
    """
    manifest_path = get_manifest_path(
        run_directory
    )

    if not manifest_path.exists():
        raise RunManifestError(
            "The pipeline-run manifest does not exist.",
            error_code="MANIFEST_NOT_FOUND",
            details={
                "manifest_path": str(manifest_path),
            },
        )

    try:
        manifest = read_json_file(
            manifest_path
        )

    except (
        FileProcessingError,
        InputValidationError,
    ) as error:
        raise RunManifestError(
            "Unable to load the pipeline-run manifest.",
            error_code="MANIFEST_READ_FAILED",
            details={
                "manifest_path": str(manifest_path),
                "original_error": str(error),
            },
        ) from error

    if not isinstance(manifest, dict):
        raise RunManifestError(
            "The pipeline-run manifest must contain a JSON object.",
            error_code="INVALID_MANIFEST_TYPE",
            details={
                "manifest_path": str(manifest_path),
                "received_type": type(manifest).__name__,
            },
        )

    validate_manifest_structure(
        manifest
    )

    return manifest


def save_run_manifest(
    run_directory: str | Path,
    manifest: Mapping[str, Any],
) -> Path:
    """
    Validate and save a pipeline-run manifest.

    The file service writes the JSON through an atomic temporary-file
    replacement to reduce the risk of partial manifest files.

    Args:
        run_directory: Root directory receiving manifest.json.
        manifest: Manifest data that must be saved.

    Returns:
        Path of the saved manifest file.

    Raises:
        RunManifestError: If the manifest is invalid or cannot be written.
    """
    manifest_dictionary = dict(
        manifest
    )

    validate_manifest_structure(
        manifest_dictionary
    )

    manifest_dictionary["updated_at_utc"] = (
        get_current_utc_timestamp()
    )

    manifest_path = get_manifest_path(
        run_directory
    )

    try:
        return write_json_file(
            data=manifest_dictionary,
            file_path=manifest_path,
            overwrite=True,
        )

    except (
        FileProcessingError,
        InputValidationError,
    ) as error:
        raise RunManifestError(
            "Unable to save the pipeline-run manifest.",
            error_code="MANIFEST_WRITE_FAILED",
            details={
                "manifest_path": str(manifest_path),
                "original_error": str(error),
            },
        ) from error


def set_run_status(
    run_directory: str | Path,
    status: str,
    *,
    message: str | None = None,
) -> dict[str, Any]:
    """
    Update the overall status of a pipeline run.

    Starting a run records its first start timestamp. Completing, failing, or
    cancelling a run records its completion timestamp.

    Args:
        run_directory: Root directory containing the run manifest.
        status: New pipeline-run status.
        message: Optional event description.

    Returns:
        Updated run-manifest dictionary.
    """
    validated_status = validate_run_status(
        status
    )

    manifest = load_run_manifest(
        run_directory
    )

    previous_status = manifest["status"]
    current_timestamp = get_current_utc_timestamp()

    manifest["status"] = validated_status

    if (
        validated_status == "running"
        and manifest.get("started_at_utc") is None
    ):
        manifest["started_at_utc"] = (
            current_timestamp
        )

    if validated_status in {
        "completed",
        "failed",
        "cancelled",
    }:
        manifest["completed_at_utc"] = (
            current_timestamp
        )

    append_manifest_event(
        manifest,
        event_type="run_status_changed",
        message=(
            message
            or (
                f"Run status changed from "
                f"'{previous_status}' to "
                f"'{validated_status}'."
            )
        ),
        details={
            "previous_status": previous_status,
            "new_status": validated_status,
        },
    )

    save_run_manifest(
        run_directory,
        manifest,
    )

    return manifest


def start_pipeline_stage(
    run_directory: str | Path,
    stage_name: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Mark a pipeline stage as running.

    Args:
        run_directory: Root directory containing the run manifest.
        stage_name: Unique name of the pipeline stage.
        metadata: Optional stage configuration or context.

    Returns:
        Updated run-manifest dictionary.

    Raises:
        RunManifestError: If the stage name is empty.
    """
    cleaned_stage_name = stage_name.strip()

    if not cleaned_stage_name:
        raise RunManifestError(
            "The pipeline stage name cannot be empty.",
            error_code="EMPTY_STAGE_NAME",
        )

    manifest = load_run_manifest(
        run_directory
    )

    existing_stage = manifest["stages"].get(
        cleaned_stage_name,
        {},
    )

    stage_record = {
        "status": "running",
        "started_at_utc": (
            existing_stage.get("started_at_utc")
            or get_current_utc_timestamp()
        ),
        "completed_at_utc": None,
        "metadata": {
            **existing_stage.get(
                "metadata",
                {},
            ),
            **dict(metadata or {}),
        },
        "metrics": existing_stage.get(
            "metrics",
            {},
        ),
        "outputs": existing_stage.get(
            "outputs",
            [],
        ),
        "error": None,
    }

    manifest["stages"][
        cleaned_stage_name
    ] = stage_record

    append_manifest_event(
        manifest,
        event_type="stage_started",
        message=(
            f"Pipeline stage "
            f"'{cleaned_stage_name}' started."
        ),
        details={
            "stage_name": cleaned_stage_name,
        },
    )

    save_run_manifest(
        run_directory,
        manifest,
    )

    return manifest


def complete_pipeline_stage(
    run_directory: str | Path,
    stage_name: str,
    *,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Mark a pipeline stage as successfully completed.

    Args:
        run_directory: Root directory containing the run manifest.
        stage_name: Name of the completed pipeline stage.
        metrics: Optional metrics generated by the stage.

    Returns:
        Updated run-manifest dictionary.

    Raises:
        RunManifestError: If the stage was never registered or started.
    """
    cleaned_stage_name = stage_name.strip()

    manifest = load_run_manifest(
        run_directory
    )

    if cleaned_stage_name not in manifest["stages"]:
        raise RunManifestError(
            "The requested pipeline stage is not registered.",
            error_code="STAGE_NOT_FOUND",
            details={
                "stage_name": cleaned_stage_name,
            },
        )

    stage_record = manifest["stages"][
        cleaned_stage_name
    ]

    stage_record["status"] = "completed"
    stage_record["completed_at_utc"] = (
        get_current_utc_timestamp()
    )
    stage_record["metrics"] = {
        **stage_record.get("metrics", {}),
        **dict(metrics or {}),
    }
    stage_record["error"] = None

    append_manifest_event(
        manifest,
        event_type="stage_completed",
        message=(
            f"Pipeline stage "
            f"'{cleaned_stage_name}' completed."
        ),
        details={
            "stage_name": cleaned_stage_name,
        },
    )

    save_run_manifest(
        run_directory,
        manifest,
    )

    return manifest


def skip_pipeline_stage(
    run_directory: str | Path,
    stage_name: str,
    *,
    reason: str,
) -> dict[str, Any]:
    """
    Mark a pipeline stage as skipped.

    Args:
        run_directory: Root directory containing the run manifest.
        stage_name: Name of the skipped stage.
        reason: Human-readable explanation for skipping the stage.

    Returns:
        Updated run-manifest dictionary.
    """
    cleaned_stage_name = stage_name.strip()

    if not cleaned_stage_name:
        raise RunManifestError(
            "The pipeline stage name cannot be empty.",
            error_code="EMPTY_STAGE_NAME",
        )

    manifest = load_run_manifest(
        run_directory
    )

    manifest["stages"][cleaned_stage_name] = {
        "status": "skipped",
        "started_at_utc": None,
        "completed_at_utc": (
            get_current_utc_timestamp()
        ),
        "metadata": {
            "skip_reason": reason,
        },
        "metrics": {},
        "outputs": [],
        "error": None,
    }

    append_manifest_event(
        manifest,
        event_type="stage_skipped",
        message=(
            f"Pipeline stage "
            f"'{cleaned_stage_name}' was skipped."
        ),
        details={
            "stage_name": cleaned_stage_name,
            "reason": reason,
        },
    )

    save_run_manifest(
        run_directory,
        manifest,
    )

    return manifest


def convert_exception_to_record(
    error: Exception,
) -> dict[str, Any]:
    """
    Convert a Python exception into manifest-compatible structured data.

    Custom backend exceptions retain their error codes and details. Unknown
    exceptions are stored using their class name and readable message.

    Args:
        error: Exception that must be recorded.

    Returns:
        Structured error dictionary.
    """
    if isinstance(
        error,
        TelecomBackendError,
    ):
        return error.to_dict()

    return {
        "error_type": type(error).__name__,
        "message": str(error),
        "error_code": None,
        "details": {},
    }


def fail_pipeline_stage(
    run_directory: str | Path,
    stage_name: str,
    error: Exception,
    *,
    fail_run: bool = True,
) -> dict[str, Any]:
    """
    Mark a pipeline stage as failed and record its exception.

    Args:
        run_directory: Root directory containing the run manifest.
        stage_name: Name of the failed pipeline stage.
        error: Exception responsible for the failure.
        fail_run: Whether the overall pipeline run should also be marked failed.

    Returns:
        Updated run-manifest dictionary.
    """
    cleaned_stage_name = stage_name.strip()

    if not cleaned_stage_name:
        raise RunManifestError(
            "The pipeline stage name cannot be empty.",
            error_code="EMPTY_STAGE_NAME",
        )

    manifest = load_run_manifest(
        run_directory
    )

    error_record = convert_exception_to_record(
        error
    )

    existing_stage = manifest["stages"].get(
        cleaned_stage_name,
        {},
    )

    manifest["stages"][cleaned_stage_name] = {
        "status": "failed",
        "started_at_utc": (
            existing_stage.get("started_at_utc")
        ),
        "completed_at_utc": (
            get_current_utc_timestamp()
        ),
        "metadata": existing_stage.get(
            "metadata",
            {},
        ),
        "metrics": existing_stage.get(
            "metrics",
            {},
        ),
        "outputs": existing_stage.get(
            "outputs",
            [],
        ),
        "error": error_record,
    }

    manifest["errors"].append(
        {
            "timestamp_utc": (
                get_current_utc_timestamp()
            ),
            "stage_name": cleaned_stage_name,
            **error_record,
        }
    )

    if fail_run:
        manifest["status"] = "failed"
        manifest["completed_at_utc"] = (
            get_current_utc_timestamp()
        )

    append_manifest_event(
        manifest,
        event_type="stage_failed",
        message=(
            f"Pipeline stage "
            f"'{cleaned_stage_name}' failed."
        ),
        details={
            "stage_name": cleaned_stage_name,
            "error_type": (
                error_record["error_type"]
            ),
            "error_code": (
                error_record["error_code"]
            ),
        },
    )

    save_run_manifest(
        run_directory,
        manifest,
    )

    return manifest


def record_output_file(
    run_directory: str | Path,
    output_file: str | Path,
    *,
    stage_name: str,
    output_type: str,
    include_checksum: bool = False,
) -> dict[str, Any]:
    """
    Add a generated output file to the run manifest.

    Args:
        run_directory: Root directory containing the run manifest.
        output_file: Generated file that must be recorded.
        stage_name: Pipeline stage that generated the file.
        output_type: Role of the output, such as dataset, plot, model, or report.
        include_checksum: Whether a SHA-256 checksum should be calculated.

    Returns:
        Updated run-manifest dictionary.
    """
    manifest = load_run_manifest(
        run_directory
    )

    output_metadata = get_file_metadata(
        output_file,
        include_checksum=include_checksum,
    )

    output_record = {
        "stage_name": stage_name,
        "output_type": output_type,
        "recorded_at_utc": (
            get_current_utc_timestamp()
        ),
        **output_metadata,
    }

    existing_paths = {
        output.get("file_path")
        for output in manifest["outputs"]
    }

    if output_record["file_path"] not in existing_paths:
        manifest["outputs"].append(
            output_record
        )

    if stage_name in manifest["stages"]:
        stage_outputs = manifest["stages"][
            stage_name
        ].setdefault(
            "outputs",
            [],
        )

        stage_output_paths = {
            output.get("file_path")
            for output in stage_outputs
        }

        if (
            output_record["file_path"]
            not in stage_output_paths
        ):
            stage_outputs.append(
                output_record
            )

    append_manifest_event(
        manifest,
        event_type="output_recorded",
        message=(
            f"Output file recorded for "
            f"stage '{stage_name}'."
        ),
        details={
            "stage_name": stage_name,
            "output_type": output_type,
            "file_path": (
                output_record["file_path"]
            ),
        },
    )

    save_run_manifest(
        run_directory,
        manifest,
    )

    return manifest


def update_run_summary(
    run_directory: str | Path,
    summary_values: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Add or update final summary values in the run manifest.

    Args:
        run_directory: Root directory containing the run manifest.
        summary_values: Summary fields that must be added or updated.

    Returns:
        Updated run-manifest dictionary.
    """
    manifest = load_run_manifest(
        run_directory
    )

    manifest["summary"] = {
        **manifest.get("summary", {}),
        **dict(summary_values),
    }

    append_manifest_event(
        manifest,
        event_type="summary_updated",
        message="Pipeline run summary updated.",
        details={
            "updated_fields": sorted(
                summary_values.keys()
            ),
        },
    )

    save_run_manifest(
        run_directory,
        manifest,
    )

    return manifest