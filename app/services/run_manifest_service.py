"""
Run-manifest persistence for the telecom anomaly detection backend.

Manifest V2 stores the technical lifecycle of one long-lived backend run.

The hierarchy is:

    run
      ├── attempts
      │     └── attempt
      │           └── stages
      │                 └── stage
      │                       └── executions
      └── inference invocations

A training attempt is treated as an immutable dataset/configuration lineage by
the higher-level run-management layer. A failed stage may later receive another
execution record, while previously written execution history remains preserved.

This service is intentionally persistence-focused. It validates and mutates
manifest records, but it does not decide whether a stage retry is allowed,
resolve prerequisite artifacts, create attempt/stage directories, execute ML
modules, or implement run locking. Those responsibilities belong to the
run-management and workflow layers.
"""

import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import settings
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
    validate_run_directory,
    validate_run_id,
)


MANIFEST_FILENAME = "manifest.json"
MANIFEST_VERSION = settings.run_manifest_version
WORKFLOW_SCHEMA_VERSION = settings.workflow_schema_version

TRAINING_STAGE_NAMES: tuple[str, ...] = (
    "part_a",
    "part_b",
    "part_c",
    "part_d",
    "inference_bootstrap",
)

VALID_RUN_STATUSES = {
    "created",
    "running",
    "partially_completed",
    "completed",
    "failed",
    "cancelled",
}

VALID_ATTEMPT_STATUSES = {
    "created",
    "running",
    "partially_completed",
    "completed",
    "failed",
    "cancelled",
}

VALID_STAGE_STATUSES = {
    "pending",
    "running",
    "completed",
    "failed",
}

VALID_EXECUTION_STATUSES = {
    "running",
    "completed",
    "failed",
    "cancelled",
}

VALID_INFERENCE_INVOCATION_STATUSES = {
    "running",
    "completed",
    "failed",
    "cancelled",
}

RECORD_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
)


# ---------------------------------------------------------------------------
# GENERAL HELPERS
# ---------------------------------------------------------------------------


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
    Build the manifest-file path for a backend run.

    Args:
        run_directory: Root directory of the backend run.

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


def validate_record_id(
    value: str,
    *,
    field_name: str,
) -> str:
    """
    Validate an attempt, execution, or invocation identifier.

    Args:
        value: Identifier to validate.
        field_name: Human-readable field name used in errors.

    Returns:
        Validated identifier.

    Raises:
        RunManifestError: If the identifier is empty or unsafe.
    """
    cleaned_value = value.strip()

    if not cleaned_value:
        raise RunManifestError(
            f"{field_name} cannot be empty.",
            error_code="EMPTY_MANIFEST_RECORD_ID",
            details={
                "field_name": field_name,
            },
        )

    if cleaned_value in {".", ".."}:
        raise RunManifestError(
            f"{field_name} is invalid.",
            error_code="INVALID_MANIFEST_RECORD_ID",
            details={
                "field_name": field_name,
                "received_value": cleaned_value,
            },
        )

    if not RECORD_ID_PATTERN.fullmatch(
        cleaned_value
    ):
        raise RunManifestError(
            f"{field_name} contains unsupported characters.",
            error_code="INVALID_MANIFEST_RECORD_ID",
            details={
                "field_name": field_name,
                "received_value": cleaned_value,
                "maximum_length": 128,
            },
        )

    return cleaned_value


def validate_stage_name(
    stage_name: str,
) -> str:
    """
    Validate one canonical training-stage name.

    Returns:
        Validated stage name.

    Raises:
        RunManifestError: If the stage is unsupported.
    """
    cleaned_stage_name = (
        stage_name.strip().lower()
    )

    if (
        cleaned_stage_name
        not in TRAINING_STAGE_NAMES
    ):
        raise RunManifestError(
            "The requested training stage is unsupported.",
            error_code="INVALID_STAGE_NAME",
            details={
                "received_stage": stage_name,
                "allowed_stages": list(
                    TRAINING_STAGE_NAMES
                ),
            },
        )

    return cleaned_stage_name


def _validate_status(
    status: str,
    *,
    allowed_statuses: set[str],
    error_code: str,
    status_type: str,
) -> str:
    """
    Shared status validator.
    """
    normalised_status = (
        status.strip().lower()
    )

    if (
        normalised_status
        not in allowed_statuses
    ):
        raise RunManifestError(
            f"Unsupported {status_type} status '{status}'.",
            error_code=error_code,
            details={
                "received_status": status,
                "allowed_statuses": sorted(
                    allowed_statuses
                ),
            },
        )

    return normalised_status


def validate_run_status(
    status: str,
) -> str:
    return _validate_status(
        status,
        allowed_statuses=VALID_RUN_STATUSES,
        error_code="INVALID_RUN_STATUS",
        status_type="run",
    )


def validate_attempt_status(
    status: str,
) -> str:
    return _validate_status(
        status,
        allowed_statuses=VALID_ATTEMPT_STATUSES,
        error_code="INVALID_ATTEMPT_STATUS",
        status_type="attempt",
    )


def validate_stage_status(
    status: str,
) -> str:
    return _validate_status(
        status,
        allowed_statuses=VALID_STAGE_STATUSES,
        error_code="INVALID_STAGE_STATUS",
        status_type="stage",
    )


def validate_execution_status(
    status: str,
) -> str:
    return _validate_status(
        status,
        allowed_statuses=VALID_EXECUTION_STATUSES,
        error_code="INVALID_EXECUTION_STATUS",
        status_type="execution",
    )


def validate_inference_invocation_status(
    status: str,
) -> str:
    return _validate_status(
        status,
        allowed_statuses=(
            VALID_INFERENCE_INVOCATION_STATUSES
        ),
        error_code="INVALID_INFERENCE_INVOCATION_STATUS",
        status_type="inference invocation",
    )


def convert_exception_to_record(
    error: Exception,
) -> dict[str, Any]:
    """
    Convert an exception into manifest-compatible structured data.

    Custom backend exceptions retain their error codes and details. Unknown
    exceptions are stored using their class name and readable message.
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


# ---------------------------------------------------------------------------
# MANIFEST STRUCTURE
# ---------------------------------------------------------------------------


def _new_stage_record() -> dict[str, Any]:
    """
    Build an empty canonical stage record for a new attempt.
    """
    return {
        "status": "pending",
        "active_execution_id": None,
        "successful_execution_id": None,
        "executions": {},
    }


def _new_attempt_stage_map() -> dict[str, dict[str, Any]]:
    """
    Build the complete A-D + inference-bootstrap stage map.
    """
    return {
        stage_name: _new_stage_record()
        for stage_name in TRAINING_STAGE_NAMES
    }


def validate_manifest_structure(
    manifest: Mapping[str, Any],
) -> None:
    """
    Validate the minimum required structure of a Manifest V2 run record.

    This performs structural validation rather than enforcing higher-level
    workflow lineage rules.
    """
    required_fields = {
        "manifest_version",
        "workflow_schema_version",
        "run_id",
        "run_directory",
        "run_metadata",
        "status",
        "created_at_utc",
        "updated_at_utc",
        "latest_attempt_id",
        "active_inference_attempt_id",
        "inference_ready",
        "attempts",
        "inference",
        "outputs",
        "errors",
        "summary",
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

    if (
        manifest["manifest_version"]
        != MANIFEST_VERSION
    ):
        raise RunManifestError(
            "The run manifest version is unsupported.",
            error_code="UNSUPPORTED_MANIFEST_VERSION",
            details={
                "received_version": (
                    manifest["manifest_version"]
                ),
                "supported_version": (
                    MANIFEST_VERSION
                ),
            },
        )

    if (
        manifest["workflow_schema_version"]
        != WORKFLOW_SCHEMA_VERSION
    ):
        raise RunManifestError(
            "The workflow schema version stored in the manifest is unsupported.",
            error_code="UNSUPPORTED_WORKFLOW_SCHEMA_VERSION",
            details={
                "received_version": (
                    manifest[
                        "workflow_schema_version"
                    ]
                ),
                "supported_version": (
                    WORKFLOW_SCHEMA_VERSION
                ),
            },
        )

    if not isinstance(
        manifest["run_id"],
        str,
    ):
        raise RunManifestError(
            "The manifest run ID must be a string.",
            error_code="INVALID_MANIFEST_RUN_ID",
        )

    validate_run_id(
        manifest["run_id"]
    )

    validate_run_status(
        str(manifest["status"])
    )

    if not isinstance(
        manifest["run_metadata"],
        dict,
    ):
        raise RunManifestError(
            "The manifest run_metadata field must be a dictionary.",
            error_code="INVALID_MANIFEST_RUN_METADATA",
        )

    if (
        manifest["latest_attempt_id"]
        is not None
        and not isinstance(
            manifest["latest_attempt_id"],
            str,
        )
    ):
        raise RunManifestError(
            "latest_attempt_id must be a string or null.",
            error_code="INVALID_LATEST_ATTEMPT_ID",
        )

    if (
        manifest["active_inference_attempt_id"]
        is not None
        and not isinstance(
            manifest[
                "active_inference_attempt_id"
            ],
            str,
        )
    ):
        raise RunManifestError(
            "active_inference_attempt_id must be a string or null.",
            error_code="INVALID_ACTIVE_INFERENCE_ATTEMPT_ID",
        )

    if not isinstance(
        manifest["inference_ready"],
        bool,
    ):
        raise RunManifestError(
            "inference_ready must be a boolean.",
            error_code="INVALID_INFERENCE_READY_FLAG",
        )

    if not isinstance(
        manifest["attempts"],
        dict,
    ):
        raise RunManifestError(
            "The manifest attempts field must be a dictionary.",
            error_code="INVALID_MANIFEST_ATTEMPTS",
        )

    if not isinstance(
        manifest["inference"],
        dict,
    ):
        raise RunManifestError(
            "The manifest inference field must be a dictionary.",
            error_code="INVALID_MANIFEST_INFERENCE",
        )

    if not isinstance(
        manifest["inference"].get(
            "invocations"
        ),
        dict,
    ):
        raise RunManifestError(
            "The manifest inference.invocations field must be a dictionary.",
            error_code="INVALID_MANIFEST_INFERENCE_INVOCATIONS",
        )

    if not isinstance(
        manifest["outputs"],
        list,
    ):
        raise RunManifestError(
            "The manifest outputs field must be a list.",
            error_code="INVALID_MANIFEST_OUTPUTS",
        )

    if not isinstance(
        manifest["errors"],
        list,
    ):
        raise RunManifestError(
            "The manifest errors field must be a list.",
            error_code="INVALID_MANIFEST_ERRORS",
        )

    if not isinstance(
        manifest["summary"],
        dict,
    ):
        raise RunManifestError(
            "The manifest summary field must be a dictionary.",
            error_code="INVALID_MANIFEST_SUMMARY",
        )

    if not isinstance(
        manifest["events"],
        list,
    ):
        raise RunManifestError(
            "The manifest events field must be a list.",
            error_code="INVALID_MANIFEST_EVENTS",
        )

    for (
        attempt_id,
        attempt_record,
    ) in manifest["attempts"].items():
        validate_record_id(
            attempt_id,
            field_name="attempt_id",
        )

        if not isinstance(
            attempt_record,
            dict,
        ):
            raise RunManifestError(
                "Every attempt record must be a dictionary.",
                error_code="INVALID_ATTEMPT_RECORD",
                details={
                    "attempt_id": attempt_id,
                },
            )

        validate_attempt_status(
            str(
                attempt_record.get(
                    "status",
                    "",
                )
            )
        )

        stages = attempt_record.get(
            "stages"
        )

        if not isinstance(
            stages,
            dict,
        ):
            raise RunManifestError(
                "Every attempt must contain a stages dictionary.",
                error_code="INVALID_ATTEMPT_STAGES",
                details={
                    "attempt_id": attempt_id,
                },
            )

        missing_stages = [
            stage_name
            for stage_name in TRAINING_STAGE_NAMES
            if stage_name not in stages
        ]

        if missing_stages:
            raise RunManifestError(
                "An attempt is missing canonical training stages.",
                error_code="MISSING_ATTEMPT_STAGES",
                details={
                    "attempt_id": attempt_id,
                    "missing_stages": (
                        missing_stages
                    ),
                },
            )

        for (
            stage_name,
            stage_record,
        ) in stages.items():
            validate_stage_name(
                stage_name
            )

            if not isinstance(
                stage_record,
                dict,
            ):
                raise RunManifestError(
                    "Every stage record must be a dictionary.",
                    error_code="INVALID_STAGE_RECORD",
                    details={
                        "attempt_id": attempt_id,
                        "stage_name": stage_name,
                    },
                )

            validate_stage_status(
                str(
                    stage_record.get(
                        "status",
                        "",
                    )
                )
            )

            executions = stage_record.get(
                "executions"
            )

            if not isinstance(
                executions,
                dict,
            ):
                raise RunManifestError(
                    "Every stage must contain an executions dictionary.",
                    error_code="INVALID_STAGE_EXECUTIONS",
                    details={
                        "attempt_id": attempt_id,
                        "stage_name": stage_name,
                    },
                )

            for (
                execution_id,
                execution_record,
            ) in executions.items():
                validate_record_id(
                    execution_id,
                    field_name="execution_id",
                )

                if not isinstance(
                    execution_record,
                    dict,
                ):
                    raise RunManifestError(
                        "Every execution record must be a dictionary.",
                        error_code="INVALID_EXECUTION_RECORD",
                        details={
                            "attempt_id": attempt_id,
                            "stage_name": stage_name,
                            "execution_id": (
                                execution_id
                            ),
                        },
                    )

                validate_execution_status(
                    str(
                        execution_record.get(
                            "status",
                            "",
                        )
                    )
                )


# ---------------------------------------------------------------------------
# EVENT / LOAD / SAVE
# ---------------------------------------------------------------------------


def append_manifest_event(
    manifest: dict[str, Any],
    event_type: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> None:
    """
    Add a timestamped event to a manifest in memory.
    """
    cleaned_event_type = (
        event_type.strip()
    )

    if not cleaned_event_type:
        raise RunManifestError(
            "Manifest event_type cannot be empty.",
            error_code="EMPTY_MANIFEST_EVENT_TYPE",
        )

    event = {
        "timestamp_utc": (
            get_current_utc_timestamp()
        ),
        "event_type": cleaned_event_type,
        "message": message,
        "details": dict(
            details or {}
        ),
    }

    manifest.setdefault(
        "events",
        [],
    ).append(event)


def create_run_manifest(
    run_directory: str | Path,
    run_id: str,
    *,
    run_metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Create the Manifest V2 record for a new backend run.

    Input datasets and exact request snapshots are attempt-level information
    and are therefore not stored directly on the run root.
    """
    cleaned_run_id = validate_run_id(
        run_id
    )

    expected_run_directory = (
        validate_run_directory(
            cleaned_run_id
        )
    )

    supplied_run_directory = Path(
        run_directory
    ).expanduser().resolve()

    if (
        supplied_run_directory
        != expected_run_directory
    ):
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

    manifest_path = get_manifest_path(
        expected_run_directory
    )

    if (
        manifest_path.exists()
        and not overwrite
    ):
        raise RunManifestError(
            "A manifest already exists for this backend run.",
            error_code="MANIFEST_ALREADY_EXISTS",
            details={
                "run_id": cleaned_run_id,
                "manifest_path": str(
                    manifest_path
                ),
            },
        )

    current_timestamp = (
        get_current_utc_timestamp()
    )

    manifest: dict[str, Any] = {
        "manifest_version": (
            MANIFEST_VERSION
        ),
        "workflow_schema_version": (
            WORKFLOW_SCHEMA_VERSION
        ),
        "run_id": cleaned_run_id,
        "run_directory": str(
            expected_run_directory
        ),
        "run_metadata": dict(
            run_metadata or {}
        ),
        "status": "created",
        "created_at_utc": (
            current_timestamp
        ),
        "updated_at_utc": (
            current_timestamp
        ),
        "started_at_utc": None,
        "completed_at_utc": None,
        "latest_attempt_id": None,
        "active_inference_attempt_id": (
            None
        ),
        "inference_ready": False,
        "attempts": {},
        "inference": {
            "invocations": {},
        },
        "outputs": [],
        "errors": [],
        "summary": {},
        "events": [],
    }

    append_manifest_event(
        manifest,
        event_type="run_created",
        message="Backend run manifest created.",
        details={
            "run_id": cleaned_run_id,
            "manifest_version": (
                MANIFEST_VERSION
            ),
        },
    )

    save_run_manifest(
        expected_run_directory,
        manifest,
    )

    return manifest


def load_run_manifest(
    run_directory: str | Path,
) -> dict[str, Any]:
    """
    Load and validate a Manifest V2 run record.
    """
    manifest_path = get_manifest_path(
        run_directory
    )

    if not manifest_path.exists():
        raise RunManifestError(
            "The backend run manifest does not exist.",
            error_code="MANIFEST_NOT_FOUND",
            details={
                "manifest_path": str(
                    manifest_path
                ),
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
            "Unable to load the backend run manifest.",
            error_code="MANIFEST_READ_FAILED",
            details={
                "manifest_path": str(
                    manifest_path
                ),
                "original_error": str(
                    error
                ),
            },
        ) from error

    if not isinstance(
        manifest,
        dict,
    ):
        raise RunManifestError(
            "The backend run manifest must contain a JSON object.",
            error_code="INVALID_MANIFEST_TYPE",
            details={
                "manifest_path": str(
                    manifest_path
                ),
                "received_type": (
                    type(manifest).__name__
                ),
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
    Validate and atomically save a Manifest V2 run record.
    """
    manifest_dictionary = dict(
        manifest
    )

    manifest_dictionary[
        "updated_at_utc"
    ] = get_current_utc_timestamp()

    validate_manifest_structure(
        manifest_dictionary
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
            "Unable to save the backend run manifest.",
            error_code="MANIFEST_WRITE_FAILED",
            details={
                "manifest_path": str(
                    manifest_path
                ),
                "original_error": str(
                    error
                ),
            },
        ) from error


# ---------------------------------------------------------------------------
# RUN-LEVEL STATE
# ---------------------------------------------------------------------------


def set_run_status(
    run_directory: str | Path,
    status: str,
    *,
    message: str | None = None,
) -> dict[str, Any]:
    """
    Update the overall run status.
    """
    validated_status = (
        validate_run_status(
            status
        )
    )

    manifest = load_run_manifest(
        run_directory
    )

    previous_status = (
        manifest["status"]
    )
    current_timestamp = (
        get_current_utc_timestamp()
    )

    manifest["status"] = (
        validated_status
    )

    if (
        validated_status == "running"
        and manifest.get(
            "started_at_utc"
        ) is None
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

    elif validated_status in {
        "created",
        "running",
        "partially_completed",
    }:
        manifest["completed_at_utc"] = (
            None
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
            "previous_status": (
                previous_status
            ),
            "new_status": (
                validated_status
            ),
        },
    )

    save_run_manifest(
        run_directory,
        manifest,
    )

    return manifest


def update_run_metadata(
    run_directory: str | Path,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Add or update non-technical run metadata.

    Intended examples include employee code, employee name, and a user-facing
    run name. Uniqueness/business rules are deliberately outside this service.
    """
    manifest = load_run_manifest(
        run_directory
    )

    manifest["run_metadata"] = {
        **manifest.get(
            "run_metadata",
            {},
        ),
        **dict(metadata),
    }

    append_manifest_event(
        manifest,
        event_type="run_metadata_updated",
        message="Run metadata updated.",
        details={
            "updated_fields": sorted(
                metadata.keys()
            ),
        },
    )

    save_run_manifest(
        run_directory,
        manifest,
    )

    return manifest


# ---------------------------------------------------------------------------
# ATTEMPTS
# ---------------------------------------------------------------------------


def create_attempt_record(
    run_directory: str | Path,
    attempt_id: str,
    *,
    request_file: str | Path | None = None,
    input_file: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create one training-attempt record.

    The higher-level run manager decides when creating a new attempt is
    required. This function only persists the supplied attempt.
    """
    cleaned_attempt_id = (
        validate_record_id(
            attempt_id,
            field_name="attempt_id",
        )
    )

    manifest = load_run_manifest(
        run_directory
    )

    if (
        cleaned_attempt_id
        in manifest["attempts"]
    ):
        raise RunManifestError(
            "The attempt already exists in the run manifest.",
            error_code="ATTEMPT_ALREADY_EXISTS",
            details={
                "attempt_id": (
                    cleaned_attempt_id
                ),
            },
        )

    input_metadata = None

    if input_file is not None:
        input_metadata = (
            get_file_metadata(
                input_file,
                include_checksum=True,
            )
        )

    current_timestamp = (
        get_current_utc_timestamp()
    )

    manifest["attempts"][
        cleaned_attempt_id
    ] = {
        "attempt_id": (
            cleaned_attempt_id
        ),
        "status": "created",
        "created_at_utc": (
            current_timestamp
        ),
        "updated_at_utc": (
            current_timestamp
        ),
        "started_at_utc": None,
        "completed_at_utc": None,
        "request_file": (
            str(
                Path(
                    request_file
                ).expanduser().resolve()
            )
            if request_file is not None
            else None
        ),
        "input": input_metadata,
        "metadata": dict(
            metadata or {}
        ),
        "stages": (
            _new_attempt_stage_map()
        ),
        "errors": [],
        "summary": {},
    }

    manifest["latest_attempt_id"] = (
        cleaned_attempt_id
    )

    append_manifest_event(
        manifest,
        event_type="attempt_created",
        message=(
            f"Training attempt "
            f"'{cleaned_attempt_id}' created."
        ),
        details={
            "attempt_id": (
                cleaned_attempt_id
            ),
        },
    )

    save_run_manifest(
        run_directory,
        manifest,
    )

    return manifest


def set_attempt_status(
    run_directory: str | Path,
    attempt_id: str,
    status: str,
    *,
    message: str | None = None,
) -> dict[str, Any]:
    """
    Update one training attempt's lifecycle status.
    """
    cleaned_attempt_id = (
        validate_record_id(
            attempt_id,
            field_name="attempt_id",
        )
    )
    validated_status = (
        validate_attempt_status(
            status
        )
    )

    manifest = load_run_manifest(
        run_directory
    )

    if (
        cleaned_attempt_id
        not in manifest["attempts"]
    ):
        raise RunManifestError(
            "The requested training attempt is not registered.",
            error_code="ATTEMPT_NOT_FOUND",
            details={
                "attempt_id": (
                    cleaned_attempt_id
                ),
            },
        )

    attempt = manifest["attempts"][
        cleaned_attempt_id
    ]

    previous_status = (
        attempt["status"]
    )
    current_timestamp = (
        get_current_utc_timestamp()
    )

    attempt["status"] = (
        validated_status
    )
    attempt["updated_at_utc"] = (
        current_timestamp
    )

    if (
        validated_status == "running"
        and attempt.get(
            "started_at_utc"
        ) is None
    ):
        attempt["started_at_utc"] = (
            current_timestamp
        )

    if validated_status in {
        "completed",
        "failed",
        "cancelled",
    }:
        attempt["completed_at_utc"] = (
            current_timestamp
        )

    elif validated_status in {
        "created",
        "running",
        "partially_completed",
    }:
        attempt["completed_at_utc"] = (
            None
        )

    append_manifest_event(
        manifest,
        event_type="attempt_status_changed",
        message=(
            message
            or (
                f"Attempt '{cleaned_attempt_id}' "
                f"status changed from "
                f"'{previous_status}' to "
                f"'{validated_status}'."
            )
        ),
        details={
            "attempt_id": (
                cleaned_attempt_id
            ),
            "previous_status": (
                previous_status
            ),
            "new_status": (
                validated_status
            ),
        },
    )

    save_run_manifest(
        run_directory,
        manifest,
    )

    return manifest


# ---------------------------------------------------------------------------
# STAGE EXECUTIONS
# ---------------------------------------------------------------------------


def start_stage_execution(
    run_directory: str | Path,
    attempt_id: str,
    stage_name: str,
    execution_id: str,
    *,
    request_file: str | Path | None = None,
    output_directory: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Register a new execution for one canonical stage and mark it running.

    Retry/lineage permission is intentionally not decided here.
    """
    cleaned_attempt_id = (
        validate_record_id(
            attempt_id,
            field_name="attempt_id",
        )
    )
    cleaned_stage_name = (
        validate_stage_name(
            stage_name
        )
    )
    cleaned_execution_id = (
        validate_record_id(
            execution_id,
            field_name="execution_id",
        )
    )

    manifest = load_run_manifest(
        run_directory
    )

    if (
        cleaned_attempt_id
        not in manifest["attempts"]
    ):
        raise RunManifestError(
            "The requested training attempt is not registered.",
            error_code="ATTEMPT_NOT_FOUND",
            details={
                "attempt_id": (
                    cleaned_attempt_id
                ),
            },
        )

    stage = manifest["attempts"][
        cleaned_attempt_id
    ]["stages"][
        cleaned_stage_name
    ]

    if (
        cleaned_execution_id
        in stage["executions"]
    ):
        raise RunManifestError(
            "The stage execution already exists.",
            error_code="EXECUTION_ALREADY_EXISTS",
            details={
                "attempt_id": (
                    cleaned_attempt_id
                ),
                "stage_name": (
                    cleaned_stage_name
                ),
                "execution_id": (
                    cleaned_execution_id
                ),
            },
        )

    active_execution_id = (
        stage.get(
            "active_execution_id"
        )
    )

    if active_execution_id is not None:
        active_record = (
            stage["executions"].get(
                active_execution_id
            )
        )

        if (
            isinstance(
                active_record,
                dict,
            )
            and active_record.get(
                "status"
            ) == "running"
        ):
            raise RunManifestError(
                "The stage already has an active running execution.",
                error_code="ACTIVE_EXECUTION_ALREADY_EXISTS",
                details={
                    "attempt_id": (
                        cleaned_attempt_id
                    ),
                    "stage_name": (
                        cleaned_stage_name
                    ),
                    "active_execution_id": (
                        active_execution_id
                    ),
                },
            )

    current_timestamp = (
        get_current_utc_timestamp()
    )

    stage["executions"][
        cleaned_execution_id
    ] = {
        "execution_id": (
            cleaned_execution_id
        ),
        "status": "running",
        "started_at_utc": (
            current_timestamp
        ),
        "completed_at_utc": None,
        "request_file": (
            str(
                Path(
                    request_file
                ).expanduser().resolve()
            )
            if request_file is not None
            else None
        ),
        "result_file": None,
        "error_file": None,
        "output_directory": (
            str(
                Path(
                    output_directory
                ).expanduser().resolve()
            )
            if output_directory
            is not None
            else None
        ),
        "metadata": dict(
            metadata or {}
        ),
        "metrics": {},
        "outputs": [],
        "error": None,
    }

    stage["status"] = "running"
    stage["active_execution_id"] = (
        cleaned_execution_id
    )

    attempt = manifest["attempts"][
        cleaned_attempt_id
    ]
    attempt["updated_at_utc"] = (
        current_timestamp
    )

    append_manifest_event(
        manifest,
        event_type="stage_execution_started",
        message=(
            f"Execution '{cleaned_execution_id}' "
            f"started for stage "
            f"'{cleaned_stage_name}'."
        ),
        details={
            "attempt_id": (
                cleaned_attempt_id
            ),
            "stage_name": (
                cleaned_stage_name
            ),
            "execution_id": (
                cleaned_execution_id
            ),
        },
    )

    save_run_manifest(
        run_directory,
        manifest,
    )

    return manifest


def _require_active_running_stage_execution(
    *,
    attempt_id: str,
    stage_name: str,
    execution_id: str,
    stage: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> None:
    """
    Require a legal running -> terminal stage transition.

    A completed/failed/cancelled execution must never be silently rewritten
    because a caller encountered a later persistence/index exception.
    """
    if (
        execution.get("status") != "running"
        or stage.get("active_execution_id") != execution_id
    ):
        raise RunManifestError(
            "Only the active running stage execution can be finalized.",
            error_code="STAGE_EXECUTION_NOT_ACTIVE_RUNNING",
            details={
                "attempt_id": attempt_id,
                "stage_name": stage_name,
                "execution_id": execution_id,
                "execution_status": execution.get("status"),
                "active_execution_id": stage.get(
                    "active_execution_id"
                ),
            },
        )


def complete_stage_execution(
    run_directory: str | Path,
    attempt_id: str,
    stage_name: str,
    execution_id: str,
    *,
    result_file: str | Path | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Mark one stage execution as successfully completed.
    """
    (
        manifest,
        attempt,
        stage,
        execution,
        cleaned_attempt_id,
        cleaned_stage_name,
        cleaned_execution_id,
    ) = _load_stage_execution(
        run_directory,
        attempt_id,
        stage_name,
        execution_id,
    )

    _require_active_running_stage_execution(
        attempt_id=cleaned_attempt_id,
        stage_name=cleaned_stage_name,
        execution_id=cleaned_execution_id,
        stage=stage,
        execution=execution,
    )

    current_timestamp = (
        get_current_utc_timestamp()
    )

    execution["status"] = "completed"
    execution["completed_at_utc"] = (
        current_timestamp
    )
    execution["result_file"] = (
        str(
            Path(
                result_file
            ).expanduser().resolve()
        )
        if result_file is not None
        else execution.get(
            "result_file"
        )
    )
    execution["metrics"] = {
        **execution.get(
            "metrics",
            {},
        ),
        **dict(metrics or {}),
    }
    execution["error"] = None

    stage["status"] = "completed"
    stage["active_execution_id"] = None
    stage["successful_execution_id"] = (
        cleaned_execution_id
    )

    attempt["updated_at_utc"] = (
        current_timestamp
    )

    append_manifest_event(
        manifest,
        event_type="stage_execution_completed",
        message=(
            f"Execution '{cleaned_execution_id}' "
            f"completed for stage "
            f"'{cleaned_stage_name}'."
        ),
        details={
            "attempt_id": (
                cleaned_attempt_id
            ),
            "stage_name": (
                cleaned_stage_name
            ),
            "execution_id": (
                cleaned_execution_id
            ),
        },
    )

    save_run_manifest(
        run_directory,
        manifest,
    )

    return manifest


def fail_stage_execution(
    run_directory: str | Path,
    attempt_id: str,
    stage_name: str,
    execution_id: str,
    error: Exception,
    *,
    error_file: str | Path | None = None,
) -> dict[str, Any]:
    """
    Mark one stage execution as failed and preserve the structured error.
    """
    (
        manifest,
        attempt,
        stage,
        execution,
        cleaned_attempt_id,
        cleaned_stage_name,
        cleaned_execution_id,
    ) = _load_stage_execution(
        run_directory,
        attempt_id,
        stage_name,
        execution_id,
    )

    _require_active_running_stage_execution(
        attempt_id=cleaned_attempt_id,
        stage_name=cleaned_stage_name,
        execution_id=cleaned_execution_id,
        stage=stage,
        execution=execution,
    )

    current_timestamp = (
        get_current_utc_timestamp()
    )
    error_record = (
        convert_exception_to_record(
            error
        )
    )

    execution["status"] = "failed"
    execution["completed_at_utc"] = (
        current_timestamp
    )
    execution["error_file"] = (
        str(
            Path(
                error_file
            ).expanduser().resolve()
        )
        if error_file is not None
        else None
    )
    execution["error"] = (
        error_record
    )

    stage["status"] = "failed"
    stage["active_execution_id"] = None

    attempt["updated_at_utc"] = (
        current_timestamp
    )

    scoped_error = {
        "timestamp_utc": (
            current_timestamp
        ),
        "scope": "stage_execution",
        "attempt_id": (
            cleaned_attempt_id
        ),
        "stage_name": (
            cleaned_stage_name
        ),
        "execution_id": (
            cleaned_execution_id
        ),
        **error_record,
    }

    attempt.setdefault(
        "errors",
        [],
    ).append(
        scoped_error
    )

    manifest["errors"].append(
        scoped_error
    )

    append_manifest_event(
        manifest,
        event_type="stage_execution_failed",
        message=(
            f"Execution '{cleaned_execution_id}' "
            f"failed for stage "
            f"'{cleaned_stage_name}'."
        ),
        details={
            "attempt_id": (
                cleaned_attempt_id
            ),
            "stage_name": (
                cleaned_stage_name
            ),
            "execution_id": (
                cleaned_execution_id
            ),
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


def cancel_stage_execution(
    run_directory: str | Path,
    attempt_id: str,
    stage_name: str,
    execution_id: str,
    *,
    reason: str,
) -> dict[str, Any]:
    """
    Mark a running stage execution as cancelled.
    """
    (
        manifest,
        attempt,
        stage,
        execution,
        cleaned_attempt_id,
        cleaned_stage_name,
        cleaned_execution_id,
    ) = _load_stage_execution(
        run_directory,
        attempt_id,
        stage_name,
        execution_id,
    )

    _require_active_running_stage_execution(
        attempt_id=cleaned_attempt_id,
        stage_name=cleaned_stage_name,
        execution_id=cleaned_execution_id,
        stage=stage,
        execution=execution,
    )

    current_timestamp = (
        get_current_utc_timestamp()
    )

    execution["status"] = "cancelled"
    execution["completed_at_utc"] = (
        current_timestamp
    )
    execution["metadata"] = {
        **execution.get(
            "metadata",
            {},
        ),
        "cancellation_reason": (
            reason
        ),
    }

    stage["status"] = "failed"
    stage["active_execution_id"] = None

    attempt["updated_at_utc"] = (
        current_timestamp
    )

    append_manifest_event(
        manifest,
        event_type="stage_execution_cancelled",
        message=(
            f"Execution '{cleaned_execution_id}' "
            f"was cancelled for stage "
            f"'{cleaned_stage_name}'."
        ),
        details={
            "attempt_id": (
                cleaned_attempt_id
            ),
            "stage_name": (
                cleaned_stage_name
            ),
            "execution_id": (
                cleaned_execution_id
            ),
            "reason": reason,
        },
    )

    save_run_manifest(
        run_directory,
        manifest,
    )

    return manifest


def _load_stage_execution(
    run_directory: str | Path,
    attempt_id: str,
    stage_name: str,
    execution_id: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    str,
    str,
    str,
]:
    """
    Load one attempt/stage/execution tuple or raise a manifest error.
    """
    cleaned_attempt_id = (
        validate_record_id(
            attempt_id,
            field_name="attempt_id",
        )
    )
    cleaned_stage_name = (
        validate_stage_name(
            stage_name
        )
    )
    cleaned_execution_id = (
        validate_record_id(
            execution_id,
            field_name="execution_id",
        )
    )

    manifest = load_run_manifest(
        run_directory
    )

    attempt = manifest[
        "attempts"
    ].get(
        cleaned_attempt_id
    )

    if attempt is None:
        raise RunManifestError(
            "The requested training attempt is not registered.",
            error_code="ATTEMPT_NOT_FOUND",
            details={
                "attempt_id": (
                    cleaned_attempt_id
                ),
            },
        )

    stage = attempt[
        "stages"
    ][
        cleaned_stage_name
    ]

    execution = stage[
        "executions"
    ].get(
        cleaned_execution_id
    )

    if execution is None:
        raise RunManifestError(
            "The requested stage execution is not registered.",
            error_code="EXECUTION_NOT_FOUND",
            details={
                "attempt_id": (
                    cleaned_attempt_id
                ),
                "stage_name": (
                    cleaned_stage_name
                ),
                "execution_id": (
                    cleaned_execution_id
                ),
            },
        )

    return (
        manifest,
        attempt,
        stage,
        execution,
        cleaned_attempt_id,
        cleaned_stage_name,
        cleaned_execution_id,
    )


# ---------------------------------------------------------------------------
# OUTPUT REGISTRATION
# ---------------------------------------------------------------------------


def record_output_file(
    run_directory: str | Path,
    output_file: str | Path,
    *,
    attempt_id: str,
    stage_name: str,
    execution_id: str,
    output_type: str,
    include_checksum: bool = False,
) -> dict[str, Any]:
    """
    Register one generated stage-execution output.

    The file is recorded both globally and inside its exact execution record.
    """
    (
        manifest,
        _attempt,
        _stage,
        execution,
        cleaned_attempt_id,
        cleaned_stage_name,
        cleaned_execution_id,
    ) = _load_stage_execution(
        run_directory,
        attempt_id,
        stage_name,
        execution_id,
    )

    output_metadata = (
        get_file_metadata(
            output_file,
            include_checksum=(
                include_checksum
            ),
        )
    )

    output_record = {
        "attempt_id": (
            cleaned_attempt_id
        ),
        "stage_name": (
            cleaned_stage_name
        ),
        "execution_id": (
            cleaned_execution_id
        ),
        "output_type": output_type,
        "recorded_at_utc": (
            get_current_utc_timestamp()
        ),
        **output_metadata,
    }

    global_paths = {
        output.get(
            "file_path"
        )
        for output
        in manifest["outputs"]
    }

    if (
        output_record["file_path"]
        not in global_paths
    ):
        manifest["outputs"].append(
            output_record
        )

    execution_paths = {
        output.get(
            "file_path"
        )
        for output
        in execution["outputs"]
    }

    if (
        output_record["file_path"]
        not in execution_paths
    ):
        execution["outputs"].append(
            output_record
        )

    append_manifest_event(
        manifest,
        event_type="output_recorded",
        message=(
            f"Output recorded for "
            f"'{cleaned_stage_name}' "
            f"execution "
            f"'{cleaned_execution_id}'."
        ),
        details={
            "attempt_id": (
                cleaned_attempt_id
            ),
            "stage_name": (
                cleaned_stage_name
            ),
            "execution_id": (
                cleaned_execution_id
            ),
            "output_type": (
                output_type
            ),
            "file_path": (
                output_record[
                    "file_path"
                ]
            ),
        },
    )

    save_run_manifest(
        run_directory,
        manifest,
    )

    return manifest


# ---------------------------------------------------------------------------
# INFERENCE-READY POINTER
# ---------------------------------------------------------------------------


def set_active_inference_attempt(
    run_directory: str | Path,
    attempt_id: str,
) -> dict[str, Any]:
    """
    Point future Part-E inference to one training attempt.

    The run-management layer is responsible for proving that the attempt has
    successfully completed Part D and inference bootstrap before calling this
    function.
    """
    cleaned_attempt_id = (
        validate_record_id(
            attempt_id,
            field_name="attempt_id",
        )
    )

    manifest = load_run_manifest(
        run_directory
    )

    if (
        cleaned_attempt_id
        not in manifest["attempts"]
    ):
        raise RunManifestError(
            "The requested inference-ready attempt is not registered.",
            error_code="ATTEMPT_NOT_FOUND",
            details={
                "attempt_id": (
                    cleaned_attempt_id
                ),
            },
        )

    previous_attempt_id = (
        manifest.get(
            "active_inference_attempt_id"
        )
    )

    manifest[
        "active_inference_attempt_id"
    ] = cleaned_attempt_id
    manifest["inference_ready"] = True

    append_manifest_event(
        manifest,
        event_type="active_inference_attempt_changed",
        message=(
            f"Training attempt "
            f"'{cleaned_attempt_id}' is now "
            f"the active inference attempt."
        ),
        details={
            "previous_attempt_id": (
                previous_attempt_id
            ),
            "active_attempt_id": (
                cleaned_attempt_id
            ),
        },
    )

    save_run_manifest(
        run_directory,
        manifest,
    )

    return manifest


def clear_active_inference_attempt(
    run_directory: str | Path,
    *,
    reason: str,
) -> dict[str, Any]:
    """
    Remove the run's active inference pointer.
    """
    manifest = load_run_manifest(
        run_directory
    )

    previous_attempt_id = (
        manifest.get(
            "active_inference_attempt_id"
        )
    )

    manifest[
        "active_inference_attempt_id"
    ] = None
    manifest["inference_ready"] = False

    append_manifest_event(
        manifest,
        event_type="active_inference_attempt_cleared",
        message="The run is no longer marked inference-ready.",
        details={
            "previous_attempt_id": (
                previous_attempt_id
            ),
            "reason": reason,
        },
    )

    save_run_manifest(
        run_directory,
        manifest,
    )

    return manifest


# ---------------------------------------------------------------------------
# INFERENCE INVOCATIONS
# ---------------------------------------------------------------------------


def start_inference_invocation(
    run_directory: str | Path,
    invocation_id: str,
    *,
    training_attempt_id: str,
    request_file: str | Path | None = None,
    input_file: str | Path | None = None,
    output_directory: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Register one Part-E inference invocation and mark it running.
    """
    cleaned_invocation_id = (
        validate_record_id(
            invocation_id,
            field_name="invocation_id",
        )
    )
    cleaned_attempt_id = (
        validate_record_id(
            training_attempt_id,
            field_name="training_attempt_id",
        )
    )

    manifest = load_run_manifest(
        run_directory
    )

    if (
        cleaned_attempt_id
        not in manifest["attempts"]
    ):
        raise RunManifestError(
            "The training attempt for the inference invocation is unknown.",
            error_code="ATTEMPT_NOT_FOUND",
            details={
                "attempt_id": (
                    cleaned_attempt_id
                ),
            },
        )

    invocations = manifest[
        "inference"
    ][
        "invocations"
    ]

    if (
        cleaned_invocation_id
        in invocations
    ):
        raise RunManifestError(
            "The inference invocation already exists.",
            error_code="INFERENCE_INVOCATION_ALREADY_EXISTS",
            details={
                "invocation_id": (
                    cleaned_invocation_id
                ),
            },
        )

    input_metadata = None

    if input_file is not None:
        input_metadata = (
            get_file_metadata(
                input_file,
                include_checksum=True,
            )
        )

    current_timestamp = (
        get_current_utc_timestamp()
    )

    invocations[
        cleaned_invocation_id
    ] = {
        "invocation_id": (
            cleaned_invocation_id
        ),
        "training_attempt_id": (
            cleaned_attempt_id
        ),
        "status": "running",
        "started_at_utc": (
            current_timestamp
        ),
        "completed_at_utc": None,
        "request_file": (
            str(
                Path(
                    request_file
                ).expanduser().resolve()
            )
            if request_file is not None
            else None
        ),
        "input": input_metadata,
        "output_directory": (
            str(
                Path(
                    output_directory
                ).expanduser().resolve()
            )
            if output_directory
            is not None
            else None
        ),
        "result_file": None,
        "error_file": None,
        "metadata": dict(
            metadata or {}
        ),
        "metrics": {},
        "outputs": [],
        "error": None,
    }

    append_manifest_event(
        manifest,
        event_type="inference_invocation_started",
        message=(
            f"Inference invocation "
            f"'{cleaned_invocation_id}' started."
        ),
        details={
            "invocation_id": (
                cleaned_invocation_id
            ),
            "training_attempt_id": (
                cleaned_attempt_id
            ),
        },
    )

    save_run_manifest(
        run_directory,
        manifest,
    )

    return manifest


def complete_inference_invocation(
    run_directory: str | Path,
    invocation_id: str,
    *,
    result_file: str | Path | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Mark one inference invocation as completed.
    """
    (
        manifest,
        invocation,
        cleaned_invocation_id,
    ) = _load_inference_invocation(
        run_directory,
        invocation_id,
    )

    invocation["status"] = (
        "completed"
    )
    invocation["completed_at_utc"] = (
        get_current_utc_timestamp()
    )
    invocation["result_file"] = (
        str(
            Path(
                result_file
            ).expanduser().resolve()
        )
        if result_file is not None
        else invocation.get(
            "result_file"
        )
    )
    invocation["metrics"] = {
        **invocation.get(
            "metrics",
            {},
        ),
        **dict(metrics or {}),
    }
    invocation["error"] = None

    append_manifest_event(
        manifest,
        event_type="inference_invocation_completed",
        message=(
            f"Inference invocation "
            f"'{cleaned_invocation_id}' "
            f"completed."
        ),
        details={
            "invocation_id": (
                cleaned_invocation_id
            ),
        },
    )

    save_run_manifest(
        run_directory,
        manifest,
    )

    return manifest


def fail_inference_invocation(
    run_directory: str | Path,
    invocation_id: str,
    error: Exception,
    *,
    error_file: str | Path | None = None,
) -> dict[str, Any]:
    """
    Mark one inference invocation as failed and preserve its error.
    """
    (
        manifest,
        invocation,
        cleaned_invocation_id,
    ) = _load_inference_invocation(
        run_directory,
        invocation_id,
    )

    current_timestamp = (
        get_current_utc_timestamp()
    )
    error_record = (
        convert_exception_to_record(
            error
        )
    )

    invocation["status"] = "failed"
    invocation["completed_at_utc"] = (
        current_timestamp
    )
    invocation["error_file"] = (
        str(
            Path(
                error_file
            ).expanduser().resolve()
        )
        if error_file is not None
        else None
    )
    invocation["error"] = (
        error_record
    )

    manifest["errors"].append(
        {
            "timestamp_utc": (
                current_timestamp
            ),
            "scope": (
                "inference_invocation"
            ),
            "invocation_id": (
                cleaned_invocation_id
            ),
            "training_attempt_id": (
                invocation[
                    "training_attempt_id"
                ]
            ),
            **error_record,
        }
    )

    append_manifest_event(
        manifest,
        event_type="inference_invocation_failed",
        message=(
            f"Inference invocation "
            f"'{cleaned_invocation_id}' failed."
        ),
        details={
            "invocation_id": (
                cleaned_invocation_id
            ),
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


def _load_inference_invocation(
    run_directory: str | Path,
    invocation_id: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    str,
]:
    """
    Load one registered inference invocation.
    """
    cleaned_invocation_id = (
        validate_record_id(
            invocation_id,
            field_name="invocation_id",
        )
    )

    manifest = load_run_manifest(
        run_directory
    )

    invocation = manifest[
        "inference"
    ][
        "invocations"
    ].get(
        cleaned_invocation_id
    )

    if invocation is None:
        raise RunManifestError(
            "The requested inference invocation is not registered.",
            error_code="INFERENCE_INVOCATION_NOT_FOUND",
            details={
                "invocation_id": (
                    cleaned_invocation_id
                ),
            },
        )

    validate_inference_invocation_status(
        str(
            invocation.get(
                "status",
                "",
            )
        )
    )

    return (
        manifest,
        invocation,
        cleaned_invocation_id,
    )


# ---------------------------------------------------------------------------
# SUMMARY HELPERS
# ---------------------------------------------------------------------------


def update_attempt_summary(
    run_directory: str | Path,
    attempt_id: str,
    summary_values: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Add or update summary values for one training attempt.
    """
    cleaned_attempt_id = (
        validate_record_id(
            attempt_id,
            field_name="attempt_id",
        )
    )

    manifest = load_run_manifest(
        run_directory
    )

    attempt = manifest[
        "attempts"
    ].get(
        cleaned_attempt_id
    )

    if attempt is None:
        raise RunManifestError(
            "The requested training attempt is not registered.",
            error_code="ATTEMPT_NOT_FOUND",
            details={
                "attempt_id": (
                    cleaned_attempt_id
                ),
            },
        )

    attempt["summary"] = {
        **attempt.get(
            "summary",
            {},
        ),
        **dict(summary_values),
    }
    attempt["updated_at_utc"] = (
        get_current_utc_timestamp()
    )

    append_manifest_event(
        manifest,
        event_type="attempt_summary_updated",
        message=(
            f"Summary updated for attempt "
            f"'{cleaned_attempt_id}'."
        ),
        details={
            "attempt_id": (
                cleaned_attempt_id
            ),
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


def update_run_summary(
    run_directory: str | Path,
    summary_values: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Add or update run-level summary values.
    """
    manifest = load_run_manifest(
        run_directory
    )

    manifest["summary"] = {
        **manifest.get(
            "summary",
            {},
        ),
        **dict(summary_values),
    }

    append_manifest_event(
        manifest,
        event_type="run_summary_updated",
        message="Backend run summary updated.",
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
