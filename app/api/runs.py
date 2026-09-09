"""
Read-only API endpoints for inspecting persisted run lifecycle state.

This router exposes both historical and currently executing runs through the
existing Run Management / Manifest V2 contracts.

Responsibilities
----------------
- list registered runs;
- return the authoritative persisted state of one run;
- return the authoritative persisted state of one training attempt;
- convert internal Manifest V2 records into the existing public workflow view
  schemas;
- translate only genuinely caller-owned missing run/attempt lookups into the
  frozen API_* 404 error codes;
- remove internal filesystem/debug information from persisted summaries,
  metrics, and historical errors.

This module deliberately does NOT:
- start, retry, cancel, or mutate A/B/C/D/E execution;
- scan run directories or guess the newest execution;
- read arbitrary result files;
- generate reports or plots;
- repair registry/manifest inconsistencies;
- expose physical server paths.

Manifest V2 remains the technical source of truth. The registry remains the
run-discovery/index layer.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from app.core import (
    AttemptNotFoundError,
    RunManagementError,
    RunNotFoundError,
)
from app.schemas.workflows import (
    AttemptDetailResponse,
    AttemptId,
    AttemptView,
    ExecutionView,
    InferenceInvocationView,
    RunId,
    RunMetadata,
    RunSummaryResponse,
    StageView,
    WorkflowErrorDetail,
    WorkflowErrorRecord,
)
from app.services.run_manifest_service import (
    TRAINING_STAGE_NAMES,
)

from .dependencies import (
    ApiLoggerDependency,
    RunManagerDependency,
)


router = APIRouter(
    prefix="/api/v1/runs",
    tags=["runs"],
)


# ---------------------------------------------------------------------------
# Public-data safety helpers
# ---------------------------------------------------------------------------


_SENSITIVE_KEY_SUFFIXES = (
    "_path",
    "_paths",
    "_file",
    "_files",
    "_directory",
    "_directories",
    "_root",
)

_SENSITIVE_KEYS = {
    "path",
    "paths",
    "file",
    "files",
    "directory",
    "directories",
    "root",
    "run_directory",
    "attempt_directory",
    "execution_directory",
    "state_directory",
    "staging_directory",
    "output_directory",
    "input_file",
    "output_file",
    "manifest_file",
    "request_file",
    "result_file",
    "error_file",
    "model_file",
    "model_files",
    "file_path",
}

_DIAGNOSTIC_KEYS = {
    "original_error",
    "traceback",
    "stack_trace",
    "exception",
    "raw_error",
}

_WINDOWS_ABSOLUTE_PATH = re.compile(
    r"^[A-Za-z]:[\\/]"
)


def _is_sensitive_key(
    key: object,
) -> bool:
    """Return whether a mapping key represents internal filesystem/debug data."""
    cleaned = str(
        key
    ).strip().lower()

    if (
        cleaned in _SENSITIVE_KEYS
        or cleaned in _DIAGNOSTIC_KEYS
    ):
        return True

    return any(
        cleaned.endswith(
            suffix
        )
        for suffix in _SENSITIVE_KEY_SUFFIXES
    )


def _looks_like_absolute_path(
    value: str,
) -> bool:
    """
    Conservatively detect common absolute filesystem path formats.

    This is defense in depth for arbitrary summary/metric values. Primary
    protection remains key-based exclusion and never exposing raw Manifest
    filesystem fields.
    """
    cleaned = value.strip()

    return bool(
        cleaned.startswith("/")
        or cleaned.startswith("\\\\")
        or _WINDOWS_ABSOLUTE_PATH.match(
            cleaned
        )
    )


def _public_json_value(
    value: Any,
) -> Any:
    """
    Convert arbitrary persisted summary/metric data into conservative JSON-safe
    public data.

    Mapping entries whose keys represent filesystem/debug information are
    dropped entirely. Absolute path-looking string values are redacted even if
    their key was not recognizable.
    """
    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        int,
    ):
        return value

    if isinstance(
        value,
        float,
    ):
        return (
            value
            if math.isfinite(
                value
            )
            else None
        )

    if isinstance(
        value,
        str,
    ):
        return (
            "[redacted]"
            if _looks_like_absolute_path(
                value
            )
            else value
        )

    if isinstance(
        value,
        Path,
    ):
        return "[redacted]"

    if isinstance(
        value,
        (date, datetime),
    ):
        return value.isoformat()

    if isinstance(
        value,
        Enum,
    ):
        return _public_json_value(
            value.value
        )

    if hasattr(
        value,
        "model_dump",
    ):
        return _public_json_value(
            value.model_dump(
                mode="python"
            )
        )

    if isinstance(
        value,
        Mapping,
    ):
        public: dict[
            str,
            Any,
        ] = {}

        for (
            key,
            child,
        ) in value.items():
            if _is_sensitive_key(
                key
            ):
                continue

            public[
                str(
                    key
                )
            ] = _public_json_value(
                child
            )

        return public

    if isinstance(
        value,
        Sequence,
    ) and not isinstance(
        value,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        return [
            _public_json_value(
                child
            )
            for child in value
        ]

    if isinstance(
        value,
        (
            set,
            frozenset,
        ),
    ):
        return [
            _public_json_value(
                child
            )
            for child in value
        ]

    # Never expose repr(value); opaque object representations may contain
    # paths, raw data, connection strings, or other implementation details.
    return {
        "type": type(
            value
        ).__name__,
    }


def _public_mapping(
    value: object,
    *,
    field_name: str,
) -> dict[str, Any]:
    """Require one mapping and return its sanitized public representation."""
    if value is None:
        return {}

    if not isinstance(
        value,
        Mapping,
    ):
        raise RunManagementError(
            f"Persisted {field_name} must be a mapping.",
            error_code="RUN_VIEW_INVALID_PUBLIC_MAPPING",
            details={
                "field_name": (
                    field_name
                ),
                "received_type": type(
                    value
                ).__name__,
            },
        )

    converted = _public_json_value(
        value
    )

    if not isinstance(
        converted,
        dict,
    ):
        raise RunManagementError(
            f"Persisted {field_name} did not resolve to an object.",
            error_code="RUN_VIEW_INVALID_PUBLIC_MAPPING",
            details={
                "field_name": (
                    field_name
                ),
            },
        )

    return converted


# ---------------------------------------------------------------------------
# Persisted-error safety helpers
# ---------------------------------------------------------------------------


_PUBLIC_ERROR_MESSAGE = (
    "A backend error occurred during this operation."
)

_VALID_ERROR_SCOPES = {
    "run",
    "attempt",
    "stage_execution",
    "inference_invocation",
}


def _error_identity(
    record: Mapping[str, Any],
) -> tuple[str, str | None]:
    """Validate and normalize the minimum persisted error identity."""
    error_type = record.get(
        "error_type"
    )

    if (
        not isinstance(
            error_type,
            str,
        )
        or not error_type.strip()
    ):
        raise RunManagementError(
            "A persisted run error has an invalid error_type.",
            error_code="RUN_VIEW_INVALID_ERROR_RECORD",
        )

    error_code = record.get(
        "error_code"
    )

    if (
        error_code is not None
        and not isinstance(
            error_code,
            str,
        )
    ):
        raise RunManagementError(
            "A persisted run error has an invalid error_code.",
            error_code="RUN_VIEW_INVALID_ERROR_RECORD",
            details={
                "received_type": type(
                    error_code
                ).__name__,
            },
        )

    normalized_code = (
        error_code.strip()
        if isinstance(
            error_code,
            str,
        )
        and error_code.strip()
        else None
    )

    return (
        error_type.strip(),
        normalized_code,
    )


def _public_error_detail(
    value: object,
) -> WorkflowErrorDetail | None:
    """
    Build a safe execution/invocation error view.

    Raw persisted message/details are intentionally never reflected.
    """
    if value is None:
        return None

    if not isinstance(
        value,
        Mapping,
    ):
        raise RunManagementError(
            "A persisted execution error must be a mapping or null.",
            error_code="RUN_VIEW_INVALID_ERROR_RECORD",
            details={
                "received_type": type(
                    value
                ).__name__,
            },
        )

    (
        error_type,
        error_code,
    ) = _error_identity(
        value
    )

    return WorkflowErrorDetail(
        error_type=error_type,
        message=_PUBLIC_ERROR_MESSAGE,
        error_code=error_code,
        details={},
    )


def _optional_record_id(
    record: Mapping[str, Any],
    field_name: str,
) -> str | None:
    """Read one optional string identifier from a persisted error record."""
    value = record.get(
        field_name
    )

    if value is None:
        return None

    if (
        not isinstance(
            value,
            str,
        )
        or not value.strip()
    ):
        raise RunManagementError(
            "A persisted run error contains an invalid lifecycle identifier.",
            error_code="RUN_VIEW_INVALID_ERROR_RECORD",
            details={
                "field_name": (
                    field_name
                ),
                "received_type": type(
                    value
                ).__name__,
            },
        )

    return value.strip()


def _public_error_record(
    value: object,
) -> WorkflowErrorRecord:
    """
    Build a safe historical error record.

    Lifecycle identifiers and timestamp/scope remain useful to the UI.
    Internal message/details are sealed.
    """
    if not isinstance(
        value,
        Mapping,
    ):
        raise RunManagementError(
            "A persisted historical error must be a mapping.",
            error_code="RUN_VIEW_INVALID_ERROR_RECORD",
            details={
                "received_type": type(
                    value
                ).__name__,
            },
        )

    (
        error_type,
        error_code,
    ) = _error_identity(
        value
    )

    scope = value.get(
        "scope"
    )

    if (
        scope is not None
        and scope not in _VALID_ERROR_SCOPES
    ):
        raise RunManagementError(
            "A persisted historical error has an invalid scope.",
            error_code="RUN_VIEW_INVALID_ERROR_RECORD",
            details={
                "scope": scope,
            },
        )

    return WorkflowErrorRecord(
        error_type=error_type,
        message=_PUBLIC_ERROR_MESSAGE,
        error_code=error_code,
        details={},
        timestamp_utc=value.get(
            "timestamp_utc"
        ),
        scope=scope,
        attempt_id=_optional_record_id(
            value,
            "attempt_id",
        ),
        training_attempt_id=(
            _optional_record_id(
                value,
                "training_attempt_id",
            )
        ),
        stage_name=_optional_record_id(
            value,
            "stage_name",
        ),
        execution_id=_optional_record_id(
            value,
            "execution_id",
        ),
        invocation_id=_optional_record_id(
            value,
            "invocation_id",
        ),
    )


def _public_error_records(
    value: object,
    *,
    field_name: str,
) -> tuple[
    WorkflowErrorRecord,
    ...,
]:
    """Convert one persisted historical-error collection."""
    if value is None:
        return ()

    if (
        not isinstance(
            value,
            Sequence,
        )
        or isinstance(
            value,
            (
                str,
                bytes,
                bytearray,
            ),
        )
    ):
        raise RunManagementError(
            f"Persisted {field_name} must be a sequence.",
            error_code="RUN_VIEW_INVALID_ERROR_COLLECTION",
            details={
                "field_name": (
                    field_name
                ),
                "received_type": type(
                    value
                ).__name__,
            },
        )

    return tuple(
        _public_error_record(
            item
        )
        for item in value
    )


# ---------------------------------------------------------------------------
# Manifest -> public-schema builders
# ---------------------------------------------------------------------------


def _require_mapping(
    value: object,
    *,
    field_name: str,
) -> Mapping[str, Any]:
    """Require one internal persisted record to be a mapping."""
    if not isinstance(
        value,
        Mapping,
    ):
        raise RunManagementError(
            f"Persisted {field_name} must be a mapping.",
            error_code="RUN_VIEW_INVALID_RECORD",
            details={
                "field_name": (
                    field_name
                ),
                "received_type": type(
                    value
                ).__name__,
            },
        )

    return value


def _matching_record_id(
    *,
    key: str,
    record: Mapping[str, Any],
    field_name: str,
) -> str:
    """Require an embedded persisted identifier to match its dictionary key."""
    embedded = record.get(
        field_name
    )

    if embedded != key:
        raise RunManagementError(
            "Persisted run-state identity does not match its record key.",
            error_code="RUN_VIEW_RECORD_ID_MISMATCH",
            details={
                "field_name": (
                    field_name
                ),
                "record_key": key,
                "embedded_id": embedded,
            },
        )

    return key


def _build_execution_view(
    execution_id: str,
    value: object,
) -> ExecutionView:
    """Convert one Manifest V2 stage execution into its public status view."""
    record = _require_mapping(
        value,
        field_name=(
            "stage execution"
        ),
    )

    _matching_record_id(
        key=execution_id,
        record=record,
        field_name="execution_id",
    )

    return ExecutionView(
        execution_id=execution_id,
        status=record.get(
            "status"
        ),
        started_at_utc=record.get(
            "started_at_utc"
        ),
        completed_at_utc=record.get(
            "completed_at_utc"
        ),
        metrics=_public_mapping(
            record.get(
                "metrics",
                {},
            ),
            field_name=(
                "execution metrics"
            ),
        ),
        error=_public_error_detail(
            record.get(
                "error"
            )
        ),
    )


def _build_stage_view(
    *,
    stage_name: str,
    value: object,
) -> StageView:
    """Convert one canonical Manifest V2 stage into its public status view."""
    record = _require_mapping(
        value,
        field_name=(
            f"{stage_name} stage"
        ),
    )

    executions_raw = _require_mapping(
        record.get(
            "executions"
        ),
        field_name=(
            f"{stage_name} executions"
        ),
    )

    executions = tuple(
        _build_execution_view(
            execution_id,
            executions_raw[
                execution_id
            ],
        )
        for execution_id in sorted(
            executions_raw
        )
    )

    execution_lookup = {
        execution.execution_id: (
            execution
        )
        for execution in executions
    }

    stage_status = record.get(
        "status"
    )
    active_execution_id = record.get(
        "active_execution_id"
    )
    successful_execution_id = record.get(
        "successful_execution_id"
    )

    running_execution_ids = tuple(
        execution.execution_id
        for execution in executions
        if execution.status
        == "running"
    )

    # Manifest V2 lifecycle writes a running stage and its active execution
    # pointer together. A read-only public view must not silently display a
    # contradictory state if the persisted record was corrupted externally or
    # by a future regression.
    if stage_status == "running":
        if (
            active_execution_id
            is None
            or len(
                running_execution_ids
            )
            != 1
            or running_execution_ids[
                0
            ]
            != active_execution_id
        ):
            raise RunManagementError(
                "The persisted running stage does not have one matching active execution.",
                error_code="RUN_VIEW_RUNNING_STAGE_INCONSISTENT",
                details={
                    "stage_name": stage_name,
                    "active_execution_id": (
                        active_execution_id
                    ),
                    "running_execution_ids": (
                        list(
                            running_execution_ids
                        )
                    ),
                },
            )
    else:
        if (
            active_execution_id
            is not None
            or running_execution_ids
        ):
            raise RunManagementError(
                "A non-running stage retains active/running execution state.",
                error_code="RUN_VIEW_NON_RUNNING_STAGE_HAS_ACTIVE_EXECUTION",
                details={
                    "stage_name": stage_name,
                    "stage_status": (
                        stage_status
                    ),
                    "active_execution_id": (
                        active_execution_id
                    ),
                    "running_execution_ids": (
                        list(
                            running_execution_ids
                        )
                    ),
                },
            )

    if active_execution_id is not None:
        active = execution_lookup.get(
            active_execution_id
        )

        if (
            active is None
            or active.status
            != "running"
        ):
            raise RunManagementError(
                "The persisted active execution pointer is inconsistent.",
                error_code="RUN_VIEW_ACTIVE_EXECUTION_INCONSISTENT",
                details={
                    "stage_name": stage_name,
                    "active_execution_id": (
                        active_execution_id
                    ),
                },
            )

    if successful_execution_id is not None:
        successful = (
            execution_lookup.get(
                successful_execution_id
            )
        )

        if (
            successful is None
            or successful.status
            != "completed"
        ):
            raise RunManagementError(
                "The persisted successful execution pointer is inconsistent.",
                error_code="RUN_VIEW_SUCCESSFUL_EXECUTION_INCONSISTENT",
                details={
                    "stage_name": stage_name,
                    "successful_execution_id": (
                        successful_execution_id
                    ),
                },
            )

    if stage_status == "completed":
        if successful_execution_id is None:
            raise RunManagementError(
                "A completed stage does not register a successful execution.",
                error_code="RUN_VIEW_COMPLETED_STAGE_WITHOUT_SUCCESS",
                details={
                    "stage_name": (
                        stage_name
                    ),
                },
            )
    elif successful_execution_id is not None:
        raise RunManagementError(
            "A non-completed stage retains a successful execution pointer.",
            error_code="RUN_VIEW_NON_COMPLETED_STAGE_HAS_SUCCESS",
            details={
                "stage_name": stage_name,
                "stage_status": stage_status,
                "successful_execution_id": (
                    successful_execution_id
                ),
            },
        )

    return StageView(
        stage=stage_name,
        status=stage_status,
        active_execution_id=(
            active_execution_id
        ),
        successful_execution_id=(
            successful_execution_id
        ),
        executions=executions,
    )


def _build_attempt_view(
    attempt_id: str,
    value: object,
) -> AttemptView:
    """Convert one Manifest V2 attempt into its public status view."""
    record = _require_mapping(
        value,
        field_name=(
            "training attempt"
        ),
    )

    _matching_record_id(
        key=attempt_id,
        record=record,
        field_name="attempt_id",
    )

    stages_raw = _require_mapping(
        record.get(
            "stages"
        ),
        field_name=(
            "attempt stages"
        ),
    )

    stages = tuple(
        _build_stage_view(
            stage_name=stage_name,
            value=stages_raw.get(
                stage_name
            ),
        )
        for stage_name in (
            TRAINING_STAGE_NAMES
        )
    )

    return AttemptView(
        attempt_id=attempt_id,
        status=record.get(
            "status"
        ),
        created_at_utc=record.get(
            "created_at_utc"
        ),
        updated_at_utc=record.get(
            "updated_at_utc"
        ),
        started_at_utc=record.get(
            "started_at_utc"
        ),
        completed_at_utc=record.get(
            "completed_at_utc"
        ),
        stages=stages,
        summary=_public_mapping(
            record.get(
                "summary",
                {},
            ),
            field_name=(
                "attempt summary"
            ),
        ),
        errors=_public_error_records(
            record.get(
                "errors",
                [],
            ),
            field_name=(
                "attempt errors"
            ),
        ),
    )


def _build_inference_invocation_view(
    invocation_id: str,
    value: object,
    *,
    known_attempt_ids: set[str],
) -> InferenceInvocationView:
    """Convert one Manifest V2 Part E invocation into its public status view."""
    record = _require_mapping(
        value,
        field_name=(
            "inference invocation"
        ),
    )

    _matching_record_id(
        key=invocation_id,
        record=record,
        field_name="invocation_id",
    )

    training_attempt_id = (
        record.get(
            "training_attempt_id"
        )
    )

    if (
        not isinstance(
            training_attempt_id,
            str,
        )
        or training_attempt_id
        not in known_attempt_ids
    ):
        raise RunManagementError(
            "An inference invocation references an unknown training attempt.",
            error_code="RUN_VIEW_INFERENCE_ATTEMPT_UNKNOWN",
            details={
                "invocation_id": (
                    invocation_id
                ),
                "training_attempt_id": (
                    training_attempt_id
                ),
            },
        )

    return InferenceInvocationView(
        invocation_id=invocation_id,
        training_attempt_id=(
            training_attempt_id
        ),
        status=record.get(
            "status"
        ),
        started_at_utc=record.get(
            "started_at_utc"
        ),
        completed_at_utc=record.get(
            "completed_at_utc"
        ),
        metrics=_public_mapping(
            record.get(
                "metrics",
                {},
            ),
            field_name=(
                "inference metrics"
            ),
        ),
        error=_public_error_detail(
            record.get(
                "error"
            )
        ),
    )


def _build_run_summary(
    manifest: Mapping[str, Any],
) -> RunSummaryResponse:
    """Convert one validated Manifest V2 document into its public run view."""
    run_id = manifest.get(
        "run_id"
    )

    if (
        not isinstance(
            run_id,
            str,
        )
        or not run_id.strip()
    ):
        raise RunManagementError(
            "The persisted run manifest has an invalid run_id.",
            error_code="RUN_VIEW_INVALID_RUN_ID",
        )

    attempts_raw = _require_mapping(
        manifest.get(
            "attempts"
        ),
        field_name="run attempts",
    )

    attempts = tuple(
        _build_attempt_view(
            attempt_id,
            attempts_raw[
                attempt_id
            ],
        )
        for attempt_id in sorted(
            attempts_raw
        )
    )

    known_attempt_ids = {
        attempt.attempt_id
        for attempt in attempts
    }

    active_inference_attempt_id = (
        manifest.get(
            "active_inference_attempt_id"
        )
    )
    inference_ready = bool(
        manifest.get(
            "inference_ready",
            False,
        )
    )

    if (
        active_inference_attempt_id
        is not None
        and active_inference_attempt_id
        not in known_attempt_ids
    ):
        raise RunManagementError(
            "The active inference attempt pointer references an unknown attempt.",
            error_code="RUN_VIEW_ACTIVE_INFERENCE_ATTEMPT_UNKNOWN",
            details={
                "active_inference_attempt_id": (
                    active_inference_attempt_id
                ),
            },
        )

    if (
        inference_ready
        and active_inference_attempt_id
        is None
    ):
        raise RunManagementError(
            "The run is marked inference-ready without an active attempt.",
            error_code="RUN_VIEW_INFERENCE_READY_WITHOUT_ACTIVE_ATTEMPT",
        )

    inference = _require_mapping(
        manifest.get(
            "inference"
        ),
        field_name=(
            "run inference state"
        ),
    )

    invocations_raw = _require_mapping(
        inference.get(
            "invocations"
        ),
        field_name=(
            "inference invocations"
        ),
    )

    invocation_views = [
        _build_inference_invocation_view(
            invocation_id,
            invocations_raw[
                invocation_id
            ],
            known_attempt_ids=(
                known_attempt_ids
            ),
        )
        for invocation_id in (
            invocations_raw
        )
    ]

    # Invocation IDs include timestamps today, but ordering by persisted
    # started_at_utc keeps this view correct even if the identifier strategy
    # changes later.
    invocation_views.sort(
        key=lambda item: (
            item.started_at_utc
            is not None,
            (
                item.started_at_utc.isoformat()
                if item.started_at_utc
                is not None
                else ""
            ),
            item.invocation_id,
        )
    )

    latest_attempt_id = (
        manifest.get(
            "latest_attempt_id"
        )
    )

    if (
        latest_attempt_id
        is not None
        and latest_attempt_id
        not in known_attempt_ids
    ):
        raise RunManagementError(
            "The latest attempt pointer references an unknown attempt.",
            error_code="RUN_VIEW_LATEST_ATTEMPT_UNKNOWN",
            details={
                "latest_attempt_id": (
                    latest_attempt_id
                ),
            },
        )

    if (
        active_inference_attempt_id
        is not None
        and not inference_ready
    ):
        raise RunManagementError(
            "The run has an active inference attempt while inference_ready is false.",
            error_code="RUN_VIEW_ACTIVE_INFERENCE_WITHOUT_READY_FLAG",
            details={
                "active_inference_attempt_id": (
                    active_inference_attempt_id
                ),
            },
        )

    if inference_ready:
        active_attempt = next(
            (
                attempt
                for attempt in attempts
                if attempt.attempt_id
                == active_inference_attempt_id
            ),
            None,
        )

        if active_attempt is None:
            raise RunManagementError(
                "The inference-ready run does not expose its active attempt.",
                error_code="RUN_VIEW_INFERENCE_READY_ATTEMPT_INVALID",
                details={
                    "active_inference_attempt_id": (
                        active_inference_attempt_id
                    ),
                },
            )

        active_stage_lookup = {
            stage.stage: stage
            for stage in (
                active_attempt.stages
            )
        }

        invalid_ready_stages = [
            stage_name
            for stage_name in (
                "part_d",
                "inference_bootstrap",
            )
            if (
                stage_name
                not in active_stage_lookup
                or active_stage_lookup[
                    stage_name
                ].status
                != "completed"
                or active_stage_lookup[
                    stage_name
                ].successful_execution_id
                is None
            )
        ]

        if invalid_ready_stages:
            raise RunManagementError(
                "The active inference attempt is not backed by completed Part D and bootstrap stages.",
                error_code="RUN_VIEW_INFERENCE_READY_ATTEMPT_INVALID",
                details={
                    "active_inference_attempt_id": (
                        active_inference_attempt_id
                    ),
                    "invalid_stages": (
                        invalid_ready_stages
                    ),
                },
            )

    return RunSummaryResponse(
        run_id=run_id,
        run_metadata=RunMetadata.model_validate(
            manifest.get(
                "run_metadata",
                {},
            )
        ),
        status=manifest.get(
            "status"
        ),
        created_at_utc=manifest.get(
            "created_at_utc"
        ),
        updated_at_utc=manifest.get(
            "updated_at_utc"
        ),
        started_at_utc=manifest.get(
            "started_at_utc"
        ),
        completed_at_utc=manifest.get(
            "completed_at_utc"
        ),
        latest_attempt_id=(
            latest_attempt_id
        ),
        active_inference_attempt_id=(
            active_inference_attempt_id
        ),
        inference_ready=inference_ready,
        attempt_count=len(
            attempts
        ),
        attempts=attempts,
        inference_invocation_count=len(
            invocation_views
        ),
        inference_invocations=tuple(
            invocation_views
        ),
        summary=_public_mapping(
            manifest.get(
                "summary",
                {},
            ),
            field_name="run summary",
        ),
        errors=_public_error_records(
            manifest.get(
                "errors",
                [],
            ),
            field_name="run errors",
        ),
    )


# ---------------------------------------------------------------------------
# Public lookup translation
# ---------------------------------------------------------------------------


def _run_is_registered(
    *,
    manager: Any,
    run_id: str,
) -> bool:
    """
    Perform a targeted registry membership check for one caller-requested run.

    This intentionally uses the registry's public ``exists`` interface rather
    than ``manager.list_runs()``. A valid run lookup must not depend on decoding
    every unrelated registry entry.
    """
    return bool(
        manager.registry.exists(
            run_id
        )
    )


def _get_public_manifest(
    *,
    manager: Any,
    run_id: str,
) -> dict[str, Any]:
    """
    Load a caller-requested run with safe 404 semantics.

    The targeted registry membership check establishes that absence genuinely
    belongs to this caller lookup. Once existence is observed, any later
    registry, directory, manifest, identity, or concurrent-removal failure
    remains an internal 500 rather than being incorrectly translated to a
    public 404.
    """
    if not _run_is_registered(
        manager=manager,
        run_id=run_id,
    ):
        raise RunNotFoundError(
            "The caller-requested run is not registered.",
            error_code="API_RUN_NOT_FOUND",
            details={
                "run_id": (
                    run_id
                ),
            },
        )

    manifest = manager.get_manifest(
        run_id
    )

    if (
        not isinstance(
            manifest,
            dict,
        )
        or manifest.get(
            "run_id"
        )
        != run_id
    ):
        raise RunManagementError(
            "Run Management returned an inconsistent manifest for the requested run.",
            error_code="RUN_VIEW_MANIFEST_ID_MISMATCH",
            details={
                "requested_run_id": (
                    run_id
                ),
                "manifest_run_id": (
                    manifest.get(
                        "run_id"
                    )
                    if isinstance(
                        manifest,
                        Mapping,
                    )
                    else None
                ),
            },
        )

    return manifest


def _get_attempt_record_for_public_lookup(
    *,
    manifest: Mapping[str, Any],
    run_id: str,
    attempt_id: str,
) -> Mapping[str, Any]:
    """Resolve one caller-requested attempt and translate only true absence."""
    attempts = _require_mapping(
        manifest.get(
            "attempts"
        ),
        field_name=(
            "run attempts"
        ),
    )

    if attempt_id not in attempts:
        raise AttemptNotFoundError(
            "The caller-requested training attempt is not registered.",
            error_code="API_ATTEMPT_NOT_FOUND",
            details={
                "run_id": run_id,
                "attempt_id": (
                    attempt_id
                ),
            },
        )

    return _require_mapping(
        attempts[
            attempt_id
        ],
        field_name=(
            "training attempt"
        ),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=list[
        RunSummaryResponse
    ],
    summary="List backend runs",
)
def list_runs(
    manager: RunManagerDependency,
    logger: ApiLoggerDependency,
) -> list[RunSummaryResponse]:
    """
    Return all registered runs using authoritative Manifest V2 state.

    This endpoint supports both a historical-runs page and discovery of runs
    that are currently executing. Runs are returned newest first.

    If one registered run has corrupt/unreadable technical state, the request
    fails closed rather than silently returning a misleading partial list.
    """
    registry_records = (
        manager.list_runs()
    )

    summaries: list[
        RunSummaryResponse
    ] = []

    for record in registry_records:
        manifest = manager.get_manifest(
            record.run_id
        )

        summary = _build_run_summary(
            manifest
        )

        if (
            summary.run_id
            != record.run_id
        ):
            raise RunManagementError(
                "Registry and Manifest V2 disagree on run identity.",
                error_code="RUN_VIEW_REGISTRY_MANIFEST_ID_MISMATCH",
                details={
                    "registry_run_id": (
                        record.run_id
                    ),
                    "manifest_run_id": (
                        summary.run_id
                    ),
                },
            )

        summaries.append(
            summary
        )

    summaries.sort(
        key=lambda item: (
            item.created_at_utc,
            item.run_id,
        ),
        reverse=True,
    )

    logger.debug(
        "Listed %s backend runs.",
        len(
            summaries
        ),
    )

    return summaries


@router.get(
    "/{run_id}",
    response_model=RunSummaryResponse,
    summary="Get backend run status",
)
def get_run(
    run_id: RunId,
    manager: RunManagerDependency,
    logger: ApiLoggerDependency,
) -> RunSummaryResponse:
    """
    Return one run's latest persisted lifecycle state.

    The same endpoint works for:
    - a completed historical run;
    - a partially completed/failed run;
    - a currently running pipeline.

    It reports persisted status only; it does not start or modify execution.
    """
    manifest = _get_public_manifest(
        manager=manager,
        run_id=run_id,
    )

    response = _build_run_summary(
        manifest
    )

    logger.debug(
        "Read backend run status for run_id=%s.",
        run_id,
    )

    return response


@router.get(
    "/{run_id}/attempts/{attempt_id}",
    response_model=AttemptDetailResponse,
    summary="Get training attempt status",
)
def get_attempt(
    run_id: RunId,
    attempt_id: AttemptId,
    manager: RunManagerDependency,
    logger: ApiLoggerDependency,
) -> AttemptDetailResponse:
    """
    Return detailed persisted stage/execution state for one training attempt.

    `inference_ready` in this attempt-level response means that this exact
    attempt is currently the run's active inference-ready attempt. An older
    completed attempt therefore returns false after another attempt becomes
    active for inference.
    """
    manifest = _get_public_manifest(
        manager=manager,
        run_id=run_id,
    )

    # Establish caller-owned attempt absence first so a genuinely unknown
    # attempt receives the frozen API_ATTEMPT_NOT_FOUND / 404 contract.
    _get_attempt_record_for_public_lookup(
        manifest=manifest,
        run_id=run_id,
        attempt_id=attempt_id,
    )

    # Then build the authoritative run view so attempt-detail cannot bypass
    # run-level pointer/inference-readiness consistency checks enforced by the
    # normal run-status endpoint.
    run_summary = _build_run_summary(
        manifest
    )

    attempt = next(
        (
            item
            for item in run_summary.attempts
            if item.attempt_id
            == attempt_id
        ),
        None,
    )

    if attempt is None:
        # Membership was proven above. Reaching this branch means the
        # manifest-to-public conversion lost a registered attempt and is an
        # internal consistency failure rather than a public 404.
        raise RunManagementError(
            "The registered attempt was not present in the public run view.",
            error_code="RUN_VIEW_ATTEMPT_CONVERSION_MISMATCH",
            details={
                "run_id": run_id,
                "attempt_id": (
                    attempt_id
                ),
            },
        )

    active_inference_attempt_id = (
        run_summary
        .active_inference_attempt_id
    )

    attempt_is_inference_ready = bool(
        run_summary.inference_ready
        and active_inference_attempt_id
        == attempt_id
    )

    logger.debug(
        (
            "Read training attempt status for "
            "run_id=%s attempt_id=%s."
        ),
        run_id,
        attempt_id,
    )

    return AttemptDetailResponse(
        run_id=run_id,
        attempt=attempt,
        active_inference_attempt_id=(
            active_inference_attempt_id
        ),
        inference_ready=(
            attempt_is_inference_ready
        ),
    )


__all__ = [
    "router",
]