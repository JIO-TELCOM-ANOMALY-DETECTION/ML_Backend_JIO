"""
V1 training-lineage policy.

This module owns the safety rule agreed for the first production version:

- completed stages are immutable inside an attempt;
- a failed/pending stage may execute again;
- a stage cannot start until its immediate prerequisite has completed;
- changing a previously completed upstream stage requires a new attempt;
- inference activation requires successful Part D + inference bootstrap.

The module inspects Manifest V2 but performs no filesystem writes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.core.exceptions import (
    AttemptNotFoundError,
    StageConflictError,
    StagePrerequisiteError,
)
from app.services.run_manifest_service import (
    TRAINING_STAGE_NAMES,
    validate_record_id,
    validate_stage_name,
)


STAGE_ORDER: tuple[str, ...] = tuple(
    TRAINING_STAGE_NAMES
)

PREREQUISITE_STAGE: dict[
    str,
    str | None,
] = {
    "part_a": None,
    "part_b": "part_a",
    "part_c": "part_b",
    "part_d": "part_c",
    "inference_bootstrap": "part_d",
}


def get_attempt_record(
    manifest: Mapping[str, Any],
    attempt_id: str,
) -> Mapping[str, Any]:
    """Return one attempt or raise a run-management not-found error."""
    cleaned_attempt_id = validate_record_id(
        attempt_id,
        field_name="attempt_id",
    )

    attempt = (
        manifest.get(
            "attempts",
            {},
        )
        .get(
            cleaned_attempt_id
        )
    )

    if not isinstance(
        attempt,
        Mapping,
    ):
        raise AttemptNotFoundError(
            "The requested training attempt does not exist.",
            error_code="ATTEMPT_NOT_FOUND",
            details={
                "attempt_id": (
                    cleaned_attempt_id
                ),
            },
        )

    return attempt


def get_stage_record(
    manifest: Mapping[str, Any],
    attempt_id: str,
    stage_name: str,
) -> Mapping[str, Any]:
    """Return one canonical stage record from an attempt."""
    attempt = get_attempt_record(
        manifest,
        attempt_id,
    )
    cleaned_stage_name = (
        validate_stage_name(
            stage_name
        )
    )

    stage = (
        attempt.get(
            "stages",
            {},
        )
        .get(
            cleaned_stage_name
        )
    )

    if not isinstance(
        stage,
        Mapping,
    ):
        raise StageConflictError(
            "The attempt does not contain the expected stage record.",
            error_code="MISSING_STAGE_RECORD",
            details={
                "attempt_id": attempt_id,
                "stage_name": (
                    cleaned_stage_name
                ),
            },
        )

    return stage


def get_successful_execution_id(
    manifest: Mapping[str, Any],
    attempt_id: str,
    stage_name: str,
) -> str | None:
    """
    Return the authoritative successful execution ID when the stage is complete.

    Both the pointer and execution status are checked so a corrupted pointer
    cannot silently become authoritative.
    """
    stage = get_stage_record(
        manifest,
        attempt_id,
        stage_name,
    )

    execution_id = stage.get(
        "successful_execution_id"
    )

    if execution_id is None:
        return None

    executions = stage.get(
        "executions",
        {},
    )
    execution = (
        executions.get(
            execution_id
        )
        if isinstance(
            executions,
            Mapping,
        )
        else None
    )

    if (
        not isinstance(
            execution,
            Mapping,
        )
        or execution.get(
            "status"
        ) != "completed"
    ):
        raise StageConflictError(
            "The stage successful-execution pointer is inconsistent.",
            error_code="INCONSISTENT_SUCCESSFUL_EXECUTION",
            details={
                "attempt_id": attempt_id,
                "stage_name": stage_name,
                "successful_execution_id": (
                    execution_id
                ),
            },
        )

    return str(
        execution_id
    )


def is_stage_completed(
    manifest: Mapping[str, Any],
    attempt_id: str,
    stage_name: str,
) -> bool:
    """Return True only when a valid successful execution exists."""
    return (
        get_successful_execution_id(
            manifest,
            attempt_id,
            stage_name,
        )
        is not None
    )


def ensure_stage_can_start(
    manifest: Mapping[str, Any],
    attempt_id: str,
    stage_name: str,
) -> None:
    """
    Enforce the V1 immutable-attempt retry/prerequisite policy.

    A stage is allowed when:
    - it has never completed successfully;
    - no execution of that stage is currently running;
    - its immediate prerequisite, if any, has completed successfully.
    """
    cleaned_stage_name = (
        validate_stage_name(
            stage_name
        )
    )
    stage = get_stage_record(
        manifest,
        attempt_id,
        cleaned_stage_name,
    )

    successful_execution_id = (
        get_successful_execution_id(
            manifest,
            attempt_id,
            cleaned_stage_name,
        )
    )

    if successful_execution_id is not None:
        raise StageConflictError(
            "A completed stage cannot be changed or rerun inside the same attempt.",
            error_code="COMPLETED_STAGE_IS_IMMUTABLE",
            details={
                "attempt_id": attempt_id,
                "stage_name": (
                    cleaned_stage_name
                ),
                "successful_execution_id": (
                    successful_execution_id
                ),
                "required_action": (
                    "Create a new training attempt to change a "
                    "previously completed stage."
                ),
            },
        )

    active_execution_id = stage.get(
        "active_execution_id"
    )

    if active_execution_id is not None:
        executions = stage.get(
            "executions",
            {},
        )
        active_execution = (
            executions.get(
                active_execution_id
            )
            if isinstance(
                executions,
                Mapping,
            )
            else None
        )

        if (
            isinstance(
                active_execution,
                Mapping,
            )
            and active_execution.get(
                "status"
            ) == "running"
        ):
            raise StageConflictError(
                "The stage already has a running execution.",
                error_code="STAGE_EXECUTION_ALREADY_RUNNING",
                details={
                    "attempt_id": attempt_id,
                    "stage_name": (
                        cleaned_stage_name
                    ),
                    "execution_id": (
                        active_execution_id
                    ),
                },
            )

    prerequisite = PREREQUISITE_STAGE[
        cleaned_stage_name
    ]

    if (
        prerequisite is not None
        and not is_stage_completed(
            manifest,
            attempt_id,
            prerequisite,
        )
    ):
        raise StagePrerequisiteError(
            "The requested stage cannot start because its prerequisite has not completed.",
            error_code="STAGE_PREREQUISITE_NOT_COMPLETED",
            details={
                "attempt_id": attempt_id,
                "stage_name": (
                    cleaned_stage_name
                ),
                "required_stage": (
                    prerequisite
                ),
            },
        )

    # Detect an impossible historical branch rather than silently continuing.
    current_index = STAGE_ORDER.index(
        cleaned_stage_name
    )

    for downstream_stage in STAGE_ORDER[
        current_index + 1:
    ]:
        if is_stage_completed(
            manifest,
            attempt_id,
            downstream_stage,
        ):
            raise StageConflictError(
                "The attempt contains completed downstream work while the requested upstream stage is incomplete.",
                error_code="INCONSISTENT_STAGE_LINEAGE",
                details={
                    "attempt_id": attempt_id,
                    "stage_name": (
                        cleaned_stage_name
                    ),
                    "completed_downstream_stage": (
                        downstream_stage
                    ),
                },
            )


def ensure_inference_activation_allowed(
    manifest: Mapping[str, Any],
    attempt_id: str,
) -> None:
    """Require successful Part D and bootstrap before activating inference."""
    get_attempt_record(
        manifest,
        attempt_id,
    )

    missing = [
        stage_name
        for stage_name in (
            "part_d",
            "inference_bootstrap",
        )
        if not is_stage_completed(
            manifest,
            attempt_id,
            stage_name,
        )
    ]

    if missing:
        raise StagePrerequisiteError(
            "The attempt is not ready to become the active inference source.",
            error_code="ATTEMPT_NOT_INFERENCE_READY",
            details={
                "attempt_id": attempt_id,
                "missing_completed_stages": (
                    missing
                ),
            },
        )


def get_active_inference_attempt_id(
    manifest: Mapping[str, Any],
) -> str:
    """
    Resolve and validate the attempt currently serving Part E.
    """
    attempt_id = manifest.get(
        "active_inference_attempt_id"
    )

    if (
        not manifest.get(
            "inference_ready",
            False,
        )
        or attempt_id is None
    ):
        raise StagePrerequisiteError(
            "The run does not currently have an inference-ready training attempt.",
            error_code="RUN_NOT_INFERENCE_READY",
        )

    cleaned_attempt_id = validate_record_id(
        str(attempt_id),
        field_name=(
            "active_inference_attempt_id"
        ),
    )

    ensure_inference_activation_allowed(
        manifest,
        cleaned_attempt_id,
    )

    return cleaned_attempt_id


def derive_attempt_status(
    attempt: Mapping[str, Any],
) -> str:
    """
    Derive the operational attempt status from canonical stage state.

    This is deliberately deterministic and does not rely on the previous
    attempt-level status field.
    """
    stages = attempt.get(
        "stages",
        {}
    )

    if not isinstance(
        stages,
        Mapping,
    ):
        return "failed"

    stage_records = [
        stages.get(
            stage_name,
            {},
        )
        for stage_name in STAGE_ORDER
    ]

    if any(
        isinstance(
            record,
            Mapping,
        )
        and record.get(
            "status"
        ) == "running"
        for record in stage_records
    ):
        return "running"

    completed_count = sum(
        1
        for record in stage_records
        if (
            isinstance(
                record,
                Mapping,
            )
            and record.get(
                "successful_execution_id"
            ) is not None
            and record.get(
                "status"
            ) == "completed"
        )
    )

    if completed_count == len(
        STAGE_ORDER
    ):
        return "completed"

    failed_exists = any(
        isinstance(
            record,
            Mapping,
        )
        and record.get(
            "status"
        ) == "failed"
        for record in stage_records
    )

    if failed_exists:
        return (
            "partially_completed"
            if completed_count > 0
            else "failed"
        )

    if completed_count > 0:
        return "partially_completed"

    return "created"


def get_next_stage(
    manifest: Mapping[str, Any],
    attempt_id: str,
) -> str | None:
    """
    Return the first canonical stage without a successful execution.
    """
    get_attempt_record(
        manifest,
        attempt_id,
    )

    for stage_name in STAGE_ORDER:
        if not is_stage_completed(
            manifest,
            attempt_id,
            stage_name,
        ):
            return stage_name

    return None


__all__ = [
    "STAGE_ORDER",
    "PREREQUISITE_STAGE",
    "get_attempt_record",
    "get_stage_record",
    "get_successful_execution_id",
    "is_stage_completed",
    "ensure_stage_can_start",
    "ensure_inference_activation_allowed",
    "get_active_inference_attempt_id",
    "derive_attempt_status",
    "get_next_stage",
]
