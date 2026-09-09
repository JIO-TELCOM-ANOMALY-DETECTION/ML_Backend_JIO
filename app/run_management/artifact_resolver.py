"""
Authoritative artifact resolution for Manifest V2.

Workflows must not search directories or reconstruct filenames. This module
resolves only artifacts explicitly registered by a successful execution and
rejects paths that escape the owning run root.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.core.exceptions import (
    ArtifactResolutionError,
)
from app.services.run_manifest_service import (
    validate_record_id,
    validate_stage_name,
)

from app.run_management.lineage import (
    PREREQUISITE_STAGE,
    get_active_inference_attempt_id,
    get_attempt_record,
    get_successful_execution_id,
)
from app.run_management.models import (
    ArtifactReference,
)


def _safe_run_path(
    run_directory: str | Path,
    candidate: str | Path,
    *,
    must_exist: bool,
    must_be_file: bool | None = None,
) -> Path:
    """
    Resolve one persisted path and prove that it remains below run_directory.
    """
    run_root = Path(
        run_directory
    ).expanduser().resolve()

    resolved = Path(
        candidate
    ).expanduser().resolve()

    try:
        resolved.relative_to(
            run_root
        )
    except ValueError as error:
        raise ArtifactResolutionError(
            "A persisted artifact path escapes the owning run directory.",
            error_code="ARTIFACT_OUTSIDE_RUN_DIRECTORY",
            details={
                "run_directory": str(
                    run_root
                ),
                "artifact_path": str(
                    resolved
                ),
            },
        ) from error

    if (
        must_exist
        and not resolved.exists()
    ):
        raise ArtifactResolutionError(
            "A required persisted artifact does not exist.",
            error_code="ARTIFACT_NOT_FOUND",
            details={
                "artifact_path": str(
                    resolved
                ),
            },
        )

    if (
        must_be_file is True
        and resolved.exists()
        and not resolved.is_file()
    ):
        raise ArtifactResolutionError(
            "A required artifact path is not a file.",
            error_code="ARTIFACT_PATH_IS_NOT_FILE",
            details={
                "artifact_path": str(
                    resolved
                ),
            },
        )

    if (
        must_be_file is False
        and resolved.exists()
        and not resolved.is_dir()
    ):
        raise ArtifactResolutionError(
            "A required artifact path is not a directory.",
            error_code="ARTIFACT_PATH_IS_NOT_DIRECTORY",
            details={
                "artifact_path": str(
                    resolved
                ),
            },
        )

    return resolved


def resolve_attempt_input(
    manifest: Mapping[str, Any],
    run_directory: str | Path,
    attempt_id: str,
) -> Path:
    """
    Resolve the immutable run-owned input snapshot for one training attempt.
    """
    attempt = get_attempt_record(
        manifest,
        attempt_id,
    )

    input_record = attempt.get(
        "input"
    )

    if not isinstance(
        input_record,
        Mapping,
    ):
        raise ArtifactResolutionError(
            "The training attempt does not contain input-file metadata.",
            error_code="ATTEMPT_INPUT_NOT_REGISTERED",
            details={
                "attempt_id": attempt_id,
            },
        )

    file_path = input_record.get(
        "file_path"
    )

    if not file_path:
        raise ArtifactResolutionError(
            "The training attempt input metadata does not contain a file path.",
            error_code="ATTEMPT_INPUT_PATH_MISSING",
            details={
                "attempt_id": attempt_id,
            },
        )

    return _safe_run_path(
        run_directory,
        str(file_path),
        must_exist=True,
        must_be_file=True,
    )


def resolve_successful_execution_record(
    manifest: Mapping[str, Any],
    attempt_id: str,
    stage_name: str,
) -> Mapping[str, Any]:
    """
    Resolve the authoritative completed execution record for one stage.
    """
    cleaned_stage_name = (
        validate_stage_name(
            stage_name
        )
    )
    attempt = get_attempt_record(
        manifest,
        attempt_id,
    )
    stage = attempt[
        "stages"
    ][
        cleaned_stage_name
    ]

    execution_id = (
        get_successful_execution_id(
            manifest,
            attempt_id,
            cleaned_stage_name,
        )
    )

    if execution_id is None:
        raise ArtifactResolutionError(
            "The requested stage has no successful execution.",
            error_code="SUCCESSFUL_STAGE_EXECUTION_NOT_FOUND",
            details={
                "attempt_id": attempt_id,
                "stage_name": (
                    cleaned_stage_name
                ),
            },
        )

    execution = (
        stage.get(
            "executions",
            {}
        )
        .get(
            execution_id
        )
    )

    if not isinstance(
        execution,
        Mapping,
    ):
        raise ArtifactResolutionError(
            "The successful stage execution record is missing.",
            error_code="SUCCESSFUL_EXECUTION_RECORD_MISSING",
            details={
                "attempt_id": attempt_id,
                "stage_name": (
                    cleaned_stage_name
                ),
                "execution_id": (
                    execution_id
                ),
            },
        )

    return execution


def resolve_stage_result(
    manifest: Mapping[str, Any],
    run_directory: str | Path,
    attempt_id: str,
    stage_name: str,
) -> ArtifactReference:
    """
    Resolve the persisted result.json associated with a successful stage.
    """
    cleaned_stage_name = (
        validate_stage_name(
            stage_name
        )
    )
    execution_id = (
        get_successful_execution_id(
            manifest,
            attempt_id,
            cleaned_stage_name,
        )
    )

    if execution_id is None:
        raise ArtifactResolutionError(
            "The requested stage has no successful result.",
            error_code="STAGE_RESULT_NOT_AVAILABLE",
            details={
                "attempt_id": attempt_id,
                "stage_name": (
                    cleaned_stage_name
                ),
            },
        )

    execution = (
        resolve_successful_execution_record(
            manifest,
            attempt_id,
            cleaned_stage_name,
        )
    )

    result_file = execution.get(
        "result_file"
    )

    if not result_file:
        raise ArtifactResolutionError(
            "The successful execution does not register a result snapshot.",
            error_code="STAGE_RESULT_FILE_NOT_REGISTERED",
            details={
                "attempt_id": attempt_id,
                "stage_name": (
                    cleaned_stage_name
                ),
                "execution_id": (
                    execution_id
                ),
            },
        )

    path = _safe_run_path(
        run_directory,
        str(result_file),
        must_exist=True,
        must_be_file=True,
    )

    return ArtifactReference(
        path=path,
        artifact_type="stage_result",
        attempt_id=attempt_id,
        stage_name=(
            cleaned_stage_name
        ),
        execution_id=(
            execution_id
        ),
    )


def resolve_stage_outputs(
    manifest: Mapping[str, Any],
    run_directory: str | Path,
    attempt_id: str,
    stage_name: str,
    *,
    output_type: str | None = None,
) -> tuple[ArtifactReference, ...]:
    """
    Resolve files explicitly registered against a successful stage execution.
    """
    cleaned_stage_name = (
        validate_stage_name(
            stage_name
        )
    )
    execution_id = (
        get_successful_execution_id(
            manifest,
            attempt_id,
            cleaned_stage_name,
        )
    )

    if execution_id is None:
        raise ArtifactResolutionError(
            "The requested stage has no successful execution.",
            error_code="STAGE_OUTPUTS_NOT_AVAILABLE",
            details={
                "attempt_id": attempt_id,
                "stage_name": (
                    cleaned_stage_name
                ),
            },
        )

    execution = (
        resolve_successful_execution_record(
            manifest,
            attempt_id,
            cleaned_stage_name,
        )
    )

    raw_outputs = execution.get(
        "outputs",
        [],
    )

    if not isinstance(
        raw_outputs,
        list,
    ):
        raise ArtifactResolutionError(
            "The execution output registry is invalid.",
            error_code="INVALID_EXECUTION_OUTPUT_REGISTRY",
            details={
                "attempt_id": attempt_id,
                "stage_name": (
                    cleaned_stage_name
                ),
                "execution_id": (
                    execution_id
                ),
            },
        )

    references: list[
        ArtifactReference
    ] = []

    for item in raw_outputs:
        if not isinstance(
            item,
            Mapping,
        ):
            raise ArtifactResolutionError(
                "An execution output record is invalid.",
                error_code="INVALID_EXECUTION_OUTPUT_RECORD",
                details={
                    "attempt_id": (
                        attempt_id
                    ),
                    "stage_name": (
                        cleaned_stage_name
                    ),
                    "execution_id": (
                        execution_id
                    ),
                },
            )

        item_type = item.get(
            "output_type"
        )

        if (
            output_type is not None
            and item_type != output_type
        ):
            continue

        file_path = item.get(
            "file_path"
        )

        if not file_path:
            raise ArtifactResolutionError(
                "An execution output record does not contain file_path.",
                error_code="EXECUTION_OUTPUT_PATH_MISSING",
                details={
                    "attempt_id": (
                        attempt_id
                    ),
                    "stage_name": (
                        cleaned_stage_name
                    ),
                    "execution_id": (
                        execution_id
                    ),
                },
            )

        references.append(
            ArtifactReference(
                path=_safe_run_path(
                    run_directory,
                    str(
                        file_path
                    ),
                    must_exist=True,
                    must_be_file=True,
                ),
                artifact_type=(
                    "registered_output"
                ),
                attempt_id=(
                    attempt_id
                ),
                stage_name=(
                    cleaned_stage_name
                ),
                execution_id=(
                    execution_id
                ),
                output_type=(
                    str(
                        item_type
                    )
                    if item_type
                    is not None
                    else None
                ),
            )
        )

    return tuple(
        references
    )


def resolve_prerequisite_results(
    manifest: Mapping[str, Any],
    run_directory: str | Path,
    attempt_id: str,
    stage_name: str,
) -> tuple[ArtifactReference, ...]:
    """
    Resolve the immediate stage-result dependency for one workflow stage.
    """
    cleaned_stage_name = (
        validate_stage_name(
            stage_name
        )
    )

    prerequisite = PREREQUISITE_STAGE[
        cleaned_stage_name
    ]

    if prerequisite is None:
        return ()

    return (
        resolve_stage_result(
            manifest,
            run_directory,
            attempt_id,
            prerequisite,
        ),
    )


def get_inference_state_directory(
    run_directory: str | Path,
    attempt_id: str,
    *,
    require_config: bool = False,
) -> Path:
    """
    Return the stable self-contained Part E saved-run directory for an attempt.
    """
    cleaned_attempt_id = (
        validate_record_id(
            attempt_id,
            field_name="attempt_id",
        )
    )

    run_root = Path(
        run_directory
    ).expanduser().resolve()

    state_directory = (
        run_root
        / "inference"
        / cleaned_attempt_id
    ).resolve()

    _safe_run_path(
        run_root,
        state_directory,
        must_exist=(
            require_config
        ),
        must_be_file=False,
    )

    if require_config:
        config_file = _safe_run_path(
            run_root,
            (
                state_directory
                / "inference_config.json"
            ),
            must_exist=True,
            must_be_file=True,
        )

        # Keep a named local variable for debugging clarity.
        _ = config_file

    return state_directory


def resolve_active_inference_state(
    manifest: Mapping[str, Any],
    run_directory: str | Path,
) -> tuple[str, Path]:
    """
    Resolve the active inference attempt and its complete saved-run root.
    """
    attempt_id = (
        get_active_inference_attempt_id(
            manifest
        )
    )

    state_directory = (
        get_inference_state_directory(
            run_directory,
            attempt_id,
            require_config=True,
        )
    )

    return (
        attempt_id,
        state_directory,
    )


__all__ = [
    "resolve_attempt_input",
    "resolve_successful_execution_record",
    "resolve_stage_result",
    "resolve_stage_outputs",
    "resolve_prerequisite_results",
    "get_inference_state_directory",
    "resolve_active_inference_state",
]
