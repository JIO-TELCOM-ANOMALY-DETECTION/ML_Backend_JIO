"""
Production Part D workflow.

Part D reconstructs the authoritative successful Part C handoff, calls the
already-tested public ``run_training()`` entry point once for every KPI, checks
the finalized training package returned by Part D, and persists the D->bootstrap
snapshot required by Phase 5.

This file intentionally executes only the canonical ``part_d`` stage.
``inference_bootstrap`` is a separate stage and is composed with Part D by the
Phase 5 training lifecycle.

Run Management owns:
- run/attempt/execution identity;
- prerequisite resolution;
- completed-stage immutability and retry legality;
- result/error persistence and manifest transitions.

The Part D ML package owns:
- model dispatch;
- fixed/Optuna execution;
- per-site training;
- validation/splits;
- model serialization;
- final Part D training-package creation.
"""

from __future__ import annotations

from logging import Logger
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.core.exceptions import RunManagementError
from app.ml.training import (
    PartDTrainingResult,
    run_training,
)
from app.run_management import RunManager
from app.schemas.workflows import (
    StageExecutionSummary,
    StandaloneStageResponse,
    TrainingStageRequest,
    WorkflowErrorDetail,
)
from app.services.file_service import ensure_directory
from app.services.run_manifest_service import (
    convert_exception_to_record,
)

from .common import (
    normalize_kpi_name,
    safe_kpi_directory_name,
    to_json_safe,
)
from .handoffs import (
    PartCHandoff,
    PartDRunnerInput,
    build_part_d_runner_inputs,
    load_part_c_handoff,
)


PART_D_SNAPSHOT_VERSION = "1.0"


def _error_detail(
    error: Exception,
) -> WorkflowErrorDetail:
    """Convert one backend exception into the workflow response error shape."""
    return WorkflowErrorDetail.model_validate(
        convert_exception_to_record(
            error
        )
    )


def _failed_response(
    *,
    request: TrainingStageRequest,
    execution_id: str,
    error: Exception,
) -> StandaloneStageResponse:
    """Build the standard failed Part D response after failure is persisted."""
    detail = _error_detail(
        error
    )

    return StandaloneStageResponse(
        status="failed",
        run_id=request.run_id,
        attempt_id=request.attempt_id,
        requested_stage="part_d",
        executions=(
            StageExecutionSummary(
                stage="part_d",
                execution_id=execution_id,
                status="failed",
                error=detail,
            ),
        ),
        message="Part D training failed.",
        next_stage="part_d",
        inference_ready=False,
        error=detail,
    )


def _persist_started_failure(
    *,
    manager: RunManager,
    execution: Any,
    error: Exception,
) -> bool:
    """
    Best-effort mark an already-started Part D execution failed.

    If failure persistence itself fails, the original exception is propagated
    instead of being masked by a second persistence exception.
    """
    try:
        manager.fail_stage(
            execution,
            error,
        )
    except Exception:
        return False

    return True


def _unique_kpi_directory_names(
    handoff: PartCHandoff,
) -> tuple[str, ...]:
    """
    Build deterministic unique filesystem names for all Part C KPI results.

    Distinct KPI identities may sanitize to the same path component. Only
    workflow-owned directory names are disambiguated; the logical KPI names
    stored in training metadata/snapshots remain unchanged.
    """
    used: set[str] = set()
    names: list[str] = []

    for index, result in enumerate(
        handoff.feature_engineering_results,
        start=1,
    ):
        base = safe_kpi_directory_name(
            result.kpi_name,
            fallback_index=index,
        )

        candidate = base
        suffix = index

        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1

        used.add(
            candidate
        )
        names.append(
            candidate
        )

    return tuple(
        names
    )


def _require_path_below(
    path: str | Path,
    *,
    root: Path,
    role: str,
    kpi_name: str,
    must_be_file: bool | None,
) -> Path:
    """
    Require a finalized Part D artifact to remain under its KPI output root.
    """
    resolved = Path(
        path
    ).expanduser().resolve()

    expected_root = Path(
        root
    ).expanduser().resolve()

    try:
        resolved.relative_to(
            expected_root
        )
    except ValueError as error:
        raise RunManagementError(
            "Part D returned an artifact outside its workflow-owned KPI "
            "output directory.",
            error_code="PART_D_OUTPUT_PATH_OUTSIDE_KPI_DIRECTORY",
            details={
                "kpi_name": kpi_name,
                "role": role,
                "path": str(
                    resolved
                ),
                "output_directory": str(
                    expected_root
                ),
            },
        ) from error

    if not resolved.exists():
        raise RunManagementError(
            "Part D returned an artifact that does not exist.",
            error_code="PART_D_OUTPUT_ARTIFACT_MISSING",
            details={
                "kpi_name": kpi_name,
                "role": role,
                "path": str(
                    resolved
                ),
            },
        )

    if (
        must_be_file is True
        and not resolved.is_file()
    ):
        raise RunManagementError(
            "Part D expected a file artifact but received a different "
            "filesystem object.",
            error_code="PART_D_OUTPUT_ARTIFACT_TYPE_INVALID",
            details={
                "kpi_name": kpi_name,
                "role": role,
                "path": str(
                    resolved
                ),
            },
        )

    if (
        must_be_file is False
        and not resolved.is_dir()
    ):
        raise RunManagementError(
            "Part D expected a directory artifact but received a different "
            "filesystem object.",
            error_code="PART_D_OUTPUT_ARTIFACT_TYPE_INVALID",
            details={
                "kpi_name": kpi_name,
                "role": role,
                "path": str(
                    resolved
                ),
            },
        )

    return resolved


def _validate_series_counts(
    *,
    result: PartDTrainingResult,
    kpi_name: str,
) -> None:
    """
    Validate the aggregate series invariants guaranteed by the Part D runner.

    The runner rejects a KPI with no valid series and fails the complete KPI run
    if any valid series cannot be trained. Therefore every valid series must
    correspond to one trained model in a successful result.
    """
    total_series = int(
        result.total_series
    )
    valid_series = int(
        result.valid_series
    )
    trained_series = int(
        result.trained_series
    )
    skipped_invalid_series = int(
        result.skipped_invalid_series
    )

    values = {
        "total_series": total_series,
        "valid_series": valid_series,
        "trained_series": trained_series,
        "skipped_invalid_series": (
            skipped_invalid_series
        ),
    }

    for field_name, value in values.items():
        if value < 0:
            raise RunManagementError(
                "Part D returned a negative series count.",
                error_code="PART_D_SERIES_COUNT_INVALID",
                details={
                    "kpi_name": kpi_name,
                    "field": field_name,
                    "value": value,
                },
            )

    if valid_series < 1:
        raise RunManagementError(
            "A successful Part D result must contain at least one valid "
            "series.",
            error_code="PART_D_VALID_SERIES_EMPTY",
            details={
                "kpi_name": kpi_name,
            },
        )

    if (
        total_series
        != valid_series
        + skipped_invalid_series
    ):
        raise RunManagementError(
            "Part D total/valid/skipped series counts are inconsistent.",
            error_code="PART_D_SERIES_COUNTS_INCONSISTENT",
            details={
                "kpi_name": kpi_name,
                **values,
            },
        )

    if trained_series != valid_series:
        raise RunManagementError(
            "A successful Part D result did not train every valid series.",
            error_code="PART_D_TRAINED_SERIES_MISMATCH",
            details={
                "kpi_name": kpi_name,
                "valid_series": valid_series,
                "trained_series": trained_series,
            },
        )

    if len(
        result.skipped_series
    ) != skipped_invalid_series:
        raise RunManagementError(
            "Part D skipped-series details do not match the skipped count.",
            error_code="PART_D_SKIPPED_SERIES_COUNT_MISMATCH",
            details={
                "kpi_name": kpi_name,
                "skipped_invalid_series": (
                    skipped_invalid_series
                ),
                "skipped_series_records": len(
                    result.skipped_series
                ),
            },
        )


def _validate_training_result(
    *,
    runner_input: PartDRunnerInput,
    result: PartDTrainingResult,
    output_directory: Path,
    requested_model_type: str,
) -> dict[str, Any]:
    """
    Validate one completed Part D result and return its bootstrap record.

    This is a cross-stage persistence check, not a reimplementation of Part D
    training validation.
    """
    kpi_name = normalize_kpi_name(
        runner_input.kpi_name
    )

    if str(
        result.status
    ) != "completed":
        raise RunManagementError(
            "Part D returned a non-completed result without raising.",
            error_code="PART_D_RESULT_STATUS_INVALID",
            details={
                "kpi_name": kpi_name,
                "status": str(
                    result.status
                ),
            },
        )

    if str(
        result.model_type
    ) != str(
        requested_model_type
    ):
        raise RunManagementError(
            "Part D returned a model_type different from the training request.",
            error_code="PART_D_MODEL_TYPE_MISMATCH",
            details={
                "kpi_name": kpi_name,
                "expected_model_type": str(
                    requested_model_type
                ),
                "actual_model_type": str(
                    result.model_type
                ),
            },
        )

    expected_input_file = Path(
        runner_input.input_file
    ).expanduser().resolve()

    actual_input_file = Path(
        result.input_file
    ).expanduser().resolve()

    if actual_input_file != expected_input_file:
        raise RunManagementError(
            "Part D result references a different Part C dataset than the one "
            "supplied by the workflow.",
            error_code="PART_D_INPUT_FILE_MISMATCH",
            details={
                "kpi_name": kpi_name,
                "expected_input_file": str(
                    expected_input_file
                ),
                "actual_input_file": str(
                    actual_input_file
                ),
            },
        )

    expected_output_directory = Path(
        output_directory
    ).expanduser().resolve()

    actual_output_directory = Path(
        result.output_directory
    ).expanduser().resolve()

    if actual_output_directory != expected_output_directory:
        raise RunManagementError(
            "Part D result output_directory differs from the workflow-owned "
            "KPI directory.",
            error_code="PART_D_OUTPUT_DIRECTORY_MISMATCH",
            details={
                "kpi_name": kpi_name,
                "expected_output_directory": str(
                    expected_output_directory
                ),
                "actual_output_directory": str(
                    actual_output_directory
                ),
            },
        )

    if int(
        result.minimum_history_points
    ) != int(
        runner_input.minimum_history_points
    ):
        raise RunManagementError(
            "Part D returned a different minimum-history contract than Part C.",
            error_code="PART_D_MINIMUM_HISTORY_MISMATCH",
            details={
                "kpi_name": kpi_name,
                "expected_minimum_history_points": int(
                    runner_input.minimum_history_points
                ),
                "actual_minimum_history_points": int(
                    result.minimum_history_points
                ),
            },
        )

    validation = result.validation_result

    if str(
        validation.model_type
    ) != str(
        requested_model_type
    ):
        raise RunManagementError(
            "Part D validation_result model_type differs from the training "
            "request.",
            error_code="PART_D_VALIDATION_MODEL_TYPE_MISMATCH",
            details={
                "kpi_name": kpi_name,
            },
        )

    if Path(
        validation.input_file
    ).expanduser().resolve() != expected_input_file:
        raise RunManagementError(
            "Part D validation_result references a different input dataset.",
            error_code="PART_D_VALIDATION_INPUT_MISMATCH",
            details={
                "kpi_name": kpi_name,
            },
        )

    if int(
        validation.minimum_history_points
    ) != int(
        runner_input.minimum_history_points
    ):
        raise RunManagementError(
            "Part D validation_result minimum-history contract differs from "
            "Part C.",
            error_code="PART_D_VALIDATION_HISTORY_MISMATCH",
            details={
                "kpi_name": kpi_name,
            },
        )

    model_feature_columns = tuple(
        str(
            value
        )
        for value in (
            validation
            .model_feature_columns
        )
    )

    if not model_feature_columns:
        raise RunManagementError(
            "Part D validation_result contains no model feature columns.",
            error_code="PART_D_MODEL_FEATURES_EMPTY",
            details={
                "kpi_name": kpi_name,
            },
        )

    if len(
        model_feature_columns
    ) != len(
        set(
            model_feature_columns
        )
    ):
        raise RunManagementError(
            "Part D validation_result contains duplicate model feature columns.",
            error_code="PART_D_MODEL_FEATURES_DUPLICATE",
            details={
                "kpi_name": kpi_name,
            },
        )

    target_column = validation.target_column

    if (
        requested_model_type == "xgboost"
        and (
            target_column is None
            or not str(
                target_column
            ).strip()
        )
    ):
        raise RunManagementError(
            "Successful XGBoost training did not expose a target column.",
            error_code="PART_D_XGBOOST_TARGET_MISSING",
            details={
                "kpi_name": kpi_name,
            },
        )

    if (
        requested_model_type
        == "isolation_forest"
        and target_column is not None
    ):
        raise RunManagementError(
            "Isolation Forest training unexpectedly exposed a target column.",
            error_code="PART_D_ISOLATION_FOREST_TARGET_PRESENT",
            details={
                "kpi_name": kpi_name,
                "target_column": str(
                    target_column
                ),
            },
        )

    _validate_series_counts(
        result=result,
        kpi_name=kpi_name,
    )

    output_paths = result.output_paths

    training_directory = _require_path_below(
        output_paths.training_directory,
        root=expected_output_directory,
        role="training directory",
        kpi_name=kpi_name,
        must_be_file=False,
    )

    model_directory = _require_path_below(
        output_paths.model_directory,
        root=training_directory,
        role="model directory",
        kpi_name=kpi_name,
        must_be_file=False,
    )

    model_files = tuple(
        _require_path_below(
            path,
            root=model_directory,
            role="model file",
            kpi_name=kpi_name,
            must_be_file=True,
        )
        for path in output_paths.model_files
    )

    if len(
        model_files
    ) != int(
        result.trained_series
    ):
        raise RunManagementError(
            "Part D model-file count does not match trained_series.",
            error_code="PART_D_MODEL_FILE_COUNT_MISMATCH",
            details={
                "kpi_name": kpi_name,
                "trained_series": int(
                    result.trained_series
                ),
                "model_file_count": len(
                    model_files
                ),
            },
        )

    if requested_model_type == "xgboost":
        invalid_suffixes = [
            str(
                path
            )
            for path in model_files
            if path.suffix.lower() != ".json"
        ]
    else:
        invalid_suffixes = [
            str(
                path
            )
            for path in model_files
            if path.suffix.lower() != ".joblib"
        ]

    if invalid_suffixes:
        raise RunManagementError(
            "One or more saved model files have a suffix inconsistent with "
            "the selected model type.",
            error_code="PART_D_MODEL_FILE_SUFFIX_MISMATCH",
            details={
                "kpi_name": kpi_name,
                "model_type": str(
                    requested_model_type
                ),
                "invalid_model_files": (
                    invalid_suffixes
                ),
            },
        )

    # Require the complete finalized model-output package, not only the subset
    # immediately consumed by inference bootstrap.
    package_files = {
        "split_plan_file": (
            output_paths.split_plan_file
        ),
        "training_config_file": (
            output_paths.training_config_file
        ),
        "feature_contract_file": (
            output_paths.feature_contract_file
        ),
        "final_hyperparameters_file": (
            output_paths.final_hyperparameters_file
        ),
        "training_summary_file": (
            output_paths.training_summary_file
        ),
        "decision_settings_file": (
            output_paths.decision_settings_file
        ),
        "usable_row_counts_file": (
            output_paths.usable_row_counts_file
        ),
        "metadata_file": (
            output_paths.metadata_file
        ),
    }

    validated_package_files = {
        role: _require_path_below(
            path,
            root=training_directory,
            role=role,
            kpi_name=kpi_name,
            must_be_file=True,
        )
        for role, path
        in package_files.items()
    }

    return {
        "kpi_name": kpi_name,
        "status": "completed",
        "model_type": str(
            result.model_type
        ),
        "training_directory": str(
            training_directory
        ),
        "model_directory": str(
            model_directory
        ),
        "metadata_file": str(
            validated_package_files[
                "metadata_file"
            ]
        ),
        "decision_settings_file": str(
            validated_package_files[
                "decision_settings_file"
            ]
        ),
        "model_files": [
            str(
                path
            )
            for path in model_files
        ],
        "model_feature_columns": list(
            model_feature_columns
        ),
        "target_column": (
            str(
                target_column
            )
            if target_column is not None
            else None
        ),
        "minimum_history_points": int(
            result.minimum_history_points
        ),
        "total_series": int(
            result.total_series
        ),
        "valid_series": int(
            result.valid_series
        ),
        "trained_series": int(
            result.trained_series
        ),
        "skipped_invalid_series": int(
            result.skipped_invalid_series
        ),
    }


def _validate_part_d_kpi_continuity(
    *,
    runner_inputs: Sequence[
        PartDRunnerInput
    ],
    records: Sequence[
        Mapping[str, Any]
    ],
) -> tuple[str, ...]:
    """
    Require exactly one completed Part D record for every Part C KPI.
    """
    expected_names = tuple(
        normalize_kpi_name(
            item.kpi_name
        )
        for item in runner_inputs
    )

    actual_names = tuple(
        normalize_kpi_name(
            record.get(
                "kpi_name"
            )
        )
        for record in records
    )

    if (
        len(
            actual_names
        )
        != len(
            expected_names
        )
        or len(
            set(
                actual_names
            )
        )
        != len(
            actual_names
        )
        or set(
            actual_names
        )
        != set(
            expected_names
        )
    ):
        raise RunManagementError(
            "Part D KPI results do not exactly match the KPI set supplied by "
            "Part C.",
            error_code="PART_D_KPI_SET_MISMATCH",
            details={
                "expected_kpis": sorted(
                    set(
                        expected_names
                    )
                ),
                "actual_kpis": sorted(
                    set(
                        actual_names
                    )
                ),
                "expected_count": len(
                    expected_names
                ),
                "actual_count": len(
                    actual_names
                ),
            },
        )

    return actual_names


def build_part_d_result_snapshot(
    *,
    dataset_config: Any,
    training_config: Any,
    records: Sequence[
        Mapping[str, Any]
    ],
) -> dict[str, Any]:
    """
    Build the persisted Part D -> inference-bootstrap contract.

    The bootstrap stage will reload this snapshot after D may have completed in
    a previous request/process.
    """
    materialized_records = tuple(
        dict(
            record
        )
        for record in records
    )

    if not materialized_records:
        raise RunManagementError(
            "Part D cannot persist an empty training snapshot.",
            error_code="PART_D_TRAINING_RECORDS_EMPTY",
        )

    return {
        "part_d_snapshot_version": (
            PART_D_SNAPSHOT_VERSION
        ),
        "stage": "part_d",
        "dataset_config": to_json_safe(
            dataset_config
        ),
        "training_config": to_json_safe(
            training_config
        ),
        "training_records": to_json_safe(
            materialized_records
        ),
    }


def run_part_d_workflow(
    *,
    request: TrainingStageRequest,
    manager: RunManager,
    logger: Logger,
) -> StandaloneStageResponse:
    """
    Execute only the canonical Part D stage.

    The final production training lifecycle is composed in Phase 5 as:

        run_part_d_workflow(...)
        -> run_inference_bootstrap_workflow(...)

    Keeping bootstrap separate allows a failed bootstrap to be retried without
    rerunning an already-completed immutable Part D stage.
    """
    del logger  # The public Part D runner does not currently accept a logger.

    # ------------------------------------------------------------------
    # 1. Create/validate Part D execution and resolve the Part C prerequisite.
    # ------------------------------------------------------------------
    execution = manager.prepare_stage_execution(
        request.run_id,
        request.attempt_id,
        "part_d",
        request_payload=request,
        metadata={
            "workflow": "part_d",
        },
    )

    # ------------------------------------------------------------------
    # 2. Reconstruct Part C and train every KPI through the public Part D API.
    # ------------------------------------------------------------------
    try:
        if len(
            execution.prerequisite_results
        ) != 1:
            raise RunManagementError(
                "Part D expected exactly one authoritative Part C result.",
                error_code="PART_D_PREREQUISITE_RESULT_COUNT_INVALID",
                details={
                    "received_count": len(
                        execution.prerequisite_results
                    ),
                },
            )

        handoff = load_part_c_handoff(
            result_file=(
                execution
                .prerequisite_results[0]
                .path
            ),
            run_directory=(
                execution.run_directory
            ),
        )

        runner_inputs = build_part_d_runner_inputs(
            handoff
        )

        if not runner_inputs:
            # load_part_c_handoff() already prevents an empty C result, but the
            # explicit guard keeps the D execution contract self-documenting.
            raise RunManagementError(
                "Part D received no KPI runner inputs from Part C.",
                error_code="PART_D_RUNNER_INPUTS_EMPTY",
            )

        directory_names = (
            _unique_kpi_directory_names(
                handoff
            )
        )

        records: list[
            dict[str, Any]
        ] = []

        total_series = 0
        valid_series = 0
        trained_series = 0
        skipped_invalid_series = 0

        tuning_enabled = bool(
            request
            .training
            .hyperparameter_tuning_enabled
        )

        tuning_method = (
            request
            .training
            .hyperparameter_tuning_method
            if tuning_enabled
            else None
        )

        for (
            runner_input,
            directory_name,
        ) in zip(
            runner_inputs,
            directory_names,
            strict=True,
        ):
            kpi_name = normalize_kpi_name(
                runner_input.kpi_name
            )

            output_directory = ensure_directory(
                execution.output_directory
                / directory_name
            )

            metadata = {
                **dict(
                    runner_input.metadata
                ),
                "hyperparameter_tuning_enabled": (
                    tuning_enabled
                ),
                "hyperparameter_tuning_method": (
                    tuning_method
                ),
            }

            training_result = run_training(
                model_type=(
                    request.training.model_type
                ),
                input_file=(
                    runner_input.input_file
                ),
                output_directory=(
                    output_directory
                ),
                dataset_columns=(
                    runner_input.dataset_columns
                ),
                feature_config=(
                    runner_input.feature_config
                ),
                minimum_history_points=(
                    runner_input
                    .minimum_history_points
                ),
                training_config=(
                    request.training
                ),
                metadata=metadata,
            )

            record = _validate_training_result(
                runner_input=runner_input,
                result=training_result,
                output_directory=(
                    output_directory
                ),
                requested_model_type=(
                    request.training.model_type
                ),
            )

            # _validate_training_result() already confirms the record KPI.
            if record["kpi_name"] != kpi_name:
                raise RunManagementError(
                    "Internal Part D KPI record identity mismatch.",
                    error_code="PART_D_RECORD_KPI_IDENTITY_MISMATCH",
                    details={
                        "expected_kpi": kpi_name,
                        "actual_kpi": record[
                            "kpi_name"
                        ],
                    },
                )

            records.append(
                record
            )

            total_series += int(
                training_result.total_series
            )
            valid_series += int(
                training_result.valid_series
            )
            trained_series += int(
                training_result.trained_series
            )
            skipped_invalid_series += int(
                training_result
                .skipped_invalid_series
            )

        kpi_names = (
            _validate_part_d_kpi_continuity(
                runner_inputs=runner_inputs,
                records=records,
            )
        )

        snapshot = build_part_d_result_snapshot(
            dataset_config=(
                handoff.dataset_config
            ),
            training_config=(
                request.training
            ),
            records=records,
        )

        summary = {
            "model_type": (
                request.training.model_type
            ),
            "kpi_count": len(
                records
            ),
            "kpi_names": list(
                kpi_names
            ),
            "total_series": (
                total_series
            ),
            "valid_series": (
                valid_series
            ),
            "trained_series": (
                trained_series
            ),
            "skipped_invalid_series": (
                skipped_invalid_series
            ),
            "hyperparameter_tuning_enabled": (
                tuning_enabled
            ),
            "hyperparameter_tuning_method": (
                tuning_method
            ),
        }

    except Exception as error:
        if not _persist_started_failure(
            manager=manager,
            execution=execution,
            error=error,
        ):
            raise

        return _failed_response(
            request=request,
            execution_id=(
                execution.execution_id
            ),
            error=error,
        )

    # ------------------------------------------------------------------
    # 3. Persist the immutable D snapshot through Run Management.
    # ------------------------------------------------------------------
    try:
        manager.complete_stage(
            execution,
            result_payload=snapshot,
            metrics=summary,
        )

    except Exception as error:
        if not _persist_started_failure(
            manager=manager,
            execution=execution,
            error=error,
        ):
            raise

        return _failed_response(
            request=request,
            execution_id=(
                execution.execution_id
            ),
            error=error,
        )

    # ------------------------------------------------------------------
    # 4. D alone is not inference-ready. Bootstrap is the next stage.
    # ------------------------------------------------------------------
    return StandaloneStageResponse(
        status="completed",
        run_id=request.run_id,
        attempt_id=request.attempt_id,
        requested_stage="part_d",
        executions=(
            StageExecutionSummary(
                stage="part_d",
                execution_id=(
                    execution.execution_id
                ),
                status="completed",
                summary=summary,
            ),
        ),
        message=(
            "Part D training completed successfully. "
            "Inference bootstrap is the next stage."
        ),
        next_stage="inference_bootstrap",
        inference_ready=False,
        summary=summary,
    )


__all__ = [
    "PART_D_SNAPSHOT_VERSION",
    "build_part_d_result_snapshot",
    "run_part_d_workflow",
]
