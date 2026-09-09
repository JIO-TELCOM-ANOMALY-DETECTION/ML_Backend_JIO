"""
Production Part C workflow.

Part C reconstructs the authoritative successful Part B handoff, runs the
already-tested feature-engineering runner once for every Part B KPI, validates
the persisted C->D contract, and saves the exact FeatureEngineeringRunnerResult
objects required by later Part D execution.

Run Management owns:
- run/attempt/execution identity;
- prerequisite resolution;
- completed-stage immutability and retry legality;
- result/error persistence and manifest transitions.

This workflow owns only Part C orchestration.
"""

from __future__ import annotations

from logging import Logger
from pathlib import Path
from typing import Any, Sequence

from app.core.exceptions import RunManagementError
from app.ml.feature_engineering import (
    FeatureEngineeringRunnerResult,
    run_feature_engineering_pipeline,
)
from app.ml.preprocessing.runner import PreprocessingRunnerResult
from app.run_management import RunManager
from app.schemas.workflows import (
    FeatureEngineeringStageRequest,
    StageExecutionSummary,
    StandaloneStageResponse,
    WorkflowErrorDetail,
)
from app.services.file_service import ensure_directory
from app.services.run_manifest_service import (
    convert_exception_to_record,
)

from .common import (
    normalize_kpi_name,
    resolve_feature_engineering_config,
    safe_kpi_directory_name,
)
from .handoffs import (
    PartBHandoff,
    build_part_c_result_snapshot,
    load_part_b_handoff,
)


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
    request: FeatureEngineeringStageRequest,
    execution_id: str,
    error: Exception,
) -> StandaloneStageResponse:
    """Build the standard failed Part C response after failure is persisted."""
    detail = _error_detail(
        error
    )

    return StandaloneStageResponse(
        status="failed",
        run_id=request.run_id,
        attempt_id=request.attempt_id,
        requested_stage="part_c",
        executions=(
            StageExecutionSummary(
                stage="part_c",
                execution_id=execution_id,
                status="failed",
                error=detail,
            ),
        ),
        message="Part C feature engineering failed.",
        next_stage="part_c",
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
    Best-effort mark an already-started Part C execution failed.

    If failure persistence itself fails, the original exception is propagated
    rather than being masked by a second persistence exception.
    """
    try:
        manager.fail_stage(
            execution,
            error,
        )
    except Exception:
        return False

    return True


def _resolve_request_feature_config(
    request: FeatureEngineeringStageRequest,
) -> Any:
    """
    Resolve the validated Part C request without inventing a profile.

    The workflow request schema guarantees that model_type_hint exists whenever
    feature_engineering is omitted.
    """
    if request.feature_engineering is not None:
        return request.feature_engineering

    if request.model_type_hint is None:
        # Defensive guard in case this workflow is called with an object that
        # bypassed normal Pydantic validation.
        raise RunManagementError(
            "Part C requires model_type_hint when feature_engineering is "
            "omitted.",
            error_code="PART_C_MODEL_TYPE_HINT_MISSING",
        )

    return resolve_feature_engineering_config(
        None,
        model_type=request.model_type_hint,
    )


def _unique_kpi_directory_names(
    handoff: PartBHandoff,
) -> tuple[str, ...]:
    """
    Build deterministic unique filesystem names for all Part B KPI results.

    Distinct logical KPI names may sanitize to the same path component. Only
    workflow-owned directory names are disambiguated; the original KPI identity
    stored in runner results remains unchanged.
    """
    used: set[str] = set()
    names: list[str] = []

    for index, preprocessing_result in enumerate(
        handoff.preprocessing_results,
        start=1,
    ):
        base = safe_kpi_directory_name(
            preprocessing_result.kpi_name,
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


def _validate_one_result_identity(
    *,
    preprocessing_result: PreprocessingRunnerResult,
    result: FeatureEngineeringRunnerResult,
) -> str:
    """
    Require the Part C result to retain the KPI identity supplied by Part B.
    """
    expected = normalize_kpi_name(
        preprocessing_result.kpi_name
    )
    actual = normalize_kpi_name(
        result.kpi_name
    )

    if actual != expected:
        raise RunManagementError(
            "Part C returned a KPI identity different from its Part B input.",
            error_code="PART_C_KPI_IDENTITY_MISMATCH",
            details={
                "expected_kpi": expected,
                "actual_kpi": actual,
            },
        )

    return actual


def _require_file_below(
    path: str | Path,
    *,
    root: Path,
    role: str,
    kpi_name: str,
) -> Path:
    """Require one Part C output file to exist below the KPI output root."""
    resolved = Path(
        path
    ).expanduser().resolve()

    try:
        resolved.relative_to(
            root
        )
    except ValueError as error:
        raise RunManagementError(
            f"{role} is outside the workflow-owned Part C KPI output "
            "directory.",
            error_code="PART_C_OUTPUT_PATH_OUTSIDE_KPI_DIRECTORY",
            details={
                "kpi_name": kpi_name,
                "role": role,
                "path": str(
                    resolved
                ),
                "output_directory": str(
                    root
                ),
            },
        ) from error

    if not resolved.is_file():
        raise RunManagementError(
            f"{role} does not exist.",
            error_code="PART_C_OUTPUT_FILE_MISSING",
            details={
                "kpi_name": kpi_name,
                "role": role,
                "path": str(
                    resolved
                ),
            },
        )

    return resolved


def _validate_result_contract(
    *,
    preprocessing_result: PreprocessingRunnerResult,
    result: FeatureEngineeringRunnerResult,
    output_directory: Path,
) -> str:
    """
    Validate the stable Part C contract that later Part D will trust.

    The Part C runner performs its own internal validation. These checks protect
    the persisted cross-stage boundary against accidental runner contract drift.
    """
    kpi_name = _validate_one_result_identity(
        preprocessing_result=(
            preprocessing_result
        ),
        result=result,
    )

    expected_root = Path(
        output_directory
    ).expanduser().resolve()

    reported_output_directory = Path(
        result.output_directory
    ).expanduser().resolve()

    if reported_output_directory != expected_root:
        raise RunManagementError(
            "Part C returned an output_directory different from the "
            "workflow-owned KPI directory.",
            error_code="PART_C_OUTPUT_DIRECTORY_MISMATCH",
            details={
                "kpi_name": kpi_name,
                "expected_output_directory": str(
                    expected_root
                ),
                "reported_output_directory": str(
                    reported_output_directory
                ),
            },
        )

    expected_source_file = Path(
        preprocessing_result.final_dataset_file
    ).expanduser().resolve()

    reported_source_file = Path(
        result.source_file
    ).expanduser().resolve()

    if reported_source_file != expected_source_file:
        raise RunManagementError(
            "Part C source_file does not match the authoritative Part B final "
            "dataset.",
            error_code="PART_C_SOURCE_FILE_MISMATCH",
            details={
                "kpi_name": kpi_name,
                "expected_source_file": str(
                    expected_source_file
                ),
                "reported_source_file": str(
                    reported_source_file
                ),
            },
        )

    _require_file_below(
        result.final_dataset_file,
        root=expected_root,
        role="Part C final dataset",
        kpi_name=kpi_name,
    )

    _require_file_below(
        result.site_summary_file,
        root=expected_root,
        role="Part C site summary",
        kpi_name=kpi_name,
    )

    if result.corrected_history_audit_file is not None:
        _require_file_below(
            result.corrected_history_audit_file,
            root=expected_root,
            role="Part C corrected-history audit",
            kpi_name=kpi_name,
        )

    feature_columns = tuple(
        result.feature_columns
    )
    resolved_feature_columns = tuple(
        result.resolved_config.feature_columns
    )

    if feature_columns != resolved_feature_columns:
        raise RunManagementError(
            "Part C feature_columns do not match its resolved feature "
            "configuration.",
            error_code="PART_C_FEATURE_CONTRACT_MISMATCH",
            details={
                "kpi_name": kpi_name,
            },
        )

    minimum_history_points = int(
        result.resolved_config.minimum_history_points
    )

    if minimum_history_points < 1:
        raise RunManagementError(
            "Part C resolved minimum_history_points must be at least one.",
            error_code="PART_C_MINIMUM_HISTORY_INVALID",
            details={
                "kpi_name": kpi_name,
                "minimum_history_points": (
                    minimum_history_points
                ),
            },
        )

    if int(
        result.total_rows
    ) < 0:
        raise RunManagementError(
            "Part C reported a negative total_rows value.",
            error_code="PART_C_RESULT_COUNT_INVALID",
            details={
                "kpi_name": kpi_name,
                "field": "total_rows",
                "value": int(
                    result.total_rows
                ),
            },
        )

    for field_name in (
        "total_sites",
        "processed_sites",
        "skipped_invalid_sites",
        "insufficient_history_sites",
        "populated_feature_cells",
        "missing_feature_cells",
    ):
        value = int(
            getattr(
                result,
                field_name,
            )
        )

        if value < 0:
            raise RunManagementError(
                "Part C returned a negative headline count.",
                error_code="PART_C_RESULT_COUNT_INVALID",
                details={
                    "kpi_name": kpi_name,
                    "field": field_name,
                    "value": value,
                },
            )

    if (
        int(
            result.processed_sites
        )
        + int(
            result.skipped_invalid_sites
        )
        > int(
            result.total_sites
        )
    ):
        raise RunManagementError(
            "Part C site counts are internally inconsistent.",
            error_code="PART_C_SITE_COUNTS_INCONSISTENT",
            details={
                "kpi_name": kpi_name,
                "total_sites": int(
                    result.total_sites
                ),
                "processed_sites": int(
                    result.processed_sites
                ),
                "skipped_invalid_sites": int(
                    result.skipped_invalid_sites
                ),
            },
        )

    return kpi_name


def _validate_part_c_kpi_continuity(
    *,
    handoff: PartBHandoff,
    results: Sequence[
        FeatureEngineeringRunnerResult
    ],
) -> tuple[str, ...]:
    """
    Require exactly one Part C result for every KPI supplied by Part B.

    No KPI may disappear, be added, or be duplicated before the successful
    Part C snapshot is persisted.
    """
    expected_names = tuple(
        normalize_kpi_name(
            preprocessing_result.kpi_name
        )
        for preprocessing_result
        in handoff.preprocessing_results
    )

    actual_names = tuple(
        normalize_kpi_name(
            result.kpi_name
        )
        for result in results
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
            "Part C KPI results do not exactly match the KPI set produced by "
            "Part B.",
            error_code="PART_C_KPI_SET_MISMATCH",
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


def run_part_c_workflow(
    *,
    request: FeatureEngineeringStageRequest,
    manager: RunManager,
    logger: Logger,
) -> StandaloneStageResponse:
    """
    Execute one production Part C stage.

    Run Management resolves the authoritative Part B result before this
    workflow reconstructs it. Part C is then called once for every persisted
    PreprocessingRunnerResult.
    """

    # ------------------------------------------------------------------
    # 1. Create/validate Part C execution and resolve the Part B prerequisite.
    # ------------------------------------------------------------------
    execution = manager.prepare_stage_execution(
        request.run_id,
        request.attempt_id,
        "part_c",
        request_payload=request,
        metadata={
            "workflow": "part_c",
        },
    )

    # ------------------------------------------------------------------
    # 2. Reconstruct Part B, resolve the Part C request, and execute every KPI.
    # ------------------------------------------------------------------
    try:
        if len(
            execution.prerequisite_results
        ) != 1:
            raise RunManagementError(
                "Part C expected exactly one authoritative Part B result.",
                error_code="PART_C_PREREQUISITE_RESULT_COUNT_INVALID",
                details={
                    "received_count": len(
                        execution.prerequisite_results
                    ),
                },
            )

        handoff = load_part_b_handoff(
            result_file=(
                execution
                .prerequisite_results[0]
                .path
            ),
            run_directory=(
                execution.run_directory
            ),
        )

        if not handoff.preprocessing_results:
            # load_part_b_handoff() already enforces this; keep the explicit
            # workflow guard so the execution contract remains self-documenting.
            raise RunManagementError(
                "Part C received no KPI results from Part B.",
                error_code="PART_C_PART_B_KPI_RESULTS_EMPTY",
            )

        feature_config = (
            _resolve_request_feature_config(
                request
            )
        )

        directory_names = (
            _unique_kpi_directory_names(
                handoff
            )
        )

        results: list[
            FeatureEngineeringRunnerResult
        ] = []

        for (
            preprocessing_result,
            directory_name,
        ) in zip(
            handoff.preprocessing_results,
            directory_names,
            strict=True,
        ):
            output_directory = (
                ensure_directory(
                    execution.output_directory
                    / directory_name
                )
            )

            result = (
                run_feature_engineering_pipeline(
                    preprocessing_result=(
                        preprocessing_result
                    ),
                    output_directory=(
                        output_directory
                    ),
                    dataset_config=(
                        handoff.dataset_config
                    ),
                    feature_config=(
                        feature_config
                    ),
                    bucket=handoff.bucket,
                    logger=logger,
                )
            )

            _validate_result_contract(
                preprocessing_result=(
                    preprocessing_result
                ),
                result=result,
                output_directory=(
                    output_directory
                ),
            )

            results.append(
                result
            )

        kpi_names = (
            _validate_part_c_kpi_continuity(
                handoff=handoff,
                results=results,
            )
        )

        # Persist exact FeatureEngineeringRunnerResult objects for later
        # annotation-driven reconstruction by Part D.
        snapshot = build_part_c_result_snapshot(
            dataset_config=(
                handoff.dataset_config
            ),
            feature_engineering_results=tuple(
                results
            ),
        )

        summary = {
            "kpi_count": len(
                results
            ),
            "kpi_names": list(
                kpi_names
            ),
            "total_rows": sum(
                int(
                    result.total_rows
                )
                for result in results
            ),
            "total_sites": sum(
                int(
                    result.total_sites
                )
                for result in results
            ),
            "processed_sites": sum(
                int(
                    result.processed_sites
                )
                for result in results
            ),
            "skipped_invalid_sites": sum(
                int(
                    result.skipped_invalid_sites
                )
                for result in results
            ),
            "insufficient_history_sites": sum(
                int(
                    result.insufficient_history_sites
                )
                for result in results
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
    # 3. Finalize result.json + Manifest V2 through Run Management.
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
    # 4. Public response exposes workflow IDs/status/summary, never paths.
    # ------------------------------------------------------------------
    return StandaloneStageResponse(
        status="completed",
        run_id=request.run_id,
        attempt_id=request.attempt_id,
        requested_stage="part_c",
        executions=(
            StageExecutionSummary(
                stage="part_c",
                execution_id=(
                    execution.execution_id
                ),
                status="completed",
                summary=summary,
            ),
        ),
        message=(
            "Part C feature engineering completed successfully."
        ),
        next_stage="part_d",
        inference_ready=False,
        summary=summary,
    )


__all__ = [
    "run_part_c_workflow",
]
