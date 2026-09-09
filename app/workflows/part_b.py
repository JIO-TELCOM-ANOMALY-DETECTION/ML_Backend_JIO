"""
Production Part B workflow.

Part B reconstructs the authoritative successful Part A handoff, runs the
already-tested normal/training preprocessing runner once for every KPI returned
by Part A, validates KPI continuity, and persists the exact B->C handoff.

Run Management owns:
- run/attempt/execution identity;
- prerequisite resolution;
- completed-stage immutability/retry legality;
- result/error persistence and manifest transitions.

This workflow owns only Part B orchestration.
"""

from __future__ import annotations

from logging import Logger
from pathlib import Path
from typing import Any, Sequence

from app.core.exceptions import RunManagementError
from app.ml.preprocessing.runner import (
    PreprocessingRunnerResult,
    run_preprocessing_pipeline,
)
from app.run_management import RunManager
from app.schemas.workflows import (
    PreprocessingStageRequest,
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
    safe_kpi_directory_name,
)
from .handoffs import (
    PartAHandoff,
    build_part_b_result_snapshot,
    load_part_a_handoff,
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
    request: PreprocessingStageRequest,
    execution_id: str,
    error: Exception,
) -> StandaloneStageResponse:
    """Build the standard failed Part B response after failure is persisted."""
    detail = _error_detail(
        error
    )

    return StandaloneStageResponse(
        status="failed",
        run_id=request.run_id,
        attempt_id=request.attempt_id,
        requested_stage="part_b",
        executions=(
            StageExecutionSummary(
                stage="part_b",
                execution_id=execution_id,
                status="failed",
                error=detail,
            ),
        ),
        message="Part B preprocessing failed.",
        next_stage="part_b",
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
    Best-effort mark an already-started execution failed.

    If failure persistence itself fails, the original exception is propagated
    instead of being hidden by a second persistence exception.
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
    handoff: PartAHandoff,
) -> tuple[str, ...]:
    """
    Build deterministic unique filesystem names for all Part A KPIs.

    Logical KPI identities are never changed. Only workflow-owned directory
    names are disambiguated when sanitization would collapse two different KPI
    names to the same filename-safe value.
    """
    used: set[str] = set()
    names: list[str] = []

    for index, kpi_result in enumerate(
        handoff.kpi_results,
        start=1,
    ):
        base = safe_kpi_directory_name(
            kpi_result.prepared_dataset.kpi_name,
            fallback_index=index,
        )

        candidate = base
        suffix = index

        while candidate in used:
            candidate = (
                f"{base}_{suffix}"
            )
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
    expected_kpi_name: object,
    result: PreprocessingRunnerResult,
) -> str:
    """
    Require one Part B result to retain the KPI identity it was called for.

    Set-level continuity is checked again after all KPI calls complete.
    """
    expected = normalize_kpi_name(
        expected_kpi_name
    )
    actual = normalize_kpi_name(
        result.kpi_name
    )

    if actual != expected:
        raise RunManagementError(
            "Part B returned a KPI identity different from its Part A input.",
            error_code="PART_B_KPI_IDENTITY_MISMATCH",
            details={
                "expected_kpi": expected,
                "actual_kpi": actual,
            },
        )

    return actual


def _validate_result_output(
    *,
    result: PreprocessingRunnerResult,
    output_directory: Path,
    kpi_name: str,
) -> None:
    """
    Require the stable Part B product to exist inside this KPI execution root.

    Part C consumes ``final_dataset_file`` as its authoritative source, so Part B
    must not be marked successful if that file is missing or escapes the
    workflow-owned KPI directory.
    """
    expected_root = Path(
        output_directory
    ).expanduser().resolve()

    final_dataset_file = Path(
        result.final_dataset_file
    ).expanduser().resolve()

    try:
        final_dataset_file.relative_to(
            expected_root
        )
    except ValueError as error:
        raise RunManagementError(
            "Part B returned a final dataset outside its workflow-owned KPI "
            "output directory.",
            error_code="PART_B_FINAL_DATASET_OUTSIDE_OUTPUT",
            details={
                "kpi_name": kpi_name,
                "output_directory": str(
                    expected_root
                ),
                "final_dataset_file": str(
                    final_dataset_file
                ),
            },
        ) from error

    if not final_dataset_file.is_file():
        raise RunManagementError(
            "Part B returned a final dataset file that does not exist.",
            error_code="PART_B_FINAL_DATASET_MISSING",
            details={
                "kpi_name": kpi_name,
                "final_dataset_file": str(
                    final_dataset_file
                ),
            },
        )


def _validate_part_b_kpi_continuity(
    *,
    handoff: PartAHandoff,
    results: Sequence[
        PreprocessingRunnerResult
    ],
) -> tuple[str, ...]:
    """
    Require exactly one Part B result for every KPI produced by Part A.

    Part A is the authoritative downstream KPI enumeration. No KPI may silently
    disappear, be added, or be duplicated before the Part B snapshot is marked
    successful.
    """
    expected_names = tuple(
        normalize_kpi_name(
            kpi_result
            .prepared_dataset
            .kpi_name
        )
        for kpi_result in handoff.kpi_results
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
            "Part B KPI results do not exactly match the KPI set produced by "
            "Part A.",
            error_code="PART_B_KPI_SET_MISMATCH",
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


def run_part_b_workflow(
    *,
    request: PreprocessingStageRequest,
    manager: RunManager,
    logger: Logger,
) -> StandaloneStageResponse:
    """
    Execute one production Part B stage.

    ``prepare_stage_execution()`` resolves the authoritative successful Part A
    prerequisite and enforces Part B lineage/retry legality before this workflow
    begins ML execution.
    """

    # ------------------------------------------------------------------
    # 1. Create/validate the Part B execution and resolve its Part A result.
    #    Errors here propagate because no Part B execution is safely available
    #    to mark failed if Run Management rejects the transition itself.
    # ------------------------------------------------------------------
    execution = manager.prepare_stage_execution(
        request.run_id,
        request.attempt_id,
        "part_b",
        request_payload=request,
        metadata={
            "workflow": "part_b",
        },
    )

    # ------------------------------------------------------------------
    # 2. Reconstruct exact Part A runner objects and run normal Part B once
    #    for every authoritative Part A KPI.
    # ------------------------------------------------------------------
    try:
        if len(
            execution.prerequisite_results
        ) != 1:
            raise RunManagementError(
                "Part B expected exactly one authoritative Part A result.",
                error_code="PART_B_PREREQUISITE_RESULT_COUNT_INVALID",
                details={
                    "received_count": len(
                        execution.prerequisite_results
                    ),
                },
            )

        handoff = load_part_a_handoff(
            result_file=(
                execution
                .prerequisite_results[0]
                .path
            ),
            run_directory=(
                execution.run_directory
            ),
        )

        if not handoff.kpi_results:
            # ``load_part_a_handoff()`` already protects this, but retaining the
            # guard here keeps the stage contract explicit.
            raise RunManagementError(
                "Part B received no KPI results from Part A.",
                error_code="PART_B_PART_A_KPI_RESULTS_EMPTY",
            )

        directory_names = (
            _unique_kpi_directory_names(
                handoff
            )
        )

        results: list[
            PreprocessingRunnerResult
        ] = []

        for (
            kpi_result,
            directory_name,
        ) in zip(
            handoff.kpi_results,
            directory_names,
            strict=True,
        ):
            kpi_name = normalize_kpi_name(
                kpi_result
                .prepared_dataset
                .kpi_name
            )

            output_directory = (
                ensure_directory(
                    execution.output_directory
                    / directory_name
                )
            )

            result = run_preprocessing_pipeline(
                kpi_result=kpi_result,
                output_directory=(
                    output_directory
                ),
                dataset_config=(
                    handoff.dataset_config
                ),
                preprocessing_config=(
                    request.preprocessing
                ),
                bucket=handoff.bucket,
                logger=logger,
            )

            returned_kpi_name = (
                _validate_one_result_identity(
                    expected_kpi_name=(
                        kpi_name
                    ),
                    result=result,
                )
            )

            _validate_result_output(
                result=result,
                output_directory=(
                    output_directory
                ),
                kpi_name=(
                    returned_kpi_name
                ),
            )

            results.append(
                result
            )

        kpi_names = (
            _validate_part_b_kpi_continuity(
                handoff=handoff,
                results=results,
            )
        )

        # Persist the exact B->C structured runner contracts. Part C will later
        # reconstruct real PreprocessingRunnerResult objects from this snapshot.
        snapshot = build_part_b_result_snapshot(
            dataset_config=(
                handoff.dataset_config
            ),
            bucket=handoff.bucket,
            preprocessing_results=tuple(
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
            "total_input_rows": sum(
                int(
                    result.total_input_rows
                )
                for result in results
            ),
            "final_rows": sum(
                int(
                    result.final_rows
                )
                for result in results
            ),
            "prophet_enabled_kpis": sum(
                1
                for result in results
                if bool(
                    result.prophet_enabled
                )
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
    # 4. Public response exposes no server filesystem paths.
    # ------------------------------------------------------------------
    return StandaloneStageResponse(
        status="completed",
        run_id=request.run_id,
        attempt_id=request.attempt_id,
        requested_stage="part_b",
        executions=(
            StageExecutionSummary(
                stage="part_b",
                execution_id=(
                    execution.execution_id
                ),
                status="completed",
                summary=summary,
            ),
        ),
        message=(
            "Part B preprocessing completed successfully."
        ),
        next_stage="part_c",
        inference_ready=False,
        summary=summary,
    )


__all__ = [
    "run_part_b_workflow",
]