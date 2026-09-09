"""
Production Part A workflow.

Part A is the entry point for one training attempt. It connects the final
workflow request schema, Run Management lifecycle, the already-tested Part A
EDA runner, and the persisted A->B handoff contract.

Selection semantics are inherited directly from ``EDAStageRequest``:

- no run_id:
    create a new run and a new attempt;
- run_id only:
    create a new attempt under the existing run;
- run_id + attempt_id:
    target that existing attempt, normally to retry a failed Part A execution.

Run Management remains responsible for deciding whether the requested retry is
legal. The workflow never bypasses lineage rules.
"""

from __future__ import annotations

from logging import Logger
from typing import Any

from app.core.exceptions import RunManagementError
from app.ml.eda.runner import (
    EDARunnerResult,
    run_eda_pipeline,
)
from app.run_management import RunManager
from app.schemas.workflows import (
    EDAStageRequest,
    StageExecutionSummary,
    StandaloneStageResponse,
    WorkflowErrorDetail,
)
from app.services.run_manifest_service import (
    convert_exception_to_record,
)

from .common import (
    build_runtime_dataset_config,
    normalize_kpi_name,
    to_json_safe,
)
from .handoffs import (
    build_part_a_result_snapshot,
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


def _run_metadata_payload(
    request: EDAStageRequest,
) -> dict[str, Any]:
    """
    Return the non-authoritative descriptive metadata stored on a new run.

    Existing runs are resolved rather than recreated, so their run-level
    metadata is not silently overwritten by a Part A retry/new-attempt call.
    """
    return request.run_metadata.model_dump(
        mode="json",
        exclude_none=True,
    )


def _validate_part_a_result(
    result: EDARunnerResult,
) -> tuple[str, ...]:
    """
    Validate the workflow-level contract Part A must satisfy before success.

    Part A is the authoritative source of downstream KPI enumeration. A
    successful persisted Part A result therefore must contain at least one KPI
    and may not contain duplicate/invalid KPI identities.
    """
    if not result.kpi_results:
        raise RunManagementError(
            "Part A completed without producing any KPI results.",
            error_code="PART_A_KPI_RESULTS_EMPTY",
        )

    seen: set[str] = set()
    ordered_names: list[str] = []

    for kpi_result in result.kpi_results:
        kpi_name = normalize_kpi_name(
            kpi_result.prepared_dataset.kpi_name
        )

        if kpi_name in seen:
            raise RunManagementError(
                "Part A produced more than one result for the same KPI.",
                error_code="DUPLICATE_PART_A_KPI_RESULT",
                details={
                    "kpi_name": kpi_name,
                },
            )

        seen.add(
            kpi_name
        )
        ordered_names.append(
            kpi_name
        )

    return tuple(
        ordered_names
    )


def _failed_response(
    *,
    run_id: str,
    attempt_id: str,
    execution_id: str,
    error: Exception,
) -> StandaloneStageResponse:
    """Build the standard failed Part A response after failure is persisted."""
    detail = _error_detail(
        error
    )

    return StandaloneStageResponse(
        status="failed",
        run_id=run_id,
        attempt_id=attempt_id,
        requested_stage="part_a",
        executions=(
            StageExecutionSummary(
                stage="part_a",
                execution_id=execution_id,
                status="failed",
                error=detail,
            ),
        ),
        message="Part A EDA failed.",
        next_stage="part_a",
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

    If failure persistence itself cannot be completed, the workflow must not
    mask the original exception with a second persistence exception. The caller
    then propagates the original error to the API error boundary.
    """
    try:
        manager.fail_stage(
            execution,
            error,
        )
    except Exception:
        return False

    return True


def run_part_a_workflow(
    *,
    request: EDAStageRequest,
    manager: RunManager,
    logger: Logger,
) -> StandaloneStageResponse:
    """
    Execute one production Part A stage.

    Errors that occur before a stage execution is created are intentionally
    propagated. Once the execution exists, runner/contract failures are
    persisted through Run Management and returned as a structured failed stage
    response.
    """

    # ------------------------------------------------------------------
    # 1. Resolve/create the run + attempt selected by EDAStageRequest.
    # ------------------------------------------------------------------
    if request.run_id is None:
        run_context = manager.create_run(
            run_metadata=_run_metadata_payload(
                request
            ),
        )

        attempt_context = manager.create_attempt(
            run_context.run_id,
            request_payload=request,
            input_file=(
                request.dataset.input_file
            ),
        )

    elif request.attempt_id is None:
        run_context = manager.resolve_run(
            request.run_id
        )

        attempt_context = manager.create_attempt(
            run_context.run_id,
            request_payload=request,
            input_file=(
                request.dataset.input_file
            ),
        )

    else:
        run_context = manager.resolve_run(
            request.run_id
        )

        attempt_context = manager.resolve_attempt(
            run_context.run_id,
            request.attempt_id,
        )

    # ------------------------------------------------------------------
    # 2. Replace the external source path with the immutable attempt copy.
    # ------------------------------------------------------------------
    runtime_dataset_config = (
        build_runtime_dataset_config(
            request.dataset,
            input_file=(
                attempt_context.input_file
            ),
        )
    )

    # ------------------------------------------------------------------
    # 3. Ask Run Management to create/validate the Part A execution.
    #    Lineage legality is decided there, not in this workflow.
    # ------------------------------------------------------------------
    execution = manager.prepare_stage_execution(
        run_context.run_id,
        attempt_context.attempt_id,
        "part_a",
        request_payload=request,
        metadata={
            "workflow": "part_a",
        },
    )

    # ------------------------------------------------------------------
    # 4. Run the existing Part A module unchanged.
    # ------------------------------------------------------------------
    try:
        result = run_eda_pipeline(
            config=runtime_dataset_config,
            logger=logger,
            output_directory=(
                execution.output_directory
            ),
        )

        kpi_names = _validate_part_a_result(
            result
        )

        # Persist the exact A->B contract. No later stage needs to rediscover
        # Part A artifacts by scanning the filesystem.
        snapshot = build_part_a_result_snapshot(
            dataset_config=(
                runtime_dataset_config
            ),
            eda_result=result,
        )

        summary = {
            "kpi_count": len(
                kpi_names
            ),
            "kpi_names": list(
                kpi_names
            ),
            "total_input_rows": int(
                result
                .preparation
                .total_input_rows
            ),
            "was_split": bool(
                result
                .preparation
                .was_split
            ),
            "bucket": to_json_safe(
                result
                .preparation
                .bucket
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
            run_id=run_context.run_id,
            attempt_id=(
                attempt_context.attempt_id
            ),
            execution_id=(
                execution.execution_id
            ),
            error=error,
        )

    # ------------------------------------------------------------------
    # 5. Finalize the execution through Run Management.
    #
    # complete_stage() owns result.json + manifest/registry persistence.
    # If finalization itself fails, try to mark the still-active execution
    # failed. If the execution was already finalized before the persistence
    # error surfaced, fail_stage() may reject that transition; in that case
    # propagate the original completion error rather than masking it.
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
            run_id=run_context.run_id,
            attempt_id=(
                attempt_context.attempt_id
            ),
            execution_id=(
                execution.execution_id
            ),
            error=error,
        )

    # ------------------------------------------------------------------
    # 6. Public response contains IDs/status/summary only, never paths.
    # ------------------------------------------------------------------
    return StandaloneStageResponse(
        status="completed",
        run_id=run_context.run_id,
        attempt_id=(
            attempt_context.attempt_id
        ),
        requested_stage="part_a",
        executions=(
            StageExecutionSummary(
                stage="part_a",
                execution_id=(
                    execution.execution_id
                ),
                status="completed",
                summary=summary,
            ),
        ),
        message=(
            "Part A EDA completed successfully."
        ),
        next_stage="part_b",
        inference_ready=False,
        summary=summary,
    )


__all__ = [
    "run_part_a_workflow",
]
