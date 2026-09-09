"""
Production training lifecycle composition.

This module provides two orchestration entry points:

1. ``run_training_stage_workflow()``
   Final standalone training-endpoint behavior:

       Part D
         ↓
       inference_bootstrap

   Retry policy:

       D incomplete/failed
       → run/retry D
       → bootstrap

       D completed + bootstrap incomplete/failed
       → reuse completed D
       → retry bootstrap only

       D completed + bootstrap completed
       → reject duplicate completed lifecycle

2. ``run_training_pipeline()``
   Master fresh-attempt workflow:

       A → B → C → D → inference_bootstrap

The master workflow delegates to the same frozen individual workflow functions
used by standalone endpoints. It does not duplicate any ML execution logic.
"""

from __future__ import annotations

from logging import Logger
from typing import Any

from app.core.exceptions import StageConflictError
from app.run_management import (
    RunManager,
    get_successful_execution_id,
    is_stage_completed,
)
from app.schemas.workflows import (
    EDAStageRequest,
    FeatureEngineeringStageRequest,
    PreprocessingStageRequest,
    StandaloneStageResponse,
    TrainingPipelineRequest,
    TrainingPipelineResponse,
    TrainingStageRequest,
)

from .inference_bootstrap import (
    run_inference_bootstrap_workflow,
)
from .part_a import run_part_a_workflow
from .part_b import run_part_b_workflow
from .part_c import run_part_c_workflow
from .part_d import run_part_d_workflow


def _merge_summary(
    *summaries: dict[str, Any],
) -> dict[str, Any]:
    """Merge workflow summaries in execution order."""
    merged: dict[
        str,
        Any,
    ] = {}

    for summary in summaries:
        merged.update(
            summary
        )

    return merged


def _successful_execution_details(
    *,
    manifest: dict[str, Any] | Any,
    attempt_id: str,
) -> dict[str, Any]:
    """Return completed lifecycle execution IDs for conflict diagnostics."""
    return {
        "part_d_execution_id": (
            get_successful_execution_id(
                manifest,
                attempt_id,
                "part_d",
            )
        ),
        "bootstrap_execution_id": (
            get_successful_execution_id(
                manifest,
                attempt_id,
                "inference_bootstrap",
            )
        ),
    }


def run_training_stage_workflow(
    *,
    request: TrainingStageRequest,
    manager: RunManager,
    logger: Logger,
) -> StandaloneStageResponse:
    """
    Execute final standalone training lifecycle: Part D then bootstrap.

    This function deliberately inspects current persisted lineage before
    choosing which stage to launch. It never reruns a successfully completed
    immutable Part D merely because bootstrap failed.
    """
    manifest = manager.get_manifest(
        request.run_id
    )

    d_completed = is_stage_completed(
        manifest,
        request.attempt_id,
        "part_d",
    )
    bootstrap_completed = is_stage_completed(
        manifest,
        request.attempt_id,
        "inference_bootstrap",
    )

    # Bootstrap has Part D as its canonical prerequisite. Seeing bootstrap
    # completed while D is incomplete means the persisted lineage is corrupt.
    if (
        bootstrap_completed
        and not d_completed
    ):
        raise StageConflictError(
            "The training attempt has a completed inference bootstrap while "
            "Part D is not completed.",
            error_code="INCONSISTENT_TRAINING_LIFECYCLE",
            details={
                "run_id": request.run_id,
                "attempt_id": (
                    request.attempt_id
                ),
                **_successful_execution_details(
                    manifest=manifest,
                    attempt_id=(
                        request.attempt_id
                    ),
                ),
            },
        )

    if (
        d_completed
        and bootstrap_completed
    ):
        raise StageConflictError(
            "Part D and inference bootstrap are already completed for this "
            "attempt.",
            error_code=(
                "TRAINING_ATTEMPT_ALREADY_INFERENCE_READY"
            ),
            details={
                "run_id": request.run_id,
                "attempt_id": (
                    request.attempt_id
                ),
                **_successful_execution_details(
                    manifest=manifest,
                    attempt_id=(
                        request.attempt_id
                    ),
                ),
            },
        )

    launched_executions = []

    d_summary: dict[
        str,
        Any,
    ] = {}

    # --------------------------------------------------------------
    # D incomplete/failed -> run or retry D.
    # --------------------------------------------------------------
    if not d_completed:
        d_response = run_part_d_workflow(
            request=request,
            manager=manager,
            logger=logger,
        )

        launched_executions.extend(
            d_response.executions
        )

        d_summary = dict(
            d_response.summary
        )

        if (
            d_response.status
            != "completed"
        ):
            # Preserve the exact standalone D failure response. Bootstrap was
            # never attempted.
            return d_response

    # --------------------------------------------------------------
    # Bootstrap always runs after a newly successful D, or by itself when
    # D was already completed from an earlier request.
    #
    # The bootstrap workflow performs its own immutable-D configuration
    # preflight before creating a bootstrap execution.
    # --------------------------------------------------------------
    bootstrap_response = (
        run_inference_bootstrap_workflow(
            request=request,
            manager=manager,
            logger=logger,
        )
    )

    launched_executions.extend(
        bootstrap_response.executions
    )

    combined_summary = _merge_summary(
        d_summary,
        dict(
            bootstrap_response.summary
        ),
    )

    if (
        bootstrap_response.status
        == "completed"
    ):
        return StandaloneStageResponse(
            status="completed",
            run_id=request.run_id,
            attempt_id=request.attempt_id,
            requested_stage="part_d",
            executions=tuple(
                launched_executions
            ),
            message=(
                "Part D training and inference bootstrap are complete. "
                "The attempt is inference-ready."
                if not d_completed
                else (
                    "The existing completed Part D result was reused and "
                    "inference bootstrap retry completed successfully."
                )
            ),
            next_stage="inference",
            inference_ready=True,
            summary=combined_summary,
        )

    # Part D is known to be complete here. A failed bootstrap therefore means
    # the lifecycle is partially completed and retryable at bootstrap only.
    return StandaloneStageResponse(
        status="partially_completed",
        run_id=request.run_id,
        attempt_id=request.attempt_id,
        requested_stage="part_d",
        executions=tuple(
            launched_executions
        ),
        message=(
            "Part D is complete, but inference bootstrap failed. "
            "Bootstrap may be retried in the same attempt."
        ),
        next_stage="inference_bootstrap",
        inference_ready=False,
        summary=combined_summary,
        error=bootstrap_response.error,
    )


def _pipeline_failure(
    *,
    run_id: str,
    attempt_id: str,
    stages: tuple,
    failed_response: StandaloneStageResponse,
    previous_completed: bool,
) -> TrainingPipelineResponse:
    """Convert one failed standalone stage response into master-pipeline form."""
    failed_stage = (
        failed_response
        .executions[-1]
        .stage
        if failed_response.executions
        else failed_response.requested_stage
    )

    return TrainingPipelineResponse(
        status=(
            "partially_completed"
            if previous_completed
            else "failed"
        ),
        run_id=run_id,
        attempt_id=attempt_id,
        stages=stages,
        message=failed_response.message,
        failed_stage=failed_stage,
        next_stage=(
            failed_response.next_stage
        ),
        inference_ready=False,
        summary=dict(
            failed_response.summary
        ),
        error=failed_response.error,
    )


def run_training_pipeline(
    *,
    request: TrainingPipelineRequest,
    manager: RunManager,
    logger: Logger,
) -> TrainingPipelineResponse:
    """
    Run one fresh training attempt through A→B→C→D→bootstrap.

    ``TrainingPipelineRequest`` intentionally has no attempt_id. Therefore every
    call creates a fresh attempt:
    - no run_id -> new logical run + fresh attempt;
    - run_id supplied -> fresh attempt below that existing logical run.

    Failed attempts are resumed through the corresponding standalone stage
    endpoint, not by resubmitting the master request.
    """
    stage_summaries = []

    # --------------------------------------------------------------
    # Part A creates/selects the new attempt for the whole master lifecycle.
    # --------------------------------------------------------------
    a_request = EDAStageRequest(
        run_id=request.run_id,
        run_metadata=request.run_metadata,
        dataset=request.dataset,
    )

    a_response = run_part_a_workflow(
        request=a_request,
        manager=manager,
        logger=logger,
    )

    stage_summaries.extend(
        a_response.executions
    )

    run_id = a_response.run_id
    attempt_id = a_response.attempt_id

    if (
        a_response.status
        != "completed"
    ):
        return _pipeline_failure(
            run_id=run_id,
            attempt_id=attempt_id,
            stages=tuple(
                stage_summaries
            ),
            failed_response=a_response,
            previous_completed=False,
        )

    # --------------------------------------------------------------
    # Part B uses only run/attempt identity + B request configuration.
    # --------------------------------------------------------------
    b_response = run_part_b_workflow(
        request=PreprocessingStageRequest(
            run_id=run_id,
            attempt_id=attempt_id,
            preprocessing=(
                request.preprocessing
            ),
        ),
        manager=manager,
        logger=logger,
    )

    stage_summaries.extend(
        b_response.executions
    )

    if (
        b_response.status
        != "completed"
    ):
        return _pipeline_failure(
            run_id=run_id,
            attempt_id=attempt_id,
            stages=tuple(
                stage_summaries
            ),
            failed_response=b_response,
            previous_completed=True,
        )

    # --------------------------------------------------------------
    # Part C preserves historical model-matched default-profile behavior.
    # --------------------------------------------------------------
    c_response = run_part_c_workflow(
        request=FeatureEngineeringStageRequest(
            run_id=run_id,
            attempt_id=attempt_id,
            feature_engineering=(
                request
                .feature_engineering
            ),
            model_type_hint=(
                request.training.model_type
                if (
                    request
                    .feature_engineering
                    is None
                )
                else None
            ),
        ),
        manager=manager,
        logger=logger,
    )

    stage_summaries.extend(
        c_response.executions
    )

    if (
        c_response.status
        != "completed"
    ):
        return _pipeline_failure(
            run_id=run_id,
            attempt_id=attempt_id,
            stages=tuple(
                stage_summaries
            ),
            failed_response=c_response,
            previous_completed=True,
        )

    # --------------------------------------------------------------
    # Final training lifecycle: canonical D followed by bootstrap.
    # --------------------------------------------------------------
    training_response = (
        run_training_stage_workflow(
            request=TrainingStageRequest(
                run_id=run_id,
                attempt_id=attempt_id,
                training=request.training,
                inference_bootstrap=(
                    request
                    .inference_bootstrap
                ),
            ),
            manager=manager,
            logger=logger,
        )
    )

    stage_summaries.extend(
        training_response.executions
    )

    if (
        training_response.status
        != "completed"
    ):
        failed_stage = (
            training_response
            .executions[-1]
            .stage
            if training_response.executions
            else "part_d"
        )

        return TrainingPipelineResponse(
            status="partially_completed",
            run_id=run_id,
            attempt_id=attempt_id,
            stages=tuple(
                stage_summaries
            ),
            message=(
                training_response.message
            ),
            failed_stage=failed_stage,
            next_stage=(
                training_response.next_stage
            ),
            inference_ready=False,
            summary=dict(
                training_response.summary
            ),
            error=(
                training_response.error
            ),
        )

    summary = {
        "completed_stage_count": len(
            stage_summaries
        ),
        "model_type": (
            request.training.model_type
        ),
        **dict(
            training_response.summary
        ),
    }

    return TrainingPipelineResponse(
        status="completed",
        run_id=run_id,
        attempt_id=attempt_id,
        stages=tuple(
            stage_summaries
        ),
        message=(
            "Training pipeline completed successfully. "
            "The attempt is inference-ready."
        ),
        failed_stage=None,
        next_stage="inference",
        inference_ready=True,
        summary=summary,
    )


__all__ = [
    "run_training_stage_workflow",
    "run_training_pipeline",
]
