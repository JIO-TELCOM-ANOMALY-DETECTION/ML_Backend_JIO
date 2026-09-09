"""
Production FastAPI route for standalone training finalization.

The final standalone training endpoint is intentionally composed as:

    Part D training
        ↓
    inference_bootstrap
        ↓
    inference-ready attempt

The HTTP route itself remains thin.

Responsibilities
----------------
- receive and validate ``TrainingStageRequest`` through Pydantic/FastAPI;
- establish whether the caller-supplied run and attempt genuinely exist so
  public missing resources can use the frozen API_* 404 contract;
- obtain the existing run-scoped workflow logger;
- delegate the complete D -> inference_bootstrap lifecycle to
  ``run_training_stage_workflow``.

This module deliberately does NOT:
- read or validate an external CSV;
- inspect Part C artifacts directly;
- check Part C completion itself;
- inspect Part D/bootstrap status itself;
- decide whether Part D or bootstrap should run;
- decide Part D retry/immutability legality;
- compare completed Part D training configuration itself;
- create/complete/fail stage executions;
- activate inference attempts directly;
- manipulate Manifest V2;
- execute XGBoost/Isolation Forest training directly;
- build saved inference state directly;
- reshape workflow success/failure responses.

The composed training workflow and Run Management remain authoritative for all
training/bootstrap lifecycle decisions.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.core import (
    AttemptNotFoundError,
    RunNotFoundError,
)
from app.schemas.workflows import (
    StandaloneStageResponse,
    TrainingStageRequest,
)
from app.workflows import (
    run_training_stage_workflow,
)

from .dependencies import (
    RunManagerDependency,
    get_workflow_logger,
)


router = APIRouter(
    prefix="/api/v1/training",
    tags=["training"],
)


def _validate_public_target(
    *,
    request: TrainingStageRequest,
    manager: Any,
) -> None:
    """
    Establish genuine caller-owned run/attempt absence before training starts.

    Only public lookup absence is translated here:

    - unknown caller-supplied run
        -> API_RUN_NOT_FOUND;
    - unknown caller-supplied attempt inside a known run
        -> API_ATTEMPT_NOT_FOUND.

    Once these checks pass, later lifecycle/manifest/artifact failures propagate
    unchanged. A registered run whose technical state is broken is an internal
    backend problem rather than a public 404.
    """
    if not manager.registry.exists(
        request.run_id
    ):
        raise RunNotFoundError(
            "The caller-requested run is not registered.",
            error_code="API_RUN_NOT_FOUND",
            details={
                "run_id": request.run_id,
            },
        )

    try:
        manager.resolve_attempt(
            request.run_id,
            request.attempt_id,
        )

    except AttemptNotFoundError as error:
        raise AttemptNotFoundError(
            "The caller-requested training attempt is not registered.",
            error_code="API_ATTEMPT_NOT_FOUND",
            details={
                "run_id": request.run_id,
                "attempt_id": (
                    request.attempt_id
                ),
            },
        ) from error


@router.post(
    "/run",
    response_model=StandaloneStageResponse,
    summary="Run standalone Part D training and inference bootstrap",
)
def run_training_stage(
    request: TrainingStageRequest,
    manager: RunManagerDependency,
) -> StandaloneStageResponse:
    """
    Execute the final standalone training lifecycle for one existing attempt.

    ``TrainingStageRequest`` contains both:
    - the discriminated XGBoost / Isolation Forest Part D training config; and
    - the inference-bootstrap policy used to freeze the saved Part E state.

    The API deliberately performs no stage-state branching.

    ``run_training_stage_workflow`` decides among the canonical lifecycle cases:

    1. Part D incomplete/failed:
        run/retry Part D, then run bootstrap after D succeeds;

    2. Part D completed + bootstrap incomplete/failed:
        reuse immutable completed Part D and retry bootstrap only;

    3. Part D completed + bootstrap completed:
        reject the duplicate completed training lifecycle;

    4. bootstrap completed while Part D is incomplete:
        treat persisted lifecycle state as internally inconsistent.

    Part C prerequisite resolution, completed-stage immutability, running-stage
    conflicts, completed-D configuration preflight, saved-state publication,
    and inference activation all remain below this API layer.
    """
    _validate_public_target(
        request=request,
        manager=manager,
    )

    logger = get_workflow_logger(
        request.run_id,
        component_name="pipeline",
    )

    return run_training_stage_workflow(
        request=request,
        manager=manager,
        logger=logger,
    )


__all__ = [
    "router",
]
