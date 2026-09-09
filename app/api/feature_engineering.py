"""
Production FastAPI route for standalone Part C feature-engineering execution.

This module is intentionally a thin HTTP boundary.

Responsibilities
----------------
- receive and validate ``FeatureEngineeringStageRequest`` through
  Pydantic/FastAPI;
- establish whether the caller-supplied run and attempt genuinely exist so
  public missing resources can use the frozen API_* 404 contract;
- obtain the existing run-scoped workflow logger;
- delegate the complete Part C lifecycle to ``run_part_c_workflow``.

This module deliberately does NOT:
- read or validate an external CSV;
- inspect Part B result files directly;
- check Part B completion itself;
- decide Part C retry legality;
- resolve explicit config vs ``model_type_hint`` itself;
- create/complete/fail stage executions;
- manipulate Manifest V2;
- execute feature-engineering ML logic directly;
- reshape workflow success/failure responses.

The workflow request schema owns the explicit-config/default-profile selection
contract, while Run Management and the Part C workflow remain authoritative for
prerequisite resolution, completed-stage immutability, active-execution
conflicts, artifact resolution, stage persistence, and feature-engineering
orchestration.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.core import (
    AttemptNotFoundError,
    RunNotFoundError,
)
from app.schemas.workflows import (
    FeatureEngineeringStageRequest,
    StandaloneStageResponse,
)
from app.workflows import (
    run_part_c_workflow,
)

from .dependencies import (
    RunManagerDependency,
    get_workflow_logger,
)


router = APIRouter(
    prefix="/api/v1/feature-engineering",
    tags=["feature-engineering"],
)


def _validate_public_target(
    *,
    request: FeatureEngineeringStageRequest,
    manager: Any,
) -> None:
    """
    Establish genuine caller-owned run/attempt absence before Part C starts.

    Only public lookup absence is translated here:

    - unknown caller-supplied run
        -> API_RUN_NOT_FOUND;
    - unknown caller-supplied attempt inside a known run
        -> API_ATTEMPT_NOT_FOUND.

    Once these checks pass, later Run Management/workflow failures propagate
    unchanged. A registered run whose manifest/directory/artifacts are broken
    is an internal backend problem, not a caller-owned 404.
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
    summary="Run standalone Part C feature engineering",
)
def run_feature_engineering_stage(
    request: FeatureEngineeringStageRequest,
    manager: RunManagerDependency,
) -> StandaloneStageResponse:
    """
    Execute one standalone Part C feature-engineering workflow.

    The request always targets an existing run + attempt.

    Part C configuration semantics are already frozen in
    ``FeatureEngineeringStageRequest``:

    - explicit ``feature_engineering`` supplied:
        the explicit configuration is authoritative;
    - entire ``feature_engineering`` config omitted:
        ``model_type_hint`` is required so the workflow can instantiate the
        correct schema-owned XGBoost or Isolation Forest default profile.

    This route deliberately does not duplicate that resolution logic.

    Likewise, it does not inspect Part B completion or Part C lifecycle state.
    The Part C workflow calls Run Management's ``prepare_stage_execution()``,
    which is authoritative for:

    - successful Part B prerequisite resolution;
    - Part C retry legality;
    - completed-stage immutability;
    - already-running execution conflicts.

    Once a Part C execution starts, the workflow owns feature-engineering
    execution, C->D handoff persistence, stage success/failure persistence, and
    the returned ``StandaloneStageResponse``.
    """
    _validate_public_target(
        request=request,
        manager=manager,
    )

    logger = get_workflow_logger(
        request.run_id,
        component_name="pipeline",
    )

    return run_part_c_workflow(
        request=request,
        manager=manager,
        logger=logger,
    )


__all__ = [
    "router",
]