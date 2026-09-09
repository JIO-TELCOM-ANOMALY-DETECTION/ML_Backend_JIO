"""
Standard production response schemas for standalone stage workflows, the master
training workflow, pure Part E inference, and structured API errors.

Physical server paths are intentionally not part of these public response
contracts. Artifact locations remain internal to the manifest/run manager.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from .common import (
    AttemptId,
    ExecutionId,
    InvocationId,
    RunId,
    TrainingStageName,
    WorkflowErrorDetail,
    WorkflowSchemaModel,
    WorkflowStageName,
)


class StageExecutionSummary(WorkflowSchemaModel):
    """Compact result for one stage execution inside an API response."""

    stage: TrainingStageName
    execution_id: ExecutionId
    status: Literal[
        "completed",
        "failed",
        "cancelled",
    ]

    summary: dict[str, Any] = Field(
        default_factory=dict,
    )
    error: WorkflowErrorDetail | None = None


class StandaloneStageResponse(WorkflowSchemaModel):
    """
    Response returned by the standalone Part A/B/C/D endpoints.

    Part D may include both a part_d execution and an inference_bootstrap
    execution in `executions`.
    """

    status: Literal[
        "completed",
        "failed",
        "partially_completed",
    ]

    run_id: RunId
    attempt_id: AttemptId

    requested_stage: TrainingStageName
    executions: tuple[
        StageExecutionSummary,
        ...,
    ]

    message: str = Field(
        min_length=1,
    )

    next_stage: WorkflowStageName | None = None
    inference_ready: bool = False

    summary: dict[str, Any] = Field(
        default_factory=dict,
    )
    error: WorkflowErrorDetail | None = None


class TrainingPipelineResponse(WorkflowSchemaModel):
    """Response for the master A -> B -> C -> D + bootstrap workflow."""

    status: Literal[
        "completed",
        "failed",
        "partially_completed",
    ]

    run_id: RunId
    attempt_id: AttemptId

    stages: tuple[
        StageExecutionSummary,
        ...,
    ]

    message: str = Field(
        min_length=1,
    )

    failed_stage: TrainingStageName | None = None
    next_stage: WorkflowStageName | None = None
    inference_ready: bool = False

    summary: dict[str, Any] = Field(
        default_factory=dict,
    )
    error: WorkflowErrorDetail | None = None


class InferenceWorkflowResponse(WorkflowSchemaModel):
    """Response for one pure Part E invocation."""

    status: Literal[
        "completed",
        "failed",
        "cancelled",
    ]

    run_id: RunId
    training_attempt_id: AttemptId
    invocation_id: InvocationId

    stage: Literal["inference"] = "inference"

    message: str = Field(
        min_length=1,
    )

    summary: dict[str, Any] = Field(
        default_factory=dict,
    )
    error: WorkflowErrorDetail | None = None


class WorkflowErrorResponse(WorkflowSchemaModel):
    """
    Generic structured body for API-level errors that occur before a normal
    workflow response can be produced.
    """

    status: Literal["failed"] = "failed"

    error: WorkflowErrorDetail

    run_id: RunId | None = None
    attempt_id: AttemptId | None = None
    execution_id: ExecutionId | None = None
    invocation_id: InvocationId | None = None
    stage: WorkflowStageName | None = None


__all__ = [
    "StageExecutionSummary",
    "StandaloneStageResponse",
    "TrainingPipelineResponse",
    "InferenceWorkflowResponse",
    "WorkflowErrorResponse",
]
