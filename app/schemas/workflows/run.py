"""
Read-only API schemas for inspecting persisted run, attempt, stage, execution,
and inference-invocation state.

These models describe data returned by future run-status endpoints. They do not
mutate manifests and do not duplicate the internal run-management dataclasses.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from .common import (
    AttemptId,
    AttemptStatus,
    ExecutionId,
    ExecutionStatus,
    InferenceInvocationStatus,
    InvocationId,
    RunId,
    RunMetadata,
    RunStatus,
    StageStatus,
    TrainingStageName,
    WorkflowErrorDetail,
    WorkflowErrorRecord,
    WorkflowSchemaModel,
)


class ExecutionView(WorkflowSchemaModel):
    """Public status view of one physical stage execution."""

    execution_id: ExecutionId
    status: ExecutionStatus

    started_at_utc: datetime | None = None
    completed_at_utc: datetime | None = None

    metrics: dict[str, Any] = Field(
        default_factory=dict,
    )
    error: WorkflowErrorDetail | None = None


class StageView(WorkflowSchemaModel):
    """Public status view of one canonical A-D/bootstrap stage."""

    stage: TrainingStageName
    status: StageStatus

    active_execution_id: ExecutionId | None = None
    successful_execution_id: ExecutionId | None = None

    executions: tuple[ExecutionView, ...] = ()


class AttemptView(WorkflowSchemaModel):
    """Public status view of one training attempt."""

    attempt_id: AttemptId
    status: AttemptStatus

    created_at_utc: datetime
    updated_at_utc: datetime
    started_at_utc: datetime | None = None
    completed_at_utc: datetime | None = None

    stages: tuple[StageView, ...] = ()

    summary: dict[str, Any] = Field(
        default_factory=dict,
    )
    errors: tuple[WorkflowErrorRecord, ...] = ()


class InferenceInvocationView(WorkflowSchemaModel):
    """Public status view of one persisted Part E invocation."""

    invocation_id: InvocationId
    training_attempt_id: AttemptId
    status: InferenceInvocationStatus

    started_at_utc: datetime | None = None
    completed_at_utc: datetime | None = None

    metrics: dict[str, Any] = Field(
        default_factory=dict,
    )
    error: WorkflowErrorDetail | None = None


class RunSummaryResponse(WorkflowSchemaModel):
    """
    Compact run-level response for GET /runs/{run_id}.
    """

    run_id: RunId
    run_metadata: RunMetadata = Field(
        default_factory=RunMetadata,
    )
    status: RunStatus

    created_at_utc: datetime
    updated_at_utc: datetime
    started_at_utc: datetime | None = None
    completed_at_utc: datetime | None = None

    latest_attempt_id: AttemptId | None = None
    active_inference_attempt_id: AttemptId | None = None
    inference_ready: bool = False

    attempt_count: int = Field(
        default=0,
        ge=0,
    )
    attempts: tuple[AttemptView, ...] = ()

    inference_invocation_count: int = Field(
        default=0,
        ge=0,
    )
    inference_invocations: tuple[
        InferenceInvocationView,
        ...,
    ] = ()

    summary: dict[str, Any] = Field(
        default_factory=dict,
    )
    errors: tuple[WorkflowErrorRecord, ...] = ()


class AttemptDetailResponse(WorkflowSchemaModel):
    """
    Detailed response for one attempt-status endpoint.
    """

    run_id: RunId
    attempt: AttemptView
    active_inference_attempt_id: AttemptId | None = None
    inference_ready: bool = False


__all__ = [
    "ExecutionView",
    "StageView",
    "AttemptView",
    "InferenceInvocationView",
    "RunSummaryResponse",
    "AttemptDetailResponse",
]