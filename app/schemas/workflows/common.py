"""
Shared schema primitives for production workflow/run-management API contracts.

This module intentionally contains no orchestration logic. It defines stable
identifiers, lifecycle status literals, run metadata, the Part-D-to-Part-E
bootstrap policy, and the discriminated Part D training-config union reused by
the standalone and master workflow request schemas.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
)

from app.schemas.training.common import TrainingModelType
from app.schemas.training.isolation_forest.isolation_forest import (
    IsolationForestTrainingConfig,
)
from app.schemas.training.xgboost.xgboost import (
    XGBoostTrainingConfig,
)


# ---------------------------------------------------------------------------
# SHARED IDENTIFIERS
# ---------------------------------------------------------------------------

SafeRecordId: TypeAlias = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
    ),
]

# Production run IDs are capped at 96 characters because Part E appends a
# 32-character timestamp/UUID suffix when building invocation_id, while
# Manifest V2 record IDs are capped at 128 characters.
RunId: TypeAlias = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=96,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$",
    ),
]

AttemptId: TypeAlias = SafeRecordId
ExecutionId: TypeAlias = SafeRecordId
InvocationId: TypeAlias = SafeRecordId


# ---------------------------------------------------------------------------
# SHARED LIFECYCLE TYPES
# ---------------------------------------------------------------------------

TrainingStageName: TypeAlias = Literal[
    "part_a",
    "part_b",
    "part_c",
    "part_d",
    "inference_bootstrap",
]

WorkflowStageName: TypeAlias = Literal[
    "part_a",
    "part_b",
    "part_c",
    "part_d",
    "inference_bootstrap",
    "inference",
]

RunStatus: TypeAlias = Literal[
    "created",
    "running",
    "partially_completed",
    "completed",
    "failed",
    "cancelled",
]

AttemptStatus: TypeAlias = Literal[
    "created",
    "running",
    "partially_completed",
    "completed",
    "failed",
    "cancelled",
]

StageStatus: TypeAlias = Literal[
    "pending",
    "running",
    "completed",
    "failed",
]

ExecutionStatus: TypeAlias = Literal[
    "running",
    "completed",
    "failed",
    "cancelled",
]

InferenceInvocationStatus: TypeAlias = ExecutionStatus


# ---------------------------------------------------------------------------
# EXISTING PART D CONTRACT
# ---------------------------------------------------------------------------

TrainingConfig: TypeAlias = Annotated[
    XGBoostTrainingConfig | IsolationForestTrainingConfig,
    Field(discriminator="model_type"),
]


# ---------------------------------------------------------------------------
# COMMON PYDANTIC MODELS
# ---------------------------------------------------------------------------


class WorkflowSchemaModel(BaseModel):
    """
    Base class for new production workflow schemas.

    Unknown fields are rejected so API requests cannot silently carry misspelled
    or unsupported workflow-management options.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class RunMetadata(WorkflowSchemaModel):
    """
    Optional human-facing metadata associated with a logical run.

    These fields are descriptive only in V1. The schema deliberately does not
    impose uniqueness or authentication rules; run_id remains the technical key.
    """

    employee_code: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
    )
    employee_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
    )
    run_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
    )


class InferenceBootstrapConfig(WorkflowSchemaModel):
    """
    History/gap policy used when a completed Part D attempt is frozen for Part E.

    This is the production successor of the tested temporary bootstrap contract.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        frozen=True,
    )

    max_gap_days: int = Field(
        default=0,
        ge=0,
    )

    max_gap_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    late_data_reconstruction_enabled: bool = True


class WorkflowErrorDetail(WorkflowSchemaModel):
    """
    JSON-safe structured error representation used by workflow/API responses.
    """

    error_type: str = Field(
        min_length=1,
        max_length=256,
    )
    message: str = Field(
        min_length=1,
    )
    error_code: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
    )


class WorkflowErrorRecord(WorkflowErrorDetail):
    """
    Manifest-compatible error view with optional lifecycle context.

    Manifest V2 stores scoped errors with timestamp and run/attempt/execution
    identifiers in addition to the core TelecomBackendError-compatible fields.
    """

    timestamp_utc: datetime | None = None
    scope: Literal[
        "run",
        "attempt",
        "stage_execution",
        "inference_invocation",
    ] | None = None

    attempt_id: AttemptId | None = None
    training_attempt_id: AttemptId | None = None
    stage_name: TrainingStageName | None = None
    execution_id: ExecutionId | None = None
    invocation_id: InvocationId | None = None


__all__ = [
    "SafeRecordId",
    "RunId",
    "AttemptId",
    "ExecutionId",
    "InvocationId",
    "TrainingStageName",
    "WorkflowStageName",
    "TrainingModelType",
    "RunStatus",
    "AttemptStatus",
    "StageStatus",
    "ExecutionStatus",
    "InferenceInvocationStatus",
    "TrainingConfig",
    "WorkflowSchemaModel",
    "RunMetadata",
    "InferenceBootstrapConfig",
    "WorkflowErrorDetail",
    "WorkflowErrorRecord",
]
