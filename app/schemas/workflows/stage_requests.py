"""
Production request schemas for independently invoking Parts A, B, C, and D.

The schemas deliberately reuse the existing A-D configuration contracts instead
of redefining ML settings. Run-management identifiers are added only at the
workflow boundary.

V1 lifecycle rule:
- Part A may create a new run/new attempt or target a specific failed attempt.
- Parts B/C/D always target an existing run + attempt.
- Whether a requested retry is actually legal is decided by Run Management,
  not by these request schemas.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from app.schemas.common.dataset import DatasetInputConfig
from app.schemas.feature_engineering import FeatureEngineeringConfig
from app.schemas.preprocessing import PreprocessingConfig

from .common import (
    AttemptId,
    InferenceBootstrapConfig,
    RunId,
    RunMetadata,
    TrainingConfig,
    TrainingModelType,
    WorkflowSchemaModel,
)


class EDAStageRequest(WorkflowSchemaModel):
    """
    Request for standalone Part A execution.

    Selection semantics:
    - run_id omitted:
        Run Management creates a new run and a new attempt.
    - run_id supplied, attempt_id omitted:
        Run Management creates a new attempt under that run.
    - run_id + attempt_id supplied:
        the request targets that existing attempt, normally to retry failed A.
    """

    run_id: RunId | None = None
    attempt_id: AttemptId | None = None

    run_metadata: RunMetadata = Field(
        default_factory=RunMetadata,
    )

    dataset: DatasetInputConfig

    @model_validator(mode="after")
    def validate_run_selection(
        self,
    ) -> "EDAStageRequest":
        if (
            self.attempt_id is not None
            and self.run_id is None
        ):
            raise ValueError(
                "attempt_id cannot be supplied without run_id."
            )

        return self


class PreprocessingStageRequest(WorkflowSchemaModel):
    """Request for standalone Part B execution."""

    run_id: RunId
    attempt_id: AttemptId
    preprocessing: PreprocessingConfig


class FeatureEngineeringStageRequest(WorkflowSchemaModel):
    """
    Request for standalone Part C execution.

    If a complete feature-engineering config is omitted, model_type_hint is
    required so the workflow can instantiate the correct schema-owned Part C
    default profile.
    """

    run_id: RunId
    attempt_id: AttemptId

    feature_engineering: FeatureEngineeringConfig | None = None
    model_type_hint: TrainingModelType | None = None

    @model_validator(mode="after")
    def require_default_profile_hint(
        self,
    ) -> "FeatureEngineeringStageRequest":
        if (
            self.feature_engineering is None
            and self.model_type_hint is None
        ):
            raise ValueError(
                "model_type_hint is required when feature_engineering "
                "is omitted."
            )

        return self


class TrainingStageRequest(WorkflowSchemaModel):
    """
    Request for standalone Part D execution plus saved-state finalization.

    After Part D succeeds, the workflow applies inference_bootstrap to build the
    persistent Part-E contract/state. Only then may the attempt become the
    run's active inference attempt.
    """

    run_id: RunId
    attempt_id: AttemptId

    training: TrainingConfig

    inference_bootstrap: InferenceBootstrapConfig = Field(
        default_factory=InferenceBootstrapConfig,
    )


__all__ = [
    "EDAStageRequest",
    "PreprocessingStageRequest",
    "FeatureEngineeringStageRequest",
    "TrainingStageRequest",
]
