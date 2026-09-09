"""
Production request schema for the master A -> B -> C -> D training workflow.

The master request always starts a fresh training attempt. Supplying run_id
places that new attempt under an existing logical run; omitting run_id creates a
new run. The master request intentionally has no attempt_id field.

The actual A-D configuration models remain owned by their existing schema
packages.
"""

from __future__ import annotations

from pydantic import Field

from app.schemas.common.dataset import DatasetInputConfig
from app.schemas.feature_engineering import FeatureEngineeringConfig
from app.schemas.preprocessing import PreprocessingConfig

from .common import (
    InferenceBootstrapConfig,
    RunId,
    RunMetadata,
    TrainingConfig,
    WorkflowSchemaModel,
)


class TrainingPipelineRequest(WorkflowSchemaModel):
    """
    Complete validated request for one fresh A-D training attempt.

    If feature_engineering is omitted, the workflow must construct:

        FeatureEngineeringConfig(
            default_profile=training.model_type
        )

    matching the behavior proven by the final A-D integration pipelines.
    """

    run_id: RunId | None = None

    run_metadata: RunMetadata = Field(
        default_factory=RunMetadata,
    )

    dataset: DatasetInputConfig
    preprocessing: PreprocessingConfig

    feature_engineering: FeatureEngineeringConfig | None = Field(
        default=None,
        description=(
            "Explicit Part C configuration. When omitted, the workflow uses "
            "the schema-owned default profile matching training.model_type."
        ),
    )

    training: TrainingConfig

    inference_bootstrap: InferenceBootstrapConfig = Field(
        default_factory=InferenceBootstrapConfig,
    )


__all__ = [
    "TrainingPipelineRequest",
]
