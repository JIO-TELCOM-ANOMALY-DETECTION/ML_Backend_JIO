"""
Define the temporary FastAPI request schemas used to test:

- Part A EDA;
- Part B preprocessing;
- Part C feature engineering;
- Part D model training with fixed or Optuna tuning; and
- Part E pure inference against a previously completed temporary training run.

Intended project location::

    app/schemas/test_pipeline_part_E.py

The temporary inference request adds only ``run_directory`` to the production
``InferenceRequest`` contract. In production, a database/service will resolve
``run_id -> run_directory``. For this test rig, the later inference-suite
scripts resolve that directory from the existing Part-D suite root and submit
it explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from app.schemas.common.dataset import DatasetInputConfig
from app.schemas.feature_engineering import FeatureEngineeringConfig
from app.schemas.inference.request import InferenceRequest
from app.schemas.preprocessing import PreprocessingConfig
from app.schemas.training.isolation_forest.isolation_forest import (
    IsolationForestTrainingConfig,
)
from app.schemas.training.xgboost.xgboost import (
    XGBoostTrainingConfig,
)


TrainingConfig: TypeAlias = Annotated[
    XGBoostTrainingConfig | IsolationForestTrainingConfig,
    Field(discriminator="model_type"),
]


class InferenceBootstrapConfig(BaseModel):
    """
    Temporary filesystem history policy created after successful training.

    These values exist only so the Part-A-to-Part-D test endpoint can leave
    behind a complete saved-run state that the real Part-E inference runner can
    consume later.

    Defaults intentionally allow only a contiguous next inference period.
    Gap-focused inference tests can override the limits explicitly.
    """

    model_config = ConfigDict(
        extra="forbid",
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


class PipelineRunRequest(BaseModel):
    """
    Store all validated inputs required for one temporary training-pipeline run.

    Existing Part-D requests remain valid. ``run_id`` and
    ``inference_bootstrap`` are optional additions used only to prepare the
    completed output directory for later Part-E inference testing.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    dataset: DatasetInputConfig = Field(
        description=(
            "Dataset path, column mapping, bucket information, KPI identity, "
            "and expected timeline settings."
        ),
    )

    preprocessing: PreprocessingConfig = Field(
        description=(
            "Complete Part B preprocessing configuration."
        ),
    )

    feature_engineering: FeatureEngineeringConfig | None = Field(
        default=None,
        description=(
            "Complete Part C feature-engineering configuration. When omitted, "
            "the workflow uses the schema-owned default profile matching the "
            "selected Part D model."
        ),
    )

    training: TrainingConfig = Field(
        description=(
            "Complete Part D model-training configuration. model_type "
            "discriminates between XGBoost and Isolation Forest settings."
        ),
    )

    output_directory: Path = Field(
        description=(
            "Root directory where A/B/C/D outputs and the temporary saved "
            "inference state are written."
        ),
    )

    run_id: str | None = Field(
        default=None,
        min_length=1,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
        description=(
            "Optional explicit saved-run ID. When omitted, the workflow derives "
            "a deterministic test run ID from output_directory."
        ),
    )

    inference_bootstrap: InferenceBootstrapConfig = Field(
        default_factory=InferenceBootstrapConfig,
        description=(
            "Temporary history/gap policy used while constructing the saved "
            "Part-E inference state after training succeeds."
        ),
    )

    @field_validator(
        "output_directory",
        mode="before",
    )
    @classmethod
    def normalise_output_directory(
        cls,
        value: str | Path,
    ) -> Path:
        return Path(
            value
        ).expanduser()


class TestInferenceRunRequest(InferenceRequest):
    """
    Temporary API request for Part-E inference.

    ``run_id``, ``input_file`` and ``inference_days`` are inherited unchanged
    from the real inference schema.

    ``run_directory`` temporarily replaces the future database lookup. The
    inference test-suite generator/runner resolves it from the already-created
    Part-D suite and sends it to the backend.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    run_directory: Path = Field(
        description=(
            "Existing completed temporary training-run directory containing "
            "inference_config.json and all saved inference state."
        ),
    )

    @field_validator(
        "run_directory",
        mode="before",
    )
    @classmethod
    def normalise_run_directory(
        cls,
        value: str | Path,
    ) -> Path:
        return Path(
            value
        ).expanduser()

    def to_inference_request(
        self,
    ) -> InferenceRequest:
        """Return the exact production Part-E request contract."""

        return InferenceRequest(
            run_id=self.run_id,
            input_file=self.input_file,
            inference_days=self.inference_days,
        )


__all__ = [
    "InferenceBootstrapConfig",
    "PipelineRunRequest",
    "TestInferenceRunRequest",
    "TrainingConfig",
]
