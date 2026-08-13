"""
Define the temporary API request schema for running Part A EDA,
Part B preprocessing, Part C feature engineering, and Part D training
through FastAPI.

This module combines the existing dataset, preprocessing, feature-engineering,
and model-specific training schemas. It does not execute the pipeline or
contain processing logic.
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


class PipelineRunRequest(BaseModel):
    """
    Store all validated inputs required for one temporary pipeline run.

    The request contains:

    - the source dataset path and dataset mappings;
    - all Part B preprocessing settings;
    - optional Part C feature-engineering settings;
    - one explicit Part D model-training configuration; and
    - the root directory where pipeline outputs are written.

    ``feature_engineering`` may be omitted. In that case the workflow selects
    the schema-owned Part C default profile matching ``training.model_type``.
    This prevents an Isolation Forest request from accidentally receiving the
    XGBoost default feature profile, and vice versa.
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
            "Root directory where the workflow should save EDA, "
            "preprocessing, feature-engineering, and training outputs."
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
        """
        Convert the supplied output-directory value into a normalized Path.

        Args:
            value:
                Output directory supplied through JSON.

        Returns:
            Path:
                Normalized output-directory path.
        """

        return Path(
            value
        ).expanduser()


__all__ = [
    "PipelineRunRequest",
    "TrainingConfig",
]
