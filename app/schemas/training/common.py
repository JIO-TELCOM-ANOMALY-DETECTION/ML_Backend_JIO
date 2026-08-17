"""
Shared schemas for Part D model training.

This module defines configuration shared by every supported training
algorithm.

Model-specific hyperparameters, tuning search spaces, data splits,
feature requirements, calibration settings, and defaults belong to their
respective model schema packages.

This module does not load datasets, train models, calculate thresholds,
or save model outputs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from app.schemas.preprocessing import OutlierDirection


TrainingModelType = Literal[
    "xgboost",
    "isolation_forest",
]


HyperparameterTuningMethod = Literal[
    "fixed",
    "optuna",
]


class CommonTrainingConfig(BaseModel):
    """
    Define settings shared by every supported training algorithm.

    The selected threshold direction controls which side or sides of the
    model-specific residual or anomaly-score distribution are considered
    during calibration.

    Hyperparameter tuning controls whether model-specific tuning is performed.
    When tuning is enabled, hyperparameter_tuning_method selects the tuning
    strategy. When tuning is disabled, one model-specific hyperparameter
    configuration is used directly.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    model_type: TrainingModelType = Field(
        description=(
            "Training algorithm selected for the current model run."
        ),
    )

    threshold_direction: OutlierDirection = Field(
        description=(
            "Apply model-specific calibration to the lower side, upper "
            "side, or both sides of the resulting score or residual "
            "distribution."
        ),
    )

    hyperparameter_tuning_enabled: bool = Field(
        description=(
            "When True, perform model-specific hyperparameter tuning. "
            "When False, use one model-specific hyperparameter "
            "configuration directly."
        ),
    )

    hyperparameter_tuning_method: HyperparameterTuningMethod = Field(
        default="fixed",
        description=(
            "Hyperparameter tuning strategy used when tuning is enabled. "
            "'fixed' evaluates complete predefined candidate configurations; "
            "'optuna' searches a model-specific hyperparameter search space. "
            "Ignored when hyperparameter_tuning_enabled is False."
        ),
    )


__all__ = [
    "CommonTrainingConfig",
    "TrainingModelType",
    "HyperparameterTuningMethod",
]