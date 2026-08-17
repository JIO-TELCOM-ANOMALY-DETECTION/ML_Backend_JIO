"""
Public interface for Part D XGBoost training implementations.

The package exposes separate fixed/direct and Optuna training paths while
sharing the same XGBoostTrainingResult contract.
"""

from .fixed_training import (
    XGBoostTrainingResult,
    train_xgboost_series as train_xgboost_fixed_series,
)

from .optuna_training import (
    train_xgboost_series as train_xgboost_optuna_series,
)


__all__ = [
    "XGBoostTrainingResult",
    "train_xgboost_fixed_series",
    "train_xgboost_optuna_series",
]