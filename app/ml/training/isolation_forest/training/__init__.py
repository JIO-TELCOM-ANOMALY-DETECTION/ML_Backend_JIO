"""
Public interface for Part D Isolation Forest training implementations.

The package exposes separate fixed/direct and Optuna training paths while
sharing the same IsolationForestTrainingResult contract.
"""

from .fixed_training import (
    IsolationForestTrainingResult,
    train_isolation_forest_series as train_isolation_forest_fixed_series,
)
from .optuna_training import (
    train_isolation_forest_series as train_isolation_forest_optuna_series,
)


__all__ = [
    "IsolationForestTrainingResult",
    "train_isolation_forest_fixed_series",
    "train_isolation_forest_optuna_series",
]