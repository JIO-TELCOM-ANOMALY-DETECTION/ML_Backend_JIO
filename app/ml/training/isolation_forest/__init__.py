"""
Public exports for Part D Isolation Forest training.

Intended project location::

    app/ml/training/isolation_forest/__init__.py

The Part D runner should import public stage functions/result contracts from
this package rather than reaching into individual implementation files.
"""

from .training import (
    IsolationForestTrainingResult,
    train_isolation_forest_fixed_series,
    train_isolation_forest_optuna_series,
)
from .testing import (
    IsolationForestTestingResult,
    test_isolation_forest_series,
)


__all__ = [
    "IsolationForestTrainingResult",
    "train_isolation_forest_fixed_series",
    "train_isolation_forest_optuna_series",
    "IsolationForestTestingResult",
    "test_isolation_forest_series",
]