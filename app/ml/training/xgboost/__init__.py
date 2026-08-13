"""
Public exports for Part D XGBoost training.

Intended project location::

    app/ml/training/xgboost/__init__.py

The runner should import the XGBoost stage functions and result contracts
from this package rather than reaching into individual implementation files.
"""

from .training import (
    XGBoostTrainingResult,
    train_xgboost_series,
)
from .residual_calibration import (
    XGBoostCalibrationResult,
    calibrate_xgboost_series,
)
from .testing import (
    XGBoostTestingResult,
    test_xgboost_series,
)


__all__ = [
    "XGBoostTrainingResult",
    "train_xgboost_series",
    "XGBoostCalibrationResult",
    "calibrate_xgboost_series",
    "XGBoostTestingResult",
    "test_xgboost_series",
]