"""
Public interface for XGBoost inference.

Intended project location::

    app/ml/inference/xgboost/__init__.py
"""

from .inference import (
    XGBoostSeriesInferenceResult,
    run_xgboost_inference,
)


__all__ = [
    "XGBoostSeriesInferenceResult",
    "run_xgboost_inference",
]