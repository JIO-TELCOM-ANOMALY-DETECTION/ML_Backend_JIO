"""
Public interface for Isolation Forest inference.

Intended project location::

    app/ml/inference/isolation_forest/__init__.py
"""

from .inference import (
    IsolationForestSeriesInferenceResult,
    run_isolation_forest_inference,
)


__all__ = [
    "IsolationForestSeriesInferenceResult",
    "run_isolation_forest_inference",
]