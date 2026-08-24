"""
Public interface for the pure-inference module.

Intended project location::

    app/ml/inference/__init__.py
"""

from .runner import (
    InferenceRunnerResult,
    InferenceSeriesOutput,
    SkippedInferenceSeries,
    run_inference_pipeline,
)


__all__ = [
    "InferenceRunnerResult",
    "InferenceSeriesOutput",
    "SkippedInferenceSeries",
    "run_inference_pipeline",
]