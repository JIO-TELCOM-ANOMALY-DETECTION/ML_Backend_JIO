"""
Public exports for Part D model training.

Intended project location::

    app/ml/training/__init__.py

The parent pipeline should import only the Part D runner entry point and its
top-level result contracts from this package.

Fixed/direct versus Optuna model-training dispatch is intentionally owned by
runner.py and is not exposed to the parent Part A/B/C/D pipeline.
"""

from .runner import (
    PartDTrainingResult,
    SkippedSeries,
    run_training,
)


__all__ = [
    "PartDTrainingResult",
    "SkippedSeries",
    "run_training",
]