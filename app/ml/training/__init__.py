"""
Public exports for Part D model training.

Intended project location::

    app/ml/training/__init__.py

The parent pipeline should import the Part D runner entry point and its
top-level result contracts from this package rather than reaching directly
into runner.py.
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
