"""Public orchestration interface for Part C feature engineering.

Intended project location::

    app/ml/feature_engineering/__init__.py

Feature calculations remain in the individual modules. This package-level
interface exposes only the runner result models and the main Part C entry point
used by workflows and API orchestration.
"""

from .runner import (
    FeatureEngineeringRunnerResult,
    FeatureEngineeringSiteRunSummary,
    run_feature_engineering_pipeline,
)

__all__ = [
    "FeatureEngineeringRunnerResult",
    "FeatureEngineeringSiteRunSummary",
    "run_feature_engineering_pipeline",
]