"""
Default values for Part D Isolation Forest training.

Intended project location::

    app/schemas/training/isolation_forest/defaults.py

This module contains Isolation Forest-specific data-only defaults.

It deliberately does not import Pydantic models from ``isolation_forest.py``
so the package cannot create circular imports. Validation and default
resolution remain in ``isolation_forest.py``.

History is not represented as a split percentage. Part D derives the required
history from Part C's resolved ``minimum_history_points`` value and applies
the split percentages below only to the remaining rows.

Isolation Forest uses ``contamination`` to define its anomaly decision
boundary, so no separate residual-calibration split is configured here.
"""

from __future__ import annotations

from typing import Final


# ---------------------------------------------------------------------------
# Fixed Isolation Forest estimator settings
# ---------------------------------------------------------------------------

# Fixed reproducibility/execution settings. These are not part of the
# user-supplied hyperparameter contract.
DEFAULT_ISOLATION_FOREST_BASE_PARAMS: Final = {
    "n_jobs": -1,
    "random_state": 42,
    "verbose": 0,
    "warm_start": False,
}


# ---------------------------------------------------------------------------
# User-configurable hyperparameter contract
# ---------------------------------------------------------------------------

# This dictionary defines both:
#
# 1. the internal single default used when tuning is OFF and the user supplies
#    no hyperparameters; and
# 2. the complete set of keys that every user-supplied hyperparameter
#    configuration/candidate must contain.
#
# Add or remove configurable hyperparameters here rather than changing
# isolation_forest.py.

DEFAULT_ISOLATION_FOREST_HYPERPARAMETERS: Final = {
    "n_estimators": 300,
    "max_samples": "auto",
    "contamination": 0.01,
    "max_features": 1.0,
    "bootstrap": False,
}


# Generic validation metadata consumed by isolation_forest.py.
# Parameter-specific validation remains data-driven from this dictionary.

ISOLATION_FOREST_HYPERPARAMETER_RULES: Final = {
    "n_estimators": {
        "type": "int",
        "gt": 0,
    },
    "max_samples": {
        "type": "max_samples",
    },
    "contamination": {
        "type": "contamination",
    },
    "max_features": {
        "type": "max_features",
    },
    "bootstrap": {
        "type": "bool",
    },
}


# ---------------------------------------------------------------------------
# Default hyperparameter tuning candidates
# ---------------------------------------------------------------------------

# Every candidate is a complete user-configurable hyperparameter set.
#
# The default list intentionally keeps contamination fixed at 1% while
# varying the forest-size and sampling behaviour. Without labelled anomalies,
# automatically selecting between substantially different contamination
# values is less reliable. Users may still provide their own complete
# candidate list containing different valid contamination values.

DEFAULT_ISOLATION_FOREST_HYPERPARAMETER_CANDIDATES: Final = (
    {
        "n_estimators": 200,
        "max_samples": "auto",
        "contamination": 0.01,
        "max_features": 1.0,
        "bootstrap": False,
    },
    {
        "n_estimators": 300,
        "max_samples": "auto",
        "contamination": 0.01,
        "max_features": 1.0,
        "bootstrap": False,
    },
    {
        "n_estimators": 300,
        "max_samples": 0.80,
        "contamination": 0.01,
        "max_features": 1.0,
        "bootstrap": False,
    },
    {
        "n_estimators": 300,
        "max_samples": "auto",
        "contamination": 0.01,
        "max_features": 0.80,
        "bootstrap": False,
    },
)


# ---------------------------------------------------------------------------
# Hyperparameter tuning controls
# ---------------------------------------------------------------------------

# Internal validation metric used to compare complete candidates when tuning
# is enabled. The tuning module should calculate the separation between the
# validation decision scores of model-labelled inliers and outliers and prefer
# larger separation.
DEFAULT_ISOLATION_FOREST_SELECTION_METRIC: Final = "score_separation"

ISOLATION_FOREST_ALLOWED_SELECTION_METRICS: Final = frozenset(
    {
        "score_separation",
    }
)


# ---------------------------------------------------------------------------
# Post-history chronological split defaults
# ---------------------------------------------------------------------------

# History is NOT included in these percentages.
#
# TUNING ENABLED:
#
#     history -> train -> validation -> test
#
# TUNING DISABLED:
#
#     history -> train -> test
#
# The schema resolves the appropriate default set from the tuning switch.

DEFAULT_ISOLATION_FOREST_TUNING_TRAIN_PERCENTAGE: Final = 70.0
DEFAULT_ISOLATION_FOREST_TUNING_VALIDATION_PERCENTAGE: Final = 15.0
DEFAULT_ISOLATION_FOREST_TUNING_TEST_PERCENTAGE: Final = 15.0

DEFAULT_ISOLATION_FOREST_DIRECT_TRAIN_PERCENTAGE: Final = 85.0
DEFAULT_ISOLATION_FOREST_DIRECT_TEST_PERCENTAGE: Final = 15.0


# ---------------------------------------------------------------------------
# Model-specific dataset-validation defaults
# ---------------------------------------------------------------------------

# Isolation Forest has no y target. X is built strictly from Part C's resolved
# feature_columns. Retained source/metadata columns are not inferred as model
# features.
DEFAULT_ISOLATION_FOREST_USE_ONLY_RESOLVED_FEATURE_COLUMNS: Final = True
DEFAULT_ISOLATION_FOREST_REQUIRE_ALL_RESOLVED_FEATURES: Final = True

# Logical column-role contract consumed by Part D validation.py.
#
# The source KPI value is explicitly NOT a target and NOT automatically an X
# feature. If Part C intentionally creates ``current_value``, that generated
# feature is allowed because it appears in the resolved Part C feature list.
DEFAULT_ISOLATION_FOREST_COLUMN_ROLE_RULES: Final = {
    "model_features": {
        "source": "part_c_resolved_features",
        "required": True,
        "use_as_x": True,
        "use_as_y": False,
    },
    "target": {
        "source": None,
        "logical_name": None,
        "required": False,
        "use_as_x": False,
        "use_as_y": False,
    },
    "dataset_value": {
        "source": "dataset_mapping",
        "logical_name": "value_column",
        "required": False,
        "use_as_x": False,
        "use_as_y": False,
    },
    "site_identity": {
        "source": "dataset_mapping",
        "logical_names": (
            "circle_column",
            "cluster_column",
            "site_name_column",
        ),
        "required": True,
        "use_as_x": False,
        "use_as_y": False,
    },
    "timestamp": {
        "source": "dataset_mapping",
        "logical_name": "timestamp_column",
        "required": True,
        "use_as_x": False,
        "use_as_y": False,
    },
    "status": {
        "source": "fixed_output",
        "column_name": "Current_Status_Label",
        "required": True,
        "use_as_x": False,
        "use_as_y": False,
        "purpose": "trusted_training_row_filter",
    },
    "usability": {
        "source": "fixed_output",
        "column_name": "kpi_usability",
        "required": True,
        "use_as_x": False,
        "use_as_y": False,
        "purpose": "series_usability_filter",
    },
    "invalid_reason": {
        "source": "fixed_output",
        "column_name": "kpi_invalid_reason",
        "required": True,
        "use_as_x": False,
        "use_as_y": False,
        "purpose": "series_usability_filter",
    },
}

# Isolation Forest's anomaly score is non-directional. For one-sided threshold
# requests, the sign of this Part C feature determines whether an identified
# anomaly belongs to the requested side. ``both`` does not require the feature
# solely for directional filtering.
DEFAULT_ISOLATION_FOREST_DIRECTION_FEATURE_RULE: Final = {
    "feature_name": "residual",
    "required_for_threshold_directions": (
        "lower_only",
        "upper_only",
    ),
}

# Status 1 = original valid
# Status 3 = safe filled
DEFAULT_ISOLATION_FOREST_TRUSTED_TRAIN_STATUSES: Final = frozenset(
    {
        1,
        3,
    }
)

DEFAULT_ISOLATION_FOREST_REQUIRE_COMPLETE_FEATURE_ROWS: Final = True


# ---------------------------------------------------------------------------
# Minimum usable rows
# ---------------------------------------------------------------------------

DEFAULT_ISOLATION_FOREST_MINIMUM_TRAIN_POINTS: Final = 1_000
DEFAULT_ISOLATION_FOREST_MINIMUM_VALIDATION_POINTS: Final = 400
DEFAULT_ISOLATION_FOREST_MINIMUM_TEST_POINTS: Final = 400


__all__ = [
    name
    for name in globals()
    if (
        name.startswith("DEFAULT_ISOLATION_FOREST_")
        or name.startswith("ISOLATION_FOREST_")
    )
]