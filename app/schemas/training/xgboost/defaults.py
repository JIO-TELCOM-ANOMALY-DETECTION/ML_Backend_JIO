"""
Default values for Part D XGBoost training.

Intended project location::

    app/schemas/training/xgboost/defaults.py

This module contains XGBoost-specific data-only defaults.

It deliberately does not import Pydantic models from ``xgboost.py`` so the
package cannot create a circular import. Validation and default resolution
remain in ``xgboost.py``.

The hyperparameter defaults and candidate list are based on the existing
offline XGBoost training pipeline. History is not represented as a split
percentage: Part D derives the required history from Part C's resolved
``minimum_history_points`` value and applies the percentages below only to
the remaining rows.
"""

from __future__ import annotations

from typing import Final


# ---------------------------------------------------------------------------
# Fixed XGBoost estimator settings
# ---------------------------------------------------------------------------

DEFAULT_XGBOOST_BASE_PARAMS: Final = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "eval_metric": "mae",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}


# ---------------------------------------------------------------------------
# User-configurable hyperparameter contract
# ---------------------------------------------------------------------------

# This dictionary defines both the internal single default and the complete
# set of keys that every user-supplied configuration/candidate must contain.
# Add or remove configurable hyperparameters here rather than in xgboost.py.

DEFAULT_XGBOOST_HYPERPARAMETERS: Final = {
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_child_weight": 5,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "gamma": 0.0,
}


# Generic validation metadata consumed by xgboost.py.

XGBOOST_HYPERPARAMETER_RULES: Final = {
    "learning_rate": {
        "type": "number",
        "gt": 0.0,
    },
    "max_depth": {
        "type": "int",
        "ge": 0,
    },
    "min_child_weight": {
        "type": "number",
        "ge": 0.0,
    },
    "subsample": {
        "type": "number",
        "gt": 0.0,
        "le": 1.0,
    },
    "colsample_bytree": {
        "type": "number",
        "gt": 0.0,
        "le": 1.0,
    },
    "reg_alpha": {
        "type": "number",
        "ge": 0.0,
    },
    "reg_lambda": {
        "type": "number",
        "ge": 0.0,
    },
    "gamma": {
        "type": "number",
        "ge": 0.0,
    },
}


# ---------------------------------------------------------------------------
# Default hyperparameter tuning candidates
# ---------------------------------------------------------------------------

DEFAULT_XGBOOST_HYPERPARAMETER_CANDIDATES: Final = (
    {
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_weight": 5,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "gamma": 0.0,
    },
    {
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 5,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "gamma": 0.0,
    },
    {
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 10,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.0,
        "reg_lambda": 5.0,
        "gamma": 0.0,
    },
    {
        "learning_rate": 0.10,
        "max_depth": 3,
        "min_child_weight": 3,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.10,
        "reg_lambda": 1.0,
        "gamma": 0.0,
    },
)


# ---------------------------------------------------------------------------
# Hyperparameter tuning controls
# ---------------------------------------------------------------------------

DEFAULT_XGBOOST_MAX_ESTIMATORS: Final = 800
DEFAULT_XGBOOST_EARLY_STOPPING_ROUNDS: Final = 40

DEFAULT_XGBOOST_PRIMARY_SELECTION_METRIC: Final = "mae"
DEFAULT_XGBOOST_SECONDARY_SELECTION_METRIC: Final = "rmse"

XGBOOST_ALLOWED_SELECTION_METRICS: Final = frozenset(
    {
        "mae",
        "rmse",
    }
)



# ---------------------------------------------------------------------------
# Residual calibration defaults
# ---------------------------------------------------------------------------

# The configured quantile represents the upper residual tail.
#
# Example:
#     residual_quantile = 0.99
#
# gives:
#     upper threshold quantile = 0.99
#     lower threshold quantile = 0.01
#
# The common Part D threshold_direction decides whether residual calibration
# uses only the lower tail, only the upper tail, or both tails.
DEFAULT_XGBOOST_RESIDUAL_QUANTILE: Final = 0.99


# ---------------------------------------------------------------------------
# Post-history chronological split defaults
# ---------------------------------------------------------------------------

# History is NOT included in these percentages.
#
# The loader first reserves:
#
#     history_points = Part C resolved minimum_history_points
#
# The remaining rows are then split chronologically using the four
# percentages below, which must total 100.

DEFAULT_XGBOOST_TRAIN_PERCENTAGE: Final = 60.0
DEFAULT_XGBOOST_VALIDATION_PERCENTAGE: Final = 10.0
DEFAULT_XGBOOST_CALIBRATION_PERCENTAGE: Final = 15.0
DEFAULT_XGBOOST_TEST_PERCENTAGE: Final = 15.0


# ---------------------------------------------------------------------------
# Model-specific dataset-validation defaults
# ---------------------------------------------------------------------------

# XGBoost must use exactly the feature columns declared by the resolved Part C
# feature contract. Original Part B columns retained beside those features are
# metadata/source columns and are never added to X automatically.
DEFAULT_XGBOOST_USE_ONLY_RESOLVED_FEATURE_COLUMNS: Final = True
DEFAULT_XGBOOST_REQUIRE_ALL_RESOLVED_FEATURES: Final = True

# Part C feature request families that directly contain the current KPI value
# and therefore must not enter X when that same KPI value is the XGBoost y
# target. They may still exist in a reused Part C dataset; loader.py simply
# excludes their resolved output columns from X.
DEFAULT_XGBOOST_EXCLUDED_PART_C_FEATURE_TYPES: Final = (
    "current_value",
    "difference",
    "percentage_change",
    "seasonal_difference",
    "rolling_zscore",
    "residual",
)

# Logical column-role contract consumed by Part D validation.py.
#
# ``dataset_mapping`` entries refer to attributes on DatasetColumnMapping and
# are resolved to the physical column names supplied for the current dataset.
# ``fixed_output`` entries are stable Part B/Part C metadata column names.
# ``part_c_resolved_features`` refers to ResolvedFeatureEngineeringConfig's
# feature_columns tuple.
#
# Keeping these rules in defaults.py means physical/model-role policy can be
# adjusted without changing the XGBoost schema implementation.
DEFAULT_XGBOOST_COLUMN_ROLE_RULES: Final = {
    "model_features": {
        "source": "part_c_resolved_features",
        "required": True,
        "use_as_x": True,
        "exclude_feature_types": (
            "current_value",
            "difference",
            "percentage_change",
            "seasonal_difference",
            "rolling_zscore",
            "residual",
        ),
        "use_as_y": False,
    },
    "target": {
        "source": "dataset_mapping",
        "logical_name": "value_column",
        "required": True,
        "use_as_x": False,
        "use_as_y": True,
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
        "purpose": "trusted_target_row_filter",
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

# Status 1 = original valid
# Status 3 = safe filled
DEFAULT_XGBOOST_TRUSTED_TARGET_STATUSES: Final = frozenset(
    {
        1,
        3,
    }
)

DEFAULT_XGBOOST_REQUIRE_COMPLETE_FEATURE_ROWS: Final = True
DEFAULT_XGBOOST_REQUIRE_NON_NULL_TARGET: Final = True


# ---------------------------------------------------------------------------
# Minimum usable rows
# ---------------------------------------------------------------------------

DEFAULT_XGBOOST_MINIMUM_TRAIN_POINTS: Final = 1_000
DEFAULT_XGBOOST_MINIMUM_VALIDATION_POINTS: Final = 400
DEFAULT_XGBOOST_MINIMUM_CALIBRATION_POINTS: Final = 400
DEFAULT_XGBOOST_MINIMUM_TEST_POINTS: Final = 400


__all__ = [
    name
    for name in globals()
    if (
        name.startswith("DEFAULT_XGBOOST_")
        or name.startswith("XGBOOST_")
    )
]