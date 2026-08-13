"""
Part D XGBoost residual calibration.

Intended project location::

    app/ml/training/xgboost/residual_calibration.py

This module calibrates anomaly thresholds for ONE already-trained XGBoost
site-series model using that series' dedicated calibration split.

Residual definition
-------------------
    residual = actual_target - predicted_target

Therefore:
- negative residuals mean the observed KPI is below the model prediction;
- positive residuals mean the observed KPI is above the model prediction.

The configured ``residual_quantile`` is the upper-tail quantile. The lower
quantile is its complement::

    lower_quantile = 1 - residual_quantile

Direction behavior
------------------
``lower_only``:
    calculate/use only the lower residual threshold.

``upper_only``:
    calculate/use only the upper residual threshold.

``both``:
    calculate/use both lower and upper residual thresholds.

This module does NOT:
- train or refit XGBoost;
- use train/validation rows;
- use test rows;
- perform final test evaluation;
- save model/output files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from app.ml.training.loader import (
    SeriesIdentity,
    SeriesSplitPlan,
    TrainingSplitPlan,
)
from app.ml.training.validation import TrainingValidationResult
from app.schemas.training.xgboost.xgboost import (
    XGBoostTrainingConfig,
)


CALIBRATION_READ_CHUNK_SIZE = 250_000
RESIDUAL_QUANTILE_METHOD = "linear"


@dataclass(
    frozen=True,
    slots=True,
)
class XGBoostCalibrationResult:
    """Residual-calibration output for one site-series XGBoost model."""

    identity: SeriesIdentity

    threshold_direction: str
    residual_quantile: float

    lower_quantile: float | None
    upper_quantile: float | None

    lower_threshold: float | None
    upper_threshold: float | None

    usable_row_count: int

    decision_settings: dict[str, Any]
    calibration_summary: dict[str, Any]


def _validate_calibration_inputs(
    *,
    trained_model: XGBRegressor,
    validation_result: TrainingValidationResult,
    split_plan: TrainingSplitPlan,
    series_plan: SeriesSplitPlan,
    training_config: XGBoostTrainingConfig,
) -> None:
    """Ensure all supplied objects describe one XGBoost calibration request."""

    if validation_result.model_type != "xgboost":
        raise ValueError(
            "XGBoost residual calibration requires "
            "validation_result.model_type='xgboost'."
        )

    if split_plan.model_type != "xgboost":
        raise ValueError(
            "XGBoost residual calibration requires "
            "split_plan.model_type='xgboost'."
        )

    if training_config.model_type != "xgboost":
        raise ValueError(
            "XGBoost residual calibration requires XGBoostTrainingConfig."
        )

    if validation_result.input_file != split_plan.input_file:
        raise ValueError(
            "validation_result and split_plan reference different Part C "
            "datasets."
        )

    if validation_result.target_column is None:
        raise ValueError(
            "XGBoost residual calibration requires a validated target column."
        )

    if not validation_result.model_feature_columns:
        raise ValueError(
            "XGBoost residual calibration requires validated model features."
        )

    if series_plan.calibration is None:
        raise ValueError(
            "XGBoost residual calibration requires a calibration split."
        )

    if not callable(
        getattr(
            trained_model,
            "predict",
            None,
        )
    ):
        raise TypeError(
            "trained_model must provide a callable predict() method."
        )


def _series_identity_mask(
    chunk: pd.DataFrame,
    *,
    identity_columns: tuple[str, str, str],
    identity: SeriesIdentity,
) -> pd.Series:
    """Return rows belonging to exactly one planned site identity."""

    expected_values = identity.as_tuple()

    mask = pd.Series(
        True,
        index=chunk.index,
        dtype=bool,
    )

    for column, expected in zip(
        identity_columns,
        expected_values,
        strict=True,
    ):
        mask &= (
            chunk[column]
            .astype("string")
            .eq(str(expected))
            .fillna(False)
        )

    return mask


def _load_calibration_rows(
    *,
    validation_result: TrainingValidationResult,
    split_plan: TrainingSplitPlan,
    series_plan: SeriesSplitPlan,
) -> pd.DataFrame:
    """Load only the dedicated calibration range for one series."""

    calibration_range = series_plan.calibration

    if calibration_range is None:
        raise ValueError(
            "Calibration split is missing."
        )

    target_column = validation_result.target_column

    if target_column is None:
        raise ValueError(
            "XGBoost target column is missing."
        )

    required_columns = list(
        dict.fromkeys(
            [
                *validation_result.site_identity_columns,
                validation_result.timestamp_column,
                validation_result.status_column,
                target_column,
                *validation_result.model_feature_columns,
            ]
        )
    )

    start_timestamp = pd.Timestamp(
        calibration_range.start_timestamp
    )

    end_timestamp = pd.Timestamp(
        calibration_range.end_timestamp
    )

    selected_chunks: list[pd.DataFrame] = []

    reader = pd.read_csv(
        split_plan.input_file,
        usecols=required_columns,
        chunksize=CALIBRATION_READ_CHUNK_SIZE,
        low_memory=False,
    )

    for chunk in reader:
        identity_mask = _series_identity_mask(
            chunk,
            identity_columns=validation_result.site_identity_columns,
            identity=series_plan.identity,
        )

        if not bool(
            identity_mask.any()
        ):
            continue

        subset = chunk.loc[
            identity_mask
        ].copy()

        subset[
            validation_result.timestamp_column
        ] = pd.to_datetime(
            subset[
                validation_result.timestamp_column
            ],
            errors="raise",
        )

        time_mask = (
            subset[
                validation_result.timestamp_column
            ].ge(start_timestamp)
            & subset[
                validation_result.timestamp_column
            ].le(end_timestamp)
        )

        subset = subset.loc[
            time_mask
        ]

        if not subset.empty:
            selected_chunks.append(
                subset
            )

    if not selected_chunks:
        raise ValueError(
            "No XGBoost calibration rows were found for series "
            f"{series_plan.identity.as_tuple()}."
        )

    rows = pd.concat(
        selected_chunks,
        ignore_index=True,
        sort=False,
    )

    rows = (
        rows
        .sort_values(
            validation_result.timestamp_column,
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    if bool(
        rows[
            validation_result.timestamp_column
        ].duplicated().any()
    ):
        raise ValueError(
            "Duplicate timestamps were found while loading the XGBoost "
            f"calibration range for series {series_plan.identity.as_tuple()}."
        )

    if len(
        rows
    ) != calibration_range.row_count:
        raise ValueError(
            "XGBoost raw calibration row count does not match loader.py "
            f"for series {series_plan.identity.as_tuple()}. "
            f"Expected {calibration_range.row_count}, found {len(rows)}."
        )

    return rows


def _prepare_usable_calibration_rows(
    rows: pd.DataFrame,
    *,
    validation_result: TrainingValidationResult,
) -> pd.DataFrame:
    """Apply the same XGBoost row-eligibility contract used for training."""

    feature_columns = list(
        validation_result.model_feature_columns
    )

    target_column = validation_result.target_column

    if target_column is None:
        raise ValueError(
            "XGBoost target column is missing."
        )

    status_column = validation_result.status_column

    prepared = rows.copy()

    for column in [
        *feature_columns,
        target_column,
        status_column,
    ]:
        prepared[
            column
        ] = pd.to_numeric(
            prepared[
                column
            ],
            errors="coerce",
        )

    mask = (
        prepared[
            status_column
        ]
        .isin(
            validation_result.trusted_statuses
        )
        .fillna(False)
    )

    if validation_result.require_complete_feature_rows:
        mask &= (
            prepared[
                feature_columns
            ]
            .notna()
            .all(axis=1)
        )

    if validation_result.require_non_null_target:
        mask &= prepared[
            target_column
        ].notna()

    return (
        prepared.loc[
            mask
        ]
        .sort_values(
            validation_result.timestamp_column,
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def _calculate_residuals(
    *,
    trained_model: XGBRegressor,
    rows: pd.DataFrame,
    validation_result: TrainingValidationResult,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Predict the calibration target and return actual/prediction/residual."""

    target_column = validation_result.target_column

    if target_column is None:
        raise ValueError(
            "XGBoost target column is missing."
        )

    x = rows.loc[
        :,
        list(
            validation_result.model_feature_columns
        ),
    ]

    actual = rows[
        target_column
    ].to_numpy(
        dtype=np.float64
    )

    predicted = np.asarray(
        trained_model.predict(
            x
        ),
        dtype=np.float64,
    )

    if predicted.shape != actual.shape:
        raise ValueError(
            "XGBoost calibration prediction shape does not match target shape."
        )

    if bool(
        np.isnan(
            predicted
        ).any()
    ) or bool(
        np.isinf(
            predicted
        ).any()
    ):
        raise ValueError(
            "XGBoost produced invalid calibration predictions."
        )

    residual = (
        actual
        - predicted
    )

    return (
        actual,
        predicted,
        residual,
    )


def _derive_thresholds(
    *,
    residual: np.ndarray,
    threshold_direction: str,
    residual_quantile: float,
) -> tuple[
    float | None,
    float | None,
    float | None,
    float | None,
]:
    """Calculate only the residual threshold(s) required by direction."""

    lower_quantile_value = (
        1.0
        - residual_quantile
    )

    lower_quantile: float | None = None
    upper_quantile: float | None = None

    lower_threshold: float | None = None
    upper_threshold: float | None = None

    if threshold_direction in {
        "lower_only",
        "both",
    }:
        lower_quantile = (
            lower_quantile_value
        )

        lower_threshold = float(
            np.quantile(
                residual,
                lower_quantile,
                method=RESIDUAL_QUANTILE_METHOD,
            )
        )

    if threshold_direction in {
        "upper_only",
        "both",
    }:
        upper_quantile = (
            residual_quantile
        )

        upper_threshold = float(
            np.quantile(
                residual,
                upper_quantile,
                method=RESIDUAL_QUANTILE_METHOD,
            )
        )

    if threshold_direction not in {
        "lower_only",
        "upper_only",
        "both",
    }:
        raise ValueError(
            "Unsupported XGBoost threshold_direction: "
            f"{threshold_direction}"
        )

    return (
        lower_quantile,
        upper_quantile,
        lower_threshold,
        upper_threshold,
    )


def _calibration_anomaly_mask(
    *,
    residual: np.ndarray,
    threshold_direction: str,
    lower_threshold: float | None,
    upper_threshold: float | None,
) -> np.ndarray:
    """Apply the derived cutoff(s) back to calibration residuals for summary."""

    if threshold_direction == "lower_only":
        if lower_threshold is None:
            raise RuntimeError(
                "lower_only calibration is missing lower_threshold."
            )

        return residual <= lower_threshold

    if threshold_direction == "upper_only":
        if upper_threshold is None:
            raise RuntimeError(
                "upper_only calibration is missing upper_threshold."
            )

        return residual >= upper_threshold

    if threshold_direction == "both":
        if (
            lower_threshold is None
            or upper_threshold is None
        ):
            raise RuntimeError(
                "both-direction calibration requires both thresholds."
            )

        return (
            (residual <= lower_threshold)
            | (residual >= upper_threshold)
        )

    raise ValueError(
        "Unsupported XGBoost threshold_direction: "
        f"{threshold_direction}"
    )


def calibrate_xgboost_series(
    *,
    trained_model: XGBRegressor,
    validation_result: TrainingValidationResult,
    split_plan: TrainingSplitPlan,
    series_plan: SeriesSplitPlan,
    training_config: XGBoostTrainingConfig,
) -> XGBoostCalibrationResult:
    """
    Calibrate residual-quantile anomaly thresholds for one XGBoost site series.
    """

    _validate_calibration_inputs(
        trained_model=trained_model,
        validation_result=validation_result,
        split_plan=split_plan,
        series_plan=series_plan,
        training_config=training_config,
    )

    raw_rows = _load_calibration_rows(
        validation_result=validation_result,
        split_plan=split_plan,
        series_plan=series_plan,
    )

    usable_rows = (
        _prepare_usable_calibration_rows(
            raw_rows,
            validation_result=validation_result,
        )
    )

    if len(
        usable_rows
    ) < validation_result.minimum_calibration_points:
        raise ValueError(
            "Insufficient usable XGBoost calibration rows for series "
            f"{series_plan.identity.as_tuple()}: "
            f"{len(usable_rows)} < minimum "
            f"{validation_result.minimum_calibration_points}."
        )

    (
        actual,
        predicted,
        residual,
    ) = _calculate_residuals(
        trained_model=trained_model,
        rows=usable_rows,
        validation_result=validation_result,
    )

    residual_quantile = float(
        training_config
        .calibration
        .residual_quantile
    )

    threshold_direction = str(
        training_config.threshold_direction
    )

    (
        lower_quantile,
        upper_quantile,
        lower_threshold,
        upper_threshold,
    ) = _derive_thresholds(
        residual=residual,
        threshold_direction=threshold_direction,
        residual_quantile=residual_quantile,
    )

    anomaly_mask = _calibration_anomaly_mask(
        residual=residual,
        threshold_direction=threshold_direction,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
    )

    residual_median = float(
        np.median(
            residual
        )
    )

    calibration_summary: dict[
        str,
        Any,
    ] = {
        "series_identity": {
            "circle": (
                series_plan.identity.circle
            ),
            "cluster": (
                series_plan.identity.cluster
            ),
            "site_name": (
                series_plan.identity.site_name
            ),
        },
        "raw_calibration_rows": len(
            raw_rows
        ),
        "usable_calibration_rows": len(
            usable_rows
        ),
        "threshold_direction": (
            threshold_direction
        ),
        "residual_definition": (
            "actual_minus_prediction"
        ),
        "residual_quantile": (
            residual_quantile
        ),
        "lower_quantile": (
            lower_quantile
        ),
        "upper_quantile": (
            upper_quantile
        ),
        "lower_threshold": (
            lower_threshold
        ),
        "upper_threshold": (
            upper_threshold
        ),
        "quantile_method": (
            RESIDUAL_QUANTILE_METHOD
        ),
        "residual_mean": float(
            np.mean(
                residual
            )
        ),
        "residual_median": (
            residual_median
        ),
        "residual_std": float(
            np.std(
                residual
            )
        ),
        "residual_mad": float(
            np.median(
                np.abs(
                    residual
                    - residual_median
                )
            )
        ),
        "residual_min": float(
            np.min(
                residual
            )
        ),
        "residual_max": float(
            np.max(
                residual
            )
        ),
        "calibration_anomaly_count": int(
            np.sum(
                anomaly_mask
            )
        ),
        "calibration_anomaly_rate": float(
            np.mean(
                anomaly_mask
            )
        ),
        "calibration_mae": float(
            np.mean(
                np.abs(
                    residual
                )
            )
        ),
        "calibration_rmse": float(
            np.sqrt(
                np.mean(
                    np.square(
                        residual
                    )
                )
            )
        ),
    }

    decision_settings: dict[
        str,
        Any,
    ] = {
        "method": (
            "residual_quantile"
        ),
        "residual_definition": (
            "actual_minus_prediction"
        ),
        "threshold_direction": (
            threshold_direction
        ),
        "residual_quantile": (
            residual_quantile
        ),
        "lower_quantile": (
            lower_quantile
        ),
        "upper_quantile": (
            upper_quantile
        ),
        "lower_threshold": (
            lower_threshold
        ),
        "upper_threshold": (
            upper_threshold
        ),
        "quantile_method": (
            RESIDUAL_QUANTILE_METHOD
        ),
        "lower_comparison": (
            "residual <= lower_threshold"
            if lower_threshold is not None
            else None
        ),
        "upper_comparison": (
            "residual >= upper_threshold"
            if upper_threshold is not None
            else None
        ),
    }

    return XGBoostCalibrationResult(
        identity=series_plan.identity,
        threshold_direction=(
            threshold_direction
        ),
        residual_quantile=(
            residual_quantile
        ),
        lower_quantile=(
            lower_quantile
        ),
        upper_quantile=(
            upper_quantile
        ),
        lower_threshold=(
            lower_threshold
        ),
        upper_threshold=(
            upper_threshold
        ),
        usable_row_count=len(
            usable_rows
        ),
        decision_settings=(
            decision_settings
        ),
        calibration_summary=(
            calibration_summary
        ),
    )


__all__ = [
    "XGBoostCalibrationResult",
    "calibrate_xgboost_series",
]