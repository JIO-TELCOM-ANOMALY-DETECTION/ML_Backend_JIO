"""
Pure XGBoost inference for ONE saved KPI/site series.

Intended project location::

    app/ml/inference/xgboost/inference.py

This module mirrors the finalized Part D XGBoost testing decision semantics,
but removes test-split, training, calibration, and evaluation behavior.

Exact saved decision behavior
-----------------------------
    residual = actual_target - predicted_target

Saved thresholds are never recalculated here.

lower_only:
    anomaly when residual <= lower_threshold

upper_only:
    anomaly when residual >= upper_threshold

both:
    anomaly when residual <= lower_threshold
    OR residual >= upper_threshold

The parent runner keeps memory bounded by loading/validating one site's Part C
rows, loading one saved model, calling ``run_xgboost_inference()``, persisting
that site's result, releasing the dataframe/model, and then moving to the next
site.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from app.core.exceptions import InputValidationError
from app.ml.inference.loader import LoadedSeriesModel
from app.ml.inference.validation import ValidatedSeriesData
from app.schemas.inference.xgboost import XGBoostDecisionSettings


XGBoostThresholdDirection = Literal[
    "lower_only",
    "upper_only",
    "both",
]


@dataclass(frozen=True, slots=True)
class XGBoostSeriesInferenceResult:
    """Pure inference result for one saved XGBoost KPI/site model."""

    kpi_name: str
    series_key: str
    identity: tuple[str, str, str]

    inference_start: pd.Timestamp
    inference_end_exclusive: pd.Timestamp

    total_window_rows: int
    scorable_rows: int
    unscorable_rows: int

    anomaly_count: int
    anomaly_rate: float
    lower_anomaly_count: int
    upper_anomaly_count: int

    threshold_direction: XGBoostThresholdDirection
    lower_threshold: float | None
    upper_threshold: float | None

    output: pd.DataFrame


def _validate_inputs(
    *,
    loaded_model: LoadedSeriesModel,
    series_data: ValidatedSeriesData,
    inference_start: pd.Timestamp,
    inference_end_exclusive: pd.Timestamp,
) -> tuple[XGBRegressor, XGBoostDecisionSettings]:
    """Validate one already-loaded one-site XGBoost request."""

    artifact = loaded_model.artifact
    data_artifact = series_data.series_plan.artifact

    if artifact.model_type != "xgboost":
        raise InputValidationError(
            "XGBoost inference requires a loaded XGBoost artifact.",
            error_code="INVALID_XGBOOST_INFERENCE_MODEL_TYPE",
            details={"model_type": artifact.model_type},
        )

    if data_artifact.model_type != "xgboost":
        raise InputValidationError(
            "XGBoost inference requires XGBoost-validated Part C data.",
            error_code="INVALID_XGBOOST_INFERENCE_DATA_TYPE",
            details={"model_type": data_artifact.model_type},
        )

    if artifact != data_artifact:
        raise InputValidationError(
            "Loaded model and validated Part C rows reference different saved "
            "XGBoost artifacts.",
            error_code="XGBOOST_INFERENCE_ARTIFACT_MISMATCH",
            details={
                "loaded_kpi": artifact.kpi_name,
                "loaded_series_key": artifact.series_key,
                "loaded_identity": artifact.identity,
                "data_kpi": data_artifact.kpi_name,
                "data_series_key": data_artifact.series_key,
                "data_identity": data_artifact.identity,
            },
        )

    if artifact.target_column is None:
        raise InputValidationError(
            "Saved XGBoost artifact is missing its target column.",
            error_code="MISSING_XGBOOST_INFERENCE_TARGET_COLUMN",
        )

    if not artifact.model_feature_columns:
        raise InputValidationError(
            "Saved XGBoost artifact contains no model features.",
            error_code="EMPTY_XGBOOST_INFERENCE_FEATURE_SET",
        )

    if (
        tuple(series_data.model_feature_columns)
        != tuple(artifact.model_feature_columns)
    ):
        raise InputValidationError(
            "Validated Part C feature order does not match the saved XGBoost "
            "feature order.",
            error_code="XGBOOST_INFERENCE_FEATURE_ORDER_MISMATCH",
            details={
                "saved_features": list(artifact.model_feature_columns),
                "validated_features": list(series_data.model_feature_columns),
            },
        )

    if series_data.target_column != artifact.target_column:
        raise InputValidationError(
            "Validated Part C target does not match the saved XGBoost target.",
            error_code="XGBOOST_INFERENCE_TARGET_MISMATCH",
            details={
                "saved_target": artifact.target_column,
                "validated_target": series_data.target_column,
            },
        )

    if not isinstance(
        artifact.decision_settings,
        XGBoostDecisionSettings,
    ):
        raise InputValidationError(
            "Saved artifact does not contain XGBoost decision settings.",
            error_code="INVALID_XGBOOST_DECISION_SETTINGS",
        )

    decision_settings = artifact.decision_settings

    if decision_settings.method != "residual_quantile":
        raise InputValidationError(
            "XGBoost inference requires residual_quantile decision settings.",
            error_code="INVALID_XGBOOST_DECISION_METHOD",
        )

    if (
        decision_settings.residual_definition
        != "actual_minus_prediction"
    ):
        raise InputValidationError(
            "XGBoost inference requires residual_definition="
            "'actual_minus_prediction'.",
            error_code="INVALID_XGBOOST_RESIDUAL_DEFINITION",
        )

    model = loaded_model.estimator

    if not callable(getattr(model, "predict", None)):
        raise InputValidationError(
            "Loaded XGBoost estimator does not provide predict().",
            error_code="INVALID_LOADED_XGBOOST_ESTIMATOR",
        )

    start = pd.Timestamp(inference_start)
    end = pd.Timestamp(inference_end_exclusive)

    if end <= start:
        raise InputValidationError(
            "inference_end_exclusive must be later than inference_start.",
            error_code="INVALID_XGBOOST_INFERENCE_WINDOW",
            details={
                "inference_start": str(start),
                "inference_end_exclusive": str(end),
            },
        )

    return model, decision_settings


def _timestamp_column(
    dataframe: pd.DataFrame,
) -> str:
    """
    Return the one parsed timestamp column retained by parent validation.
    """

    candidates = [
        column
        for column in dataframe.columns
        if pd.api.types.is_datetime64_any_dtype(dataframe[column])
    ]

    if len(candidates) != 1:
        raise InputValidationError(
            "Validated XGBoost site data must contain exactly one parsed "
            "timestamp column.",
            error_code="XGBOOST_INFERENCE_TIMESTAMP_COLUMN_AMBIGUOUS",
            details={"datetime_columns": candidates},
        )

    return candidates[0]


def _select_window(
    *,
    series_data: ValidatedSeriesData,
    inference_start: pd.Timestamp,
    inference_end_exclusive: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Series]:
    """Select only the requested forward inference window."""

    dataframe = series_data.dataframe
    timestamp_column = _timestamp_column(dataframe)

    start = pd.Timestamp(inference_start)
    end = pd.Timestamp(inference_end_exclusive)

    window_mask = (
        dataframe[timestamp_column].ge(start)
        & dataframe[timestamp_column].lt(end)
    )

    source_indices = dataframe.index[window_mask]

    rows = (
        dataframe.loc[window_mask]
        .copy()
        .reset_index(drop=True)
    )

    if rows.empty:
        raise InputValidationError(
            "No XGBoost Part C rows exist inside the requested inference "
            "window.",
            error_code="EMPTY_XGBOOST_INFERENCE_WINDOW",
            details={
                "kpi_name": series_data.series_plan.artifact.kpi_name,
                "identity": series_data.series_plan.artifact.identity,
                "inference_start": str(start),
                "inference_end_exclusive": str(end),
            },
        )

    upstream_ready = (
        series_data.scorable_mask
        .loc[source_indices]
        .reset_index(drop=True)
        .astype(bool)
    )

    return rows, upstream_ready


def _scorable_mask(
    *,
    rows: pd.DataFrame,
    upstream_ready: pd.Series,
    feature_columns: tuple[str, ...],
    target_column: str,
) -> pd.Series:
    """
    Require finite model features AND finite actual target.

    The target requirement is mandatory because anomaly inference uses
    residual = actual - prediction.
    """

    feature_values = rows.loc[
        :,
        list(feature_columns),
    ].to_numpy(dtype=np.float64)

    finite_features = pd.Series(
        np.isfinite(feature_values).all(axis=1),
        index=rows.index,
        dtype=bool,
    )

    target_values = rows[
        target_column
    ].to_numpy(dtype=np.float64)

    finite_target = pd.Series(
        np.isfinite(target_values),
        index=rows.index,
        dtype=bool,
    )

    return (
        upstream_ready
        & finite_features
        & finite_target
    )


def _predict_residuals(
    *,
    model: XGBRegressor,
    rows: pd.DataFrame,
    feature_columns: tuple[str, ...],
    target_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate actual, predicted, and actual-minus-prediction residuals.

    This mirrors Part D XGBoost final testing.
    """

    x = rows.loc[
        :,
        list(feature_columns),
    ]

    actual = rows[
        target_column
    ].to_numpy(dtype=np.float64)

    predicted = np.asarray(
        model.predict(x),
        dtype=np.float64,
    )

    if predicted.shape != actual.shape:
        raise InputValidationError(
            "XGBoost inference prediction shape does not match target shape.",
            error_code="XGBOOST_INFERENCE_PREDICTION_SHAPE_MISMATCH",
            details={
                "prediction_shape": list(predicted.shape),
                "target_shape": list(actual.shape),
            },
        )

    if not bool(np.isfinite(predicted).all()):
        raise InputValidationError(
            "XGBoost produced non-finite inference predictions.",
            error_code="INVALID_XGBOOST_INFERENCE_PREDICTIONS",
        )

    residual = actual - predicted

    if not bool(np.isfinite(residual).all()):
        raise InputValidationError(
            "XGBoost produced non-finite inference residuals.",
            error_code="INVALID_XGBOOST_INFERENCE_RESIDUALS",
        )

    return actual, predicted, residual


def _apply_saved_thresholds(
    *,
    residual: np.ndarray,
    decision_settings: XGBoostDecisionSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply saved calibrated thresholds exactly as final testing does."""

    direction = str(
        decision_settings.threshold_direction
    )

    lower_mask = np.zeros(
        residual.shape,
        dtype=bool,
    )
    upper_mask = np.zeros(
        residual.shape,
        dtype=bool,
    )

    if direction in {"lower_only", "both"}:
        if decision_settings.lower_threshold is None:
            raise InputValidationError(
                "Saved lower-side XGBoost inference is missing "
                "lower_threshold.",
                error_code="MISSING_XGBOOST_LOWER_THRESHOLD",
            )

        lower_mask = (
            residual
            <= float(decision_settings.lower_threshold)
        )

    if direction in {"upper_only", "both"}:
        if decision_settings.upper_threshold is None:
            raise InputValidationError(
                "Saved upper-side XGBoost inference is missing "
                "upper_threshold.",
                error_code="MISSING_XGBOOST_UPPER_THRESHOLD",
            )

        upper_mask = (
            residual
            >= float(decision_settings.upper_threshold)
        )

    if direction == "lower_only":
        anomaly_mask = lower_mask
    elif direction == "upper_only":
        anomaly_mask = upper_mask
    elif direction == "both":
        anomaly_mask = lower_mask | upper_mask
    else:
        raise InputValidationError(
            "Unsupported saved XGBoost threshold_direction.",
            error_code="UNSUPPORTED_XGBOOST_THRESHOLD_DIRECTION",
            details={"threshold_direction": direction},
        )

    return anomaly_mask, lower_mask, upper_mask


def _build_output(
    *,
    artifact,
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create standardized output while preserving unscorable requested rows.
    """

    timestamp_column = _timestamp_column(rows)

    return pd.DataFrame(
        {
            "kpi_name": artifact.kpi_name,
            "series_key": artifact.series_key,
            "circle": artifact.identity[0],
            "cluster": artifact.identity[1],
            "site_name": artifact.identity[2],
            "timestamp": rows[timestamp_column].to_numpy(),
            "scorable": False,
            "actual": np.nan,
            "prediction": np.nan,
            "residual": np.nan,
            "lower_anomaly": pd.Series(
                pd.NA,
                index=rows.index,
                dtype="boolean",
            ),
            "upper_anomaly": pd.Series(
                pd.NA,
                index=rows.index,
                dtype="boolean",
            ),
            "is_anomaly": pd.Series(
                pd.NA,
                index=rows.index,
                dtype="boolean",
            ),
        }
    )


def run_xgboost_inference(
    *,
    loaded_model: LoadedSeriesModel,
    series_data: ValidatedSeriesData,
    inference_start: pd.Timestamp,
    inference_end_exclusive: pd.Timestamp,
) -> XGBoostSeriesInferenceResult:
    """
    Run pure XGBoost inference for ONE KPI/site series.

    Only rows in ``[inference_start, inference_end_exclusive)`` become output.
    Earlier rows remain feature/history context only.
    """

    model, decision_settings = _validate_inputs(
        loaded_model=loaded_model,
        series_data=series_data,
        inference_start=inference_start,
        inference_end_exclusive=inference_end_exclusive,
    )

    artifact = loaded_model.artifact
    target_column = artifact.target_column

    if target_column is None:
        raise InputValidationError(
            "XGBoost target column is missing.",
            error_code="MISSING_XGBOOST_INFERENCE_TARGET_COLUMN",
        )

    window_rows, upstream_ready = _select_window(
        series_data=series_data,
        inference_start=inference_start,
        inference_end_exclusive=inference_end_exclusive,
    )

    scorable = _scorable_mask(
        rows=window_rows,
        upstream_ready=upstream_ready,
        feature_columns=artifact.model_feature_columns,
        target_column=target_column,
    )

    output = _build_output(
        artifact=artifact,
        rows=window_rows,
    )

    scoring_rows = (
        window_rows.loc[scorable]
        .copy()
        .reset_index(drop=True)
    )

    anomaly_count = 0
    lower_count = 0
    upper_count = 0

    if not scoring_rows.empty:
        actual, predicted, residual = _predict_residuals(
            model=model,
            rows=scoring_rows,
            feature_columns=artifact.model_feature_columns,
            target_column=target_column,
        )

        anomaly_mask, lower_mask, upper_mask = _apply_saved_thresholds(
            residual=residual,
            decision_settings=decision_settings,
        )

        scoring_indices = output.index[scorable]

        output.loc[scoring_indices, "scorable"] = True
        output.loc[scoring_indices, "actual"] = actual
        output.loc[scoring_indices, "prediction"] = predicted
        output.loc[scoring_indices, "residual"] = residual
        output.loc[scoring_indices, "lower_anomaly"] = lower_mask
        output.loc[scoring_indices, "upper_anomaly"] = upper_mask
        output.loc[scoring_indices, "is_anomaly"] = anomaly_mask

        anomaly_count = int(np.sum(anomaly_mask))
        lower_count = int(np.sum(lower_mask))
        upper_count = int(np.sum(upper_mask))

    total_rows = len(window_rows)
    scorable_rows = int(scorable.sum())
    unscorable_rows = total_rows - scorable_rows

    anomaly_rate = (
        float(anomaly_count / scorable_rows)
        if scorable_rows > 0
        else 0.0
    )

    direction = str(
        decision_settings.threshold_direction
    )

    return XGBoostSeriesInferenceResult(
        kpi_name=artifact.kpi_name,
        series_key=artifact.series_key,
        identity=artifact.identity,
        inference_start=pd.Timestamp(inference_start),
        inference_end_exclusive=pd.Timestamp(
            inference_end_exclusive
        ),
        total_window_rows=total_rows,
        scorable_rows=scorable_rows,
        unscorable_rows=unscorable_rows,
        anomaly_count=anomaly_count,
        anomaly_rate=anomaly_rate,
        lower_anomaly_count=lower_count,
        upper_anomaly_count=upper_count,
        threshold_direction=direction,  # type: ignore[arg-type]
        lower_threshold=decision_settings.lower_threshold,
        upper_threshold=decision_settings.upper_threshold,
        output=output,
    )


__all__ = [
    "XGBoostSeriesInferenceResult",
    "run_xgboost_inference",
]