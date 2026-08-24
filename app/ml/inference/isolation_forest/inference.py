"""
Pure Isolation Forest inference for ONE saved KPI/site series.

Intended project location::

    app/ml/inference/isolation_forest/inference.py

This mirrors the finalized Part D Isolation Forest testing semantics:

1. model anomaly when ``predict(X) == -1``;
2. lower_only keeps model anomalies where direction_feature < 0;
3. upper_only keeps model anomalies where direction_feature > 0;
4. both keeps every model anomaly.

For ``both``, the direction feature is not required solely for filtering. If
it is already part of the saved model feature set, lower/upper subtype counts
are also reported.

``decision_function`` and ``score_samples`` are returned for row-level
inspection, but the model-anomaly rule remains ``predict == -1``.

The parent runner handles one model at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.core.exceptions import InputValidationError
from app.ml.inference.loader import LoadedSeriesModel
from app.ml.inference.validation import ValidatedSeriesData
from app.schemas.inference.isolation_forest import (
    IsolationForestDecisionSettings,
)


IsolationForestThresholdDirection = Literal[
    "lower_only",
    "upper_only",
    "both",
]


@dataclass(frozen=True, slots=True)
class IsolationForestSeriesInferenceResult:
    """Pure inference result for one saved Isolation Forest KPI/site model."""

    kpi_name: str
    series_key: str
    identity: tuple[str, str, str]

    inference_start: pd.Timestamp
    inference_end_exclusive: pd.Timestamp

    total_window_rows: int
    scorable_rows: int
    unscorable_rows: int

    model_anomaly_count: int
    model_anomaly_rate: float

    anomaly_count: int
    anomaly_rate: float

    lower_anomaly_count: int
    upper_anomaly_count: int
    direction_counts_available: bool

    threshold_direction: IsolationForestThresholdDirection
    direction_feature: str
    model_offset: float
    contamination: float | Literal["auto"]

    output: pd.DataFrame


def _validate_inputs(
    *,
    loaded_model: LoadedSeriesModel,
    series_data: ValidatedSeriesData,
    inference_start: pd.Timestamp,
    inference_end_exclusive: pd.Timestamp,
) -> tuple[IsolationForest, IsolationForestDecisionSettings]:
    artifact = loaded_model.artifact
    data_artifact = series_data.series_plan.artifact

    if artifact.model_type != "isolation_forest":
        raise InputValidationError(
            "Isolation Forest inference requires an Isolation Forest artifact.",
            error_code="INVALID_ISOLATION_FOREST_INFERENCE_MODEL_TYPE",
        )

    if data_artifact.model_type != "isolation_forest":
        raise InputValidationError(
            "Isolation Forest inference requires Isolation-Forest-validated "
            "Part C data.",
            error_code="INVALID_ISOLATION_FOREST_INFERENCE_DATA_TYPE",
        )

    if artifact != data_artifact:
        raise InputValidationError(
            "Loaded model and validated Part C rows reference different saved "
            "Isolation Forest artifacts.",
            error_code="ISOLATION_FOREST_INFERENCE_ARTIFACT_MISMATCH",
        )

    if artifact.target_column is not None or series_data.target_column is not None:
        raise InputValidationError(
            "Isolation Forest inference must not receive a y target.",
            error_code="ISOLATION_FOREST_INFERENCE_TARGET_NOT_ALLOWED",
        )

    if not artifact.model_feature_columns:
        raise InputValidationError(
            "Saved Isolation Forest artifact contains no model features.",
            error_code="EMPTY_ISOLATION_FOREST_INFERENCE_FEATURE_SET",
        )

    if (
        tuple(series_data.model_feature_columns)
        != tuple(artifact.model_feature_columns)
    ):
        raise InputValidationError(
            "Validated Part C feature order does not match the saved "
            "Isolation Forest feature order.",
            error_code="ISOLATION_FOREST_INFERENCE_FEATURE_ORDER_MISMATCH",
        )

    if not isinstance(
        artifact.decision_settings,
        IsolationForestDecisionSettings,
    ):
        raise InputValidationError(
            "Saved artifact does not contain Isolation Forest decision "
            "settings.",
            error_code="INVALID_ISOLATION_FOREST_DECISION_SETTINGS",
        )

    decision_settings = artifact.decision_settings

    if decision_settings.method != "isolation_forest":
        raise InputValidationError(
            "Isolation Forest inference requires isolation_forest decision "
            "settings.",
            error_code="INVALID_ISOLATION_FOREST_DECISION_METHOD",
        )

    if decision_settings.model_prediction_rule != "predict == -1":
        raise InputValidationError(
            "Isolation Forest inference requires model_prediction_rule="
            "'predict == -1'.",
            error_code="INVALID_ISOLATION_FOREST_PREDICTION_RULE",
        )

    if (
        decision_settings.decision_function_rule
        != "decision_function < 0 indicates model anomaly"
    ):
        raise InputValidationError(
            "Isolation Forest saved decision_function rule is unsupported.",
            error_code="INVALID_ISOLATION_FOREST_DECISION_FUNCTION_RULE",
        )

    direction = str(decision_settings.threshold_direction)

    if direction not in {"lower_only", "upper_only", "both"}:
        raise InputValidationError(
            "Unsupported saved Isolation Forest threshold_direction.",
            error_code="UNSUPPORTED_ISOLATION_FOREST_THRESHOLD_DIRECTION",
        )

    if direction in {"lower_only", "upper_only"}:
        if (
            decision_settings.direction_feature
            not in artifact.model_feature_columns
        ):
            raise InputValidationError(
                "One-sided Isolation Forest inference requires its saved "
                "direction feature in model_feature_columns.",
                error_code="MISSING_ISOLATION_FOREST_DIRECTION_FEATURE",
            )

        if series_data.direction_feature != decision_settings.direction_feature:
            raise InputValidationError(
                "Validated Part C direction feature does not match the saved "
                "Isolation Forest decision contract.",
                error_code="ISOLATION_FOREST_DIRECTION_FEATURE_MISMATCH",
            )

    model = loaded_model.estimator

    for method_name in (
        "predict",
        "decision_function",
        "score_samples",
    ):
        if not callable(getattr(model, method_name, None)):
            raise InputValidationError(
                "Loaded Isolation Forest estimator is missing a required "
                "inference method.",
                error_code="INVALID_LOADED_ISOLATION_FOREST_ESTIMATOR",
                details={"missing_method": method_name},
            )

    start = pd.Timestamp(inference_start)
    end = pd.Timestamp(inference_end_exclusive)

    if end <= start:
        raise InputValidationError(
            "inference_end_exclusive must be later than inference_start.",
            error_code="INVALID_ISOLATION_FOREST_INFERENCE_WINDOW",
        )

    return model, decision_settings


def _timestamp_column(
    dataframe: pd.DataFrame,
) -> str:
    candidates = [
        column
        for column in dataframe.columns
        if pd.api.types.is_datetime64_any_dtype(dataframe[column])
    ]

    if len(candidates) != 1:
        raise InputValidationError(
            "Validated Isolation Forest site data must contain exactly one "
            "parsed timestamp column.",
            error_code="ISOLATION_FOREST_TIMESTAMP_COLUMN_AMBIGUOUS",
        )

    return candidates[0]


def _select_window(
    *,
    series_data: ValidatedSeriesData,
    inference_start: pd.Timestamp,
    inference_end_exclusive: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Series]:
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
            "No Isolation Forest Part C rows exist inside the requested "
            "inference window.",
            error_code="EMPTY_ISOLATION_FOREST_INFERENCE_WINDOW",
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
) -> pd.Series:
    feature_values = rows.loc[
        :,
        list(feature_columns),
    ].to_numpy(dtype=np.float64)

    finite_features = pd.Series(
        np.isfinite(feature_values).all(axis=1),
        index=rows.index,
        dtype=bool,
    )

    return upstream_ready & finite_features


def _run_model(
    *,
    model: IsolationForest,
    rows: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = rows.loc[
        :,
        list(feature_columns),
    ]

    predicted_labels = np.asarray(
        model.predict(x),
        dtype=np.int64,
    )

    decision_scores = np.asarray(
        model.decision_function(x),
        dtype=np.float64,
    )

    raw_scores = np.asarray(
        model.score_samples(x),
        dtype=np.float64,
    )

    if not (
        predicted_labels.shape
        == decision_scores.shape
        == raw_scores.shape
    ):
        raise InputValidationError(
            "Isolation Forest inference prediction/score shapes differ.",
            error_code="ISOLATION_FOREST_INFERENCE_SHAPE_MISMATCH",
        )

    if not bool(np.isfinite(decision_scores).all()):
        raise InputValidationError(
            "Isolation Forest produced non-finite decision scores.",
            error_code="INVALID_ISOLATION_FOREST_DECISION_SCORES",
        )

    if not bool(np.isfinite(raw_scores).all()):
        raise InputValidationError(
            "Isolation Forest produced non-finite score_samples values.",
            error_code="INVALID_ISOLATION_FOREST_RAW_SCORES",
        )

    allowed = np.isin(predicted_labels, [-1, 1])

    if not bool(allowed.all()):
        raise InputValidationError(
            "Isolation Forest predict() returned labels outside {-1, 1}.",
            error_code="INVALID_ISOLATION_FOREST_PREDICTED_LABELS",
        )

    return predicted_labels, decision_scores, raw_scores


def _apply_direction(
    *,
    model_anomaly_mask: np.ndarray,
    rows: pd.DataFrame,
    decision_settings: IsolationForestDecisionSettings,
    direction_feature_available: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    direction = str(decision_settings.threshold_direction)
    direction_feature = decision_settings.direction_feature

    lower_mask = np.zeros(
        model_anomaly_mask.shape,
        dtype=bool,
    )
    upper_mask = np.zeros(
        model_anomaly_mask.shape,
        dtype=bool,
    )

    if direction in {"lower_only", "upper_only"}:
        if not direction_feature_available:
            raise InputValidationError(
                "One-sided Isolation Forest inference requires its saved "
                "direction feature.",
                error_code="MISSING_ISOLATION_FOREST_DIRECTION_FEATURE",
            )

        values = rows[
            direction_feature
        ].to_numpy(dtype=np.float64)

        lower_mask = (
            model_anomaly_mask
            & (values < 0.0)
        )

        upper_mask = (
            model_anomaly_mask
            & (values > 0.0)
        )

        final_mask = (
            lower_mask
            if direction == "lower_only"
            else upper_mask
        )

        return (
            final_mask,
            lower_mask,
            upper_mask,
            True,
        )

    if direction == "both":
        if direction_feature_available:
            values = rows[
                direction_feature
            ].to_numpy(dtype=np.float64)

            lower_mask = (
                model_anomaly_mask
                & (values < 0.0)
            )

            upper_mask = (
                model_anomaly_mask
                & (values > 0.0)
            )

        return (
            model_anomaly_mask.copy(),
            lower_mask,
            upper_mask,
            direction_feature_available,
        )

    raise InputValidationError(
        "Unsupported saved Isolation Forest threshold_direction.",
        error_code="UNSUPPORTED_ISOLATION_FOREST_THRESHOLD_DIRECTION",
    )


def _build_output(
    *,
    artifact,
    rows: pd.DataFrame,
) -> pd.DataFrame:
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
            "model_prediction": pd.Series(
                pd.NA,
                index=rows.index,
                dtype="Int64",
            ),
            "decision_score": np.nan,
            "score_samples": np.nan,
            "model_anomaly": pd.Series(
                pd.NA,
                index=rows.index,
                dtype="boolean",
            ),
            "direction_value": np.nan,
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


def run_isolation_forest_inference(
    *,
    loaded_model: LoadedSeriesModel,
    series_data: ValidatedSeriesData,
    inference_start: pd.Timestamp,
    inference_end_exclusive: pd.Timestamp,
) -> IsolationForestSeriesInferenceResult:
    model, decision_settings = _validate_inputs(
        loaded_model=loaded_model,
        series_data=series_data,
        inference_start=inference_start,
        inference_end_exclusive=inference_end_exclusive,
    )

    artifact = loaded_model.artifact

    window_rows, upstream_ready = _select_window(
        series_data=series_data,
        inference_start=inference_start,
        inference_end_exclusive=inference_end_exclusive,
    )

    scorable = _scorable_mask(
        rows=window_rows,
        upstream_ready=upstream_ready,
        feature_columns=artifact.model_feature_columns,
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

    direction_feature = decision_settings.direction_feature

    direction_feature_available = (
        direction_feature
        in artifact.model_feature_columns
        and direction_feature
        in scoring_rows.columns
    )

    model_anomaly_count = 0
    anomaly_count = 0
    lower_count = 0
    upper_count = 0
    direction_counts_available = direction_feature_available

    if not scoring_rows.empty:
        (
            predicted_labels,
            decision_scores,
            raw_scores,
        ) = _run_model(
            model=model,
            rows=scoring_rows,
            feature_columns=artifact.model_feature_columns,
        )

        model_anomaly_mask = (
            predicted_labels == -1
        )

        (
            final_anomaly_mask,
            lower_mask,
            upper_mask,
            direction_counts_available,
        ) = _apply_direction(
            model_anomaly_mask=model_anomaly_mask,
            rows=scoring_rows,
            decision_settings=decision_settings,
            direction_feature_available=direction_feature_available,
        )

        scoring_indices = output.index[scorable]

        output.loc[scoring_indices, "scorable"] = True
        output.loc[scoring_indices, "model_prediction"] = predicted_labels
        output.loc[scoring_indices, "decision_score"] = decision_scores
        output.loc[scoring_indices, "score_samples"] = raw_scores
        output.loc[scoring_indices, "model_anomaly"] = model_anomaly_mask

        if direction_feature_available:
            output.loc[
                scoring_indices,
                "direction_value",
            ] = scoring_rows[
                direction_feature
            ].to_numpy(dtype=np.float64)

        output.loc[scoring_indices, "lower_anomaly"] = lower_mask
        output.loc[scoring_indices, "upper_anomaly"] = upper_mask
        output.loc[scoring_indices, "is_anomaly"] = final_anomaly_mask

        model_anomaly_count = int(
            np.sum(model_anomaly_mask)
        )
        anomaly_count = int(
            np.sum(final_anomaly_mask)
        )
        lower_count = int(
            np.sum(lower_mask)
        )
        upper_count = int(
            np.sum(upper_mask)
        )

    total_rows = len(window_rows)
    scorable_rows = int(scorable.sum())
    unscorable_rows = total_rows - scorable_rows

    model_anomaly_rate = (
        float(model_anomaly_count / scorable_rows)
        if scorable_rows > 0
        else 0.0
    )

    anomaly_rate = (
        float(anomaly_count / scorable_rows)
        if scorable_rows > 0
        else 0.0
    )

    return IsolationForestSeriesInferenceResult(
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
        model_anomaly_count=model_anomaly_count,
        model_anomaly_rate=model_anomaly_rate,
        anomaly_count=anomaly_count,
        anomaly_rate=anomaly_rate,
        lower_anomaly_count=lower_count,
        upper_anomaly_count=upper_count,
        direction_counts_available=direction_counts_available,
        threshold_direction=str(
            decision_settings.threshold_direction
        ),  # type: ignore[arg-type]
        direction_feature=direction_feature,
        model_offset=float(
            getattr(
                model,
                "offset_",
                decision_settings.model_offset,
            )
        ),
        contamination=decision_settings.contamination,
        output=output,
    )


__all__ = [
    "IsolationForestSeriesInferenceResult",
    "run_isolation_forest_inference",
]