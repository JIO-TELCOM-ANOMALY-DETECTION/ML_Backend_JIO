"""Generate rolling-history features for Part C feature engineering.

Intended project location::

    app/ml/feature_engineering/rolling_features.py

Supported request families
--------------------------

1. ``rolling``
   Generate one or more historical rolling statistics:

   - mean
   - median
   - median absolute deviation (MAD)
   - standard deviation
   - minimum
   - maximum
   - range
   - interquartile range (IQR)

2. ``rolling_zscore``
   Standardize the selected current value against the mean and standard
   deviation of earlier observations only::

       rolling_zscore_window(t) = (
           source(t) - previous_mean_window(t)
       ) / previous_std_window(t)

3. ``ewma``
   Generate an exponentially weighted moving average from earlier observations
   only. ``span`` controls the recursive decay; it is not treated as a strict
   fixed-size window. The first row is therefore ``NaN`` and calculation begins
   once at least one earlier numeric observation is available.

4. ``rolling_slope``
   Fit an ordinary least-squares line to earlier observations in each rolling
   window and return its slope in KPI units per timeline point.

Past-only rule
--------------

Every rolling-history calculation starts from::

    historical_source = selected_source.shift(1)

Consequently, the current row never enters its own rolling statistic, EWMA, or
slope. ``rolling_zscore`` uses the unshifted selected source only as the current
numerator value; its mean and standard deviation are still strictly past-only.

General rules
-------------

- input represents one valid site already ordered by timestamp;
- ``raw_value`` and ``corrected_history`` are selected from
  ``PreparedFeatureSources``;
- windows and spans are expressed in exact timeline points;
- schema-defined output names and request order are preserved;
- initial unavailable-history rows remain ``NaN``;
- zero standard deviation and infinite results become ``NaN``;
- source row count and index are preserved; and
- prepared source Series and the original Part B dataframe are never modified.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pandas as pd

from app.schemas.feature_engineering import (
    EWMAFeatureRequest,
    RollingFeatureRequest,
    RollingSlopeFeatureRequest,
    RollingZScoreFeatureRequest,
)

from .source_preparation import PreparedFeatureSources


RollingRequest: TypeAlias = (
    RollingFeatureRequest
    | RollingZScoreFeatureRequest
    | EWMAFeatureRequest
    | RollingSlopeFeatureRequest
)


def _select_numeric_source(
    *,
    request: RollingRequest,
    sources: PreparedFeatureSources,
) -> pd.Series:
    """Return one copied numeric source aligned with the prepared site.

    Source preparation owns raw/corrected-history construction. This module
    only selects the source explicitly requested by the schema and verifies
    that it contains finite numeric values or ``NaN``.
    """

    if not isinstance(sources, PreparedFeatureSources):
        raise TypeError(
            "sources must be an instance of PreparedFeatureSources; received "
            f"{type(sources).__name__}."
        )

    selected = pd.to_numeric(
        sources.bundle.select(request.source),
        errors="coerce",
    ).astype("float64")

    if not selected.index.equals(sources.bundle.raw_value.index):
        raise ValueError(
            "Selected rolling source is not aligned with raw_value."
        )

    finite_values = selected.dropna().to_numpy(dtype="float64")

    if finite_values.size and not np.isfinite(finite_values).all():
        raise ValueError(
            f"Selected rolling source {request.source!r} contains positive or "
            "negative infinity."
        )

    return selected.copy()


def _resolve_minimum_periods(
    *,
    window: int,
    configured_minimum: int | None,
    minimum_allowed: int = 1,
) -> int:
    """Resolve a request's optional minimum-period setting for one window."""

    if isinstance(window, bool) or not isinstance(window, int):
        raise TypeError("Rolling window must be an integer number of points.")

    if window <= 0:
        raise ValueError("Rolling window must be greater than zero.")

    minimum_periods = (
        window
        if configured_minimum is None
        else configured_minimum
    )

    if isinstance(minimum_periods, bool) or not isinstance(
        minimum_periods,
        int,
    ):
        raise TypeError("minimum_periods must be an integer or None.")

    if minimum_periods < minimum_allowed:
        raise ValueError(
            f"minimum_periods must be at least {minimum_allowed}."
        )

    if minimum_periods > window:
        raise ValueError(
            "minimum_periods cannot exceed the rolling window."
        )

    return minimum_periods


def _median_absolute_deviation(values: np.ndarray) -> float:
    """Return the median absolute deviation of finite values in one window."""

    numeric = np.asarray(values, dtype="float64")
    finite = numeric[np.isfinite(numeric)]

    if finite.size == 0:
        return np.nan

    median = float(np.median(finite))
    return float(np.median(np.abs(finite - median)))


def _ordinary_least_squares_slope(values: np.ndarray) -> float:
    """Return OLS slope against original positions in one rolling window.

    Missing values are ignored, but their positions are not compressed. This
    preserves the meaning of slope as KPI change per timeline point.
    """

    numeric = np.asarray(values, dtype="float64")
    positions = np.arange(numeric.size, dtype="float64")
    finite_mask = np.isfinite(numeric)

    if int(finite_mask.sum()) < 2:
        return np.nan

    x = positions[finite_mask]
    y = numeric[finite_mask]

    x_centered = x - x.mean()
    denominator = float(np.dot(x_centered, x_centered))

    if denominator == 0.0:
        return np.nan

    y_centered = y - y.mean()
    numerator = float(np.dot(x_centered, y_centered))

    return numerator / denominator


def _generate_one_rolling_statistic(
    *,
    historical_source: pd.Series,
    statistic: str,
    window: int,
    minimum_periods: int,
    std_ddof: int,
) -> pd.Series:
    """Generate one past-only rolling statistic for one window."""

    rolling = historical_source.rolling(
        window=window,
        min_periods=minimum_periods,
    )

    if statistic == "mean":
        result = rolling.mean()

    elif statistic == "median":
        result = rolling.median()

    elif statistic == "mad":
        result = rolling.apply(
            _median_absolute_deviation,
            raw=True,
        )

    elif statistic == "std":
        result = rolling.std(ddof=std_ddof)

    elif statistic == "min":
        result = rolling.min()

    elif statistic == "max":
        result = rolling.max()

    elif statistic == "range":
        result = rolling.max() - rolling.min()

    elif statistic == "iqr":
        result = rolling.quantile(0.75) - rolling.quantile(0.25)

    else:
        raise ValueError(
            f"Unsupported rolling statistic: {statistic!r}"
        )

    return result.astype("float64")


def generate_rolling_statistical_features(
    request: RollingFeatureRequest,
    *,
    sources: PreparedFeatureSources,
) -> pd.DataFrame:
    """Generate requested past-only rolling statistical columns.

    Output order follows the schema contract: windows first, then statistics in
    request order. With ``minimum_periods=None``, each complete rolling window
    is required.
    """

    if not isinstance(request, RollingFeatureRequest):
        raise TypeError(
            "generate_rolling_statistical_features requires "
            "RollingFeatureRequest; received "
            f"{type(request).__name__}."
        )

    source = _select_numeric_source(
        request=request,
        sources=sources,
    )
    historical_source = source.shift(1)
    output = pd.DataFrame(index=source.index)

    for window in request.windows:
        minimum_periods = _resolve_minimum_periods(
            window=window,
            configured_minimum=request.minimum_periods,
        )

        for statistic in request.statistics:
            column_name = f"rolling_{statistic}_{window}"
            output[column_name] = _generate_one_rolling_statistic(
                historical_source=historical_source,
                statistic=statistic,
                window=window,
                minimum_periods=minimum_periods,
                std_ddof=request.std_ddof,
            )

    return output


def generate_rolling_zscore_features(
    request: RollingZScoreFeatureRequest,
    *,
    sources: PreparedFeatureSources,
) -> pd.DataFrame:
    """Generate current-versus-past rolling z-score features.

    The current value comes from the selected unshifted source. Rolling mean
    and standard deviation use only earlier rows. A zero standard deviation
    produces ``NaN`` according to the current schema.
    """

    if not isinstance(request, RollingZScoreFeatureRequest):
        raise TypeError(
            "generate_rolling_zscore_features requires "
            "RollingZScoreFeatureRequest; received "
            f"{type(request).__name__}."
        )

    if request.zero_std_strategy != "nan":
        raise ValueError(
            "Unsupported rolling-zscore zero_std_strategy: "
            f"{request.zero_std_strategy!r}"
        )

    source = _select_numeric_source(
        request=request,
        sources=sources,
    )
    historical_source = source.shift(1)
    output = pd.DataFrame(index=source.index)

    for window in request.windows:
        minimum_periods = _resolve_minimum_periods(
            window=window,
            configured_minimum=request.minimum_periods,
        )
        rolling = historical_source.rolling(
            window=window,
            min_periods=minimum_periods,
        )
        previous_mean = rolling.mean()
        previous_std = rolling.std(ddof=request.std_ddof)
        safe_std = previous_std.mask(previous_std.eq(0.0))

        values = (source - previous_mean) / safe_std
        values = values.replace(
            [np.inf, -np.inf],
            np.nan,
        ).astype("float64")

        output[f"rolling_zscore_{window}"] = values

    return output


def generate_ewma_features(
    request: EWMAFeatureRequest,
    *,
    sources: PreparedFeatureSources,
) -> pd.DataFrame:
    """Generate recursive past-only exponentially weighted moving averages.

    ``span`` determines exponential decay rather than a strict observation
    count. Because the schema exposes no EWMA ``minimum_periods`` field, this
    implementation starts once one earlier numeric observation exists. The
    initial row remains ``NaN`` because the source is shifted first.
    """

    if not isinstance(request, EWMAFeatureRequest):
        raise TypeError(
            "generate_ewma_features requires EWMAFeatureRequest; received "
            f"{type(request).__name__}."
        )

    if request.adjust is not False:
        raise ValueError(
            "Only recursive EWMA with adjust=False is supported."
        )

    source = _select_numeric_source(
        request=request,
        sources=sources,
    )
    historical_source = source.shift(1)
    output = pd.DataFrame(index=source.index)

    for span in request.spans:
        if isinstance(span, bool) or not isinstance(span, int):
            raise TypeError("EWMA span must be an integer number of points.")

        if span <= 0:
            raise ValueError("EWMA span must be greater than zero.")

        values = historical_source.ewm(
            span=span,
            adjust=False,
            min_periods=1,
        ).mean()

        output[f"ewma_{span}"] = values.astype("float64")

    return output


def generate_rolling_slope_features(
    request: RollingSlopeFeatureRequest,
    *,
    sources: PreparedFeatureSources,
) -> pd.DataFrame:
    """Generate past-only OLS slope features for requested windows."""

    if not isinstance(request, RollingSlopeFeatureRequest):
        raise TypeError(
            "generate_rolling_slope_features requires "
            "RollingSlopeFeatureRequest; received "
            f"{type(request).__name__}."
        )

    source = _select_numeric_source(
        request=request,
        sources=sources,
    )
    historical_source = source.shift(1)
    output = pd.DataFrame(index=source.index)

    for window in request.windows:
        minimum_periods = _resolve_minimum_periods(
            window=window,
            configured_minimum=request.minimum_periods,
            minimum_allowed=2,
        )

        values = historical_source.rolling(
            window=window,
            min_periods=minimum_periods,
        ).apply(
            _ordinary_least_squares_slope,
            raw=True,
        )

        output[f"rolling_slope_{window}"] = values.astype("float64")

    return output


def _expected_columns(
    request: RollingRequest,
) -> tuple[str, ...]:
    """Return the deterministic schema-defined output names for a request."""

    if isinstance(request, RollingFeatureRequest):
        return tuple(
            f"rolling_{statistic}_{window}"
            for window in request.windows
            for statistic in request.statistics
        )

    if isinstance(request, RollingZScoreFeatureRequest):
        return tuple(
            f"rolling_zscore_{window}"
            for window in request.windows
        )

    if isinstance(request, EWMAFeatureRequest):
        return tuple(
            f"ewma_{span}"
            for span in request.spans
        )

    if isinstance(request, RollingSlopeFeatureRequest):
        return tuple(
            f"rolling_slope_{window}"
            for window in request.windows
        )

    raise TypeError(
        f"Unsupported rolling request: {type(request).__name__}."
    )


def _validate_rolling_output(
    output: pd.DataFrame,
    *,
    expected_index: pd.Index,
    expected_columns: tuple[str, ...],
) -> None:
    """Defensively verify row, index, column, and numeric output contracts."""

    if len(output) != len(expected_index):
        raise RuntimeError(
            "Rolling-feature generation unexpectedly changed the row count."
        )

    if not output.index.equals(expected_index):
        raise RuntimeError(
            "Rolling-feature generation unexpectedly changed the source index."
        )

    if tuple(output.columns) != expected_columns:
        raise RuntimeError(
            "Rolling-feature generation produced unexpected columns. Expected "
            f"{list(expected_columns)}, received {output.columns.tolist()}."
        )

    if output.columns.has_duplicates:
        duplicates = output.columns[
            output.columns.duplicated()
        ].tolist()
        raise RuntimeError(
            "Rolling-feature generation produced duplicate columns: "
            f"{duplicates}"
        )

    numeric = output.apply(
        pd.to_numeric,
        errors="coerce",
    )
    finite_values = numeric.to_numpy(dtype="float64")
    finite_mask = ~np.isnan(finite_values)

    if finite_mask.any() and not np.isfinite(finite_values[finite_mask]).all():
        raise RuntimeError(
            "Rolling-feature generation produced positive or negative "
            "infinity."
        )


def generate_rolling_features(
    request: RollingRequest,
    *,
    sources: PreparedFeatureSources,
) -> pd.DataFrame:
    """Dispatch one supported rolling request for a prepared valid site.

    Args:
        request:
            One validated rolling, rolling-zscore, EWMA, or rolling-slope
            request from the feature-engineering schema.
        sources:
            Prepared raw and optional corrected-history sources for one site.

    Returns:
        pd.DataFrame:
            Requested rolling features aligned with the prepared source index.
    """

    if not isinstance(
        request,
        (
            RollingFeatureRequest,
            RollingZScoreFeatureRequest,
            EWMAFeatureRequest,
            RollingSlopeFeatureRequest,
        ),
    ):
        raise TypeError(
            "rolling_features.py supports only RollingFeatureRequest, "
            "RollingZScoreFeatureRequest, EWMAFeatureRequest, and "
            "RollingSlopeFeatureRequest; received "
            f"{type(request).__name__}."
        )

    if not isinstance(sources, PreparedFeatureSources):
        raise TypeError(
            "sources must be an instance of PreparedFeatureSources; received "
            f"{type(sources).__name__}."
        )

    if len(sources.bundle.raw_value) == 0:
        raise ValueError(
            "Cannot generate rolling features from an empty prepared site "
            "source."
        )

    if isinstance(request, RollingFeatureRequest):
        output = generate_rolling_statistical_features(
            request,
            sources=sources,
        )

    elif isinstance(request, RollingZScoreFeatureRequest):
        output = generate_rolling_zscore_features(
            request,
            sources=sources,
        )

    elif isinstance(request, EWMAFeatureRequest):
        output = generate_ewma_features(
            request,
            sources=sources,
        )

    else:
        output = generate_rolling_slope_features(
            request,
            sources=sources,
        )

    _validate_rolling_output(
        output,
        expected_index=sources.bundle.raw_value.index,
        expected_columns=_expected_columns(request),
    )

    return output


__all__ = [
    "RollingRequest",
    "generate_ewma_features",
    "generate_rolling_features",
    "generate_rolling_slope_features",
    "generate_rolling_statistical_features",
    "generate_rolling_zscore_features",
]