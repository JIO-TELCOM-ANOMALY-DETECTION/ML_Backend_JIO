"""Generate exact point-based lag features for Part C feature engineering.

Intended project location::

    app/ml/feature_engineering/lag_features.py

Supported lag-source methods:

1. ``raw_value``
   Shift the unchanged Part B KPI value by the requested lag.

2. ``corrected_history``
   Shift the temporary corrected-history Series prepared once by
   ``source_preparation.py``.

3. ``raw_value`` plus local median cleaning
   Shift/reference the raw source and replace an unacceptable referenced point
   with a robust median calculated from accepted raw observations.

Adaptive leakage-safe local median
----------------------------------

For feature row ``t`` and lag ``L``, the referenced timestamp is ``t - L``.
The configured local-median half-window is ``h``. The lag module allows the
full left side and only the right-side observations that remain strictly before
``t``::

    safe_right_points = min(h, L - 1)

The candidate window around the referenced point is therefore::

    [referenced - h, ..., referenced + safe_right_points]

Large lags retain the full centered median whenever it is safe. Small lags use
an asymmetric truncated window. The current row and future rows can never enter
the median.

General rules:

- input represents one site already ordered by timestamp;
- lag values are expressed in exact timeline points;
- output columns are named ``lag_<points>``;
- initial warm-up rows remain ``NaN``;
- source row count and index are preserved;
- the original Part B dataframe and source Series are never modified; and
- ``corrected_history`` cannot be combined with local median cleaning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.schemas.feature_engineering import LagFeatureRequest

from .source_preparation import (
    LocalMedianCandidateResult,
    PreparedFeatureSources,
    prepare_local_median_candidates,
    prepare_referenced_values,
)


def _resolve_safe_right_window_points(
    *,
    lag: int,
    half_window_points: int,
) -> int:
    """Return the largest right-side window that remains before feature row t."""

    if isinstance(lag, bool) or not isinstance(lag, int):
        raise TypeError("lag must be an integer number of points.")

    if lag <= 0:
        raise ValueError("lag must be greater than zero.")

    if isinstance(half_window_points, bool) or not isinstance(
        half_window_points,
        int,
    ):
        raise TypeError("half_window_points must be an integer.")

    if half_window_points <= 0:
        raise ValueError("half_window_points must be greater than zero.")

    return min(half_window_points, lag - 1)


def _prepare_adaptive_local_median_candidates(
    *,
    sources: PreparedFeatureSources,
    request: LagFeatureRequest,
    bucket_minutes: int,
    lag: int,
    half_window_points: int,
) -> LocalMedianCandidateResult:
    """Prepare one leakage-safe local-median candidate Series for a lag."""

    cleaning = request.point_cleaning

    if not cleaning.enabled:
        raise ValueError(
            "Adaptive local-median candidates cannot be prepared when point "
            "cleaning is disabled."
        )

    if request.source != "raw_value":
        raise ValueError(
            "Local median cleaning is supported only with source='raw_value'."
        )

    if sources.statuses is None:
        raise ValueError(
            "A status Series is required when lag local-median cleaning is "
            "enabled."
        )

    safe_right_points = _resolve_safe_right_window_points(
        lag=lag,
        half_window_points=half_window_points,
    )
    available_window_points = (
        half_window_points
        + safe_right_points
        + 1
    )

    if cleaning.minimum_valid_points > available_window_points:
        raise ValueError(
            "point_cleaning.minimum_valid_points cannot be satisfied for "
            f"lag={lag}. The leakage-safe local window contains at most "
            f"{available_window_points} positions: {half_window_points} on "
            f"the left, the referenced point, and {safe_right_points} on the "
            "right."
        )

    candidates = prepare_local_median_candidates(
        raw_values=sources.bundle.raw_value,
        statuses=sources.statuses,
        config=cleaning,
        half_window_points=half_window_points,
        right_window_points=safe_right_points,
    )

    if candidates.resolved_right_window_points != safe_right_points:
        raise RuntimeError(
            "Source preparation returned local-median candidates with an "
            "unexpected right-side window."
        )

    return candidates


def _validate_lag_output(
    output: pd.DataFrame,
    *,
    expected_index: pd.Index,
    expected_columns: tuple[str, ...],
) -> None:
    """Defensively verify the lag-feature output contract."""

    if len(output) != len(expected_index):
        raise RuntimeError(
            "Lag-feature generation unexpectedly changed the row count."
        )

    if not output.index.equals(expected_index):
        raise RuntimeError(
            "Lag-feature generation unexpectedly changed the source index."
        )

    if tuple(output.columns) != expected_columns:
        raise RuntimeError(
            "Lag-feature generation produced unexpected columns. Expected "
            f"{list(expected_columns)}, received {output.columns.tolist()}."
        )

    if output.columns.has_duplicates:
        duplicates = output.columns[
            output.columns.duplicated()
        ].tolist()
        raise RuntimeError(
            f"Lag-feature generation produced duplicate columns: {duplicates}"
        )

    numeric = output.apply(
        pd.to_numeric,
        errors="coerce",
    )
    finite_values = numeric.to_numpy(dtype="float64")
    finite_mask = ~np.isnan(finite_values)

    if finite_mask.any() and not np.isfinite(finite_values[finite_mask]).all():
        raise RuntimeError(
            "Lag-feature generation produced positive or negative infinity."
        )


def generate_lag_features(
    request: LagFeatureRequest,
    *,
    sources: PreparedFeatureSources,
    bucket_minutes: int,
) -> pd.DataFrame:
    """Generate requested exact point-based lag columns for one valid site.

    Formula::

        lag_n(t) = selected_source(t - n)

    When ``point_cleaning.enabled`` is true, only an unacceptable referenced
    raw point is replaced. The lag determines the largest safe right-side
    portion of the configured local-median window; source preparation performs
    the actual median calculation.
    """

    if not isinstance(request, LagFeatureRequest):
        raise TypeError(
            "generate_lag_features requires LagFeatureRequest; received "
            f"{type(request).__name__}."
        )

    if not isinstance(sources, PreparedFeatureSources):
        raise TypeError(
            "sources must be an instance of PreparedFeatureSources; received "
            f"{type(sources).__name__}."
        )

    if isinstance(bucket_minutes, bool) or not isinstance(bucket_minutes, int):
        raise TypeError("bucket_minutes must be an integer number of minutes.")

    if bucket_minutes <= 0:
        raise ValueError("bucket_minutes must be greater than zero.")

    source_index = sources.bundle.raw_value.index

    if len(source_index) == 0:
        raise ValueError(
            "Cannot generate lag features from an empty prepared site source."
        )

    half_window_points: int | None = None
    candidate_cache: dict[int, LocalMedianCandidateResult] = {}

    if request.point_cleaning.enabled:
        if request.source != "raw_value":
            raise ValueError(
                "Local median cleaning is supported only with "
                "source='raw_value'."
            )

        half_window_points = request.point_cleaning.resolve_half_window_points(
            bucket_minutes=bucket_minutes,
        )

    output = pd.DataFrame(index=source_index)

    for lag in request.lags:
        local_candidates: LocalMedianCandidateResult | None = None

        if half_window_points is not None:
            safe_right_points = _resolve_safe_right_window_points(
                lag=lag,
                half_window_points=half_window_points,
            )

            local_candidates = candidate_cache.get(safe_right_points)

            if local_candidates is None:
                local_candidates = _prepare_adaptive_local_median_candidates(
                    sources=sources,
                    request=request,
                    bucket_minutes=bucket_minutes,
                    lag=lag,
                    half_window_points=half_window_points,
                )
                candidate_cache[safe_right_points] = local_candidates

        referenced = prepare_referenced_values(
            sources=sources.bundle,
            statuses=sources.statuses,
            source_name=request.source,
            offset_points=lag,
            local_cleaning=request.point_cleaning,
            bucket_minutes=bucket_minutes,
            local_candidates=local_candidates,
        )

        column_name = f"lag_{lag}"
        lag_values = referenced.values.astype("float64", copy=True)
        lag_values.name = column_name

        warm_up_rows = min(lag, len(lag_values))
        if warm_up_rows and not lag_values.iloc[:warm_up_rows].isna().all():
            raise RuntimeError(
                f"Lag feature '{column_name}' populated one or more warm-up "
                "rows that should be NaN."
            )

        output[column_name] = lag_values

    expected_columns = tuple(
        f"lag_{lag}"
        for lag in request.lags
    )
    _validate_lag_output(
        output,
        expected_index=source_index,
        expected_columns=expected_columns,
    )

    return output


__all__ = [
    "generate_lag_features",
]