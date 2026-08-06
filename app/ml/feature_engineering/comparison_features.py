"""Generate lag-based comparison features for Part C feature engineering.

Intended project location::

    app/ml/feature_engineering/comparison_features.py

Supported feature requests:

1. ``difference``
   Calculate the absolute change between the selected current source and an
   exact earlier reference::

       difference_lag(t) = source(t) - source(t - lag)

2. ``percentage_change``
   Calculate the relative change against an exact earlier reference::

       percentage_change_lag(t) = (
           source(t) - source(t - lag)
       ) / source(t - lag)

   The result is a decimal ratio rather than a value multiplied by 100. A
   denominator of zero produces ``NaN`` according to the current schema.

3. ``seasonal_difference``
   Calculate the absolute difference from an intentionally seasonal lag such
   as one day or one week::

       seasonal_difference_lag(t) = source(t) - source(t - lag)

``difference`` and ``seasonal_difference`` use the same mathematical formula.
They remain separate request types so configuration, model reporting, and UI
labels can distinguish general short-term movement from seasonal deviation.

Adaptive leakage-safe local median
----------------------------------

Comparison requests use the same adaptive lag-reference cleaning rule as
``lag_features.py``. For feature row ``t``, lag ``L``, and configured nominal
half-window ``h``::

    safe_right_points = min(h, L - 1)

The local-median candidate around referenced point ``t - L`` therefore spans::

    [referenced - h, ..., referenced + safe_right_points]

The current row and future rows can never enter the cleaned lag reference.
Local median cleaning is supported only with ``source='raw_value'`` and applies
only to the referenced historical value. The current value remains the selected
source at row ``t``.

General rules:

- input represents one valid site already ordered by timestamp;
- every lag is expressed in exact timeline points;
- output columns follow the names defined by the schema;
- initial lag warm-up rows remain ``NaN``;
- zero denominators and infinite outputs become ``NaN``;
- source row count and index are preserved;
- original source Series and the Part B dataframe are never modified; and
- ``corrected_history`` cannot be combined with local median cleaning.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pandas as pd

from app.schemas.feature_engineering import (
    DifferenceFeatureRequest,
    PercentageChangeFeatureRequest,
    SeasonalDifferenceFeatureRequest,
)

from .source_preparation import (
    LocalMedianCandidateResult,
    PreparedFeatureSources,
    prepare_local_median_candidates,
    prepare_referenced_values,
)


ComparisonFeatureRequest: TypeAlias = (
    DifferenceFeatureRequest
    | PercentageChangeFeatureRequest
    | SeasonalDifferenceFeatureRequest
)


def _resolve_safe_right_window_points(
    *,
    lag: int,
    half_window_points: int,
) -> int:
    """Return the largest right-side window strictly preceding feature row t."""

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
    request: ComparisonFeatureRequest,
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
            "A status Series is required when comparison local-median "
            "cleaning is enabled."
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


def _select_current_values(
    *,
    request: ComparisonFeatureRequest,
    sources: PreparedFeatureSources,
) -> pd.Series:
    """Return one copied finite-or-NaN current source aligned with the site."""

    current = pd.to_numeric(
        sources.bundle.select(request.source),
        errors="coerce",
    ).astype("float64")

    finite_values = current.dropna().to_numpy(dtype="float64")

    if finite_values.size and not np.isfinite(finite_values).all():
        raise ValueError(
            f"Selected current source {request.source!r} contains positive or "
            "negative infinity."
        )

    return current.copy()


def _calculate_comparison_values(
    *,
    request: ComparisonFeatureRequest,
    current_values: pd.Series,
    referenced_values: pd.Series,
) -> pd.Series:
    """Calculate one requested comparison using aligned current/reference data."""

    if not current_values.index.equals(referenced_values.index):
        raise ValueError(
            "Current and referenced comparison values are not index-aligned."
        )

    difference = current_values - referenced_values

    if isinstance(
        request,
        (
            DifferenceFeatureRequest,
            SeasonalDifferenceFeatureRequest,
        ),
    ):
        result = difference

    elif isinstance(request, PercentageChangeFeatureRequest):
        if request.zero_denominator_strategy != "nan":
            raise ValueError(
                "Unsupported percentage-change zero-denominator strategy: "
                f"{request.zero_denominator_strategy!r}"
            )

        denominator = referenced_values.mask(
            referenced_values.eq(0.0)
        )
        result = difference / denominator

    else:
        raise TypeError(
            "Unsupported comparison request: "
            f"{type(request).__name__}."
        )

    # The percentage schema requires zero denominators to become NaN. This
    # replacement is also a final defensive guard against any unexpected
    # floating-point infinity from direct/internal calls.
    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    ).astype("float64")

    return result


def _column_name_for_request(
    request: ComparisonFeatureRequest,
    *,
    lag: int,
) -> str:
    """Return the deterministic schema-compatible output name for one lag."""

    if isinstance(request, DifferenceFeatureRequest):
        return f"difference_{lag}"

    if isinstance(request, PercentageChangeFeatureRequest):
        return f"percentage_change_{lag}"

    if isinstance(request, SeasonalDifferenceFeatureRequest):
        return f"seasonal_difference_{lag}"

    raise TypeError(
        "Unsupported comparison request: "
        f"{type(request).__name__}."
    )


def _validate_comparison_output(
    output: pd.DataFrame,
    *,
    expected_index: pd.Index,
    expected_columns: tuple[str, ...],
    request_lags: list[int],
) -> None:
    """Defensively verify the comparison-feature output contract."""

    if len(output) != len(expected_index):
        raise RuntimeError(
            "Comparison-feature generation unexpectedly changed the row count."
        )

    if not output.index.equals(expected_index):
        raise RuntimeError(
            "Comparison-feature generation unexpectedly changed the source "
            "index."
        )

    if tuple(output.columns) != expected_columns:
        raise RuntimeError(
            "Comparison-feature generation produced unexpected columns. "
            f"Expected {list(expected_columns)}, received "
            f"{output.columns.tolist()}."
        )

    if output.columns.has_duplicates:
        duplicates = output.columns[
            output.columns.duplicated()
        ].tolist()
        raise RuntimeError(
            "Comparison-feature generation produced duplicate columns: "
            f"{duplicates}"
        )

    for lag, column_name in zip(
        request_lags,
        expected_columns,
        strict=True,
    ):
        warm_up_rows = min(lag, len(output))

        if (
            warm_up_rows
            and not output[column_name]
            .iloc[:warm_up_rows]
            .isna()
            .all()
        ):
            raise RuntimeError(
                f"Comparison feature '{column_name}' populated one or more "
                "warm-up rows that should be NaN."
            )

    numeric = output.apply(
        pd.to_numeric,
        errors="coerce",
    )
    finite_values = numeric.to_numpy(dtype="float64")
    finite_mask = ~np.isnan(finite_values)

    if finite_mask.any() and not np.isfinite(finite_values[finite_mask]).all():
        raise RuntimeError(
            "Comparison-feature generation produced positive or negative "
            "infinity."
        )


def generate_comparison_features(
    request: ComparisonFeatureRequest,
    *,
    sources: PreparedFeatureSources,
    bucket_minutes: int,
) -> pd.DataFrame:
    """Generate one lag-based comparison request for a valid site.

    Args:
        request:
            Validated ``difference``, ``percentage_change``, or
            ``seasonal_difference`` request.
        sources:
            Prepared raw and optional corrected-history sources for one
            timestamp-ordered site.
        bucket_minutes:
            Resolved dataset bucket used only when converting local-cleaning
            durations from minutes to points.

    Returns:
        pd.DataFrame:
            One column per requested lag, aligned with the source site index.

    Raises:
        TypeError:
            If an unsupported request or prepared-source object is supplied.
        ValueError:
            If source preparation, local-cleaning settings, or percentage
            denominators cannot be handled under the validated contract.
    """

    if not isinstance(
        request,
        (
            DifferenceFeatureRequest,
            PercentageChangeFeatureRequest,
            SeasonalDifferenceFeatureRequest,
        ),
    ):
        raise TypeError(
            "generate_comparison_features requires DifferenceFeatureRequest, "
            "PercentageChangeFeatureRequest, or "
            "SeasonalDifferenceFeatureRequest; received "
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
            "Cannot generate comparison features from an empty prepared site "
            "source."
        )

    current_values = _select_current_values(
        request=request,
        sources=sources,
    )

    if not current_values.index.equals(source_index):
        raise ValueError(
            "Selected current comparison source is not aligned with raw_value."
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

        comparison_values = _calculate_comparison_values(
            request=request,
            current_values=current_values,
            referenced_values=referenced.values,
        )

        column_name = _column_name_for_request(
            request,
            lag=lag,
        )
        comparison_values.name = column_name
        output[column_name] = comparison_values

    expected_columns = tuple(
        _column_name_for_request(
            request,
            lag=lag,
        )
        for lag in request.lags
    )

    _validate_comparison_output(
        output,
        expected_index=source_index,
        expected_columns=expected_columns,
        request_lags=request.lags,
    )

    return output


__all__ = [
    "ComparisonFeatureRequest",
    "generate_comparison_features",
]