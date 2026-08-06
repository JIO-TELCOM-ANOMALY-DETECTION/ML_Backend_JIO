"""Generate timestamp-based time features for Part C feature engineering.

Intended project location::

    app/ml/feature_engineering/time_features.py

This module supports:

- configurable time-of-day bucket features; and
- cyclical time-of-day sine and cosine features.

Rules:

- only the configured timestamp column is used;
- KPI values, source preparation, status labels, and cleaning are irrelevant;
- timestamps are parsed for calculation but the source dataframe is unchanged;
- bucket numbering starts at zero from midnight;
- every requested time bucket must divide 1,440 minutes exactly;
- generated output preserves the source row count and index; and
- calendar-derived features remain the responsibility of
  ``calendar_features.py``.
"""

from __future__ import annotations

from math import pi
from typing import Final, TypeAlias

import numpy as np
import pandas as pd

from app.schemas.feature_engineering import (
    TimeBucketFeatureRequest,
    TimeCyclicFeatureRequest,
)

from .common import require_columns


MINUTES_PER_DAY: Final = 1_440
TimeFeatureRequest: TypeAlias = (
    TimeBucketFeatureRequest
    | TimeCyclicFeatureRequest
)


def _prepare_timestamps(
    site_dataframe: pd.DataFrame,
    *,
    timestamp_column: str,
) -> pd.Series:
    """Return one validated datetime Series aligned with the site dataframe.

    Timestamp validity and exact bucket spacing should already be checked by
    ``feature_validation.py``. This function performs a defensive parse so that
    feature generation fails clearly instead of silently producing invalid
    values.
    """

    if not isinstance(site_dataframe, pd.DataFrame):
        raise TypeError(
            "site_dataframe must be a pandas DataFrame."
        )

    if not isinstance(timestamp_column, str) or not timestamp_column.strip():
        raise ValueError(
            "timestamp_column must be a non-empty string."
        )

    require_columns(
        site_dataframe,
        [timestamp_column],
        context="time-feature generation",
    )

    if site_dataframe.empty:
        raise ValueError(
            "Cannot generate time features from an empty site dataframe."
        )

    timestamps = pd.to_datetime(
        site_dataframe[timestamp_column],
        errors="coerce",
    )

    invalid_mask = timestamps.isna()

    if invalid_mask.any():
        invalid_indices = (
            timestamps.index[invalid_mask]
            .tolist()[:10]
        )
        raise ValueError(
            f"Timestamp column '{timestamp_column}' contains "
            f"{int(invalid_mask.sum()):,} missing or invalid values. "
            f"First invalid indices: {invalid_indices}"
        )

    # Preserve the dataframe index explicitly. No timezone conversion or
    # timestamp mutation is performed.
    timestamps.index = site_dataframe.index
    timestamps.name = timestamp_column

    return timestamps


def _minutes_since_midnight(
    timestamps: pd.Series,
) -> pd.Series:
    """Return fractional minutes elapsed since local midnight.

    Seconds, microseconds, and nanoseconds are included defensively. For the
    bucket-aligned telecom timelines these components should normally be zero.
    """

    minutes = (
        timestamps.dt.hour.astype("float64") * 60.0
        + timestamps.dt.minute.astype("float64")
        + timestamps.dt.second.astype("float64") / 60.0
        + timestamps.dt.microsecond.astype("float64") / 60_000_000.0
        + timestamps.dt.nanosecond.astype("float64") / 60_000_000_000.0
    )

    minutes.name = "minutes_since_midnight"
    return minutes


def _validate_output_alignment(
    output: pd.DataFrame,
    *,
    expected_index: pd.Index,
    context: str,
) -> None:
    """Defensively verify generator row and index preservation."""

    if len(output) != len(expected_index):
        raise RuntimeError(
            f"{context} unexpectedly changed the row count."
        )

    if not output.index.equals(expected_index):
        raise RuntimeError(
            f"{context} unexpectedly changed the source index."
        )

    if output.columns.empty:
        raise RuntimeError(
            f"{context} produced no feature columns."
        )

    if output.columns.has_duplicates:
        duplicates = (
            output.columns[
                output.columns.duplicated()
            ]
            .tolist()
        )
        raise RuntimeError(
            f"{context} produced duplicate columns: {duplicates}"
        )


def generate_time_bucket_features(
    request: TimeBucketFeatureRequest,
    *,
    timestamps: pd.Series,
) -> pd.DataFrame:
    """Generate requested zero-based time-of-day bucket identifiers.

    Formula::

        time_bucket_Nm = floor(minutes_since_midnight / N)

    Examples for a 30-minute bucket:

    - 00:00 -> 0
    - 00:29 -> 0
    - 00:30 -> 1
    - 23:59 -> 47
    """

    if not isinstance(request, TimeBucketFeatureRequest):
        raise TypeError(
            "generate_time_bucket_features requires "
            "TimeBucketFeatureRequest; received "
            f"{type(request).__name__}."
        )

    minutes = _minutes_since_midnight(
        timestamps
    )
    output = pd.DataFrame(
        index=timestamps.index,
    )

    for bucket_minutes in request.bucket_minutes:
        if bucket_minutes <= 0:
            raise ValueError(
                "Every requested time bucket must be greater than zero."
            )

        if MINUTES_PER_DAY % bucket_minutes != 0:
            raise ValueError(
                "Every requested time bucket must divide 1,440 minutes "
                f"exactly; received {bucket_minutes}."
            )

        column_name = f"time_bucket_{bucket_minutes}m"

        output[column_name] = np.floor(
            minutes / float(bucket_minutes)
        ).astype("int64")

    _validate_output_alignment(
        output,
        expected_index=timestamps.index,
        context="Time-bucket generation",
    )

    return output


def generate_time_cyclic_features(
    request: TimeCyclicFeatureRequest,
    *,
    timestamps: pd.Series,
) -> pd.DataFrame:
    """Generate requested cyclical encodings of time within the day.

    Formula::

        angle = 2 * pi * minutes_since_midnight / 1,440
        sin_tod = sin(angle)
        cos_tod = cos(angle)

    The feature repeats continuously at midnight and does not use the date.
    """

    if not isinstance(request, TimeCyclicFeatureRequest):
        raise TypeError(
            "generate_time_cyclic_features requires "
            "TimeCyclicFeatureRequest; received "
            f"{type(request).__name__}."
        )

    minutes = _minutes_since_midnight(
        timestamps
    )
    angle = (
        2.0
        * pi
        * minutes
        / float(MINUTES_PER_DAY)
    )

    output = pd.DataFrame(
        index=timestamps.index,
    )

    for feature_name in request.features:
        if feature_name == "sin_tod":
            output[feature_name] = np.sin(
                angle
            ).astype("float64")

        elif feature_name == "cos_tod":
            output[feature_name] = np.cos(
                angle
            ).astype("float64")

        else:
            # The schema should prevent this branch. It remains as a defensive
            # runtime guard for direct/internal calls.
            raise ValueError(
                "Unsupported cyclical time feature: "
                f"{feature_name!r}"
            )

    _validate_output_alignment(
        output,
        expected_index=timestamps.index,
        context="Cyclical time-feature generation",
    )

    return output


def generate_time_features(
    request: TimeFeatureRequest,
    *,
    site_dataframe: pd.DataFrame,
    timestamp_column: str,
) -> pd.DataFrame:
    """Dispatch one validated time request for a timestamp-ordered site.

    Args:
        request:
            Validated ``time_bucket`` or ``time_cyclic`` request.
        site_dataframe:
            One valid site dataframe already ordered by timestamp.
        timestamp_column:
            Resolved physical timestamp-column name from the shared dataset
            schema.

    Returns:
        pd.DataFrame:
            Requested numeric time features aligned with the site dataframe.

    Raises:
        TypeError:
            If an unsupported request model is supplied.
        ValueError:
            If timestamps cannot be used safely for feature generation.
    """

    timestamps = _prepare_timestamps(
        site_dataframe,
        timestamp_column=timestamp_column,
    )

    if isinstance(request, TimeBucketFeatureRequest):
        return generate_time_bucket_features(
            request,
            timestamps=timestamps,
        )

    if isinstance(request, TimeCyclicFeatureRequest):
        return generate_time_cyclic_features(
            request,
            timestamps=timestamps,
        )

    raise TypeError(
        "time_features.py supports only TimeBucketFeatureRequest and "
        "TimeCyclicFeatureRequest; received "
        f"{type(request).__name__}."
    )


__all__ = [
    "MINUTES_PER_DAY",
    "TimeFeatureRequest",
    "generate_time_bucket_features",
    "generate_time_cyclic_features",
    "generate_time_features",
]
