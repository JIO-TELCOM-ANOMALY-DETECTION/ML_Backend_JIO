"""Default Part C feature profile.

Project location::

    app/schemas/feature_engineering/defaults.py

This module owns the default feature profile only. Custom requests bypass this
profile and are validated by ``feature_engineering.py``. Intrinsic formula
settings, such as EWMA ``adjust=False`` or rolling standard-deviation ``ddof``,
remain on their respective request schemas because they define feature behavior
rather than the default list of requested features.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


FeatureSource = Literal[
    "raw_value",
    "corrected_history",
]

RollingStatistic = Literal[
    "mean",
    "median",
    "mad",
    "std",
    "min",
    "max",
    "range",
    "iqr",
]

TimeCyclicFeature = Literal[
    "sin_tod",
    "cos_tod",
]

CalendarFeature = Literal[
    "is_weekend",
    "day_of_week",
    "day_sin",
    "day_cos",
]


def _validate_unique_positive_integers(
    values: list[int],
    *,
    field_name: str,
) -> list[int]:
    if not values:
        raise ValueError(f"{field_name} must contain at least one value.")
    if any(value <= 0 for value in values):
        raise ValueError(
            f"Every value in {field_name} must be greater than zero."
        )
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} cannot contain duplicate values.")
    return sorted(values)


def _validate_unique_strings(
    values: list[str],
    *,
    field_name: str,
) -> list[str]:
    if not values:
        raise ValueError(f"{field_name} must contain at least one value.")
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} cannot contain duplicate values.")
    return values


def _duration_to_points(
    *,
    duration_minutes: int,
    bucket_minutes: int,
    field_name: str,
) -> int:
    """Convert an exact duration to timeline points independently."""

    if duration_minutes % bucket_minutes != 0:
        raise ValueError(
            f"{field_name}={duration_minutes} minutes must be exactly "
            f"divisible by the resolved dataset bucket of "
            f"{bucket_minutes} minutes."
        )

    points = duration_minutes // bucket_minutes

    if points <= 0:
        raise ValueError(f"{field_name} resolves to zero points.")

    return points


class DefaultFeatureProfileConfig(BaseModel):
    """
    Define the schema-owned feature list used when ``features`` is omitted.

    These defaults mirror the useful common core of the previous feature
    scripts while remaining bucket-aware. A custom request bypasses this
    profile and generates exactly the supplied feature list.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    lag_source: FeatureSource = Field(
        default="corrected_history",
    )

    include_previous_point_lag: bool = Field(
        default=True,
    )

    include_daily_lag: bool = Field(
        default=True,
    )

    include_weekly_lag: bool = Field(
        default=True,
    )

    rolling_source: FeatureSource = Field(
        default="corrected_history",
    )

    rolling_statistics: list[RollingStatistic] = Field(
        default_factory=lambda: ["median", "mad"],
    )

    rolling_window_minutes: list[int] = Field(
        default_factory=lambda: [60],
        description="Default rolling durations resolved to points.",
    )

    time_bucket_minutes: list[int] = Field(
        default_factory=lambda: [30],
        description=(
            "Default time-bucket features. Any custom request may supply any "
            "duration that divides 1,440 exactly."
        ),
    )

    time_cyclic_features: list[TimeCyclicFeature] = Field(
        default_factory=lambda: ["sin_tod", "cos_tod"],
    )

    calendar_features: list[CalendarFeature] = Field(
        default_factory=lambda: ["is_weekend"],
    )

    @field_validator("rolling_statistics")
    @classmethod
    def validate_rolling_statistics(
        cls,
        values: list[RollingStatistic],
    ) -> list[RollingStatistic]:
        """Validate default rolling statistics."""

        return _validate_unique_strings(
            values,
            field_name="rolling_statistics",
        )

    @field_validator("rolling_window_minutes")
    @classmethod
    def validate_rolling_windows(
        cls,
        values: list[int],
    ) -> list[int]:
        """Validate default rolling durations."""

        return _validate_unique_positive_integers(
            values,
            field_name="rolling_window_minutes",
        )

    @field_validator("time_bucket_minutes")
    @classmethod
    def validate_default_time_buckets(
        cls,
        values: list[int],
    ) -> list[int]:
        """Validate default time-bucket durations."""

        values = _validate_unique_positive_integers(
            values,
            field_name="time_bucket_minutes",
        )

        invalid = [
            value
            for value in values
            if 1_440 % value != 0
        ]

        if invalid:
            raise ValueError(
                "Every default time bucket must divide 1,440 minutes exactly. "
                f"Invalid values: {invalid}"
            )

        return values

    @field_validator("time_cyclic_features")
    @classmethod
    def validate_time_cyclic_features(
        cls,
        values: list[TimeCyclicFeature],
    ) -> list[TimeCyclicFeature]:
        """Validate default cyclical columns."""

        return _validate_unique_strings(
            values,
            field_name="time_cyclic_features",
        )

    @field_validator("calendar_features")
    @classmethod
    def validate_calendar_features(
        cls,
        values: list[CalendarFeature],
    ) -> list[CalendarFeature]:
        """Validate default calendar columns."""

        return _validate_unique_strings(
            values,
            field_name="calendar_features",
        )

    def build_request_payloads(
        self,
        *,
        bucket_minutes: int,
        points_per_day: int,
    ) -> list[dict[str, Any]]:
        """Build plain payloads that the main schema validates into requests."""

        requests: list[dict[str, Any]] = []
        lags: list[int] = []

        if self.include_previous_point_lag:
            lags.append(1)

        if self.include_daily_lag:
            lags.append(points_per_day)

        if self.include_weekly_lag:
            lags.append(points_per_day * 7)

        if lags:
            requests.append(
                {
                    "type": "lag",
                    "lags": sorted(set(lags)),
                    "source": self.lag_source,
                }
            )

        rolling_windows = [
            _duration_to_points(
                duration_minutes=duration,
                bucket_minutes=bucket_minutes,
                field_name="rolling_window_minutes",
            )
            for duration in self.rolling_window_minutes
        ]

        requests.append(
            {
                "type": "rolling",
                "statistics": list(self.rolling_statistics),
                "windows": rolling_windows,
                "source": self.rolling_source,
            }
        )

        if self.time_bucket_minutes:
            requests.append(
                {
                    "type": "time_bucket",
                    "bucket_minutes": list(self.time_bucket_minutes),
                }
            )

        if self.time_cyclic_features:
            requests.append(
                {
                    "type": "time_cyclic",
                    "features": list(self.time_cyclic_features),
                }
            )

        if self.calendar_features:
            requests.append(
                {
                    "type": "calendar",
                    "features": list(self.calendar_features),
                }
            )

        return requests




__all__ = [
    "DefaultFeatureProfileConfig",
]