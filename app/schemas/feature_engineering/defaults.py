"""Default Part C feature profiles.

Intended project location::

    app/schemas/feature_engineering/defaults.py

This module owns the algorithm-specific default feature lists used when the
caller omits ``features``. It deliberately returns plain request payloads so it
does not import feature request models from ``feature_engineering.py`` and
therefore cannot create a circular import.

Custom feature requests bypass these profiles and are validated directly by
``feature_engineering.py``.
"""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator


FeatureSource: TypeAlias = Literal[
    "raw_value",
    "corrected_history",
]

RollingStatistic: TypeAlias = Literal[
    "mean",
    "median",
    "mad",
    "std",
    "min",
    "max",
    "range",
    "iqr",
]

TimeCyclicFeature: TypeAlias = Literal[
    "sin_tod",
    "cos_tod",
]

CalendarFeature: TypeAlias = Literal[
    "is_weekend",
    "day_of_week",
    "day_sin",
    "day_cos",
]

DefaultFeatureProfileName: TypeAlias = Literal[
    "xgboost",
    "isolation_forest",
]


def _validate_unique_positive_integers(
    values: list[int],
    *,
    field_name: str,
) -> list[int]:
    """Validate a non-empty list of unique positive integers."""

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
    """Validate a non-empty list of unique strings."""

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
    """Convert a duration to exact timeline points."""

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


class _DefaultProfileBase(BaseModel):
    """Shared strict validation for algorithm-specific default profiles."""

    model_config = ConfigDict(
        extra="forbid",
    )

    time_bucket_minutes: list[int] = Field(
        default_factory=lambda: [30],
        description=(
            "Default time-of-day bucket features. Every duration must divide "
            "1,440 minutes exactly."
        ),
    )

    calendar_features: list[CalendarFeature] = Field(
        default_factory=lambda: ["is_weekend"],
    )

    @field_validator("time_bucket_minutes")
    @classmethod
    def validate_default_time_buckets(
        cls,
        values: list[int],
    ) -> list[int]:
        """Validate day-dividing default time buckets."""

        values = _validate_unique_positive_integers(
            values,
            field_name="time_bucket_minutes",
        )

        invalid = [value for value in values if 1_440 % value != 0]

        if invalid:
            raise ValueError(
                "Every default time bucket must divide 1,440 minutes exactly. "
                f"Invalid values: {invalid}"
            )

        return values

    @field_validator("calendar_features")
    @classmethod
    def validate_calendar_features(
        cls,
        values: list[CalendarFeature],
    ) -> list[CalendarFeature]:
        """Reject duplicate default calendar features."""

        return _validate_unique_strings(
            values,
            field_name="calendar_features",
        )


class XGBoostDefaultFeatureProfileConfig(_DefaultProfileBase):
    """Default features mirroring the earlier XGBoost feature script."""

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

    @field_validator("rolling_statistics")
    @classmethod
    def validate_rolling_statistics(
        cls,
        values: list[RollingStatistic],
    ) -> list[RollingStatistic]:
        """Reject duplicate default rolling statistics."""

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

    def build_request_payloads(
        self,
        *,
        bucket_minutes: int,
        points_per_day: int,
    ) -> list[dict[str, Any]]:
        """Build the XGBoost default request payloads."""

        payloads: list[dict[str, Any]] = []
        lags: list[int] = []

        if self.include_previous_point_lag:
            lags.append(1)
        if self.include_daily_lag:
            lags.append(points_per_day)
        if self.include_weekly_lag:
            lags.append(points_per_day * 7)

        if lags:
            payloads.append(
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

        payloads.append(
            {
                "type": "rolling",
                "statistics": list(self.rolling_statistics),
                "windows": rolling_windows,
                "source": self.rolling_source,
            }
        )

        payloads.append(
            {
                "type": "time_bucket",
                "bucket_minutes": list(self.time_bucket_minutes),
            }
        )

        payloads.append(
            {
                "type": "calendar",
                "features": list(self.calendar_features),
            }
        )

        return payloads


class IsolationForestDefaultFeatureProfileConfig(_DefaultProfileBase):
    """Default features mirroring the earlier Isolation Forest script."""

    current_value_source: FeatureSource = Field(
        default="raw_value",
    )

    lag_source: FeatureSource = Field(
        default="raw_value",
    )

    include_daily_lag: bool = Field(
        default=True,
    )

    rolling_source: FeatureSource = Field(
        default="raw_value",
    )

    rolling_statistics: list[RollingStatistic] = Field(
        default_factory=lambda: ["median"],
    )

    rolling_window_minutes: list[int] = Field(
        default_factory=lambda: [60],
    )

    residual_source: FeatureSource = Field(
        default="raw_value",
    )

    baseline_source: FeatureSource = Field(
        default="raw_value",
    )

    time_cyclic_features: list[TimeCyclicFeature] = Field(
        default_factory=lambda: ["sin_tod", "cos_tod"],
    )

    @field_validator("rolling_statistics")
    @classmethod
    def validate_rolling_statistics(
        cls,
        values: list[RollingStatistic],
    ) -> list[RollingStatistic]:
        """Reject duplicate default rolling statistics."""

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

    @field_validator("time_cyclic_features")
    @classmethod
    def validate_time_cyclic_features(
        cls,
        values: list[TimeCyclicFeature],
    ) -> list[TimeCyclicFeature]:
        """Reject duplicate default cyclic features."""

        return _validate_unique_strings(
            values,
            field_name="time_cyclic_features",
        )

    def build_request_payloads(
        self,
        *,
        bucket_minutes: int,
        points_per_day: int,
    ) -> list[dict[str, Any]]:
        """Build the Isolation Forest default request payloads."""

        rolling_windows = [
            _duration_to_points(
                duration_minutes=duration,
                bucket_minutes=bucket_minutes,
                field_name="rolling_window_minutes",
            )
            for duration in self.rolling_window_minutes
        ]

        payloads: list[dict[str, Any]] = [
            {
                "type": "current_value",
                "source": self.current_value_source,
            },
            {
                "type": "residual",
                "source": self.residual_source,
                "baseline": {
                    "method": "weighted_seasonal",
                    "source": self.baseline_source,
                },
            },
            {
                "type": "rolling",
                "statistics": list(self.rolling_statistics),
                "windows": rolling_windows,
                "source": self.rolling_source,
            },
        ]

        if self.include_daily_lag:
            payloads.append(
                {
                    "type": "lag",
                    "lags": [points_per_day],
                    "source": self.lag_source,
                }
            )

        payloads.append(
            {
                "type": "time_bucket",
                "bucket_minutes": list(self.time_bucket_minutes),
            }
        )

        payloads.append(
            {
                "type": "time_cyclic",
                "features": list(self.time_cyclic_features),
            }
        )

        payloads.append(
            {
                "type": "calendar",
                "features": list(self.calendar_features),
            }
        )

        return payloads


class DefaultFeatureProfilesConfig(BaseModel):
    """Hold both algorithm-specific default feature profiles."""

    model_config = ConfigDict(
        extra="forbid",
    )

    xgboost: XGBoostDefaultFeatureProfileConfig = Field(
        default_factory=XGBoostDefaultFeatureProfileConfig,
    )

    isolation_forest: IsolationForestDefaultFeatureProfileConfig = Field(
        default_factory=IsolationForestDefaultFeatureProfileConfig,
    )

    def build_request_payloads(
        self,
        *,
        profile: DefaultFeatureProfileName,
        bucket_minutes: int,
        points_per_day: int,
    ) -> list[dict[str, Any]]:
        """Build payloads for the selected algorithm profile."""

        if profile == "xgboost":
            return self.xgboost.build_request_payloads(
                bucket_minutes=bucket_minutes,
                points_per_day=points_per_day,
            )

        if profile == "isolation_forest":
            return self.isolation_forest.build_request_payloads(
                bucket_minutes=bucket_minutes,
                points_per_day=points_per_day,
            )

        raise ValueError(f"Unsupported default feature profile: {profile}")


# Backwards-compatible name for code that imported the earlier single-profile
# class. The object now contains both algorithm profiles.
DefaultFeatureProfileConfig = DefaultFeatureProfilesConfig


__all__ = [
    "DefaultFeatureProfileConfig",
    "DefaultFeatureProfileName",
    "DefaultFeatureProfilesConfig",
    "IsolationForestDefaultFeatureProfileConfig",
    "XGBoostDefaultFeatureProfileConfig",
]