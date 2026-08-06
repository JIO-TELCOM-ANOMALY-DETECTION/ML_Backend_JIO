"""
Request schemas for Part C telecom feature engineering.

Revision label: points window fixed.

This revision aligns schema history calculations and validation with the
adaptive leakage-safe local-median windows used by lag features.

Intended project location::

    app/schemas/feature_engineering/feature_engineering.py

This module defines:

- the structured list of requested features;
- per-feature source selection;
- temporary corrected-history settings;
- robust local-median cleaning around referenced points;
- validated feature-request and resolution settings;
- validation of feature parameters and output-name collisions; and
- bucket-aware resolution of defaults and minimum history requirements.

This module does not read dataframes, generate features, save files, update
manifests, or run model training. Those responsibilities belong to the Part C
feature modules, runner, workflow, and services.
"""

from __future__ import annotations

from math import isclose
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from app.schemas.common.dataset import BucketConfig

from .defaults import (
    DEFAULT_BASELINE_ANCHOR_STATUSES,
    DEFAULT_BASELINE_ANCHOR_WINDOW_EACH_SIDE_POINTS,
    DEFAULT_BASELINE_METHOD,
    DEFAULT_BASELINE_MINIMUM_HISTORY_DAYS,
    DEFAULT_BASELINE_RECENT_FALLBACK_DAYS,
    DEFAULT_BASELINE_RECENT_WEIGHT,
    DEFAULT_BASELINE_SLOT_FALLBACK_POINTS,
    DEFAULT_BASELINE_SOURCE,
    DEFAULT_BASELINE_WEEKLY_FALLBACK_WEEKS,
    DEFAULT_BASELINE_WEEKLY_WEIGHT,
    DEFAULT_CORRECTED_HISTORY_BAD_STATUSES,
    DEFAULT_CORRECTED_HISTORY_FORWARD_FILL_REMAINING,
    DEFAULT_CORRECTED_HISTORY_MINIMUM_PERIODS,
    DEFAULT_CORRECTED_HISTORY_REPLACEMENT_METHOD,
    DEFAULT_CORRECTED_HISTORY_REPLACE_MISSING_VALUES,
    DEFAULT_CORRECTED_HISTORY_WINDOW_MINUTES,
    DEFAULT_LOCAL_MEDIAN_ACCEPTED_STATUSES,
    DEFAULT_LOCAL_MEDIAN_BAD_STATUSES,
    DEFAULT_LOCAL_MEDIAN_ENABLED,
    DEFAULT_LOCAL_MEDIAN_HALF_WINDOW_MINUTES,
    DEFAULT_LOCAL_MEDIAN_MINIMUM_VALID_POINTS,
    DEFAULT_LOCAL_MEDIAN_UNAVAILABLE_STRATEGY,
    DEFAULT_RETAIN_ORIGINAL_COLUMNS,
    DEFAULT_SAVE_INTERMEDIATE_OUTPUTS,
    DefaultFeatureProfileName,
    DefaultFeatureProfilesConfig,
)


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

BaselineMethod = Literal[
    "weighted_seasonal",
]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_unique_positive_integers(
    values: list[int],
    *,
    field_name: str,
) -> list[int]:
    """Validate one non-empty list of unique positive integers."""

    if not values:
        raise ValueError(
            f"{field_name} must contain at least one value."
        )

    if any(value <= 0 for value in values):
        raise ValueError(
            f"Every value in {field_name} must be greater than zero."
        )

    if len(values) != len(set(values)):
        raise ValueError(
            f"{field_name} cannot contain duplicate values."
        )

    return sorted(values)


def _validate_unique_strings(
    values: list[str],
    *,
    field_name: str,
) -> list[str]:
    """Validate one non-empty list of unique strings."""

    if not values:
        raise ValueError(
            f"{field_name} must contain at least one value."
        )

    if len(values) != len(set(values)):
        raise ValueError(
            f"{field_name} cannot contain duplicate values."
        )

    return values


def _bucket_to_minutes(
    bucket: BucketConfig,
) -> int:
    """Convert the shared bucket configuration to whole minutes."""

    unit = bucket.bucket_type.strip().lower()

    unit_multipliers = {
        "minute": 1,
        "minutes": 1,
        "min": 1,
        "mins": 1,
        "hour": 60,
        "hours": 60,
        "hr": 60,
        "hrs": 60,
        "day": 1_440,
        "days": 1_440,
    }

    if unit not in unit_multipliers:
        raise ValueError(
            "Feature engineering supports bucket units expressed as "
            "minutes, hours, or days."
        )

    bucket_minutes = bucket.bucket_size * unit_multipliers[unit]

    if bucket_minutes <= 0:
        raise ValueError(
            "Resolved bucket duration must be greater than zero."
        )

    if 1_440 % bucket_minutes != 0:
        raise ValueError(
            "The resolved dataset bucket must divide 1,440 minutes exactly "
            "so that points_per_day is an integer."
        )

    return bucket_minutes


def _duration_to_points(
    *,
    duration_minutes: int,
    bucket_minutes: int,
    field_name: str,
) -> int:
    """Convert an exact duration to timeline points."""

    if duration_minutes % bucket_minutes != 0:
        raise ValueError(
            f"{field_name}={duration_minutes} minutes must be exactly "
            f"divisible by the resolved dataset bucket of "
            f"{bucket_minutes} minutes."
        )

    points = duration_minutes // bucket_minutes

    if points <= 0:
        raise ValueError(
            f"{field_name} resolves to zero points."
        )

    return points


# ---------------------------------------------------------------------------
# Shared cleaning configuration
# ---------------------------------------------------------------------------


class LocalMedianCleaningConfig(BaseModel):
    """
    Define optional robust cleaning around one referenced historical point.

    When enabled, a feature module may replace a referenced value whose status
    belongs to ``bad_statuses`` with the median of nearby values whose statuses
    belong to ``accepted_statuses``. This is separate from selecting raw values
    or the shared corrected-history series. Local median cleaning is
    supported only for ``raw_value`` references; it cannot be layered on
    top of ``corrected_history``.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    enabled: bool = Field(
        default=DEFAULT_LOCAL_MEDIAN_ENABLED,
        description=(
            "Apply local median cleaning when a referenced historical point "
            "has an unacceptable status."
        ),
    )

    method: Literal["local_median"] = Field(
        default="local_median",
        description="Robust point-cleaning method used by the first version.",
    )

    bad_statuses: set[int] = Field(
        default_factory=lambda: set(DEFAULT_LOCAL_MEDIAN_BAD_STATUSES),
        description=(
            "Statuses whose referenced values should be replaced before the "
            "feature is calculated."
        ),
    )

    accepted_statuses: set[int] = Field(
        default_factory=lambda: set(DEFAULT_LOCAL_MEDIAN_ACCEPTED_STATUSES),
        description=(
            "Statuses allowed to contribute values to the local median."
        ),
    )

    half_window_minutes: int | None = Field(
        default=DEFAULT_LOCAL_MEDIAN_HALF_WINDOW_MINUTES,
        gt=0,
        description=(
            "Nominal minutes searched on each side of the referenced point. "
            "Leakage-sensitive feature modules may truncate the right side "
            "while retaining the configured left-side history."
        ),
    )

    half_window_points: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Optional nominal half-window in data points. Supply this "
            "instead of half_window_minutes. Leakage-sensitive feature "
            "modules may use fewer right-side points when required."
        ),
    )

    minimum_valid_points: int = Field(
        default=DEFAULT_LOCAL_MEDIAN_MINIMUM_VALID_POINTS,
        gt=0,
        description=(
            "Minimum accepted values required before a local median can "
            "replace the referenced point."
        ),
    )

    unavailable_strategy: Literal[
        "nan",
        "keep_source",
    ] = Field(
        default=DEFAULT_LOCAL_MEDIAN_UNAVAILABLE_STRATEGY,
        description=(
            "Fallback when the local median lacks enough valid raw points. "
            "nan leaves the referenced value unavailable; keep_source retains "
            "a finite raw source value."
        ),
    )

    @model_validator(mode="after")
    def validate_window_definition(
        self,
    ) -> LocalMedianCleaningConfig:
        """Validate status groups and the selected half-window representation."""

        if not self.enabled:
            return self

        if not self.bad_statuses:
            raise ValueError(
                "bad_statuses cannot be empty when local cleaning is enabled."
            )

        if not self.accepted_statuses:
            raise ValueError(
                "accepted_statuses cannot be empty when local cleaning is "
                "enabled."
            )

        overlap = self.bad_statuses & self.accepted_statuses

        if overlap:
            raise ValueError(
                "bad_statuses and accepted_statuses cannot overlap. "
                f"Overlapping statuses: {sorted(overlap)}"
            )

        has_minutes = self.half_window_minutes is not None
        has_points = self.half_window_points is not None

        if has_minutes == has_points:
            raise ValueError(
                "Exactly one of half_window_minutes or half_window_points "
                "must be supplied when local cleaning is enabled."
            )

        return self

    def resolve_half_window_points(
        self,
        *,
        bucket_minutes: int,
    ) -> int:
        """Resolve the configured half-window to points."""

        if not self.enabled:
            return 0

        if self.half_window_points is not None:
            available_window_points = 2 * self.half_window_points + 1
            if self.minimum_valid_points > available_window_points:
                raise ValueError(
                    "minimum_valid_points cannot exceed the complete local "
                    "median window size."
                )
            return self.half_window_points

        assert self.half_window_minutes is not None

        half_window_points = _duration_to_points(
            duration_minutes=self.half_window_minutes,
            bucket_minutes=bucket_minutes,
            field_name="half_window_minutes",
        )

        available_window_points = 2 * half_window_points + 1
        if self.minimum_valid_points > available_window_points:
            raise ValueError(
                "minimum_valid_points cannot exceed the complete local "
                "median window size."
            )

        return half_window_points


class CorrectedHistoryConfig(BaseModel):
    """
    Define the optional shared temporary history used by value-based features.

    The original Part B value column remains unchanged. The runner constructs
    this temporary series only when at least one requested feature selects
    ``corrected_history``.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    bad_statuses: set[int] = Field(
        default_factory=lambda: set(DEFAULT_CORRECTED_HISTORY_BAD_STATUSES),
        description=(
            "Statuses masked before trailing-median replacement."
        ),
    )

    replace_missing_values: bool = Field(
        default=DEFAULT_CORRECTED_HISTORY_REPLACE_MISSING_VALUES,
        description=(
            "Treat unresolved numeric NaN values as unavailable history."
        ),
    )

    replacement_method: Literal["trailing_median"] = Field(
        default=DEFAULT_CORRECTED_HISTORY_REPLACEMENT_METHOD,
        description=(
            "Method used to replace unavailable values in temporary history."
        ),
    )

    replacement_window_minutes: int | None = Field(
        default=DEFAULT_CORRECTED_HISTORY_WINDOW_MINUTES,
        gt=0,
        description=(
            "Trailing-median duration. The default represents the previous "
            "one hour for any compatible dataset frequency."
        ),
    )

    replacement_window_points: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Optional exact trailing-median window in points. Supply this "
            "instead of replacement_window_minutes."
        ),
    )

    minimum_periods: int = Field(
        default=DEFAULT_CORRECTED_HISTORY_MINIMUM_PERIODS,
        gt=0,
        description=(
            "Minimum previous clean values needed for trailing-median "
            "replacement."
        ),
    )

    forward_fill_remaining: bool = Field(
        default=DEFAULT_CORRECTED_HISTORY_FORWARD_FILL_REMAINING,
        description=(
            "Forward-fill values that remain unavailable after trailing "
            "median replacement."
        ),
    )

    @model_validator(mode="after")
    def validate_replacement_window(
        self,
    ) -> CorrectedHistoryConfig:
        """Require exactly one replacement-window representation."""

        has_minutes = self.replacement_window_minutes is not None
        has_points = self.replacement_window_points is not None

        if has_minutes == has_points:
            raise ValueError(
                "Exactly one of replacement_window_minutes or "
                "replacement_window_points must be supplied."
            )

        if not self.bad_statuses and not self.replace_missing_values:
            raise ValueError(
                "Corrected history must clean at least one status or replace "
                "missing values."
            )

        return self

    def resolve_window_points(
        self,
        *,
        bucket_minutes: int,
    ) -> int:
        """Resolve the temporary-history replacement window to points."""

        if self.replacement_window_points is not None:
            window_points = self.replacement_window_points
        else:
            assert self.replacement_window_minutes is not None

            window_points = _duration_to_points(
                duration_minutes=self.replacement_window_minutes,
                bucket_minutes=bucket_minutes,
                field_name="replacement_window_minutes",
            )

        if self.minimum_periods > window_points:
            raise ValueError(
                "corrected_history.minimum_periods cannot exceed the resolved "
                "replacement window."
            )

        return window_points


# ---------------------------------------------------------------------------
# Feature request models
# ---------------------------------------------------------------------------


class FeatureRequestBase(BaseModel):
    """Provide strict shared Pydantic behaviour for every feature request."""

    model_config = ConfigDict(
        extra="forbid",
    )


class CurrentValueFeatureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["current_value"]


class LagFeatureRequest(FeatureRequestBase):
    """Request one or more exact point-based lag features."""

    type: Literal["lag"]

    lags: list[int] = Field(
        min_length=1,
        description="Exact positive lag values expressed in timeline points.",
    )

    source: FeatureSource = Field(
        default="corrected_history",
        description="Value source shifted to create each lag.",
    )

    point_cleaning: LocalMedianCleaningConfig = Field(
        default_factory=LocalMedianCleaningConfig,
        description=(
            "Robust median cleaning applied around an unacceptable referenced "
            "lag point."
        ),
    )

    @field_validator("lags")
    @classmethod
    def validate_lags(
        cls,
        values: list[int],
    ) -> list[int]:
        """Validate exact lag points."""

        return _validate_unique_positive_integers(
            values,
            field_name="lags",
        )

    @model_validator(mode="after")
    def validate_source_cleaning_combination(
        self,
    ) -> LagFeatureRequest:
        """Reject corrected_history combined with local median cleaning."""

        if (
            self.source == "corrected_history"
            and self.point_cleaning.enabled
        ):
            raise ValueError(
                "point_cleaning.enabled=True is supported only when "
                "source='raw_value'. corrected_history and local median "
                "cleaning are separate methods and cannot be combined."
            )

        return self


class RollingFeatureRequest(FeatureRequestBase):
    """Request rolling statistical features over previous observations."""

    type: Literal["rolling"]

    statistics: list[RollingStatistic] = Field(
        min_length=1,
        description="Rolling statistics generated for every selected window.",
    )

    windows: list[int] = Field(
        min_length=1,
        description="Positive rolling-window sizes expressed in points.",
    )

    source: FeatureSource = Field(
        default="corrected_history",
        description="Value source used for all requested rolling statistics.",
    )

    minimum_periods: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Minimum observations required. None requires each complete "
            "rolling window."
        ),
    )

    std_ddof: Literal[0, 1] = Field(
        default=1,
        description=(
            "Standard-deviation divisor adjustment. One uses sample standard "
            "deviation; zero uses population standard deviation."
        ),
    )

    @field_validator("statistics")
    @classmethod
    def validate_statistics(
        cls,
        values: list[RollingStatistic],
    ) -> list[RollingStatistic]:
        """Reject duplicate rolling statistics."""

        return _validate_unique_strings(
            values,
            field_name="statistics",
        )

    @field_validator("windows")
    @classmethod
    def validate_windows(
        cls,
        values: list[int],
    ) -> list[int]:
        """Validate rolling windows."""

        return _validate_unique_positive_integers(
            values,
            field_name="windows",
        )

    @model_validator(mode="after")
    def validate_minimum_periods(
        self,
    ) -> RollingFeatureRequest:
        """Ensure one shared minimum_periods works for every selected window."""

        if (
            self.minimum_periods is not None
            and self.minimum_periods > min(self.windows)
        ):
            raise ValueError(
                "minimum_periods cannot exceed the smallest rolling window."
            )

        return self


class TimeBucketFeatureRequest(FeatureRequestBase):
    """Request one or more configurable time-of-day bucket features."""

    type: Literal["time_bucket"]

    bucket_minutes: list[int] = Field(
        min_length=1,
        description=(
            "Time-bucket durations. Every value must divide 1,440 minutes "
            "exactly."
        ),
    )

    @field_validator("bucket_minutes")
    @classmethod
    def validate_bucket_minutes(
        cls,
        values: list[int],
    ) -> list[int]:
        """Validate arbitrary day-dividing bucket sizes."""

        values = _validate_unique_positive_integers(
            values,
            field_name="bucket_minutes",
        )

        invalid = [
            value
            for value in values
            if 1_440 % value != 0
        ]

        if invalid:
            raise ValueError(
                "Every time bucket must divide 1,440 minutes exactly. "
                f"Invalid values: {invalid}"
            )

        return values


class TimeCyclicFeatureRequest(FeatureRequestBase):
    """Request cyclical encodings of time within the day."""

    type: Literal["time_cyclic"]

    features: list[TimeCyclicFeature] = Field(
        min_length=1,
        description="Requested time-of-day sine and cosine columns.",
    )

    @field_validator("features")
    @classmethod
    def validate_features(
        cls,
        values: list[TimeCyclicFeature],
    ) -> list[TimeCyclicFeature]:
        """Reject duplicate cyclical features."""

        return _validate_unique_strings(
            values,
            field_name="features",
        )


class CalendarFeatureRequest(FeatureRequestBase):
    """Request calendar-derived timestamp features."""

    type: Literal["calendar"]

    features: list[CalendarFeature] = Field(
        min_length=1,
        description="Requested calendar feature columns.",
    )

    @field_validator("features")
    @classmethod
    def validate_features(
        cls,
        values: list[CalendarFeature],
    ) -> list[CalendarFeature]:
        """Reject duplicate calendar features."""

        return _validate_unique_strings(
            values,
            field_name="features",
        )


class LagComparisonFeatureRequestBase(FeatureRequestBase):
    """Shared fields for features comparing the current and lagged values."""

    lags: list[int] = Field(
        min_length=1,
        description="Exact positive comparison lags expressed in points.",
    )

    source: FeatureSource = Field(
        default="corrected_history",
        description=(
            "Source used for both the current and referenced lagged values."
        ),
    )

    point_cleaning: LocalMedianCleaningConfig = Field(
        default_factory=LocalMedianCleaningConfig,
        description=(
            "Robust median cleaning applied around an unacceptable referenced "
            "lag point."
        ),
    )

    @field_validator("lags")
    @classmethod
    def validate_lags(
        cls,
        values: list[int],
    ) -> list[int]:
        """Validate comparison lags."""

        return _validate_unique_positive_integers(
            values,
            field_name="lags",
        )

    @model_validator(mode="after")
    def validate_source_cleaning_combination(
        self,
    ) -> LagComparisonFeatureRequestBase:
        """Reject corrected_history combined with local median cleaning."""

        if (
            self.source == "corrected_history"
            and self.point_cleaning.enabled
        ):
            raise ValueError(
                "point_cleaning.enabled=True is supported only when "
                "source='raw_value'. corrected_history and local median "
                "cleaning are separate methods and cannot be combined."
            )

        return self


class DifferenceFeatureRequest(LagComparisonFeatureRequestBase):
    """Request current value minus lagged value."""

    type: Literal["difference"]


class PercentageChangeFeatureRequest(LagComparisonFeatureRequestBase):
    """Request percentage change relative to an exact lagged value."""

    type: Literal["percentage_change"]

    zero_denominator_strategy: Literal["nan"] = Field(
        default="nan",
        description=(
            "Return NaN instead of infinity when the lagged denominator is "
            "zero."
        ),
    )


class SeasonalDifferenceFeatureRequest(LagComparisonFeatureRequestBase):
    """
    Request a named seasonal difference using exact user-provided lag points.

    Formula::

        seasonal_difference_lag(t) = source(t) - source(t - lag)
    """

    type: Literal["seasonal_difference"]


class RollingZScoreFeatureRequest(FeatureRequestBase):
    """Request current value standardized by previous rolling statistics."""

    type: Literal["rolling_zscore"]

    windows: list[int] = Field(
        min_length=1,
        description="Previous-value windows used for rolling mean and std.",
    )

    source: FeatureSource = Field(
        default="corrected_history",
        description="Value source used by the z-score calculation.",
    )

    minimum_periods: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Minimum previous observations required. None requires a complete "
            "window."
        ),
    )

    std_ddof: Literal[0, 1] = Field(
        default=1,
        description="Sample or population standard deviation selection.",
    )

    zero_std_strategy: Literal["nan"] = Field(
        default="nan",
        description="Return NaN when rolling standard deviation is zero.",
    )

    @field_validator("windows")
    @classmethod
    def validate_windows(
        cls,
        values: list[int],
    ) -> list[int]:
        """Validate z-score windows."""

        return _validate_unique_positive_integers(
            values,
            field_name="windows",
        )

    @model_validator(mode="after")
    def validate_minimum_periods(
        self,
    ) -> RollingZScoreFeatureRequest:
        """Ensure minimum_periods fits the smallest selected window."""

        if (
            self.minimum_periods is not None
            and self.minimum_periods > min(self.windows)
        ):
            raise ValueError(
                "minimum_periods cannot exceed the smallest z-score window."
            )

        return self


class EWMAFeatureRequest(FeatureRequestBase):
    """Request exponentially weighted moving-average features."""

    type: Literal["ewma"]

    spans: list[int] = Field(
        min_length=1,
        description="Positive EWMA spans controlling recent-value weighting.",
    )

    source: FeatureSource = Field(
        default="corrected_history",
        description="Value source used by the EWMA calculation.",
    )

    adjust: Literal[False] = Field(
        default=False,
        description=(
            "Use the recursive time-series EWMA formula rather than adjusted "
            "full-history weights."
        ),
    )

    @field_validator("spans")
    @classmethod
    def validate_spans(
        cls,
        values: list[int],
    ) -> list[int]:
        """Validate EWMA spans."""

        return _validate_unique_positive_integers(
            values,
            field_name="spans",
        )


class RollingSlopeFeatureRequest(FeatureRequestBase):
    """Request ordinary least-squares slope over previous rolling windows."""

    type: Literal["rolling_slope"]

    windows: list[int] = Field(
        min_length=1,
        description="Previous-value windows used for linear trend slopes.",
    )

    source: FeatureSource = Field(
        default="corrected_history",
        description="Value source used by the rolling slope.",
    )

    minimum_periods: int | None = Field(
        default=None,
        ge=2,
        description=(
            "Minimum previous observations required. None requires a complete "
            "window."
        ),
    )

    @field_validator("windows")
    @classmethod
    def validate_windows(
        cls,
        values: list[int],
    ) -> list[int]:
        """Validate slope windows."""

        values = _validate_unique_positive_integers(
            values,
            field_name="windows",
        )

        if any(value < 2 for value in values):
            raise ValueError(
                "Every rolling-slope window must contain at least two points."
            )

        return values

    @model_validator(mode="after")
    def validate_minimum_periods(
        self,
    ) -> RollingSlopeFeatureRequest:
        """Ensure minimum_periods fits the smallest selected slope window."""

        if (
            self.minimum_periods is not None
            and self.minimum_periods > min(self.windows)
        ):
            raise ValueError(
                "minimum_periods cannot exceed the smallest slope window."
            )

        return self


class WeightedSeasonalBaselineConfig(BaseModel):
    """
    Define the first supported baseline method.

    The implementation can later dispatch on ``method`` to add alternative
    baseline algorithms without changing residual requests or the runner API.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    method: BaselineMethod = Field(
        default=DEFAULT_BASELINE_METHOD,
        description="Baseline method selected for generation.",
    )

    source: FeatureSource = Field(
        default=DEFAULT_BASELINE_SOURCE,
        description="Source from which accepted baseline anchors are selected.",
    )

    anchor_statuses: set[int] = Field(
        default_factory=lambda: set(DEFAULT_BASELINE_ANCHOR_STATUSES),
        description="Statuses permitted to supply seasonal anchor values.",
    )

    recent_fallback_days: int = Field(
        default=DEFAULT_BASELINE_RECENT_FALLBACK_DAYS,
        gt=0,
        description=(
            "Maximum earlier matching weekday/weekend days searched for the "
            "recent anchor."
        ),
    )

    weekly_fallback_weeks: int = Field(
        default=DEFAULT_BASELINE_WEEKLY_FALLBACK_WEEKS,
        gt=0,
        description="Maximum earlier weekly anchors searched.",
    )

    recent_weight: float = Field(
        default=DEFAULT_BASELINE_RECENT_WEIGHT,
        ge=0.0,
        le=1.0,
        description="Weight assigned when both recent and weekly anchors exist.",
    )

    weekly_weight: float = Field(
        default=DEFAULT_BASELINE_WEEKLY_WEIGHT,
        ge=0.0,
        le=1.0,
        description="Weight assigned when both recent and weekly anchors exist.",
    )

    anchor_window_each_side_points: int = Field(
        default=DEFAULT_BASELINE_ANCHOR_WINDOW_EACH_SIDE_POINTS,
        ge=0,
        description=(
            "Number of neighbouring points on each side used to form a robust "
            "local median around every historical anchor. A positive value is "
            "supported only when baseline source='raw_value'."
        ),
    )

    slot_fallback_points: int = Field(
        default=DEFAULT_BASELINE_SLOT_FALLBACK_POINTS,
        gt=0,
        description=(
            "Maximum earlier accepted values used by the same-time-slot median "
            "fallback."
        ),
    )

    slot_tolerance_minutes: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Time-of-day tolerance around the target slot. None resolves to "
            "one dataset bucket on each side."
        ),
    )

    minimum_history_days: int = Field(
        default=DEFAULT_BASELINE_MINIMUM_HISTORY_DAYS,
        gt=0,
        description=(
            "Warm-up history required before baseline and residual values are "
            "expected to begin."
        ),
    )

    @model_validator(mode="after")
    def validate_baseline_settings(
        self,
    ) -> WeightedSeasonalBaselineConfig:
        """Validate baseline anchors and weight normalization."""

        if not self.anchor_statuses:
            raise ValueError(
                "anchor_statuses cannot be empty."
            )

        if (
            self.source == "corrected_history"
            and self.anchor_window_each_side_points > 0
        ):
            raise ValueError(
                "anchor_window_each_side_points must be 0 when "
                "source='corrected_history'. A positive anchor window applies "
                "local median cleaning, which cannot be combined with "
                "corrected_history."
            )

        if not isclose(
            self.recent_weight + self.weekly_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "recent_weight and weekly_weight must sum to exactly 1.0."
            )

        return self

    def resolve_slot_tolerance_points(
        self,
        *,
        bucket_minutes: int,
    ) -> int:
        """Resolve the same-time-slot tolerance to timeline points."""

        tolerance_minutes = (
            self.slot_tolerance_minutes
            if self.slot_tolerance_minutes is not None
            else bucket_minutes
        )

        return _duration_to_points(
            duration_minutes=tolerance_minutes,
            bucket_minutes=bucket_minutes,
            field_name="slot_tolerance_minutes",
        )


class BaselineFeatureRequest(FeatureRequestBase):
    """Request the configured baseline as a standalone feature."""

    type: Literal["baseline"]

    baseline: WeightedSeasonalBaselineConfig = Field(
        default_factory=WeightedSeasonalBaselineConfig,
        description="Settings for the selected baseline method.",
    )


class ResidualFeatureRequest(FeatureRequestBase):
    """
    Request residual generation and automatically produce its baseline.

    Formula::

        residual = selected current value - generated baseline
    """

    type: Literal["residual"]

    source: FeatureSource = Field(
        default="raw_value",
        description="Current-value source used as the residual numerator.",
    )

    baseline: WeightedSeasonalBaselineConfig = Field(
        default_factory=WeightedSeasonalBaselineConfig,
        description=(
            "Baseline settings automatically executed before residual "
            "calculation."
        ),
    )

    include_baseline_column: Literal[True] = Field(
        default=True,
        description=(
            "Retain the automatically generated baseline column alongside the "
            "residual."
        ),
    )


FeatureRequest: TypeAlias = Annotated[
    CurrentValueFeatureRequest
    | LagFeatureRequest
    | RollingFeatureRequest
    | TimeBucketFeatureRequest
    | TimeCyclicFeatureRequest
    | CalendarFeatureRequest
    | DifferenceFeatureRequest
    | PercentageChangeFeatureRequest
    | SeasonalDifferenceFeatureRequest
    | RollingZScoreFeatureRequest
    | EWMAFeatureRequest
    | RollingSlopeFeatureRequest
    | BaselineFeatureRequest
    | ResidualFeatureRequest,
    Field(discriminator="type"),
]


_FEATURE_REQUEST_ADAPTER = TypeAdapter(FeatureRequest)


# Feature families that may intentionally be requested more than once.
# Repeated canonical output names from these request types are renamed with
# deterministic _s2, _s3, ... suffixes during bucket resolution.
_REPEATABLE_FEATURE_REQUEST_TYPES = (
    LagFeatureRequest,
    RollingFeatureRequest,
    DifferenceFeatureRequest,
    PercentageChangeFeatureRequest,
    SeasonalDifferenceFeatureRequest,
    RollingZScoreFeatureRequest,
    EWMAFeatureRequest,
    RollingSlopeFeatureRequest,
)


def _is_repeatable_feature_request(
    request: FeatureRequest,
) -> bool:
    """Return whether one request type may intentionally repeat outputs."""

    return isinstance(
        request,
        _REPEATABLE_FEATURE_REQUEST_TYPES,
    )


def _resolve_feature_column_groups(
    features: list[FeatureRequest],
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Resolve globally unique output names while preserving request identity.

    The first occurrence of a canonical feature keeps its normal schema name.
    Later occurrences of the same canonical output are allowed only when both
    requests belong to the same explicitly repeatable feature family. They are
    renamed deterministically using ``_s2``, ``_s3``, and so on.

    Returns:
        tuple:
            A flattened tuple of all resolved feature columns followed by one
            resolved output-column tuple per request, in request order.

    Raises:
        ValueError:
            If a non-repeatable output collides, or if two different request
            families attempt to own the same canonical output name.
    """

    occurrence_count: dict[str, int] = {}
    first_owner_type: dict[str, str] = {}
    first_owner_repeatable: dict[str, bool] = {}
    resolved_columns: list[str] = []
    resolved_groups: list[tuple[str, ...]] = []

    for request in features:
        request_type = str(
            getattr(request, "type", type(request).__name__)
        )
        request_repeatable = _is_repeatable_feature_request(request)
        request_columns: list[str] = []

        for canonical_column in get_feature_columns(request):
            count = occurrence_count.get(canonical_column, 0) + 1

            if count == 1:
                resolved_name = canonical_column
                first_owner_type[canonical_column] = request_type
                first_owner_repeatable[canonical_column] = request_repeatable
            else:
                original_type = first_owner_type[canonical_column]
                original_repeatable = first_owner_repeatable[canonical_column]

                if (
                    not request_repeatable
                    or not original_repeatable
                    or request_type != original_type
                ):
                    raise ValueError(
                        "Requested features create a disallowed output "
                        f"collision for '{canonical_column}'. Repetition is "
                        "supported only within explicitly repeatable feature "
                        "families of the same type."
                    )

                resolved_name = f"{canonical_column}_s{count}"

            occurrence_count[canonical_column] = count
            request_columns.append(resolved_name)
            resolved_columns.append(resolved_name)

        resolved_groups.append(tuple(request_columns))

    if len(resolved_columns) != len(set(resolved_columns)):
        raise ValueError(
            "Resolved feature naming unexpectedly produced duplicate output "
            "columns."
        )

    return tuple(resolved_columns), tuple(resolved_groups)


# ---------------------------------------------------------------------------
# Resolved configuration and main schema
# ---------------------------------------------------------------------------


class ResolvedFeatureEngineeringConfig(BaseModel):
    """Store bucket-aware settings passed by the runner to feature modules."""

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    bucket_minutes: int = Field(
        gt=0,
    )

    points_per_day: int = Field(
        gt=0,
    )

    corrected_history: CorrectedHistoryConfig

    corrected_history_window_points: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Resolved trailing-median window. None when no requested feature "
            "uses corrected_history."
        ),
    )

    features: tuple[FeatureRequest, ...] = Field(
        min_length=1,
    )

    feature_columns: tuple[str, ...] = Field(
        min_length=1,
    )

    feature_column_groups: tuple[tuple[str, ...], ...] = Field(
        min_length=1,
        description=(
            "Resolved output names for each request in features, preserving "
            "request order. Repeated allowed outputs use _s2, _s3, and so on."
        ),
    )

    @model_validator(mode="after")
    def validate_feature_column_groups(
        self,
    ) -> ResolvedFeatureEngineeringConfig:
        """Require one resolved output-name group per feature request."""

        if len(self.feature_column_groups) != len(self.features):
            raise ValueError(
                "feature_column_groups must contain exactly one entry for "
                "every resolved feature request."
            )

        flattened = tuple(
            column
            for group in self.feature_column_groups
            for column in group
        )

        if flattened != self.feature_columns:
            raise ValueError(
                "Flattened feature_column_groups must exactly match "
                "feature_columns in request order."
            )

        for request, group in zip(
            self.features,
            self.feature_column_groups,
            strict=True,
        ):
            if len(group) != len(get_feature_columns(request)):
                raise ValueError(
                    "Each feature_column_groups entry must contain the same "
                    "number of columns produced by its feature request."
                )

        return self

    minimum_history_points: int = Field(
        gt=0,
    )

    maximum_lookback_points: int = Field(
        ge=0,
    )

    uses_corrected_history: bool
    uses_local_median_cleaning: bool
    requires_status_column: bool
    retain_original_columns: bool
    save_intermediate_outputs: bool


def get_feature_columns(
    request: FeatureRequest,
) -> tuple[str, ...]:
    """Return deterministic output names created by one request."""

    if isinstance(request, CurrentValueFeatureRequest):
        return ("current_value",)

    if isinstance(request, LagFeatureRequest):
        return tuple(
            f"lag_{lag}"
            for lag in request.lags
        )

    if isinstance(request, RollingFeatureRequest):
        return tuple(
            f"rolling_{statistic}_{window}"
            for window in request.windows
            for statistic in request.statistics
        )

    if isinstance(request, TimeBucketFeatureRequest):
        return tuple(
            f"time_bucket_{minutes}m"
            for minutes in request.bucket_minutes
        )

    if isinstance(request, TimeCyclicFeatureRequest):
        return tuple(request.features)

    if isinstance(request, CalendarFeatureRequest):
        return tuple(request.features)

    if isinstance(request, DifferenceFeatureRequest):
        return tuple(
            f"difference_{lag}"
            for lag in request.lags
        )

    if isinstance(request, PercentageChangeFeatureRequest):
        return tuple(
            f"percentage_change_{lag}"
            for lag in request.lags
        )

    if isinstance(request, SeasonalDifferenceFeatureRequest):
        return tuple(
            f"seasonal_difference_{lag}"
            for lag in request.lags
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

    if isinstance(request, BaselineFeatureRequest):
        return ("baseline",)

    if isinstance(request, ResidualFeatureRequest):
        return ("baseline", "residual")

    raise TypeError(
        f"Unsupported feature request: {type(request).__name__}"
    )


def get_request_history_requirements(
    request: FeatureRequest,
    *,
    points_per_day: int,
) -> tuple[int, int]:
    """
    Return minimum history and maximum practical lookback for one request.

    The first value is the minimum row count expected before the feature can
    become usable. The second is the furthest configured historical reference.
    """

    if isinstance(
        request,
        (
            CurrentValueFeatureRequest,
            TimeBucketFeatureRequest,
            TimeCyclicFeatureRequest,
            CalendarFeatureRequest,
        ),
    ):
        return 1, 0

    if isinstance(
        request,
        (
            LagFeatureRequest,
            DifferenceFeatureRequest,
            PercentageChangeFeatureRequest,
            SeasonalDifferenceFeatureRequest,
        ),
    ):
        maximum_lag = max(request.lags)
        return maximum_lag + 1, maximum_lag

    if isinstance(request, RollingFeatureRequest):
        maximum_window = max(request.windows)
        return maximum_window + 1, maximum_window

    if isinstance(request, RollingZScoreFeatureRequest):
        maximum_window = max(request.windows)
        return maximum_window + 1, maximum_window

    if isinstance(request, RollingSlopeFeatureRequest):
        maximum_window = max(request.windows)
        return maximum_window + 1, maximum_window

    if isinstance(request, EWMAFeatureRequest):
        maximum_span = max(request.spans)

        # One previous observation is required because EWMA uses source.shift(1).
        # Therefore, the earliest possible usable output is the second row.
        return 2, maximum_span
    
    if isinstance(request, BaselineFeatureRequest):
        baseline = request.baseline

    elif isinstance(request, ResidualFeatureRequest):
        baseline = request.baseline

    else:
        raise TypeError(
            f"Unsupported feature request: {type(request).__name__}"
        )

    minimum_history = baseline.minimum_history_days * points_per_day
    maximum_seasonal_days = max(
        baseline.recent_fallback_days,
        baseline.weekly_fallback_weeks * 7,
    )
    maximum_lookback = (
        maximum_seasonal_days * points_per_day
        + baseline.anchor_window_each_side_points
    )

    return minimum_history, maximum_lookback


class FeatureEngineeringConfig(BaseModel):
    """
    Define the complete Part C feature-engineering request.

    ``features=None`` selects the schema-owned default profile. Supplying a
    non-empty list generates exactly the requested features after validation.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    features: list[FeatureRequest] | None = Field(
        default=None,
        description=(
            "Structured requested features. None selects default_features. "
            "An explicit empty list is rejected."
        ),
    )

    default_profile: DefaultFeatureProfileName = Field(
        default="xgboost",
        description=(
            "Algorithm-specific default profile used only when features is "
            "omitted."
        ),
    )

    default_features: DefaultFeatureProfilesConfig = Field(
        default_factory=DefaultFeatureProfilesConfig,
        description=(
            "Schema-owned XGBoost and Isolation Forest defaults used only "
            "when features is omitted."
        ),
    )

    corrected_history: CorrectedHistoryConfig = Field(
        default_factory=CorrectedHistoryConfig,
        description=(
            "Shared temporary-history cleaning settings used by features that "
            "select corrected_history."
        ),
    )

    retain_original_columns: bool = Field(
        default=DEFAULT_RETAIN_ORIGINAL_COLUMNS,
        description=(
            "Retain all Part B columns and append generated feature columns."
        ),
    )

    save_intermediate_outputs: bool = Field(
        default=DEFAULT_SAVE_INTERMEDIATE_OUTPUTS,
        description=(
            "Allow the runner to save optional corrected-history or feature "
            "audit outputs."
        ),
    )

    @model_validator(mode="after")
    def validate_custom_features(
        self,
    ) -> FeatureEngineeringConfig:
        """Reject empty requests and disallowed output collisions.

        Selected lag/comparison/rolling feature families may intentionally
        repeat. Their final ``_s2``/``_s3`` names are resolved later, after the
        dataset bucket is known. Non-repeatable collisions remain invalid.
        """

        if self.features is None:
            return self

        if not self.features:
            raise ValueError(
                "features cannot be an empty list. Omit the field to use the "
                "default profile."
            )

        # Run the same deterministic naming policy used during resolution so
        # invalid collisions fail immediately during request validation.
        _resolve_feature_column_groups(
            list(self.features)
        )

        return self

    def resolve_for_bucket(
        self,
        bucket: BucketConfig,
    ) -> ResolvedFeatureEngineeringConfig:
        """
        Resolve defaults and history requirements for one dataset frequency.

        The workflow or runner should call this once after Part A has resolved
        the bucket and then pass the result to the individual feature modules.
        """

        bucket_minutes = _bucket_to_minutes(
            bucket
        )
        points_per_day = 1_440 // bucket_minutes

        corrected_history_window_points: int | None = None

        if self.features is not None:
            effective_features = list(self.features)
        else:
            default_payloads = self.default_features.build_request_payloads(
                profile=self.default_profile,
                bucket_minutes=bucket_minutes,
                points_per_day=points_per_day,
            )
            effective_features = [
                _FEATURE_REQUEST_ADAPTER.validate_python(payload)
                for payload in default_payloads
            ]

        if not effective_features:
            raise ValueError(
                "Feature resolution produced no requested features."
            )

        feature_columns, feature_column_groups = (
            _resolve_feature_column_groups(
                effective_features
            )
        )
        minimum_history_points = 1
        maximum_lookback_points = 0
        uses_corrected_history = False
        uses_local_median_cleaning = False
        requires_status_column = False

        for request in effective_features:
            request_minimum, request_lookback = (
                get_request_history_requirements(
                    request,
                    points_per_day=points_per_day,
                )
            )

            minimum_history_points = max(
                minimum_history_points,
                request_minimum,
            )
            maximum_lookback_points = max(
                maximum_lookback_points,
                request_lookback,
            )

            source_values: list[FeatureSource] = []

            if hasattr(request, "source"):
                source_values.append(request.source)

            if isinstance(request, BaselineFeatureRequest):
                source_values.append(request.baseline.source)
                requires_status_column = True

            if isinstance(request, ResidualFeatureRequest):
                source_values.append(request.baseline.source)
                requires_status_column = True

            if any(
                source == "corrected_history"
                for source in source_values
            ):
                uses_corrected_history = True
                requires_status_column = True

            point_cleaning = getattr(
                request,
                "point_cleaning",
                None,
            )

            if (
                point_cleaning is not None
                and point_cleaning.enabled
            ):
                half_window_points = (
                    point_cleaning.resolve_half_window_points(
                        bucket_minutes=bucket_minutes,
                    )
                )

                request_lags = getattr(
                    request,
                    "lags",
                    None,
                )

                if not request_lags:
                    raise TypeError(
                        "Local median point cleaning is supported only for "
                        "lag-based feature requests that define positive "
                        "lags."
                    )

                # Lag and comparison modules use an adaptive leakage-safe
                # window. For lag L and nominal half-window h, the full left
                # side is retained while the right side is capped so that the
                # current feature row can never enter the median:
                #
                #     safe_right = min(h, L - 1)
                #
                # Validate every requested lag because small lags can expose
                # fewer candidate positions than the nominal 2h + 1 window.
                for lag in request_lags:
                    safe_right_points = min(
                        half_window_points,
                        lag - 1,
                    )
                    available_window_points = (
                        half_window_points
                        + safe_right_points
                        + 1
                    )

                    if (
                        point_cleaning.minimum_valid_points
                        > available_window_points
                    ):
                        raise ValueError(
                            "point_cleaning.minimum_valid_points cannot be "
                            f"satisfied for lag={lag}. The leakage-safe local "
                            "window contains at most "
                            f"{available_window_points} positions: "
                            f"{half_window_points} on the left, the referenced "
                            f"point, and {safe_right_points} on the right."
                        )

                uses_local_median_cleaning = True
                requires_status_column = True

                # The earliest observation required for any requested lag is
                # t - maximum_lag - h. The adaptive right side contains newer
                # historical observations and therefore does not increase the
                # maximum lookback.
                minimum_history_points = max(
                    minimum_history_points,
                    request_lookback + half_window_points + 1,
                )
                maximum_lookback_points = max(
                    maximum_lookback_points,
                    request_lookback + half_window_points,
                )

            if isinstance(request, BaselineFeatureRequest):
                request.baseline.resolve_slot_tolerance_points(
                    bucket_minutes=bucket_minutes,
                )
                if request.baseline.anchor_window_each_side_points > 0:
                    uses_local_median_cleaning = True

            if isinstance(request, ResidualFeatureRequest):
                request.baseline.resolve_slot_tolerance_points(
                    bucket_minutes=bucket_minutes,
                )
                if request.baseline.anchor_window_each_side_points > 0:
                    uses_local_median_cleaning = True

        if uses_corrected_history:
            corrected_history_window_points = (
                self.corrected_history.resolve_window_points(
                    bucket_minutes=bucket_minutes,
                )
            )
            minimum_history_points = max(
                minimum_history_points,
                self.corrected_history.minimum_periods + 1,
            )
            maximum_lookback_points = max(
                maximum_lookback_points,
                corrected_history_window_points,
            )

        return ResolvedFeatureEngineeringConfig(
            bucket_minutes=bucket_minutes,
            points_per_day=points_per_day,
            corrected_history=self.corrected_history,
            corrected_history_window_points=(
                corrected_history_window_points
            ),
            features=tuple(effective_features),
            feature_columns=feature_columns,
            feature_column_groups=feature_column_groups,
            minimum_history_points=minimum_history_points,
            maximum_lookback_points=maximum_lookback_points,
            uses_corrected_history=uses_corrected_history,
            uses_local_median_cleaning=uses_local_median_cleaning,
            requires_status_column=requires_status_column,
            retain_original_columns=self.retain_original_columns,
            save_intermediate_outputs=self.save_intermediate_outputs,
        )


__all__ = [
    "BaselineFeatureRequest",
    "BaselineMethod",
    "CalendarFeature",
    "CalendarFeatureRequest",
    "CorrectedHistoryConfig",
    "CurrentValueFeatureRequest",
    "DefaultFeatureProfileName",
    "DefaultFeatureProfilesConfig",
    "DifferenceFeatureRequest",
    "EWMAFeatureRequest",
    "FeatureEngineeringConfig",
    "FeatureRequest",
    "FeatureSource",
    "LagFeatureRequest",
    "LocalMedianCleaningConfig",
    "PercentageChangeFeatureRequest",
    "ResidualFeatureRequest",
    "ResolvedFeatureEngineeringConfig",
    "RollingFeatureRequest",
    "RollingSlopeFeatureRequest",
    "RollingStatistic",
    "RollingZScoreFeatureRequest",
    "SeasonalDifferenceFeatureRequest",
    "TimeBucketFeatureRequest",
    "TimeCyclicFeature",
    "TimeCyclicFeatureRequest",
    "WeightedSeasonalBaselineConfig",
    "get_feature_columns",
    "get_request_history_requirements",
]