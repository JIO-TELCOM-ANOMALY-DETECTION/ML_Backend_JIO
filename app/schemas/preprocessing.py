"""
Request schemas for Part B telecom preprocessing.

This module defines how a prepared KPI dataset should be processed after
Part A has resolved the source columns, KPI identity, expected timeline, and
bucket configuration.

The shared dataset contract remains in::

    app/schemas/common/dataset.py

This module covers only Part B settings:

- duplicate and off-grid timestamp treatment;
- valid-value cleaning;
- series-usability checks;
- outlier labelling;
- frequency-specific missing-gap profiles;
- missing-value filling phases 1 to 4;
- Prophet controls; and
- intermediate-output control.

Frequency-dependent minute settings are converted to validated point counts
only after Part A returns the resolved ``BucketConfig``. The processing runner
should call ``PreprocessingConfig.resolve_for_bucket`` once and pass the
returned ``ResolvedFrequencyConfig`` to the individual processing stages.
"""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from app.schemas.common.dataset import BucketConfig


OutlierMethod = Literal[
    "global_iqr",
    "one_day_centered_iqr",
    "user_bounds",
]

OutlierDirection = Literal[
    "lower_only",
    "upper_only",
    "both",
]

SeasonalityMode = Literal[
    "additive",
    "multiplicative",
]


class TimelineCorrectionConfig(BaseModel):
    """
    Define fixed timeline-correction behaviour.

    Duplicate rows are identified by the resolved site-identity columns plus
    the timestamp column. The first physical row is retained and later
    duplicates are removed before timeline expansion.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    duplicate_timestamp_strategy: Literal["keep_first"] = Field(
        default="keep_first",
        description=(
            "Keep the first row for a duplicate site-timestamp key and "
            "remove every later duplicate row."
        ),
    )

    off_grid_timestamp_strategy: Literal["reject"] = Field(
        default="reject",
        description=(
            "Reject timestamps that do not lie on the resolved bucket grid."
        ),
    )


class ValidValueConfig(BaseModel):
    """Define valid KPI bounds and boundary-tolerance behaviour."""

    model_config = ConfigDict(
        extra="forbid",
    )

    lower_bound: float = Field(
        default=0.0,
        description="Inclusive lower bound for a physically valid KPI value.",
    )

    upper_bound: float = Field(
        default=100.0,
        description="Inclusive upper bound for a physically valid KPI value.",
    )

    boundary_tolerance: float = Field(
        default=0.001,
        ge=0.0,
        description=(
            "Tolerance outside the valid bounds. Values within this distance "
            "may be clipped back to the nearest boundary."
        ),
    )

    clip_values_within_tolerance: bool = Field(
        default=True,
        description=(
            "Clip slightly out-of-range source values when they remain within "
            "boundary_tolerance."
        ),
    )

    clip_generated_values: bool = Field(
        default=True,
        description=(
            "Clip generated filling values and Prophet predictions to the "
            "configured valid range."
        ),
    )

    @model_validator(mode="after")
    def validate_value_range(
        self,
    ) -> ValidValueConfig:
        """
        Validate the configured physical KPI range.

        Returns:
            ValidValueConfig:
                The validated value configuration.

        Raises:
            ValueError:
                If the upper bound is not greater than the lower bound or the
                tolerance would make the two boundary regions overlap.
        """

        value_span = self.upper_bound - self.lower_bound

        if value_span <= 0:
            raise ValueError(
                "upper_bound must be greater than lower_bound."
            )

        if self.boundary_tolerance * 2 >= value_span:
            raise ValueError(
                "boundary_tolerance is too large for the configured valid "
                "value range."
            )

        return self


class SeriesUsabilityConfig(BaseModel):
    """Define when a site-level KPI series is skipped by filling stages."""

    model_config = ConfigDict(
        extra="forbid",
    )

    maximum_missing_percentage: float = Field(
        default=20.0,
        ge=0.0,
        le=100.0,
        description=(
            "A series is invalid only when the percentage of original labels "
            "0, -1, and -2 is greater than this value. Equality remains valid."
        ),
    )


class OutlierConfig(BaseModel):
    """Define the one outlier-labelling method selected for the run."""

    model_config = ConfigDict(
        extra="forbid",
    )

    method: OutlierMethod = Field(
        description="Outlier method applied independently to each site series.",
    )

    direction: OutlierDirection = Field(
        default="lower_only",
        description=(
            "Detect lower outliers, upper outliers, or both lower and upper "
            "outliers."
        ),
    )

    iqr_multiplier: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "IQR multiplier for global or one-day-centred IQR. Defaults to "
            "1.5 when an IQR method is selected."
        ),
    )

    user_lower_bound: float | None = Field(
        default=None,
        description=(
            "Required user threshold when method is user_bounds and direction "
            "includes lower-side detection."
        ),
    )

    user_upper_bound: float | None = Field(
        default=None,
        description=(
            "Required user threshold when method is user_bounds and direction "
            "includes upper-side detection."
        ),
    )

    centered_window_minutes: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Total duration around a target used by one-day-centred IQR. "
            "Defaults to 1,440 minutes. The resolved point window includes "
            "the centre point."
        ),
    )

    centered_min_valid_minutes: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Valid-observation requirement expressed as an equivalent duration. "
            "Defaults to 720 minutes and is converted to a point count."
        ),
    )

    @model_validator(mode="after")
    def validate_method_fields(
        self,
    ) -> OutlierConfig:
        """
        Validate fields allowed and required by the selected outlier method.

        Returns:
            OutlierConfig:
                The validated and default-resolved outlier configuration.

        Raises:
            ValueError:
                If method-specific fields are missing, contradictory, or
                supplied for an unrelated method.
        """

        if self.method == "user_bounds":
            if self.iqr_multiplier is not None:
                raise ValueError(
                    "iqr_multiplier cannot be supplied for user_bounds."
                )

            if (
                self.centered_window_minutes is not None
                or self.centered_min_valid_minutes is not None
            ):
                raise ValueError(
                    "Centred-window settings cannot be supplied for "
                    "user_bounds."
                )

            if self.direction == "lower_only":
                if self.user_lower_bound is None:
                    raise ValueError(
                        "lower_only user_bounds requires user_lower_bound."
                    )
                if self.user_upper_bound is not None:
                    raise ValueError(
                        "user_upper_bound is not used for lower_only."
                    )

            elif self.direction == "upper_only":
                if self.user_upper_bound is None:
                    raise ValueError(
                        "upper_only user_bounds requires user_upper_bound."
                    )
                if self.user_lower_bound is not None:
                    raise ValueError(
                        "user_lower_bound is not used for upper_only."
                    )

            else:
                if (
                    self.user_lower_bound is None
                    or self.user_upper_bound is None
                ):
                    raise ValueError(
                        "both-direction user_bounds requires both user bounds."
                    )

                if self.user_lower_bound >= self.user_upper_bound:
                    raise ValueError(
                        "user_lower_bound must be less than user_upper_bound."
                    )

            return self

        if (
            self.user_lower_bound is not None
            or self.user_upper_bound is not None
        ):
            raise ValueError(
                "User bounds can be supplied only when method is user_bounds."
            )

        if self.iqr_multiplier is None:
            self.iqr_multiplier = 1.5

        if self.method == "global_iqr":
            if (
                self.centered_window_minutes is not None
                or self.centered_min_valid_minutes is not None
            ):
                raise ValueError(
                    "Centred-window settings can be supplied only for "
                    "one_day_centered_iqr."
                )

            return self

        if self.centered_window_minutes is None:
            self.centered_window_minutes = 1_440

        if self.centered_min_valid_minutes is None:
            self.centered_min_valid_minutes = 720

        if self.centered_min_valid_minutes > self.centered_window_minutes:
            raise ValueError(
                "centered_min_valid_minutes cannot exceed "
                "centered_window_minutes."
            )

        return self


class GapRangeConfig(BaseModel):
    """Define one inclusive original-gap point range."""

    model_config = ConfigDict(
        extra="forbid",
    )

    minimum_points: int = Field(
        ge=1,
        description="First original gap size included in this phase.",
    )

    maximum_points: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Last original gap size included in this phase. None means no "
            "upper limit."
        ),
    )

    @model_validator(mode="after")
    def validate_point_range(
        self,
    ) -> GapRangeConfig:
        """
        Validate one inclusive point range.

        Returns:
            GapRangeConfig:
                The validated gap range.

        Raises:
            ValueError:
                If the maximum is lower than the minimum.
        """

        if (
            self.maximum_points is not None
            and self.maximum_points < self.minimum_points
        ):
            raise ValueError(
                "maximum_points cannot be lower than minimum_points."
            )

        return self


class GapProfileConfig(BaseModel):
    """Define complete non-overlapping original-gap ranges for all phases."""

    model_config = ConfigDict(
        extra="forbid",
    )

    phase_1: GapRangeConfig
    phase_2: GapRangeConfig
    phase_3: GapRangeConfig
    phase_4_pass_1: GapRangeConfig
    phase_4_pass_2: GapRangeConfig

    @model_validator(mode="after")
    def validate_complete_profile(
        self,
    ) -> GapProfileConfig:
        """
        Validate that every positive gap size belongs to exactly one phase.

        Returns:
            GapProfileConfig:
                The validated complete gap profile.

        Raises:
            ValueError:
                If ranges overlap, leave holes, begin after point one, or do
                not end with an open-ended Phase 4 Pass 2 range.
        """

        ranges = [
            self.phase_1,
            self.phase_2,
            self.phase_3,
            self.phase_4_pass_1,
            self.phase_4_pass_2,
        ]

        if ranges[0].minimum_points != 1:
            raise ValueError(
                "The gap profile must begin at one point."
            )

        for current_range, next_range in zip(
            ranges[:-1],
            ranges[1:],
            strict=True,
        ):
            if current_range.maximum_points is None:
                raise ValueError(
                    "Only phase_4_pass_2 may be open-ended."
                )

            expected_next_start = current_range.maximum_points + 1

            if next_range.minimum_points != expected_next_start:
                raise ValueError(
                    "Gap ranges must be contiguous and non-overlapping. "
                    f"Expected next minimum {expected_next_start}, received "
                    f"{next_range.minimum_points}."
                )

        if self.phase_4_pass_2.maximum_points is not None:
            raise ValueError(
                "phase_4_pass_2 must be open-ended."
            )

        return self


class Phase1Config(BaseModel):
    """Define bounded local interpolation and anchor-search settings."""

    model_config = ConfigDict(
        extra="forbid",
    )

    maximum_anchor_search_minutes: int = Field(
        default=60,
        gt=0,
        description=(
            "Maximum distance searched in either direction for a trusted "
            "anchor. Search may pass outliers but may not exceed this limit."
        ),
    )

    maximum_two_sided_distance_minutes: int = Field(
        default=30,
        gt=0,
        description=(
            "Maximum distance allowed for each anchor in the stricter "
            "two-sided weighted fallback."
        ),
    )

    maximum_anchor_distance_ratio: float = Field(
        default=2.0,
        ge=1.0,
        description=(
            "Largest accepted ratio between the farther and nearer two-sided "
            "anchor distances."
        ),
    )

    @model_validator(mode="after")
    def validate_search_distances(
        self,
    ) -> Phase1Config:
        """
        Validate the relationship between Phase 1 search distances.

        Returns:
            Phase1Config:
                The validated Phase 1 configuration.

        Raises:
            ValueError:
                If the stricter two-sided distance exceeds the overall search
                limit.
        """

        if (
            self.maximum_two_sided_distance_minutes
            > self.maximum_anchor_search_minutes
        ):
            raise ValueError(
                "maximum_two_sided_distance_minutes cannot exceed "
                "maximum_anchor_search_minutes."
            )

        return self


class Phase2Config(BaseModel):
    """Define recent same-type and last-week seasonal filling settings."""

    model_config = ConfigDict(
        extra="forbid",
    )

    recent_same_type_lookback_days: int = Field(
        default=14,
        gt=0,
        description=(
            "Maximum calendar days searched backward for the nearest earlier "
            "weekday/weekend-matching timestamp."
        ),
    )

    last_week_lookback_days: int = Field(
        default=7,
        gt=0,
        description="Calendar-day offset used for the last-week reference.",
    )

    recent_weight: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Weight applied to the recent same-type reference.",
    )

    last_week_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight applied to the exact last-week reference.",
    )

    neighbor_offset_minutes: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Offset on each side of a seasonal timestamp. None means exactly "
            "one resolved bucket step."
        ),
    )

    @model_validator(mode="after")
    def validate_weights(
        self,
    ) -> Phase2Config:
        """
        Validate that Phase 2 seasonal weights form one complete mixture.

        Returns:
            Phase2Config:
                The validated Phase 2 configuration.

        Raises:
            ValueError:
                If the two weights do not sum to one.
        """

        if abs(
            self.recent_weight
            + self.last_week_weight
            - 1.0
        ) > 1e-9:
            raise ValueError(
                "recent_weight and last_week_weight must sum to 1.0."
            )

        return self


class Phase3Config(BaseModel):
    """Define multi-day seasonal-median filling settings."""

    model_config = ConfigDict(
        extra="forbid",
    )

    maximum_same_type_days: int = Field(
        default=4,
        gt=0,
        description=(
            "Maximum previous weekday/weekend-matching days used to build a "
            "seasonal reference pool."
        ),
    )

    maximum_lookback_days: int = Field(
        default=30,
        gt=0,
        description="Maximum calendar days searched into the past.",
    )

    level_1_window_minutes: int = Field(
        default=15,
        gt=0,
        description="Half-window duration used by Phase 3 Level 1.",
    )

    level_2_window_minutes: int = Field(
        default=30,
        gt=0,
        description="Relaxed half-window duration used by Phase 3 Level 2.",
    )

    level_1_min_clean_values: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Minimum clean values required by Level 1. Defaults to 5 for a "
            "5-minute bucket and 3 for a 15-minute bucket."
        ),
    )

    level_2_min_clean_values: int = Field(
        default=2,
        gt=0,
        description="Minimum pooled clean values required by Level 2.",
    )

    level_4_min_clean_values: int = Field(
        default=2,
        gt=0,
        description=(
            "Minimum clean values required from the first usable previous day "
            "of any type."
        ),
    )

    @model_validator(mode="after")
    def validate_phase_3_ranges(
        self,
    ) -> Phase3Config:
        """
        Validate Phase 3 lookback and window relationships.

        Returns:
            Phase3Config:
                The validated Phase 3 configuration.

        Raises:
            ValueError:
                If the relaxed window is narrower than Level 1 or fewer days
                are searched than the number of requested same-type days.
        """

        if self.level_2_window_minutes < self.level_1_window_minutes:
            raise ValueError(
                "level_2_window_minutes cannot be lower than "
                "level_1_window_minutes."
            )

        if self.maximum_lookback_days < self.maximum_same_type_days:
            raise ValueError(
                "maximum_lookback_days cannot be lower than "
                "maximum_same_type_days."
            )

        return self


class Phase4ProphetConfig(BaseModel):
    """Define optional two-pass Prophet filling and model parameters."""

    model_config = ConfigDict(
        extra="forbid",
    )

    enabled: bool = Field(
        default=True,
        description="Run both ordered Phase 4 Prophet passes.",
    )

    allow_full_series_fallback: bool = Field(
        default=True,
        description=(
            "Permit a full-series fallback when a past-only Prophet model is "
            "unavailable or fails. These fills receive status label 4."
        ),
    )

    minimum_history_days: int = Field(
        default=14,
        gt=0,
        description=(
            "Minimum trusted timestamp span required for either Prophet mode."
        ),
    )

    minimum_history_points: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Minimum trusted observations required for Prophet. None derives "
            "the value from minimum_history_days and the resolved bucket."
        ),
    )

    changepoint_prior_scale: float = Field(
        default=0.3,
        gt=0.0,
        description="Prophet trend-flexibility parameter.",
    )

    seasonality_prior_scale: float = Field(
        default=20.0,
        gt=0.0,
        description="Prophet seasonality-strength parameter.",
    )

    seasonality_mode: SeasonalityMode = Field(
        default="additive",
        description="Prophet additive or multiplicative seasonality mode.",
    )

    daily_seasonality: bool = Field(
        default=True,
        description="Enable Prophet daily seasonality.",
    )

    weekly_seasonality: bool = Field(
        default=True,
        description="Enable Prophet weekly seasonality.",
    )

    yearly_seasonality: bool = Field(
        default=False,
        description="Enable Prophet yearly seasonality.",
    )

    interval_width: float = Field(
        default=0.95,
        gt=0.0,
        lt=1.0,
        description="Prophet uncertainty interval width.",
    )

    noise_scale: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Multiplier applied to the fitted-model residual standard "
            "deviation before prediction noise is added."
        ),
    )

    random_seed: int = Field(
        default=42,
        ge=0,
        description="Base seed used to derive stable per-gap random seeds.",
    )


class ResolvedFrequencyConfig(BaseModel):
    """Store validated point counts derived from the resolved bucket."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    bucket_minutes: int
    gap_profile: GapProfileConfig

    centered_window_points: int | None = None
    centered_min_valid_points: int | None = None

    phase_1_maximum_anchor_search_points: int
    phase_1_maximum_two_sided_distance_points: int

    phase_2_neighbor_offset_points: int

    phase_3_level_1_half_window_points: int
    phase_3_level_2_half_window_points: int
    phase_3_level_1_min_clean_values: int

    points_per_day: int | None = None
    prophet_minimum_history_points: int | None = None


class PreprocessingConfig(BaseModel):
    """Define the complete Part B preprocessing request configuration."""

    model_config = ConfigDict(
        extra="forbid",
    )

    timeline_correction: TimelineCorrectionConfig = Field(
        default_factory=TimelineCorrectionConfig,
    )

    valid_values: ValidValueConfig = Field(
        default_factory=ValidValueConfig,
    )

    series_usability: SeriesUsabilityConfig = Field(
        default_factory=SeriesUsabilityConfig,
    )

    outlier: OutlierConfig

    gap_profile: GapProfileConfig | None = Field(
        default=None,
        description=(
            "Optional complete custom gap profile. When omitted, built-in "
            "profiles are used for 5-minute and 15-minute buckets. Other "
            "bucket sizes require this field."
        ),
    )

    phase_1: Phase1Config = Field(
        default_factory=Phase1Config,
    )

    phase_2: Phase2Config = Field(
        default_factory=Phase2Config,
    )

    phase_3: Phase3Config = Field(
        default_factory=Phase3Config,
    )

    phase_4: Phase4ProphetConfig = Field(
        default_factory=Phase4ProphetConfig,
    )

    save_intermediate_outputs: bool = Field(
        default=False,
        description=(
            "Write selected processing checkpoints in addition to the final "
            "cleaned dataset."
        ),
    )

    @model_validator(mode="after")
    def validate_cross_section_settings(
        self,
    ) -> PreprocessingConfig:
        """
        Validate relationships between value cleaning and user outlier bounds.

        Returns:
            PreprocessingConfig:
                The validated preprocessing configuration.

        Raises:
            ValueError:
                If a supplied user outlier threshold lies outside the valid KPI
                range.
        """

        if self.outlier.method != "user_bounds":
            return self

        lower = self.outlier.user_lower_bound
        upper = self.outlier.user_upper_bound
        valid_lower = self.valid_values.lower_bound
        valid_upper = self.valid_values.upper_bound

        if lower is not None and not valid_lower <= lower <= valid_upper:
            raise ValueError(
                "user_lower_bound must lie inside the configured valid "
                "value range."
            )

        if upper is not None and not valid_lower <= upper <= valid_upper:
            raise ValueError(
                "user_upper_bound must lie inside the configured valid "
                "value range."
            )

        return self

    def resolve_for_bucket(
        self,
        bucket: BucketConfig,
    ) -> ResolvedFrequencyConfig:
        """
        Resolve and validate every frequency-dependent preprocessing setting.

        This method should be called after data preparation has returned one
        uniform ``BucketConfig`` for the KPI dataset.

        Args:
            bucket:
                Resolved bucket configuration returned by Part A preparation.

        Returns:
            ResolvedFrequencyConfig:
                Validated point counts and the selected gap profile.

        Raises:
            ValueError:
                If the bucket cannot be expressed as whole minutes, a custom
                profile is missing, a duration is incompatible with the bucket,
                or a clean-reference requirement exceeds its possible pool.
        """

        bucket_minutes = _bucket_to_whole_minutes(
            bucket=bucket,
        )

        gap_profile = self.gap_profile

        if gap_profile is None:
            if bucket_minutes == 5:
                gap_profile = build_5_minute_gap_profile()
            elif bucket_minutes == 15:
                gap_profile = build_15_minute_gap_profile()
            else:
                raise ValueError(
                    "A complete custom gap_profile is required when the "
                    "resolved bucket is not 5 or 15 minutes."
                )

        centered_window_points: int | None = None
        centered_min_valid_points: int | None = None

        if self.outlier.method == "one_day_centered_iqr":
            if self.outlier.centered_window_minutes is None:
                raise ValueError(
                    "centered_window_minutes was not resolved."
                )

            if self.outlier.centered_min_valid_minutes is None:
                raise ValueError(
                    "centered_min_valid_minutes was not resolved."
                )

            centered_span_points = _minutes_to_points(
                minutes=self.outlier.centered_window_minutes,
                bucket_minutes=bucket_minutes,
                setting_name="centered_window_minutes",
            )

            centered_window_points = centered_span_points + 1

            centered_min_valid_points = _minutes_to_points(
                minutes=self.outlier.centered_min_valid_minutes,
                bucket_minutes=bucket_minutes,
                setting_name="centered_min_valid_minutes",
            )

            if centered_min_valid_points > centered_window_points:
                raise ValueError(
                    "Resolved centred-IQR minimum valid points cannot exceed "
                    "the resolved rolling window."
                )

        phase_1_search_points = _minutes_to_points(
            minutes=self.phase_1.maximum_anchor_search_minutes,
            bucket_minutes=bucket_minutes,
            setting_name="maximum_anchor_search_minutes",
        )

        phase_1_two_sided_points = _minutes_to_points(
            minutes=self.phase_1.maximum_two_sided_distance_minutes,
            bucket_minutes=bucket_minutes,
            setting_name="maximum_two_sided_distance_minutes",
        )

        neighbor_minutes = (
            self.phase_2.neighbor_offset_minutes
            if self.phase_2.neighbor_offset_minutes is not None
            else bucket_minutes
        )

        phase_2_neighbor_points = _minutes_to_points(
            minutes=neighbor_minutes,
            bucket_minutes=bucket_minutes,
            setting_name="neighbor_offset_minutes",
        )

        phase_3_level_1_points = _minutes_to_points(
            minutes=self.phase_3.level_1_window_minutes,
            bucket_minutes=bucket_minutes,
            setting_name="level_1_window_minutes",
        )

        phase_3_level_2_points = _minutes_to_points(
            minutes=self.phase_3.level_2_window_minutes,
            bucket_minutes=bucket_minutes,
            setting_name="level_2_window_minutes",
        )

        level_1_min_clean = self.phase_3.level_1_min_clean_values

        if level_1_min_clean is None:
            if bucket_minutes == 5:
                level_1_min_clean = 5
            elif bucket_minutes == 15:
                level_1_min_clean = 3
            else:
                raise ValueError(
                    "phase_3.level_1_min_clean_values is required when the "
                    "resolved bucket is not 5 or 15 minutes."
                )

        level_1_pool_size = (
            self.phase_3.maximum_same_type_days
            * (phase_3_level_1_points * 2 + 1)
        )

        level_2_pool_size = (
            self.phase_3.maximum_same_type_days
            * (phase_3_level_2_points * 2 + 1)
        )

        level_4_pool_size = phase_3_level_1_points * 2 + 1

        if level_1_min_clean > level_1_pool_size:
            raise ValueError(
                "phase_3.level_1_min_clean_values exceeds the maximum "
                "possible Level 1 reference pool."
            )

        if self.phase_3.level_2_min_clean_values > level_2_pool_size:
            raise ValueError(
                "phase_3.level_2_min_clean_values exceeds the maximum "
                "possible Level 2 reference pool."
            )

        if self.phase_3.level_4_min_clean_values > level_4_pool_size:
            raise ValueError(
                "phase_3.level_4_min_clean_values exceeds the maximum "
                "possible Level 4 reference window."
            )

        points_per_day: int | None = None

        if 1_440 % bucket_minutes == 0:
            points_per_day = 1_440 // bucket_minutes

        prophet_history_points: int | None = None

        if self.phase_4.enabled:
            prophet_history_points = self.phase_4.minimum_history_points

            if prophet_history_points is None:
                if points_per_day is None:
                    raise ValueError(
                        "phase_4.minimum_history_points is required when the "
                        "resolved bucket does not divide evenly into one day."
                    )

                prophet_history_points = (
                    self.phase_4.minimum_history_days
                    * points_per_day
                )

        return ResolvedFrequencyConfig(
            bucket_minutes=bucket_minutes,
            gap_profile=gap_profile,
            centered_window_points=centered_window_points,
            centered_min_valid_points=centered_min_valid_points,
            phase_1_maximum_anchor_search_points=(
                phase_1_search_points
            ),
            phase_1_maximum_two_sided_distance_points=(
                phase_1_two_sided_points
            ),
            phase_2_neighbor_offset_points=(
                phase_2_neighbor_points
            ),
            phase_3_level_1_half_window_points=(
                phase_3_level_1_points
            ),
            phase_3_level_2_half_window_points=(
                phase_3_level_2_points
            ),
            phase_3_level_1_min_clean_values=(
                level_1_min_clean
            ),
            points_per_day=points_per_day,
            prophet_minimum_history_points=(
                prophet_history_points
            ),
        )


def build_5_minute_gap_profile() -> GapProfileConfig:
    """
    Build the agreed default gap profile for 5-minute KPI data.

    Returns:
        GapProfileConfig:
            Phase ranges 1-3, 4-12, 13-72, 73-288, and 289+.

    Raises:
        pydantic.ValidationError:
            If the built-in profile is accidentally made inconsistent.
    """

    return GapProfileConfig(
        phase_1=GapRangeConfig(
            minimum_points=1,
            maximum_points=3,
        ),
        phase_2=GapRangeConfig(
            minimum_points=4,
            maximum_points=12,
        ),
        phase_3=GapRangeConfig(
            minimum_points=13,
            maximum_points=72,
        ),
        phase_4_pass_1=GapRangeConfig(
            minimum_points=73,
            maximum_points=288,
        ),
        phase_4_pass_2=GapRangeConfig(
            minimum_points=289,
            maximum_points=None,
        ),
    )


def build_15_minute_gap_profile() -> GapProfileConfig:
    """
    Build the agreed default gap profile for 15-minute KPI data.

    Returns:
        GapProfileConfig:
            Phase ranges 1, 2-4, 5-24, 25-96, and 97+.

    Raises:
        pydantic.ValidationError:
            If the built-in profile is accidentally made inconsistent.
    """

    return GapProfileConfig(
        phase_1=GapRangeConfig(
            minimum_points=1,
            maximum_points=1,
        ),
        phase_2=GapRangeConfig(
            minimum_points=2,
            maximum_points=4,
        ),
        phase_3=GapRangeConfig(
            minimum_points=5,
            maximum_points=24,
        ),
        phase_4_pass_1=GapRangeConfig(
            minimum_points=25,
            maximum_points=96,
        ),
        phase_4_pass_2=GapRangeConfig(
            minimum_points=97,
            maximum_points=None,
        ),
    )


def _bucket_to_whole_minutes(
    bucket: BucketConfig,
) -> int:
    """
    Convert a resolved bucket configuration into a positive whole-minute step.

    Args:
        bucket:
            Resolved bucket unit and size returned by data preparation.

    Returns:
        int:
            Positive whole minutes between expected observations.

    Raises:
        ValueError:
            If the bucket unit is unsupported or cannot be represented as a
            positive whole number of minutes.
    """

    normalized_type = bucket.bucket_type.strip().casefold()

    if normalized_type in {"minute", "minutes"}:
        minutes = bucket.bucket_size

    elif normalized_type in {"hour", "hours"}:
        minutes = bucket.bucket_size * 60

    elif normalized_type in {"day", "days"}:
        minutes = bucket.bucket_size * 1_440

    elif normalized_type in {"second", "seconds"}:
        total_seconds = bucket.bucket_size

        if total_seconds % 60 != 0:
            raise ValueError(
                "Preprocessing requires a bucket that can be represented as "
                "whole minutes."
            )

        minutes = total_seconds // 60

    else:
        raise ValueError(
            "Unsupported bucket_type for preprocessing. Supported units are "
            "seconds, minutes, hours, and days."
        )

    if minutes <= 0:
        raise ValueError(
            "The resolved preprocessing bucket must be positive."
        )

    if 1_440 % minutes != 0:
        raise ValueError(
            "The bucket must divide evenly into 1,440 minutes so that "
            "every day contains a whole number of observations."
        )

    return int(minutes)


def _minutes_to_points(
    minutes: int,
    bucket_minutes: int,
    setting_name: str,
) -> int:
    """
    Convert a duration into an exact positive number of bucket steps.

    Args:
        minutes:
            Duration configured in minutes.
        bucket_minutes:
            Positive whole-minute bucket size.
        setting_name:
            Human-readable setting name used in validation errors.

    Returns:
        int:
            Exact number of timeline steps represented by the duration.

    Raises:
        ValueError:
            If the duration is not positive or is not divisible by the bucket.
    """

    if minutes <= 0:
        raise ValueError(
            f"{setting_name} must be positive."
        )

    if minutes % bucket_minutes != 0:
        raise ValueError(
            f"{setting_name}={minutes} minutes is incompatible with a "
            f"{bucket_minutes}-minute bucket. It must resolve to a whole "
            "number of points."
        )

    points = minutes // bucket_minutes

    if points <= 0:
        raise ValueError(
            f"{setting_name} must resolve to at least one point."
        )

    return points


__all__ = [
    "GapProfileConfig",
    "GapRangeConfig",
    "OutlierConfig",
    "Phase1Config",
    "Phase2Config",
    "Phase3Config",
    "Phase4ProphetConfig",
    "PreprocessingConfig",
    "ResolvedFrequencyConfig",
    "SeriesUsabilityConfig",
    "TimelineCorrectionConfig",
    "ValidValueConfig",
    "build_5_minute_gap_profile",
    "build_15_minute_gap_profile",
]