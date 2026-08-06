"""Generate seasonal baseline features for Part C feature engineering.

Intended project location::

    app/ml/feature_engineering/baseline_features.py

Supported baseline method
-------------------------

``weighted_seasonal`` estimates the expected KPI value at row ``t`` from
strictly earlier seasonal observations. The method uses this fallback ladder:

1. Search for the most recent usable anchor at the same timestamp-of-day and
   with the same broad day type (weekday or weekend), up to
   ``recent_fallback_days`` earlier days.
2. Search exact weekly anchors at ``t - 7 days``, ``t - 14 days``, and so on,
   up to ``weekly_fallback_weeks``.
3. When both anchors exist, combine them using ``recent_weight`` and
   ``weekly_weight``. When only one exists, use that anchor directly.
4. When neither seasonal anchor exists, use the median of the latest
   ``slot_fallback_points`` accepted anchors from earlier calendar days whose
   time-of-day lies within the configured slot tolerance.
5. Return ``NaN`` when every level is unavailable.

Robust local anchors
--------------------

When ``anchor_window_each_side_points`` is positive, a historical anchor is
represented by the median of accepted source values around that anchor. The
window is leakage-safe for every current row: its right side is capped so that
the current row and future rows can never enter the anchor median.

A positive local-anchor window is supported only for ``raw_value``. The schema
already rejects combining this local-median method with ``corrected_history``.

General rules
-------------

- input represents one valid site already ordered by timestamp;
- only configured ``anchor_statuses`` may supply anchor values;
- minimum warm-up is ``minimum_history_days * points_per_day``;
- all seasonal and slot fallbacks are strictly past-only;
- source row count and index are preserved;
- prepared sources and the original Part B dataframe are never modified; and
- output contains exactly one ``float64`` column named ``baseline``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from app.schemas.feature_engineering import (
    BaselineFeatureRequest,
    WeightedSeasonalBaselineConfig,
)

from .common import require_columns
from .source_preparation import PreparedFeatureSources


MINUTES_PER_DAY: Final = 1_440
BASELINE_COLUMN: Final = "baseline"


@dataclass(frozen=True, slots=True)
class WeightedSeasonalBaselineResult:
    """Generated baseline values and mutually informative audit Series."""

    values: pd.Series
    recent_anchor_values: pd.Series
    weekly_anchor_values: pd.Series
    slot_fallback_values: pd.Series
    weighted_anchor_mask: pd.Series
    recent_only_mask: pd.Series
    weekly_only_mask: pd.Series
    slot_fallback_mask: pd.Series
    warm_up_mask: pd.Series
    unresolved_mask: pd.Series

    def __post_init__(self) -> None:
        """Require complete index alignment and boolean audit masks."""

        expected_index = self.values.index

        for field_name in (
            "recent_anchor_values",
            "weekly_anchor_values",
            "slot_fallback_values",
            "weighted_anchor_mask",
            "recent_only_mask",
            "weekly_only_mask",
            "slot_fallback_mask",
            "warm_up_mask",
            "unresolved_mask",
        ):
            value = getattr(self, field_name)

            if not value.index.equals(expected_index):
                raise ValueError(
                    f"{field_name} is not aligned with baseline values."
                )

        for field_name in (
            "weighted_anchor_mask",
            "recent_only_mask",
            "weekly_only_mask",
            "slot_fallback_mask",
            "warm_up_mask",
            "unresolved_mask",
        ):
            if getattr(self, field_name).dtype != bool:
                raise ValueError(f"{field_name} must be a boolean Series.")

    @property
    def resolved_count(self) -> int:
        """Return the number of rows containing a finite baseline value."""

        return int(self.values.notna().sum())

    @property
    def unresolved_count(self) -> int:
        """Return the number of rows without a baseline value."""

        return int(self.unresolved_mask.sum())


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------


def _validate_bucket_minutes(bucket_minutes: int) -> int:
    """Validate and return a whole-day-dividing dataset bucket."""

    if isinstance(bucket_minutes, bool) or not isinstance(bucket_minutes, int):
        raise TypeError("bucket_minutes must be an integer number of minutes.")

    if bucket_minutes <= 0:
        raise ValueError("bucket_minutes must be greater than zero.")

    if MINUTES_PER_DAY % bucket_minutes != 0:
        raise ValueError(
            "bucket_minutes must divide 1,440 minutes exactly for seasonal "
            "baseline generation."
        )

    return bucket_minutes


def _prepare_timestamps(
    site_dataframe: pd.DataFrame,
    *,
    timestamp_column: str,
    expected_index: pd.Index,
) -> pd.Series:
    """Return validated, unique, increasing timestamps aligned with sources."""

    if not isinstance(site_dataframe, pd.DataFrame):
        raise TypeError("site_dataframe must be a pandas DataFrame.")

    if not isinstance(timestamp_column, str) or not timestamp_column.strip():
        raise ValueError("timestamp_column must be a non-empty string.")

    require_columns(
        site_dataframe,
        [timestamp_column],
        context="seasonal baseline generation",
    )

    if site_dataframe.empty:
        raise ValueError(
            "Cannot generate a seasonal baseline from an empty site dataframe."
        )

    if not site_dataframe.index.equals(expected_index):
        raise ValueError(
            "site_dataframe is not index-aligned with prepared feature "
            "sources."
        )

    timestamps = pd.to_datetime(
        site_dataframe[timestamp_column],
        errors="coerce",
    )

    if timestamps.isna().any():
        invalid_indices = timestamps.index[timestamps.isna()].tolist()[:10]
        raise ValueError(
            f"Timestamp column '{timestamp_column}' contains "
            f"{int(timestamps.isna().sum()):,} missing or invalid values. "
            f"First invalid indices: {invalid_indices}"
        )

    if timestamps.duplicated().any():
        examples = (
            timestamps.loc[timestamps.duplicated(keep=False)]
            .head(5)
            .astype("string")
            .tolist()
        )
        raise ValueError(
            "Seasonal baseline generation requires unique timestamps. "
            f"Examples: {examples}"
        )

    if not timestamps.is_monotonic_increasing:
        raise ValueError(
            "site_dataframe must already be ordered by timestamp before "
            "seasonal baseline generation."
        )

    timestamps.index = expected_index
    timestamps.name = timestamp_column

    return timestamps


def _select_baseline_source(
    *,
    config: WeightedSeasonalBaselineConfig,
    sources: PreparedFeatureSources,
) -> pd.Series:
    """Select one copied finite-or-NaN source for baseline anchors."""

    if not isinstance(sources, PreparedFeatureSources):
        raise TypeError(
            "sources must be an instance of PreparedFeatureSources; received "
            f"{type(sources).__name__}."
        )

    selected = pd.to_numeric(
        sources.bundle.select(config.source),
        errors="coerce",
    ).astype("float64")

    if not selected.index.equals(sources.bundle.raw_value.index):
        raise ValueError(
            "Selected baseline source is not aligned with raw_value."
        )

    finite_values = selected.dropna().to_numpy(dtype="float64")

    if finite_values.size and not np.isfinite(finite_values).all():
        raise ValueError(
            f"Selected baseline source {config.source!r} contains positive or "
            "negative infinity."
        )

    return selected.copy()


def _select_anchor_statuses(
    *,
    config: WeightedSeasonalBaselineConfig,
    sources: PreparedFeatureSources,
    expected_index: pd.Index,
) -> pd.Series:
    """Return one aligned integer status Series for anchor filtering."""

    if sources.statuses is None:
        raise ValueError(
            "A status Series is required for seasonal baseline generation."
        )

    statuses = pd.to_numeric(
        sources.statuses,
        errors="coerce",
    )

    if not statuses.index.equals(expected_index):
        raise ValueError(
            "Prepared statuses are not aligned with the baseline source."
        )

    if statuses.isna().any():
        raise ValueError(
            "Prepared statuses contain missing or non-numeric values."
        )

    if not config.anchor_statuses:
        raise ValueError("baseline.anchor_statuses cannot be empty.")

    return statuses.astype("int16", copy=True)


# ---------------------------------------------------------------------------
# Leakage-safe local anchor candidates
# ---------------------------------------------------------------------------


def _prepare_local_anchor_candidates(
    *,
    accepted_values: pd.Series,
    left_window_points: int,
    right_window_points: int,
) -> pd.Series:
    """Return local anchor medians for one asymmetric positional window."""

    if left_window_points < 0:
        raise ValueError("left_window_points cannot be negative.")

    if right_window_points < 0:
        raise ValueError("right_window_points cannot be negative.")

    if right_window_points > left_window_points:
        raise ValueError(
            "right_window_points cannot exceed left_window_points."
        )

    if left_window_points == 0:
        result = accepted_values.copy()
        result.name = "local_anchor_left_0_right_0"
        return result.astype("float64")

    total_window_points = (
        left_window_points
        + right_window_points
        + 1
    )

    if right_window_points == left_window_points:
        candidates = accepted_values.rolling(
            window=total_window_points,
            center=True,
            min_periods=1,
        ).median()
    else:
        positional_values = accepted_values.reset_index(drop=True)

        if right_window_points > 0:
            padding = pd.Series(
                np.nan,
                index=range(right_window_points),
                dtype="float64",
            )
            positional_values = pd.concat(
                [positional_values, padding],
                ignore_index=True,
            )

        candidates = (
            positional_values
            .rolling(
                window=total_window_points,
                center=False,
                min_periods=1,
            )
            .median()
            .shift(-right_window_points)
            .iloc[:len(accepted_values)]
        )
        candidates.index = accepted_values.index

    candidates.name = (
        f"local_anchor_left_{left_window_points}_"
        f"right_{right_window_points}"
    )

    return candidates.astype("float64")


class _AnchorResolver:
    """Resolve accepted historical anchors with cached safe local medians."""

    def __init__(
        self,
        *,
        source: pd.Series,
        statuses: pd.Series,
        accepted_statuses: set[int],
        window_each_side_points: int,
    ) -> None:
        if window_each_side_points < 0:
            raise ValueError(
                "anchor_window_each_side_points cannot be negative."
            )

        if not source.index.equals(statuses.index):
            raise ValueError(
                "Baseline source and statuses are not index-aligned."
            )

        self._source = source
        self._accepted_mask = (
            statuses.isin(accepted_statuses)
            & source.notna()
        ).astype(bool)
        self._accepted_values = source.where(self._accepted_mask)
        self._window = window_each_side_points
        self._candidate_cache: dict[int, pd.Series] = {}

    @property
    def accepted_mask(self) -> pd.Series:
        """Return the exact-anchor acceptance mask as a read-only convention."""

        return self._accepted_mask

    def resolve(
        self,
        *,
        anchor_position: int,
        current_position: int,
    ) -> float:
        """Return one accepted, leakage-safe robust anchor or ``NaN``."""

        if anchor_position < 0 or anchor_position >= len(self._source):
            return np.nan

        if current_position <= anchor_position:
            return np.nan

        if not bool(self._accepted_mask.iloc[anchor_position]):
            return np.nan

        if self._window == 0:
            value = self._source.iloc[anchor_position]
            return float(value) if pd.notna(value) else np.nan

        safe_right_points = min(
            self._window,
            current_position - anchor_position - 1,
        )

        candidates = self._candidate_cache.get(safe_right_points)

        if candidates is None:
            candidates = _prepare_local_anchor_candidates(
                accepted_values=self._accepted_values,
                left_window_points=self._window,
                right_window_points=safe_right_points,
            )
            self._candidate_cache[safe_right_points] = candidates

        value = candidates.iloc[anchor_position]
        return float(value) if pd.notna(value) else np.nan


# ---------------------------------------------------------------------------
# Seasonal and slot lookup helpers
# ---------------------------------------------------------------------------


def _minutes_since_midnight(timestamps: pd.Series) -> np.ndarray:
    """Return fractional local minutes elapsed since midnight."""

    return (
        timestamps.dt.hour.to_numpy(dtype="float64") * 60.0
        + timestamps.dt.minute.to_numpy(dtype="float64")
        + timestamps.dt.second.to_numpy(dtype="float64") / 60.0
        + timestamps.dt.microsecond.to_numpy(dtype="float64")
        / 60_000_000.0
        + timestamps.dt.nanosecond.to_numpy(dtype="float64")
        / 60_000_000_000.0
    )


def _resolve_slot_numbers(
    *,
    timestamps: pd.Series,
    bucket_minutes: int,
) -> np.ndarray:
    """Return exact zero-based time-of-day slots for regular bucketed rows."""

    minutes = _minutes_since_midnight(timestamps)
    raw_slots = minutes / float(bucket_minutes)
    rounded_slots = np.rint(raw_slots)

    if not np.allclose(raw_slots, rounded_slots, rtol=0.0, atol=1e-9):
        raise ValueError(
            "Seasonal baseline timestamps must align exactly to the resolved "
            "dataset bucket."
        )

    points_per_day = MINUTES_PER_DAY // bucket_minutes
    slots = rounded_slots.astype("int64")

    if ((slots < 0) | (slots >= points_per_day)).any():
        raise ValueError("Resolved time-of-day slot is outside one day.")

    return slots


def _find_recent_day_type_anchor(
    *,
    current_position: int,
    current_timestamp: pd.Timestamp,
    current_is_weekend: bool,
    maximum_days: int,
    timestamp_to_position: dict[pd.Timestamp, int],
    anchor_resolver: _AnchorResolver,
) -> float:
    """Return the nearest usable same-day-type anchor at the same time."""

    for days_back in range(1, maximum_days + 1):
        candidate_timestamp = current_timestamp - pd.Timedelta(days=days_back)
        candidate_is_weekend = candidate_timestamp.dayofweek >= 5

        if candidate_is_weekend != current_is_weekend:
            continue

        anchor_position = timestamp_to_position.get(candidate_timestamp)

        if anchor_position is None:
            continue

        value = anchor_resolver.resolve(
            anchor_position=anchor_position,
            current_position=current_position,
        )

        if np.isfinite(value):
            return value

    return np.nan


def _find_weekly_anchor(
    *,
    current_position: int,
    current_timestamp: pd.Timestamp,
    maximum_weeks: int,
    timestamp_to_position: dict[pd.Timestamp, int],
    anchor_resolver: _AnchorResolver,
) -> float:
    """Return the nearest usable exact weekly anchor."""

    for weeks_back in range(1, maximum_weeks + 1):
        candidate_timestamp = current_timestamp - pd.Timedelta(
            days=7 * weeks_back
        )
        anchor_position = timestamp_to_position.get(candidate_timestamp)

        if anchor_position is None:
            continue

        value = anchor_resolver.resolve(
            anchor_position=anchor_position,
            current_position=current_position,
        )

        if np.isfinite(value):
            return value

    return np.nan


def _resolve_tolerated_slots(
    *,
    target_slot: int,
    tolerance_points: int,
    points_per_day: int,
) -> tuple[int, ...]:
    """Return unique circular time slots inside the configured tolerance."""

    return tuple(
        dict.fromkeys(
            (target_slot + offset) % points_per_day
            for offset in range(-tolerance_points, tolerance_points + 1)
        )
    )


def _resolve_slot_fallback(
    *,
    current_position: int,
    target_slot: int,
    tolerance_points: int,
    points_per_day: int,
    maximum_values: int,
    slot_history: dict[int, deque[int]],
    anchor_resolver: _AnchorResolver,
) -> float:
    """Return the median of latest usable prior-day same-slot anchors."""

    candidate_positions: list[int] = []

    for slot in _resolve_tolerated_slots(
        target_slot=target_slot,
        tolerance_points=tolerance_points,
        points_per_day=points_per_day,
    ):
        candidate_positions.extend(slot_history.get(slot, ()))

    if not candidate_positions:
        return np.nan

    candidate_positions.sort(reverse=True)
    resolved_values: list[float] = []

    for anchor_position in candidate_positions:
        value = anchor_resolver.resolve(
            anchor_position=anchor_position,
            current_position=current_position,
        )

        if np.isfinite(value):
            resolved_values.append(value)

        if len(resolved_values) >= maximum_values:
            break

    if not resolved_values:
        return np.nan

    return float(np.median(np.asarray(resolved_values, dtype="float64")))


# ---------------------------------------------------------------------------
# Public generation functions
# ---------------------------------------------------------------------------


def generate_weighted_seasonal_baseline(
    config: WeightedSeasonalBaselineConfig,
    *,
    sources: PreparedFeatureSources,
    site_dataframe: pd.DataFrame,
    timestamp_column: str,
    bucket_minutes: int,
) -> WeightedSeasonalBaselineResult:
    """Generate one weighted-seasonal baseline for a valid site.

    Args:
        config:
            Validated weighted-seasonal baseline configuration.
        sources:
            Prepared raw and optional corrected-history sources.
        site_dataframe:
            One valid site dataframe already ordered by timestamp.
        timestamp_column:
            Resolved physical timestamp-column name.
        bucket_minutes:
            Resolved dataset frequency in whole minutes.

    Returns:
        WeightedSeasonalBaselineResult:
            Baseline values plus anchor/fallback audit Series.
    """

    if not isinstance(config, WeightedSeasonalBaselineConfig):
        raise TypeError(
            "generate_weighted_seasonal_baseline requires "
            "WeightedSeasonalBaselineConfig; received "
            f"{type(config).__name__}."
        )

    if config.method != "weighted_seasonal":
        raise ValueError(
            f"Unsupported baseline method: {config.method!r}"
        )

    resolved_bucket = _validate_bucket_minutes(bucket_minutes)
    points_per_day = MINUTES_PER_DAY // resolved_bucket

    source = _select_baseline_source(
        config=config,
        sources=sources,
    )

    if len(source) == 0:
        raise ValueError(
            "Cannot generate a baseline from an empty prepared site source."
        )

    timestamps = _prepare_timestamps(
        site_dataframe,
        timestamp_column=timestamp_column,
        expected_index=source.index,
    )
    statuses = _select_anchor_statuses(
        config=config,
        sources=sources,
        expected_index=source.index,
    )

    if (
        config.source == "corrected_history"
        and config.anchor_window_each_side_points > 0
    ):
        raise ValueError(
            "A positive anchor_window_each_side_points is not supported with "
            "source='corrected_history'."
        )

    slot_tolerance_points = config.resolve_slot_tolerance_points(
        bucket_minutes=resolved_bucket,
    )
    minimum_history_points = (
        config.minimum_history_days
        * points_per_day
    )

    timestamp_values = list(timestamps)
    timestamp_to_position = {
        timestamp: position
        for position, timestamp in enumerate(timestamp_values)
    }
    weekend_flags = (
        timestamps.dt.dayofweek >= 5
    ).to_numpy(dtype=bool)
    normalized_dates = timestamps.dt.normalize().to_numpy()
    slot_numbers = _resolve_slot_numbers(
        timestamps=timestamps,
        bucket_minutes=resolved_bucket,
    )

    anchor_resolver = _AnchorResolver(
        source=source,
        statuses=statuses,
        accepted_statuses=set(config.anchor_statuses),
        window_each_side_points=(
            config.anchor_window_each_side_points
        ),
    )

    baseline_values = np.full(len(source), np.nan, dtype="float64")
    recent_values = np.full(len(source), np.nan, dtype="float64")
    weekly_values = np.full(len(source), np.nan, dtype="float64")
    slot_values = np.full(len(source), np.nan, dtype="float64")

    weighted_mask = np.zeros(len(source), dtype=bool)
    recent_only_mask = np.zeros(len(source), dtype=bool)
    weekly_only_mask = np.zeros(len(source), dtype=bool)
    slot_mask = np.zeros(len(source), dtype=bool)
    warm_up_mask = np.arange(len(source)) < minimum_history_points

    # Each slot retains only the latest configured count from earlier calendar
    # days. Current-day positions are staged and become available only after a
    # date transition, preventing same-day fallback leakage.
    slot_history: dict[int, deque[int]] = {
        slot: deque(maxlen=config.slot_fallback_points)
        for slot in range(points_per_day)
    }
    staged_day_positions: list[int] = []
    active_date: np.datetime64 | None = None

    for position in range(len(source)):
        row_date = normalized_dates[position]

        if active_date is None:
            active_date = row_date

        elif row_date != active_date:
            for staged_position in staged_day_positions:
                if bool(anchor_resolver.accepted_mask.iloc[staged_position]):
                    slot_history[int(slot_numbers[staged_position])].append(
                        staged_position
                    )

            staged_day_positions = []
            active_date = row_date

        if not warm_up_mask[position]:
            current_timestamp = timestamp_values[position]

            recent = _find_recent_day_type_anchor(
                current_position=position,
                current_timestamp=current_timestamp,
                current_is_weekend=bool(weekend_flags[position]),
                maximum_days=config.recent_fallback_days,
                timestamp_to_position=timestamp_to_position,
                anchor_resolver=anchor_resolver,
            )
            weekly = _find_weekly_anchor(
                current_position=position,
                current_timestamp=current_timestamp,
                maximum_weeks=config.weekly_fallback_weeks,
                timestamp_to_position=timestamp_to_position,
                anchor_resolver=anchor_resolver,
            )

            recent_values[position] = recent
            weekly_values[position] = weekly

            has_recent = np.isfinite(recent)
            has_weekly = np.isfinite(weekly)

            if has_recent and has_weekly:
                baseline_values[position] = (
                    config.recent_weight * recent
                    + config.weekly_weight * weekly
                )
                weighted_mask[position] = True

            elif has_recent:
                baseline_values[position] = recent
                recent_only_mask[position] = True

            elif has_weekly:
                baseline_values[position] = weekly
                weekly_only_mask[position] = True

            else:
                slot_value = _resolve_slot_fallback(
                    current_position=position,
                    target_slot=int(slot_numbers[position]),
                    tolerance_points=slot_tolerance_points,
                    points_per_day=points_per_day,
                    maximum_values=config.slot_fallback_points,
                    slot_history=slot_history,
                    anchor_resolver=anchor_resolver,
                )
                slot_values[position] = slot_value

                if np.isfinite(slot_value):
                    baseline_values[position] = slot_value
                    slot_mask[position] = True

        staged_day_positions.append(position)

    baseline = pd.Series(
        baseline_values,
        index=source.index,
        name=BASELINE_COLUMN,
        dtype="float64",
    )
    recent_series = pd.Series(
        recent_values,
        index=source.index,
        name="recent_anchor",
        dtype="float64",
    )
    weekly_series = pd.Series(
        weekly_values,
        index=source.index,
        name="weekly_anchor",
        dtype="float64",
    )
    slot_series = pd.Series(
        slot_values,
        index=source.index,
        name="slot_fallback",
        dtype="float64",
    )

    return WeightedSeasonalBaselineResult(
        values=baseline,
        recent_anchor_values=recent_series,
        weekly_anchor_values=weekly_series,
        slot_fallback_values=slot_series,
        weighted_anchor_mask=pd.Series(
            weighted_mask,
            index=source.index,
            dtype=bool,
        ),
        recent_only_mask=pd.Series(
            recent_only_mask,
            index=source.index,
            dtype=bool,
        ),
        weekly_only_mask=pd.Series(
            weekly_only_mask,
            index=source.index,
            dtype=bool,
        ),
        slot_fallback_mask=pd.Series(
            slot_mask,
            index=source.index,
            dtype=bool,
        ),
        warm_up_mask=pd.Series(
            warm_up_mask,
            index=source.index,
            dtype=bool,
        ),
        unresolved_mask=baseline.isna().astype(bool),
    )


def generate_baseline_features(
    request: BaselineFeatureRequest,
    *,
    sources: PreparedFeatureSources,
    site_dataframe: pd.DataFrame,
    timestamp_column: str,
    bucket_minutes: int,
) -> pd.DataFrame:
    """Generate the standalone schema-defined ``baseline`` feature column."""

    if not isinstance(request, BaselineFeatureRequest):
        raise TypeError(
            "generate_baseline_features requires BaselineFeatureRequest; "
            f"received {type(request).__name__}."
        )

    result = generate_weighted_seasonal_baseline(
        request.baseline,
        sources=sources,
        site_dataframe=site_dataframe,
        timestamp_column=timestamp_column,
        bucket_minutes=bucket_minutes,
    )

    output = result.values.to_frame()

    if len(output) != len(sources.bundle.raw_value):
        raise RuntimeError(
            "Baseline generation unexpectedly changed the row count."
        )

    if not output.index.equals(sources.bundle.raw_value.index):
        raise RuntimeError(
            "Baseline generation unexpectedly changed the source index."
        )

    if tuple(output.columns) != (BASELINE_COLUMN,):
        raise RuntimeError(
            "Baseline generation produced an unexpected output column."
        )

    finite_values = output[BASELINE_COLUMN].dropna().to_numpy(
        dtype="float64"
    )

    if finite_values.size and not np.isfinite(finite_values).all():
        raise RuntimeError(
            "Baseline generation produced positive or negative infinity."
        )

    return output


__all__ = [
    "BASELINE_COLUMN",
    "MINUTES_PER_DAY",
    "WeightedSeasonalBaselineResult",
    "generate_baseline_features",
    "generate_weighted_seasonal_baseline",
]