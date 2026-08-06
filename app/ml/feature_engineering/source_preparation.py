"""Value-source preparation for Part C feature engineering.

Revision: asymmetric local-median boundary handling fixed.

Intended project location::

    app/ml/feature_engineering/source_preparation.py

This module prepares the value series consumed by feature generators. It does
not decide which features are requested and does not append feature columns to
the dataset. Those responsibilities belong to ``runner.py``.

The supported source/cleaning combinations are:

1. ``raw_value``
   Use the original Part B KPI value unchanged.

2. ``raw_value`` plus local median cleaning
   Shift/reference the raw source first, then replace an unacceptable
   referenced point with a median calculated from nearby accepted raw values.
   Local median cleaning is not permitted with ``corrected_history``.

3. ``corrected_history``
   Build one temporary full-series history in which configured bad statuses
   and unresolved values are replaced using a trailing median and optional
   forward fill.

The timestamp sequence is never changed here. Only temporary value Series are
prepared. The original Part B dataframe and KPI value column remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from app.schemas.feature_engineering import (
    CorrectedHistoryConfig,
    FeatureSource,
    LocalMedianCleaningConfig,
    ResolvedFeatureEngineeringConfig,
)

from .common import (
    FeatureSourceBundle,
    build_feature_source_bundle,
    build_raw_value_series,
    require_columns,
)


DEFAULT_STATUS_COLUMN: Final = "Current_Status_Label"


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CorrectedHistoryResult:
    """Prepared corrected-history series and its audit masks."""

    values: pd.Series
    unavailable_mask: pd.Series
    trailing_median_replaced_mask: pd.Series
    forward_filled_mask: pd.Series
    unresolved_mask: pd.Series

    def __post_init__(self) -> None:
        """Require complete alignment between the result and all audit masks."""

        expected_index = self.values.index

        for field_name in (
            "unavailable_mask",
            "trailing_median_replaced_mask",
            "forward_filled_mask",
            "unresolved_mask",
        ):
            mask = getattr(self, field_name)

            if not mask.index.equals(expected_index):
                raise ValueError(
                    f"{field_name} is not aligned with corrected history."
                )

            if mask.dtype != bool:
                raise ValueError(
                    f"{field_name} must be a boolean pandas Series."
                )

    @property
    def unavailable_count(self) -> int:
        return int(self.unavailable_mask.sum())

    @property
    def trailing_median_replaced_count(self) -> int:
        return int(self.trailing_median_replaced_mask.sum())

    @property
    def forward_filled_count(self) -> int:
        return int(self.forward_filled_mask.sum())

    @property
    def unresolved_count(self) -> int:
        return int(self.unresolved_mask.sum())


@dataclass(frozen=True, slots=True)
class LocalMedianCandidateResult:
    """Local-median candidate available at every original timestamp.

    ``half_window_points`` is the configured maximum number of observations on
    the left side of each candidate timestamp. ``right_window_points`` records
    the actual number used on the right side. A value of ``None`` is retained
    only for backwards compatibility with older direct constructors and means
    the original symmetric window.
    """

    values: pd.Series
    accepted_value_mask: pd.Series
    accepted_statuses: frozenset[int]
    half_window_points: int
    minimum_valid_points: int
    right_window_points: int | None = None

    def __post_init__(self) -> None:
        if not self.values.index.equals(self.accepted_value_mask.index):
            raise ValueError(
                "Local-median candidates and accepted-value mask are not "
                "aligned."
            )

        if self.accepted_value_mask.dtype != bool:
            raise ValueError(
                "accepted_value_mask must be a boolean pandas Series."
            )

        if not self.accepted_statuses:
            raise ValueError(
                "accepted_statuses cannot be empty for local median candidates."
            )

        if self.half_window_points <= 0:
            raise ValueError(
                "half_window_points must be greater than zero."
            )

        right_window_points = self.resolved_right_window_points

        if right_window_points < 0:
            raise ValueError(
                "right_window_points cannot be negative."
            )

        if right_window_points > self.half_window_points:
            raise ValueError(
                "right_window_points cannot exceed half_window_points."
            )

        if self.minimum_valid_points <= 0:
            raise ValueError(
                "minimum_valid_points must be greater than zero."
            )

        if self.minimum_valid_points > self.total_window_points:
            raise ValueError(
                "minimum_valid_points cannot exceed the complete local "
                "median window size."
            )

    @property
    def resolved_right_window_points(self) -> int:
        """Return the actual right-side window represented by this result."""

        if self.right_window_points is None:
            return self.half_window_points

        return self.right_window_points

    @property
    def total_window_points(self) -> int:
        """Return the maximum candidate positions in the asymmetric window."""

        return (
            self.half_window_points
            + self.resolved_right_window_points
            + 1
        )


@dataclass(frozen=True, slots=True)
class ReferencedValueResult:
    """One lagged/referenced source after optional local median cleaning."""

    values: pd.Series
    referenced_source: pd.Series
    referenced_status: pd.Series
    unacceptable_reference_mask: pd.Series
    local_median_replaced_mask: pd.Series
    source_fallback_kept_mask: pd.Series
    unresolved_mask: pd.Series
    offset_points: int
    source_name: FeatureSource

    def __post_init__(self) -> None:
        expected_index = self.values.index

        for field_name in (
            "referenced_source",
            "referenced_status",
            "unacceptable_reference_mask",
            "local_median_replaced_mask",
            "source_fallback_kept_mask",
            "unresolved_mask",
        ):
            value = getattr(self, field_name)

            if not value.index.equals(expected_index):
                raise ValueError(
                    f"{field_name} is not aligned with referenced values."
                )

    @property
    def local_median_replaced_count(self) -> int:
        return int(self.local_median_replaced_mask.sum())

    @property
    def source_fallback_kept_count(self) -> int:
        return int(self.source_fallback_kept_mask.sum())

    @property
    def unresolved_count(self) -> int:
        return int(self.unresolved_mask.sum())


@dataclass(frozen=True, slots=True)
class PreparedFeatureSources:
    """All shared value sources prepared for one valid site."""

    bundle: FeatureSourceBundle
    statuses: pd.Series | None
    corrected_history_result: CorrectedHistoryResult | None

    def __post_init__(self) -> None:
        expected_index = self.bundle.raw_value.index

        if self.statuses is not None and not self.statuses.index.equals(
            expected_index
        ):
            raise ValueError(
                "Status series is not aligned with prepared feature sources."
            )

        if self.corrected_history_result is None:
            return

        if not self.corrected_history_result.values.index.equals(
            expected_index
        ):
            raise ValueError(
                "Corrected-history result is not aligned with raw values."
            )


# ---------------------------------------------------------------------------
# Shared normalization and alignment
# ---------------------------------------------------------------------------


def _require_aligned_series(
    *,
    reference: pd.Series,
    candidates: dict[str, pd.Series],
) -> None:
    """Require all supplied Series to use the reference index and row count."""

    for name, candidate in candidates.items():
        if len(candidate) != len(reference):
            raise ValueError(
                f"{name} contains {len(candidate):,} rows but the reference "
                f"contains {len(reference):,}."
            )

        if not candidate.index.equals(reference.index):
            raise ValueError(
                f"{name} is not aligned with the reference Series."
            )


def _normalize_status_series(
    site_dataframe: pd.DataFrame,
    *,
    status_column: str,
    expected_index: pd.Index,
) -> pd.Series:
    """Return one aligned integer status Series without repairing labels."""

    require_columns(
        site_dataframe,
        [status_column],
        context="feature-source status preparation",
    )

    numeric = pd.to_numeric(
        site_dataframe[status_column],
        errors="coerce",
    )

    if numeric.isna().any():
        raise ValueError(
            f"Status column '{status_column}' contains missing or non-numeric "
            "values. Input validation must run before source preparation."
        )

    fractional = numeric.mod(1).ne(0)
    if fractional.any():
        raise ValueError(
            f"Status column '{status_column}' contains non-integer labels."
        )

    statuses = numeric.astype("int16")
    statuses.index = expected_index
    statuses.name = status_column

    return statuses


def _ensure_numeric_finite_or_nan(
    values: pd.Series,
    *,
    context: str,
) -> pd.Series:
    """Normalize one value Series and reject infinity."""

    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).astype("float64")

    finite_values = numeric.dropna().to_numpy(dtype="float64")

    if finite_values.size and not np.isfinite(finite_values).all():
        raise ValueError(
            f"{context} contains positive or negative infinity."
        )

    return numeric


# ---------------------------------------------------------------------------
# Shared corrected-history preparation
# ---------------------------------------------------------------------------


def prepare_corrected_history(
    *,
    raw_values: pd.Series,
    statuses: pd.Series,
    config: CorrectedHistoryConfig,
    window_points: int,
) -> CorrectedHistoryResult:
    """Build one temporary corrected-history Series for a valid site.

    Cleaning formula:

    1. Mark configured bad statuses as unavailable.
    2. Optionally mark remaining numeric NaN values as unavailable.
    3. Calculate a trailing median from earlier clean raw values only.
    4. Replace unavailable positions where that trailing median exists.
    5. Optionally forward-fill only still-unavailable target positions.

    The current row never enters its own trailing-median replacement because
    the clean source is shifted by one row before rolling.
    """

    if window_points <= 0:
        raise ValueError("window_points must be greater than zero.")

    if config.minimum_periods > window_points:
        raise ValueError(
            "corrected_history.minimum_periods cannot exceed window_points."
        )

    raw = _ensure_numeric_finite_or_nan(
        raw_values,
        context="raw feature source",
    )
    status = pd.to_numeric(statuses, errors="coerce")

    _require_aligned_series(
        reference=raw,
        candidates={"statuses": status},
    )

    if status.isna().any():
        raise ValueError(
            "statuses contains missing or non-numeric values during corrected-"
            "history preparation."
        )

    unavailable_mask = status.isin(config.bad_statuses)

    if config.replace_missing_values:
        unavailable_mask = unavailable_mask | raw.isna()

    unavailable_mask = unavailable_mask.astype(bool)

    clean_raw_source = raw.mask(unavailable_mask)

    if config.replacement_method != "trailing_median":
        raise ValueError(
            "Unsupported corrected-history replacement method: "
            f"{config.replacement_method!r}"
        )

    trailing_median = (
        clean_raw_source
        .shift(1)
        .rolling(
            window=window_points,
            min_periods=config.minimum_periods,
        )
        .median()
    )

    # Remove every unavailable value from the temporary history first.
    # This is essential for finite outliers: when no trailing median exists,
    # the original bad value must not silently remain in corrected_history.
    corrected = raw.mask(unavailable_mask)

    trailing_replacement_mask = (
        unavailable_mask
        & trailing_median.notna()
    )
    corrected.loc[trailing_replacement_mask] = trailing_median.loc[
        trailing_replacement_mask
    ]

    forward_filled_mask = pd.Series(
        False,
        index=raw.index,
        dtype=bool,
    )

    remaining_target_mask = unavailable_mask & corrected.isna()

    if config.forward_fill_remaining and remaining_target_mask.any():
        forward_candidates = corrected.ffill()
        forward_filled_mask = (
            remaining_target_mask
            & forward_candidates.notna()
        )
        corrected.loc[forward_filled_mask] = forward_candidates.loc[
            forward_filled_mask
        ]

    unresolved_mask = corrected.isna()

    corrected.name = "corrected_history"

    return CorrectedHistoryResult(
        values=corrected.astype("float64"),
        unavailable_mask=unavailable_mask.astype(bool),
        trailing_median_replaced_mask=trailing_replacement_mask.astype(bool),
        forward_filled_mask=forward_filled_mask.astype(bool),
        unresolved_mask=unresolved_mask.astype(bool),
    )


# ---------------------------------------------------------------------------
# Local median candidates and referenced-point cleaning
# ---------------------------------------------------------------------------


def prepare_local_median_candidates(
    *,
    raw_values: pd.Series,
    statuses: pd.Series,
    config: LocalMedianCleaningConfig,
    half_window_points: int,
    right_window_points: int | None = None,
) -> LocalMedianCandidateResult:
    """Calculate reusable local-median candidates at every timestamp.

    The left side always uses the configured ``half_window_points``. The right
    side defaults to the same size, preserving the original centered-window
    behaviour used by callers such as seasonal baseline generation.

    Lag features may pass a smaller ``right_window_points`` value after
    deciding how much of the centered window is safe for a particular lag. For
    a candidate timestamp ``r``, the resulting window is::

        [r - half_window_points, ..., r + right_window_points]

    Only finite raw observations whose statuses belong to
    ``accepted_statuses`` contribute to the median. This function calculates
    candidates only; the requesting feature module remains responsible for
    deciding the leakage-safe right-side width.
    """

    if not config.enabled:
        raise ValueError(
            "Local median candidates cannot be prepared when cleaning is "
            "disabled."
        )

    if isinstance(half_window_points, bool) or not isinstance(
        half_window_points,
        int,
    ):
        raise TypeError("half_window_points must be an integer.")

    if half_window_points <= 0:
        raise ValueError("half_window_points must be greater than zero.")

    if right_window_points is None:
        resolved_right_window_points = half_window_points
    else:
        if isinstance(right_window_points, bool) or not isinstance(
            right_window_points,
            int,
        ):
            raise TypeError("right_window_points must be an integer or None.")

        resolved_right_window_points = right_window_points

    if resolved_right_window_points < 0:
        raise ValueError("right_window_points cannot be negative.")

    if resolved_right_window_points > half_window_points:
        raise ValueError(
            "right_window_points cannot exceed half_window_points."
        )

    raw = _ensure_numeric_finite_or_nan(
        raw_values,
        context="raw values for local median cleaning",
    )
    status = pd.to_numeric(statuses, errors="coerce")

    _require_aligned_series(
        reference=raw,
        candidates={"statuses": status},
    )

    if status.isna().any():
        raise ValueError(
            "statuses contains missing or non-numeric values during local-"
            "median preparation."
        )

    accepted_value_mask = (
        status.isin(config.accepted_statuses)
        & raw.notna()
    )
    accepted_values = raw.where(accepted_value_mask)

    complete_window_points = (
        half_window_points
        + resolved_right_window_points
        + 1
    )

    if config.minimum_valid_points > complete_window_points:
        raise ValueError(
            "local cleaning minimum_valid_points cannot exceed the resolved "
            "local window size."
        )

    if resolved_right_window_points == half_window_points:
        # Preserve the original symmetric calculation exactly for existing
        # callers that do not request an adaptive right-side window.
        candidates = accepted_values.rolling(
            window=complete_window_points,
            center=True,
            min_periods=config.minimum_valid_points,
        ).median()
    else:
        # Calculate on a temporary positional index and pad the right edge
        # with unavailable rows. A trailing result at position r + right then
        # represents the desired asymmetric interval
        # [r - left, ..., r + right]. Padding preserves valid partial windows
        # at the end of the site, while the trailing roll naturally preserves
        # partial windows at the beginning. Restore the original index after
        # trimming the temporary padding.
        positional_values = accepted_values.reset_index(drop=True)

        if resolved_right_window_points > 0:
            right_padding = pd.Series(
                np.nan,
                index=range(resolved_right_window_points),
                dtype="float64",
            )
            positional_values = pd.concat(
                [positional_values, right_padding],
                ignore_index=True,
            )

        candidates = (
            positional_values
            .rolling(
                window=complete_window_points,
                center=False,
                min_periods=config.minimum_valid_points,
            )
            .median()
            .shift(-resolved_right_window_points)
            .iloc[:len(accepted_values)]
        )
        candidates.index = accepted_values.index

    candidates.name = (
        "local_median_candidate_"
        f"left_{half_window_points}_"
        f"right_{resolved_right_window_points}"
    )

    return LocalMedianCandidateResult(
        values=candidates.astype("float64"),
        accepted_value_mask=accepted_value_mask.astype(bool),
        accepted_statuses=frozenset(config.accepted_statuses),
        half_window_points=half_window_points,
        minimum_valid_points=config.minimum_valid_points,
        right_window_points=resolved_right_window_points,
    )


def prepare_referenced_values(
    *,
    sources: FeatureSourceBundle,
    statuses: pd.Series | None,
    source_name: FeatureSource,
    offset_points: int,
    local_cleaning: LocalMedianCleaningConfig,
    bucket_minutes: int,
    local_candidates: LocalMedianCandidateResult | None = None,
) -> ReferencedValueResult:
    """Reference an earlier source point and optionally clean that point.

    For a lag ``n`` this returns the source value at ``t - n`` aligned to the
    current row ``t``. Local median cleaning, when enabled, is applied to that
    referenced historical point—not to the current row.
    """
    if (
        source_name == "corrected_history"
        and local_cleaning.enabled
    ):
        raise ValueError(
            "Invalid source-cleaning combination: corrected_history cannot "
            "be combined with local median cleaning."
        )

    if offset_points < 0:
        raise ValueError("offset_points cannot be negative.")

    source = sources.select(source_name)
    source = _ensure_numeric_finite_or_nan(
        source,
        context=f"{source_name} source",
    )

    _require_aligned_series(
        reference=source,
        candidates={
            "raw_value": sources.raw_value,
        },
    )

    if statuses is None:
        if local_cleaning.enabled:
            raise ValueError(
                "statuses is required when local median cleaning is enabled."
            )
        status = pd.Series(
            pd.NA,
            index=source.index,
            dtype="Int16",
            name="status",
        )
    else:
        status = pd.to_numeric(statuses, errors="coerce")
        _require_aligned_series(
            reference=source,
            candidates={
                "statuses": status,
            },
        )
        if status.isna().any():
            raise ValueError(
                "statuses contains missing or non-numeric values during "
                "referenced-value preparation."
            )

    referenced_source = source.shift(offset_points)
    referenced_status = status.shift(offset_points)

    if not local_cleaning.enabled:
        empty_mask = pd.Series(
            False,
            index=source.index,
            dtype=bool,
        )

        return ReferencedValueResult(
            values=referenced_source.rename(
                f"{source_name}_reference_{offset_points}"
            ),
            referenced_source=referenced_source,
            referenced_status=referenced_status,
            unacceptable_reference_mask=(
                referenced_source.isna()
            ).astype(bool),
            local_median_replaced_mask=empty_mask.copy(),
            source_fallback_kept_mask=empty_mask.copy(),
            unresolved_mask=referenced_source.isna().astype(bool),
            offset_points=offset_points,
            source_name=source_name,
        )

    half_window_points = local_cleaning.resolve_half_window_points(
        bucket_minutes=bucket_minutes,
    )

    if local_candidates is None:
        local_candidates = prepare_local_median_candidates(
            raw_values=sources.raw_value,
            statuses=status,
            config=local_cleaning,
            half_window_points=half_window_points,
        )
    else:
        if local_candidates.half_window_points != half_window_points:
            raise ValueError(
                "Supplied local-median candidates use a different half-window "
                "from the requested cleaning configuration."
            )
        if (
            local_candidates.resolved_right_window_points
            > half_window_points
        ):
            raise ValueError(
                "Supplied local-median candidates use a right-side window "
                "larger than the configured half-window."
            )
        if (
            local_candidates.minimum_valid_points
            != local_cleaning.minimum_valid_points
        ):
            raise ValueError(
                "Supplied local-median candidates use a different "
                "minimum_valid_points setting."
            )
        if (
            local_candidates.accepted_statuses
            != frozenset(local_cleaning.accepted_statuses)
        ):
            raise ValueError(
                "Supplied local-median candidates use different "
                "accepted_statuses from the requested cleaning configuration."
            )
        _require_aligned_series(
            reference=source,
            candidates={
                "local_candidates": local_candidates.values,
            },
        )

    referenced_candidate = local_candidates.values.shift(offset_points)

    unacceptable_reference_mask = (
        referenced_status.isin(local_cleaning.bad_statuses)
        | referenced_source.isna()
    )

    cleaned = referenced_source.copy()

    local_replacement_mask = (
        unacceptable_reference_mask
        & referenced_candidate.notna()
    )
    cleaned.loc[local_replacement_mask] = referenced_candidate.loc[
        local_replacement_mask
    ]

    unresolved_after_local = (
        unacceptable_reference_mask
        & referenced_candidate.isna()
    )

    source_fallback_kept_mask = pd.Series(
        False,
        index=source.index,
        dtype=bool,
    )

    if local_cleaning.unavailable_strategy == "nan":
        cleaned.loc[unresolved_after_local] = np.nan

    elif local_cleaning.unavailable_strategy == "keep_source":
        source_fallback_kept_mask = (
            unresolved_after_local
            & referenced_source.notna()
        )

    else:
        raise ValueError(
            "Unsupported local-cleaning unavailable strategy: "
            f"{local_cleaning.unavailable_strategy!r}"
        )

    unresolved_mask = cleaned.isna()

    cleaned.name = f"{source_name}_reference_{offset_points}"

    return ReferencedValueResult(
        values=cleaned.astype("float64"),
        referenced_source=referenced_source.astype("float64"),
        referenced_status=referenced_status,
        unacceptable_reference_mask=unacceptable_reference_mask.astype(bool),
        local_median_replaced_mask=local_replacement_mask.astype(bool),
        source_fallback_kept_mask=source_fallback_kept_mask.astype(bool),
        unresolved_mask=unresolved_mask.astype(bool),
        offset_points=offset_points,
        source_name=source_name,
    )


# ---------------------------------------------------------------------------
# Runner-facing source preparation
# ---------------------------------------------------------------------------


def prepare_feature_sources(
    site_dataframe: pd.DataFrame,
    *,
    value_column: str,
    resolved_config: ResolvedFeatureEngineeringConfig,
    status_column: str = DEFAULT_STATUS_COLUMN,
) -> PreparedFeatureSources:
    """Prepare all shared value sources required by one valid site.

    The runner should call this once after pre-source validation. Local median
    candidates are not generated automatically because different requested
    features may use different local-cleaning configurations.
    """

    raw_value = build_raw_value_series(
        site_dataframe,
        value_column=value_column,
    )
    raw_value = _ensure_numeric_finite_or_nan(
        raw_value,
        context="raw feature source",
    )
    raw_value.index = site_dataframe.index
    raw_value.name = "raw_value"

    statuses: pd.Series | None = None

    if (
        resolved_config.requires_status_column
        or resolved_config.uses_corrected_history
        or resolved_config.uses_local_median_cleaning
    ):
        statuses = _normalize_status_series(
            site_dataframe,
            status_column=status_column,
            expected_index=raw_value.index,
        )

    corrected_result: CorrectedHistoryResult | None = None
    corrected_history: pd.Series | None = None

    if resolved_config.uses_corrected_history:
        if statuses is None:
            raise RuntimeError(
                "Corrected history requires a prepared status Series."
            )

        window_points = resolved_config.corrected_history_window_points

        if window_points is None:
            raise ValueError(
                "Resolved configuration uses corrected_history but does not "
                "provide corrected_history_window_points."
            )

        corrected_result = prepare_corrected_history(
            raw_values=raw_value,
            statuses=statuses,
            config=resolved_config.corrected_history,
            window_points=window_points,
        )
        corrected_history = corrected_result.values

    bundle = build_feature_source_bundle(
        site_dataframe,
        value_column=value_column,
        corrected_history=corrected_history,
    )

    return PreparedFeatureSources(
        bundle=bundle,
        statuses=statuses,
        corrected_history_result=corrected_result,
    )


__all__ = [
    "CorrectedHistoryResult",
    "LocalMedianCandidateResult",
    "PreparedFeatureSources",
    "ReferencedValueResult",
    "prepare_corrected_history",
    "prepare_feature_sources",
    "prepare_local_median_candidates",
    "prepare_referenced_values",
]