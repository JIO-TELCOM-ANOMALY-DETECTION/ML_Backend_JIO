"""
Fill original Phase 1 missing-value gaps using bounded local anchors.

This module runs after ``missing_gap_analysis.py``. It expects a dataset that
already contains the immutable ``Original_Gap_Phase`` classification and
processes only rows belonging to Phase 1 gaps.

The preprocessing runner supplies:

- the exact classified dataset path returned by missing-gap analysis;
- the output directory;
- the shared dataset column mapping;
- the validated ``PreprocessingConfig``; and
- the frequency-dependent ``ResolvedFrequencyConfig``.

Phase 1 filling ladder
----------------------

1. ``phase1_linear_interpolation``
   Use trusted anchors immediately before and after the complete original gap
   and linearly interpolate every point in the gap.

2. ``phase1_neighbor_weighted``
   Use the nearest trusted anchors on both sides when both are within the
   configured two-sided distance and their distances are sufficiently
   balanced. Each target point is weighted by its actual distance from the two
   anchors.

3. ``phase1_neighbor_before_only`` or
   ``phase1_neighbor_after_only``
   When two-sided filling is unavailable or rejected, use the closer trusted
   anchor for the complete gap. A tie is resolved in favour of the previous
   anchor.

4. Failure
   Leave the value, current status, and fill method unresolved so a later phase
   may retry the original gap.

Trust rules
-----------

A reference is trusted only when:

- ``Current_Status_Label`` is ``1`` or ``3``;
- its KPI value is finite; and
- its KPI value lies inside the configured valid range.

Outliers (status ``2``) are never used, but the bounded search may pass them.
Unresolved missing rows (``-2``, ``-1``, ``0``), leakage fills (``4``), missing
values, and unexpected states are hard search barriers. Earlier safe Phase 1
fills (status ``3``) may be reused as anchors for later Phase 1 gaps.

Successful fills update only:

- the configured KPI value column;
- ``Current_Status_Label`` to ``3``; and
- ``Fill_Method``.

``Original_Condition_Label`` and ``Original_Gap_Phase`` are never modified.
Invalid site series are skipped completely.
"""

from __future__ import annotations

from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.exceptions import (
    FileProcessingError,
    InputValidationError,
)
from app.ml.eda.data_integrity import (
    COMPOSITE_ID_COLUMN,
    COMPOSITE_ID_SEPARATOR,
    MISSING_ID_VALUE,
)
from app.schemas.common.dataset import DatasetColumnMapping
from app.schemas.preprocessing import (
    PreprocessingConfig,
    ResolvedFrequencyConfig,
)
from app.services.file_service import (
    ensure_directory,
    read_csv_file,
    sanitise_filename,
    write_csv_file,
)

from .missing_gap_analysis import (
    MISSING_ORIGINAL_LABELS,
    ORIGINAL_GAP_PHASE_COLUMN,
    PHASE_1,
)
from .timeline_correction import (
    CURRENT_STATUS_LABEL_COLUMN,
    INVALID_USABILITY_VALUE,
    KPI_USABILITY_COLUMN,
    ORIGINAL_CONDITION_LABEL_COLUMN,
    VALID_USABILITY_VALUE,
)


FILL_METHOD_COLUMN = "Fill_Method"

METHOD_LINEAR_INTERPOLATION = "phase1_linear_interpolation"
METHOD_NEIGHBOR_WEIGHTED = "phase1_neighbor_weighted"
METHOD_NEIGHBOR_BEFORE_ONLY = "phase1_neighbor_before_only"
METHOD_NEIGHBOR_AFTER_ONLY = "phase1_neighbor_after_only"

OUTPUT_FILE_SUFFIX = "_phase1_filled.csv"
SUMMARY_FILE_SUFFIX = "_phase1_summary.csv"

TRUSTED_CURRENT_LABELS = {1, 3}
UNRESOLVED_CURRENT_LABELS = {-2, -1, 0}

_ALLOWED_ORIGINAL_LABELS = {-2, -1, 0, 1}
_ALLOWED_INPUT_CURRENT_LABELS = {-2, -1, 0, 1, 2}
_ALLOWED_GAP_PHASES = {1, 2, 3, 4, 5}
_ALLOWED_USABILITY_VALUES = {
    VALID_USABILITY_VALUE,
    INVALID_USABILITY_VALUE,
}

_STATUS_PROCESSED = "processed"
_STATUS_SKIPPED_UNUSABLE = "skipped_unusable_series"
_STATUS_NO_PHASE1_GAPS = "no_phase1_gaps"

_INTERNAL_TIMESTAMP_COLUMN = "__phase1_timestamp__"
_INTERNAL_ROW_ORDER_COLUMN = "__phase1_row_order__"


@dataclass(frozen=True, slots=True)
class Phase1Anchor:
    """
    Store one trusted local anchor found for a Phase 1 gap.

    Attributes:
        position:
            Zero-based position in the timestamp-sorted site arrays.

        value:
            Finite trusted KPI value.

        boundary_distance_points:
            Number of bucket steps from the nearest boundary of the target gap.
    """

    position: int
    value: float
    boundary_distance_points: int


@dataclass(frozen=True, slots=True)
class Phase1GapOutcome:
    """
    Store the result of attempting one complete Phase 1 gap.

    Attributes:
        filled:
            Whether every point in the gap was filled successfully.

        method:
            Applied fill method, or ``None`` when unresolved.

        fill_values:
            One generated value per target row, ordered chronologically.

        before_anchor_distance:
            Distance from the gap start to the previous trusted anchor.

        after_anchor_distance:
            Distance from the gap end to the following trusted anchor.
    """

    filled: bool
    method: str | None
    fill_values: tuple[float, ...]
    before_anchor_distance: int | None
    after_anchor_distance: int | None


@dataclass(frozen=True, slots=True)
class Phase1FillingData:
    """
    Store the in-memory Phase 1 filling result.

    Attributes:
        dataframe:
            Dataset copy containing Phase 1 values, status updates, and
            ``Fill_Method``.

        summary:
            One-row-per-site Phase 1 summary.

        total_series:
            Number of unique site series.

        processed_series:
            Number of usable sites containing at least one Phase 1 gap.

        skipped_series:
            Number of unusable sites containing Phase 1 gaps.

        eligible_gaps:
            Number of original Phase 1 gaps.

        eligible_points:
            Number of rows belonging to original Phase 1 gaps.

        filled_gaps:
            Number of gaps filled completely.

        filled_points:
            Number of values filled successfully.

        unresolved_gaps:
            Number of eligible gaps left unresolved.

        unresolved_points:
            Number of eligible rows left unresolved.

        linear_points:
            Number of rows filled by immediate two-sided interpolation.

        weighted_points:
            Number of rows filled by bounded two-sided weighted anchors.

        before_only_points:
            Number of rows filled from the previous anchor only.

        after_only_points:
            Number of rows filled from the following anchor only.
    """

    dataframe: pd.DataFrame
    summary: pd.DataFrame
    total_series: int
    processed_series: int
    skipped_series: int
    eligible_gaps: int
    eligible_points: int
    filled_gaps: int
    filled_points: int
    unresolved_gaps: int
    unresolved_points: int
    linear_points: int
    weighted_points: int
    before_only_points: int
    after_only_points: int


@dataclass(frozen=True, slots=True)
class Phase1FillingResult:
    """
    Store saved Phase 1 outputs and headline statistics.

    Attributes:
        source_file:
            Missing-gap-classified input dataset.

        output_file:
            Newly written Phase 1 dataset.

        summary_file:
            Per-site Phase 1 summary CSV.

        total_rows:
            Number of rows preserved in the output.

        total_series:
            Number of unique site series.

        processed_series:
            Number of usable sites containing Phase 1 gaps.

        skipped_series:
            Number of unusable sites containing Phase 1 gaps.

        eligible_gaps:
            Number of original Phase 1 gaps.

        eligible_points:
            Number of rows in original Phase 1 gaps.

        filled_gaps:
            Number of gaps filled completely.

        filled_points:
            Number of values filled successfully.

        unresolved_gaps:
            Number of Phase 1 gaps left unresolved.

        unresolved_points:
            Number of Phase 1 rows left unresolved.

        linear_points:
            Number of linear-interpolation fills.

        weighted_points:
            Number of weighted-neighbour fills.

        before_only_points:
            Number of previous-anchor-only fills.

        after_only_points:
            Number of following-anchor-only fills.
    """

    source_file: Path
    output_file: Path
    summary_file: Path
    total_rows: int
    total_series: int
    processed_series: int
    skipped_series: int
    eligible_gaps: int
    eligible_points: int
    filled_gaps: int
    filled_points: int
    unresolved_gaps: int
    unresolved_points: int
    linear_points: int
    weighted_points: int
    before_only_points: int
    after_only_points: int


def add_composite_id(
    dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
) -> pd.DataFrame:
    """
    Return a dataframe copy containing the shared composite site identifier.

    Args:
        dataframe:
            Dataset containing the configured site identity columns.

        columns:
            Shared physical dataset column mapping.

    Returns:
        pd.DataFrame:
            Copy containing ``Composite_ID``.

    Raises:
        KeyError:
            If a configured site identity column is missing.
    """

    result = dataframe.copy()
    site_columns = list(
        columns.get_site_identity_columns()
    )

    identity_values = (
        result[site_columns]
        .astype("string")
        .fillna(MISSING_ID_VALUE)
    )

    result[COMPOSITE_ID_COLUMN] = (
        identity_values.agg(
            COMPOSITE_ID_SEPARATOR.join,
            axis=1,
        )
    )

    return result


def parse_integer_column(
    dataframe: pd.DataFrame,
    column_name: str,
    allowed_values: set[int],
    *,
    nullable: bool = False,
) -> pd.Series:
    """
    Parse and validate one integer-valued processing column.

    Args:
        dataframe:
            Dataset containing the target column.

        column_name:
            Column to parse.

        allowed_values:
            Accepted integer values.

        nullable:
            Whether missing values are accepted.

    Returns:
        pd.Series:
            Parsed nullable or non-null integer series.

    Raises:
        InputValidationError:
            If values are non-integral, unsupported, or unexpectedly missing.
    """

    numeric = pd.to_numeric(
        dataframe[column_name],
        errors="coerce",
    )

    invalid_mask = (
        numeric.notna()
        & (
            ~np.isfinite(numeric)
            | numeric.mod(1).ne(0)
            | ~numeric.isin(allowed_values)
        )
    )

    if not nullable:
        invalid_mask = invalid_mask | numeric.isna()

    if invalid_mask.any():
        raise InputValidationError(
            f"{column_name} contains invalid values for Phase 1 filling.",
            error_code="INVALID_PHASE1_INTEGER_COLUMN",
            details={
                "column_name": column_name,
                "allowed_values": sorted(allowed_values),
                "nullable": nullable,
                "invalid_rows": int(invalid_mask.sum()),
                "sample_values": (
                    dataframe.loc[
                        invalid_mask,
                        column_name,
                    ]
                    .head(20)
                    .astype("string")
                    .tolist()
                ),
            },
        )

    if nullable:
        return numeric.astype("Int8")

    return numeric.astype(np.int8)


def validate_required_columns(
    dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
) -> list[str]:
    """
    Validate the missing-gap-classified input schema.

    Args:
        dataframe:
            Input dataset from ``missing_gap_analysis.py``.

        columns:
            Shared dataset column mapping.

    Returns:
        list[str]:
            Input columns in their existing order.

    Raises:
        InputValidationError:
            If the dataset is empty, required fields are missing, or Phase 1
            output columns already exist.
    """

    if dataframe.empty:
        raise InputValidationError(
            "Phase 1 cannot process an empty dataframe.",
            error_code="EMPTY_PHASE1_DATASET",
        )

    required_columns = [
        *columns.get_site_identity_columns(),
        columns.timestamp_column,
        columns.value_column,
        ORIGINAL_CONDITION_LABEL_COLUMN,
        CURRENT_STATUS_LABEL_COLUMN,
        KPI_USABILITY_COLUMN,
        ORIGINAL_GAP_PHASE_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise InputValidationError(
            "The missing-gap-classified dataset is missing columns required "
            "for Phase 1 filling.",
            error_code="MISSING_PHASE1_COLUMNS",
            details={
                "missing_columns": missing_columns,
                "required_columns": required_columns,
                "available_columns": list(dataframe.columns),
            },
        )

    if FILL_METHOD_COLUMN in dataframe.columns:
        raise InputValidationError(
            "The dataset already contains Fill_Method. Phase 1 will not "
            "overwrite an earlier filling result.",
            error_code="PHASE1_FILL_METHOD_ALREADY_EXISTS",
        )

    reserved_columns = {
        COMPOSITE_ID_COLUMN,
        _INTERNAL_TIMESTAMP_COLUMN,
        _INTERNAL_ROW_ORDER_COLUMN,
    }

    conflicts = sorted(
        reserved_columns.intersection(dataframe.columns)
    )

    if conflicts:
        raise InputValidationError(
            "The dataset contains reserved Phase 1 internal columns.",
            error_code="RESERVED_PHASE1_COLUMN_CONFLICT",
            details={
                "conflicting_columns": conflicts,
            },
        )

    return list(dataframe.columns)


def normalize_usability(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """
    Validate and normalize site usability values.

    Args:
        dataframe:
            Working dataframe containing ``Composite_ID``.

    Returns:
        pd.Series:
            Lower-case normalized usability values.

    Raises:
        InputValidationError:
            If values are unsupported or inconsistent within a site.
    """

    usability = (
        dataframe[KPI_USABILITY_COLUMN]
        .astype("string")
        .str.strip()
        .str.casefold()
    )

    invalid_mask = (
        usability.isna()
        | ~usability.isin(_ALLOWED_USABILITY_VALUES)
    )

    if invalid_mask.any():
        raise InputValidationError(
            "kpi_usability contains invalid values for Phase 1.",
            error_code="INVALID_PHASE1_USABILITY",
            details={
                "allowed_values": sorted(
                    _ALLOWED_USABILITY_VALUES
                ),
                "invalid_rows": int(invalid_mask.sum()),
                "sample_values": (
                    dataframe.loc[
                        invalid_mask,
                        KPI_USABILITY_COLUMN,
                    ]
                    .head(20)
                    .astype("string")
                    .tolist()
                ),
            },
        )

    site_counts = (
        pd.DataFrame(
            {
                COMPOSITE_ID_COLUMN: dataframe[
                    COMPOSITE_ID_COLUMN
                ],
                KPI_USABILITY_COLUMN: usability,
            }
        )
        .groupby(
            COMPOSITE_ID_COLUMN,
            dropna=False,
        )[KPI_USABILITY_COLUMN]
        .nunique(dropna=False)
    )

    inconsistent_sites = site_counts.loc[
        site_counts.ne(1)
    ]

    if not inconsistent_sites.empty:
        raise InputValidationError(
            "kpi_usability is not constant within one or more site series.",
            error_code="INCONSISTENT_PHASE1_USABILITY",
            details={
                "sample_composite_ids": (
                    inconsistent_sites.index
                    .astype(str)
                    .tolist()[:20]
                ),
            },
        )

    return usability


def prepare_phase1_dataframe(
    dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
    config: PreprocessingConfig,
    resolved_frequency: ResolvedFrequencyConfig,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Validate and prepare the classified dataset for Phase 1 filling.

    Args:
        dataframe:
            Missing-gap-classified input dataset.

        columns:
            Shared dataset column mapping.

        config:
            Validated preprocessing configuration.

        resolved_frequency:
            Frequency-dependent point settings resolved by the runner.

    Returns:
        tuple[pd.DataFrame, list[str]]:
            Prepared working dataframe and original column order.

    Raises:
        InputValidationError:
            If timestamps, labels, values, usability, phase assignments,
            frequency spacing, or gap sizes are inconsistent.
    """

    input_columns = validate_required_columns(
        dataframe=dataframe,
        columns=columns,
    )

    result = add_composite_id(
        dataframe=dataframe,
        columns=columns,
    )

    timestamps = pd.to_datetime(
        result[columns.timestamp_column],
        errors="coerce",
    )

    invalid_timestamp_mask = timestamps.isna()

    if invalid_timestamp_mask.any():
        raise InputValidationError(
            "The Phase 1 input contains missing or invalid timestamps.",
            error_code="INVALID_PHASE1_TIMESTAMPS",
            details={
                "invalid_rows": int(
                    invalid_timestamp_mask.sum()
                ),
                "sample_values": (
                    result.loc[
                        invalid_timestamp_mask,
                        columns.timestamp_column,
                    ]
                    .head(20)
                    .astype("string")
                    .tolist()
                ),
            },
        )

    result[_INTERNAL_TIMESTAMP_COLUMN] = timestamps
    result[_INTERNAL_ROW_ORDER_COLUMN] = np.arange(
        len(result),
        dtype=np.int64,
    )

    original_labels = parse_integer_column(
        dataframe=result,
        column_name=ORIGINAL_CONDITION_LABEL_COLUMN,
        allowed_values=_ALLOWED_ORIGINAL_LABELS,
    )
    current_labels = parse_integer_column(
        dataframe=result,
        column_name=CURRENT_STATUS_LABEL_COLUMN,
        allowed_values=_ALLOWED_INPUT_CURRENT_LABELS,
    )
    gap_phases = parse_integer_column(
        dataframe=result,
        column_name=ORIGINAL_GAP_PHASE_COLUMN,
        allowed_values=_ALLOWED_GAP_PHASES,
        nullable=True,
    )

    result[ORIGINAL_CONDITION_LABEL_COLUMN] = original_labels
    result[CURRENT_STATUS_LABEL_COLUMN] = current_labels
    result[ORIGINAL_GAP_PHASE_COLUMN] = gap_phases

    invalid_transition_mask = (
        (
            original_labels.isin(MISSING_ORIGINAL_LABELS)
            & current_labels.ne(original_labels)
        )
        | (
            original_labels.eq(1)
            & ~current_labels.isin({1, 2})
        )
    )

    if invalid_transition_mask.any():
        raise InputValidationError(
            "Original and current labels contain invalid transitions before "
            "Phase 1.",
            error_code="INVALID_PRE_PHASE1_STATUS_TRANSITION",
            details={
                "invalid_rows": int(
                    invalid_transition_mask.sum()
                ),
            },
        )

    missing_origin_mask = original_labels.isin(
        MISSING_ORIGINAL_LABELS
    )
    phase_present_mask = gap_phases.notna()

    if (
        (missing_origin_mask & ~phase_present_mask).any()
        or (~missing_origin_mask & phase_present_mask).any()
    ):
        raise InputValidationError(
            "Original_Gap_Phase does not match immutable original missing "
            "labels.",
            error_code="INVALID_PHASE1_GAP_PHASE_ALIGNMENT",
            details={
                "missing_without_phase": int(
                    (
                        missing_origin_mask
                        & ~phase_present_mask
                    ).sum()
                ),
                "non_missing_with_phase": int(
                    (
                        ~missing_origin_mask
                        & phase_present_mask
                    ).sum()
                ),
            },
        )

    numeric_values = pd.to_numeric(
        result[columns.value_column],
        errors="coerce",
    )
    finite_mask = pd.Series(
        np.isfinite(
            numeric_values.to_numpy(
                dtype=float,
                na_value=np.nan,
            )
        ),
        index=result.index,
    )

    missing_with_value = (
        missing_origin_mask
        & numeric_values.notna()
    )
    valid_without_value = (
        original_labels.eq(1)
        & ~finite_mask
    )

    if missing_with_value.any():
        raise InputValidationError(
            "Rows with original labels -2, -1, or 0 must remain missing "
            "before Phase 1.",
            error_code="PHASE1_MISSING_ORIGIN_WITH_VALUE",
            details={
                "invalid_rows": int(
                    missing_with_value.sum()
                ),
            },
        )

    if valid_without_value.any():
        raise InputValidationError(
            "Rows with Original_Condition_Label 1 must contain finite KPI "
            "values before Phase 1.",
            error_code="PHASE1_VALID_ORIGIN_WITHOUT_VALUE",
            details={
                "invalid_rows": int(
                    valid_without_value.sum()
                ),
            },
        )

    lower_bound = config.valid_values.lower_bound
    upper_bound = config.valid_values.upper_bound
    valid_origin_outside_range = (
        original_labels.eq(1)
        & (
            numeric_values.lt(lower_bound)
            | numeric_values.gt(upper_bound)
        )
    )

    if valid_origin_outside_range.any():
        raise InputValidationError(
            "Original-valid values fall outside the configured KPI range.",
            error_code="PHASE1_VALID_VALUE_OUTSIDE_RANGE",
            details={
                "invalid_rows": int(
                    valid_origin_outside_range.sum()
                ),
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            },
        )

    result[columns.value_column] = numeric_values
    result[KPI_USABILITY_COLUMN] = normalize_usability(
        result
    )

    duplicate_mask = result.duplicated(
        subset=[
            COMPOSITE_ID_COLUMN,
            _INTERNAL_TIMESTAMP_COLUMN,
        ],
        keep=False,
    )

    if duplicate_mask.any():
        raise InputValidationError(
            "Duplicate site-timestamp rows remain before Phase 1.",
            error_code="DUPLICATE_TIMESTAMPS_BEFORE_PHASE1",
            details={
                "duplicate_rows": int(duplicate_mask.sum()),
            },
        )

    if resolved_frequency.bucket_minutes <= 0:
        raise InputValidationError(
            "The resolved Phase 1 bucket must be positive.",
            error_code="NON_POSITIVE_PHASE1_BUCKET",
            details={
                "bucket_minutes": (
                    resolved_frequency.bucket_minutes
                ),
            },
        )

    if (
        resolved_frequency.phase_1_maximum_two_sided_distance_points
        > resolved_frequency.phase_1_maximum_anchor_search_points
    ):
        raise InputValidationError(
            "The resolved two-sided Phase 1 distance exceeds the overall "
            "anchor-search distance.",
            error_code="INVALID_RESOLVED_PHASE1_DISTANCES",
        )

    expected_frequency = pd.Timedelta(
        minutes=resolved_frequency.bucket_minutes
    )
    spacing_errors: list[dict[str, Any]] = []

    phase_1_range = resolved_frequency.gap_profile.phase_1

    for composite_id, site_dataframe in result.groupby(
        COMPOSITE_ID_COLUMN,
        sort=False,
        dropna=False,
    ):
        ordered = site_dataframe.sort_values(
            _INTERNAL_TIMESTAMP_COLUMN,
            kind="stable",
        )

        differences = ordered[
            _INTERNAL_TIMESTAMP_COLUMN
        ].diff()
        invalid_spacing = (
            differences.notna()
            & differences.ne(expected_frequency)
        )

        if invalid_spacing.any():
            first_bad = invalid_spacing.loc[
                invalid_spacing
            ].index[0]
            spacing_errors.append(
                {
                    COMPOSITE_ID_COLUMN: str(composite_id),
                    "timestamp": str(
                        result.at[
                            first_bad,
                            _INTERNAL_TIMESTAMP_COLUMN,
                        ]
                    ),
                    "actual_difference": str(
                        differences.loc[first_bad]
                    ),
                    "expected_difference": str(
                        expected_frequency
                    ),
                }
            )

            if len(spacing_errors) >= 20:
                break

        phase1_mask = (
            ordered[ORIGINAL_GAP_PHASE_COLUMN]
            .eq(PHASE_1)
            .fillna(False)
        )

        if not phase1_mask.any():
            continue

        new_gap = (
            phase1_mask
            & ~phase1_mask.shift(
                1,
                fill_value=False,
            )
        )
        gap_ids = new_gap.cumsum()

        for _, gap_dataframe in ordered.loc[
            phase1_mask
        ].groupby(
            gap_ids.loc[phase1_mask],
            sort=True,
        ):
            gap_size = int(len(gap_dataframe))
            maximum = phase_1_range.maximum_points
            size_valid = (
                gap_size >= phase_1_range.minimum_points
                and (
                    maximum is None
                    or gap_size <= maximum
                )
            )

            if not size_valid:
                raise InputValidationError(
                    "A block marked Original_Gap_Phase 1 does not fit the "
                    "resolved Phase 1 gap range.",
                    error_code="PHASE1_BLOCK_SIZE_OUTSIDE_PROFILE",
                    details={
                        COMPOSITE_ID_COLUMN: str(
                            composite_id
                        ),
                        "gap_start": str(
                            gap_dataframe[
                                _INTERNAL_TIMESTAMP_COLUMN
                            ].min()
                        ),
                        "gap_end": str(
                            gap_dataframe[
                                _INTERNAL_TIMESTAMP_COLUMN
                            ].max()
                        ),
                        "gap_size": gap_size,
                        "minimum_points": (
                            phase_1_range.minimum_points
                        ),
                        "maximum_points": maximum,
                    },
                )

    if spacing_errors:
        raise InputValidationError(
            "One or more site timelines are incomplete or off-grid before "
            "Phase 1.",
            error_code="INVALID_PHASE1_TIMELINE_SPACING",
            details={
                "sample": spacing_errors,
            },
        )

    return result, input_columns


def is_trusted_anchor(
    value: float,
    current_label: int,
    config: PreprocessingConfig,
) -> bool:
    """
    Determine whether one row may anchor a Phase 1 fill.

    Args:
        value:
            Candidate KPI value.

        current_label:
            Candidate current status label.

        config:
            Valid-value configuration source.

    Returns:
        bool:
            True only for finite in-range values with current label 1 or 3.

    Raises:
        None.
    """

    return bool(
        current_label in TRUSTED_CURRENT_LABELS
        and np.isfinite(value)
        and config.valid_values.lower_bound
        <= value
        <= config.valid_values.upper_bound
    )


def find_local_anchor(
    *,
    start_position: int,
    direction: int,
    values: np.ndarray,
    current_labels: np.ndarray,
    maximum_distance_points: int,
    config: PreprocessingConfig,
) -> Phase1Anchor | None:
    """
    Find the nearest trusted anchor in one direction from a gap boundary.

    Search may move past outliers, but unresolved missing rows, leakage fills,
    missing values, and unexpected states stop the search.

    Args:
        start_position:
            Position immediately inside the target gap boundary. For a previous
            search this is the first gap position; for a following search it is
            the final gap position.

        direction:
            ``-1`` for previous or ``1`` for following.

        values:
            Current site KPI values.

        current_labels:
            Current site status labels.

        maximum_distance_points:
            Maximum bounded search distance from the gap boundary.

        config:
            Valid-value configuration source.

    Returns:
        Phase1Anchor | None:
            Nearest trusted anchor, or ``None`` when blocked or unavailable.

    Raises:
        ValueError:
            If direction or search distance is invalid.
    """

    if direction not in {-1, 1}:
        raise ValueError(
            "direction must be -1 or 1."
        )

    if maximum_distance_points <= 0:
        raise ValueError(
            "maximum_distance_points must be positive."
        )

    for distance in range(
        1,
        maximum_distance_points + 1,
    ):
        position = start_position + direction * distance

        if position < 0 or position >= len(values):
            return None

        label = int(current_labels[position])
        value = float(values[position])

        if is_trusted_anchor(
            value=value,
            current_label=label,
            config=config,
        ):
            return Phase1Anchor(
                position=position,
                value=value,
                boundary_distance_points=distance,
            )

        if label == 2:
            # Outliers are transparent to the bounded search but never trusted.
            continue

        if label in UNRESOLVED_CURRENT_LABELS:
            return None

        if label == 4:
            return None

        # A missing value or an unexpected state is a hard barrier, even if its
        # label would otherwise look trusted.
        if not np.isfinite(value):
            return None

        return None

    return None


def clip_generated_values(
    values: np.ndarray,
    config: PreprocessingConfig,
) -> np.ndarray:
    """
    Optionally clip generated Phase 1 values to the valid KPI range.

    Args:
        values:
            Generated numeric values.

        config:
            Valid-value configuration source.

    Returns:
        np.ndarray:
            Floating generated values, clipped when configured.

    Raises:
        InputValidationError:
            If generated values are non-finite or outside the valid range when
            clipping is disabled.
    """

    generated = np.asarray(
        values,
        dtype=float,
    )

    if not np.isfinite(generated).all():
        raise InputValidationError(
            "Phase 1 generated a non-finite fill value.",
            error_code="NON_FINITE_PHASE1_FILL",
        )

    lower_bound = config.valid_values.lower_bound
    upper_bound = config.valid_values.upper_bound

    if config.valid_values.clip_generated_values:
        return np.clip(
            generated,
            lower_bound,
            upper_bound,
        )

    outside_range = (
        (generated < lower_bound)
        | (generated > upper_bound)
    )

    if outside_range.any():
        raise InputValidationError(
            "Phase 1 generated values outside the configured KPI range while "
            "generated-value clipping is disabled.",
            error_code="PHASE1_FILL_OUTSIDE_VALID_RANGE",
            details={
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "sample_values": (
                    generated[outside_range][:20].tolist()
                ),
            },
        )

    return generated


def build_linear_fill_values(
    *,
    gap_positions: np.ndarray,
    before_anchor: Phase1Anchor,
    after_anchor: Phase1Anchor,
    config: PreprocessingConfig,
) -> np.ndarray:
    """
    Interpolate every gap point between immediate boundary anchors.

    Args:
        gap_positions:
            Chronological positions belonging to one complete gap.

        before_anchor:
            Trusted anchor immediately before the gap.

        after_anchor:
            Trusted anchor immediately after the gap.

        config:
            Valid-value and clipping settings.

    Returns:
        np.ndarray:
            One interpolated value per gap position.

    Raises:
        InputValidationError:
            If anchor positions do not surround the gap.
    """

    if (
        before_anchor.position >= gap_positions[0]
        or after_anchor.position <= gap_positions[-1]
    ):
        raise InputValidationError(
            "Phase 1 linear anchors do not surround the target gap.",
            error_code="INVALID_PHASE1_LINEAR_ANCHORS",
        )

    total_span = (
        after_anchor.position
        - before_anchor.position
    )

    before_weights = (
        after_anchor.position
        - gap_positions
    ) / total_span
    after_weights = (
        gap_positions
        - before_anchor.position
    ) / total_span

    generated = (
        before_weights * before_anchor.value
        + after_weights * after_anchor.value
    )

    return clip_generated_values(
        values=generated,
        config=config,
    )


def select_phase1_fill(
    *,
    gap_positions: np.ndarray,
    before_anchor: Phase1Anchor | None,
    after_anchor: Phase1Anchor | None,
    config: PreprocessingConfig,
    resolved_frequency: ResolvedFrequencyConfig,
) -> Phase1GapOutcome:
    """
    Apply the configured Phase 1 filling ladder to one complete gap.

    Args:
        gap_positions:
            Chronological positions of the complete original Phase 1 gap.

        before_anchor:
            Nearest previous trusted anchor, when available.

        after_anchor:
            Nearest following trusted anchor, when available.

        config:
            Complete preprocessing configuration.

        resolved_frequency:
            Frequency-dependent Phase 1 point limits.

    Returns:
        Phase1GapOutcome:
            Fill decision and generated values.

    Raises:
        InputValidationError:
            If the supplied gap position sequence is invalid.
    """

    if len(gap_positions) == 0:
        raise InputValidationError(
            "Phase 1 received an empty gap.",
            error_code="EMPTY_PHASE1_GAP",
        )

    if not np.array_equal(
        np.diff(gap_positions),
        np.ones(
            max(len(gap_positions) - 1, 0),
            dtype=int,
        ),
    ):
        raise InputValidationError(
            "Phase 1 gap positions are not consecutive.",
            error_code="NON_CONSECUTIVE_PHASE1_GAP",
        )

    before_distance = (
        before_anchor.boundary_distance_points
        if before_anchor is not None
        else None
    )
    after_distance = (
        after_anchor.boundary_distance_points
        if after_anchor is not None
        else None
    )

    # Level 1: both gap-boundary neighbours are immediately trusted.
    if (
        before_anchor is not None
        and after_anchor is not None
        and before_anchor.boundary_distance_points == 1
        and after_anchor.boundary_distance_points == 1
    ):
        generated = build_linear_fill_values(
            gap_positions=gap_positions,
            before_anchor=before_anchor,
            after_anchor=after_anchor,
            config=config,
        )

        return Phase1GapOutcome(
            filled=True,
            method=METHOD_LINEAR_INTERPOLATION,
            fill_values=tuple(
                float(value)
                for value in generated
            ),
            before_anchor_distance=before_distance,
            after_anchor_distance=after_distance,
        )

    # Level 2: both nearest anchors are local and reasonably balanced.
    if (
        before_anchor is not None
        and after_anchor is not None
    ):
        minimum_distance = min(
            before_anchor.boundary_distance_points,
            after_anchor.boundary_distance_points,
        )
        maximum_distance = max(
            before_anchor.boundary_distance_points,
            after_anchor.boundary_distance_points,
        )
        distance_ratio = (
            maximum_distance / minimum_distance
        )

        if (
            before_anchor.boundary_distance_points
            <= resolved_frequency.phase_1_maximum_two_sided_distance_points
            and after_anchor.boundary_distance_points
            <= resolved_frequency.phase_1_maximum_two_sided_distance_points
            and distance_ratio
            <= config.phase_1.maximum_anchor_distance_ratio
        ):
            generated = build_linear_fill_values(
                gap_positions=gap_positions,
                before_anchor=before_anchor,
                after_anchor=after_anchor,
                config=config,
            )

            return Phase1GapOutcome(
                filled=True,
                method=METHOD_NEIGHBOR_WEIGHTED,
                fill_values=tuple(
                    float(value)
                    for value in generated
                ),
                before_anchor_distance=before_distance,
                after_anchor_distance=after_distance,
            )

    # Levels 3 and 4: choose one local anchor for the complete gap. Ties favour
    # the previous anchor for deterministic behaviour.
    selected_anchor: Phase1Anchor | None = None
    selected_method: str | None = None

    if (
        before_anchor is not None
        and after_anchor is not None
    ):
        if (
            before_anchor.boundary_distance_points
            <= after_anchor.boundary_distance_points
        ):
            selected_anchor = before_anchor
            selected_method = (
                METHOD_NEIGHBOR_BEFORE_ONLY
            )
        else:
            selected_anchor = after_anchor
            selected_method = (
                METHOD_NEIGHBOR_AFTER_ONLY
            )

    elif before_anchor is not None:
        selected_anchor = before_anchor
        selected_method = (
            METHOD_NEIGHBOR_BEFORE_ONLY
        )

    elif after_anchor is not None:
        selected_anchor = after_anchor
        selected_method = (
            METHOD_NEIGHBOR_AFTER_ONLY
        )

    if selected_anchor is not None and selected_method is not None:
        generated = clip_generated_values(
            values=np.full(
                len(gap_positions),
                selected_anchor.value,
                dtype=float,
            ),
            config=config,
        )

        return Phase1GapOutcome(
            filled=True,
            method=selected_method,
            fill_values=tuple(
                float(value)
                for value in generated
            ),
            before_anchor_distance=before_distance,
            after_anchor_distance=after_distance,
        )

    return Phase1GapOutcome(
        filled=False,
        method=None,
        fill_values=(),
        before_anchor_distance=before_distance,
        after_anchor_distance=after_distance,
    )


def find_phase1_gap_positions(
    site_dataframe: pd.DataFrame,
) -> list[np.ndarray]:
    """
    Return chronological position arrays for frozen Phase 1 gap blocks.

    This function does not recalculate gap sizes or phases. It only groups
    adjacent rows already marked ``Original_Gap_Phase == 1`` so the complete
    frozen block can be filled atomically.

    Args:
        site_dataframe:
            Timestamp-sorted site dataframe with a zero-based local index.

    Returns:
        list[np.ndarray]:
            One consecutive local-position array per Phase 1 gap.

    Raises:
        None.
    """

    phase1_mask = (
        site_dataframe[ORIGINAL_GAP_PHASE_COLUMN]
        .eq(PHASE_1)
        .fillna(False)
    )

    if not phase1_mask.any():
        return []

    new_gap = (
        phase1_mask
        & ~phase1_mask.shift(
            1,
            fill_value=False,
        )
    )
    gap_ids = new_gap.cumsum()

    return [
        group.index.to_numpy(
            dtype=np.int64,
        )
        for _, group in site_dataframe.loc[
            phase1_mask
        ].groupby(
            gap_ids.loc[phase1_mask],
            sort=True,
        )
    ]


def process_one_site(
    *,
    site_dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
    config: PreprocessingConfig,
    resolved_frequency: ResolvedFrequencyConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Fill every frozen Phase 1 gap in one site series.

    Args:
        site_dataframe:
            Prepared rows belonging to one site.

        columns:
            Shared dataset column mapping.

        config:
            Validated preprocessing settings.

        resolved_frequency:
            Frequency-dependent Phase 1 settings.

    Returns:
        tuple[pd.DataFrame, dict[str, Any]]:
            Updated site dataframe and one summary record.

    Raises:
        InputValidationError:
            If a target gap no longer has the expected unresolved state or
            internal Phase 1 accounting fails.
    """

    ordered = site_dataframe.sort_values(
        _INTERNAL_TIMESTAMP_COLUMN,
        kind="stable",
    ).copy()
    ordered["__phase1_source_index__"] = ordered.index
    ordered = ordered.reset_index(drop=True)

    values = pd.to_numeric(
        ordered[columns.value_column],
        errors="coerce",
    ).to_numpy(
        dtype=float,
        copy=True,
    )
    current_labels = pd.to_numeric(
        ordered[CURRENT_STATUS_LABEL_COLUMN],
        errors="raise",
    ).to_numpy(
        dtype=np.int8,
        copy=True,
    )
    original_labels = pd.to_numeric(
        ordered[ORIGINAL_CONDITION_LABEL_COLUMN],
        errors="raise",
    ).to_numpy(
        dtype=np.int8,
        copy=False,
    )
    fill_methods = np.full(
        len(ordered),
        "",
        dtype=object,
    )

    phase1_gaps = find_phase1_gap_positions(
        ordered
    )
    eligible_points = int(
        sum(len(gap) for gap in phase1_gaps)
    )

    usability_values = (
        ordered[KPI_USABILITY_COLUMN]
        .astype("string")
        .drop_duplicates()
        .tolist()
    )

    if len(usability_values) != 1:
        raise InputValidationError(
            "A site contains mixed kpi_usability values during Phase 1.",
            error_code="MIXED_PHASE1_SITE_USABILITY",
            details={
                "values": usability_values,
            },
        )

    usability = str(usability_values[0])
    composite_id = str(
        ordered[COMPOSITE_ID_COLUMN].iloc[0]
    )

    summary: dict[str, Any] = {
        COMPOSITE_ID_COLUMN: composite_id,
        KPI_USABILITY_COLUMN: usability,
        "status": (
            _STATUS_NO_PHASE1_GAPS
            if not phase1_gaps
            else _STATUS_PROCESSED
        ),
        "eligible_gaps": len(phase1_gaps),
        "eligible_points": eligible_points,
        "filled_gaps": 0,
        "filled_points": 0,
        "unresolved_gaps": 0,
        "unresolved_points": 0,
        "linear_points": 0,
        "weighted_points": 0,
        "before_only_points": 0,
        "after_only_points": 0,
    }

    if not phase1_gaps:
        ordered[columns.value_column] = values
        ordered[CURRENT_STATUS_LABEL_COLUMN] = (
            current_labels
        )
        ordered[FILL_METHOD_COLUMN] = fill_methods

        return (
            ordered.set_index(
                "__phase1_source_index__"
            ),
            summary,
        )

    if usability != VALID_USABILITY_VALUE:
        summary["status"] = (
            _STATUS_SKIPPED_UNUSABLE
        )
        summary["unresolved_gaps"] = len(
            phase1_gaps
        )
        summary["unresolved_points"] = (
            eligible_points
        )

        ordered[columns.value_column] = values
        ordered[CURRENT_STATUS_LABEL_COLUMN] = (
            current_labels
        )
        ordered[FILL_METHOD_COLUMN] = fill_methods

        return (
            ordered.set_index(
                "__phase1_source_index__"
            ),
            summary,
        )

    maximum_search = (
        resolved_frequency
        .phase_1_maximum_anchor_search_points
    )

    for gap_positions in phase1_gaps:
        gap_original_labels = original_labels[
            gap_positions
        ]
        gap_current_labels = current_labels[
            gap_positions
        ]
        gap_values = values[gap_positions]

        state_valid = bool(
            np.isin(
                gap_original_labels,
                list(MISSING_ORIGINAL_LABELS),
            ).all()
            and np.isin(
                gap_current_labels,
                list(UNRESOLVED_CURRENT_LABELS),
            ).all()
            and np.isnan(gap_values).all()
        )

        if not state_valid:
            raise InputValidationError(
                "A frozen Phase 1 gap no longer has the expected unresolved "
                "state.",
                error_code="PHASE1_TARGET_STATE_MISMATCH",
                details={
                    COMPOSITE_ID_COLUMN: composite_id,
                    "gap_start": str(
                        ordered.loc[
                            gap_positions[0],
                            _INTERNAL_TIMESTAMP_COLUMN,
                        ]
                    ),
                    "gap_end": str(
                        ordered.loc[
                            gap_positions[-1],
                            _INTERNAL_TIMESTAMP_COLUMN,
                        ]
                    ),
                    "gap_size": int(
                        len(gap_positions)
                    ),
                },
            )

        before_anchor = find_local_anchor(
            start_position=int(gap_positions[0]),
            direction=-1,
            values=values,
            current_labels=current_labels,
            maximum_distance_points=maximum_search,
            config=config,
        )
        after_anchor = find_local_anchor(
            start_position=int(gap_positions[-1]),
            direction=1,
            values=values,
            current_labels=current_labels,
            maximum_distance_points=maximum_search,
            config=config,
        )

        outcome = select_phase1_fill(
            gap_positions=gap_positions,
            before_anchor=before_anchor,
            after_anchor=after_anchor,
            config=config,
            resolved_frequency=resolved_frequency,
        )

        if not outcome.filled or outcome.method is None:
            summary["unresolved_gaps"] += 1
            summary["unresolved_points"] += len(
                gap_positions
            )
            continue

        generated = np.asarray(
            outcome.fill_values,
            dtype=float,
        )

        if len(generated) != len(gap_positions):
            raise InputValidationError(
                "Phase 1 generated the wrong number of values for a gap.",
                error_code="PHASE1_GENERATED_LENGTH_MISMATCH",
                details={
                    "gap_points": int(
                        len(gap_positions)
                    ),
                    "generated_points": int(
                        len(generated)
                    ),
                },
            )

        values[gap_positions] = generated
        current_labels[gap_positions] = 3
        fill_methods[gap_positions] = outcome.method

        summary["filled_gaps"] += 1
        summary["filled_points"] += len(
            gap_positions
        )

        method_count_key = {
            METHOD_LINEAR_INTERPOLATION: (
                "linear_points"
            ),
            METHOD_NEIGHBOR_WEIGHTED: (
                "weighted_points"
            ),
            METHOD_NEIGHBOR_BEFORE_ONLY: (
                "before_only_points"
            ),
            METHOD_NEIGHBOR_AFTER_ONLY: (
                "after_only_points"
            ),
        }[outcome.method]
        summary[method_count_key] += len(
            gap_positions
        )

    if (
        summary["filled_gaps"]
        + summary["unresolved_gaps"]
        != summary["eligible_gaps"]
    ):
        raise InputValidationError(
            "Phase 1 gap accounting failed for one site.",
            error_code="PHASE1_GAP_ACCOUNTING_MISMATCH",
            details={
                COMPOSITE_ID_COLUMN: composite_id,
                "summary": summary,
            },
        )

    if (
        summary["filled_points"]
        + summary["unresolved_points"]
        != summary["eligible_points"]
    ):
        raise InputValidationError(
            "Phase 1 point accounting failed for one site.",
            error_code="PHASE1_POINT_ACCOUNTING_MISMATCH",
            details={
                COMPOSITE_ID_COLUMN: composite_id,
                "summary": summary,
            },
        )

    ordered[columns.value_column] = values
    ordered[CURRENT_STATUS_LABEL_COLUMN] = (
        current_labels
    )
    ordered[FILL_METHOD_COLUMN] = fill_methods

    return (
        ordered.set_index(
            "__phase1_source_index__"
        ),
        summary,
    )


def validate_phase1_output(
    *,
    input_dataframe: pd.DataFrame,
    output_dataframe: pd.DataFrame,
    input_columns: list[str],
    columns: DatasetColumnMapping,
    config: PreprocessingConfig,
) -> None:
    """
    Validate Phase 1 row, schema, provenance, and fill invariants.

    Args:
        input_dataframe:
            Original classified dataset.

        output_dataframe:
            Phase 1 result.

        input_columns:
            Input columns in existing order.

        columns:
            Shared dataset column mapping.

        config:
            Valid-value configuration source.

    Returns:
        None.

    Raises:
        InputValidationError:
            If Phase 1 changes protected data or creates inconsistent fills.
    """

    expected_columns = [
        *input_columns,
        FILL_METHOD_COLUMN,
    ]

    if list(output_dataframe.columns) != expected_columns:
        raise InputValidationError(
            "Phase 1 produced an unexpected output schema.",
            error_code="INVALID_PHASE1_OUTPUT_COLUMNS",
            details={
                "expected_columns": expected_columns,
                "actual_columns": list(
                    output_dataframe.columns
                ),
            },
        )

    if len(output_dataframe) != len(input_dataframe):
        raise InputValidationError(
            "Phase 1 changed the dataset row count.",
            error_code="PHASE1_ROW_COUNT_CHANGED",
            details={
                "input_rows": len(input_dataframe),
                "output_rows": len(output_dataframe),
            },
        )

    mutable_columns = {
        columns.value_column,
        CURRENT_STATUS_LABEL_COLUMN,
    }
    protected_columns = [
        column
        for column in input_columns
        if column not in mutable_columns
    ]

    changed_protected_columns = [
        column
        for column in protected_columns
        if not output_dataframe[column].equals(
            input_dataframe[column]
        )
    ]

    if changed_protected_columns:
        raise InputValidationError(
            "Phase 1 modified protected input columns.",
            error_code="PHASE1_MODIFIED_PROTECTED_COLUMNS",
            details={
                "changed_columns": (
                    changed_protected_columns
                ),
            },
        )

    input_values = pd.to_numeric(
        input_dataframe[columns.value_column],
        errors="coerce",
    )
    output_values = pd.to_numeric(
        output_dataframe[columns.value_column],
        errors="coerce",
    )
    input_status = pd.to_numeric(
        input_dataframe[CURRENT_STATUS_LABEL_COLUMN],
        errors="coerce",
    )
    output_status = pd.to_numeric(
        output_dataframe[CURRENT_STATUS_LABEL_COLUMN],
        errors="coerce",
    )
    original_labels = pd.to_numeric(
        output_dataframe[ORIGINAL_CONDITION_LABEL_COLUMN],
        errors="coerce",
    )
    gap_phases = pd.to_numeric(
        output_dataframe[ORIGINAL_GAP_PHASE_COLUMN],
        errors="coerce",
    )
    methods = (
        output_dataframe[FILL_METHOD_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    filled_mask = output_status.eq(3)
    method_mask = methods.ne("")
    phase1_mask = gap_phases.eq(PHASE_1)

    invalid_filled_rows = (
        filled_mask
        & (
            ~phase1_mask
            | ~original_labels.isin(
                MISSING_ORIGINAL_LABELS
            )
            | output_values.isna()
            | ~np.isfinite(output_values)
            | ~method_mask
        )
    )

    if invalid_filled_rows.any():
        raise InputValidationError(
            "Phase 1 produced inconsistent safely filled rows.",
            error_code="INVALID_PHASE1_FILLED_ROWS",
            details={
                "invalid_rows": int(
                    invalid_filled_rows.sum()
                ),
            },
        )

    if (method_mask != filled_mask).any():
        raise InputValidationError(
            "Fill_Method and Current_Status_Label 3 are not aligned.",
            error_code="PHASE1_METHOD_STATUS_MISMATCH",
            details={
                "mismatched_rows": int(
                    (method_mask != filled_mask).sum()
                ),
            },
        )

    allowed_methods = {
        METHOD_LINEAR_INTERPOLATION,
        METHOD_NEIGHBOR_WEIGHTED,
        METHOD_NEIGHBOR_BEFORE_ONLY,
        METHOD_NEIGHBOR_AFTER_ONLY,
    }
    invalid_methods = (
        method_mask
        & ~methods.isin(allowed_methods)
    )

    if invalid_methods.any():
        raise InputValidationError(
            "Phase 1 produced unsupported fill-method values.",
            error_code="INVALID_PHASE1_FILL_METHOD",
            details={
                "sample_values": (
                    methods.loc[invalid_methods]
                    .head(20)
                    .tolist()
                ),
            },
        )

    lower_bound = config.valid_values.lower_bound
    upper_bound = config.valid_values.upper_bound
    filled_outside_range = (
        filled_mask
        & (
            output_values.lt(lower_bound)
            | output_values.gt(upper_bound)
        )
    )

    if filled_outside_range.any():
        raise InputValidationError(
            "Phase 1 filled values outside the configured KPI range.",
            error_code="PHASE1_OUTPUT_OUTSIDE_VALID_RANGE",
            details={
                "invalid_rows": int(
                    filled_outside_range.sum()
                ),
            },
        )

    non_phase1_mask = ~phase1_mask
    changed_non_phase1_value = (
        non_phase1_mask
        & ~(
            output_values.eq(input_values)
            | (
                output_values.isna()
                & input_values.isna()
            )
        )
    )
    changed_non_phase1_status = (
        non_phase1_mask
        & output_status.ne(input_status)
    )

    if (
        changed_non_phase1_value.any()
        or changed_non_phase1_status.any()
    ):
        raise InputValidationError(
            "Phase 1 modified rows outside original Phase 1 gaps.",
            error_code="PHASE1_MODIFIED_NON_TARGET_ROWS",
            details={
                "changed_values": int(
                    changed_non_phase1_value.sum()
                ),
                "changed_statuses": int(
                    changed_non_phase1_status.sum()
                ),
            },
        )

    invalid_site_mask = (
        output_dataframe[KPI_USABILITY_COLUMN]
        .astype("string")
        .str.strip()
        .str.casefold()
        .eq(INVALID_USABILITY_VALUE)
    )
    invalid_site_value_changed = (
        invalid_site_mask
        & ~(
            output_values.eq(input_values)
            | (
                output_values.isna()
                & input_values.isna()
            )
        )
    )
    invalid_site_status_changed = (
        invalid_site_mask
        & output_status.ne(input_status)
    )

    if (
        invalid_site_value_changed.any()
        or invalid_site_status_changed.any()
        or (invalid_site_mask & method_mask).any()
    ):
        raise InputValidationError(
            "Phase 1 modified an invalid site series.",
            error_code="PHASE1_MODIFIED_INVALID_SITE",
        )

    unresolved_phase1_mask = (
        phase1_mask
        & ~filled_mask
    )
    invalid_unresolved_state = (
        unresolved_phase1_mask
        & (
            output_values.notna()
            | ~output_status.isin(
                UNRESOLVED_CURRENT_LABELS
            )
            | method_mask
        )
    )

    if invalid_unresolved_state.any():
        raise InputValidationError(
            "Unresolved Phase 1 rows were not left unchanged.",
            error_code="INVALID_UNRESOLVED_PHASE1_STATE",
            details={
                "invalid_rows": int(
                    invalid_unresolved_state.sum()
                ),
            },
        )


def fill_phase_1_missing_values(
    *,
    dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
    config: PreprocessingConfig,
    resolved_frequency: ResolvedFrequencyConfig,
    logger: Logger,
) -> Phase1FillingData:
    """
    Fill all eligible Phase 1 gaps in one classified KPI dataframe.

    Args:
        dataframe:
            Dataset returned by ``missing_gap_analysis.py``.

        columns:
            Shared dataset column mapping.

        config:
            Complete validated preprocessing configuration.

        resolved_frequency:
            Frequency-dependent Phase 1 point settings.

        logger:
            Run-specific logger.

    Returns:
        Phase1FillingData:
            Updated dataframe, per-site summary, and aggregate counts.

    Raises:
        InputValidationError:
            If input validation, filling, or output invariants fail.
    """

    working, input_columns = prepare_phase1_dataframe(
        dataframe=dataframe,
        columns=columns,
        config=config,
        resolved_frequency=resolved_frequency,
    )

    updated = working.copy()
    updated[FILL_METHOD_COLUMN] = ""
    summary_rows: list[dict[str, Any]] = []

    grouped_indices = updated.groupby(
        COMPOSITE_ID_COLUMN,
        sort=False,
        dropna=False,
    ).groups

    for _, indices in grouped_indices.items():
        site_dataframe = updated.loc[
            indices
        ].copy()

        processed_site, site_summary = process_one_site(
            site_dataframe=site_dataframe,
            columns=columns,
            config=config,
            resolved_frequency=resolved_frequency,
        )

        source_indices = processed_site.index
        updated.loc[
            source_indices,
            columns.value_column,
        ] = processed_site[
            columns.value_column
        ].to_numpy()
        updated.loc[
            source_indices,
            CURRENT_STATUS_LABEL_COLUMN,
        ] = processed_site[
            CURRENT_STATUS_LABEL_COLUMN
        ].to_numpy()
        updated.loc[
            source_indices,
            FILL_METHOD_COLUMN,
        ] = processed_site[
            FILL_METHOD_COLUMN
        ].to_numpy()

        summary_rows.append(site_summary)

    aligned_updates = updated.reindex(
        dataframe.index
    )

    output_dataframe = dataframe.copy()
    output_dataframe[columns.value_column] = (
        aligned_updates[columns.value_column]
        .to_numpy()
    )
    output_dataframe[CURRENT_STATUS_LABEL_COLUMN] = (
        aligned_updates[CURRENT_STATUS_LABEL_COLUMN]
        .to_numpy()
    )
    output_dataframe[FILL_METHOD_COLUMN] = (
        aligned_updates[FILL_METHOD_COLUMN]
        .to_numpy()
    )

    validate_phase1_output(
        input_dataframe=dataframe,
        output_dataframe=output_dataframe,
        input_columns=input_columns,
        columns=columns,
        config=config,
    )

    summary = pd.DataFrame(
        summary_rows,
        columns=[
            COMPOSITE_ID_COLUMN,
            KPI_USABILITY_COLUMN,
            "status",
            "eligible_gaps",
            "eligible_points",
            "filled_gaps",
            "filled_points",
            "unresolved_gaps",
            "unresolved_points",
            "linear_points",
            "weighted_points",
            "before_only_points",
            "after_only_points",
        ],
    )

    aggregate_columns = [
        "eligible_gaps",
        "eligible_points",
        "filled_gaps",
        "filled_points",
        "unresolved_gaps",
        "unresolved_points",
        "linear_points",
        "weighted_points",
        "before_only_points",
        "after_only_points",
    ]
    totals = {
        column: int(
            pd.to_numeric(
                summary[column],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )
        for column in aggregate_columns
    }

    processed_series = int(
        summary["status"].eq(
            _STATUS_PROCESSED
        ).sum()
    )
    skipped_series = int(
        summary["status"].eq(
            _STATUS_SKIPPED_UNUSABLE
        ).sum()
    )

    logger.info(
        "Phase 1 filling completed: eligible_gaps=%d, eligible_points=%d, "
        "filled_gaps=%d, filled_points=%d, unresolved_gaps=%d, "
        "unresolved_points=%d, linear=%d, weighted=%d, before_only=%d, "
        "after_only=%d, processed_series=%d, skipped_series=%d.",
        totals["eligible_gaps"],
        totals["eligible_points"],
        totals["filled_gaps"],
        totals["filled_points"],
        totals["unresolved_gaps"],
        totals["unresolved_points"],
        totals["linear_points"],
        totals["weighted_points"],
        totals["before_only_points"],
        totals["after_only_points"],
        processed_series,
        skipped_series,
    )

    return Phase1FillingData(
        dataframe=output_dataframe,
        summary=summary,
        total_series=len(summary),
        processed_series=processed_series,
        skipped_series=skipped_series,
        eligible_gaps=totals["eligible_gaps"],
        eligible_points=totals["eligible_points"],
        filled_gaps=totals["filled_gaps"],
        filled_points=totals["filled_points"],
        unresolved_gaps=totals["unresolved_gaps"],
        unresolved_points=totals["unresolved_points"],
        linear_points=totals["linear_points"],
        weighted_points=totals["weighted_points"],
        before_only_points=totals[
            "before_only_points"
        ],
        after_only_points=totals[
            "after_only_points"
        ],
    )


def save_phase_1_outputs(
    *,
    filling_data: Phase1FillingData,
    source_file: Path,
    output_directory: Path,
) -> tuple[Path, Path]:
    """
    Save the Phase 1 dataset and per-site summary.

    Args:
        filling_data:
            In-memory Phase 1 result.

        source_file:
            Classified source file used for naming.

        output_directory:
            Destination directory supplied by the preprocessing runner.

    Returns:
        tuple[Path, Path]:
            Written dataset path followed by summary path.

    Raises:
        FileProcessingError:
            If output files cannot be created.
    """

    resolved_output_directory = ensure_directory(
        output_directory
    )
    safe_stem = sanitise_filename(
        source_file.stem
    )

    output_file = (
        resolved_output_directory
        / f"{safe_stem}{OUTPUT_FILE_SUFFIX}"
    )
    summary_file = (
        resolved_output_directory
        / f"{safe_stem}{SUMMARY_FILE_SUFFIX}"
    )

    write_csv_file(
        filling_data.dataframe,
        output_file,
    )
    write_csv_file(
        filling_data.summary,
        summary_file,
    )

    return output_file, summary_file


def create_phase_1_filled_dataset(
    *,
    source_file: str | Path,
    output_directory: str | Path,
    columns: DatasetColumnMapping,
    config: PreprocessingConfig,
    resolved_frequency: ResolvedFrequencyConfig,
    logger: Logger,
) -> Phase1FillingResult:
    """
    Read, fill, and save one Phase 1 KPI dataset.

    The preprocessing runner should pass
    ``missing_gap_result.output_file`` as ``source_file``. It should resolve
    ``PreprocessingConfig.resolve_for_bucket`` once and reuse that result for
    every preprocessing stage.

    Args:
        source_file:
            Exact missing-gap-classified dataset path from the previous stage.

        output_directory:
            KPI preprocessing directory selected by the runner.

        columns:
            Shared dataset column mapping.

        config:
            Complete validated preprocessing configuration.

        resolved_frequency:
            Frequency-dependent Phase 1 point settings.

        logger:
            Run-specific logger.

    Returns:
        Phase1FillingResult:
            Saved paths and Phase 1 statistics.

    Raises:
        FileProcessingError:
            If the input or output files cannot be processed.

        InputValidationError:
            If Phase 1 input, filling, or output invariants fail.
    """

    resolved_source_file = Path(
        source_file
    ).expanduser().resolve()
    resolved_output_directory = Path(
        output_directory
    ).expanduser().resolve()

    if not resolved_source_file.is_file():
        raise FileProcessingError(
            "The missing-gap-classified dataset does not exist.",
            error_code="MISSING_PHASE1_SOURCE_FILE",
            details={
                "source_file": str(
                    resolved_source_file
                ),
            },
        )

    logger.info(
        "Starting Phase 1 missing-value filling for %s.",
        resolved_source_file.name,
    )

    dataframe = read_csv_file(
        resolved_source_file
    )

    filling_data = fill_phase_1_missing_values(
        dataframe=dataframe,
        columns=columns,
        config=config,
        resolved_frequency=resolved_frequency,
        logger=logger,
    )

    output_file, summary_file = save_phase_1_outputs(
        filling_data=filling_data,
        source_file=resolved_source_file,
        output_directory=resolved_output_directory,
    )

    result = Phase1FillingResult(
        source_file=resolved_source_file,
        output_file=output_file,
        summary_file=summary_file,
        total_rows=len(filling_data.dataframe),
        total_series=filling_data.total_series,
        processed_series=(
            filling_data.processed_series
        ),
        skipped_series=filling_data.skipped_series,
        eligible_gaps=filling_data.eligible_gaps,
        eligible_points=filling_data.eligible_points,
        filled_gaps=filling_data.filled_gaps,
        filled_points=filling_data.filled_points,
        unresolved_gaps=filling_data.unresolved_gaps,
        unresolved_points=(
            filling_data.unresolved_points
        ),
        linear_points=filling_data.linear_points,
        weighted_points=filling_data.weighted_points,
        before_only_points=(
            filling_data.before_only_points
        ),
        after_only_points=(
            filling_data.after_only_points
        ),
    )

    logger.info(
        "Phase 1 dataset saved to %s; summary saved to %s.",
        result.output_file,
        result.summary_file,
    )

    return result


__all__ = [
    "FILL_METHOD_COLUMN",
    "METHOD_LINEAR_INTERPOLATION",
    "METHOD_NEIGHBOR_AFTER_ONLY",
    "METHOD_NEIGHBOR_BEFORE_ONLY",
    "METHOD_NEIGHBOR_WEIGHTED",
    "Phase1Anchor",
    "Phase1FillingData",
    "Phase1FillingResult",
    "Phase1GapOutcome",
    "create_phase_1_filled_dataset",
    "fill_phase_1_missing_values",
    "find_local_anchor",
    "select_phase1_fill",
]