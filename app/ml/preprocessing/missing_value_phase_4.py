"""
Fill remaining missing values with two ordered Prophet passes.

This module is the final missing-value filling stage. It runs after
``missing_value_phase_3.py`` and uses the immutable ``Original_Gap_Phase``
classification created by ``missing_gap_analysis.py``.

Two ordered passes
------------------

Pass 1 processes:

- primary targets: original gaps classified as Phase 4 Pass 1
  (``Original_Gap_Phase == 4``); and
- retry targets: unresolved points from original Phase 1, 2, and 3 gaps.

Pass 2 processes:

- primary targets: original gaps classified as Phase 4 Pass 2
  (``Original_Gap_Phase == 5``).

A fixed training snapshot is taken at the beginning of each pass. Prophet fills
created earlier within the same pass are therefore not reused for training.
Pass 1 safe fills with status label ``3`` are visible in the fresh Pass 2
snapshot. Pass 1 leakage-risk fills with status label ``4`` are excluded.

Prophet modes
-------------

For every target gap, the script first attempts a past-only Prophet model
trained strictly before the original gap start.

- successful past-only predictions receive status label ``3``;
- method names end with ``prophet_forward_safe``.

When enabled, unresolved points then use a cached full-series Prophet fallback
trained from all trusted rows in the fixed pass snapshot.

- successful full-series predictions receive status label ``4``;
- method names end with ``prophet_full_series_leakage``.

Trusted training rows have ``Current_Status_Label`` in ``{1, 3}``, finite KPI
values, and values inside the configured physical range. Outliers, unresolved
points, and label-4 fills are never training references.

Runner responsibility
---------------------

The preprocessing runner should inspect ``config.phase_4.enabled``. When it is
false, the runner should skip this module entirely. This module also rejects an
accidental call with Phase 4 disabled.

The ``prophet`` package is imported lazily only when Phase 4 actually runs.
Therefore, importing the backend remains possible when Prophet is disabled and
the optional package is not installed.

Only these dataset fields may change:

- the configured KPI value column;
- ``Current_Status_Label``; and
- ``Fill_Method``.

All other columns, row count, and row order are preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
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
    PHASE_2,
    PHASE_3,
    PHASE_4_PASS_1,
    PHASE_4_PASS_2,
)
from .missing_value_phase_1 import FILL_METHOD_COLUMN
from .timeline_correction import (
    CURRENT_STATUS_LABEL_COLUMN,
    INVALID_USABILITY_VALUE,
    KPI_USABILITY_COLUMN,
    ORIGINAL_CONDITION_LABEL_COLUMN,
    VALID_USABILITY_VALUE,
)


PASS_1_SAFE_METHOD = "phase4_pass1_prophet_forward_safe"
PASS_1_LEAKAGE_METHOD = (
    "phase4_pass1_prophet_full_series_leakage"
)
PASS_2_SAFE_METHOD = "phase4_pass2_prophet_forward_safe"
PASS_2_LEAKAGE_METHOD = (
    "phase4_pass2_prophet_full_series_leakage"
)

PHASE_4_SAFE_METHODS = {
    PASS_1_SAFE_METHOD,
    PASS_2_SAFE_METHOD,
}
PHASE_4_LEAKAGE_METHODS = {
    PASS_1_LEAKAGE_METHOD,
    PASS_2_LEAKAGE_METHOD,
}
PHASE_4_METHODS = (
    PHASE_4_SAFE_METHODS
    | PHASE_4_LEAKAGE_METHODS
)

OUTPUT_FILE_SUFFIX = "_phase4_filled.csv"
SUMMARY_FILE_SUFFIX = "_phase4_summary.csv"

TRUSTED_CURRENT_LABELS = {1, 3}
UNRESOLVED_CURRENT_LABELS = {-2, -1, 0}
RESOLVED_FILL_LABELS = {3, 4}

_ALLOWED_ORIGINAL_LABELS = {-2, -1, 0, 1}
_ALLOWED_CURRENT_LABELS = {-2, -1, 0, 1, 2, 3}
_ALLOWED_GAP_PHASES = {1, 2, 3, 4, 5}
_ALLOWED_USABILITY_VALUES = {
    VALID_USABILITY_VALUE,
    INVALID_USABILITY_VALUE,
}

_STATUS_PROCESSED = "processed"
_STATUS_SKIPPED_UNUSABLE = "skipped_unusable_series"
_STATUS_NO_TARGETS = "no_phase4_targets"

_PASS_1 = 1
_PASS_2 = 2

_INTERNAL_TIMESTAMP_COLUMN = "__phase4_timestamp__"
_INTERNAL_ROW_ORDER_COLUMN = "__phase4_row_order__"


@dataclass(frozen=True, slots=True)
class OriginalGapBlock:
    """
    Store one immutable original missing-value gap.

    Attributes:
        gap_number:
            One-based gap number within the site.

        gap_phase:
            Original gap phase from 1 through 5.

        gap_start:
            First timestamp in the original gap.

        gap_end:
            Final timestamp in the original gap.

        positions:
            Timestamp-sorted site-array positions belonging to the gap.
    """

    gap_number: int
    gap_phase: int
    gap_start: pd.Timestamp
    gap_end: pd.Timestamp
    positions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ProphetModelBundle:
    """
    Store one fitted Prophet model and its residual-noise scale.

    Attributes:
        model:
            Fitted Prophet-compatible model.

        training:
            Exact trusted training dataframe used for the model.

        residual_standard_deviation:
            Population standard deviation of finite training residuals.
    """

    model: Any
    training: pd.DataFrame
    residual_standard_deviation: float


@dataclass(slots=True)
class FullSeriesModelCache:
    """
    Store one lazy full-series model cache for a site and pass.

    Attributes:
        attempted:
            Whether eligibility and fitting have already been attempted.

        bundle:
            Fitted bundle, or ``None`` when unavailable.

        ineligible:
            Whether the fixed pass snapshot lacked required history.

        failed:
            Whether an eligible fitting attempt raised an exception.
    """

    attempted: bool = False
    bundle: ProphetModelBundle | None = None
    ineligible: bool = False
    failed: bool = False


@dataclass(frozen=True, slots=True)
class Phase4FillingData:
    """
    Store the in-memory Phase 4 result.

    Attributes:
        dataframe:
            Dataset after both Prophet passes.

        summary:
            One row per site and pass.

        total_series:
            Number of unique site series.

        processed_series:
            Unique usable sites having at least one Phase 4 target.

        skipped_series:
            Unique invalid sites having at least one Phase 4 target.

        pass_1_counts:
            Aggregated Pass 1 target, fill, failure, and model counts.

        pass_2_counts:
            Aggregated Pass 2 target, fill, failure, and model counts.

        total_safe_filled_points:
            Safe status-3 predictions across both passes.

        total_leakage_filled_points:
            Full-series status-4 predictions across both passes.

        total_unresolved_points:
            Eligible target points left unresolved after both passes.
    """

    dataframe: pd.DataFrame
    summary: pd.DataFrame
    total_series: int
    processed_series: int
    skipped_series: int
    pass_1_counts: dict[str, int]
    pass_2_counts: dict[str, int]
    total_safe_filled_points: int
    total_leakage_filled_points: int
    total_unresolved_points: int


@dataclass(frozen=True, slots=True)
class Phase4FillingResult:
    """
    Store saved Phase 4 paths and aggregate statistics.

    Attributes:
        source_file:
            Exact Phase 3 dataset supplied by the preprocessing runner.

        output_file:
            Final Phase 4 filled dataset.

        summary_file:
            Per-site, per-pass Phase 4 summary CSV.

        total_rows:
            Number of rows preserved in the output.

        total_series:
            Number of unique site series.

        processed_series:
            Unique usable sites having at least one target.

        skipped_series:
            Unique invalid sites having at least one target.

        pass_1_counts:
            Aggregated Pass 1 counts.

        pass_2_counts:
            Aggregated Pass 2 counts.

        total_safe_filled_points:
            Safe status-3 Prophet predictions.

        total_leakage_filled_points:
            Leakage-risk status-4 Prophet predictions.

        total_unresolved_points:
            Eligible target points still unresolved after Phase 4.
    """

    source_file: Path
    output_file: Path
    summary_file: Path
    total_rows: int
    total_series: int
    processed_series: int
    skipped_series: int
    pass_1_counts: dict[str, int]
    pass_2_counts: dict[str, int]
    total_safe_filled_points: int
    total_leakage_filled_points: int
    total_unresolved_points: int


def new_pass_counts() -> dict[str, int]:
    """
    Build a zeroed Phase 4 pass-statistics dictionary.

    Returns:
        dict[str, int]:
            Mutable count dictionary used for one site and pass.

    Raises:
        None.
    """

    return {
        "primary_gaps": 0,
        "primary_points": 0,
        "retry_gaps": 0,
        "retry_points": 0,
        "target_points": 0,
        "already_resolved_points": 0,
        "eligible_unresolved_points": 0,
        "safe_filled_points": 0,
        "leakage_filled_points": 0,
        "passed_points": 0,
        "failed_points": 0,
        "skipped_unusable_points": 0,
        "state_mismatch_points": 0,
        "safe_models_fitted": 0,
        "safe_model_failures": 0,
        "safe_model_ineligible_events": 0,
        "full_models_fitted": 0,
        "full_model_failures": 0,
        "full_model_ineligible_events": 0,
    }


def load_prophet_class() -> type[Any]:
    """
    Import and return Prophet only when the enabled stage actually runs.

    Returns:
        type[Any]:
            The ``prophet.Prophet`` class.

    Raises:
        FileProcessingError:
            If the optional Prophet package is unavailable.
    """

    try:
        from prophet import Prophet
    except ImportError as error:
        raise FileProcessingError(
            "Phase 4 is enabled but the optional 'prophet' package is not "
            "installed in the backend environment.",
            error_code="PROPHET_PACKAGE_NOT_INSTALLED",
            details={
                "required_package": "prophet",
            },
        ) from error

    return Prophet


def add_composite_id(
    dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
) -> pd.DataFrame:
    """
    Return a dataframe copy containing the shared composite site identifier.

    Args:
        dataframe:
            Dataset containing configured site identity columns.

        columns:
            Shared dataset column mapping.

    Returns:
        pd.DataFrame:
            Copy containing ``Composite_ID``.

    Raises:
        KeyError:
            If a configured identity column is unavailable.
    """

    result = dataframe.copy()
    identity_columns = list(
        columns.get_site_identity_columns()
    )

    identity_values = (
        result[identity_columns]
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
) -> pd.Series:
    """
    Parse and validate one required integer state column.

    Args:
        dataframe:
            Dataset containing the state column.

        column_name:
            Column to parse.

        allowed_values:
            Complete accepted integer values.

    Returns:
        pd.Series:
            Parsed ``int8`` values.

    Raises:
        InputValidationError:
            If values are missing, non-integral, or unsupported.
    """

    numeric = pd.to_numeric(
        dataframe[column_name],
        errors="coerce",
    )

    invalid_mask = (
        numeric.isna()
        | ~np.isfinite(numeric)
        | numeric.mod(1).ne(0)
        | ~numeric.isin(allowed_values)
    )

    if invalid_mask.any():
        raise InputValidationError(
            f"{column_name} contains invalid pre-Phase-4 values.",
            error_code="INVALID_PHASE4_INTEGER_COLUMN",
            details={
                "column_name": column_name,
                "allowed_values": sorted(allowed_values),
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

    return numeric.astype(np.int8)


def parse_gap_phase_column(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """
    Parse and validate nullable original gap-phase values.

    Args:
        dataframe:
            Dataset containing ``Original_Gap_Phase``.

    Returns:
        pd.Series:
            Nullable ``Int8`` phase values.

    Raises:
        InputValidationError:
            If a non-null phase is non-integral or outside 1 through 5.
    """

    raw = dataframe[ORIGINAL_GAP_PHASE_COLUMN]
    numeric = pd.to_numeric(
        raw,
        errors="coerce",
    )

    invalid_non_null = (
        raw.notna()
        & (
            numeric.isna()
            | ~np.isfinite(numeric)
            | numeric.mod(1).ne(0)
            | ~numeric.isin(_ALLOWED_GAP_PHASES)
        )
    )

    if invalid_non_null.any():
        raise InputValidationError(
            "Original_Gap_Phase contains invalid Phase 4 values.",
            error_code="INVALID_PHASE4_GAP_PHASE",
            details={
                "invalid_rows": int(invalid_non_null.sum()),
                "sample_values": (
                    raw.loc[invalid_non_null]
                    .head(20)
                    .astype("string")
                    .tolist()
                ),
            },
        )

    return numeric.astype("Int8")


def validate_required_columns(
    dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
) -> list[str]:
    """
    Validate the Phase 3 output schema required by Phase 4.

    Args:
        dataframe:
            Dataset returned by Phase 3.

        columns:
            Shared dataset column mapping.

    Returns:
        list[str]:
            Input columns in their existing order.

    Raises:
        InputValidationError:
            If the dataframe is empty, required columns are missing, or
            reserved Phase 4 columns are present.
    """

    if dataframe.empty:
        raise InputValidationError(
            "Phase 4 cannot process an empty dataframe.",
            error_code="EMPTY_PHASE4_DATASET",
        )

    required_columns = [
        *columns.get_site_identity_columns(),
        columns.timestamp_column,
        columns.value_column,
        ORIGINAL_CONDITION_LABEL_COLUMN,
        CURRENT_STATUS_LABEL_COLUMN,
        KPI_USABILITY_COLUMN,
        ORIGINAL_GAP_PHASE_COLUMN,
        FILL_METHOD_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise InputValidationError(
            "The Phase 3 dataset is missing columns required by Phase 4.",
            error_code="MISSING_PHASE4_COLUMNS",
            details={
                "missing_columns": missing_columns,
                "required_columns": required_columns,
                "available_columns": list(dataframe.columns),
            },
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
            "The Phase 3 dataset contains reserved Phase 4 columns.",
            error_code="RESERVED_PHASE4_COLUMN_CONFLICT",
            details={
                "conflicting_columns": conflicts,
            },
        )

    return list(dataframe.columns)


def validate_usability(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """
    Normalize and validate site usability values.

    Args:
        dataframe:
            Working dataset containing ``Composite_ID``.

    Returns:
        pd.Series:
            Normalized usability strings.

    Raises:
        InputValidationError:
            If values are missing, unsupported, or inconsistent within a site.
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
            "kpi_usability contains invalid Phase 4 values.",
            error_code="INVALID_PHASE4_USABILITY",
            details={
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

    per_site_counts = (
        pd.DataFrame(
            {
                COMPOSITE_ID_COLUMN: dataframe[COMPOSITE_ID_COLUMN],
                KPI_USABILITY_COLUMN: usability,
            }
        )
        .groupby(
            COMPOSITE_ID_COLUMN,
            dropna=False,
        )[KPI_USABILITY_COLUMN]
        .nunique(dropna=False)
    )

    inconsistent = per_site_counts.loc[
        per_site_counts != 1
    ]

    if not inconsistent.empty:
        raise InputValidationError(
            "kpi_usability is not constant within one or more sites.",
            error_code="INCONSISTENT_PHASE4_USABILITY",
            details={
                "sample_composite_ids": (
                    inconsistent.index
                    .astype(str)
                    .tolist()[:20]
                ),
            },
        )

    return usability


def validate_fill_methods(
    dataframe: pd.DataFrame,
    current_labels: pd.Series,
) -> pd.Series:
    """
    Validate earlier fill provenance and reject accidental Phase 4 reruns.

    Args:
        dataframe:
            Phase 3 dataset.

        current_labels:
            Parsed pre-Phase-4 current labels.

    Returns:
        pd.Series:
            Normalized fill-method strings.

    Raises:
        InputValidationError:
            If a Phase 4 method or label 4 already exists, or earlier
            provenance is inconsistent.
    """

    methods = (
        dataframe[FILL_METHOD_COLUMN]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    existing_phase4 = methods.isin(PHASE_4_METHODS) | (
        methods.str.startswith("phase4_")
    )

    if existing_phase4.any():
        raise InputValidationError(
            "The dataset already contains Phase 4 fill methods.",
            error_code="PHASE4_ALREADY_APPLIED",
            details={
                "rows": int(existing_phase4.sum()),
                "sample_methods": (
                    methods.loc[existing_phase4]
                    .drop_duplicates()
                    .head(20)
                    .tolist()
                ),
            },
        )

    safe_without_method = (
        current_labels.eq(3)
        & methods.eq("")
    )

    if safe_without_method.any():
        raise InputValidationError(
            "Rows with Current_Status_Label 3 require Fill_Method provenance.",
            error_code="PHASE4_SAFE_FILL_WITHOUT_METHOD",
            details={
                "invalid_rows": int(safe_without_method.sum()),
            },
        )

    method_without_safe_label = (
        methods.ne("")
        & current_labels.ne(3)
    )

    if method_without_safe_label.any():
        raise InputValidationError(
            "Earlier Fill_Method values are present on rows not labelled 3.",
            error_code="PHASE4_METHOD_WITHOUT_SAFE_LABEL",
            details={
                "invalid_rows": int(method_without_safe_label.sum()),
            },
        )

    return methods


def prepare_phase4_dataframe(
    *,
    dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
    config: PreprocessingConfig,
    resolved_frequency: ResolvedFrequencyConfig,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Validate and prepare one Phase 3 dataset for Prophet filling.

    Args:
        dataframe:
            Dataset returned by Phase 3.

        columns:
            Shared dataset column mapping.

        config:
            Complete preprocessing configuration.

        resolved_frequency:
            Frequency-dependent preprocessing settings.

    Returns:
        tuple[pd.DataFrame, list[str]]:
            Prepared working dataframe and original input-column order.

    Raises:
        InputValidationError:
            If Phase 4 is disabled or any schema, label, value, timeline, or
            resolved configuration invariant fails.
    """

    if not config.phase_4.enabled:
        raise InputValidationError(
            "Phase 4 is disabled. The preprocessing runner should skip this "
            "module instead of calling it.",
            error_code="PHASE4_CALLED_WHILE_DISABLED",
        )

    input_columns = validate_required_columns(
        dataframe=dataframe,
        columns=columns,
    )

    working = add_composite_id(
        dataframe=dataframe,
        columns=columns,
    )
    working[_INTERNAL_ROW_ORDER_COLUMN] = np.arange(
        len(working),
        dtype=np.int64,
    )

    parsed_timestamps = pd.to_datetime(
        working[columns.timestamp_column],
        errors="coerce",
    )

    invalid_timestamps = parsed_timestamps.isna()

    if invalid_timestamps.any():
        raise InputValidationError(
            "The Phase 3 dataset contains invalid timestamps.",
            error_code="INVALID_PHASE4_TIMESTAMPS",
            details={
                "invalid_rows": int(invalid_timestamps.sum()),
            },
        )

    if getattr(parsed_timestamps.dt, "tz", None) is not None:
        raise InputValidationError(
            "Prophet Phase 4 requires timezone-naive timestamps.",
            error_code="TIMEZONE_AWARE_PHASE4_TIMESTAMPS",
        )

    working[_INTERNAL_TIMESTAMP_COLUMN] = parsed_timestamps

    original_labels = parse_integer_column(
        dataframe=working,
        column_name=ORIGINAL_CONDITION_LABEL_COLUMN,
        allowed_values=_ALLOWED_ORIGINAL_LABELS,
    )
    current_labels = parse_integer_column(
        dataframe=working,
        column_name=CURRENT_STATUS_LABEL_COLUMN,
        allowed_values=_ALLOWED_CURRENT_LABELS,
    )
    gap_phases = parse_gap_phase_column(
        dataframe=working,
    )

    working[ORIGINAL_CONDITION_LABEL_COLUMN] = original_labels
    working[CURRENT_STATUS_LABEL_COLUMN] = current_labels
    working[ORIGINAL_GAP_PHASE_COLUMN] = gap_phases
    working[KPI_USABILITY_COLUMN] = validate_usability(
        working
    )
    working[FILL_METHOD_COLUMN] = validate_fill_methods(
        dataframe=working,
        current_labels=current_labels,
    )

    missing_origin = original_labels.isin(
        MISSING_ORIGINAL_LABELS
    )
    valid_origin = original_labels.eq(1)

    invalid_transition = (
        (
            missing_origin
            & ~current_labels.isin(
                {-2, -1, 0, 3}
            )
        )
        | (
            valid_origin
            & ~current_labels.isin({1, 2})
        )
    )

    if invalid_transition.any():
        raise InputValidationError(
            "Original and current labels contain invalid pre-Phase-4 states.",
            error_code="INVALID_PHASE4_LABEL_TRANSITION",
            details={
                "invalid_rows": int(invalid_transition.sum()),
            },
        )

    missing_without_phase = (
        missing_origin
        & gap_phases.isna()
    )
    nonmissing_with_phase = (
        ~missing_origin
        & gap_phases.notna()
    )

    if missing_without_phase.any() or nonmissing_with_phase.any():
        raise InputValidationError(
            "Original_Gap_Phase does not match immutable original labels.",
            error_code="INVALID_PHASE4_GAP_PHASE_STATE",
            details={
                "missing_without_phase": int(
                    missing_without_phase.sum()
                ),
                "nonmissing_with_phase": int(
                    nonmissing_with_phase.sum()
                ),
            },
        )

    numeric_values = pd.to_numeric(
        working[columns.value_column],
        errors="coerce",
    )

    unresolved = current_labels.isin(
        UNRESOLVED_CURRENT_LABELS
    )
    present_state = current_labels.isin(
        TRUSTED_CURRENT_LABELS | {2}
    )

    unresolved_with_value = (
        unresolved
        & numeric_values.notna()
    )
    present_state_without_value = (
        present_state
        & (
            numeric_values.isna()
            | ~np.isfinite(numeric_values)
        )
    )

    if unresolved_with_value.any() or present_state_without_value.any():
        raise InputValidationError(
            "KPI values do not match current pre-Phase-4 labels.",
            error_code="INVALID_PHASE4_VALUE_STATE",
            details={
                "unresolved_with_value": int(
                    unresolved_with_value.sum()
                ),
                "present_state_without_value": int(
                    present_state_without_value.sum()
                ),
            },
        )

    lower_bound = config.valid_values.lower_bound
    upper_bound = config.valid_values.upper_bound

    present_outside_range = (
        numeric_values.notna()
        & (
            ~np.isfinite(numeric_values)
            | numeric_values.lt(lower_bound)
            | numeric_values.gt(upper_bound)
        )
    )

    if present_outside_range.any():
        raise InputValidationError(
            "The Phase 3 dataset contains present KPI values outside the "
            "configured valid range.",
            error_code="PHASE4_VALUE_OUTSIDE_VALID_RANGE",
            details={
                "invalid_rows": int(present_outside_range.sum()),
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            },
        )

    duplicate_mask = working.duplicated(
        subset=[
            COMPOSITE_ID_COLUMN,
            _INTERNAL_TIMESTAMP_COLUMN,
        ],
        keep=False,
    )

    if duplicate_mask.any():
        raise InputValidationError(
            "Duplicate site-timestamp rows remain before Phase 4.",
            error_code="DUPLICATE_TIMESTAMPS_BEFORE_PHASE4",
            details={
                "duplicate_rows": int(duplicate_mask.sum()),
            },
        )

    expected_step = pd.Timedelta(
        minutes=resolved_frequency.bucket_minutes
    )

    if expected_step <= pd.Timedelta(0):
        raise InputValidationError(
            "The resolved Phase 4 bucket frequency must be positive.",
            error_code="NON_POSITIVE_PHASE4_FREQUENCY",
        )

    spacing_errors: list[dict[str, str]] = []

    for composite_id, site_dataframe in working.groupby(
        COMPOSITE_ID_COLUMN,
        sort=False,
        dropna=False,
    ):
        timestamps = (
            site_dataframe[_INTERNAL_TIMESTAMP_COLUMN]
            .sort_values(kind="stable")
        )
        differences = timestamps.diff()
        bad_spacing = (
            differences.notna()
            & differences.ne(expected_step)
        )

        if bad_spacing.any():
            bad_index = bad_spacing.loc[
                bad_spacing
            ].index[0]
            ordered_indices = list(timestamps.index)
            position = ordered_indices.index(bad_index)
            previous_index = ordered_indices[position - 1]

            spacing_errors.append(
                {
                    COMPOSITE_ID_COLUMN: str(composite_id),
                    "previous_timestamp": str(
                        working.at[
                            previous_index,
                            _INTERNAL_TIMESTAMP_COLUMN,
                        ]
                    ),
                    "current_timestamp": str(
                        working.at[
                            bad_index,
                            _INTERNAL_TIMESTAMP_COLUMN,
                        ]
                    ),
                    "expected_difference": str(expected_step),
                }
            )

            if len(spacing_errors) >= 20:
                break

    if spacing_errors:
        raise InputValidationError(
            "One or more site timelines are incomplete before Phase 4.",
            error_code="INCOMPLETE_TIMELINE_BEFORE_PHASE4",
            details={
                "sample": spacing_errors,
            },
        )

    if resolved_frequency.prophet_minimum_history_points <= 0:
        raise InputValidationError(
            "Resolved Prophet minimum history points must be positive.",
            error_code="NON_POSITIVE_PROPHET_HISTORY_POINTS",
            details={
                "prophet_minimum_history_points": (
                    resolved_frequency
                    .prophet_minimum_history_points
                ),
            },
        )

    if config.phase_4.minimum_history_days <= 0:
        raise InputValidationError(
            "Prophet minimum history days must be positive.",
            error_code="NON_POSITIVE_PROPHET_HISTORY_DAYS",
        )

    return working, input_columns


def build_original_gap_blocks(
    ordered: pd.DataFrame,
) -> list[OriginalGapBlock]:
    """
    Reconstruct immutable original gap blocks from frozen row classifications.

    The function does not infer gap sizes from current missing values. It uses
    contiguous rows carrying one frozen ``Original_Gap_Phase``.

    Args:
        ordered:
            One site dataframe sorted by timestamp.

    Returns:
        list[OriginalGapBlock]:
            Original gaps ordered by timestamp.

    Raises:
        InputValidationError:
            If one contiguous original gap contains multiple phase values.
    """

    gap_present = ordered[
        ORIGINAL_GAP_PHASE_COLUMN
    ].notna()

    if not gap_present.any():
        return []

    new_gap = (
        gap_present
        & ~gap_present.shift(
            1,
            fill_value=False,
        )
    )
    gap_ids = new_gap.cumsum()

    gap_rows = ordered.loc[
        gap_present
    ].copy()
    gap_rows["__phase4_gap_id__"] = gap_ids.loc[
        gap_present
    ]

    index_to_position = {
        index: position
        for position, index in enumerate(ordered.index)
    }

    blocks: list[OriginalGapBlock] = []

    for gap_number, (_, gap_dataframe) in enumerate(
        gap_rows.groupby(
            "__phase4_gap_id__",
            sort=True,
            dropna=False,
        ),
        start=1,
    ):
        unique_phases = (
            gap_dataframe[ORIGINAL_GAP_PHASE_COLUMN]
            .dropna()
            .astype(int)
            .unique()
        )

        if len(unique_phases) != 1:
            raise InputValidationError(
                "One original gap contains multiple frozen phase values.",
                error_code="MIXED_PHASE_VALUES_WITHIN_ORIGINAL_GAP",
                details={
                    "gap_number": gap_number,
                    "phase_values": unique_phases.tolist(),
                },
            )

        positions = tuple(
            index_to_position[index]
            for index in gap_dataframe.index
        )

        blocks.append(
            OriginalGapBlock(
                gap_number=gap_number,
                gap_phase=int(unique_phases[0]),
                gap_start=pd.Timestamp(
                    gap_dataframe[
                        _INTERNAL_TIMESTAMP_COLUMN
                    ].iloc[0]
                ),
                gap_end=pd.Timestamp(
                    gap_dataframe[
                        _INTERNAL_TIMESTAMP_COLUMN
                    ].iloc[-1]
                ),
                positions=positions,
            )
        )

    return blocks


def stable_seed(
    *parts: Any,
) -> int:
    """
    Derive one deterministic NumPy seed from configuration and gap identity.

    Args:
        *parts:
            Values identifying the dataset, site, pass, gap, and model mode.

    Returns:
        int:
            Stable unsigned 32-bit seed.

    Raises:
        None.
    """

    payload = "||".join(
        str(part)
        for part in parts
    ).encode("utf-8")

    digest = hashlib.sha256(payload).digest()

    return int.from_bytes(
        digest[:4],
        byteorder="big",
        signed=False,
    )


def build_training_dataframe(
    *,
    timestamps: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    lower_bound: float,
    upper_bound: float,
    before_timestamp: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Build one trusted Prophet training dataframe from a fixed pass snapshot.

    Args:
        timestamps:
            Site timestamps from the fixed pass snapshot.

        values:
            Snapshot KPI values.

        labels:
            Snapshot current labels.

        lower_bound:
            Inclusive valid KPI lower bound.

        upper_bound:
            Inclusive valid KPI upper bound.

        before_timestamp:
            Optional strict historical cutoff. When provided, only timestamps
            earlier than this value are included.

    Returns:
        pd.DataFrame:
            Sorted unique Prophet dataframe with ``ds`` and ``y``.

    Raises:
        None.
    """

    trusted_mask = np.isin(
        labels,
        list(TRUSTED_CURRENT_LABELS),
    )
    finite_mask = np.isfinite(values)
    range_mask = (
        (values >= lower_bound)
        & (values <= upper_bound)
    )

    mask = (
        trusted_mask
        & finite_mask
        & range_mask
    )

    if before_timestamp is not None:
        mask = (
            mask
            & (
                timestamps
                < np.datetime64(before_timestamp)
            )
        )

    training = pd.DataFrame(
        {
            "ds": pd.to_datetime(
                timestamps[mask]
            ),
            "y": values[mask].astype(float),
        }
    )

    return (
        training
        .drop_duplicates(
            subset=["ds"],
            keep="first",
        )
        .sort_values(
            "ds",
            kind="stable",
        )
        .reset_index(drop=True)
    )


def training_is_eligible(
    *,
    training: pd.DataFrame,
    minimum_history_points: int,
    minimum_history_days: int,
    bucket_minutes: int,
) -> bool:
    """
    Determine whether one Prophet training frame satisfies both thresholds.

    The effective history duration includes the final observation's bucket
    interval. For example, 96 consecutive 15-minute observations represent
    one complete day even though the difference between their first and last
    timestamps is only 23 hours and 45 minutes.

    Args:
        training:
            Trusted Prophet dataframe.

        minimum_history_points:
            Minimum trusted observations.

        minimum_history_days:
            Minimum inclusive observation duration.

        bucket_minutes:
            Positive resolved duration represented by each observation.

    Returns:
        bool:
            True when both point count and inclusive-duration conditions are
            satisfied.

    Raises:
        None.
    """

    if training.empty:
        return False

    if len(training) < minimum_history_points:
        return False

    if bucket_minutes <= 0:
        return False

    effective_history_span = (
        training["ds"].iloc[-1]
        - training["ds"].iloc[0]
        + pd.Timedelta(
            minutes=bucket_minutes
        )
    )

    return bool(
        effective_history_span
        >= pd.Timedelta(
            days=minimum_history_days
        )
    )


def fit_prophet_model(
    *,
    training: pd.DataFrame,
    config: PreprocessingConfig,
    prophet_class: type[Any],
) -> ProphetModelBundle:
    """
    Fit one Prophet model using every configured model parameter.

    Args:
        training:
            Eligible trusted training dataframe.

        config:
            Complete preprocessing configuration.

        prophet_class:
            Imported Prophet class or test-compatible replacement.

    Returns:
        ProphetModelBundle:
            Fitted model, training data, and residual standard deviation.

    Raises:
        Exception:
            Any exception raised by Prophet construction, fitting, or
            residual prediction is propagated to pass-level handling.
    """

    phase4_config = config.phase_4

    model = prophet_class(
        changepoint_prior_scale=float(
            phase4_config.changepoint_prior_scale
        ),
        seasonality_prior_scale=float(
            phase4_config.seasonality_prior_scale
        ),
        seasonality_mode=(
            phase4_config.seasonality_mode
        ),
        daily_seasonality=(
            phase4_config.daily_seasonality
        ),
        weekly_seasonality=(
            phase4_config.weekly_seasonality
        ),
        yearly_seasonality=(
            phase4_config.yearly_seasonality
        ),
        interval_width=float(
            phase4_config.interval_width
        ),
    )

    model.fit(training)

    residual_standard_deviation = (
        compute_training_residual_standard_deviation(
            model=model,
            training=training,
        )
    )

    return ProphetModelBundle(
        model=model,
        training=training,
        residual_standard_deviation=(
            residual_standard_deviation
        ),
    )


def compute_training_residual_standard_deviation(
    *,
    model: Any,
    training: pd.DataFrame,
) -> float:
    """
    Calculate population standard deviation of finite training residuals.

    Args:
        model:
            Fitted Prophet-compatible model.

        training:
            Exact training dataframe.

    Returns:
        float:
            Non-negative finite residual standard deviation. Returns zero when
            residuals cannot be formed.

    Raises:
        Exception:
            Prediction exceptions are propagated to the fitting caller.
    """

    fitted = model.predict(
        training[["ds"]]
    )[["ds", "yhat"]]

    aligned = training.merge(
        fitted,
        on="ds",
        how="inner",
        validate="one_to_one",
    )

    if aligned.empty:
        return 0.0

    residuals = (
        aligned["y"].to_numpy(dtype=float)
        - aligned["yhat"].to_numpy(dtype=float)
    )
    residuals = residuals[
        np.isfinite(residuals)
    ]

    if residuals.size == 0:
        return 0.0

    residual_standard_deviation = float(
        np.std(
            residuals,
            ddof=0,
        )
    )

    if (
        not np.isfinite(
            residual_standard_deviation
        )
        or residual_standard_deviation < 0
    ):
        return 0.0

    return residual_standard_deviation


def predict_with_residual_noise(
    *,
    bundle: ProphetModelBundle,
    timestamps: list[pd.Timestamp],
    config: PreprocessingConfig,
    seed_parts: tuple[Any, ...],
) -> tuple[dict[pd.Timestamp, float], set[pd.Timestamp]]:
    """
    Predict target timestamps and add deterministic residual-based noise.

    Args:
        bundle:
            Fitted model and residual statistics.

        timestamps:
            Target timestamps.

        config:
            Valid-value and Phase 4 noise configuration.

        seed_parts:
            Stable dataset, site, pass, gap, and mode identity.

    Returns:
        tuple[dict[pd.Timestamp, float], set[pd.Timestamp]]:
            Successful timestamp-value mapping and failed timestamps.

    Raises:
        Exception:
            Model prediction exceptions are propagated to pass-level handling.
    """

    if not timestamps:
        return {}, set()

    future = pd.DataFrame(
        {
            "ds": pd.to_datetime(
                timestamps
            ),
        }
    )

    forecast = bundle.model.predict(
        future
    )[["ds", "yhat"]]

    forecast = forecast.drop_duplicates(
        subset=["ds"],
        keep="first",
    )

    prediction_map = {
        pd.Timestamp(row.ds): float(row.yhat)
        for row in forecast.itertuples(
            index=False
        )
        if np.isfinite(row.yhat)
    }

    rng = np.random.default_rng(
        stable_seed(
            config.phase_4.random_seed,
            *seed_parts,
        )
    )

    lower_bound = config.valid_values.lower_bound
    upper_bound = config.valid_values.upper_bound
    noise_standard_deviation = (
        bundle.residual_standard_deviation
        * config.phase_4.noise_scale
    )

    predictions: dict[pd.Timestamp, float] = {}
    failures: set[pd.Timestamp] = set()

    for timestamp in timestamps:
        normalized_timestamp = pd.Timestamp(
            timestamp
        )
        prediction = prediction_map.get(
            normalized_timestamp
        )

        if prediction is None or not np.isfinite(
            prediction
        ):
            failures.add(
                normalized_timestamp
            )
            continue

        if (
            noise_standard_deviation > 0
            and np.isfinite(
                noise_standard_deviation
            )
        ):
            noise = float(
                rng.normal(
                    loc=0.0,
                    scale=noise_standard_deviation,
                )
            )
        else:
            noise = 0.0

        generated = float(
            prediction + noise
        )

        if not np.isfinite(generated):
            failures.add(
                normalized_timestamp
            )
            continue

        if config.valid_values.clip_generated_values:
            generated = float(
                np.clip(
                    generated,
                    lower_bound,
                    upper_bound,
                )
            )

        if not (
            lower_bound
            <= generated
            <= upper_bound
        ):
            failures.add(
                normalized_timestamp
            )
            continue

        predictions[
            normalized_timestamp
        ] = generated

    return predictions, failures


def classify_gap_states(
    *,
    gap: OriginalGapBlock,
    values: np.ndarray,
    labels: np.ndarray,
    original_labels: np.ndarray,
    lower_bound: float,
    upper_bound: float,
) -> tuple[list[int], int, int]:
    """
    Classify current point states inside one immutable original gap.

    Args:
        gap:
            Original gap block.

        values:
            Live working values.

        labels:
            Live current labels.

        original_labels:
            Immutable original-condition labels.

        lower_bound:
            Inclusive valid lower bound.

        upper_bound:
            Inclusive valid upper bound.

    Returns:
        tuple[list[int], int, int]:
            Eligible unresolved positions, already-resolved point count, and
            state-mismatch count.

    Raises:
        None.
    """

    unresolved_positions: list[int] = []
    already_resolved = 0
    state_mismatch = 0

    for position in gap.positions:
        original_label = int(
            original_labels[position]
        )
        current_label = int(
            labels[position]
        )
        value = values[position]

        if (
            original_label
            not in MISSING_ORIGINAL_LABELS
        ):
            state_mismatch += 1
            continue

        if (
            current_label
            in RESOLVED_FILL_LABELS
            and np.isfinite(value)
            and lower_bound
            <= value
            <= upper_bound
        ):
            already_resolved += 1
            continue

        if (
            current_label
            in UNRESOLVED_CURRENT_LABELS
            and not np.isfinite(value)
        ):
            unresolved_positions.append(
                position
            )
            continue

        state_mismatch += 1

    return (
        unresolved_positions,
        already_resolved,
        state_mismatch,
    )


def select_pass_gaps(
    *,
    gaps: list[OriginalGapBlock],
    pass_number: int,
    labels: np.ndarray,
    values: np.ndarray,
) -> list[tuple[OriginalGapBlock, bool]]:
    """
    Select primary and retry gaps for one ordered Phase 4 pass.

    Args:
        gaps:
            All original gaps for one site.

        pass_number:
            Pass 1 or Pass 2.

        labels:
            Current live labels at pass start.

        values:
            Current live values at pass start.

    Returns:
        list[tuple[OriginalGapBlock, bool]]:
            Selected gaps with a Boolean indicating retry status.

    Raises:
        ValueError:
            If ``pass_number`` is unsupported.
    """

    selected: list[
        tuple[OriginalGapBlock, bool]
    ] = []

    for gap in gaps:
        if pass_number == _PASS_1:
            if gap.gap_phase == PHASE_4_PASS_1:
                selected.append(
                    (gap, False)
                )
                continue

            if gap.gap_phase in {
                PHASE_1,
                PHASE_2,
                PHASE_3,
            }:
                has_unresolved = any(
                    (
                        int(labels[position])
                        in UNRESOLVED_CURRENT_LABELS
                        and not np.isfinite(
                            values[position]
                        )
                    )
                    for position in gap.positions
                )

                if has_unresolved:
                    selected.append(
                        (gap, True)
                    )

        elif pass_number == _PASS_2:
            if gap.gap_phase == PHASE_4_PASS_2:
                selected.append(
                    (gap, False)
                )

        else:
            raise ValueError(
                f"Unsupported Phase 4 pass: {pass_number}"
            )

    return selected


def get_or_fit_full_series_bundle(
    *,
    cache: FullSeriesModelCache,
    snapshot_timestamps: np.ndarray,
    snapshot_values: np.ndarray,
    snapshot_labels: np.ndarray,
    config: PreprocessingConfig,
    resolved_frequency: ResolvedFrequencyConfig,
    prophet_class: type[Any],
    counts: dict[str, int],
) -> ProphetModelBundle | None:
    """
    Lazily fit and cache one full-series Prophet model for a site and pass.

    Args:
        cache:
            Mutable pass-local full-series cache.

        snapshot_timestamps:
            Fixed pass snapshot timestamps.

        snapshot_values:
            Fixed pass snapshot values.

        snapshot_labels:
            Fixed pass snapshot labels.

        config:
            Complete preprocessing configuration.

        resolved_frequency:
            Resolved Prophet minimum history points.

        prophet_class:
            Imported Prophet class or test replacement.

        counts:
            Mutable pass counts.

    Returns:
        ProphetModelBundle | None:
            Cached fitted model, or ``None`` when unavailable.

    Raises:
        None.
    """

    if cache.attempted:
        return cache.bundle

    cache.attempted = True

    training = build_training_dataframe(
        timestamps=snapshot_timestamps,
        values=snapshot_values,
        labels=snapshot_labels,
        lower_bound=(
            config.valid_values.lower_bound
        ),
        upper_bound=(
            config.valid_values.upper_bound
        ),
    )

    if not training_is_eligible(
        training=training,
        minimum_history_points=(
            resolved_frequency
            .prophet_minimum_history_points
        ),
        minimum_history_days=(
            config.phase_4.minimum_history_days
        ),
        bucket_minutes=(
            resolved_frequency.bucket_minutes
        ),
    ):
        cache.ineligible = True
        counts[
            "full_model_ineligible_events"
        ] += 1
        return None

    try:
        cache.bundle = fit_prophet_model(
            training=training,
            config=config,
            prophet_class=prophet_class,
        )
        counts["full_models_fitted"] += 1
    except Exception:
        cache.failed = True
        counts["full_model_failures"] += 1
        cache.bundle = None

    return cache.bundle


def apply_predictions(
    *,
    predictions: dict[pd.Timestamp, float],
    timestamp_to_position: dict[pd.Timestamp, int],
    values: np.ndarray,
    labels: np.ndarray,
    fill_methods: np.ndarray,
    new_label: int,
    method: str,
) -> int:
    """
    Apply successful model predictions to live site arrays.

    Args:
        predictions:
            Timestamp-value mapping.

        timestamp_to_position:
            Site timestamp-position lookup.

        values:
            Mutable live values.

        labels:
            Mutable live labels.

        fill_methods:
            Mutable live method strings.

        new_label:
            Status label 3 for safe or 4 for leakage-risk predictions.

        method:
            Exact Phase 4 method name.

    Returns:
        int:
            Number of predictions applied.

    Raises:
        None.
    """

    applied = 0

    for timestamp, prediction in (
        predictions.items()
    ):
        position = timestamp_to_position.get(
            pd.Timestamp(timestamp)
        )

        if position is None:
            continue

        if (
            int(labels[position])
            not in UNRESOLVED_CURRENT_LABELS
            or np.isfinite(values[position])
        ):
            continue

        values[position] = float(prediction)
        labels[position] = new_label
        fill_methods[position] = method
        applied += 1

    return applied


def process_one_pass(
    *,
    pass_number: int,
    selected_gaps: list[
        tuple[OriginalGapBlock, bool]
    ],
    composite_id: str,
    dataset_identifier: str,
    usability: str,
    timestamps: np.ndarray,
    timestamp_to_position: dict[pd.Timestamp, int],
    values: np.ndarray,
    labels: np.ndarray,
    original_labels: np.ndarray,
    fill_methods: np.ndarray,
    config: PreprocessingConfig,
    resolved_frequency: ResolvedFrequencyConfig,
    prophet_class: type[Any],
) -> dict[str, int]:
    """
    Process one fixed-snapshot Prophet pass for one site.

    Args:
        pass_number:
            Ordered pass number.

        selected_gaps:
            Primary/retry gaps selected for this pass.

        composite_id:
            Site identifier used in deterministic seeds.

        dataset_identifier:
            Source dataset identifier used in deterministic seeds.

        usability:
            Site usability.

        timestamps:
            Timestamp-sorted site timestamps.

        timestamp_to_position:
            Timestamp-position lookup.

        values:
            Mutable live working values.

        labels:
            Mutable live current labels.

        original_labels:
            Immutable original labels.

        fill_methods:
            Mutable live fill-method values.

        config:
            Complete preprocessing configuration.

        resolved_frequency:
            Resolved Prophet history threshold.

        prophet_class:
            Imported Prophet class or test replacement.

    Returns:
        dict[str, int]:
            Pass statistics for this site.

    Raises:
        None.
    """

    counts = new_pass_counts()

    if not selected_gaps:
        return counts

    snapshot_values = values.copy()
    snapshot_labels = labels.copy()
    snapshot_timestamps = timestamps.copy()

    full_series_cache = FullSeriesModelCache()

    if pass_number == _PASS_1:
        safe_method = PASS_1_SAFE_METHOD
        leakage_method = PASS_1_LEAKAGE_METHOD
    else:
        safe_method = PASS_2_SAFE_METHOD
        leakage_method = PASS_2_LEAKAGE_METHOD

    for gap, is_retry in selected_gaps:
        gap_point_count = len(gap.positions)

        if is_retry:
            counts["retry_gaps"] += 1
        else:
            counts["primary_gaps"] += 1
            counts["primary_points"] += gap_point_count

        counts["target_points"] += gap_point_count

        (
            unresolved_positions,
            already_resolved,
            state_mismatch,
        ) = classify_gap_states(
            gap=gap,
            values=values,
            labels=labels,
            original_labels=original_labels,
            lower_bound=(
                config.valid_values.lower_bound
            ),
            upper_bound=(
                config.valid_values.upper_bound
            ),
        )

        if is_retry:
            counts["retry_points"] += len(
                unresolved_positions
            )

        counts[
            "already_resolved_points"
        ] += already_resolved
        counts[
            "state_mismatch_points"
        ] += state_mismatch

        if not unresolved_positions:
            continue

        if usability == INVALID_USABILITY_VALUE:
            counts[
                "skipped_unusable_points"
            ] += len(unresolved_positions)
            continue

        counts[
            "eligible_unresolved_points"
        ] += len(unresolved_positions)

        target_timestamps = [
            pd.Timestamp(
                timestamps[position]
            )
            for position in unresolved_positions
        ]

        safe_training = build_training_dataframe(
            timestamps=snapshot_timestamps,
            values=snapshot_values,
            labels=snapshot_labels,
            lower_bound=(
                config.valid_values.lower_bound
            ),
            upper_bound=(
                config.valid_values.upper_bound
            ),
            before_timestamp=gap.gap_start,
        )

        safe_predictions: dict[
            pd.Timestamp,
            float,
        ] = {}

        if training_is_eligible(
            training=safe_training,
            minimum_history_points=(
                resolved_frequency
                .prophet_minimum_history_points
            ),
            minimum_history_days=(
                config.phase_4.minimum_history_days
            ),
            bucket_minutes=(
                resolved_frequency.bucket_minutes
            ),
        ):
            try:
                safe_bundle = fit_prophet_model(
                    training=safe_training,
                    config=config,
                    prophet_class=prophet_class,
                )
                counts[
                    "safe_models_fitted"
                ] += 1

                (
                    safe_predictions,
                    _,
                ) = predict_with_residual_noise(
                    bundle=safe_bundle,
                    timestamps=target_timestamps,
                    config=config,
                    seed_parts=(
                        dataset_identifier,
                        composite_id,
                        pass_number,
                        gap.gap_number,
                        gap.gap_start,
                        gap.gap_end,
                        "safe",
                    ),
                )
            except Exception:
                counts[
                    "safe_model_failures"
                ] += 1
                safe_predictions = {}
        else:
            counts[
                "safe_model_ineligible_events"
            ] += 1

        safe_applied = apply_predictions(
            predictions=safe_predictions,
            timestamp_to_position=timestamp_to_position,
            values=values,
            labels=labels,
            fill_methods=fill_methods,
            new_label=3,
            method=safe_method,
        )

        counts[
            "safe_filled_points"
        ] += safe_applied

        remaining_positions = [
            position
            for position in unresolved_positions
            if (
                int(labels[position])
                in UNRESOLVED_CURRENT_LABELS
                and not np.isfinite(
                    values[position]
                )
            )
        ]

        if (
            remaining_positions
            and config.phase_4.allow_full_series_fallback
        ):
            full_bundle = (
                get_or_fit_full_series_bundle(
                    cache=full_series_cache,
                    snapshot_timestamps=(
                        snapshot_timestamps
                    ),
                    snapshot_values=(
                        snapshot_values
                    ),
                    snapshot_labels=(
                        snapshot_labels
                    ),
                    config=config,
                    resolved_frequency=(
                        resolved_frequency
                    ),
                    prophet_class=prophet_class,
                    counts=counts,
                )
            )

            if full_bundle is not None:
                remaining_timestamps = [
                    pd.Timestamp(
                        timestamps[position]
                    )
                    for position in (
                        remaining_positions
                    )
                ]

                try:
                    (
                        leakage_predictions,
                        _,
                    ) = predict_with_residual_noise(
                        bundle=full_bundle,
                        timestamps=remaining_timestamps,
                        config=config,
                        seed_parts=(
                            dataset_identifier,
                            composite_id,
                            pass_number,
                            gap.gap_number,
                            gap.gap_start,
                            gap.gap_end,
                            "full_series",
                        ),
                    )
                except Exception:
                    counts[
                        "full_model_failures"
                    ] += 1
                    leakage_predictions = {}

                leakage_applied = (
                    apply_predictions(
                        predictions=(
                            leakage_predictions
                        ),
                        timestamp_to_position=(
                            timestamp_to_position
                        ),
                        values=values,
                        labels=labels,
                        fill_methods=fill_methods,
                        new_label=4,
                        method=leakage_method,
                    )
                )

                counts[
                    "leakage_filled_points"
                ] += leakage_applied

        remaining_after_models = sum(
            1
            for position in unresolved_positions
            if (
                int(labels[position])
                in UNRESOLVED_CURRENT_LABELS
                and not np.isfinite(
                    values[position]
                )
            )
        )

        counts["failed_points"] += (
            remaining_after_models
        )

    counts["passed_points"] = (
        counts["safe_filled_points"]
        + counts["leakage_filled_points"]
    )

    return counts


def process_one_site(
    *,
    site_dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
    config: PreprocessingConfig,
    resolved_frequency: ResolvedFrequencyConfig,
    prophet_class: type[Any],
    dataset_identifier: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Run both ordered Phase 4 passes for one site series.

    Args:
        site_dataframe:
            Prepared rows for one site.

        columns:
            Shared dataset column mapping.

        config:
            Complete preprocessing configuration.

        resolved_frequency:
            Resolved Prophet history threshold.

        prophet_class:
            Imported Prophet class or test replacement.

        dataset_identifier:
            Stable source identifier used in random seeds.

    Returns:
        tuple[pd.DataFrame, list[dict[str, Any]]]:
            Updated timestamp-sorted site dataframe and two pass-summary rows.

    Raises:
        None.
    """

    ordered = site_dataframe.sort_values(
        _INTERNAL_TIMESTAMP_COLUMN,
        kind="stable",
    ).copy()

    timestamps = (
        ordered[_INTERNAL_TIMESTAMP_COLUMN]
        .to_numpy(
            dtype="datetime64[ns]"
        )
    )
    timestamp_to_position = {
        pd.Timestamp(timestamp): position
        for position, timestamp in enumerate(
            timestamps
        )
    }

    values = pd.to_numeric(
        ordered[columns.value_column],
        errors="coerce",
    ).to_numpy(dtype=float,
               copy=True
               )
    labels = pd.to_numeric(
        ordered[CURRENT_STATUS_LABEL_COLUMN],
        errors="raise",
    ).to_numpy(dtype=np.int8,
               copy=True
               )
    original_labels = pd.to_numeric(
        ordered[ORIGINAL_CONDITION_LABEL_COLUMN],
        errors="raise",
    ).to_numpy(dtype=np.int8)
    fill_methods = (
        ordered[FILL_METHOD_COLUMN]
        .astype("string")
        .fillna("")
        .to_numpy(dtype=object)
    )

    composite_id = str(
        ordered[COMPOSITE_ID_COLUMN].iloc[0]
    )
    usability = str(
        ordered[KPI_USABILITY_COLUMN].iloc[0]
    )

    gaps = build_original_gap_blocks(
        ordered
    )

    summary_rows: list[
        dict[str, Any]
    ] = []

    site_had_target = False # unused variable in this script

    for pass_number in (
        _PASS_1,
        _PASS_2,
    ):
        selected_gaps = select_pass_gaps(
            gaps=gaps,
            pass_number=pass_number,
            labels=labels,
            values=values,
        )

        if selected_gaps:
            site_had_target = True # unsused in this script

        counts = process_one_pass(
            pass_number=pass_number,
            selected_gaps=selected_gaps,
            composite_id=composite_id,
            dataset_identifier=(
                dataset_identifier
            ),
            usability=usability,
            timestamps=timestamps,
            timestamp_to_position=(
                timestamp_to_position
            ),
            values=values,
            labels=labels,
            original_labels=original_labels,
            fill_methods=fill_methods,
            config=config,
            resolved_frequency=(
                resolved_frequency
            ),
            prophet_class=prophet_class,
        )

        if not selected_gaps:
            status = _STATUS_NO_TARGETS
        elif usability == (
            INVALID_USABILITY_VALUE
        ):
            status = _STATUS_SKIPPED_UNUSABLE
        else:
            status = _STATUS_PROCESSED

        summary_rows.append(
            {
                COMPOSITE_ID_COLUMN: (
                    composite_id
                ),
                KPI_USABILITY_COLUMN: usability,
                "pass_number": pass_number,
                "status": status,
                **counts,
            }
        )

    ordered[columns.value_column] = values
    ordered[CURRENT_STATUS_LABEL_COLUMN] = labels
    ordered[FILL_METHOD_COLUMN] = (
        fill_methods
    )

    return ordered, summary_rows


def validate_phase4_output(
    *,
    input_dataframe: pd.DataFrame,
    output_dataframe: pd.DataFrame,
    input_columns: list[str],
    columns: DatasetColumnMapping,
    config: PreprocessingConfig,
) -> None:
    """
    Validate final Phase 4 output invariants.

    Args:
        input_dataframe:
            Phase 3 input dataset.

        output_dataframe:
            Dataset after both Phase 4 passes.

        input_columns:
            Original input-column order.

        columns:
            Shared dataset column mapping.

        config:
            Complete preprocessing configuration.

    Returns:
        None.

    Raises:
        InputValidationError:
            If schema, row count, protected fields, state changes, generated
            values, or fill-method provenance are invalid.
    """

    if list(output_dataframe.columns) != input_columns:
        raise InputValidationError(
            "Phase 4 changed the dataset schema or column order.",
            error_code="INVALID_PHASE4_OUTPUT_COLUMNS",
            details={
                "expected_columns": input_columns,
                "actual_columns": list(
                    output_dataframe.columns
                ),
            },
        )

    if len(output_dataframe) != len(
        input_dataframe
    ):
        raise InputValidationError(
            "Phase 4 changed the dataset row count.",
            error_code="PHASE4_ROW_COUNT_CHANGED",
            details={
                "input_rows": len(input_dataframe),
                "output_rows": len(output_dataframe),
            },
        )

    mutable_columns = {
        columns.value_column,
        CURRENT_STATUS_LABEL_COLUMN,
        FILL_METHOD_COLUMN,
    }

    changed_protected = [
        column
        for column in input_columns
        if (
            column not in mutable_columns
            and not output_dataframe[column].equals(
                input_dataframe[column]
            )
        )
    ]

    if changed_protected:
        raise InputValidationError(
            "Phase 4 modified protected input columns.",
            error_code="PHASE4_MODIFIED_PROTECTED_COLUMNS",
            details={
                "changed_columns": changed_protected,
            },
        )

    input_labels = pd.to_numeric(
        input_dataframe[CURRENT_STATUS_LABEL_COLUMN],
        errors="coerce",
    )
    output_labels = pd.to_numeric(
        output_dataframe[CURRENT_STATUS_LABEL_COLUMN],
        errors="coerce",
    )
    input_values = pd.to_numeric(
        input_dataframe[columns.value_column],
        errors="coerce",
    )
    output_values = pd.to_numeric(
        output_dataframe[columns.value_column],
        errors="coerce",
    )
    input_methods = (
        input_dataframe[FILL_METHOD_COLUMN]
        .astype("string")
        .fillna("")
        .str.strip()
    )
    output_methods = (
        output_dataframe[FILL_METHOD_COLUMN]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    newly_safe = (
        input_labels.isin(
            UNRESOLVED_CURRENT_LABELS
        )
        & output_labels.eq(3)
    )
    newly_leakage = (
        input_labels.isin(
            UNRESOLVED_CURRENT_LABELS
        )
        & output_labels.eq(4)
    )
    newly_filled = (
        newly_safe
        | newly_leakage
    )

    illegal_label_change = (
        ~newly_filled
        & output_labels.ne(input_labels)
    )

    if illegal_label_change.any():
        raise InputValidationError(
            "Phase 4 changed labels outside successful target fills.",
            error_code="ILLEGAL_PHASE4_LABEL_CHANGE",
            details={
                "invalid_rows": int(
                    illegal_label_change.sum()
                ),
            },
        )

    changed_existing_values = (
        ~newly_filled
        & ~(
            output_values.eq(input_values)
            | (
                output_values.isna()
                & input_values.isna()
            )
        )
    )

    if changed_existing_values.any():
        raise InputValidationError(
            "Phase 4 changed values outside newly filled target rows.",
            error_code="ILLEGAL_PHASE4_VALUE_CHANGE",
            details={
                "invalid_rows": int(
                    changed_existing_values.sum()
                ),
            },
        )

    safe_method_mask = output_methods.isin(
        PHASE_4_SAFE_METHODS
    )
    leakage_method_mask = output_methods.isin(
        PHASE_4_LEAKAGE_METHODS
    )

    if not safe_method_mask.equals(
        newly_safe
    ):
        raise InputValidationError(
            "Safe Phase 4 method provenance does not match new label-3 fills.",
            error_code="INVALID_PHASE4_SAFE_METHOD_ASSIGNMENT",
            details={
                "safe_method_rows": int(
                    safe_method_mask.sum()
                ),
                "new_safe_rows": int(
                    newly_safe.sum()
                ),
            },
        )

    if not leakage_method_mask.equals(
        newly_leakage
    ):
        raise InputValidationError(
            "Leakage Phase 4 method provenance does not match label-4 fills.",
            error_code="INVALID_PHASE4_LEAKAGE_METHOD_ASSIGNMENT",
            details={
                "leakage_method_rows": int(
                    leakage_method_mask.sum()
                ),
                "new_leakage_rows": int(
                    newly_leakage.sum()
                ),
            },
        )

    earlier_methods_changed = (
        ~newly_filled
        & output_methods.ne(
            input_methods
        )
    )

    if earlier_methods_changed.any():
        raise InputValidationError(
            "Phase 4 changed earlier fill-method provenance.",
            error_code="PHASE4_CHANGED_EARLIER_METHODS",
            details={
                "invalid_rows": int(
                    earlier_methods_changed.sum()
                ),
            },
        )

    generated_value_invalid = (
        newly_filled
        & (
            output_values.isna()
            | ~np.isfinite(output_values)
            | output_values.lt(
                config.valid_values.lower_bound
            )
            | output_values.gt(
                config.valid_values.upper_bound
            )
        )
    )

    if generated_value_invalid.any():
        raise InputValidationError(
            "Phase 4 generated values outside the configured valid range.",
            error_code="INVALID_PHASE4_GENERATED_VALUE",
            details={
                "invalid_rows": int(
                    generated_value_invalid.sum()
                ),
            },
        )

    still_unresolved = (
        input_labels.isin(
            UNRESOLVED_CURRENT_LABELS
        )
        & output_labels.isin(
            UNRESOLVED_CURRENT_LABELS
        )
    )

    unresolved_state_invalid = (
        still_unresolved
        & (
            output_values.notna()
            | output_methods.ne(
                input_methods
            )
        )
    )

    if unresolved_state_invalid.any():
        raise InputValidationError(
            "Unresolved Phase 4 targets were not left unchanged.",
            error_code="INVALID_UNRESOLVED_PHASE4_STATE",
            details={
                "invalid_rows": int(
                    unresolved_state_invalid.sum()
                ),
            },
        )


def aggregate_pass_counts(
    *,
    summary: pd.DataFrame,
    pass_number: int,
) -> dict[str, int]:
    """
    Aggregate all numeric Phase 4 metrics for one pass.

    Args:
        summary:
            One-row-per-site-per-pass summary.

        pass_number:
            Pass to aggregate.

    Returns:
        dict[str, int]:
            Aggregated count dictionary.

    Raises:
        None.
    """

    result = new_pass_counts()

    pass_rows = summary.loc[
        summary["pass_number"].eq(
            pass_number
        )
    ]

    for column in result:
        result[column] = int(
            pd.to_numeric(
                pass_rows[column],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

    return result


def fill_phase_4_missing_values(
    *,
    dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
    config: PreprocessingConfig,
    resolved_frequency: ResolvedFrequencyConfig,
    logger: Logger,
    dataset_identifier: str,
    prophet_class: type[Any] | None = None,
) -> Phase4FillingData:
    """
    Run both ordered Prophet passes on one Phase 3 dataset.

    Args:
        dataframe:
            Dataset returned by Phase 3.

        columns:
            Shared dataset column mapping.

        config:
            Complete validated preprocessing configuration.

        resolved_frequency:
            Frequency-dependent Prophet history threshold and gap profile.

        logger:
            Run-specific logger.

        dataset_identifier:
            Stable source identifier used for deterministic random seeds.

        prophet_class:
            Optional Prophet-compatible class for tests. When omitted, the
            optional ``prophet`` package is imported lazily.

    Returns:
        Phase4FillingData:
            Final dataframe, per-site-per-pass summary, and aggregate counts.

    Raises:
        FileProcessingError:
            If Prophet is enabled but the optional package is unavailable.

        InputValidationError:
            If input or output invariants fail.
    """

    working, input_columns = prepare_phase4_dataframe(
        dataframe=dataframe,
        columns=columns,
        config=config,
        resolved_frequency=resolved_frequency,
    )

    resolved_prophet_class = (
        prophet_class
        if prophet_class is not None
        else load_prophet_class()
    )

    updated = working.copy()
    summary_rows: list[
        dict[str, Any]
    ] = []

    grouped_indices = updated.groupby(
        COMPOSITE_ID_COLUMN,
        sort=False,
        dropna=False,
    ).groups

    for _, indices in grouped_indices.items():
        site_dataframe = updated.loc[
            indices
        ].copy()

        (
            processed_site,
            site_summary_rows,
        ) = process_one_site(
            site_dataframe=site_dataframe,
            columns=columns,
            config=config,
            resolved_frequency=resolved_frequency,
            prophet_class=(
                resolved_prophet_class
            ),
            dataset_identifier=(
                dataset_identifier
            ),
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

        summary_rows.extend(
            site_summary_rows
        )

    aligned = updated.reindex(
        dataframe.index
    )

    output_dataframe = dataframe.copy()
    output_dataframe[columns.value_column] = (
        aligned[columns.value_column]
        .to_numpy()
    )
    output_dataframe[
        CURRENT_STATUS_LABEL_COLUMN
    ] = aligned[
        CURRENT_STATUS_LABEL_COLUMN
    ].to_numpy()
    output_dataframe[FILL_METHOD_COLUMN] = (
        aligned[FILL_METHOD_COLUMN]
        .to_numpy()
    )

    validate_phase4_output(
        input_dataframe=dataframe,
        output_dataframe=output_dataframe,
        input_columns=input_columns,
        columns=columns,
        config=config,
    )

    count_columns = list(
        new_pass_counts()
    )

    summary_columns = [
        COMPOSITE_ID_COLUMN,
        KPI_USABILITY_COLUMN,
        "pass_number",
        "status",
        *count_columns,
    ]

    summary = pd.DataFrame(
        summary_rows,
        columns=summary_columns,
    )

    pass_1_counts = aggregate_pass_counts(
        summary=summary,
        pass_number=_PASS_1,
    )
    pass_2_counts = aggregate_pass_counts(
        summary=summary,
        pass_number=_PASS_2,
    )

    processed_site_ids = set(
        summary.loc[
            summary["status"].eq(
                _STATUS_PROCESSED
            ),
            COMPOSITE_ID_COLUMN,
        ].astype(str)
    )
    skipped_site_ids = set(
        summary.loc[
            summary["status"].eq(
                _STATUS_SKIPPED_UNUSABLE
            ),
            COMPOSITE_ID_COLUMN,
        ].astype(str)
    )

    total_safe_filled_points = (
        pass_1_counts["safe_filled_points"]
        + pass_2_counts["safe_filled_points"]
    )
    total_leakage_filled_points = (
        pass_1_counts[
            "leakage_filled_points"
        ]
        + pass_2_counts[
            "leakage_filled_points"
        ]
    )
    total_unresolved_points = (
        pass_1_counts["failed_points"]
        + pass_2_counts["failed_points"]
    )

    logger.info(
        "Phase 4 Prophet filling completed: "
        "pass1_safe=%d, pass1_leakage=%d, pass1_failed=%d, "
        "pass2_safe=%d, pass2_leakage=%d, pass2_failed=%d, "
        "processed_series=%d, skipped_series=%d.",
        pass_1_counts["safe_filled_points"],
        pass_1_counts[
            "leakage_filled_points"
        ],
        pass_1_counts["failed_points"],
        pass_2_counts["safe_filled_points"],
        pass_2_counts[
            "leakage_filled_points"
        ],
        pass_2_counts["failed_points"],
        len(processed_site_ids),
        len(skipped_site_ids),
    )

    return Phase4FillingData(
        dataframe=output_dataframe,
        summary=summary,
        total_series=int(
            summary[
                COMPOSITE_ID_COLUMN
            ].nunique(dropna=False)
        ),
        processed_series=len(
            processed_site_ids
        ),
        skipped_series=len(
            skipped_site_ids
        ),
        pass_1_counts=pass_1_counts,
        pass_2_counts=pass_2_counts,
        total_safe_filled_points=(
            total_safe_filled_points
        ),
        total_leakage_filled_points=(
            total_leakage_filled_points
        ),
        total_unresolved_points=(
            total_unresolved_points
        ),
    )


def save_phase_4_outputs(
    *,
    filling_data: Phase4FillingData,
    source_file: Path,
    output_directory: Path,
) -> tuple[Path, Path]:
    """
    Save the final Phase 4 dataset and per-site-per-pass summary.

    Args:
        filling_data:
            In-memory Phase 4 result.

        source_file:
            Phase 3 source path used for output naming.

        output_directory:
            Destination directory supplied by the preprocessing runner.

    Returns:
        tuple[Path, Path]:
            Final dataset path followed by summary path.

    Raises:
        FileProcessingError:
            If either output cannot be created.
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


def create_phase_4_filled_dataset(
    *,
    source_file: str | Path,
    output_directory: str | Path,
    columns: DatasetColumnMapping,
    config: PreprocessingConfig,
    resolved_frequency: ResolvedFrequencyConfig,
    logger: Logger,
) -> Phase4FillingResult:
    """
    Read, fill, and save the final two-pass Prophet KPI dataset.

    The preprocessing runner should call this function only when
    ``config.phase_4.enabled`` is true and should pass
    ``phase_3_result.output_file`` as ``source_file``.

    Args:
        source_file:
            Exact Phase 3 output dataset.

        output_directory:
            KPI preprocessing directory selected by the runner.

        columns:
            Shared dataset column mapping.

        config:
            Complete validated preprocessing configuration.

        resolved_frequency:
            Frequency-dependent Prophet history threshold and gap profile.

        logger:
            Run-specific logger.

    Returns:
        Phase4FillingResult:
            Final output paths and aggregate pass statistics.

    Raises:
        FileProcessingError:
            If the source file, Prophet package, or outputs cannot be
            processed.

        InputValidationError:
            If Phase 4 is disabled or input/output invariants fail.
    """

    resolved_source_file = Path(
        source_file
    ).expanduser().resolve()
    resolved_output_directory = Path(
        output_directory
    ).expanduser().resolve()

    if not resolved_source_file.is_file():
        raise FileProcessingError(
            "The Phase 3 dataset does not exist.",
            error_code="MISSING_PHASE4_SOURCE_FILE",
            details={
                "source_file": str(
                    resolved_source_file
                ),
            },
        )

    if not config.phase_4.enabled:
        raise InputValidationError(
            "Phase 4 is disabled. The preprocessing runner should return the "
            "Phase 3 output as final instead of calling Phase 4.",
            error_code="PHASE4_WRAPPER_CALLED_WHILE_DISABLED",
        )

    logger.info(
        "Starting two-pass Prophet Phase 4 filling for %s.",
        resolved_source_file.name,
    )

    dataframe = read_csv_file(
        resolved_source_file
    )

    filling_data = fill_phase_4_missing_values(
        dataframe=dataframe,
        columns=columns,
        config=config,
        resolved_frequency=resolved_frequency,
        logger=logger,
        dataset_identifier=(
            resolved_source_file.stem
        ),
    )

    (
        output_file,
        summary_file,
    ) = save_phase_4_outputs(
        filling_data=filling_data,
        source_file=resolved_source_file,
        output_directory=(
            resolved_output_directory
        ),
    )

    result = Phase4FillingResult(
        source_file=resolved_source_file,
        output_file=output_file,
        summary_file=summary_file,
        total_rows=len(
            filling_data.dataframe
        ),
        total_series=(
            filling_data.total_series
        ),
        processed_series=(
            filling_data.processed_series
        ),
        skipped_series=(
            filling_data.skipped_series
        ),
        pass_1_counts=dict(
            filling_data.pass_1_counts
        ),
        pass_2_counts=dict(
            filling_data.pass_2_counts
        ),
        total_safe_filled_points=(
            filling_data
            .total_safe_filled_points
        ),
        total_leakage_filled_points=(
            filling_data
            .total_leakage_filled_points
        ),
        total_unresolved_points=(
            filling_data
            .total_unresolved_points
        ),
    )

    logger.info(
        "Final Phase 4 dataset saved to %s.",
        result.output_file,
    )
    logger.info(
        "Phase 4 summary saved to %s.",
        result.summary_file,
    )

    return result


__all__ = [
    "PASS_1_LEAKAGE_METHOD",
    "PASS_1_SAFE_METHOD",
    "PASS_2_LEAKAGE_METHOD",
    "PASS_2_SAFE_METHOD",
    "PHASE_4_LEAKAGE_METHODS",
    "PHASE_4_METHODS",
    "PHASE_4_SAFE_METHODS",
    "FullSeriesModelCache",
    "OriginalGapBlock",
    "Phase4FillingData",
    "Phase4FillingResult",
    "ProphetModelBundle",
    "create_phase_4_filled_dataset",
    "fill_phase_4_missing_values",
    "load_prophet_class",
    "save_phase_4_outputs",
]