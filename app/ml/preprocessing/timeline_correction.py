"""
Create the initial Part B preprocessing dataset.

This module performs only the first preprocessing stage after Part A EDA:

1. reads one prepared KPI-specific CSV;
2. reuses Part A site, gap, duplicate, and integrity results;
3. keeps the first physical row for duplicate site-timestamp keys;
4. inserts the site-specific timestamps recorded in Part A ``gap_details.csv``;
5. converts invalid KPI values to ``NaN`` using the configured bounds;
6. creates immutable original-condition and mutable current-status labels;
7. calculates the missing percentage for every site series;
8. marks each series as valid or invalid; and
9. writes one new timeline-corrected dataset.

This module does not detect outliers or fill missing KPI values. Status labels
2, 3, and 4 are reserved for later preprocessing stages.
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
    DUPLICATE_TIMESTAMPS_COLUMN,
    EXPECTED_TIMESTAMPS_COLUMN,
    MISSING_ID_VALUE,
    MISSING_TIMESTAMPS_COLUMN,
    OFF_GRID_TIMESTAMPS_COLUMN,
    DataIntegrityResult,
)
from app.schemas.common.dataset import (
    BucketConfig,
    DatasetColumnMapping,
)
from app.schemas.preprocessing import PreprocessingConfig
from app.services.file_service import (
    ensure_directory,
    read_csv_file,
    sanitise_filename,
    write_csv_file,
)


ORIGINAL_CONDITION_LABEL_COLUMN = "Original_Condition_Label"
CURRENT_STATUS_LABEL_COLUMN = "Current_Status_Label"
MISSING_PERCENTAGE_COLUMN = "missing_percentage"
KPI_USABILITY_COLUMN = "kpi_usability"
KPI_INVALID_REASON_COLUMN = "kpi_invalid_reason"

VALID_USABILITY_VALUE = "valid"
INVALID_USABILITY_VALUE = "invalid"
MISSING_PERCENTAGE_REASON = "missing_percentage_above_limit"

GAP_START_COLUMN = "Gap_Start"
GAP_END_COLUMN = "Gap_End"
DUPLICATE_TIMESTAMP_COLUMN = "Timestamp"
DUPLICATE_OCCURRENCES_COLUMN = "Occurrences"

OUTPUT_FILE_SUFFIX = "_timeline_corrected_initial_labels.csv"

_INTERNAL_SITE_ORDER_COLUMN = "__preprocessing_site_order__"


@dataclass(frozen=True, slots=True)
class TimelineCorrectionResult:
    """
    Store the output and headline statistics of initial preprocessing.

    Attributes:
        source_file:
            Prepared KPI-specific CSV received from Part A.

        output_file:
            Newly written timeline-corrected and initially labelled CSV.

        site_summary_file:
            Part A site-summary CSV used as the authoritative site list.

        gap_details_file:
            Part A gap-details CSV used to insert missing timestamps.

        duplicate_details_file:
            Part A duplicate-details CSV used to audit duplicate correction.

        total_input_rows:
            Number of source rows before duplicate removal.

        duplicate_rows_removed:
            Number of later duplicate rows discarded.

        inserted_timestamp_rows:
            Number of new rows created from Part A gap details.

        original_missing_rows:
            Number of original rows labelled ``0``.

        invalid_rows_converted:
            Number of original invalid values converted to ``NaN`` and
            labelled ``-1``.

        original_valid_rows:
            Number of original rows labelled ``1``.

        valid_series:
            Number of site series marked valid.

        invalid_series:
            Number of site series marked invalid.

        final_rows:
            Number of rows in the written output dataset.
    """

    source_file: Path
    output_file: Path
    site_summary_file: Path
    gap_details_file: Path
    duplicate_details_file: Path
    total_input_rows: int
    duplicate_rows_removed: int
    inserted_timestamp_rows: int
    original_missing_rows: int
    invalid_rows_converted: int
    original_valid_rows: int
    valid_series: int
    invalid_series: int
    final_rows: int


def missing_or_blank_mask(
    series: pd.Series,
) -> pd.Series:
    """
    Identify actual missing values and whitespace-only text.

    Numeric zero is retained as a valid supplied value.

    Args:
        series:
            Source series to inspect.

    Returns:
        pd.Series:
            Boolean mask aligned with ``series``.

    Raises:
        None.
    """

    mask = series.isna()

    if (
        pd.api.types.is_object_dtype(series.dtype)
        or pd.api.types.is_string_dtype(series.dtype)
    ):
        blank_mask = (
            series.astype("string")
            .str.strip()
            .eq("")
            .fillna(False)
        )
        mask = mask | blank_mask

    return mask


def convert_bucket_to_frequency(
    bucket: BucketConfig,
) -> pd.Timedelta:
    """
    Convert the resolved Part A bucket into a positive pandas Timedelta.

    Args:
        bucket:
            Uniform bucket configuration returned by dataset preparation.

    Returns:
        pd.Timedelta:
            Expected interval between consecutive timestamps.

    Raises:
        InputValidationError:
            If the bucket unit is unsupported or the resulting frequency is
            not positive.
    """

    unit_mapping = {
        "second": "s",
        "seconds": "s",
        "minute": "min",
        "minutes": "min",
        "hour": "h",
        "hours": "h",
        "day": "D",
        "days": "D",
    }

    normalized_type = bucket.bucket_type.strip().casefold()
    pandas_unit = unit_mapping.get(normalized_type)

    if pandas_unit is None:
        raise InputValidationError(
            "The resolved bucket type is unsupported by preprocessing.",
            error_code="UNSUPPORTED_PREPROCESSING_BUCKET_TYPE",
            details={
                "bucket_type": bucket.bucket_type,
                "supported_bucket_types": sorted(unit_mapping),
            },
        )

    try:
        frequency = pd.to_timedelta(
            bucket.bucket_size,
            unit=pandas_unit,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise InputValidationError(
            "The resolved bucket could not be converted into a frequency.",
            error_code="INVALID_PREPROCESSING_BUCKET",
            details={
                "bucket_type": bucket.bucket_type,
                "bucket_size": bucket.bucket_size,
                "original_error": str(error),
            },
        ) from error

    if frequency <= pd.Timedelta(0):
        raise InputValidationError(
            "The preprocessing frequency must be positive.",
            error_code="NON_POSITIVE_PREPROCESSING_FREQUENCY",
            details={
                "frequency": str(frequency),
            },
        )

    return frequency


def add_composite_id(
    dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
) -> pd.DataFrame:
    """
    Return a dataframe copy containing the Part A composite site identifier.

    The construction exactly follows Part A: configured site values are
    converted to strings, missing identity components become ``<MISSING>``,
    and components are joined using ``" | "``.

    Args:
        dataframe:
            Dataframe containing all configured site identity columns.

        columns:
            Dataset column mapping.

    Returns:
        pd.DataFrame:
            Copy containing ``Composite_ID``.

    Raises:
        KeyError:
            If a required site identity column is unavailable.
    """

    result = dataframe.copy()

    if result.empty:
        result[COMPOSITE_ID_COLUMN] = pd.Series(
            index=result.index,
            dtype="string",
        )
        return result

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


def validate_required_files(
    source_file: Path,
    site_summary_file: Path,
    data_integrity: DataIntegrityResult,
) -> None:
    """
    Validate that the prepared dataset and required Part A outputs exist.

    Args:
        source_file:
            Prepared KPI-specific source CSV.

        site_summary_file:
            Part A site-summary CSV.

        data_integrity:
            Structured Part A integrity result containing report paths.

    Returns:
        None.

    Raises:
        FileProcessingError:
            If any required file is missing.
    """

    required_files = {
        "prepared KPI dataset": source_file,
        "site-analysis summary": site_summary_file,
        "series integrity summary": data_integrity.series_summary_file,
        "gap details": data_integrity.gap_details_file,
        "duplicate details": data_integrity.duplicate_details_file,
    }

    missing_files = {
        description: str(path)
        for description, path in required_files.items()
        if not Path(path).is_file()
    }

    if missing_files:
        raise FileProcessingError(
            "Initial preprocessing is missing required Part A files.",
            error_code="MISSING_PART_A_PREPROCESSING_INPUT",
            details={
                "missing_files": missing_files,
            },
        )


def validate_part_a_integrity_state(
    data_integrity: DataIntegrityResult,
    series_summary: pd.DataFrame,
) -> None:
    """
    Validate that Part A results can be applied without reconstructing a timeline.

    The current Part A result exposes gap records but not explicit timeline
    boundaries. Therefore, this stage rejects source rows outside the selected
    timeline rather than calculating a potentially incorrect missing
    percentage. Off-grid and invalid timestamps are also rejected.

    Args:
        data_integrity:
            Part A integrity result.

        series_summary:
            Part A per-site integrity summary.

    Returns:
        None.

    Raises:
        InputValidationError:
            If Part A found invalid timestamps, rows outside the selected
            timeline, off-grid timestamps, or required summary columns are
            missing.
    """

    required_columns = {
        COMPOSITE_ID_COLUMN,
        EXPECTED_TIMESTAMPS_COLUMN,
        OFF_GRID_TIMESTAMPS_COLUMN,
    }

    missing_columns = sorted(
        required_columns.difference(series_summary.columns)
    )

    if missing_columns:
        raise InputValidationError(
            "The Part A series summary is missing preprocessing fields.",
            error_code="MISSING_PART_A_SERIES_SUMMARY_COLUMNS",
            details={
                "missing_columns": missing_columns,
                "available_columns": list(series_summary.columns),
            },
        )

    if data_integrity.rows_with_invalid_timestamp > 0:
        raise InputValidationError(
            "Initial preprocessing cannot preserve rows with invalid "
            "timestamps.",
            error_code="INVALID_TIMESTAMPS_REPORTED_BY_PART_A",
            details={
                "rows_with_invalid_timestamp": (
                    data_integrity.rows_with_invalid_timestamp
                ),
            },
        )

    if data_integrity.rows_outside_expected_timeline > 0:
        raise InputValidationError(
            "Part A reported rows outside the selected expected timeline. "
            "The current integrity result does not expose timeline boundaries, "
            "so preprocessing will not guess which original rows to remove.",
            error_code="ROWS_OUTSIDE_EXPECTED_TIMELINE",
            details={
                "rows_outside_expected_timeline": (
                    data_integrity.rows_outside_expected_timeline
                ),
            },
        )

    total_off_grid = int(
        pd.to_numeric(
            series_summary[OFF_GRID_TIMESTAMPS_COLUMN],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    if total_off_grid > 0:
        raise InputValidationError(
            "Part A reported timestamps outside the expected bucket grid.",
            error_code="OFF_GRID_TIMESTAMPS_REPORTED_BY_PART_A",
            details={
                "total_off_grid_timestamps": total_off_grid,
            },
        )


def validate_source_columns(
    dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
) -> list[str]:
    """
    Validate the prepared source schema and return its original column order.

    Args:
        dataframe:
            Prepared KPI-specific dataframe.

        columns:
            Dataset column mapping.

    Returns:
        list[str]:
            Original physical source columns in their existing order.

    Raises:
        InputValidationError:
            If the dataframe is empty, required source columns are missing, or
            this stage's generated columns already exist.
    """

    if dataframe.empty:
        raise InputValidationError(
            "Initial preprocessing cannot process an empty dataframe.",
            error_code="EMPTY_INITIAL_PREPROCESSING_DATASET",
        )

    required_columns = [
        *columns.get_site_identity_columns(),
        columns.timestamp_column,
        columns.value_column,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise InputValidationError(
            "The prepared KPI dataframe is missing required columns.",
            error_code="MISSING_INITIAL_PREPROCESSING_COLUMNS",
            details={
                "missing_columns": missing_columns,
                "required_columns": required_columns,
                "available_columns": list(dataframe.columns),
            },
        )

    generated_columns = {
        ORIGINAL_CONDITION_LABEL_COLUMN,
        CURRENT_STATUS_LABEL_COLUMN,
        MISSING_PERCENTAGE_COLUMN,
        KPI_USABILITY_COLUMN,
        KPI_INVALID_REASON_COLUMN,
        _INTERNAL_SITE_ORDER_COLUMN,
    }

    conflicts = sorted(
        generated_columns.intersection(dataframe.columns)
    )

    if conflicts:
        raise InputValidationError(
            "The prepared KPI dataframe already contains columns generated by "
            "initial preprocessing.",
            error_code="INITIAL_PREPROCESSING_COLUMN_CONFLICT",
            details={
                "conflicting_columns": conflicts,
            },
        )

    return list(dataframe.columns)


def prepare_site_summary(
    site_summary: pd.DataFrame,
    columns: DatasetColumnMapping,
) -> pd.DataFrame:
    """
    Validate the Part A site summary and attach stable site order.

    Args:
        site_summary:
            Site-analysis CSV generated by Part A.

        columns:
            Dataset column mapping.

    Returns:
        pd.DataFrame:
            Unique site summary containing ``Composite_ID`` and a stable
            internal site-order column.

    Raises:
        InputValidationError:
            If required site columns are missing or duplicate site identities
            are present.
    """

    site_columns = list(columns.get_site_identity_columns())

    missing_columns = [
        column
        for column in site_columns
        if column not in site_summary.columns
    ]

    if missing_columns:
        raise InputValidationError(
            "The Part A site summary is missing site identity columns.",
            error_code="MISSING_PREPROCESSING_SITE_COLUMNS",
            details={
                "missing_columns": missing_columns,
                "available_columns": list(site_summary.columns),
            },
        )

    prepared = add_composite_id(
        site_summary[site_columns].copy(),
        columns,
    )

    duplicate_sites = prepared.duplicated(
        subset=[COMPOSITE_ID_COLUMN],
        keep=False,
    )

    if duplicate_sites.any():
        raise InputValidationError(
            "The Part A site summary contains duplicate site identities.",
            error_code="DUPLICATE_PART_A_SITE_IDENTITIES",
            details={
                "sample": (
                    prepared.loc[duplicate_sites]
                    .head(20)
                    .to_dict(orient="records")
                ),
            },
        )

    prepared = prepared.reset_index(drop=True)
    prepared[_INTERNAL_SITE_ORDER_COLUMN] = np.arange(
        len(prepared),
        dtype=np.int64,
    )

    return prepared


def prepare_source_dataframe(
    dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
    site_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Parse source timestamps and validate the source site set.

    Args:
        dataframe:
            Prepared KPI-specific dataframe.

        columns:
            Dataset column mapping.

        site_summary:
            Prepared authoritative Part A site summary.

    Returns:
        pd.DataFrame:
            Source dataframe with parsed timestamps and ``Composite_ID``.

    Raises:
        InputValidationError:
            If timestamps are missing or unparseable, or if the source and
            Part A site summary contain different site sets.
    """

    result = dataframe.copy()
    timestamp_column = columns.timestamp_column

    parsed_timestamps = pd.to_datetime(
        result[timestamp_column],
        errors="coerce",
    )

    invalid_timestamp_mask = parsed_timestamps.isna()

    if invalid_timestamp_mask.any():
        sample_columns = [
            *columns.get_site_identity_columns(),
            timestamp_column,
        ]

        raise InputValidationError(
            "The prepared KPI dataframe contains missing or unparseable "
            "timestamps.",
            error_code="INVALID_SOURCE_TIMESTAMPS_IN_PREPROCESSING",
            details={
                "invalid_timestamp_rows": int(invalid_timestamp_mask.sum()),
                "sample": (
                    result.loc[
                        invalid_timestamp_mask,
                        sample_columns,
                    ]
                    .head(20)
                    .to_dict(orient="records")
                ),
            },
        )

    result[timestamp_column] = parsed_timestamps
    result = add_composite_id(result, columns)

    source_site_ids = set(
        result[COMPOSITE_ID_COLUMN].drop_duplicates().tolist()
    )
    summary_site_ids = set(
        site_summary[COMPOSITE_ID_COLUMN].tolist()
    )

    if source_site_ids != summary_site_ids:
        raise InputValidationError(
            "The prepared KPI dataframe and Part A site summary do not contain "
            "the same site identities.",
            error_code="PREPROCESSING_SITE_SET_MISMATCH",
            details={
                "sites_only_in_source": sorted(
                    source_site_ids - summary_site_ids
                )[:20],
                "sites_only_in_summary": sorted(
                    summary_site_ids - source_site_ids
                )[:20],
            },
        )

    return result


def validate_duplicate_report(
    dataframe: pd.DataFrame,
    duplicate_details: pd.DataFrame,
    columns: DatasetColumnMapping,
) -> None:
    """
    Confirm that every duplicate group reported by Part A still exists.

    Duplicate keys are also recalculated locally because the actual correction
    must identify which later physical rows to discard. Extra duplicate groups
    are permitted because Part A reports only rows inside its expected
    timeline, while this correction applies to every source row.

    Args:
        dataframe:
            Parsed source dataframe containing ``Composite_ID``.

        duplicate_details:
            Part A duplicate-details CSV.

        columns:
            Dataset column mapping.

    Returns:
        None.

    Raises:
        InputValidationError:
            If required duplicate-report columns are missing or a reported
            duplicate group no longer matches the prepared dataset.
    """

    required_columns = {
        COMPOSITE_ID_COLUMN,
        DUPLICATE_TIMESTAMP_COLUMN,
        DUPLICATE_OCCURRENCES_COLUMN,
    }

    missing_columns = sorted(
        required_columns.difference(duplicate_details.columns)
    )

    if missing_columns:
        raise InputValidationError(
            "The Part A duplicate report is missing required columns.",
            error_code="MISSING_DUPLICATE_REPORT_COLUMNS",
            details={
                "missing_columns": missing_columns,
                "available_columns": list(duplicate_details.columns),
            },
        )

    if duplicate_details.empty:
        return

    report = duplicate_details[
        [
            COMPOSITE_ID_COLUMN,
            DUPLICATE_TIMESTAMP_COLUMN,
            DUPLICATE_OCCURRENCES_COLUMN,
        ]
    ].copy()

    report[DUPLICATE_TIMESTAMP_COLUMN] = pd.to_datetime(
        report[DUPLICATE_TIMESTAMP_COLUMN],
        errors="coerce",
    )

    if report[DUPLICATE_TIMESTAMP_COLUMN].isna().any():
        raise InputValidationError(
            "The Part A duplicate report contains invalid timestamps.",
            error_code="INVALID_DUPLICATE_REPORT_TIMESTAMP",
        )

    report[DUPLICATE_OCCURRENCES_COLUMN] = pd.to_numeric(
        report[DUPLICATE_OCCURRENCES_COLUMN],
        errors="coerce",
    )

    if report[DUPLICATE_OCCURRENCES_COLUMN].isna().any():
        raise InputValidationError(
            "The Part A duplicate report contains invalid occurrence counts.",
            error_code="INVALID_DUPLICATE_REPORT_OCCURRENCES",
        )

    detected = (
        dataframe
        .groupby(
            [
                COMPOSITE_ID_COLUMN,
                columns.timestamp_column,
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="Detected_Occurrences"
        )
        .rename(
            columns={
                columns.timestamp_column: DUPLICATE_TIMESTAMP_COLUMN,
            }
        )
    )

    comparison = report.merge(
        detected,
        on=[
            COMPOSITE_ID_COLUMN,
            DUPLICATE_TIMESTAMP_COLUMN,
        ],
        how="left",
        validate="one_to_one",
    )

    mismatch_mask = (
        comparison["Detected_Occurrences"].isna()
        | (
            comparison["Detected_Occurrences"]
            != comparison[DUPLICATE_OCCURRENCES_COLUMN]
        )
    )

    if mismatch_mask.any():
        raise InputValidationError(
            "The Part A duplicate report does not match the prepared KPI "
            "dataset.",
            error_code="STALE_DUPLICATE_REPORT",
            details={
                "sample": (
                    comparison.loc[mismatch_mask]
                    .head(20)
                    .to_dict(orient="records")
                ),
            },
        )


def remove_duplicate_timestamps(
    dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
) -> tuple[pd.DataFrame, int]:
    """
    Keep the first physical site-timestamp row and discard later duplicates.

    Args:
        dataframe:
            Parsed source dataframe.

        columns:
            Dataset column mapping.

    Returns:
        tuple[pd.DataFrame, int]:
            Deduplicated dataframe and number of discarded rows.

    Raises:
        None.
    """

    key_columns = [
        *columns.get_site_identity_columns(),
        columns.timestamp_column,
    ]

    duplicate_mask = dataframe.duplicated(
        subset=key_columns,
        keep="first",
    )

    removed_rows = int(duplicate_mask.sum())

    deduplicated = dataframe.loc[
        ~duplicate_mask
    ].copy()

    return (
        deduplicated.reset_index(drop=True),
        removed_rows,
    )


def clean_values_and_create_labels(
    dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
    config: PreprocessingConfig,
) -> tuple[pd.DataFrame, dict[int, int]]:
    """
    Convert invalid source values and initialise condition/status labels.

    Label assignment for original rows:

    - ``0``: originally missing or blank value;
    - ``-1``: supplied numeric value outside the accepted range;
    - ``1``: valid supplied numeric value.

    Values slightly outside a boundary are clipped only when
    ``clip_values_within_tolerance`` is enabled and their distance from the
    nearest boundary does not exceed ``boundary_tolerance``.

    Args:
        dataframe:
            Deduplicated source dataframe.

        columns:
            Dataset column mapping.

        config:
            Validated preprocessing configuration.

    Returns:
        tuple[pd.DataFrame, dict[int, int]]:
            Labelled dataframe and counts for original labels ``-1``, ``0``,
            and ``1``.

    Raises:
        InputValidationError:
            If a non-blank supplied value cannot be parsed as numeric.
    """

    result = dataframe.copy()
    value_column = columns.value_column
    raw_values = result[value_column]

    original_missing_mask = missing_or_blank_mask(raw_values)

    numeric_values = pd.to_numeric(
        raw_values.mask(original_missing_mask, np.nan),
        errors="coerce",
    )

    unexpected_non_numeric_mask = (
        ~original_missing_mask
        & numeric_values.isna()
    )

    if unexpected_non_numeric_mask.any():
        sample_columns = [
            *columns.get_site_identity_columns(),
            columns.timestamp_column,
            value_column,
        ]

        raise InputValidationError(
            "The KPI value column contains non-blank values that are not "
            "numeric.",
            error_code="NON_NUMERIC_KPI_VALUES",
            details={
                "rows": int(unexpected_non_numeric_mask.sum()),
                "sample": (
                    result.loc[
                        unexpected_non_numeric_mask,
                        sample_columns,
                    ]
                    .head(20)
                    .to_dict(orient="records")
                ),
            },
        )

    value_config = config.valid_values
    lower_bound = value_config.lower_bound
    upper_bound = value_config.upper_bound
    tolerance = value_config.boundary_tolerance

    if value_config.clip_values_within_tolerance:
        lower_tolerance_mask = (
            numeric_values.notna()
            & numeric_values.lt(lower_bound)
            & numeric_values.ge(lower_bound - tolerance)
        )
        upper_tolerance_mask = (
            numeric_values.notna()
            & numeric_values.gt(upper_bound)
            & numeric_values.le(upper_bound + tolerance)
        )

        numeric_values = numeric_values.mask(
            lower_tolerance_mask,
            lower_bound,
        )
        numeric_values = numeric_values.mask(
            upper_tolerance_mask,
            upper_bound,
        )

    invalid_value_mask = (
        ~original_missing_mask
        & (
            numeric_values.lt(lower_bound)
            | numeric_values.gt(upper_bound)
            | ~np.isfinite(numeric_values)
        )
    )

    labels = np.full(
        len(result),
        1,
        dtype=np.int8,
    )
    labels[original_missing_mask.to_numpy()] = 0
    labels[invalid_value_mask.to_numpy()] = -1

    result[value_column] = numeric_values.mask(
        invalid_value_mask,
        np.nan,
    )
    result[ORIGINAL_CONDITION_LABEL_COLUMN] = labels
    result[CURRENT_STATUS_LABEL_COLUMN] = labels.copy()

    label_counts = {
        label: int((labels == label).sum())
        for label in (-1, 0, 1)
    }

    return result, label_counts


def get_constant_site_metadata(
    site_dataframe: pd.DataFrame,
    original_columns: list[str],
    columns: DatasetColumnMapping,
    site_row: pd.Series,
) -> dict[str, Any]:
    """
    Resolve values used to populate inserted rows for one site.

    Site identity values come from the Part A site summary. For every other
    original metadata column, exactly one non-missing value is copied when the
    column is constant within the site. Non-constant or entirely missing
    optional metadata is left missing. Timestamp and KPI value are populated
    separately.

    Args:
        site_dataframe:
            Deduplicated original rows for one site.

        original_columns:
            Original source columns in their existing order.

        columns:
            Dataset column mapping.

        site_row:
            Authoritative site-summary row.

    Returns:
        dict[str, Any]:
            Metadata values for inserted rows.

    Raises:
        None.
    """

    timestamp_column = columns.timestamp_column
    value_column = columns.value_column
    site_columns = set(columns.get_site_identity_columns())
    metadata: dict[str, Any] = {}

    for column in original_columns:
        if column in {timestamp_column, value_column}:
            continue

        if column in site_columns:
            metadata[column] = site_row[column]
            continue

        series = site_dataframe[column]
        usable_mask = ~missing_or_blank_mask(series)
        unique_values = series.loc[usable_mask].drop_duplicates()

        if len(unique_values) == 1:
            metadata[column] = unique_values.iloc[0]
        else:
            metadata[column] = pd.NA

    return metadata


def prepare_gap_details(
    gap_details: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate and parse the Part A gap-details dataframe.

    Args:
        gap_details:
            Part A gap-details CSV.

    Returns:
        pd.DataFrame:
            Parsed gap details.

    Raises:
        InputValidationError:
            If required columns are missing or gap values are invalid.
    """

    required_columns = {
        COMPOSITE_ID_COLUMN,
        GAP_START_COLUMN,
        GAP_END_COLUMN,
        MISSING_TIMESTAMPS_COLUMN,
    }

    missing_columns = sorted(
        required_columns.difference(gap_details.columns)
    )

    if missing_columns:
        raise InputValidationError(
            "The Part A gap-details file is missing required columns.",
            error_code="MISSING_GAP_DETAILS_COLUMNS",
            details={
                "missing_columns": missing_columns,
                "available_columns": list(gap_details.columns),
            },
        )

    prepared = gap_details.copy()

    if prepared.empty:
        return prepared

    prepared[GAP_START_COLUMN] = pd.to_datetime(
        prepared[GAP_START_COLUMN],
        errors="coerce",
    )
    prepared[GAP_END_COLUMN] = pd.to_datetime(
        prepared[GAP_END_COLUMN],
        errors="coerce",
    )
    prepared[MISSING_TIMESTAMPS_COLUMN] = pd.to_numeric(
        prepared[MISSING_TIMESTAMPS_COLUMN],
        errors="coerce",
    )

    invalid_mask = (
        prepared[GAP_START_COLUMN].isna()
        | prepared[GAP_END_COLUMN].isna()
        | prepared[MISSING_TIMESTAMPS_COLUMN].isna()
        | prepared[MISSING_TIMESTAMPS_COLUMN].le(0)
        | (
            prepared[GAP_END_COLUMN]
            < prepared[GAP_START_COLUMN]
        )
    )

    if invalid_mask.any():
        raise InputValidationError(
            "The Part A gap-details file contains invalid gap records.",
            error_code="INVALID_GAP_DETAILS",
            details={
                "sample": (
                    prepared.loc[invalid_mask]
                    .head(20)
                    .to_dict(orient="records")
                ),
            },
        )

    prepared[MISSING_TIMESTAMPS_COLUMN] = (
        prepared[MISSING_TIMESTAMPS_COLUMN].astype(int)
    )

    return prepared


def create_inserted_rows(
    dataframe: pd.DataFrame,
    site_summary: pd.DataFrame,
    gap_details: pd.DataFrame,
    original_columns: list[str],
    columns: DatasetColumnMapping,
    frequency: pd.Timedelta,
) -> pd.DataFrame:
    """
    Create missing timestamp rows from Part A site-specific gap records.

    Args:
        dataframe:
            Deduplicated and labelled original dataframe.

        site_summary:
            Prepared Part A site summary.

        gap_details:
            Parsed Part A gap details.

        original_columns:
            Original source columns in their existing order.

        columns:
            Dataset column mapping.

        frequency:
            Expected bucket interval.

    Returns:
        pd.DataFrame:
            Inserted rows containing all original columns and labels ``-2``.

    Raises:
        InputValidationError:
            If a gap references an unknown site, its point count does not match
            its boundaries, gap rows overlap, or a supposedly missing key
            already exists in the source dataframe.
    """

    final_columns = [
        *original_columns,
        ORIGINAL_CONDITION_LABEL_COLUMN,
        CURRENT_STATUS_LABEL_COLUMN,
    ]

    if gap_details.empty:
        return pd.DataFrame(columns=final_columns)

    site_rows_by_id = {
        str(row[COMPOSITE_ID_COLUMN]): row
        for _, row in site_summary.iterrows()
    }

    grouped_source = {
        str(composite_id): group
        for composite_id, group in dataframe.groupby(
            COMPOSITE_ID_COLUMN,
            sort=False,
            dropna=False,
        )
    }

    existing_keys = set(
        zip(
            dataframe[COMPOSITE_ID_COLUMN].astype(str),
            dataframe[columns.timestamp_column],
            strict=True,
        )
    )

    inserted_records: list[dict[str, Any]] = []
    generated_keys: set[tuple[str, pd.Timestamp]] = set()

    for gap_index, gap_row in gap_details.iterrows():
        composite_id = str(gap_row[COMPOSITE_ID_COLUMN])
        site_row = site_rows_by_id.get(composite_id)

        if site_row is None:
            raise InputValidationError(
                "A Part A gap references a site absent from the site summary.",
                error_code="UNKNOWN_SITE_IN_GAP_DETAILS",
                details={
                    "gap_row_index": int(gap_index),
                    "composite_id": composite_id,
                },
            )

        site_dataframe = grouped_source.get(composite_id)

        if site_dataframe is None or site_dataframe.empty:
            raise InputValidationError(
                "A Part A gap references a site with no source rows.",
                error_code="NO_SOURCE_ROWS_FOR_GAP_SITE",
                details={
                    "gap_row_index": int(gap_index),
                    "composite_id": composite_id,
                },
            )

        gap_timestamps = pd.date_range(
            start=gap_row[GAP_START_COLUMN],
            end=gap_row[GAP_END_COLUMN],
            freq=frequency,
        )

        expected_count = int(
            gap_row[MISSING_TIMESTAMPS_COLUMN]
        )

        if len(gap_timestamps) != expected_count:
            raise InputValidationError(
                "A Part A gap's boundaries and missing-point count do not "
                "match the resolved bucket.",
                error_code="GAP_POINT_COUNT_MISMATCH",
                details={
                    "gap_row_index": int(gap_index),
                    "composite_id": composite_id,
                    "gap_start": str(gap_row[GAP_START_COLUMN]),
                    "gap_end": str(gap_row[GAP_END_COLUMN]),
                    "reported_missing_points": expected_count,
                    "generated_missing_points": len(gap_timestamps),
                    "frequency": str(frequency),
                },
            )

        metadata = get_constant_site_metadata(
            site_dataframe=site_dataframe,
            original_columns=original_columns,
            columns=columns,
            site_row=site_row,
        )

        for timestamp in gap_timestamps:
            key = (
                composite_id,
                pd.Timestamp(timestamp),
            )

            if key in existing_keys:
                raise InputValidationError(
                    "A timestamp recorded as missing by Part A already exists "
                    "in the prepared KPI dataset.",
                    error_code="STALE_GAP_DETAILS",
                    details={
                        "composite_id": composite_id,
                        "timestamp": str(timestamp),
                    },
                )

            if key in generated_keys:
                raise InputValidationError(
                    "Part A gap records overlap and would insert the same "
                    "site-timestamp row more than once.",
                    error_code="OVERLAPPING_GAP_DETAILS",
                    details={
                        "composite_id": composite_id,
                        "timestamp": str(timestamp),
                    },
                )

            record = {
                column: pd.NA
                for column in original_columns
            }
            record.update(metadata)
            record[columns.timestamp_column] = pd.Timestamp(timestamp)
            record[columns.value_column] = np.nan
            record[ORIGINAL_CONDITION_LABEL_COLUMN] = -2
            record[CURRENT_STATUS_LABEL_COLUMN] = -2

            inserted_records.append(record)
            generated_keys.add(key)

    return pd.DataFrame(
        inserted_records,
        columns=final_columns,
    )


def combine_and_sort_rows(
    original_rows: pd.DataFrame,
    inserted_rows: pd.DataFrame,
    site_summary: pd.DataFrame,
    original_columns: list[str],
    columns: DatasetColumnMapping,
) -> pd.DataFrame:
    """
    Combine original and inserted rows in Part A site order and timestamp order.

    Args:
        original_rows:
            Deduplicated, cleaned, and labelled source rows.

        inserted_rows:
            Rows created from Part A gap details.

        site_summary:
            Prepared Part A site summary containing stable site order.

        original_columns:
            Original source columns in their existing order.

        columns:
            Dataset column mapping.

    Returns:
        pd.DataFrame:
            Combined sorted dataframe containing ``Composite_ID`` internally.

    Raises:
        InputValidationError:
            If duplicate site-timestamp keys remain after combination.
    """

    final_columns = [
        *original_columns,
        ORIGINAL_CONDITION_LABEL_COLUMN,
        CURRENT_STATUS_LABEL_COLUMN,
    ]

    combined = pd.concat(
        [
            original_rows[final_columns + [COMPOSITE_ID_COLUMN]],
            add_composite_id(
                inserted_rows[final_columns],
                columns,
            ),
        ],
        ignore_index=True,
        sort=False,
    )

    site_order = site_summary[
        [
            COMPOSITE_ID_COLUMN,
            _INTERNAL_SITE_ORDER_COLUMN,
        ]
    ]

    combined = combined.merge(
        site_order,
        on=COMPOSITE_ID_COLUMN,
        how="left",
        sort=False,
        validate="many_to_one",
    )

    if combined[_INTERNAL_SITE_ORDER_COLUMN].isna().any():
        raise InputValidationError(
            "Some preprocessing rows could not be assigned Part A site order.",
            error_code="MISSING_PREPROCESSING_SITE_ORDER",
        )

    duplicate_mask = combined.duplicated(
        subset=[
            COMPOSITE_ID_COLUMN,
            columns.timestamp_column,
        ],
        keep=False,
    )

    if duplicate_mask.any():
        raise InputValidationError(
            "Duplicate site-timestamp keys remain after timeline correction.",
            error_code="DUPLICATES_AFTER_TIMELINE_CORRECTION",
            details={
                "sample": (
                    combined.loc[
                        duplicate_mask,
                        [
                            COMPOSITE_ID_COLUMN,
                            columns.timestamp_column,
                        ],
                    ]
                    .head(20)
                    .to_dict(orient="records")
                ),
            },
        )

    combined = (
        combined
        .sort_values(
            [
                _INTERNAL_SITE_ORDER_COLUMN,
                columns.timestamp_column,
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    return combined


def validate_final_series_lengths(
    dataframe: pd.DataFrame,
    series_summary: pd.DataFrame,
) -> None:
    """
    Confirm that every corrected series contains its Part A expected row count.

    Args:
        dataframe:
            Combined timeline-corrected dataframe.

        series_summary:
            Part A per-site integrity summary.

    Returns:
        None.

    Raises:
        InputValidationError:
            If one or more corrected series do not contain exactly the Part A
            expected number of timestamps.
    """

    expected = series_summary[
        [
            COMPOSITE_ID_COLUMN,
            EXPECTED_TIMESTAMPS_COLUMN,
        ]
    ].copy()

    expected[EXPECTED_TIMESTAMPS_COLUMN] = pd.to_numeric(
        expected[EXPECTED_TIMESTAMPS_COLUMN],
        errors="coerce",
    )

    if expected[EXPECTED_TIMESTAMPS_COLUMN].isna().any():
        raise InputValidationError(
            "The Part A series summary contains invalid expected row counts.",
            error_code="INVALID_EXPECTED_SERIES_LENGTH",
        )

    actual = (
        dataframe
        .groupby(
            COMPOSITE_ID_COLUMN,
            dropna=False,
        )
        .size()
        .reset_index(
            name="Actual_Timestamps"
        )
    )

    comparison = expected.merge(
        actual,
        on=COMPOSITE_ID_COLUMN,
        how="outer",
        validate="one_to_one",
    )

    mismatch_mask = (
        comparison[EXPECTED_TIMESTAMPS_COLUMN].isna()
        | comparison["Actual_Timestamps"].isna()
        | (
            comparison[EXPECTED_TIMESTAMPS_COLUMN]
            != comparison["Actual_Timestamps"]
        )
    )

    if mismatch_mask.any():
        raise InputValidationError(
            "Timeline correction did not produce the Part A expected number "
            "of rows for every site.",
            error_code="FINAL_SERIES_LENGTH_MISMATCH",
            details={
                "sample": (
                    comparison.loc[mismatch_mask]
                    .head(20)
                    .to_dict(orient="records")
                ),
            },
        )


def apply_series_usability(
    dataframe: pd.DataFrame,
    config: PreprocessingConfig,
) -> tuple[pd.DataFrame, int, int]:
    """
    Calculate site missing percentages and mark each series valid or invalid.

    Missing percentage uses original condition labels ``-2``, ``-1``, and
    ``0`` divided by the final expected row count for that site. A series is
    invalid only when its percentage is strictly greater than
    ``maximum_missing_percentage``. Equality remains valid.

    Args:
        dataframe:
            Timeline-corrected dataframe containing ``Composite_ID`` and
            original-condition labels.

        config:
            Validated preprocessing configuration.

    Returns:
        tuple[pd.DataFrame, int, int]:
            Dataframe with usability columns, valid-series count, and
            invalid-series count.

    Raises:
        None.
    """

    result = dataframe.copy()
    missing_labels = {-2, -1, 0}
    threshold = (
        config.series_usability.maximum_missing_percentage
    )

    series_statistics = (
        result
        .groupby(
            COMPOSITE_ID_COLUMN,
            dropna=False,
        )[ORIGINAL_CONDITION_LABEL_COLUMN]
        .agg(
            total_rows="size",
            missing_rows=lambda values: int(
                values.isin(missing_labels).sum()
            ),
        )
        .reset_index()
    )

    series_statistics[MISSING_PERCENTAGE_COLUMN] = (
        series_statistics["missing_rows"]
        / series_statistics["total_rows"]
        * 100.0
    )

    invalid_mask = (
        series_statistics[MISSING_PERCENTAGE_COLUMN]
        > threshold
    )

    series_statistics[KPI_USABILITY_COLUMN] = np.where(
        invalid_mask,
        INVALID_USABILITY_VALUE,
        VALID_USABILITY_VALUE,
    )
    series_statistics[KPI_INVALID_REASON_COLUMN] = np.where(
        invalid_mask,
        MISSING_PERCENTAGE_REASON,
        "",
    )

    result = result.merge(
        series_statistics[
            [
                COMPOSITE_ID_COLUMN,
                MISSING_PERCENTAGE_COLUMN,
                KPI_USABILITY_COLUMN,
                KPI_INVALID_REASON_COLUMN,
            ]
        ],
        on=COMPOSITE_ID_COLUMN,
        how="left",
        validate="many_to_one",
    )

    valid_series = int((~invalid_mask).sum())
    invalid_series = int(invalid_mask.sum())

    return result, valid_series, invalid_series


def save_timeline_corrected_dataset(
    dataframe: pd.DataFrame,
    source_file: Path,
    output_directory: Path,
    original_columns: list[str],
) -> Path:
    """
    Save the new timeline-corrected dataset in final column order.

    Args:
        dataframe:
            Fully corrected and usability-labelled dataframe.

        source_file:
            Prepared KPI-specific source file used to construct the output
            filename.

        output_directory:
            Destination directory.

        original_columns:
            Original source columns in their existing order.

    Returns:
        Path:
            Written CSV path.

    Raises:
        FileProcessingError:
            If the output directory or CSV cannot be written.
    """

    resolved_output_directory = ensure_directory(
        output_directory
    )

    output_name = sanitise_filename(
        f"{source_file.stem}{OUTPUT_FILE_SUFFIX}"
    )
    output_file = (
        resolved_output_directory
        / output_name
    )

    output_columns = [
        *original_columns,
        ORIGINAL_CONDITION_LABEL_COLUMN,
        CURRENT_STATUS_LABEL_COLUMN,
        MISSING_PERCENTAGE_COLUMN,
        KPI_USABILITY_COLUMN,
        KPI_INVALID_REASON_COLUMN,
    ]

    write_csv_file(
        dataframe[output_columns],
        output_file,
    )

    return output_file


def create_timeline_corrected_dataset(
    *,
    source_file: str | Path,
    site_summary_file: str | Path,
    data_integrity: DataIntegrityResult,
    output_directory: str | Path,
    columns: DatasetColumnMapping,
    bucket: BucketConfig,
    config: PreprocessingConfig,
    logger: Logger,
) -> TimelineCorrectionResult:
    """
    Create the first Part B preprocessing dataset from Part A results.

    This is the main orchestration function for the module. The future
    preprocessing runner should pass exact paths and structured results from
    one ``KPIEDARunResult`` rather than searching Part A directories.

    Args:
        source_file:
            ``kpi_result.prepared_dataset.file_path``.

        site_summary_file:
            ``kpi_result.site_analysis.summary_file``.

        data_integrity:
            ``kpi_result.data_integrity``.

        output_directory:
            Directory selected by the preprocessing runner for this KPI.

        columns:
            Shared dataset column mapping from ``DatasetInputConfig``.

        bucket:
            Resolved uniform bucket from
            ``eda_result.preparation.bucket``.

        config:
            Validated Part B preprocessing configuration.

        logger:
            Run-specific logger.

    Returns:
        TimelineCorrectionResult:
            Output path and correction statistics.

    Raises:
        InputValidationError:
            If source data, Part A results, timestamps, gaps, duplicates,
            values, or final invariants are invalid.

        FileProcessingError:
            If required files cannot be read or the output cannot be written.
    """

    resolved_source_file = Path(
        source_file
    ).expanduser().resolve()
    resolved_site_summary_file = Path(
        site_summary_file
    ).expanduser().resolve()
    resolved_output_directory = Path(
        output_directory
    ).expanduser().resolve()

    validate_required_files(
        source_file=resolved_source_file,
        site_summary_file=resolved_site_summary_file,
        data_integrity=data_integrity,
    )

    if (
        resolved_source_file
        != Path(data_integrity.source_file).expanduser().resolve()
    ):
        raise InputValidationError(
            "The supplied prepared KPI file does not match the Part A "
            "data-integrity result.",
            error_code="PART_A_SOURCE_FILE_MISMATCH",
            details={
                "supplied_source_file": str(resolved_source_file),
                "integrity_source_file": str(data_integrity.source_file),
            },
        )

    logger.info(
        "Starting timeline correction and initial labelling for %s.",
        resolved_source_file.name,
    )

    source_dataframe = read_csv_file(
        resolved_source_file
    )
    site_summary_dataframe = read_csv_file(
        resolved_site_summary_file
    )
    series_summary_dataframe = read_csv_file(
        data_integrity.series_summary_file
    )
    gap_details_dataframe = read_csv_file(
        data_integrity.gap_details_file
    )
    duplicate_details_dataframe = read_csv_file(
        data_integrity.duplicate_details_file
    )

    original_columns = validate_source_columns(
        dataframe=source_dataframe,
        columns=columns,
    )

    validate_part_a_integrity_state(
        data_integrity=data_integrity,
        series_summary=series_summary_dataframe,
    )

    prepared_site_summary = prepare_site_summary(
        site_summary=site_summary_dataframe,
        columns=columns,
    )

    prepared_source = prepare_source_dataframe(
        dataframe=source_dataframe,
        columns=columns,
        site_summary=prepared_site_summary,
    )

    validate_duplicate_report(
        dataframe=prepared_source,
        duplicate_details=duplicate_details_dataframe,
        columns=columns,
    )

    deduplicated_source, duplicate_rows_removed = (
        remove_duplicate_timestamps(
            dataframe=prepared_source,
            columns=columns,
        )
    )

    labelled_source, label_counts = (
        clean_values_and_create_labels(
            dataframe=deduplicated_source,
            columns=columns,
            config=config,
        )
    )

    prepared_gap_details = prepare_gap_details(
        gap_details=gap_details_dataframe,
    )

    frequency = convert_bucket_to_frequency(
        bucket=bucket,
    )

    inserted_rows = create_inserted_rows(
        dataframe=labelled_source,
        site_summary=prepared_site_summary,
        gap_details=prepared_gap_details,
        original_columns=original_columns,
        columns=columns,
        frequency=frequency,
    )

    combined = combine_and_sort_rows(
        original_rows=labelled_source,
        inserted_rows=inserted_rows,
        site_summary=prepared_site_summary,
        original_columns=original_columns,
        columns=columns,
    )

    validate_final_series_lengths(
        dataframe=combined,
        series_summary=series_summary_dataframe,
    )

    combined, valid_series, invalid_series = (
        apply_series_usability(
            dataframe=combined,
            config=config,
        )
    )

    output_file = save_timeline_corrected_dataset(
        dataframe=combined,
        source_file=resolved_source_file,
        output_directory=resolved_output_directory,
        original_columns=original_columns,
    )

    result = TimelineCorrectionResult(
        source_file=resolved_source_file,
        output_file=output_file,
        site_summary_file=resolved_site_summary_file,
        gap_details_file=Path(
            data_integrity.gap_details_file
        ).expanduser().resolve(),
        duplicate_details_file=Path(
            data_integrity.duplicate_details_file
        ).expanduser().resolve(),
        total_input_rows=len(source_dataframe),
        duplicate_rows_removed=duplicate_rows_removed,
        inserted_timestamp_rows=len(inserted_rows),
        original_missing_rows=label_counts.get(0, 0),
        invalid_rows_converted=label_counts.get(-1, 0),
        original_valid_rows=label_counts.get(1, 0),
        valid_series=valid_series,
        invalid_series=invalid_series,
        final_rows=len(combined),
    )

    logger.info(
        "Initial preprocessing completed for %s: input_rows=%d, "
        "duplicate_rows_removed=%d, inserted_rows=%d, "
        "original_missing=%d, invalid_converted=%d, original_valid=%d, "
        "valid_series=%d, invalid_series=%d, final_rows=%d.",
        resolved_source_file.name,
        result.total_input_rows,
        result.duplicate_rows_removed,
        result.inserted_timestamp_rows,
        result.original_missing_rows,
        result.invalid_rows_converted,
        result.original_valid_rows,
        result.valid_series,
        result.invalid_series,
        result.final_rows,
    )

    logger.info(
        "Timeline-corrected dataset saved to %s.",
        result.output_file,
    )

    return result


__all__ = [
    "CURRENT_STATUS_LABEL_COLUMN",
    "KPI_INVALID_REASON_COLUMN",
    "KPI_USABILITY_COLUMN",
    "MISSING_PERCENTAGE_COLUMN",
    "ORIGINAL_CONDITION_LABEL_COLUMN",
    "TimelineCorrectionResult",
    "create_timeline_corrected_dataset",
]