"""
Label KPI outliers after timeline correction and initial labelling.

This module supports exactly one selected outlier method per preprocessing run:

- ``global_iqr``;
- ``one_day_centered_iqr``; or
- ``user_bounds``.

Every method supports exactly one selected direction:

- ``lower_only``;
- ``upper_only``; or
- ``both``.

The module expects the dataset shape produced by ``timeline_correction.py``.
It changes only ``Current_Status_Label`` from ``1`` to ``2`` for detected
outliers. KPI values and ``Original_Condition_Label`` remain unchanged.

Invalid site series are skipped. IQR reference values come only from rows that
were originally valid and trusted:

    Original_Condition_Label == 1
    finite KPI value

The one-day-centred method is the established batch detector. Its centred
window may use observations before and after the target timestamp.
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

from .timeline_correction import (
    CURRENT_STATUS_LABEL_COLUMN,
    INVALID_USABILITY_VALUE,
    KPI_USABILITY_COLUMN,
    ORIGINAL_CONDITION_LABEL_COLUMN,
    VALID_USABILITY_VALUE,
)


OUTPUT_FILE_SUFFIX = "_outlier_labelled.csv"
SUMMARY_FILE_SUFFIX = "_outlier_summary.csv"

MINIMUM_GLOBAL_IQR_REFERENCE_POINTS = 4

STATUS_OK = "ok"
STATUS_SKIPPED_UNUSABLE = "skipped_unusable_series"
STATUS_INSUFFICIENT_REFERENCES = "insufficient_reference_points"
STATUS_NO_CENTERED_BOUNDS = "no_centered_bounds_available"

_ALLOWED_ORIGINAL_LABELS = {-2, -1, 0, 1}
_ALLOWED_INITIAL_CURRENT_LABELS = {-2, -1, 0, 1}


@dataclass(frozen=True, slots=True)
class OutlierDetection:
    """
    Store row-aligned detector output for one site series.

    Attributes:
        outlier_mask:
            Rows detected as outliers by the selected method and direction.

        lower_outlier_mask:
            Rows below the applicable lower threshold.

        upper_outlier_mask:
            Rows above the applicable upper threshold.

        evaluated_mask:
            Original-valid rows for which the selected detector had enough
            information to make a decision.

        lower_bound:
            Scalar or row-aligned lower threshold. ``None`` when lower-side
            detection is not selected.

        upper_bound:
            Scalar or row-aligned upper threshold. ``None`` when upper-side
            detection is not selected.

        q1:
            Global or rolling first quartile when an IQR method is selected.

        q3:
            Global or rolling third quartile when an IQR method is selected.

        iqr:
            Global or rolling interquartile range when an IQR method is
            selected.

        status:
            Detector status for the site series.
    """

    outlier_mask: pd.Series
    lower_outlier_mask: pd.Series
    upper_outlier_mask: pd.Series
    evaluated_mask: pd.Series
    lower_bound: float | pd.Series | None
    upper_bound: float | pd.Series | None
    q1: float | pd.Series | None
    q3: float | pd.Series | None
    iqr: float | pd.Series | None
    status: str


@dataclass(frozen=True, slots=True)
class OutlierLabellingData:
    """
    Store the in-memory outlier-labelling result.

    Attributes:
        dataframe:
            Dataset copy with detected rows changed to current status ``2``.

        summary:
            One-row-per-site detector summary.

        total_outliers:
            Total number of rows labelled ``2``.

        lower_outliers:
            Number of lower-side outliers.

        upper_outliers:
            Number of upper-side outliers.

        evaluated_points:
            Number of original-valid rows actually evaluated.

        unevaluated_valid_points:
            Number of original-valid rows that could not be evaluated, such
            as centred-window boundary rows without enough references.

        processed_series:
            Number of usable site series passed to the selected detector.

        skipped_series:
            Number of unusable site series skipped.
    """

    dataframe: pd.DataFrame
    summary: pd.DataFrame
    total_outliers: int
    lower_outliers: int
    upper_outliers: int
    evaluated_points: int
    unevaluated_valid_points: int
    processed_series: int
    skipped_series: int


@dataclass(frozen=True, slots=True)
class OutlierLabellingResult:
    """
    Store saved outlier-labelling outputs and headline statistics.

    Attributes:
        source_file:
            Timeline-corrected dataset received from the previous stage.

        output_file:
            New outlier-labelled dataset.

        summary_file:
            Per-site detector summary CSV.

        method:
            Selected detector method.

        direction:
            Selected outlier direction.

        total_rows:
            Number of rows preserved in the output dataset.

        total_series:
            Number of unique site series.

        processed_series:
            Number of usable series processed.

        skipped_series:
            Number of unusable series skipped.

        evaluated_points:
            Number of valid points evaluated.

        unevaluated_valid_points:
            Number of original-valid points without an applicable threshold.

        total_outliers:
            Total number of rows marked with current status ``2``.

        lower_outliers:
            Number of lower-side outliers.

        upper_outliers:
            Number of upper-side outliers.
    """

    source_file: Path
    output_file: Path
    summary_file: Path
    method: str
    direction: str
    total_rows: int
    total_series: int
    processed_series: int
    skipped_series: int
    evaluated_points: int
    unevaluated_valid_points: int
    total_outliers: int
    lower_outliers: int
    upper_outliers: int


def add_composite_id(
    dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
) -> pd.DataFrame:
    """
    Return a dataframe copy containing the Part A composite site identifier.

    Args:
        dataframe:
            Dataset containing the configured site identity columns.

        columns:
            Shared physical column mapping.

    Returns:
        pd.DataFrame:
            Copy containing ``Composite_ID``.

    Raises:
        KeyError:
            If a configured site identity column is absent.
    """

    result = dataframe.copy()
    site_columns = list(columns.get_site_identity_columns())

    identity_values = (
        result[site_columns]
        .astype("string")
        .fillna(MISSING_ID_VALUE)
    )

    result[COMPOSITE_ID_COLUMN] = identity_values.agg(
        COMPOSITE_ID_SEPARATOR.join,
        axis=1,
    )

    return result


def _parse_integer_labels(
    dataframe: pd.DataFrame,
    column_name: str,
) -> pd.Series:
    """
    Parse one label column as integer values.

    Args:
        dataframe:
            Dataset containing the label column.

        column_name:
            Label column to parse.

    Returns:
        pd.Series:
            Integer label series aligned with the dataframe.

    Raises:
        InputValidationError:
            If a label is missing, non-numeric, or non-integer.
    """

    numeric_labels = pd.to_numeric(
        dataframe[column_name],
        errors="coerce",
    )

    invalid_mask = (
        numeric_labels.isna()
        | numeric_labels.mod(1).ne(0)
    )

    if invalid_mask.any():
        raise InputValidationError(
            f"{column_name} contains missing, non-numeric, or non-integer "
            "labels.",
            error_code="INVALID_OUTLIER_INPUT_LABELS",
            details={
                "column": column_name,
                "invalid_rows": int(invalid_mask.sum()),
            },
        )

    return numeric_labels.astype(np.int8)


def validate_outlier_input(
    dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
    config: PreprocessingConfig,
    resolved_frequency: ResolvedFrequencyConfig,
) -> pd.DataFrame:
    """
    Validate and normalize the timeline-corrected detector input.

    This stage is intentionally strict. It rejects an accidental rerun or a
    dataset already modified by a later filling stage.

    Args:
        dataframe:
            Timeline-corrected dataset.

        columns:
            Shared physical column mapping.

        config:
            Validated preprocessing configuration.

        resolved_frequency:
            Frequency-dependent settings resolved once by the runner.

    Returns:
        pd.DataFrame:
            Validated copy with parsed timestamps, numeric KPI values, integer
            labels, normalized usability, and internal ``Composite_ID``.

    Raises:
        InputValidationError:
            If required columns, timestamps, values, labels, usability, site
            keys, or centred-window settings are invalid.
    """

    if dataframe.empty:
        raise InputValidationError(
            "Outlier labelling cannot process an empty dataframe.",
            error_code="EMPTY_OUTLIER_DATASET",
        )

    required_columns = [
        *columns.get_site_identity_columns(),
        columns.timestamp_column,
        columns.value_column,
        ORIGINAL_CONDITION_LABEL_COLUMN,
        CURRENT_STATUS_LABEL_COLUMN,
        KPI_USABILITY_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise InputValidationError(
            "The timeline-corrected dataset is missing columns required for "
            "outlier labelling.",
            error_code="MISSING_OUTLIER_INPUT_COLUMNS",
            details={
                "missing_columns": missing_columns,
                "required_columns": required_columns,
                "available_columns": list(dataframe.columns),
            },
        )

    result = dataframe.copy()

    parsed_timestamps = pd.to_datetime(
        result[columns.timestamp_column],
        errors="coerce",
    )

    if parsed_timestamps.isna().any():
        raise InputValidationError(
            "The outlier input contains missing or unparseable timestamps.",
            error_code="INVALID_OUTLIER_TIMESTAMPS",
            details={
                "invalid_rows": int(parsed_timestamps.isna().sum()),
            },
        )

    result[columns.timestamp_column] = parsed_timestamps
    result[columns.value_column] = pd.to_numeric(
        result[columns.value_column],
        errors="coerce",
    )

    original_labels = _parse_integer_labels(
        dataframe=result,
        column_name=ORIGINAL_CONDITION_LABEL_COLUMN,
    )
    current_labels = _parse_integer_labels(
        dataframe=result,
        column_name=CURRENT_STATUS_LABEL_COLUMN,
    )

    unexpected_original = ~original_labels.isin(
        _ALLOWED_ORIGINAL_LABELS
    )
    unexpected_current = ~current_labels.isin(
        _ALLOWED_INITIAL_CURRENT_LABELS
    )

    if unexpected_original.any():
        raise InputValidationError(
            "Original_Condition_Label contains unsupported values.",
            error_code="UNEXPECTED_ORIGINAL_LABEL_BEFORE_OUTLIERS",
            details={
                "labels": sorted(
                    original_labels.loc[unexpected_original]
                    .unique()
                    .tolist()
                ),
            },
        )

    if unexpected_current.any():
        raise InputValidationError(
            "Current_Status_Label shows that outlier labelling or filling has "
            "already modified this dataset.",
            error_code="OUTLIER_STAGE_ALREADY_APPLIED",
            details={
                "labels": sorted(
                    current_labels.loc[unexpected_current]
                    .unique()
                    .tolist()
                ),
            },
        )

    if not current_labels.equals(original_labels):
        mismatch_mask = current_labels.ne(original_labels)

        raise InputValidationError(
            "Current_Status_Label must match Original_Condition_Label before "
            "the outlier stage begins.",
            error_code="OUTLIER_INPUT_LABEL_MISMATCH",
            details={
                "mismatched_rows": int(mismatch_mask.sum()),
            },
        )

    result[ORIGINAL_CONDITION_LABEL_COLUMN] = original_labels
    result[CURRENT_STATUS_LABEL_COLUMN] = current_labels

    values = result[columns.value_column]
    original_valid_mask = original_labels.eq(1)
    original_missing_mask = original_labels.isin({-2, -1, 0})

    valid_but_missing = original_valid_mask & (
        values.isna()
        | ~np.isfinite(values)
    )
    missing_origin_but_present = (
        original_missing_mask
        & values.notna()
    )

    if valid_but_missing.any():
        raise InputValidationError(
            "Rows labelled original-valid contain missing or non-finite KPI "
            "values.",
            error_code="VALID_LABEL_WITH_INVALID_VALUE",
            details={
                "rows": int(valid_but_missing.sum()),
            },
        )

    if missing_origin_but_present.any():
        raise InputValidationError(
            "Rows with original labels -2, -1, or 0 contain KPI values.",
            error_code="MISSING_LABEL_WITH_PRESENT_VALUE",
            details={
                "rows": int(missing_origin_but_present.sum()),
            },
        )

    valid_lower = config.valid_values.lower_bound
    valid_upper = config.valid_values.upper_bound

    outside_valid_range = (
        original_valid_mask
        & (
            values.lt(valid_lower)
            | values.gt(valid_upper)
        )
    )

    if outside_valid_range.any():
        raise InputValidationError(
            "Original-valid rows lie outside the configured KPI valid range.",
            error_code="OUTLIER_INPUT_OUTSIDE_VALID_RANGE",
            details={
                "rows": int(outside_valid_range.sum()),
                "valid_lower_bound": valid_lower,
                "valid_upper_bound": valid_upper,
            },
        )

    usability = (
        result[KPI_USABILITY_COLUMN]
        .astype("string")
        .str.strip()
        .str.casefold()
    )

    invalid_usability = ~usability.isin(
        {
            VALID_USABILITY_VALUE,
            INVALID_USABILITY_VALUE,
        }
    )

    if invalid_usability.any():
        raise InputValidationError(
            "kpi_usability contains values other than valid or invalid.",
            error_code="INVALID_KPI_USABILITY_FOR_OUTLIERS",
            details={
                "values": sorted(
                    usability.loc[invalid_usability]
                    .dropna()
                    .unique()
                    .tolist()
                ),
            },
        )

    result[KPI_USABILITY_COLUMN] = usability
    result = add_composite_id(result, columns)

    duplicate_key_mask = result.duplicated(
        subset=[
            COMPOSITE_ID_COLUMN,
            columns.timestamp_column,
        ],
        keep=False,
    )

    if duplicate_key_mask.any():
        raise InputValidationError(
            "Duplicate site-timestamp rows remain before outlier labelling.",
            error_code="DUPLICATE_KEYS_BEFORE_OUTLIERS",
            details={
                "rows": int(duplicate_key_mask.sum()),
            },
        )

    usability_counts = (
        result
        .groupby(
            COMPOSITE_ID_COLUMN,
            dropna=False,
        )[KPI_USABILITY_COLUMN]
        .nunique(dropna=False)
    )

    inconsistent_sites = usability_counts.gt(1)

    if inconsistent_sites.any():
        raise InputValidationError(
            "kpi_usability must be constant within each site series.",
            error_code="INCONSISTENT_SITE_USABILITY",
            details={
                "composite_ids": (
                    usability_counts.index[inconsistent_sites]
                    .astype(str)
                    .tolist()[:20]
                ),
            },
        )

    if config.outlier.method == "one_day_centered_iqr":
        window_points = resolved_frequency.centered_window_points
        minimum_points = (
            resolved_frequency.centered_min_valid_points
        )

        if window_points is None or minimum_points is None:
            raise InputValidationError(
                "The one-day-centred detector requires resolved window and "
                "minimum-valid point counts.",
                error_code="MISSING_CENTERED_IQR_RESOLUTION",
            )

        if window_points <= 0 or window_points % 2 == 0:
            raise InputValidationError(
                "The centred-IQR rolling window must be a positive odd number "
                "of points.",
                error_code="INVALID_CENTERED_IQR_WINDOW",
                details={
                    "centered_window_points": window_points,
                },
            )

        if minimum_points <= 0 or minimum_points > window_points:
            raise InputValidationError(
                "The centred-IQR minimum valid points must lie between one and "
                "the rolling-window size.",
                error_code="INVALID_CENTERED_IQR_MINIMUM",
                details={
                    "centered_min_valid_points": minimum_points,
                    "centered_window_points": window_points,
                },
            )

    return result


def _false_mask(
    index: pd.Index,
) -> pd.Series:
    """
    Create a row-aligned all-False mask.

    Args:
        index:
            Target row index.

    Returns:
        pd.Series:
            Boolean series containing only False.

    Raises:
        None.
    """

    return pd.Series(
        False,
        index=index,
        dtype=bool,
    )


def _build_directional_masks(
    *,
    values: pd.Series,
    eligible_mask: pd.Series,
    direction: str,
    lower_bound: float | pd.Series | None,
    upper_bound: float | pd.Series | None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Build lower, upper, and combined outlier masks.

    Threshold comparisons are strict. Values exactly equal to a configured or
    calculated threshold are not outliers.

    Args:
        values:
            Numeric KPI values.

        eligible_mask:
            Rows allowed to be judged.

        direction:
            ``lower_only``, ``upper_only``, or ``both``.

        lower_bound:
            Scalar or row-aligned lower threshold.

        upper_bound:
            Scalar or row-aligned upper threshold.

    Returns:
        tuple[pd.Series, pd.Series, pd.Series]:
            Lower mask, upper mask, and combined mask.

    Raises:
        InputValidationError:
            If a selected direction is missing its required threshold.
    """

    lower_mask = _false_mask(values.index)
    upper_mask = _false_mask(values.index)

    if direction in {"lower_only", "both"}:
        if lower_bound is None:
            raise InputValidationError(
                "Lower-side detection requires a lower threshold.",
                error_code="MISSING_LOWER_OUTLIER_THRESHOLD",
            )

        lower_mask = (
            eligible_mask
            & values.lt(lower_bound)
        )

    if direction in {"upper_only", "both"}:
        if upper_bound is None:
            raise InputValidationError(
                "Upper-side detection requires an upper threshold.",
                error_code="MISSING_UPPER_OUTLIER_THRESHOLD",
            )

        upper_mask = (
            eligible_mask
            & values.gt(upper_bound)
        )

    return (
        lower_mask,
        upper_mask,
        lower_mask | upper_mask,
    )


def detect_global_iqr(
    values: pd.Series,
    original_labels: pd.Series,
    config: PreprocessingConfig,
) -> OutlierDetection:
    """
    Detect per-site global IQR outliers.

    The detector requires at least four original-valid finite reference values,
    matching the established global-IQR implementation.

    Args:
        values:
            Site KPI values in timeline order.

        original_labels:
            Immutable original-condition labels.

        config:
            Validated preprocessing configuration.

    Returns:
        OutlierDetection:
            Row-aligned masks, scalar thresholds, quartiles, and status.

    Raises:
        InputValidationError:
            If the IQR multiplier was not resolved.
    """

    eligible_mask = (
        original_labels.eq(1)
        & values.notna()
        & np.isfinite(values)
    )
    reference = values.loc[eligible_mask]

    empty_mask = _false_mask(values.index)

    if len(reference) < MINIMUM_GLOBAL_IQR_REFERENCE_POINTS:
        return OutlierDetection(
            outlier_mask=empty_mask,
            lower_outlier_mask=empty_mask.copy(),
            upper_outlier_mask=empty_mask.copy(),
            evaluated_mask=empty_mask.copy(),
            lower_bound=None,
            upper_bound=None,
            q1=None,
            q3=None,
            iqr=None,
            status=STATUS_INSUFFICIENT_REFERENCES,
        )

    multiplier = config.outlier.iqr_multiplier

    if multiplier is None:
        raise InputValidationError(
            "The global-IQR multiplier was not resolved.",
            error_code="MISSING_GLOBAL_IQR_MULTIPLIER",
        )

    q1 = float(reference.quantile(0.25))
    q3 = float(reference.quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    (
        lower_mask,
        upper_mask,
        outlier_mask,
    ) = _build_directional_masks(
        values=values,
        eligible_mask=eligible_mask,
        direction=config.outlier.direction,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )

    return OutlierDetection(
        outlier_mask=outlier_mask,
        lower_outlier_mask=lower_mask,
        upper_outlier_mask=upper_mask,
        evaluated_mask=eligible_mask,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        q1=q1,
        q3=q3,
        iqr=iqr,
        status=STATUS_OK,
    )


def detect_centered_iqr(
    values: pd.Series,
    original_labels: pd.Series,
    config: PreprocessingConfig,
    resolved_frequency: ResolvedFrequencyConfig,
) -> OutlierDetection:
    """
    Detect per-site one-day-centred rolling-IQR outliers.

    The rolling reference contains only original-valid finite values. The
    current point remains eligible for evaluation only when its rolling
    threshold exists.

    Args:
        values:
            Site KPI values in timeline order.

        original_labels:
            Immutable original-condition labels.

        config:
            Validated preprocessing configuration.

        resolved_frequency:
            Resolved centred window and minimum-valid point counts.

    Returns:
        OutlierDetection:
            Row-aligned rolling thresholds, masks, and status.

    Raises:
        InputValidationError:
            If required centred settings or the IQR multiplier are unavailable.
    """

    window_points = resolved_frequency.centered_window_points
    minimum_points = (
        resolved_frequency.centered_min_valid_points
    )
    multiplier = config.outlier.iqr_multiplier

    if (
        window_points is None
        or minimum_points is None
        or multiplier is None
    ):
        raise InputValidationError(
            "The one-day-centred IQR settings were not fully resolved.",
            error_code="INCOMPLETE_CENTERED_IQR_SETTINGS",
        )

    original_valid_mask = (
        original_labels.eq(1)
        & values.notna()
        & np.isfinite(values)
    )

    rolling_reference = values.where(
        original_valid_mask
    )

    rolling_q1 = rolling_reference.rolling(
        window=window_points,
        center=True,
        min_periods=minimum_points,
    ).quantile(0.25)

    rolling_q3 = rolling_reference.rolling(
        window=window_points,
        center=True,
        min_periods=minimum_points,
    ).quantile(0.75)

    rolling_iqr = rolling_q3 - rolling_q1
    rolling_lower = (
        rolling_q1
        - multiplier * rolling_iqr
    )
    rolling_upper = (
        rolling_q3
        + multiplier * rolling_iqr
    )

    bounds_available = (
        rolling_lower.notna()
        & rolling_upper.notna()
    )
    evaluated_mask = (
        original_valid_mask
        & bounds_available
    )

    (
        lower_mask,
        upper_mask,
        outlier_mask,
    ) = _build_directional_masks(
        values=values,
        eligible_mask=evaluated_mask,
        direction=config.outlier.direction,
        lower_bound=rolling_lower,
        upper_bound=rolling_upper,
    )

    status = (
        STATUS_OK
        if evaluated_mask.any()
        else STATUS_NO_CENTERED_BOUNDS
    )

    return OutlierDetection(
        outlier_mask=outlier_mask,
        lower_outlier_mask=lower_mask,
        upper_outlier_mask=upper_mask,
        evaluated_mask=evaluated_mask,
        lower_bound=rolling_lower,
        upper_bound=rolling_upper,
        q1=rolling_q1,
        q3=rolling_q3,
        iqr=rolling_iqr,
        status=status,
    )


def detect_user_bounds(
    values: pd.Series,
    original_labels: pd.Series,
    config: PreprocessingConfig,
) -> OutlierDetection:
    """
    Detect outliers using user-specified site-independent bounds.

    Args:
        values:
            Site KPI values.

        original_labels:
            Immutable original-condition labels.

        config:
            Validated preprocessing configuration.

    Returns:
        OutlierDetection:
            Row-aligned masks and configured scalar thresholds.

    Raises:
        InputValidationError:
            If the selected direction lacks its required bound.
    """

    eligible_mask = (
        original_labels.eq(1)
        & values.notna()
        & np.isfinite(values)
    )

    lower_bound = config.outlier.user_lower_bound
    upper_bound = config.outlier.user_upper_bound

    (
        lower_mask,
        upper_mask,
        outlier_mask,
    ) = _build_directional_masks(
        values=values,
        eligible_mask=eligible_mask,
        direction=config.outlier.direction,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )

    return OutlierDetection(
        outlier_mask=outlier_mask,
        lower_outlier_mask=lower_mask,
        upper_outlier_mask=upper_mask,
        evaluated_mask=eligible_mask,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        q1=None,
        q3=None,
        iqr=None,
        status=STATUS_OK,
    )


def detect_site_outliers(
    values: pd.Series,
    original_labels: pd.Series,
    config: PreprocessingConfig,
    resolved_frequency: ResolvedFrequencyConfig,
) -> OutlierDetection:
    """
    Dispatch one site series to the selected outlier method.

    Args:
        values:
            Site KPI values.

        original_labels:
            Immutable original-condition labels.

        config:
            Validated preprocessing configuration.

        resolved_frequency:
            Frequency-dependent point counts.

    Returns:
        OutlierDetection:
            Result from the selected detector.

    Raises:
        InputValidationError:
            If an unsupported method reaches the processing stage.
    """

    method = config.outlier.method

    if method == "global_iqr":
        return detect_global_iqr(
            values=values,
            original_labels=original_labels,
            config=config,
        )

    if method == "one_day_centered_iqr":
        return detect_centered_iqr(
            values=values,
            original_labels=original_labels,
            config=config,
            resolved_frequency=resolved_frequency,
        )

    if method == "user_bounds":
        return detect_user_bounds(
            values=values,
            original_labels=original_labels,
            config=config,
        )

    raise InputValidationError(
        "An unsupported outlier method reached the detector.",
        error_code="UNSUPPORTED_OUTLIER_METHOD",
        details={
            "method": method,
        },
    )


def _finite_bound_statistics(
    bound: float | pd.Series | None,
) -> tuple[float | None, float | None]:
    """
    Return minimum and maximum finite values from a scalar or series threshold.

    Args:
        bound:
            Scalar threshold, row-aligned threshold series, or ``None``.

    Returns:
        tuple[float | None, float | None]:
            Minimum and maximum finite threshold values.

    Raises:
        None.
    """

    if bound is None:
        return None, None

    if np.isscalar(bound):
        numeric = float(bound)

        if not np.isfinite(numeric):
            return None, None

        return numeric, numeric

    numeric_bound = pd.to_numeric(
        bound,
        errors="coerce",
    )
    finite_values = numeric_bound.loc[
        numeric_bound.notna()
        & np.isfinite(numeric_bound)
    ]

    if finite_values.empty:
        return None, None

    return (
        float(finite_values.min()),
        float(finite_values.max()),
    )


def _scalar_statistic(
    statistic: float | pd.Series | None,
    reducer: str,
) -> float | None:
    """
    Reduce a scalar or rolling statistic for the per-site summary.

    Args:
        statistic:
            Scalar, row-aligned series, or ``None``.

        reducer:
            ``median`` or ``mean``.

    Returns:
        float | None:
            Reduced finite value.

    Raises:
        ValueError:
            If ``reducer`` is unsupported.
    """

    if statistic is None:
        return None

    if np.isscalar(statistic):
        numeric = float(statistic)

        return numeric if np.isfinite(numeric) else None

    numeric_statistic = pd.to_numeric(
        statistic,
        errors="coerce",
    )
    finite_values = numeric_statistic.loc[
        numeric_statistic.notna()
        & np.isfinite(numeric_statistic)
    ]

    if finite_values.empty:
        return None

    if reducer == "median":
        return float(finite_values.median())

    if reducer == "mean":
        return float(finite_values.mean())

    raise ValueError(
        f"Unsupported statistic reducer: {reducer}"
    )


def label_outliers(
    dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
    config: PreprocessingConfig,
    resolved_frequency: ResolvedFrequencyConfig,
    logger: Logger,
) -> OutlierLabellingData:
    """
    Apply the selected detector to every usable site series in memory.

    The returned dataframe preserves the input row count, input row order, and
    all input columns. Only ``Current_Status_Label`` may change.

    Args:
        dataframe:
            Timeline-corrected dataset.

        columns:
            Shared physical column mapping.

        config:
            Validated preprocessing configuration.

        resolved_frequency:
            Frequency settings resolved by
            ``PreprocessingConfig.resolve_for_bucket``.

        logger:
            Run-specific logger.

    Returns:
        OutlierLabellingData:
            Labelled dataframe, site summary, and detector totals.

    Raises:
        InputValidationError:
            If input validation fails or an unsupported detector state occurs.
    """

    original_columns = list(dataframe.columns)

    working = validate_outlier_input(
        dataframe=dataframe,
        columns=columns,
        config=config,
        resolved_frequency=resolved_frequency,
    )

    timestamp_column = columns.timestamp_column
    value_column = columns.value_column
    site_columns = list(
        columns.get_site_identity_columns()
    )

    # Preserve exact input order after per-site timeline calculations.
    working["__outlier_input_order__"] = np.arange(
        len(working),
        dtype=np.int64,
    )

    summary_rows: list[dict[str, Any]] = []

    total_outliers = 0
    lower_outliers = 0
    upper_outliers = 0
    evaluated_points = 0
    unevaluated_valid_points = 0
    processed_series = 0
    skipped_series = 0

    grouped = working.groupby(
        COMPOSITE_ID_COLUMN,
        sort=False,
        dropna=False,
    )

    for composite_id, site_dataframe in grouped:
        site_dataframe = site_dataframe.sort_values(
            timestamp_column,
            kind="stable",
        )

        site_index = site_dataframe.index
        usability = str(
            site_dataframe[KPI_USABILITY_COLUMN].iloc[0]
        )

        values = site_dataframe[value_column]
        original_labels = site_dataframe[
            ORIGINAL_CONDITION_LABEL_COLUMN
        ]
        original_valid_mask = (
            original_labels.eq(1)
            & values.notna()
            & np.isfinite(values)
        )
        reference_points = int(
            original_valid_mask.sum()
        )

        site_identity = {
            column: site_dataframe[column].iloc[0]
            for column in site_columns
        }

        if usability != VALID_USABILITY_VALUE:
            skipped_series += 1

            summary_rows.append(
                {
                    COMPOSITE_ID_COLUMN: composite_id,
                    **site_identity,
                    "Outlier_Method": config.outlier.method,
                    "Outlier_Direction": config.outlier.direction,
                    KPI_USABILITY_COLUMN: usability,
                    "Reference_Points": reference_points,
                    "Evaluated_Points": 0,
                    "Unevaluated_Valid_Points": reference_points,
                    "Lower_Outliers": 0,
                    "Upper_Outliers": 0,
                    "Total_Outliers": 0,
                    "Q1": None,
                    "Q3": None,
                    "IQR": None,
                    "Lower_Bound_Min": None,
                    "Lower_Bound_Max": None,
                    "Upper_Bound_Min": None,
                    "Upper_Bound_Max": None,
                    "Status": STATUS_SKIPPED_UNUSABLE,
                }
            )
            continue

        processed_series += 1

        detection = detect_site_outliers(
            values=values,
            original_labels=original_labels,
            config=config,
            resolved_frequency=resolved_frequency,
        )

        detected_indices = site_index[
            detection.outlier_mask.to_numpy()
        ]

        if len(detected_indices) > 0:
            working.loc[
                detected_indices,
                CURRENT_STATUS_LABEL_COLUMN,
            ] = 2

        site_lower_outliers = int(
            detection.lower_outlier_mask.sum()
        )
        site_upper_outliers = int(
            detection.upper_outlier_mask.sum()
        )
        site_total_outliers = int(
            detection.outlier_mask.sum()
        )
        site_evaluated = int(
            detection.evaluated_mask.sum()
        )
        site_unevaluated = (
            reference_points
            - site_evaluated
        )

        total_outliers += site_total_outliers
        lower_outliers += site_lower_outliers
        upper_outliers += site_upper_outliers
        evaluated_points += site_evaluated
        unevaluated_valid_points += site_unevaluated

        lower_min, lower_max = (
            _finite_bound_statistics(
                detection.lower_bound
            )
        )
        upper_min, upper_max = (
            _finite_bound_statistics(
                detection.upper_bound
            )
        )

        summary_rows.append(
            {
                COMPOSITE_ID_COLUMN: composite_id,
                **site_identity,
                "Outlier_Method": config.outlier.method,
                "Outlier_Direction": config.outlier.direction,
                KPI_USABILITY_COLUMN: usability,
                "Reference_Points": reference_points,
                "Evaluated_Points": site_evaluated,
                "Unevaluated_Valid_Points": site_unevaluated,
                "Lower_Outliers": site_lower_outliers,
                "Upper_Outliers": site_upper_outliers,
                "Total_Outliers": site_total_outliers,
                "Q1": _scalar_statistic(
                    detection.q1,
                    "median",
                ),
                "Q3": _scalar_statistic(
                    detection.q3,
                    "median",
                ),
                "IQR": _scalar_statistic(
                    detection.iqr,
                    "median",
                ),
                "Lower_Bound_Min": lower_min,
                "Lower_Bound_Max": lower_max,
                "Upper_Bound_Min": upper_min,
                "Upper_Bound_Max": upper_max,
                "Status": detection.status,
            }
        )

    ordered_current_labels = (
        working
        .sort_values(
            "__outlier_input_order__",
            kind="stable",
        )[CURRENT_STATUS_LABEL_COLUMN]
        .to_numpy(dtype=np.int8)
    )

    # Preserve every original physical column and representation exactly.
    # Parsed timestamps and the internal Composite_ID are calculation-only.
    labelled = dataframe.reset_index(drop=True).copy()
    labelled[CURRENT_STATUS_LABEL_COLUMN] = (
        ordered_current_labels
    )
    labelled = labelled[original_columns]

    # Final invariants: only current status may differ.
    original_values = dataframe.reset_index(drop=True)
    changed_columns = [
        column
        for column in original_columns
        if column != CURRENT_STATUS_LABEL_COLUMN
        and not labelled[column].equals(
            original_values[column]
        )
    ]

    if changed_columns:
        raise InputValidationError(
            "Outlier labelling changed columns other than "
            "Current_Status_Label.",
            error_code="OUTLIER_STAGE_CHANGED_PROTECTED_COLUMNS",
            details={
                "changed_columns": changed_columns,
            },
        )

    final_current = _parse_integer_labels(
        dataframe=labelled,
        column_name=CURRENT_STATUS_LABEL_COLUMN,
    )

    allowed_final_labels = {-2, -1, 0, 1, 2}
    unexpected_final = ~final_current.isin(
        allowed_final_labels
    )

    if unexpected_final.any():
        raise InputValidationError(
            "Outlier labelling produced unsupported current labels.",
            error_code="INVALID_LABEL_AFTER_OUTLIERS",
        )

    final_outlier_count = int(
        final_current.eq(2).sum()
    )

    if final_outlier_count != total_outliers:
        raise InputValidationError(
            "The final label-2 count does not match detector accounting.",
            error_code="OUTLIER_COUNT_MISMATCH",
            details={
                "detected_outliers": total_outliers,
                "final_label_2_count": final_outlier_count,
            },
        )

    original_provenance = _parse_integer_labels(
        dataframe=labelled,
        column_name=ORIGINAL_CONDITION_LABEL_COLUMN,
    )

    provenance_mismatch = (
        original_provenance.ne(
            _parse_integer_labels(
                dataframe=original_values,
                column_name=ORIGINAL_CONDITION_LABEL_COLUMN,
            )
        )
    )

    if provenance_mismatch.any():
        raise InputValidationError(
            "Original_Condition_Label changed during outlier labelling.",
            error_code="ORIGINAL_LABEL_CHANGED_BY_OUTLIERS",
        )

    summary = pd.DataFrame(
        summary_rows
    )

    logger.info(
        "Outlier labelling completed in memory: method=%s, direction=%s, "
        "processed_series=%d, skipped_series=%d, evaluated_points=%d, "
        "outliers=%d.",
        config.outlier.method,
        config.outlier.direction,
        processed_series,
        skipped_series,
        evaluated_points,
        total_outliers,
    )

    return OutlierLabellingData(
        dataframe=labelled,
        summary=summary,
        total_outliers=total_outliers,
        lower_outliers=lower_outliers,
        upper_outliers=upper_outliers,
        evaluated_points=evaluated_points,
        unevaluated_valid_points=(
            unevaluated_valid_points
        ),
        processed_series=processed_series,
        skipped_series=skipped_series,
    )


def save_outlier_outputs(
    *,
    labelled_data: OutlierLabellingData,
    source_file: Path,
    output_directory: Path,
) -> tuple[Path, Path]:
    """
    Save the labelled dataset and small per-site detector summary.

    Args:
        labelled_data:
            In-memory detector result.

        source_file:
            Timeline-corrected input path used for output naming.

        output_directory:
            Destination directory supplied by the preprocessing runner.

    Returns:
        tuple[Path, Path]:
            Outlier-labelled dataset path and summary path.

    Raises:
        FileProcessingError:
            If output creation fails.
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
        labelled_data.dataframe,
        output_file,
    )
    write_csv_file(
        labelled_data.summary,
        summary_file,
    )

    return output_file, summary_file


def create_outlier_labelled_dataset(
    *,
    source_file: str | Path,
    output_directory: str | Path,
    columns: DatasetColumnMapping,
    config: PreprocessingConfig,
    resolved_frequency: ResolvedFrequencyConfig,
    logger: Logger,
) -> OutlierLabellingResult:
    """
    Read, label, and save one timeline-corrected KPI dataset.

    The future preprocessing runner should supply
    ``timeline_result.output_file`` as ``source_file``. For a load-once runner,
    call ``label_outliers`` directly with the in-memory dataframe and use
    ``save_outlier_outputs`` only when an intermediate checkpoint is required.

    Args:
        source_file:
            Exact output path returned by ``timeline_correction.py``.

        output_directory:
            KPI preprocessing directory selected by the runner.

        columns:
            Shared dataset column mapping.

        config:
            Validated preprocessing configuration.

        resolved_frequency:
            Result of ``config.resolve_for_bucket(resolved_bucket)``.

        logger:
            Run-specific logger.

    Returns:
        OutlierLabellingResult:
            Saved paths, detector selection, and headline counts.

    Raises:
        FileProcessingError:
            If the source file is missing or an output cannot be written.

        InputValidationError:
            If input validation or detector accounting fails.
    """

    resolved_source_file = Path(
        source_file
    ).expanduser().resolve()
    resolved_output_directory = Path(
        output_directory
    ).expanduser().resolve()

    if not resolved_source_file.is_file():
        raise FileProcessingError(
            "The timeline-corrected dataset does not exist.",
            error_code="MISSING_TIMELINE_CORRECTED_DATASET",
            details={
                "source_file": str(resolved_source_file),
            },
        )

    logger.info(
        "Starting outlier labelling for %s using method=%s and direction=%s.",
        resolved_source_file.name,
        config.outlier.method,
        config.outlier.direction,
    )

    dataframe = read_csv_file(
        resolved_source_file
    )

    labelled_data = label_outliers(
        dataframe=dataframe,
        columns=columns,
        config=config,
        resolved_frequency=resolved_frequency,
        logger=logger,
    )

    output_file, summary_file = save_outlier_outputs(
        labelled_data=labelled_data,
        source_file=resolved_source_file,
        output_directory=resolved_output_directory,
    )

    total_series = (
        labelled_data.processed_series
        + labelled_data.skipped_series
    )

    result = OutlierLabellingResult(
        source_file=resolved_source_file,
        output_file=output_file,
        summary_file=summary_file,
        method=config.outlier.method,
        direction=config.outlier.direction,
        total_rows=len(labelled_data.dataframe),
        total_series=total_series,
        processed_series=(
            labelled_data.processed_series
        ),
        skipped_series=labelled_data.skipped_series,
        evaluated_points=(
            labelled_data.evaluated_points
        ),
        unevaluated_valid_points=(
            labelled_data.unevaluated_valid_points
        ),
        total_outliers=labelled_data.total_outliers,
        lower_outliers=labelled_data.lower_outliers,
        upper_outliers=labelled_data.upper_outliers,
    )

    logger.info(
        "Outlier-labelled dataset saved to %s. Summary saved to %s.",
        result.output_file,
        result.summary_file,
    )

    return result


__all__ = [
    "MINIMUM_GLOBAL_IQR_REFERENCE_POINTS",
    "OutlierDetection",
    "OutlierLabellingData",
    "OutlierLabellingResult",
    "create_outlier_labelled_dataset",
    "detect_centered_iqr",
    "detect_global_iqr",
    "detect_site_outliers",
    "detect_user_bounds",
    "label_outliers",
    "save_outlier_outputs",
]