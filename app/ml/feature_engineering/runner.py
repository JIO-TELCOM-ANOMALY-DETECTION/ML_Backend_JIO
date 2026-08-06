"""Orchestrate Part C feature engineering for one preprocessed KPI dataset.

Intended project location::

    app/ml/feature_engineering/runner.py

This runner receives the exact ``PreprocessingRunnerResult`` produced by Part B
and uses its ``final_dataset_file`` as the Part C source. It does not search
folders or reconstruct earlier-stage filenames.

Processing order
----------------

1. validate the Part B handoff and runner arguments;
2. resolve physical dataset columns;
3. resolve ``FeatureEngineeringConfig`` once for the Part A bucket;
4. load, normalize, and validate the complete Part B dataframe;
5. process every site independently in chronological order;
6. skip invalid sites while retaining their rows with ``NaN`` features;
7. prepare shared raw/corrected-history sources once per valid site;
8. dispatch every validated feature request in schema order;
9. assemble and validate the complete feature-engineered dataset; and
10. save the final dataset, site summary, and optional source audit.

The runner performs orchestration only. Feature formulas remain inside the
individual feature modules, while request validation and bucket-aware history
resolution remain inside the schema.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from logging import Logger
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from app.core.exceptions import (
    FileProcessingError,
    InputValidationError,
)
from app.ml.preprocessing.runner import PreprocessingRunnerResult
from app.schemas.common.dataset import (
    BucketConfig,
    DatasetInputConfig,
)
from app.schemas.feature_engineering import (
    BaselineFeatureRequest,
    CalendarFeatureRequest,
    CurrentValueFeatureRequest,
    DifferenceFeatureRequest,
    EWMAFeatureRequest,
    FeatureEngineeringConfig,
    FeatureRequest,
    LagFeatureRequest,
    PercentageChangeFeatureRequest,
    ResidualFeatureRequest,
    ResolvedFeatureEngineeringConfig,
    RollingFeatureRequest,
    RollingSlopeFeatureRequest,
    RollingZScoreFeatureRequest,
    SeasonalDifferenceFeatureRequest,
    TimeBucketFeatureRequest,
    TimeCyclicFeatureRequest,
    get_feature_columns,
)
from app.services.file_service import (
    ensure_directory,
    read_csv_file,
    sanitise_filename,
    write_csv_file,
)

from .baseline_features import generate_baseline_features
from .calendar_features import generate_calendar_features
from .common import (
    FeatureInputColumns,
    append_feature_output,
    iter_site_groups,
    normalize_feature_output,
    resolve_feature_input_columns,
)
from .comparison_features import generate_comparison_features
from .feature_validation import (
    DEFAULT_STATUS_COLUMN,
    DEFAULT_USABILITY_COLUMN,
    FeatureDatasetValidationResult,
    SiteFeatureValidationResult,
    prepare_and_validate_feature_dataframe,
    validate_generated_feature_dataframe,
)
from .lag_features import generate_lag_features
from .residual_features import generate_residual_features
from .rolling_features import generate_rolling_features
from .source_preparation import (
    PreparedFeatureSources,
    prepare_feature_sources,
)
from .time_features import generate_time_features
from .value_features import generate_value_features


FINAL_DATASET_SUFFIX: Final = "_feature_engineered.csv"
SUMMARY_DIRECTORY_NAME: Final = "summaries"
INTERMEDIATE_DIRECTORY_NAME: Final = "intermediate"
SITE_SUMMARY_FILE_NAME: Final = "01_feature_engineering_site_summary.csv"
CORRECTED_HISTORY_AUDIT_FILE_NAME: Final = "01_corrected_history_audit.csv"


@dataclass(frozen=True, slots=True)
class FeatureEngineeringSiteRunSummary:
    """Store one site's Part C validation and generation statistics.

    Attributes:
        composite_id:
            Readable site identity constructed from configured site columns.
        row_count:
            Number of rows belonging to the site.
        usability:
            Normalized Part B usability value.
        minimum_history_points:
            Run-level history requirement resolved by the schema.
        has_sufficient_history:
            Whether the site contains at least that many rows.
        can_generate_features:
            Whether feature modules were executed for the site.
        skip_reason:
            Reason feature generation was skipped, otherwise ``None``.
        generated_feature_columns:
            Number of schema-requested columns generated for a processable site.
        populated_feature_cells:
            Number of non-missing feature cells generated for the site.
        missing_feature_cells:
            Number of missing/warm-up feature cells for the site.
        corrected_history_unavailable_points:
            Raw points masked while constructing corrected history.
        corrected_history_trailing_median_replacements:
            Masked points replaced using past-only trailing medians.
        corrected_history_forward_fills:
            Remaining masked points replaced using forward fill.
        corrected_history_unresolved_points:
            Points still missing in corrected history.
        minimum_timestamp:
            Earliest site timestamp after normalization.
        maximum_timestamp:
            Latest site timestamp after normalization.
    """

    composite_id: str
    row_count: int
    usability: str
    minimum_history_points: int
    has_sufficient_history: bool
    can_generate_features: bool
    skip_reason: str | None
    generated_feature_columns: int
    populated_feature_cells: int
    missing_feature_cells: int
    corrected_history_unavailable_points: int
    corrected_history_trailing_median_replacements: int
    corrected_history_forward_fills: int
    corrected_history_unresolved_points: int
    minimum_timestamp: pd.Timestamp | None
    maximum_timestamp: pd.Timestamp | None


@dataclass(frozen=True, slots=True)
class FeatureEngineeringRunnerResult:
    """Store Part C paths, resolved settings, and headline run statistics.

    Attributes:
        kpi_name:
            KPI identity inherited from the Part B result.
        source_file:
            Exact final preprocessed dataset received from Part B.
        output_directory:
            Workflow-owned Part C output directory.
        summary_directory:
            Runner-created directory containing the site summary.
        intermediate_directory:
            Optional directory containing source-preparation audits.
        resolved_config:
            Bucket-aware feature configuration resolved once for the run.
        validation_result:
            Complete dataset/site readiness result from input validation.
        final_dataset_file:
            Saved feature-engineered dataset.
        site_summary_file:
            Saved one-row-per-site feature-engineering summary.
        corrected_history_audit_file:
            Optional corrected-history audit file, otherwise ``None``.
        feature_columns:
            Deterministic generated feature columns in schema order.
        total_rows:
            Number of rows retained in the output.
        total_sites:
            Number of site series in the input.
        processed_sites:
            Valid sites for which feature generators were executed.
        skipped_invalid_sites:
            Invalid sites retained with all feature values left missing.
        insufficient_history_sites:
            Valid sites below the resolved history recommendation.
        populated_feature_cells:
            Total non-missing generated feature cells.
        missing_feature_cells:
            Total missing generated feature cells, including warm-up rows and
            skipped invalid sites.
        site_summaries:
            Structured per-site generation statistics.
    """

    kpi_name: str
    source_file: Path
    output_directory: Path
    summary_directory: Path
    intermediate_directory: Path | None
    resolved_config: ResolvedFeatureEngineeringConfig
    validation_result: FeatureDatasetValidationResult
    final_dataset_file: Path
    site_summary_file: Path
    corrected_history_audit_file: Path | None
    feature_columns: tuple[str, ...]
    total_rows: int
    total_sites: int
    processed_sites: int
    skipped_invalid_sites: int
    insufficient_history_sites: int
    populated_feature_cells: int
    missing_feature_cells: int
    site_summaries: tuple[FeatureEngineeringSiteRunSummary, ...]


def _validate_nonblank_column_name(
    value: str,
    *,
    field_name: str,
) -> str:
    """Validate one non-empty runtime column name and return it stripped."""

    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(
            f"{field_name} must be a non-empty string.",
            error_code="INVALID_FEATURE_RUNTIME_COLUMN_NAME",
            details={"field_name": field_name},
        )

    return value.strip()


def validate_preprocessing_handoff(
    preprocessing_result: PreprocessingRunnerResult,
) -> Path:
    """Validate the exact Part B final dataset supplied to Part C.

    Args:
        preprocessing_result:
            Structured Part B result for one KPI.

    Returns:
        Path:
            Resolved existing Part B final dataset path.

    Raises:
        InputValidationError:
            If the supplied object is not a Part B runner result.
        FileProcessingError:
            If the final preprocessed dataset does not exist.
    """

    if not isinstance(preprocessing_result, PreprocessingRunnerResult):
        raise InputValidationError(
            "preprocessing_result must be PreprocessingRunnerResult.",
            error_code="INVALID_FEATURE_PREPROCESSING_RESULT",
            details={
                "received_type": type(preprocessing_result).__name__,
            },
        )

    source_file = Path(
        preprocessing_result.final_dataset_file
    ).expanduser().resolve()

    if not source_file.is_file():
        raise FileProcessingError(
            "The final preprocessed dataset returned by Part B does not "
            "exist.",
            error_code="MISSING_FEATURE_SOURCE_FILE",
            details={"source_file": str(source_file)},
        )

    return source_file


def _resolve_kpi_name(
    preprocessing_result: PreprocessingRunnerResult,
    *,
    source_file: Path,
) -> str:
    """Resolve a non-empty KPI name for logs and output filenames."""

    raw_name = getattr(preprocessing_result, "kpi_name", None)

    if raw_name is not None:
        normalized = str(raw_name).strip()
        if normalized:
            return normalized

    fallback = source_file.stem.strip()
    if fallback:
        return fallback

    raise InputValidationError(
        "The Part B result does not contain a usable KPI name.",
        error_code="MISSING_FEATURE_KPI_NAME",
    )


def save_named_dataframe(
    *,
    dataframe: pd.DataFrame,
    output_directory: Path,
    filename: str,
) -> Path:
    """Save one dataframe using a stable runner-owned filename."""

    resolved_directory = ensure_directory(output_directory)
    output_file = resolved_directory / filename
    write_csv_file(dataframe, output_file)
    return output_file


def _resolve_and_validate_config(
    *,
    dataset_config: DatasetInputConfig,
    feature_config: FeatureEngineeringConfig,
    bucket: BucketConfig,
    preprocessing_result: PreprocessingRunnerResult,
) -> tuple[FeatureInputColumns, ResolvedFeatureEngineeringConfig]:
    """Resolve physical columns and one bucket-aware feature configuration."""

    if not isinstance(dataset_config, DatasetInputConfig):
        raise InputValidationError(
            "dataset_config must be DatasetInputConfig.",
            error_code="INVALID_FEATURE_DATASET_CONFIG",
            details={"received_type": type(dataset_config).__name__},
        )

    if not isinstance(feature_config, FeatureEngineeringConfig):
        raise InputValidationError(
            "feature_config must be FeatureEngineeringConfig.",
            error_code="INVALID_FEATURE_CONFIG",
            details={"received_type": type(feature_config).__name__},
        )

    if not isinstance(bucket, BucketConfig):
        raise InputValidationError(
            "bucket must be the resolved BucketConfig returned by Part A.",
            error_code="INVALID_FEATURE_BUCKET",
            details={"received_type": type(bucket).__name__},
        )

    try:
        input_columns = resolve_feature_input_columns(dataset_config)
        resolved_config = feature_config.resolve_for_bucket(bucket)
    except (TypeError, ValueError) as error:
        raise InputValidationError(
            "Feature-engineering configuration or dataset columns could not "
            "be resolved.",
            error_code="FEATURE_CONFIG_RESOLUTION_FAILED",
            details={"reason": str(error)},
        ) from error

    preprocessing_bucket_minutes = getattr(
        preprocessing_result.resolved_frequency,
        "bucket_minutes",
        None,
    )

    if (
        preprocessing_bucket_minutes is not None
        and int(preprocessing_bucket_minutes)
        != resolved_config.bucket_minutes
    ):
        raise InputValidationError(
            "Part B and Part C were given different resolved buckets.",
            error_code="FEATURE_BUCKET_MISMATCH",
            details={
                "preprocessing_bucket_minutes": int(
                    preprocessing_bucket_minutes
                ),
                "feature_bucket_minutes": resolved_config.bucket_minutes,
            },
        )

    return input_columns, resolved_config


def _dispatch_feature_request(
    request: FeatureRequest,
    *,
    site_dataframe: pd.DataFrame,
    sources: PreparedFeatureSources,
    timestamp_column: str,
    bucket_minutes: int,
) -> pd.DataFrame:
    """Dispatch one validated feature request to its calculation module.

    This function performs no feature calculation. It only maps schema request
    classes to the correct generator and supplies the already prepared inputs.
    """

    if isinstance(request, CurrentValueFeatureRequest):
        return generate_value_features(
            request,
            sources=sources,
        )

    if isinstance(request, LagFeatureRequest):
        return generate_lag_features(
            request,
            sources=sources,
            bucket_minutes=bucket_minutes,
        )

    if isinstance(
        request,
        (
            DifferenceFeatureRequest,
            PercentageChangeFeatureRequest,
            SeasonalDifferenceFeatureRequest,
        ),
    ):
        return generate_comparison_features(
            request,
            sources=sources,
            bucket_minutes=bucket_minutes,
        )

    if isinstance(
        request,
        (
            RollingFeatureRequest,
            RollingZScoreFeatureRequest,
            EWMAFeatureRequest,
            RollingSlopeFeatureRequest,
        ),
    ):
        return generate_rolling_features(
            request,
            sources=sources,
        )

    if isinstance(
        request,
        (
            TimeBucketFeatureRequest,
            TimeCyclicFeatureRequest,
        ),
    ):
        return generate_time_features(
            request,
            site_dataframe=site_dataframe,
            timestamp_column=timestamp_column,
        )

    if isinstance(request, CalendarFeatureRequest):
        return generate_calendar_features(
            request,
            site_dataframe=site_dataframe,
            timestamp_column=timestamp_column,
        )

    if isinstance(request, BaselineFeatureRequest):
        return generate_baseline_features(
            request,
            sources=sources,
            site_dataframe=site_dataframe,
            timestamp_column=timestamp_column,
            bucket_minutes=bucket_minutes,
        )

    if isinstance(request, ResidualFeatureRequest):
        return generate_residual_features(
            request,
            sources=sources,
            site_dataframe=site_dataframe,
            timestamp_column=timestamp_column,
            bucket_minutes=bucket_minutes,
        )

    raise TypeError(
        "Unsupported feature request class: "
        f"{type(request).__name__}."
    )


def _generate_site_features(
    *,
    composite_id: str,
    site_dataframe: pd.DataFrame,
    sources: PreparedFeatureSources,
    resolved_config: ResolvedFeatureEngineeringConfig,
    timestamp_column: str,
) -> pd.DataFrame:
    """Generate every requested feature for one valid site in request order."""

    site_features = pd.DataFrame(index=site_dataframe.index)

    for request, resolved_columns in zip(
        resolved_config.features,
        resolved_config.feature_column_groups,
        strict=True,
    ):
        request_type = getattr(request, "type", type(request).__name__)

        try:
            generated = _dispatch_feature_request(
                request,
                site_dataframe=site_dataframe,
                sources=sources,
                timestamp_column=timestamp_column,
                bucket_minutes=resolved_config.bucket_minutes,
            )

            canonical_columns = get_feature_columns(request)
            normalized = normalize_feature_output(
                generated,
                expected_index=site_dataframe.index,
                context=(
                    f"feature '{request_type}' for site "
                    f"'{composite_id}'"
                ),
            )

            if tuple(normalized.columns) != canonical_columns:
                raise ValueError(
                    "Generator columns do not match the schema contract. "
                    f"Expected {list(canonical_columns)}, received "
                    f"{normalized.columns.tolist()}."
                )

            if len(resolved_columns) != len(canonical_columns):
                raise ValueError(
                    "Resolved feature-column mapping does not match the "
                    f"generator output count for feature '{request_type}'."
                )

            # Calculation modules always return canonical names. The schema
            # owns repetition naming, so rename only after generator-contract
            # validation and before attachment to the site feature matrix.
            normalized.columns = list(resolved_columns)

            site_features = append_feature_output(
                site_features,
                normalized,
                context=(
                    f"feature '{request_type}' for site "
                    f"'{composite_id}'"
                ),
            )

        except (TypeError, ValueError, RuntimeError) as error:
            raise InputValidationError(
                f"Feature '{request_type}' failed for site "
                f"'{composite_id}'.",
                error_code="SITE_FEATURE_GENERATION_FAILED",
                details={
                    "site": composite_id,
                    "feature_type": str(request_type),
                    "reason": str(error),
                },
            ) from error

    expected_all_columns = tuple(resolved_config.feature_columns)

    if tuple(site_features.columns) != expected_all_columns:
        raise InputValidationError(
            f"Site '{composite_id}' produced an unexpected final feature "
            "column sequence.",
            error_code="SITE_FEATURE_COLUMN_MISMATCH",
            details={
                "expected": list(expected_all_columns),
                "received": site_features.columns.tolist(),
            },
        )

    return site_features


def _build_corrected_history_audit(
    *,
    site_dataframe: pd.DataFrame,
    sources: PreparedFeatureSources,
    input_columns: FeatureInputColumns,
) -> pd.DataFrame | None:
    """Build one site-level corrected-history audit dataframe when available."""

    corrected_result = sources.corrected_history_result

    if corrected_result is None:
        return None

    audit = site_dataframe.loc[
        :,
        [
            *input_columns.site_identity_columns,
            input_columns.timestamp_column,
        ],
    ].copy()

    audit["raw_value"] = sources.bundle.raw_value
    audit["corrected_history"] = corrected_result.values
    audit["corrected_history_unavailable"] = (
        corrected_result.unavailable_mask
    )
    audit["corrected_history_trailing_median_replaced"] = (
        corrected_result.trailing_median_replaced_mask
    )
    audit["corrected_history_forward_filled"] = (
        corrected_result.forward_filled_mask
    )
    audit["corrected_history_unresolved"] = (
        corrected_result.unresolved_mask
    )

    return audit


def _build_site_run_summary(
    *,
    validation: SiteFeatureValidationResult,
    site_features: pd.DataFrame | None,
    sources: PreparedFeatureSources | None,
    feature_column_count: int,
) -> FeatureEngineeringSiteRunSummary:
    """Build one immutable site summary from validation and generation data."""

    if site_features is None:
        populated_cells = 0
        missing_cells = validation.row_count * feature_column_count
    else:
        populated_cells = int(site_features.notna().sum().sum())
        missing_cells = int(site_features.isna().sum().sum())

    corrected = (
        sources.corrected_history_result
        if sources is not None
        else None
    )

    return FeatureEngineeringSiteRunSummary(
        composite_id=validation.composite_id,
        row_count=validation.row_count,
        usability=validation.usability,
        minimum_history_points=validation.minimum_history_points,
        has_sufficient_history=validation.has_sufficient_history,
        can_generate_features=validation.can_generate_features,
        skip_reason=validation.skip_reason,
        generated_feature_columns=(
            feature_column_count
            if validation.can_generate_features
            else 0
        ),
        populated_feature_cells=populated_cells,
        missing_feature_cells=missing_cells,
        corrected_history_unavailable_points=(
            corrected.unavailable_count
            if corrected is not None
            else 0
        ),
        corrected_history_trailing_median_replacements=(
            corrected.trailing_median_replaced_count
            if corrected is not None
            else 0
        ),
        corrected_history_forward_fills=(
            corrected.forward_filled_count
            if corrected is not None
            else 0
        ),
        corrected_history_unresolved_points=(
            corrected.unresolved_count
            if corrected is not None
            else 0
        ),
        minimum_timestamp=validation.minimum_timestamp,
        maximum_timestamp=validation.maximum_timestamp,
    )


def _site_summaries_to_dataframe(
    summaries: tuple[FeatureEngineeringSiteRunSummary, ...],
) -> pd.DataFrame:
    """Convert structured site summaries to one stable tabular report."""

    return pd.DataFrame(
        [asdict(summary) for summary in summaries]
    )


def _restore_external_index(
    dataframe: pd.DataFrame,
    *,
    original_index: pd.Index,
) -> pd.DataFrame:
    """Restore the source dataframe index after unique internal row tracking."""

    if len(dataframe) != len(original_index):
        raise RuntimeError(
            "Cannot restore the original index because the row count changed."
        )

    restored = dataframe.copy()
    restored.index = original_index.copy()
    return restored


def run_feature_engineering_pipeline(
    *,
    preprocessing_result: PreprocessingRunnerResult,
    output_directory: str | Path,
    dataset_config: DatasetInputConfig,
    feature_config: FeatureEngineeringConfig,
    bucket: BucketConfig,
    logger: Logger,
    usability_column: str = DEFAULT_USABILITY_COLUMN,
    status_column: str = DEFAULT_STATUS_COLUMN,
) -> FeatureEngineeringRunnerResult:
    """Run complete Part C feature engineering for one preprocessed KPI.

    The workflow should call this function once per
    ``PreprocessingRunnerResult`` and pass the same resolved ``BucketConfig``
    used by Part B.

    Args:
        preprocessing_result:
            Exact structured Part B result for one KPI.
        output_directory:
            Exact run-owned Part C directory created by the workflow.
        dataset_config:
            Shared validated dataset contract containing physical column names.
        feature_config:
            Validated Part C request. It may contain a custom ``features`` list
            parsed from JSON or select a schema-owned default profile.
        bucket:
            Uniform resolved bucket returned by Part A and used by Part B.
        logger:
            Run-specific logger shared by the workflow and ML stages.
        usability_column:
            Fixed Part B site-usability output column.
        status_column:
            Fixed Part B current-status output column.

    Returns:
        FeatureEngineeringRunnerResult:
            Final output paths, resolved configuration, validation report, and
            site-level headline statistics.

    Raises:
        FileProcessingError:
            If required Part B or output files cannot be processed.
        InputValidationError:
            If configuration, handoff, dataframe, feature generation, or final
            cross-module validation fails.
    """

    if not isinstance(logger, Logger):
        raise InputValidationError(
            "logger must be an instance of logging.Logger.",
            error_code="INVALID_FEATURE_LOGGER",
            details={"received_type": type(logger).__name__},
        )

    resolved_usability_column = _validate_nonblank_column_name(
        usability_column,
        field_name="usability_column",
    )
    resolved_status_column = _validate_nonblank_column_name(
        status_column,
        field_name="status_column",
    )

    source_file = validate_preprocessing_handoff(preprocessing_result)
    resolved_output_directory = ensure_directory(
        Path(output_directory).expanduser().resolve()
    )
    summary_directory = ensure_directory(
        resolved_output_directory / SUMMARY_DIRECTORY_NAME
    )

    input_columns, resolved_config = _resolve_and_validate_config(
        dataset_config=dataset_config,
        feature_config=feature_config,
        bucket=bucket,
        preprocessing_result=preprocessing_result,
    )

    intermediate_directory: Path | None = None
    if resolved_config.save_intermediate_outputs:
        intermediate_directory = ensure_directory(
            resolved_output_directory / INTERMEDIATE_DIRECTORY_NAME
        )

    kpi_name = _resolve_kpi_name(
        preprocessing_result,
        source_file=source_file,
    )
    safe_kpi_name = sanitise_filename(kpi_name)
    if not safe_kpi_name:
        safe_kpi_name = sanitise_filename(source_file.stem)
    if not safe_kpi_name:
        raise InputValidationError(
            "A safe output filename could not be derived for Part C.",
            error_code="INVALID_FEATURE_OUTPUT_NAME",
        )

    logger.info(
        "Starting Part C feature engineering for KPI '%s'. Source: %s. "
        "Bucket: %d minute(s). Features: %s.",
        kpi_name,
        source_file,
        resolved_config.bucket_minutes,
        list(resolved_config.feature_columns),
    )

    try:
        source_dataframe = read_csv_file(source_file)
    except Exception as error:
        raise FileProcessingError(
            "The final Part B dataset could not be read for feature "
            "engineering.",
            error_code="FEATURE_SOURCE_READ_FAILED",
            details={
                "source_file": str(source_file),
                "reason": str(error),
            },
        ) from error

    expected_part_b_rows = getattr(
        preprocessing_result,
        "final_rows",
        None,
    )
    if (
        expected_part_b_rows is not None
        and int(expected_part_b_rows) != len(source_dataframe)
    ):
        raise InputValidationError(
            "The Part B result row count does not match its final dataset.",
            error_code="FEATURE_SOURCE_ROW_COUNT_MISMATCH",
            details={
                "part_b_reported_rows": int(expected_part_b_rows),
                "source_file_rows": len(source_dataframe),
            },
        )

    try:
        normalized_dataframe, validation_result = (
            prepare_and_validate_feature_dataframe(
                source_dataframe,
                input_columns=input_columns,
                resolved_config=resolved_config,
                usability_column=resolved_usability_column,
                status_column=resolved_status_column,
            )
        )
    except (TypeError, ValueError) as error:
        raise InputValidationError(
            "The Part B dataframe failed Part C input validation.",
            error_code="FEATURE_INPUT_VALIDATION_FAILED",
            details={"reason": str(error)},
        ) from error

    original_index = normalized_dataframe.index.copy()
    working_dataframe = normalized_dataframe.copy()
    working_dataframe.index = pd.RangeIndex(
        start=0,
        stop=len(working_dataframe),
        step=1,
        name="_feature_row_id",
    )

    feature_matrix = pd.DataFrame(
        np.nan,
        index=working_dataframe.index,
        columns=list(resolved_config.feature_columns),
        dtype="float64",
    )

    validation_by_site = {
        result.composite_id: result
        for result in validation_result.site_results
    }

    site_run_summaries: list[FeatureEngineeringSiteRunSummary] = []
    corrected_history_audits: list[pd.DataFrame] = []

    for site_group in iter_site_groups(
        working_dataframe,
        site_identity_columns=input_columns.site_identity_columns,
        timestamp_column=input_columns.timestamp_column,
    ):
        site_validation = validation_by_site.get(site_group.composite_id)

        if site_validation is None:
            raise InputValidationError(
                "A site produced by grouping is missing from the validation "
                "result.",
                error_code="FEATURE_SITE_VALIDATION_LOOKUP_FAILED",
                details={"site": site_group.composite_id},
            )

        if not site_validation.can_generate_features:
            site_run_summaries.append(
                _build_site_run_summary(
                    validation=site_validation,
                    site_features=None,
                    sources=None,
                    feature_column_count=len(
                        resolved_config.feature_columns
                    ),
                )
            )
            logger.info(
                "Skipping Part C feature generation for invalid site '%s'.",
                site_group.composite_id,
            )
            continue

        try:
            sources = prepare_feature_sources(
                site_group.dataframe,
                value_column=input_columns.value_column,
                resolved_config=resolved_config,
                status_column=resolved_status_column,
            )

            site_features = _generate_site_features(
                composite_id=site_group.composite_id,
                site_dataframe=site_group.dataframe,
                sources=sources,
                resolved_config=resolved_config,
                timestamp_column=input_columns.timestamp_column,
            )

        except InputValidationError:
            raise
        except (TypeError, ValueError, RuntimeError) as error:
            raise InputValidationError(
                f"Source preparation or feature generation failed for site "
                f"'{site_group.composite_id}'.",
                error_code="FEATURE_SITE_PROCESSING_FAILED",
                details={
                    "site": site_group.composite_id,
                    "reason": str(error),
                },
            ) from error

        feature_matrix.loc[
            site_group.dataframe.index,
            list(site_features.columns),
        ] = site_features.to_numpy(dtype="float64")

        audit = _build_corrected_history_audit(
            site_dataframe=site_group.dataframe,
            sources=sources,
            input_columns=input_columns,
        )
        if audit is not None:
            corrected_history_audits.append(audit)

        site_run_summaries.append(
            _build_site_run_summary(
                validation=site_validation,
                site_features=site_features,
                sources=sources,
                feature_column_count=len(
                    resolved_config.feature_columns
                ),
            )
        )

    feature_matrix = feature_matrix.loc[
        :,
        list(resolved_config.feature_columns),
    ]

    if resolved_config.retain_original_columns:
        final_working_dataframe = pd.concat(
            [working_dataframe, feature_matrix],
            axis=1,
            copy=False,
        )
    else:
        final_working_dataframe = feature_matrix.copy()

    final_dataframe = _restore_external_index(
        final_working_dataframe,
        original_index=original_index,
    )
    normalized_external = _restore_external_index(
        working_dataframe,
        original_index=original_index,
    )

    try:
        validate_generated_feature_dataframe(
            source_dataframe=normalized_external,
            feature_dataframe=final_dataframe,
            input_columns=input_columns,
            resolved_config=resolved_config,
            usability_column=resolved_usability_column,
        )
    except (TypeError, ValueError) as error:
        raise InputValidationError(
            "The completed feature-engineered dataframe failed final "
            "validation.",
            error_code="FEATURE_FINAL_VALIDATION_FAILED",
            details={"reason": str(error)},
        ) from error

    site_summaries_tuple = tuple(site_run_summaries)
    site_summary_dataframe = _site_summaries_to_dataframe(
        site_summaries_tuple
    )

    final_dataset_file = save_named_dataframe(
        dataframe=final_dataframe,
        output_directory=resolved_output_directory,
        filename=f"{safe_kpi_name}{FINAL_DATASET_SUFFIX}",
    )
    site_summary_file = save_named_dataframe(
        dataframe=site_summary_dataframe,
        output_directory=summary_directory,
        filename=SITE_SUMMARY_FILE_NAME,
    )

    corrected_history_audit_file: Path | None = None
    if (
        resolved_config.save_intermediate_outputs
        and resolved_config.uses_corrected_history
    ):
        if intermediate_directory is None:
            raise InputValidationError(
                "Intermediate feature outputs are enabled but no directory "
                "was created.",
                error_code="MISSING_FEATURE_INTERMEDIATE_DIRECTORY",
            )

        audit_columns = [
            *input_columns.site_identity_columns,
            input_columns.timestamp_column,
            "raw_value",
            "corrected_history",
            "corrected_history_unavailable",
            "corrected_history_trailing_median_replaced",
            "corrected_history_forward_filled",
            "corrected_history_unresolved",
        ]

        if corrected_history_audits:
            corrected_history_audit = pd.concat(
                corrected_history_audits,
                axis=0,
                ignore_index=True,
                copy=False,
            )
            corrected_history_audit = corrected_history_audit.loc[
                :,
                audit_columns,
            ]
        else:
            corrected_history_audit = pd.DataFrame(
                columns=audit_columns
            )

        corrected_history_audit_file = save_named_dataframe(
            dataframe=corrected_history_audit,
            output_directory=intermediate_directory,
            filename=CORRECTED_HISTORY_AUDIT_FILE_NAME,
        )

    populated_feature_cells = int(
        final_dataframe.loc[
            :,
            list(resolved_config.feature_columns),
        ].notna().sum().sum()
    )
    total_feature_cells = (
        len(final_dataframe)
        * len(resolved_config.feature_columns)
    )
    missing_feature_cells = (
        total_feature_cells - populated_feature_cells
    )

    result = FeatureEngineeringRunnerResult(
        kpi_name=kpi_name,
        source_file=source_file,
        output_directory=resolved_output_directory,
        summary_directory=summary_directory,
        intermediate_directory=intermediate_directory,
        resolved_config=resolved_config,
        validation_result=validation_result,
        final_dataset_file=final_dataset_file,
        site_summary_file=site_summary_file,
        corrected_history_audit_file=corrected_history_audit_file,
        feature_columns=tuple(resolved_config.feature_columns),
        total_rows=len(final_dataframe),
        total_sites=validation_result.total_sites,
        processed_sites=validation_result.processable_sites,
        skipped_invalid_sites=validation_result.invalid_sites,
        insufficient_history_sites=(
            validation_result.insufficient_history_sites
        ),
        populated_feature_cells=populated_feature_cells,
        missing_feature_cells=missing_feature_cells,
        site_summaries=site_summaries_tuple,
    )

    logger.info(
        "Part C feature engineering completed for KPI '%s'. Final dataset: "
        "%s. Sites: processed=%d, invalid_skipped=%d, "
        "insufficient_history=%d. Feature columns=%d.",
        result.kpi_name,
        result.final_dataset_file,
        result.processed_sites,
        result.skipped_invalid_sites,
        result.insufficient_history_sites,
        len(result.feature_columns),
    )

    return result


__all__ = [
    "CORRECTED_HISTORY_AUDIT_FILE_NAME",
    "FINAL_DATASET_SUFFIX",
    "FeatureEngineeringRunnerResult",
    "FeatureEngineeringSiteRunSummary",
    "SITE_SUMMARY_FILE_NAME",
    "run_feature_engineering_pipeline",
    "save_named_dataframe",
    "validate_preprocessing_handoff",
]
