"""
Inference-specific Part B preprocessing runner.

Intended project location::

    app/ml/preprocessing/runner_inference.py

The normal ``app.ml.preprocessing.runner`` remains unchanged and authoritative
for training. This module reuses the same Part B stage implementations and the
same ``PreprocessingRunnerResult`` class, but applies one inference-only rule:

After timeline correction calculates the current-window missing percentage and
provisional ``kpi_usability``, the saved processed cache is consulted. For every
site already present in that cache, the cached ``kpi_usability`` and
``kpi_invalid_reason`` are restored across all rows BEFORE outlier labelling and
missing-value filling begin.

This prevents a shorter inference working window from changing the usability
classification established in the saved processed history. ``missing_percentage``
is intentionally left as the freshly calculated runtime diagnostic; only the
site-level usability decision and its reason are enforced.
"""

from __future__ import annotations

from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from app.core.exceptions import InputValidationError
from app.ml.eda.runner import KPIEDARunResult
from app.schemas.common.dataset import (
    BucketConfig,
    DatasetColumnMapping,
    DatasetInputConfig,
)
from app.schemas.preprocessing import PreprocessingConfig
from app.services.file_service import (
    ensure_directory,
    read_csv_file,
    sanitise_filename,
)

from .missing_gap_analysis import (
    MissingGapAnalysisData,
    analyse_missing_gaps,
)
from .missing_value_phase_1 import (
    Phase1FillingData,
    fill_phase_1_missing_values,
)
from .missing_value_phase_2 import (
    Phase2FillingData,
    fill_phase_2_missing_values,
)
from .missing_value_phase_3 import (
    Phase3FillingData,
    fill_phase_3_missing_values,
)
from .missing_value_phase_4 import (
    Phase4FillingData,
    fill_phase_4_missing_values,
)
from .outlier_labelling import (
    OutlierLabellingData,
    label_outliers,
)
from .runner import (
    FINAL_DATASET_SUFFIX,
    GAP_CHECKPOINT_FILE_NAME,
    GAP_SUMMARY_FILE_NAME,
    INTERMEDIATE_DIRECTORY_NAME,
    OUTLIER_CHECKPOINT_FILE_NAME,
    OUTLIER_SUMMARY_FILE_NAME,
    PHASE_1_CHECKPOINT_FILE_NAME,
    PHASE_1_SUMMARY_FILE_NAME,
    PHASE_2_CHECKPOINT_FILE_NAME,
    PHASE_2_SUMMARY_FILE_NAME,
    PHASE_3_CHECKPOINT_FILE_NAME,
    PHASE_3_SUMMARY_FILE_NAME,
    PHASE_4_SUMMARY_FILE_NAME,
    SUMMARY_DIRECTORY_NAME,
    TIMELINE_CHECKPOINT_FILE_NAME,
    PreprocessingRunnerResult,
    count_unresolved_points,
    resolve_kpi_name,
    save_named_dataframe,
    save_optional_checkpoint,
    validate_final_runner_output,
    validate_part_a_handoff,
)
from .timeline_correction import (
    INVALID_USABILITY_VALUE,
    KPI_INVALID_REASON_COLUMN,
    KPI_USABILITY_COLUMN,
    VALID_USABILITY_VALUE,
    create_timeline_corrected_dataset,
)


SeriesIdentity = tuple[str, str, str]
_ALLOWED_USABILITY_VALUES = {
    VALID_USABILITY_VALUE,
    INVALID_USABILITY_VALUE,
}


@dataclass(frozen=True, slots=True)
class CachedUsabilityEnforcementResult:
    """Result of restoring saved per-site usability onto the runtime timeline."""

    dataframe: pd.DataFrame
    cached_series: int
    matched_series: int
    uncached_runtime_series: int
    changed_series: int
    valid_series: int
    invalid_series: int


def _normalise_identity_column(
    series: pd.Series,
    *,
    column: str,
    role: str,
) -> pd.Series:
    """Normalize one identity column and reject missing/blank identifiers."""

    text = series.astype("string")
    invalid = series.isna() | text.str.strip().eq("")

    if bool(invalid.any()):
        raise InputValidationError(
            f"{role} contains missing or blank site identity values.",
            error_code="INVALID_INFERENCE_USABILITY_SITE_IDENTITY",
            details={
                "column": column,
                "invalid_rows": int(invalid.sum()),
            },
        )

    return text.str.strip()


def _cached_usability_map(
    *,
    saved_processed_cache: pd.DataFrame,
    columns: DatasetColumnMapping,
    kpi_name: str,
) -> dict[SeriesIdentity, tuple[str, str]]:
    """Resolve one authoritative usability/reason pair per cached site."""

    if not isinstance(saved_processed_cache, pd.DataFrame):
        raise InputValidationError(
            "saved_processed_cache must be a pandas DataFrame.",
            error_code="INVALID_INFERENCE_USABILITY_CACHE_TYPE",
            details={"kpi_name": kpi_name},
        )

    if saved_processed_cache.empty:
        raise InputValidationError(
            "The saved processed cache is empty and cannot provide usability.",
            error_code="EMPTY_INFERENCE_USABILITY_CACHE",
            details={"kpi_name": kpi_name},
        )

    identity_columns = list(columns.get_site_identity_columns())
    if len(identity_columns) != 3:
        raise InputValidationError(
            "Inference usability enforcement requires exactly three site "
            "identity columns.",
            error_code="INVALID_INFERENCE_USABILITY_IDENTITY_WIDTH",
            details={
                "kpi_name": kpi_name,
                "identity_columns": identity_columns,
            },
        )

    required_columns = [
        *identity_columns,
        KPI_USABILITY_COLUMN,
        KPI_INVALID_REASON_COLUMN,
    ]
    missing_columns = [
        column
        for column in required_columns
        if column not in saved_processed_cache.columns
    ]

    if missing_columns:
        raise InputValidationError(
            "The saved processed cache is missing usability columns required "
            "by inference-specific Part B.",
            error_code="MISSING_INFERENCE_USABILITY_CACHE_COLUMNS",
            details={
                "kpi_name": kpi_name,
                "missing_columns": missing_columns,
                "available_columns": list(saved_processed_cache.columns),
            },
        )

    working = saved_processed_cache[required_columns].copy()

    for column in identity_columns:
        working[column] = _normalise_identity_column(
            working[column],
            column=column,
            role=f"saved processed cache for KPI {kpi_name!r}",
        )

    working[KPI_USABILITY_COLUMN] = (
        working[KPI_USABILITY_COLUMN]
        .astype("string")
        .str.strip()
        .str.casefold()
    )
    invalid_usability = (
        working[KPI_USABILITY_COLUMN].isna()
        | ~working[KPI_USABILITY_COLUMN].isin(_ALLOWED_USABILITY_VALUES)
    )

    if bool(invalid_usability.any()):
        raise InputValidationError(
            "The saved processed cache contains unsupported kpi_usability "
            "values.",
            error_code="INVALID_INFERENCE_USABILITY_CACHE_VALUES",
            details={
                "kpi_name": kpi_name,
                "invalid_rows": int(invalid_usability.sum()),
            },
        )

    working[KPI_INVALID_REASON_COLUMN] = (
        working[KPI_INVALID_REASON_COLUMN]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    result: dict[SeriesIdentity, tuple[str, str]] = {}
    grouped = working.groupby(
        identity_columns,
        sort=False,
        dropna=False,
        observed=True,
    )

    for raw_identity, group in grouped:
        identity_values = (
            raw_identity
            if isinstance(raw_identity, tuple)
            else (raw_identity,)
        )

        if len(identity_values) != 3:
            raise InputValidationError(
                "Inference usability requires exactly three site identity "
                "columns.",
                error_code="INVALID_INFERENCE_USABILITY_IDENTITY_WIDTH",
                details={
                    "kpi_name": kpi_name,
                    "identity_columns": identity_columns,
                },
            )

        identity: SeriesIdentity = (
            str(identity_values[0]).strip(),
            str(identity_values[1]).strip(),
            str(identity_values[2]).strip(),
        )

        usability_values = (
            group[KPI_USABILITY_COLUMN]
            .drop_duplicates()
            .tolist()
        )

        if len(usability_values) != 1:
            raise InputValidationError(
                "The saved processed cache contains inconsistent usability "
                "for one site.",
                error_code="INCONSISTENT_INFERENCE_USABILITY_CACHE_SITE",
                details={
                    "kpi_name": kpi_name,
                    "identity": identity,
                    "values": [str(value) for value in usability_values],
                },
            )

        usability = str(usability_values[0])
        nonblank_reasons = (
            group.loc[
                group[KPI_INVALID_REASON_COLUMN].ne(""),
                KPI_INVALID_REASON_COLUMN,
            ]
            .drop_duplicates()
            .tolist()
        )

        if len(nonblank_reasons) > 1:
            raise InputValidationError(
                "The saved processed cache contains inconsistent invalid "
                "reasons for one site.",
                error_code="INCONSISTENT_INFERENCE_USABILITY_CACHE_REASON",
                details={
                    "kpi_name": kpi_name,
                    "identity": identity,
                    "reasons": [str(value) for value in nonblank_reasons],
                },
            )

        reason = (
            ""
            if usability == VALID_USABILITY_VALUE
            else (
                str(nonblank_reasons[0])
                if nonblank_reasons
                else ""
            )
        )
        result[identity] = (usability, reason)

    return result


def enforce_cached_series_usability(
    *,
    timeline_dataframe: pd.DataFrame,
    saved_processed_cache: pd.DataFrame,
    columns: DatasetColumnMapping,
    kpi_name: str,
    logger: Logger,
) -> CachedUsabilityEnforcementResult:
    """
    Restore cached site usability before any inference-time filling occurs.

    Sites present in the saved processed cache are authoritative. Runtime sites
    absent from the cache retain the provisional usability calculated by the
    normal timeline-correction stage.
    """

    if not isinstance(timeline_dataframe, pd.DataFrame) or timeline_dataframe.empty:
        raise InputValidationError(
            "The inference timeline dataframe is empty or invalid.",
            error_code="EMPTY_INFERENCE_USABILITY_TIMELINE",
            details={"kpi_name": kpi_name},
        )

    identity_columns = list(columns.get_site_identity_columns())
    if len(identity_columns) != 3:
        raise InputValidationError(
            "Inference usability enforcement requires exactly three site "
            "identity columns.",
            error_code="INVALID_INFERENCE_USABILITY_IDENTITY_WIDTH",
            details={
                "kpi_name": kpi_name,
                "identity_columns": identity_columns,
            },
        )

    required_columns = [
        *identity_columns,
        KPI_USABILITY_COLUMN,
        KPI_INVALID_REASON_COLUMN,
    ]
    missing_columns = [
        column
        for column in required_columns
        if column not in timeline_dataframe.columns
    ]

    if missing_columns:
        raise InputValidationError(
            "The runtime timeline is missing columns required to enforce "
            "cached usability.",
            error_code="MISSING_INFERENCE_USABILITY_TIMELINE_COLUMNS",
            details={
                "kpi_name": kpi_name,
                "missing_columns": missing_columns,
            },
        )

    cached = _cached_usability_map(
        saved_processed_cache=saved_processed_cache,
        columns=columns,
        kpi_name=kpi_name,
    )

    result = timeline_dataframe.copy()

    # CSV reloads can infer an all-empty invalid-reason column as float64.
    # Convert to string before filling/assigning textual reasons so pandas
    # 3.x does not reject values such as "" as incompatible with float64.
    result[KPI_INVALID_REASON_COLUMN] = (
        result[KPI_INVALID_REASON_COLUMN]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    normalized_identity_columns: list[pd.Series] = []

    for column in identity_columns:
        normalized_identity_columns.append(
            _normalise_identity_column(
                result[column],
                column=column,
                role=f"runtime Part B timeline for KPI {kpi_name!r}",
            )
        )

    runtime_identities = pd.Series(
        list(
            zip(
                normalized_identity_columns[0],
                normalized_identity_columns[1],
                normalized_identity_columns[2],
            )
        ),
        index=result.index,
        dtype="object",
    )

    runtime_series = set(runtime_identities.tolist())
    matched_series = runtime_series.intersection(cached)
    changed_series = 0

    for identity in matched_series:
        authoritative_usability, authoritative_reason = cached[identity]
        mask = runtime_identities.map(
            lambda value, expected=identity: value == expected
        )

        current_values = (
            result.loc[mask, KPI_USABILITY_COLUMN]
            .astype("string")
            .str.strip()
            .str.casefold()
            .drop_duplicates()
            .tolist()
        )
        current_reasons = (
            result.loc[mask, KPI_INVALID_REASON_COLUMN]
            .astype("string")
            .fillna("")
            .str.strip()
            .drop_duplicates()
            .tolist()
        )

        if (
            current_values != [authoritative_usability]
            or current_reasons != [authoritative_reason]
        ):
            changed_series += 1

        result.loc[mask, KPI_USABILITY_COLUMN] = authoritative_usability
        result.loc[mask, KPI_INVALID_REASON_COLUMN] = authoritative_reason

    final_usability = (
        result.assign(__runtime_identity=runtime_identities)
        .groupby(
            "__runtime_identity",
            sort=False,
            dropna=False,
        )[KPI_USABILITY_COLUMN]
        .agg(
            lambda values: tuple(
                values.astype("string")
                .str.strip()
                .str.casefold()
                .drop_duplicates()
                .tolist()
            )
        )
    )

    inconsistent = final_usability.loc[
        final_usability.map(len).ne(1)
    ]
    if not inconsistent.empty:
        raise InputValidationError(
            "Cached usability enforcement left inconsistent site usability.",
            error_code="INCONSISTENT_ENFORCED_INFERENCE_USABILITY",
            details={
                "kpi_name": kpi_name,
                "sample": {
                    str(identity): list(values)
                    for identity, values in inconsistent.head(20).items()
                },
            },
        )

    final_values = final_usability.map(lambda values: values[0])
    invalid_final = ~final_values.isin(_ALLOWED_USABILITY_VALUES)
    if bool(invalid_final.any()):
        raise InputValidationError(
            "Cached usability enforcement produced unsupported values.",
            error_code="INVALID_ENFORCED_INFERENCE_USABILITY",
            details={"kpi_name": kpi_name},
        )

    valid_series = int(final_values.eq(VALID_USABILITY_VALUE).sum())
    invalid_series = int(final_values.eq(INVALID_USABILITY_VALUE).sum())

    logger.info(
        "Inference usability enforced for KPI '%s': cached_series=%d, "
        "matched_runtime_series=%d, changed_series=%d, "
        "uncached_runtime_series=%d, final_valid=%d, final_invalid=%d.",
        kpi_name,
        len(cached),
        len(matched_series),
        changed_series,
        len(runtime_series - set(cached)),
        valid_series,
        invalid_series,
    )

    return CachedUsabilityEnforcementResult(
        dataframe=result,
        cached_series=len(cached),
        matched_series=len(matched_series),
        uncached_runtime_series=len(runtime_series - set(cached)),
        changed_series=changed_series,
        valid_series=valid_series,
        invalid_series=invalid_series,
    )


def run_preprocessing_pipeline_for_inference(
    *,
    kpi_result: KPIEDARunResult,
    output_directory: str | Path,
    dataset_config: DatasetInputConfig,
    preprocessing_config: PreprocessingConfig,
    saved_processed_cache: pd.DataFrame,
    bucket: BucketConfig,
    logger: Logger,
) -> PreprocessingRunnerResult:
    """
    Run Part B for inference while preserving cached site usability.

    The workflow should call this function once per
    ``eda_result.kpi_results`` entry and pass
    ``eda_result.preparation.bucket`` as ``bucket``.

    Args:
        kpi_result:
            Exact structured Part A result for one prepared KPI dataset.

        output_directory:
            Exact run-owned KPI processing directory created by the workflow.

        dataset_config:
            Shared validated dataset contract. The runner uses its physical
            column mapping; source paths continue to come from ``kpi_result``.

        preprocessing_config:
            Complete validated Part B request settings.

        saved_processed_cache:
            The already-loaded saved processed cache for this KPI. Its
            per-site kpi_usability and kpi_invalid_reason values are treated
            as authoritative during inference.

        bucket:
            Uniform resolved bucket returned by Part A preparation.

        logger:
            Run-specific logger created by the workflow and shared with every
            preprocessing stage.

    Returns:
        PreprocessingRunnerResult:
            Final output paths, resolved settings, and headline statistics.

    Raises:
        FileProcessingError:
            If required Part A inputs, Prophet, or output files cannot be
            processed.

        InputValidationError:
            If Part A handoff, configuration resolution, stage validation, or
            final cross-stage invariants fail.
    """

    source_file, site_summary_file = (
        validate_part_a_handoff(
            kpi_result
        )
    )

    resolved_output_directory = (
        ensure_directory(
            Path(output_directory)
            .expanduser()
            .resolve()
        )
    )
    summary_directory = ensure_directory(
        resolved_output_directory
        / SUMMARY_DIRECTORY_NAME
    )

    intermediate_directory: Path | None = None

    if preprocessing_config.save_intermediate_outputs:
        intermediate_directory = ensure_directory(
            resolved_output_directory
            / INTERMEDIATE_DIRECTORY_NAME
        )

    resolved_frequency = (
        preprocessing_config.resolve_for_bucket(
            bucket
        )
    )
    kpi_name = resolve_kpi_name(
        kpi_result
    )
    safe_kpi_name = sanitise_filename(
        kpi_name
    )

    if not safe_kpi_name:
        safe_kpi_name = sanitise_filename(
            source_file.stem
        )

    logger.info(
        "Starting inference-specific Part B preprocessing for KPI '%s'. Source: %s. "
        "Bucket: %s minute(s). Prophet enabled: %s.",
        kpi_name,
        source_file,
        resolved_frequency.bucket_minutes,
        preprocessing_config.phase_4.enabled,
    )

    with TemporaryDirectory(
        prefix=".preprocessing_work_",
        dir=resolved_output_directory,
    ) as temporary_directory_name:
        temporary_directory = Path(
            temporary_directory_name
        )

        timeline_result = (
            create_timeline_corrected_dataset(
                source_file=source_file,
                site_summary_file=(
                    site_summary_file
                ),
                data_integrity=(
                    kpi_result.data_integrity
                ),
                output_directory=(
                    temporary_directory
                ),
                columns=dataset_config.columns,
                bucket=bucket,
                config=preprocessing_config,
                logger=logger,
            )
        )

        timeline_dataframe = read_csv_file(
            timeline_result.output_file
        )

        usability_enforcement = enforce_cached_series_usability(
            timeline_dataframe=timeline_dataframe,
            saved_processed_cache=saved_processed_cache,
            columns=dataset_config.columns,
            kpi_name=kpi_name,
            logger=logger,
        )
        timeline_dataframe = usability_enforcement.dataframe

        timeline_checkpoint_file = (
            save_optional_checkpoint(
                enabled=(
                    preprocessing_config
                    .save_intermediate_outputs
                ),
                dataframe=timeline_dataframe,
                intermediate_directory=(
                    intermediate_directory
                ),
                filename=(
                    TIMELINE_CHECKPOINT_FILE_NAME
                ),
            )
        )

        outlier_data: OutlierLabellingData = (
            label_outliers(
                dataframe=timeline_dataframe,
                columns=dataset_config.columns,
                config=preprocessing_config,
                resolved_frequency=(
                    resolved_frequency
                ),
                logger=logger,
            )
        )

        outlier_summary_file = (
            save_named_dataframe(
                dataframe=outlier_data.summary,
                output_directory=(
                    summary_directory
                ),
                filename=(
                    OUTLIER_SUMMARY_FILE_NAME
                ),
            )
        )
        outlier_checkpoint_file = (
            save_optional_checkpoint(
                enabled=(
                    preprocessing_config
                    .save_intermediate_outputs
                ),
                dataframe=(
                    outlier_data.dataframe
                ),
                intermediate_directory=(
                    intermediate_directory
                ),
                filename=(
                    OUTLIER_CHECKPOINT_FILE_NAME
                ),
            )
        )

        gap_data: MissingGapAnalysisData = (
            analyse_missing_gaps(
                dataframe=(
                    outlier_data.dataframe
                ),
                columns=dataset_config.columns,
                resolved_frequency=(
                    resolved_frequency
                ),
                logger=logger,
            )
        )

        gap_summary_file = save_named_dataframe(
            dataframe=gap_data.gap_summary,
            output_directory=summary_directory,
            filename=GAP_SUMMARY_FILE_NAME,
        )
        gap_checkpoint_file = (
            save_optional_checkpoint(
                enabled=(
                    preprocessing_config
                    .save_intermediate_outputs
                ),
                dataframe=gap_data.dataframe,
                intermediate_directory=(
                    intermediate_directory
                ),
                filename=(
                    GAP_CHECKPOINT_FILE_NAME
                ),
            )
        )

        phase_1_data: Phase1FillingData = (
            fill_phase_1_missing_values(
                dataframe=gap_data.dataframe,
                columns=dataset_config.columns,
                config=preprocessing_config,
                resolved_frequency=(
                    resolved_frequency
                ),
                logger=logger,
            )
        )

        phase_1_summary_file = (
            save_named_dataframe(
                dataframe=phase_1_data.summary,
                output_directory=(
                    summary_directory
                ),
                filename=(
                    PHASE_1_SUMMARY_FILE_NAME
                ),
            )
        )
        phase_1_checkpoint_file = (
            save_optional_checkpoint(
                enabled=(
                    preprocessing_config
                    .save_intermediate_outputs
                ),
                dataframe=(
                    phase_1_data.dataframe
                ),
                intermediate_directory=(
                    intermediate_directory
                ),
                filename=(
                    PHASE_1_CHECKPOINT_FILE_NAME
                ),
            )
        )

        phase_2_data: Phase2FillingData = (
            fill_phase_2_missing_values(
                dataframe=(
                    phase_1_data.dataframe
                ),
                columns=dataset_config.columns,
                config=preprocessing_config,
                resolved_frequency=(
                    resolved_frequency
                ),
                logger=logger,
            )
        )

        phase_2_summary_file = (
            save_named_dataframe(
                dataframe=phase_2_data.summary,
                output_directory=(
                    summary_directory
                ),
                filename=(
                    PHASE_2_SUMMARY_FILE_NAME
                ),
            )
        )
        phase_2_checkpoint_file = (
            save_optional_checkpoint(
                enabled=(
                    preprocessing_config
                    .save_intermediate_outputs
                ),
                dataframe=(
                    phase_2_data.dataframe
                ),
                intermediate_directory=(
                    intermediate_directory
                ),
                filename=(
                    PHASE_2_CHECKPOINT_FILE_NAME
                ),
            )
        )

        phase_3_data: Phase3FillingData = (
            fill_phase_3_missing_values(
                dataframe=(
                    phase_2_data.dataframe
                ),
                columns=dataset_config.columns,
                config=preprocessing_config,
                resolved_frequency=(
                    resolved_frequency
                ),
                logger=logger,
            )
        )

        phase_3_summary_file = (
            save_named_dataframe(
                dataframe=phase_3_data.summary,
                output_directory=(
                    summary_directory
                ),
                filename=(
                    PHASE_3_SUMMARY_FILE_NAME
                ),
            )
        )
        phase_3_checkpoint_file = (
            save_optional_checkpoint(
                enabled=(
                    preprocessing_config
                    .save_intermediate_outputs
                ),
                dataframe=(
                    phase_3_data.dataframe
                ),
                intermediate_directory=(
                    intermediate_directory
                ),
                filename=(
                    PHASE_3_CHECKPOINT_FILE_NAME
                ),
            )
        )

        phase_4_data: Phase4FillingData | None = None
        phase_4_summary_file: Path | None = None

        if preprocessing_config.phase_4.enabled:
            phase_4_data = (
                fill_phase_4_missing_values(
                    dataframe=(
                        phase_3_data.dataframe
                    ),
                    columns=dataset_config.columns,
                    config=preprocessing_config,
                    resolved_frequency=(
                        resolved_frequency
                    ),
                    logger=logger,
                    dataset_identifier=(
                        safe_kpi_name
                    ),
                )
            )

            phase_4_summary_file = (
                save_named_dataframe(
                    dataframe=phase_4_data.summary,
                    output_directory=(
                        summary_directory
                    ),
                    filename=(
                        PHASE_4_SUMMARY_FILE_NAME
                    ),
                )
            )
            final_dataframe = (
                phase_4_data.dataframe
            )
        else:
            logger.info(
                "Phase 4 is disabled for KPI '%s'; Phase 3 output becomes "
                "the final dataset.",
                kpi_name,
            )
            final_dataframe = (
                phase_3_data.dataframe
            )

        validate_final_runner_output(
            timeline_dataframe=(
                timeline_dataframe
            ),
            final_dataframe=final_dataframe,
            timeline_result=timeline_result,
            prophet_enabled=(
                preprocessing_config
                .phase_4.enabled
            ),
        )

        final_dataset_file = (
            save_named_dataframe(
                dataframe=final_dataframe,
                output_directory=(
                    resolved_output_directory
                ),
                filename=(
                    f"{safe_kpi_name}"
                    f"{FINAL_DATASET_SUFFIX}"
                ),
            )
        )

    unresolved_points = (
        count_unresolved_points(
            final_dataframe
        )
    )

    if phase_4_data is None:
        phase_4_safe_filled_points = 0
        phase_4_leakage_filled_points = 0
        phase_4_pass_1_counts: dict[
            str,
            int,
        ] = {}
        phase_4_pass_2_counts: dict[
            str,
            int,
        ] = {}
    else:
        phase_4_safe_filled_points = (
            phase_4_data
            .total_safe_filled_points
        )
        phase_4_leakage_filled_points = (
            phase_4_data
            .total_leakage_filled_points
        )
        phase_4_pass_1_counts = dict(
            phase_4_data.pass_1_counts
        )
        phase_4_pass_2_counts = dict(
            phase_4_data.pass_2_counts
        )

    result = PreprocessingRunnerResult(
        kpi_name=kpi_name,
        source_file=source_file,
        output_directory=(
            resolved_output_directory
        ),
        resolved_frequency=(
            resolved_frequency
        ),
        prophet_enabled=(
            preprocessing_config
            .phase_4.enabled
        ),
        final_dataset_file=(
            final_dataset_file
        ),
        outlier_summary_file=(
            outlier_summary_file
        ),
        gap_summary_file=gap_summary_file,
        phase_1_summary_file=(
            phase_1_summary_file
        ),
        phase_2_summary_file=(
            phase_2_summary_file
        ),
        phase_3_summary_file=(
            phase_3_summary_file
        ),
        phase_4_summary_file=(
            phase_4_summary_file
        ),
        timeline_checkpoint_file=(
            timeline_checkpoint_file
        ),
        outlier_checkpoint_file=(
            outlier_checkpoint_file
        ),
        gap_checkpoint_file=(
            gap_checkpoint_file
        ),
        phase_1_checkpoint_file=(
            phase_1_checkpoint_file
        ),
        phase_2_checkpoint_file=(
            phase_2_checkpoint_file
        ),
        phase_3_checkpoint_file=(
            phase_3_checkpoint_file
        ),
        total_input_rows=(
            timeline_result.total_input_rows
        ),
        final_rows=len(final_dataframe),
        duplicate_rows_removed=(
            timeline_result
            .duplicate_rows_removed
        ),
        inserted_timestamp_rows=(
            timeline_result
            .inserted_timestamp_rows
        ),
        invalid_rows_converted=(
            timeline_result
            .invalid_rows_converted
        ),
        valid_series=(
            usability_enforcement.valid_series
        ),
        invalid_series=(
            usability_enforcement.invalid_series
        ),
        outliers_flagged=(
            outlier_data.total_outliers
        ),
        original_gaps=gap_data.total_gaps,
        original_missing_points=(
            gap_data.total_missing_points
        ),
        phase_1_filled_points=(
            phase_1_data.filled_points
        ),
        phase_2_filled_points=(
            phase_2_data.filled_points
        ),
        phase_3_filled_points=(
            phase_3_data.filled_points
        ),
        phase_4_safe_filled_points=(
            phase_4_safe_filled_points
        ),
        phase_4_leakage_filled_points=(
            phase_4_leakage_filled_points
        ),
        unresolved_points=unresolved_points,
        phase_2_method_counts=dict(
            phase_2_data.method_counts
        ),
        phase_3_method_counts=dict(
            phase_3_data.method_counts
        ),
        phase_4_pass_1_counts=(
            phase_4_pass_1_counts
        ),
        phase_4_pass_2_counts=(
            phase_4_pass_2_counts
        ),
    )

    logger.info(
        "Inference-specific Part B preprocessing completed for KPI '%s'. Final dataset: %s. "
        "Filled points: phase1=%d, phase2=%d, phase3=%d, "
        "phase4_safe=%d, phase4_leakage=%d. Unresolved=%d.",
        result.kpi_name,
        result.final_dataset_file,
        result.phase_1_filled_points,
        result.phase_2_filled_points,
        result.phase_3_filled_points,
        result.phase_4_safe_filled_points,
        result.phase_4_leakage_filled_points,
        result.unresolved_points,
    )

    return result


__all__ = [
    "CachedUsabilityEnforcementResult",
    "enforce_cached_series_usability",
    "run_preprocessing_pipeline_for_inference",
]