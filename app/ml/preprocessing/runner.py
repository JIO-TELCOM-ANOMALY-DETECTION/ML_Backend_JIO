"""
Orchestrate all Part B preprocessing stages for one prepared KPI dataset.

This runner receives one structured ``KPIEDARunResult`` from Part A. It does
not search directories or reconstruct Part A filenames. Exact source, site,
and data-integrity paths are read from the supplied result object.

Processing order
----------------

1. timeline correction and initial labels;
2. selected outlier labelling method;
3. immutable original-gap classification;
4. Phase 1 filling;
5. Phase 2 filling;
6. Phase 3 filling; and
7. optional two-pass Prophet Phase 4 filling.

The workflow remains responsible for:

- creating the run and KPI processing directory;
- configuring the run logger;
- calling this runner once for each ``KPIEDARunResult``; and
- updating the run manifest from the structured result returned here.

The runner resolves ``PreprocessingConfig`` for the Part A bucket exactly once
and passes the same ``ResolvedFrequencyConfig`` to every frequency-dependent
stage.

Output policy
-------------

The final cleaned dataset and all stage summaries are always saved. Dataset
checkpoints are saved only when ``save_intermediate_outputs`` is true.
Temporary files required by the timeline-correction wrapper are created inside
a short-lived work directory and are removed automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd

from app.core.exceptions import (
    FileProcessingError,
    InputValidationError,
)
from app.ml.eda.runner import KPIEDARunResult
from app.schemas.common.dataset import (
    BucketConfig,
    DatasetInputConfig,
)
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
from .timeline_correction import (
    CURRENT_STATUS_LABEL_COLUMN,
    ORIGINAL_CONDITION_LABEL_COLUMN,
    TimelineCorrectionResult,
    create_timeline_corrected_dataset,
)


FINAL_DATASET_SUFFIX = "_final_cleaned.csv"
SUMMARY_DIRECTORY_NAME = "summaries"
INTERMEDIATE_DIRECTORY_NAME = "intermediate"

OUTLIER_SUMMARY_FILE_NAME = "01_outlier_summary.csv"
GAP_SUMMARY_FILE_NAME = "02_original_gap_summary.csv"
PHASE_1_SUMMARY_FILE_NAME = "03_phase_1_summary.csv"
PHASE_2_SUMMARY_FILE_NAME = "04_phase_2_summary.csv"
PHASE_3_SUMMARY_FILE_NAME = "05_phase_3_summary.csv"
PHASE_4_SUMMARY_FILE_NAME = "06_phase_4_summary.csv"

TIMELINE_CHECKPOINT_FILE_NAME = "01_timeline_corrected.csv"
OUTLIER_CHECKPOINT_FILE_NAME = "02_outlier_labelled.csv"
GAP_CHECKPOINT_FILE_NAME = "03_original_gap_classified.csv"
PHASE_1_CHECKPOINT_FILE_NAME = "04_phase_1_filled.csv"
PHASE_2_CHECKPOINT_FILE_NAME = "05_phase_2_filled.csv"
PHASE_3_CHECKPOINT_FILE_NAME = "06_phase_3_filled.csv"

UNRESOLVED_CURRENT_LABELS = {-2, -1, 0}


@dataclass(frozen=True, slots=True)
class PreprocessingRunnerResult:
    """
    Store all Part B paths, settings, and headline statistics for one KPI.

    Attributes:
        kpi_name:
            KPI identity returned by Part A dataset preparation.

        source_file:
            Exact prepared KPI file received from Part A.

        output_directory:
            Workflow-owned directory used for this KPI's Part B outputs.

        resolved_frequency:
            Frequency-dependent preprocessing settings resolved once from the
            Part A bucket.

        prophet_enabled:
            Whether the runner executed Phase 4.

        final_dataset_file:
            Stable final cleaned dataset path.

        outlier_summary_file:
            Per-site outlier summary.

        gap_summary_file:
            One-row-per-original-gap summary containing start, end, size, and
            frozen phase.

        phase_1_summary_file:
            Per-site Phase 1 summary.

        phase_2_summary_file:
            Per-site Phase 2 summary.

        phase_3_summary_file:
            Per-site Phase 3 summary.

        phase_4_summary_file:
            Per-site-per-pass Phase 4 summary, or ``None`` when disabled.

        timeline_checkpoint_file:
            Saved timeline-corrected dataset when intermediate outputs are
            enabled, otherwise ``None``.

        outlier_checkpoint_file:
            Saved outlier-labelled dataset when enabled, otherwise ``None``.

        gap_checkpoint_file:
            Saved original-gap-classified dataset when enabled, otherwise
            ``None``.

        phase_1_checkpoint_file:
            Saved Phase 1 dataset when enabled, otherwise ``None``.

        phase_2_checkpoint_file:
            Saved Phase 2 dataset when enabled, otherwise ``None``.

        phase_3_checkpoint_file:
            Saved Phase 3 dataset when enabled, otherwise ``None``.

        total_input_rows:
            Rows in the prepared KPI file before timeline correction.

        final_rows:
            Rows in the corrected final dataset.

        duplicate_rows_removed:
            Later duplicate site-timestamp rows removed by timeline correction.

        inserted_timestamp_rows:
            Missing timestamp rows inserted from Part A gap details.

        invalid_rows_converted:
            Original invalid KPI values converted to missing values.

        valid_series:
            Site series marked valid for filling.

        invalid_series:
            Site series marked invalid for filling.

        outliers_flagged:
            Rows labelled as outliers.

        original_gaps:
            Number of immutable original missing-value gaps.

        original_missing_points:
            Number of rows carrying original labels ``-2``, ``-1``, or ``0``.

        phase_1_filled_points:
            Points filled during Phase 1.

        phase_2_filled_points:
            Points filled during Phase 2, including retries.

        phase_3_filled_points:
            Points filled during Phase 3, including retries.

        phase_4_safe_filled_points:
            Past-only Prophet fills labelled ``3``.

        phase_4_leakage_filled_points:
            Full-series Prophet fallback fills labelled ``4``.

        unresolved_points:
            Rows still carrying current labels ``-2``, ``-1``, or ``0`` in
            the final dataset, including rows in invalid site series.

        phase_2_method_counts:
            Point counts for all Phase 2 methods.

        phase_3_method_counts:
            Point counts for all Phase 3 methods.

        phase_4_pass_1_counts:
            Aggregated Phase 4 Pass 1 counts, empty when Phase 4 is disabled.

        phase_4_pass_2_counts:
            Aggregated Phase 4 Pass 2 counts, empty when Phase 4 is disabled.
    """

    kpi_name: str
    source_file: Path
    output_directory: Path
    resolved_frequency: ResolvedFrequencyConfig
    prophet_enabled: bool

    final_dataset_file: Path
    outlier_summary_file: Path
    gap_summary_file: Path
    phase_1_summary_file: Path
    phase_2_summary_file: Path
    phase_3_summary_file: Path
    phase_4_summary_file: Path | None

    timeline_checkpoint_file: Path | None
    outlier_checkpoint_file: Path | None
    gap_checkpoint_file: Path | None
    phase_1_checkpoint_file: Path | None
    phase_2_checkpoint_file: Path | None
    phase_3_checkpoint_file: Path | None

    total_input_rows: int
    final_rows: int
    duplicate_rows_removed: int
    inserted_timestamp_rows: int
    invalid_rows_converted: int
    valid_series: int
    invalid_series: int
    outliers_flagged: int
    original_gaps: int
    original_missing_points: int
    phase_1_filled_points: int
    phase_2_filled_points: int
    phase_3_filled_points: int
    phase_4_safe_filled_points: int
    phase_4_leakage_filled_points: int
    unresolved_points: int

    phase_2_method_counts: dict[str, int]
    phase_3_method_counts: dict[str, int]
    phase_4_pass_1_counts: dict[str, int]
    phase_4_pass_2_counts: dict[str, int]


def resolve_kpi_name(
    kpi_result: KPIEDARunResult,
) -> str:
    """
    Resolve a stable KPI name from the Part A result.

    Args:
        kpi_result:
            Structured Part A result for one KPI.

    Returns:
        str:
            Non-empty KPI name suitable for logs and output naming.

    Raises:
        InputValidationError:
            If neither Part A KPI identity nor the source filename provides a
            usable name.
    """

    prepared_dataset = kpi_result.prepared_dataset
    raw_name = getattr(
        prepared_dataset,
        "kpi_name",
        None,
    )

    if raw_name is not None:
        normalized = str(raw_name).strip()

        if normalized:
            return normalized

    source_file = Path(
        prepared_dataset.file_path
    )
    fallback = source_file.stem.strip()

    if fallback:
        return fallback

    raise InputValidationError(
        "The Part A KPI result does not contain a usable KPI name.",
        error_code="MISSING_PREPROCESSING_KPI_NAME",
    )


def validate_part_a_handoff(
    kpi_result: KPIEDARunResult,
) -> tuple[Path, Path]:
    """
    Validate exact Part A files and cross-result path consistency.

    Args:
        kpi_result:
            Structured Part A result containing prepared, site, and integrity
            outputs for one KPI.

    Returns:
        tuple[Path, Path]:
            Resolved prepared source file and site-summary file.

    Raises:
        FileProcessingError:
            If a required Part A file is unavailable.

        InputValidationError:
            If the supplied Part A result mixes paths from different KPI runs.
    """

    source_file = Path(
        kpi_result.prepared_dataset.file_path
    ).expanduser().resolve()
    site_summary_file = Path(
        kpi_result.site_analysis.summary_file
    ).expanduser().resolve()

    if not source_file.is_file():
        raise FileProcessingError(
            "The prepared KPI file returned by Part A does not exist.",
            error_code="MISSING_PREPROCESSING_SOURCE_FILE",
            details={
                "source_file": str(source_file),
            },
        )

    if not site_summary_file.is_file():
        raise FileProcessingError(
            "The site-summary file returned by Part A does not exist.",
            error_code="MISSING_PREPROCESSING_SITE_SUMMARY",
            details={
                "site_summary_file": str(site_summary_file),
            },
        )

    integrity_source = Path(
        kpi_result.data_integrity.source_file
    ).expanduser().resolve()
    integrity_site_summary = Path(
        kpi_result.data_integrity.site_summary_file
    ).expanduser().resolve()

    if integrity_source != source_file:
        raise InputValidationError(
            "The Part A data-integrity result belongs to a different source "
            "file than the prepared KPI result.",
            error_code="MISMATCHED_PREPROCESSING_INTEGRITY_SOURCE",
            details={
                "prepared_source": str(source_file),
                "integrity_source": str(integrity_source),
            },
        )

    if integrity_site_summary != site_summary_file:
        raise InputValidationError(
            "The Part A data-integrity result uses a different site-summary "
            "file than the site-analysis result.",
            error_code="MISMATCHED_PREPROCESSING_SITE_SUMMARY",
            details={
                "site_analysis_summary": str(site_summary_file),
                "integrity_site_summary": str(
                    integrity_site_summary
                ),
            },
        )

    return source_file, site_summary_file


def save_named_dataframe(
    *,
    dataframe: pd.DataFrame,
    output_directory: Path,
    filename: str,
) -> Path:
    """
    Save one dataframe using a stable runner-owned filename.

    Args:
        dataframe:
            Dataframe to save.

        output_directory:
            Existing or creatable destination directory.

        filename:
            Stable filename selected by the runner.

    Returns:
        Path:
            Resolved written file path.

    Raises:
        FileProcessingError:
            If the directory or CSV cannot be created.
    """

    resolved_directory = ensure_directory(
        output_directory
    )
    output_file = (
        resolved_directory
        / filename
    )

    write_csv_file(
        dataframe,
        output_file,
    )

    return output_file


def save_optional_checkpoint(
    *,
    enabled: bool,
    dataframe: pd.DataFrame,
    intermediate_directory: Path | None,
    filename: str,
) -> Path | None:
    """
    Save one dataset checkpoint only when requested by the user.

    Args:
        enabled:
            Whether intermediate dataset checkpoints are enabled.

        dataframe:
            Stage dataframe.

        intermediate_directory:
            Runner-owned checkpoint directory, or ``None`` when disabled.

        filename:
            Stable checkpoint filename.

    Returns:
        Path | None:
            Written checkpoint path, or ``None`` when disabled.

    Raises:
        InputValidationError:
            If checkpoint saving is enabled without a destination directory.

        FileProcessingError:
            If the checkpoint cannot be written.
    """

    if not enabled:
        return None

    if intermediate_directory is None:
        raise InputValidationError(
            "Intermediate outputs are enabled but no intermediate directory "
            "was created.",
            error_code="MISSING_PREPROCESSING_INTERMEDIATE_DIRECTORY",
        )

    return save_named_dataframe(
        dataframe=dataframe,
        output_directory=intermediate_directory,
        filename=filename,
    )


def count_unresolved_points(
    dataframe: pd.DataFrame,
) -> int:
    """
    Count final rows that remain unresolved after all enabled fill stages.

    Args:
        dataframe:
            Final preprocessing dataframe.

    Returns:
        int:
            Number of rows whose current label remains ``-2``, ``-1``, or
            ``0``.

    Raises:
        InputValidationError:
            If the current-status column is missing or contains non-integral
            values.
    """

    if CURRENT_STATUS_LABEL_COLUMN not in dataframe.columns:
        raise InputValidationError(
            "The final preprocessing dataframe is missing the current-status "
            "label column.",
            error_code="MISSING_FINAL_CURRENT_STATUS_COLUMN",
        )

    numeric_labels = pd.to_numeric(
        dataframe[CURRENT_STATUS_LABEL_COLUMN],
        errors="coerce",
    )

    invalid_labels = (
        numeric_labels.isna()
        | ~np.isfinite(numeric_labels)
        | numeric_labels.mod(1).ne(0)
    )

    if invalid_labels.any():
        raise InputValidationError(
            "The final current-status label column contains invalid values.",
            error_code="INVALID_FINAL_CURRENT_STATUS_VALUES",
            details={
                "invalid_rows": int(invalid_labels.sum()),
            },
        )

    return int(
        numeric_labels.isin(
            UNRESOLVED_CURRENT_LABELS
        ).sum()
    )


def validate_final_runner_output(
    *,
    timeline_dataframe: pd.DataFrame,
    final_dataframe: pd.DataFrame,
    timeline_result: TimelineCorrectionResult,
    prophet_enabled: bool,
) -> None:
    """
    Validate final runner-level invariants across all preprocessing stages.

    Individual modules already validate their own transformations. This final
    check verifies invariants that span the complete Part B chain.

    Args:
        timeline_dataframe:
            Dataset immediately after timeline correction.

        final_dataframe:
            Dataset after all enabled filling stages.

        timeline_result:
            Timeline-correction counts returned by the first stage.

        prophet_enabled:
            Whether Phase 4 was executed.

    Returns:
        None.

    Raises:
        InputValidationError:
            If row accounting, original labels, or disabled-Prophet state is
            inconsistent.
    """

    if len(final_dataframe) != timeline_result.final_rows:
        raise InputValidationError(
            "The final preprocessing row count does not match timeline "
            "correction.",
            error_code="PREPROCESSING_FINAL_ROW_COUNT_MISMATCH",
            details={
                "timeline_rows": timeline_result.final_rows,
                "final_rows": len(final_dataframe),
            },
        )

    if len(timeline_dataframe) != len(final_dataframe):
        raise InputValidationError(
            "A preprocessing stage changed the corrected dataset row count.",
            error_code="PREPROCESSING_STAGE_ROW_COUNT_CHANGED",
            details={
                "timeline_rows": len(timeline_dataframe),
                "final_rows": len(final_dataframe),
            },
        )

    if (
        ORIGINAL_CONDITION_LABEL_COLUMN
        not in timeline_dataframe.columns
        or ORIGINAL_CONDITION_LABEL_COLUMN
        not in final_dataframe.columns
    ):
        raise InputValidationError(
            "Original-condition labels are unavailable for final validation.",
            error_code="MISSING_FINAL_ORIGINAL_LABEL_COLUMN",
        )

    timeline_original_labels = pd.to_numeric(
        timeline_dataframe[
            ORIGINAL_CONDITION_LABEL_COLUMN
        ],
        errors="coerce",
    )
    final_original_labels = pd.to_numeric(
        final_dataframe[
            ORIGINAL_CONDITION_LABEL_COLUMN
        ],
        errors="coerce",
    )

    invalid_timeline_labels = (
        timeline_original_labels.isna()
        | ~np.isfinite(timeline_original_labels)
        | timeline_original_labels.mod(1).ne(0)
    )
    invalid_final_labels = (
        final_original_labels.isna()
        | ~np.isfinite(final_original_labels)
        | final_original_labels.mod(1).ne(0)
    )

    if (
        invalid_timeline_labels.any()
        or invalid_final_labels.any()
    ):
        raise InputValidationError(
            "Original-condition labels are invalid during final validation.",
            error_code="INVALID_FINAL_ORIGINAL_LABEL_VALUES",
            details={
                "timeline_invalid_rows": int(
                    invalid_timeline_labels.sum()
                ),
                "final_invalid_rows": int(
                    invalid_final_labels.sum()
                ),
            },
        )

    if not np.array_equal(
        timeline_original_labels.to_numpy(
            dtype=np.int64
        ),
        final_original_labels.to_numpy(
            dtype=np.int64
        ),
    ):
        raise InputValidationError(
            "One or more stages modified immutable original-condition labels.",
            error_code="PREPROCESSING_ORIGINAL_LABELS_CHANGED",
        )

    if not prophet_enabled:
        final_current_labels = pd.to_numeric(
            final_dataframe[
                CURRENT_STATUS_LABEL_COLUMN
            ],
            errors="coerce",
        )

        if final_current_labels.eq(4).any():
            raise InputValidationError(
                "Status label 4 exists even though Prophet Phase 4 was "
                "disabled.",
                error_code="LEAKAGE_LABEL_WITH_PHASE4_DISABLED",
                details={
                    "label_4_rows": int(
                        final_current_labels.eq(4).sum()
                    ),
                },
            )


def run_preprocessing_pipeline(
    *,
    kpi_result: KPIEDARunResult,
    output_directory: str | Path,
    dataset_config: DatasetInputConfig,
    preprocessing_config: PreprocessingConfig,
    bucket: BucketConfig,
    logger: Logger,
) -> PreprocessingRunnerResult:
    """
    Run complete Part B preprocessing for one Part A KPI result.

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
        "Starting Part B preprocessing for KPI '%s'. Source: %s. "
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
            timeline_result.valid_series
        ),
        invalid_series=(
            timeline_result.invalid_series
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
        "Part B preprocessing completed for KPI '%s'. Final dataset: %s. "
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
    "PreprocessingRunnerResult",
    "count_unresolved_points",
    "run_preprocessing_pipeline",
    "validate_final_runner_output",
    "validate_part_a_handoff",
]