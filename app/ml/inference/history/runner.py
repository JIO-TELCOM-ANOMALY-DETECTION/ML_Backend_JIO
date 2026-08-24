"""
Orchestrate the inference-time history subsystem.

Intended project location::

    app/ml/inference/history/runner.py

The parent inference runner resolves the run ID and saved inference contract
before calling this module. This runner therefore receives only resolved
paths/settings and the new raw file.

Flow
----
load normal inputs -> validate
-> if reconstruction: load extra older raw-history slice
-> prepare raw -> Part A -> inference-specific Part B with cached usability
-> prepare correct Part C input -> Part C -> return pending history state

Nothing persistent is changed by ``run_history_pipeline``. The parent
inference runner must finish model inference successfully and only then call
``commit_history_pipeline``.
"""

from __future__ import annotations

from dataclasses import dataclass
from logging import Logger
from pathlib import Path

import pandas as pd

from app.core.exceptions import InputValidationError
from app.ml.eda.runner import EDARunnerResult, run_eda_pipeline
from app.ml.feature_engineering.runner import (
    FeatureEngineeringRunnerResult,
    run_feature_engineering_pipeline,
)
from app.ml.preprocessing.runner import PreprocessingRunnerResult
from app.ml.preprocessing.runner_inference import (
    run_preprocessing_pipeline_for_inference,
)
from app.schemas.common.dataset import DatasetInputConfig, ExpectedTimelineConfig
from app.schemas.inference.request import (
    ResolvedStoragePaths,
    SavedDatasetContract,
    SavedFeatureEngineeringContract,
    SavedHistoryContract,
    SavedPreprocessingContract,
)
from app.services.file_service import ensure_directory, sanitise_filename

from .loader import (
    HistoryLoadResult,
    LoadedReconstructionRawHistory,
    load_history_inputs,
    load_reconstruction_raw_history_slice,
)
from .preparation import (
    PartCPreparationResult,
    RawHistoryPreparationResult,
    prepare_part_c_inputs,
    prepare_raw_working_input,
)
from .updater import HistoryCommitResult, commit_history_updates
from .validation import HistoryValidationResult, validate_history_inputs


@dataclass(frozen=True, slots=True)
class HistoryRunContext:
    """Internal resolved input supplied by the main inference runner."""

    incoming_file: Path
    inference_days: int

    dataset: SavedDatasetContract
    preprocessing: SavedPreprocessingContract
    feature_engineering: SavedFeatureEngineeringContract
    history: SavedHistoryContract
    storage: ResolvedStoragePaths

    working_directory: Path
    enforce_cached_usability: bool = True


@dataclass(frozen=True, slots=True)
class HistoryRunnerResult:
    """Complete non-persistent result ready for model inference."""

    context: HistoryRunContext

    loaded: HistoryLoadResult
    validation: HistoryValidationResult
    raw_preparation: RawHistoryPreparationResult

    runtime_dataset_config: DatasetInputConfig

    eda_result: EDARunnerResult
    preprocessing_results: tuple[PreprocessingRunnerResult, ...]

    part_c_preparation: PartCPreparationResult
    feature_results: tuple[FeatureEngineeringRunnerResult, ...]

    @property
    def mode(self) -> str:
        return self.validation.mode

    def get_feature_result(
        self,
        kpi_name: str,
    ) -> FeatureEngineeringRunnerResult:
        for result in self.feature_results:
            if result.kpi_name == kpi_name:
                return result

        raise KeyError(
            f"No history Part C result exists for KPI {kpi_name!r}."
        )


def _validate_context(
    context: HistoryRunContext,
    *,
    logger: Logger,
) -> None:
    if not isinstance(logger, Logger):
        raise InputValidationError(
            "logger must be an instance of logging.Logger.",
            error_code="INVALID_HISTORY_RUNNER_LOGGER",
            details={"received_type": type(logger).__name__},
        )

    if context.inference_days < 1:
        raise InputValidationError(
            "History inference_days must be at least one.",
            error_code="INVALID_HISTORY_RUNNER_INFERENCE_DAYS",
            details={"inference_days": context.inference_days},
        )

    if not isinstance(context.enforce_cached_usability, bool):
        raise InputValidationError(
            "enforce_cached_usability must be a boolean.",
            error_code="INVALID_HISTORY_USABILITY_ENFORCEMENT_FLAG",
            details={
                "received_type": type(
                    context.enforce_cached_usability
                ).__name__,
            },
        )

    expected_kpis = set(context.dataset.kpi_names)

    if set(context.storage.processed_history_by_kpi) != expected_kpis:
        raise InputValidationError(
            "Processed-history paths do not match the trained KPI set.",
            error_code="HISTORY_RUNNER_PROCESSED_HISTORY_KPI_MISMATCH",
        )

    if set(context.storage.processed_cache_by_kpi) != expected_kpis:
        raise InputValidationError(
            "Processed-cache paths do not match the trained KPI set.",
            error_code="HISTORY_RUNNER_PROCESSED_CACHE_KPI_MISMATCH",
        )

    resolved_b = context.preprocessing.resolved
    resolved_c = context.feature_engineering.resolved

    if resolved_b.bucket_minutes != resolved_c.bucket_minutes:
        raise InputValidationError(
            "Saved Part B and Part C bucket resolutions do not match.",
            error_code="HISTORY_RUNNER_BUCKET_RESOLUTION_MISMATCH",
        )

    if resolved_b.points_per_day != resolved_c.points_per_day:
        raise InputValidationError(
            "Saved Part B and Part C points_per_day values do not match.",
            error_code="HISTORY_RUNNER_POINTS_PER_DAY_MISMATCH",
        )


def _to_abc_timezone_naive(
    value: pd.Timestamp,
) -> pd.Timestamp:
    """
    Convert one Part-E internal timestamp to the A/B/C timestamp contract.

    Part E performs history comparison/classification in UTC-aware form.
    Existing Parts A/B/C, including Prophet Phase 4, expect timezone-naive
    timestamps. Convert aware timestamps to UTC first and then remove timezone
    metadata; already-naive timestamps are returned unchanged.
    """

    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is not None:
        timestamp = (
            timestamp
            .tz_convert("UTC")
            .tz_localize(None)
        )

    return timestamp


def _build_runtime_dataset_config(
    *,
    context: HistoryRunContext,
    raw_preparation: RawHistoryPreparationResult,
) -> DatasetInputConfig:
    """
    Build the Part A input contract for the prepared inference working file.

    The saved training date range is never reused. An explicit inference-time
    date range is used so Part A creates all expected buckets through the
    requested X-day end even when source rows are absent near that boundary.
    """

    base = context.dataset.build_runtime_dataset_config(
        input_file=raw_preparation.working_raw_file,
    )

    timeline = ExpectedTimelineConfig(
        timeline_mode="date_range",
        expected_start=_to_abc_timezone_naive(
            raw_preparation.expected_timeline_start
        ).to_pydatetime(),
        expected_end=_to_abc_timezone_naive(
            raw_preparation.expected_timeline_end
        ).to_pydatetime(),
    )

    return base.model_copy(
        update={"expected_timeline": timeline}
    )


def _safe_kpi_name(value: str) -> str:
    safe = sanitise_filename(str(value).strip())
    return safe or "kpi"


def _validate_part_a_bucket(
    *,
    context: HistoryRunContext,
    eda_result: EDARunnerResult,
) -> None:
    actual = eda_result.preparation.bucket

    if actual != context.dataset.bucket:
        raise InputValidationError(
            "Inference-time Part A resolved a bucket different from the "
            "saved training contract.",
            error_code="HISTORY_RUNNER_PART_A_BUCKET_MISMATCH",
            details={
                "saved_bucket": context.dataset.bucket.model_dump(mode="json"),
                "runtime_bucket": actual.model_dump(mode="json"),
            },
        )


def _run_part_b(
    *,
    context: HistoryRunContext,
    loaded: HistoryLoadResult,
    runtime_dataset_config: DatasetInputConfig,
    eda_result: EDARunnerResult,
    output_directory: Path,
    logger: Logger,
) -> tuple[PreprocessingRunnerResult, ...]:
    results: list[PreprocessingRunnerResult] = []

    for kpi_result in eda_result.kpi_results:
        raw_name = getattr(
            kpi_result.prepared_dataset,
            "kpi_name",
            None,
        )
        kpi_name = str(raw_name).strip() if raw_name is not None else ""

        if not kpi_name:
            kpi_name = Path(
                kpi_result.prepared_dataset.file_path
            ).stem.strip()

        if context.enforce_cached_usability:
            result = run_preprocessing_pipeline_for_inference(
                kpi_result=kpi_result,
                output_directory=ensure_directory(
                    output_directory / _safe_kpi_name(kpi_name)
                ),
                dataset_config=runtime_dataset_config,
                preprocessing_config=context.preprocessing.config,
                saved_processed_cache=(
                    loaded
                    .get_processed_cache(kpi_name)
                    .dataframe
                ),
                bucket=context.dataset.bucket,
                logger=logger,
            )
        else:
            raise InputValidationError(
                "Inference history runner requires cached-usability "
                "enforcement for Part B.",
                error_code="DISABLED_INFERENCE_USABILITY_ENFORCEMENT",
                details={"kpi_name": kpi_name},
            )

        if result.resolved_frequency != context.preprocessing.resolved:
            raise InputValidationError(
                "Inference-time Part B resolved settings differ from the "
                "saved training-time Part B contract.",
                error_code="HISTORY_RUNNER_PART_B_RESOLUTION_MISMATCH",
                details={"kpi_name": result.kpi_name},
            )

        results.append(result)

    names = [item.kpi_name for item in results]

    if len(names) != len(set(names)):
        raise InputValidationError(
            "Part B returned duplicate KPI results.",
            error_code="HISTORY_RUNNER_DUPLICATE_PART_B_KPI",
            details={"received_kpis": names},
        )

    if set(names) != set(context.dataset.kpi_names):
        raise InputValidationError(
            "Inference-time Part B results do not match the trained KPI set.",
            error_code="HISTORY_RUNNER_PART_B_KPI_SET_MISMATCH",
            details={
                "expected_kpis": sorted(context.dataset.kpi_names),
                "received_kpis": sorted(names),
            },
        )

    by_kpi = {item.kpi_name: item for item in results}

    return tuple(
        by_kpi[kpi_name]
        for kpi_name in context.dataset.kpi_names
    )


def _run_part_c(
    *,
    context: HistoryRunContext,
    runtime_dataset_config: DatasetInputConfig,
    preparation: PartCPreparationResult,
    output_directory: Path,
    logger: Logger,
) -> tuple[FeatureEngineeringRunnerResult, ...]:
    results: list[FeatureEngineeringRunnerResult] = []

    for kpi_name in context.dataset.kpi_names:
        prepared = preparation.get_kpi(kpi_name)

        result = run_feature_engineering_pipeline(
            preprocessing_result=prepared.part_c_preprocessing_result,
            output_directory=ensure_directory(
                output_directory / _safe_kpi_name(kpi_name)
            ),
            dataset_config=runtime_dataset_config,
            feature_config=context.feature_engineering.config,
            bucket=context.dataset.bucket,
            logger=logger,
        )

        if result.resolved_config != context.feature_engineering.resolved:
            raise InputValidationError(
                "Inference-time Part C resolved settings differ from the "
                "saved training-time Part C contract.",
                error_code="HISTORY_RUNNER_PART_C_RESOLUTION_MISMATCH",
                details={"kpi_name": result.kpi_name},
            )

        results.append(result)

    names = [item.kpi_name for item in results]

    if len(names) != len(set(names)):
        raise InputValidationError(
            "Part C returned duplicate KPI results.",
            error_code="HISTORY_RUNNER_DUPLICATE_PART_C_KPI",
            details={"received_kpis": names},
        )

    if set(names) != set(context.dataset.kpi_names):
        raise InputValidationError(
            "Inference-time Part C results do not match the trained KPI set.",
            error_code="HISTORY_RUNNER_PART_C_KPI_SET_MISMATCH",
            details={
                "expected_kpis": sorted(context.dataset.kpi_names),
                "received_kpis": sorted(names),
            },
        )

    by_kpi = {item.kpi_name: item for item in results}

    return tuple(
        by_kpi[kpi_name]
        for kpi_name in context.dataset.kpi_names
    )


def run_history_pipeline(
    *,
    context: HistoryRunContext,
    logger: Logger,
) -> HistoryRunnerResult:
    """
    Run history preparation through existing Parts A, B, and C.

    Permanent raw/processed/cache files remain untouched.
    """

    _validate_context(context, logger=logger)

    root = ensure_directory(
        Path(context.working_directory).expanduser().resolve()
    )
    preparation_dir = ensure_directory(root / "history_preparation")
    part_a_dir = ensure_directory(root / "part_a")
    part_b_dir = ensure_directory(root / "part_b")
    part_c_dir = ensure_directory(root / "part_c")

    loaded = load_history_inputs(
        incoming_file=context.incoming_file,
        raw_cache_file=context.storage.raw_cache_file,
        processed_cache_by_kpi=context.storage.processed_cache_by_kpi,
        expected_kpis=context.dataset.kpi_names,
    )

    validation = validate_history_inputs(
        loaded=loaded,
        columns=context.dataset.columns,
        expected_kpis=context.dataset.kpi_names,
        single_kpi_name=context.dataset.single_kpi_name,
        preprocessing_config=context.preprocessing.config,
        history_config=context.history,
        bucket_minutes=context.preprocessing.resolved.bucket_minutes,
        points_per_day=context.preprocessing.resolved.points_per_day,
        inference_days=context.inference_days,
    )

    reconstruction_raw_history: (
        LoadedReconstructionRawHistory | None
    ) = None

    if validation.mode == "reconstruction":
        reconstruction_raw_history = (
            load_reconstruction_raw_history_slice(
                raw_history_file=(
                    context.storage.raw_history_file
                ),
                raw_cache_dataframe=(
                    loaded.raw_cache_dataframe
                ),
                timestamp_column=(
                    context.dataset.columns.timestamp_column
                ),
                cache_days=context.history.cache_days,
            )
        )

    raw_preparation = prepare_raw_working_input(
        loaded=loaded,
        validation=validation,
        columns=context.dataset.columns,
        single_kpi_name=context.dataset.single_kpi_name,
        history_config=context.history,
        output_directory=preparation_dir,
        reconstruction_raw_history=(
            reconstruction_raw_history
        ),
    )

    # Preparation has already copied the older slice into its temporary
    # working dataset. Do not keep a duplicate reconstruction dataframe alive
    # while Parts A/B/C and later model inference run.
    del reconstruction_raw_history

    runtime_dataset_config = _build_runtime_dataset_config(
        context=context,
        raw_preparation=raw_preparation,
    )

    eda_result = run_eda_pipeline(
        runtime_dataset_config,
        logger,
        output_directory=part_a_dir,
    )

    _validate_part_a_bucket(
        context=context,
        eda_result=eda_result,
    )

    preprocessing_results = _run_part_b(
        context=context,
        loaded=loaded,
        runtime_dataset_config=runtime_dataset_config,
        eda_result=eda_result,
        output_directory=part_b_dir,
        logger=logger,
    )

    part_c_preparation = prepare_part_c_inputs(
        loaded=loaded,
        validation=validation,
        preprocessing_results=preprocessing_results,
        columns=context.dataset.columns,
        expected_kpis=context.dataset.kpi_names,
        output_directory=preparation_dir,
    )

    feature_results = _run_part_c(
        context=context,
        runtime_dataset_config=runtime_dataset_config,
        preparation=part_c_preparation,
        output_directory=part_c_dir,
        logger=logger,
    )

    return HistoryRunnerResult(
        context=context,
        loaded=loaded,
        validation=validation,
        raw_preparation=raw_preparation,
        runtime_dataset_config=runtime_dataset_config,
        eda_result=eda_result,
        preprocessing_results=preprocessing_results,
        part_c_preparation=part_c_preparation,
        feature_results=feature_results,
    )


def commit_history_pipeline(
    result: HistoryRunnerResult,
) -> HistoryCommitResult:
    """
    Persist history only after the parent model-inference step succeeds.
    """

    context = result.context

    return commit_history_updates(
        raw_preparation=result.raw_preparation,
        part_c_preparation=result.part_c_preparation,
        columns=context.dataset.columns,
        single_kpi_name=context.dataset.single_kpi_name,
        history_config=context.history,
        bucket_minutes=context.preprocessing.resolved.bucket_minutes,
        expected_kpis=context.dataset.kpi_names,
        raw_history_file=context.storage.raw_history_file,
        raw_cache_file=context.storage.raw_cache_file,
        processed_history_by_kpi=context.storage.processed_history_by_kpi,
        processed_cache_by_kpi=context.storage.processed_cache_by_kpi,
    )


__all__ = [
    "HistoryRunContext",
    "HistoryRunnerResult",
    "commit_history_pipeline",
    "run_history_pipeline",
]