"""
Main pure-inference orchestrator.

Intended project location::

    app/ml/inference/runner.py

The service/API layer resolves ``run_id -> run_directory`` before calling this
module. This runner then coordinates the complete saved-model inference flow:

    request + run_directory
        -> loader
        -> history subsystem (A -> B -> C)
        -> Part C / saved-model runtime validation
        -> one site at a time:
             load one site's validated feature rows
             load one saved model
             run model-specific inference
             stage that site's output
             release dataframe/model
        -> write staged manifest
        -> commit persistent history exactly once
        -> publish the completed inference-result directory atomically

Memory rule
-----------
The loader keeps only paths/contracts for all sites. The runner never
deserializes all models at once and never accumulates all row-level inference
dataframes in memory. At most one site's validated dataframe and one estimator
are held for model execution.

Failure rule
------------
``run_history_pipeline`` does not persist history. Therefore any failure during
validation, model loading, model inference, or output staging leaves persistent
history unchanged.

History is committed only after every ready site has completed inference and
its output has been successfully staged.

This module does not query the run database. The caller must provide the
already-resolved ``run_directory``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import json
from logging import Logger, getLogger
from pathlib import Path
import shutil
from typing import Any, Literal
from uuid import uuid4

import pandas as pd

from app.core.exceptions import (
    FileProcessingError,
    InputValidationError,
)
from app.ml.inference.history import (
    HistoryCommitResult,
    HistoryRunContext,
    commit_history_pipeline,
    run_history_pipeline,
)
from app.ml.inference.isolation_forest import (
    IsolationForestSeriesInferenceResult,
    run_isolation_forest_inference,
)
from app.ml.inference.loader import (
    InferenceLoadResult,
    load_inference_artifacts,
    load_series_model,
)
from app.ml.inference.validation import (
    InferenceSeriesValidationPlan,
    InferenceValidationResult,
    load_and_validate_series_data,
    validate_inference_handoff,
)
from app.ml.inference.xgboost import (
    XGBoostSeriesInferenceResult,
    run_xgboost_inference,
)
from app.schemas.inference.request import InferenceRequest
from app.services.file_service import (
    ensure_directory,
    sanitise_filename,
)


logger = getLogger(__name__)

InferenceModelType = Literal[
    "xgboost",
    "isolation_forest",
]


@dataclass(frozen=True, slots=True)
class InferenceSeriesOutput:
    """Lightweight persisted-output summary for one successfully scored site."""

    model_type: InferenceModelType

    kpi_name: str
    series_key: str
    identity: tuple[str, str, str]

    result_file: Path

    total_window_rows: int
    scorable_rows: int
    unscorable_rows: int

    anomaly_count: int
    anomaly_rate: float

    model_anomaly_count: int | None
    lower_anomaly_count: int
    upper_anomaly_count: int


@dataclass(frozen=True, slots=True)
class SkippedInferenceSeries:
    """Saved site model intentionally skipped because current Part C is invalid."""

    kpi_name: str
    series_key: str
    identity: tuple[str, str, str]
    reason: str


@dataclass(frozen=True, slots=True)
class InferenceRunnerResult:
    """Final lightweight result returned after model inference and history commit."""

    run_id: str
    model_type: InferenceModelType
    invocation_id: str

    inference_start: pd.Timestamp
    inference_end_exclusive: pd.Timestamp

    output_directory: Path
    manifest_file: Path

    total_saved_series: int
    completed_series: int
    skipped_series_count: int
    unmodelled_part_c_series: int

    outputs: tuple[InferenceSeriesOutput, ...]
    skipped_series: tuple[SkippedInferenceSeries, ...]

    history_commit: HistoryCommitResult


def _require_logger(
    supplied: Logger | None,
) -> Logger:
    if supplied is None:
        return logger

    if not isinstance(
        supplied,
        Logger,
    ):
        raise InputValidationError(
            "logger must be an instance of logging.Logger.",
            error_code="INVALID_INFERENCE_RUNNER_LOGGER",
            details={
                "received_type": type(
                    supplied
                ).__name__,
            },
        )

    return supplied


def _build_invocation_id(
    run_id: str,
) -> str:
    stamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    safe_run_id = (
        sanitise_filename(
            run_id
        )
        or "run"
    )

    return (
        f"{safe_run_id}_"
        f"{stamp}_"
        f"{uuid4().hex[:8]}"
    )


def _safe_output_name(
    *,
    index: int,
    series_plan: InferenceSeriesValidationPlan,
) -> str:
    artifact = (
        series_plan.artifact
    )

    components = [
        artifact.kpi_name,
        artifact.identity[0],
        artifact.identity[1],
        artifact.identity[2],
        artifact.series_key,
    ]

    safe = [
        sanitise_filename(
            str(value)
        )
        or "value"
        for value in components
    ]

    return (
        f"{index:05d}_"
        + "__".join(
            safe
        )
        + ".csv"
    )


def _json_safe(
    value: Any,
) -> Any:
    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    if isinstance(
        value,
        datetime,
    ):
        return value.isoformat()

    if isinstance(
        value,
        tuple,
    ):
        return [
            _json_safe(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        list,
    ):
        return [
            _json_safe(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): _json_safe(
                item
            )
            for key, item in value.items()
        }

    return value


def _write_csv(
    dataframe: pd.DataFrame,
    path: Path,
    *,
    role: str,
) -> None:
    try:
        dataframe.to_csv(
            path,
            index=False,
        )
    except Exception as error:
        raise FileProcessingError(
            f"Could not save {role}.",
            error_code="INFERENCE_OUTPUT_WRITE_FAILED",
            details={
                "path": str(
                    path
                ),
                "reason": str(
                    error
                ),
            },
        ) from error

    if not path.is_file():
        raise FileProcessingError(
            f"{role} was not created.",
            error_code="MISSING_STAGED_INFERENCE_OUTPUT",
            details={
                "path": str(
                    path
                ),
            },
        )


def _write_json(
    payload: dict[str, Any],
    path: Path,
    *,
    role: str,
) -> None:
    try:
        with path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                _json_safe(
                    payload
                ),
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write(
                "\n"
            )
    except Exception as error:
        raise FileProcessingError(
            f"Could not save {role}.",
            error_code="INFERENCE_MANIFEST_WRITE_FAILED",
            details={
                "path": str(
                    path
                ),
                "reason": str(
                    error
                ),
            },
        ) from error


def _build_history_context(
    *,
    load_result: InferenceLoadResult,
    working_directory: Path,
) -> HistoryRunContext:
    resolved = (
        load_result
        .resolved_request
    )

    return HistoryRunContext(
        incoming_file=(
            resolved
            .request
            .input_file
        ),
        inference_days=(
            resolved
            .request
            .inference_days
        ),
        dataset=(
            resolved
            .saved
            .dataset
        ),
        preprocessing=(
            resolved
            .saved
            .preprocessing
        ),
        feature_engineering=(
            resolved
            .saved
            .feature_engineering
        ),
        history=(
            resolved
            .saved
            .history
        ),
        storage=(
            resolved
            .storage
        ),
        working_directory=(
            working_directory
        ),
        # Part E keeps the saved processed-cache usability decision
        # authoritative while rerunning Part B on a shorter working window.
        enforce_cached_usability=True,
    )


def _run_one_series(
    *,
    load_result: InferenceLoadResult,
    series_plan: InferenceSeriesValidationPlan,
    inference_start: pd.Timestamp,
    inference_end_exclusive: pd.Timestamp,
) -> (
    XGBoostSeriesInferenceResult
    | IsolationForestSeriesInferenceResult
):
    """
    Load data + estimator for exactly one saved site and run its model.

    All all-site metadata remains lightweight. This function intentionally
    contains the only parent-runner call to ``load_series_model``.
    """

    series_data = (
        load_and_validate_series_data(
            series_plan=series_plan,
            columns=(
                load_result
                .saved
                .dataset
                .columns
            ),
        )
    )

    loaded_model = load_series_model(
        series_plan.artifact
    )

    try:
        if (
            series_plan
            .artifact
            .model_type
            == "xgboost"
        ):
            return run_xgboost_inference(
                loaded_model=loaded_model,
                series_data=series_data,
                inference_start=(
                    inference_start
                ),
                inference_end_exclusive=(
                    inference_end_exclusive
                ),
            )

        if (
            series_plan
            .artifact
            .model_type
            == "isolation_forest"
        ):
            return (
                run_isolation_forest_inference(
                    loaded_model=(
                        loaded_model
                    ),
                    series_data=(
                        series_data
                    ),
                    inference_start=(
                        inference_start
                    ),
                    inference_end_exclusive=(
                        inference_end_exclusive
                    ),
                )
            )

        raise InputValidationError(
            "Unsupported model type reached parent inference execution.",
            error_code="UNSUPPORTED_INFERENCE_RUNNER_MODEL_TYPE",
            details={
                "model_type": (
                    series_plan
                    .artifact
                    .model_type
                ),
            },
        )

    finally:
        # Keep model/data lifetime scoped to exactly one site.
        del loaded_model
        del series_data


def _output_summary(
    *,
    model_type: InferenceModelType,
    result_file: Path,
    result: (
        XGBoostSeriesInferenceResult
        | IsolationForestSeriesInferenceResult
    ),
) -> InferenceSeriesOutput:
    if model_type == "xgboost":
        if not isinstance(
            result,
            XGBoostSeriesInferenceResult,
        ):
            raise InputValidationError(
                "XGBoost runner received a non-XGBoost inference result.",
                error_code="XGBOOST_INFERENCE_RESULT_TYPE_MISMATCH",
            )

        return InferenceSeriesOutput(
            model_type="xgboost",
            kpi_name=(
                result.kpi_name
            ),
            series_key=(
                result.series_key
            ),
            identity=(
                result.identity
            ),
            result_file=(
                result_file
            ),
            total_window_rows=(
                result
                .total_window_rows
            ),
            scorable_rows=(
                result
                .scorable_rows
            ),
            unscorable_rows=(
                result
                .unscorable_rows
            ),
            anomaly_count=(
                result
                .anomaly_count
            ),
            anomaly_rate=(
                result
                .anomaly_rate
            ),
            model_anomaly_count=None,
            lower_anomaly_count=(
                result
                .lower_anomaly_count
            ),
            upper_anomaly_count=(
                result
                .upper_anomaly_count
            ),
        )

    if not isinstance(
        result,
        IsolationForestSeriesInferenceResult,
    ):
        raise InputValidationError(
            "Isolation Forest runner received a non-Isolation-Forest "
            "inference result.",
            error_code="ISOLATION_FOREST_INFERENCE_RESULT_TYPE_MISMATCH",
        )

    return InferenceSeriesOutput(
        model_type=(
            "isolation_forest"
        ),
        kpi_name=result.kpi_name,
        series_key=(
            result.series_key
        ),
        identity=result.identity,
        result_file=(
            result_file
        ),
        total_window_rows=(
            result
            .total_window_rows
        ),
        scorable_rows=(
            result.scorable_rows
        ),
        unscorable_rows=(
            result
            .unscorable_rows
        ),
        anomaly_count=(
            result.anomaly_count
        ),
        anomaly_rate=(
            result.anomaly_rate
        ),
        model_anomaly_count=(
            result
            .model_anomaly_count
        ),
        lower_anomaly_count=(
            result
            .lower_anomaly_count
        ),
        upper_anomaly_count=(
            result
            .upper_anomaly_count
        ),
    )


def _skipped_series(
    validation_result: InferenceValidationResult,
) -> tuple[
    SkippedInferenceSeries,
    ...,
]:
    skipped: list[
        SkippedInferenceSeries
    ] = []

    for kpi in validation_result.kpis:
        for series in kpi.series:
            if (
                series
                .ready_for_model_inference
            ):
                continue

            skipped.append(
                SkippedInferenceSeries(
                    kpi_name=(
                        series
                        .artifact
                        .kpi_name
                    ),
                    series_key=(
                        series
                        .artifact
                        .series_key
                    ),
                    identity=(
                        series
                        .artifact
                        .identity
                    ),
                    reason=(
                        series.skip_reason
                        or (
                            "Current Part C "
                            "series is not "
                            "ready for model "
                            "inference."
                        )
                    ),
                )
            )

    return tuple(
        skipped
    )


def _manifest_payload(
    *,
    load_result: InferenceLoadResult,
    validation_result: InferenceValidationResult,
    invocation_id: str,
    inference_start: pd.Timestamp,
    inference_end_exclusive: pd.Timestamp,
    staged_outputs: tuple[
        InferenceSeriesOutput,
        ...,
    ],
    skipped: tuple[
        SkippedInferenceSeries,
        ...,
    ],
    history_committed: bool,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "run_id": (
            load_result
            .request
            .run_id
        ),
        "invocation_id": (
            invocation_id
        ),
        "model_type": (
            load_result
            .saved
            .model_type
        ),
        "input_file": str(
            load_result
            .request
            .input_file
        ),
        "inference_days": (
            load_result
            .request
            .inference_days
        ),
        "inference_start": (
            inference_start
        ),
        "inference_end_exclusive": (
            inference_end_exclusive
        ),
        "history_committed": (
            history_committed
        ),
        "total_saved_series": (
            validation_result
            .total_saved_series
        ),
        "completed_series": len(
            staged_outputs
        ),
        "skipped_saved_series": len(
            skipped
        ),
        "unmodelled_part_c_series": (
            validation_result
            .unmodelled_part_c_series
        ),
        "outputs": [
            {
                "model_type": (
                    item.model_type
                ),
                "kpi_name": (
                    item.kpi_name
                ),
                "series_key": (
                    item.series_key
                ),
                "identity": (
                    item.identity
                ),
                # Results stay in the same published directory as this
                # manifest, so portable filenames are sufficient.
                "result_file": (
                    item
                    .result_file
                    .name
                ),
                "total_window_rows": (
                    item
                    .total_window_rows
                ),
                "scorable_rows": (
                    item
                    .scorable_rows
                ),
                "unscorable_rows": (
                    item
                    .unscorable_rows
                ),
                "anomaly_count": (
                    item
                    .anomaly_count
                ),
                "anomaly_rate": (
                    item
                    .anomaly_rate
                ),
                "model_anomaly_count": (
                    item
                    .model_anomaly_count
                ),
                "lower_anomaly_count": (
                    item
                    .lower_anomaly_count
                ),
                "upper_anomaly_count": (
                    item
                    .upper_anomaly_count
                ),
            }
            for item in staged_outputs
        ],
        "skipped_series": [
            {
                "kpi_name": (
                    item.kpi_name
                ),
                "series_key": (
                    item.series_key
                ),
                "identity": (
                    item.identity
                ),
                "reason": (
                    item.reason
                ),
            }
            for item in skipped
        ],
    }


def _publish_staging_directory(
    *,
    staging_directory: Path,
    final_directory: Path,
) -> None:
    if final_directory.exists():
        raise FileProcessingError(
            "Final inference result directory already exists.",
            error_code="INFERENCE_RESULT_DIRECTORY_ALREADY_EXISTS",
            details={
                "path": str(
                    final_directory
                ),
            },
        )

    try:
        staging_directory.replace(
            final_directory
        )
    except Exception as error:
        raise FileProcessingError(
            "Completed inference outputs could not be published.",
            error_code="INFERENCE_RESULT_PUBLISH_FAILED",
            details={
                "staging_directory": str(
                    staging_directory
                ),
                "final_directory": str(
                    final_directory
                ),
                "reason": str(
                    error
                ),
            },
        ) from error


def run_inference_pipeline(
    *,
    request: InferenceRequest,
    run_directory: str | Path,
    logger_override: Logger | None = None,
) -> InferenceRunnerResult:
    """
    Run one complete pure-inference request.

    ``run_directory`` must already have been resolved from ``request.run_id``
    by the caller/service.

    Persistent history is changed only after every ready site model completes
    and all row-level outputs are successfully staged.
    """

    active_logger = _require_logger(
        logger_override
    )

    load_result = (
        load_inference_artifacts(
            request=request,
            run_directory=(
                run_directory
            ),
        )
    )

    model_type = (
        load_result
        .saved
        .model_type
    )

    if model_type not in {
        "xgboost",
        "isolation_forest",
    }:
        raise InputValidationError(
            "Unsupported model type in saved inference contract.",
            error_code="UNSUPPORTED_INFERENCE_RUNNER_MODEL_TYPE",
            details={
                "model_type": (
                    model_type
                ),
            },
        )

    invocation_id = (
        _build_invocation_id(
            request.run_id
        )
    )

    results_root = (
        ensure_directory(
            load_result
            .storage
            .inference_results_directory
        )
    )

    staging_directory = (
        results_root
        / (
            ".staging_"
            + invocation_id
        )
    )

    final_directory = (
        results_root
        / invocation_id
    )

    working_root = (
        load_result
        .run_directory
        / ".inference_work"
    )

    working_directory = (
        ensure_directory(
            working_root
            / invocation_id
        )
    )

    if final_directory.exists():
        raise FileProcessingError(
            "Final inference result directory already exists.",
            error_code="INFERENCE_RESULT_DIRECTORY_ALREADY_EXISTS",
            details={
                "path": str(
                    final_directory
                ),
            },
        )

    if staging_directory.exists():
        raise FileProcessingError(
            "Inference staging directory already exists.",
            error_code="INFERENCE_STAGING_DIRECTORY_ALREADY_EXISTS",
            details={
                "path": str(
                    staging_directory
                ),
            },
        )

    ensure_directory(
        staging_directory
    )

    active_logger.info(
        "Starting pure inference run %s for saved run %s.",
        invocation_id,
        request.run_id,
    )

    history_result = None
    history_committed = False

    try:
        history_context = (
            _build_history_context(
                load_result=(
                    load_result
                ),
                working_directory=(
                    working_directory
                ),
            )
        )

        history_result = (
            run_history_pipeline(
                context=history_context,
                logger=active_logger,
            )
        )

        inference_start = (
            history_result
            .validation
            .new_period_start
        )

        inference_end_exclusive = (
            history_result
            .validation
            .new_period_end_exclusive
        )

        validation_result = (
            validate_inference_handoff(
                load_result=(
                    load_result
                ),
                feature_results=(
                    history_result
                    .feature_results
                ),
            )
        )

        skipped = _skipped_series(
            validation_result
        )

        staged_outputs: list[
            InferenceSeriesOutput
        ] = []

        ready_series = tuple(
            validation_result
            .iter_ready_series()
        )

        for index, series_plan in enumerate(
            ready_series,
            start=1,
        ):
            artifact = (
                series_plan.artifact
            )

            active_logger.info(
                "Running %s inference for KPI=%s site=%s.",
                artifact.model_type,
                artifact.kpi_name,
                artifact.identity,
            )

            result = _run_one_series(
                load_result=(
                    load_result
                ),
                series_plan=(
                    series_plan
                ),
                inference_start=(
                    inference_start
                ),
                inference_end_exclusive=(
                    inference_end_exclusive
                ),
            )

            filename = _safe_output_name(
                index=index,
                series_plan=(
                    series_plan
                ),
            )

            staged_file = (
                staging_directory
                / filename
            )

            _write_csv(
                result.output,
                staged_file,
                role=(
                    "row-level inference "
                    f"output for KPI "
                    f"{artifact.kpi_name!r}, "
                    f"site "
                    f"{artifact.identity!r}"
                ),
            )

            staged_outputs.append(
                _output_summary(
                    model_type=(
                        artifact.model_type
                    ),
                    result_file=(
                        staged_file
                    ),
                    result=result,
                )
            )

            # ``result.output`` is intentionally not accumulated.
            del result
            gc.collect()

        staged_outputs_tuple = tuple(
            staged_outputs
        )

        manifest_file_staged = (
            staging_directory
            / "manifest.json"
        )

        # The outputs are now fully saved to staging. History is still
        # untouched at this point.
        _write_json(
            _manifest_payload(
                load_result=(
                    load_result
                ),
                validation_result=(
                    validation_result
                ),
                invocation_id=(
                    invocation_id
                ),
                inference_start=(
                    inference_start
                ),
                inference_end_exclusive=(
                    inference_end_exclusive
                ),
                staged_outputs=(
                    staged_outputs_tuple
                ),
                skipped=(
                    skipped
                ),
                history_committed=False,
            ),
            manifest_file_staged,
            role="staged inference manifest",
        )

        history_commit = (
            commit_history_pipeline(
                history_result
            )
        )
        history_committed = True

        # Update staged manifest only after commit succeeds.
        _write_json(
            _manifest_payload(
                load_result=(
                    load_result
                ),
                validation_result=(
                    validation_result
                ),
                invocation_id=(
                    invocation_id
                ),
                inference_start=(
                    inference_start
                ),
                inference_end_exclusive=(
                    inference_end_exclusive
                ),
                staged_outputs=(
                    staged_outputs_tuple
                ),
                skipped=(
                    skipped
                ),
                history_committed=True,
            ),
            manifest_file_staged,
            role="committed inference manifest",
        )

        _publish_staging_directory(
            staging_directory=(
                staging_directory
            ),
            final_directory=(
                final_directory
            ),
        )

        final_outputs = tuple(
            InferenceSeriesOutput(
                model_type=(
                    item.model_type
                ),
                kpi_name=(
                    item.kpi_name
                ),
                series_key=(
                    item.series_key
                ),
                identity=(
                    item.identity
                ),
                result_file=(
                    final_directory
                    / item
                    .result_file
                    .name
                ),
                total_window_rows=(
                    item
                    .total_window_rows
                ),
                scorable_rows=(
                    item.scorable_rows
                ),
                unscorable_rows=(
                    item
                    .unscorable_rows
                ),
                anomaly_count=(
                    item.anomaly_count
                ),
                anomaly_rate=(
                    item.anomaly_rate
                ),
                model_anomaly_count=(
                    item
                    .model_anomaly_count
                ),
                lower_anomaly_count=(
                    item
                    .lower_anomaly_count
                ),
                upper_anomaly_count=(
                    item
                    .upper_anomaly_count
                ),
            )
            for item in staged_outputs_tuple
        )

        active_logger.info(
            "Completed pure inference run %s: %d completed, %d skipped.",
            invocation_id,
            len(
                final_outputs
            ),
            len(
                skipped
            ),
        )

        return InferenceRunnerResult(
            run_id=(
                request.run_id
            ),
            model_type=model_type,
            invocation_id=(
                invocation_id
            ),
            inference_start=(
                pd.Timestamp(
                    inference_start
                )
            ),
            inference_end_exclusive=(
                pd.Timestamp(
                    inference_end_exclusive
                )
            ),
            output_directory=(
                final_directory
            ),
            manifest_file=(
                final_directory
                / "manifest.json"
            ),
            total_saved_series=(
                validation_result
                .total_saved_series
            ),
            completed_series=len(
                final_outputs
            ),
            skipped_series_count=len(
                skipped
            ),
            unmodelled_part_c_series=(
                validation_result
                .unmodelled_part_c_series
            ),
            outputs=(
                final_outputs
            ),
            skipped_series=(
                skipped
            ),
            history_commit=(
                history_commit
            ),
        )

    except Exception:
        # If publication did not happen, staged outputs are not considered a
        # completed inference run. Persistent history is guaranteed unchanged
        # unless commit_history_pipeline itself had already succeeded.
        if (
            not history_committed
            and staging_directory.exists()
        ):
            shutil.rmtree(
                staging_directory,
                ignore_errors=True,
            )

        # Once history has committed it cannot be safely rolled back here.
        # If publication fails after that point, retain the staged outputs for
        # recovery rather than deleting the only completed result artifacts.
        raise

    finally:
        # History working files are temporary and can be reconstructed.
        if working_directory.exists():
            shutil.rmtree(
                working_directory,
                ignore_errors=True,
            )

        if history_result is not None:
            del history_result

        gc.collect()


__all__ = [
    "InferenceRunnerResult",
    "InferenceSeriesOutput",
    "SkippedInferenceSeries",
    "run_inference_pipeline",
]