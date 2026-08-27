"""
Temporary end-to-end workflow used to test Parts A through E.

Intended project location::

    app/workflows/test_pipeline_workflow_part_E.py

Training endpoint flow
----------------------
A -> B -> C -> D
    -> bootstrap a filesystem-backed saved inference run

Inference endpoint flow
-----------------------
temporary run_directory + production InferenceRequest
    -> app.ml.inference.run_inference_pipeline(...)

No database is used here. The later inference-suite scripts resolve the
completed ``run_directory`` from the existing Part-D suite root and submit it
to the inference endpoint explicitly.

The Part-E ML runner itself remains unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from math import ceil
from pathlib import Path
import re
import shutil
from typing import Any, Mapping

import pandas as pd
from pydantic import BaseModel

from app.ml.eda.runner import (
    EDARunnerResult,
    run_eda_pipeline,
)
from app.ml.feature_engineering import (
    FeatureEngineeringRunnerResult,
    run_feature_engineering_pipeline,
)
from app.ml.inference import (
    InferenceRunnerResult,
    run_inference_pipeline,
)
from app.ml.preprocessing.runner import (
    PreprocessingRunnerResult,
    run_preprocessing_pipeline,
)
from app.ml.training import (
    PartDTrainingResult,
    run_training,
)
from app.schemas.feature_engineering import FeatureEngineeringConfig
from app.schemas.inference.isolation_forest import (
    IsolationForestInferenceConfig,
    IsolationForestKPIInferenceConfig,
    IsolationForestSeriesInferenceConfig,
)
from app.schemas.inference.request import (
    INFERENCE_CONFIG_FILE_NAME,
    SavedDatasetContract,
    SavedFeatureEngineeringContract,
    SavedHistoryContract,
    SavedInferenceContract,
    SavedPreprocessingContract,
    SavedStorageContract,
)
from app.schemas.inference.xgboost import (
    XGBoostInferenceConfig,
    XGBoostKPIInferenceConfig,
    XGBoostSeriesInferenceConfig,
)
from app.schemas.test_pipeline_part_E import (
    PipelineRunRequest,
    TestInferenceRunRequest,
    TrainingConfig,
)
from app.services.file_service import (
    ensure_directory,
    read_csv_file,
    sanitise_filename,
    write_csv_file,
)


INFERENCE_STATE_DIRECTORY_NAME = "inference_state"
INFERENCE_HISTORY_DIRECTORY_NAME = "history"
INFERENCE_CACHE_DIRECTORY_NAME = "cache"
INFERENCE_RESULTS_DIRECTORY_NAME = "inference_results"


@dataclass(frozen=True, slots=True)
class KPIWorkflowResult:
    """Store Part B, Part C, and Part D results for one KPI."""

    kpi_name: str
    preprocessing_result: PreprocessingRunnerResult
    feature_engineering_result: FeatureEngineeringRunnerResult
    training_result: PartDTrainingResult


@dataclass(frozen=True, slots=True)
class InferenceBootstrapResult:
    """Saved filesystem state created for future Part-E calls."""

    run_id: str
    run_directory: Path
    config_file: Path

    state_directory: Path

    raw_history_file: Path
    raw_cache_file: Path

    processed_history_by_kpi: dict[str, Path]
    processed_cache_by_kpi: dict[str, Path]

    inference_results_directory: Path

    history_contract: SavedHistoryContract


@dataclass(frozen=True, slots=True)
class TestPipelineWorkflowResult:
    """Store the complete temporary A/B/C/D result plus Part-E saved state."""

    output_directory: Path
    eda_result: EDARunnerResult
    training_config: TrainingConfig
    kpi_results: tuple[KPIWorkflowResult, ...]
    inference_bootstrap: InferenceBootstrapResult


def _json_safe(
    value: Any,
) -> Any:
    """
    Convert project objects to JSON-safe values.

    Paths are deliberately serialized with POSIX separators because the saved
    inference schema requires portable run-relative paths.
    """

    if value is None:
        return None

    if isinstance(
        value,
        BaseModel,
    ):
        return _json_safe(
            value.model_dump(
                mode="python"
            )
        )

    if isinstance(
        value,
        Path,
    ):
        return value.as_posix()

    if isinstance(
        value,
        datetime,
    ):
        return value.isoformat()

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (tuple, list),
    ):
        return [
            _json_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        (set, frozenset),
    ):
        return [
            _json_safe(item)
            for item in sorted(
                value,
                key=lambda item: str(item),
            )
        ]

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    raise TypeError(
        "Unsupported JSON value while saving temporary inference state: "
        f"{type(value).__name__}"
    )


def _write_json(
    path: Path,
    payload: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            _json_safe(
                payload
            ),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _load_json_object(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            f"Expected JSON object in saved Part D artifact: {path}"
        )

    return payload


def _derive_run_id(
    *,
    requested_run_id: str | None,
    output_root: Path,
) -> str:
    if requested_run_id is not None:
        return requested_run_id

    candidate = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        output_root.name.strip(),
    ).strip(
        "._-"
    )

    if not candidate:
        candidate = "temporary_run"

    if not candidate[0].isalnum():
        candidate = (
            "run_"
            + candidate
        )

    return candidate


def _relative_posix(
    *,
    path: Path,
    run_root: Path,
) -> str:
    resolved_path = path.expanduser().resolve()
    resolved_root = run_root.expanduser().resolve()

    try:
        relative = resolved_path.relative_to(
            resolved_root
        )
    except ValueError as error:
        raise ValueError(
            "A temporary inference artifact is outside the saved run root. "
            f"path={resolved_path}, root={resolved_root}"
        ) from error

    return relative.as_posix()


def _copy_csv(
    *,
    source: Path,
    destination: Path,
) -> pd.DataFrame:
    dataframe = read_csv_file(
        source
    )

    ensure_directory(
        destination.parent
    )

    write_csv_file(
        dataframe,
        destination,
    )

    return dataframe


def _cache_dataframe(
    dataframe: pd.DataFrame,
    *,
    timestamp_column: str,
    cache_days: int,
    bucket_minutes: int,
    role: str,
) -> pd.DataFrame:
    if timestamp_column not in dataframe.columns:
        raise ValueError(
            f"{role} does not contain timestamp column {timestamp_column!r}."
        )

    working = dataframe.copy()

    parsed = pd.to_datetime(
        working[timestamp_column],
        errors="coerce",
        utc=True,
    )

    if bool(
        parsed.isna().any()
    ):
        raise ValueError(
            f"{role} contains invalid timestamps while creating initial "
            "Part-E cache state."
        )

    maximum = pd.Timestamp(
        parsed.max()
    )

    cutoff = (
        maximum
        - pd.Timedelta(
            days=cache_days
        )
        + pd.Timedelta(
            minutes=bucket_minutes
        )
    )

    cached = working.loc[
        parsed.ge(
            cutoff
        )
        & parsed.le(
            maximum
        )
    ].copy()

    if cached.empty:
        raise ValueError(
            f"{role} produced an empty initial Part-E cache."
        )

    cached[
        timestamp_column
    ] = parsed.loc[
        cached.index
    ]

    return cached.reset_index(
        drop=True
    )


def _part_b_history_days(
    request: PipelineRunRequest,
) -> int:
    config = request.preprocessing

    return max(
        1,
        int(
            config
            .phase_2
            .recent_same_type_lookback_days
        ),
        int(
            config
            .phase_2
            .last_week_lookback_days
        ),
        int(
            config
            .phase_3
            .maximum_lookback_days
        ),
        (
            int(
                config
                .phase_4
                .minimum_history_days
            )
            if config.phase_4.enabled
            else 0
        ),
    )


def _build_history_contract(
    *,
    request: PipelineRunRequest,
    preprocessing_result: PreprocessingRunnerResult,
    feature_result: FeatureEngineeringRunnerResult,
) -> SavedHistoryContract:
    resolved_b = (
        preprocessing_result
        .resolved_frequency
    )

    resolved_c = (
        feature_result
        .resolved_config
    )

    points_per_day = int(
        resolved_c.points_per_day
    )

    if points_per_day <= 0:
        raise ValueError(
            "Part C resolved points_per_day must be positive."
        )

    part_b_days = _part_b_history_days(
        request
    )

    part_b_points = max(
        1,
        part_b_days
        * points_per_day,
    )

    if request.preprocessing.phase_4.enabled:
        prophet_points = (
            resolved_b
            .prophet_minimum_history_points
        )

        if prophet_points is None:
            raise ValueError(
                "Enabled Part B Prophet Phase 4 did not resolve minimum "
                "history points."
            )

        part_b_points = max(
            part_b_points,
            int(
                prophet_points
            ),
        )

    part_c_points = int(
        resolved_c.minimum_history_points
    )

    maximum_lookback_points = int(
        resolved_c.maximum_lookback_points
    )

    final_required_points = max(
        part_b_points,
        part_c_points,
        maximum_lookback_points + 1,
    )

    final_required_days = max(
        part_b_days,
        1,
        ceil(
            final_required_points
            / points_per_day
        ),
        ceil(
            (
                maximum_lookback_points
                + 1
            )
            / points_per_day
        ),
    )

    max_gap_days = (
        request
        .inference_bootstrap
        .max_gap_days
    )

    cache_days = (
        final_required_days
        + max_gap_days
    )

    return SavedHistoryContract(
        part_b_required_history_points=(
            part_b_points
        ),
        part_b_required_history_days=(
            part_b_days
        ),
        part_c_required_history_points=(
            part_c_points
        ),
        part_c_maximum_lookback_points=(
            maximum_lookback_points
        ),
        final_required_history_points=(
            final_required_points
        ),
        final_required_history_days=(
            final_required_days
        ),
        cache_days=(
            cache_days
        ),
        max_gap_days=(
            max_gap_days
        ),
        max_gap_ratio=(
            request
            .inference_bootstrap
            .max_gap_ratio
        ),
        late_data_reconstruction_enabled=(
            request
            .inference_bootstrap
            .late_data_reconstruction_enabled
        ),
    )


def _same_resolved_contracts(
    kpi_results: tuple[
        KPIWorkflowResult,
        ...,
    ],
) -> tuple[
    Any,
    Any,
]:
    if not kpi_results:
        raise ValueError(
            "Cannot bootstrap Part-E state without completed KPI results."
        )

    first_b = (
        kpi_results[
            0
        ]
        .preprocessing_result
        .resolved_frequency
    )

    first_c = (
        kpi_results[
            0
        ]
        .feature_engineering_result
        .resolved_config
    )

    for item in kpi_results[
        1:
    ]:
        if (
            item
            .preprocessing_result
            .resolved_frequency
            != first_b
        ):
            raise ValueError(
                "All KPI Part B resolved contracts must match before one "
                "saved inference run can be created."
            )

        if (
            item
            .feature_engineering_result
            .resolved_config
            != first_c
        ):
            raise ValueError(
                "All KPI Part C resolved contracts must match before one "
                "saved inference run can be created."
            )

    return (
        first_b,
        first_c,
    )


def _series_contract_payloads(
    *,
    training_result: PartDTrainingResult,
    run_root: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    tuple[str, ...],
    str | None,
]:
    """
    Read the finalized Part-D metadata/decision artifacts for one KPI.

    Returns:
        series payload by series_key,
        ordered model feature columns,
        target column.
    """

    output_paths = (
        training_result
        .output_paths
    )

    metadata = _load_json_object(
        output_paths.metadata_file
    )

    decision_settings = _load_json_object(
        output_paths.decision_settings_file
    )

    raw_models = metadata.get(
        "series_models"
    )

    if not isinstance(
        raw_models,
        dict,
    ) or not raw_models:
        raise ValueError(
            "Part D metadata does not contain saved series_models."
        )

    if set(
        raw_models
    ) != set(
        decision_settings
    ):
        raise ValueError(
            "Part D metadata and decision-settings artifacts contain "
            "different series keys."
        )

    training_relative = Path(
        _relative_posix(
            path=output_paths.training_directory,
            run_root=run_root,
        )
    )

    result: dict[
        str,
        dict[str, Any],
    ] = {}

    for series_key, raw_model in raw_models.items():
        if not isinstance(
            raw_model,
            dict,
        ):
            raise ValueError(
                f"Invalid Part D series metadata for {series_key!r}."
            )

        raw_decision = decision_settings[
            series_key
        ]

        if not isinstance(
            raw_decision,
            dict,
        ):
            raise ValueError(
                f"Invalid Part D decision artifact for {series_key!r}."
            )

        identity = raw_model.get(
            "identity"
        )

        if not isinstance(
            identity,
            dict,
        ):
            identity = raw_decision.get(
                "identity"
            )

        if not isinstance(
            identity,
            dict,
        ):
            raise ValueError(
                f"Saved series {series_key!r} does not contain an identity."
            )

        model_file = raw_model.get(
            "model_file"
        )

        if not model_file:
            raise ValueError(
                f"Saved series {series_key!r} does not contain model_file."
            )

        portable_model_file = (
            training_relative
            / Path(
                str(
                    model_file
                )
            )
        ).as_posix()

        result[
            str(
                series_key
            )
        ] = {
            "identity": identity,
            "model_file": (
                portable_model_file
            ),
            "decision_settings": (
                raw_decision.get(
                    "decision_settings"
                )
            ),
        }

    validation = (
        training_result
        .validation_result
    )

    return (
        result,
        tuple(
            validation
            .model_feature_columns
        ),
        validation.target_column,
    )


def _build_model_contract(
    *,
    model_type: str,
    threshold_direction: str,
    kpi_results: tuple[
        KPIWorkflowResult,
        ...,
    ],
    run_root: Path,
) -> (
    XGBoostInferenceConfig
    | IsolationForestInferenceConfig
):
    if model_type == "xgboost":
        kpis: dict[
            str,
            XGBoostKPIInferenceConfig,
        ] = {}

        for item in kpi_results:
            (
                series_payloads,
                model_features,
                target_column,
            ) = _series_contract_payloads(
                training_result=(
                    item.training_result
                ),
                run_root=run_root,
            )

            if target_column is None:
                raise ValueError(
                    "Saved XGBoost training result has no target column."
                )

            series = {
                series_key: (
                    XGBoostSeriesInferenceConfig
                    .model_validate(
                        payload
                    )
                )
                for series_key, payload
                in series_payloads.items()
            }

            kpis[
                item.kpi_name
            ] = XGBoostKPIInferenceConfig(
                model_feature_columns=(
                    model_features
                ),
                target_column=(
                    target_column
                ),
                series=series,
            )

        return XGBoostInferenceConfig(
            threshold_direction=(
                threshold_direction
            ),
            kpis=kpis,
        )

    if model_type == "isolation_forest":
        kpis_if: dict[
            str,
            IsolationForestKPIInferenceConfig,
        ] = {}

        for item in kpi_results:
            (
                series_payloads,
                model_features,
                _,
            ) = _series_contract_payloads(
                training_result=(
                    item.training_result
                ),
                run_root=run_root,
            )

            series_if = {
                series_key: (
                    IsolationForestSeriesInferenceConfig
                    .model_validate(
                        payload
                    )
                )
                for series_key, payload
                in series_payloads.items()
            }

            kpis_if[
                item.kpi_name
            ] = IsolationForestKPIInferenceConfig(
                model_feature_columns=(
                    model_features
                ),
                series=series_if,
            )

        return IsolationForestInferenceConfig(
            threshold_direction=(
                threshold_direction
            ),
            kpis=kpis_if,
        )

    raise ValueError(
        f"Unsupported model_type while building Part-E contract: {model_type}"
    )


def _bootstrap_inference_state(
    *,
    request: PipelineRunRequest,
    output_root: Path,
    eda_result: EDARunnerResult,
    feature_config: FeatureEngineeringConfig,
    kpi_results: tuple[
        KPIWorkflowResult,
        ...,
    ],
) -> InferenceBootstrapResult:
    """
    Create the saved filesystem state consumed by the real Part-E runner.

    This is temporary test-pipeline responsibility only. In the final backend,
    the normal training/run service can own equivalent saved-run registration
    and persistence.
    """

    if (
        output_root
        / INFERENCE_CONFIG_FILE_NAME
    ).exists():
        raise FileExistsError(
            "Temporary saved inference configuration already exists:\n"
            f"  {output_root / INFERENCE_CONFIG_FILE_NAME}"
        )

    run_id = _derive_run_id(
        requested_run_id=(
            request.run_id
        ),
        output_root=(
            output_root
        ),
    )

    (
        resolved_b,
        resolved_c,
    ) = _same_resolved_contracts(
        kpi_results
    )

    history_contract = (
        _build_history_contract(
            request=request,
            preprocessing_result=(
                kpi_results[
                    0
                ]
                .preprocessing_result
            ),
            feature_result=(
                kpi_results[
                    0
                ]
                .feature_engineering_result
            ),
        )
    )

    state_directory = ensure_directory(
        output_root
        / INFERENCE_STATE_DIRECTORY_NAME
    )

    full_history_directory = ensure_directory(
        state_directory
        / INFERENCE_HISTORY_DIRECTORY_NAME
    )

    cache_directory = ensure_directory(
        state_directory
        / INFERENCE_CACHE_DIRECTORY_NAME
    )

    processed_history_directory = ensure_directory(
        full_history_directory
        / "processed"
    )

    processed_cache_directory = ensure_directory(
        cache_directory
        / "processed"
    )

    raw_history_file = (
        full_history_directory
        / "raw_history.csv"
    )

    raw_cache_file = (
        cache_directory
        / "raw_cache.csv"
    )

    raw_dataframe = _copy_csv(
        source=(
            Path(
                request
                .dataset
                .input_file
            )
            .expanduser()
            .resolve()
        ),
        destination=(
            raw_history_file
        ),
    )

    bucket_minutes = int(
        resolved_c.bucket_minutes
    )

    raw_cache_dataframe = (
        _cache_dataframe(
            raw_dataframe,
            timestamp_column=(
                request
                .dataset
                .columns
                .timestamp_column
            ),
            cache_days=(
                history_contract
                .cache_days
            ),
            bucket_minutes=(
                bucket_minutes
            ),
            role=(
                "initial raw history"
            ),
        )
    )

    write_csv_file(
        raw_cache_dataframe,
        raw_cache_file,
    )

    processed_history_by_kpi: dict[
        str,
        Path,
    ] = {}

    processed_cache_by_kpi: dict[
        str,
        Path,
    ] = {}

    for index, item in enumerate(
        kpi_results,
        start=1,
    ):
        safe_kpi_name = (
            sanitise_filename(
                item.kpi_name
            )
            or f"kpi_{index}"
        )

        processed_history_file = (
            processed_history_directory
            / f"{safe_kpi_name}.csv"
        )

        processed_cache_file = (
            processed_cache_directory
            / f"{safe_kpi_name}.csv"
        )

        processed_dataframe = _copy_csv(
            source=(
                item
                .preprocessing_result
                .final_dataset_file
            ),
            destination=(
                processed_history_file
            ),
        )

        processed_cache_dataframe = (
            _cache_dataframe(
                processed_dataframe,
                timestamp_column=(
                    request
                    .dataset
                    .columns
                    .timestamp_column
                ),
                cache_days=(
                    history_contract
                    .cache_days
                ),
                bucket_minutes=(
                    bucket_minutes
                ),
                role=(
                    "initial processed history "
                    f"for KPI {item.kpi_name!r}"
                ),
            )
        )

        write_csv_file(
            processed_cache_dataframe,
            processed_cache_file,
        )

        processed_history_by_kpi[
            item.kpi_name
        ] = processed_history_file

        processed_cache_by_kpi[
            item.kpi_name
        ] = processed_cache_file

    inference_results_directory = ensure_directory(
        output_root
        / INFERENCE_RESULTS_DIRECTORY_NAME
    )

    kpi_names = tuple(
        item.kpi_name
        for item in kpi_results
    )

    model_contract = _build_model_contract(
        model_type=(
            request
            .training
            .model_type
        ),
        threshold_direction=(
            request
            .training
            .threshold_direction
        ),
        kpi_results=(
            kpi_results
        ),
        run_root=(
            output_root
        ),
    )

    saved_contract = SavedInferenceContract(
        run_id=run_id,
        created_at_utc=datetime.now(
            timezone.utc
        ),
        model_type=(
            request
            .training
            .model_type
        ),
        dataset=SavedDatasetContract(
            columns=(
                request
                .dataset
                .columns
            ),
            bucket=(
                eda_result
                .preparation
                .bucket
            ),
            kpi_names=(
                kpi_names
            ),
            single_kpi_name=(
                request
                .dataset
                .single_kpi_name
            ),
        ),
        preprocessing=SavedPreprocessingContract(
            config=(
                request.preprocessing
            ),
            resolved=(
                resolved_b
            ),
        ),
        feature_engineering=(
            SavedFeatureEngineeringContract(
                config=(
                    feature_config
                ),
                resolved=(
                    resolved_c
                ),
            )
        ),
        history=(
            history_contract
        ),
        storage=SavedStorageContract(
            raw_history_file=(
                _relative_posix(
                    path=raw_history_file,
                    run_root=output_root,
                )
            ),
            raw_cache_file=(
                _relative_posix(
                    path=raw_cache_file,
                    run_root=output_root,
                )
            ),
            processed_history_by_kpi={
                kpi_name: _relative_posix(
                    path=path,
                    run_root=output_root,
                )
                for kpi_name, path
                in processed_history_by_kpi.items()
            },
            processed_cache_by_kpi={
                kpi_name: _relative_posix(
                    path=path,
                    run_root=output_root,
                )
                for kpi_name, path
                in processed_cache_by_kpi.items()
            },
            inference_results_directory=(
                _relative_posix(
                    path=inference_results_directory,
                    run_root=output_root,
                )
            ),
        ),
        model=(
            model_contract
        ),
    )

    config_file = (
        output_root
        / INFERENCE_CONFIG_FILE_NAME
    )

    _write_json(
        config_file,
        saved_contract,
    )

    return InferenceBootstrapResult(
        run_id=run_id,
        run_directory=(
            output_root
        ),
        config_file=(
            config_file
        ),
        state_directory=(
            state_directory
        ),
        raw_history_file=(
            raw_history_file
        ),
        raw_cache_file=(
            raw_cache_file
        ),
        processed_history_by_kpi=(
            processed_history_by_kpi
        ),
        processed_cache_by_kpi=(
            processed_cache_by_kpi
        ),
        inference_results_directory=(
            inference_results_directory
        ),
        history_contract=(
            history_contract
        ),
    )


def run_test_pipeline_workflow(
    *,
    request: PipelineRunRequest,
    logger,
) -> TestPipelineWorkflowResult:
    """
    Run A -> B -> C -> D, then create the initial saved Part-E state.

    Existing Part-D execution behavior remains unchanged. The only addition
    after every KPI has trained successfully is the inference bootstrap step.
    """

    output_root = ensure_directory(
        Path(
            request.output_directory
        ).expanduser().resolve()
    )

    eda_root = ensure_directory(
        output_root
        / "eda"
    )

    preprocessing_root = ensure_directory(
        output_root
        / "preprocessing"
    )

    feature_engineering_root = ensure_directory(
        output_root
        / "feature_engineering"
    )

    part_d_root = ensure_directory(
        output_root
        / "part_d"
    )

    feature_config = (
        request.feature_engineering
        if request.feature_engineering is not None
        else FeatureEngineeringConfig(
            default_profile=(
                request
                .training
                .model_type
            )
        )
    )

    resolved_tuning_method = (
        request
        .training
        .hyperparameter_tuning_method
        if (
            request
            .training
            .hyperparameter_tuning_enabled
        )
        else None
    )

    logger.info(
        "Starting temporary Part A->D training pipeline with Part-E state "
        "bootstrap. Model=%s, tuning=%s, method=%s, output=%s.",
        request.training.model_type,
        request.training.hyperparameter_tuning_enabled,
        resolved_tuning_method,
        output_root,
    )

    eda_result = run_eda_pipeline(
        config=request.dataset,
        logger=logger,
        output_directory=(
            eda_root
        ),
    )

    kpi_workflow_results: list[
        KPIWorkflowResult
    ] = []

    for index, kpi_result in enumerate(
        eda_result.kpi_results,
        start=1,
    ):
        raw_kpi_name = str(
            kpi_result
            .prepared_dataset
            .kpi_name
        ).strip()

        safe_kpi_name = (
            sanitise_filename(
                raw_kpi_name
            )
            or f"kpi_{index}"
        )

        kpi_preprocessing_output_directory = (
            ensure_directory(
                preprocessing_root
                / safe_kpi_name
            )
        )

        preprocessing_result = (
            run_preprocessing_pipeline(
                kpi_result=(
                    kpi_result
                ),
                output_directory=(
                    kpi_preprocessing_output_directory
                ),
                dataset_config=(
                    request.dataset
                ),
                preprocessing_config=(
                    request.preprocessing
                ),
                bucket=(
                    eda_result
                    .preparation
                    .bucket
                ),
                logger=logger,
            )
        )

        kpi_feature_engineering_output_directory = (
            ensure_directory(
                feature_engineering_root
                / safe_kpi_name
            )
        )

        feature_engineering_result = (
            run_feature_engineering_pipeline(
                preprocessing_result=(
                    preprocessing_result
                ),
                output_directory=(
                    kpi_feature_engineering_output_directory
                ),
                dataset_config=(
                    request.dataset
                ),
                feature_config=(
                    feature_config
                ),
                bucket=(
                    eda_result
                    .preparation
                    .bucket
                ),
                logger=logger,
            )
        )

        kpi_part_d_output_directory = (
            ensure_directory(
                part_d_root
                / safe_kpi_name
            )
        )

        training_result = run_training(
            model_type=(
                request
                .training
                .model_type
            ),
            input_file=(
                feature_engineering_result
                .final_dataset_file
            ),
            output_directory=(
                kpi_part_d_output_directory
            ),
            dataset_columns=(
                request
                .dataset
                .columns
            ),
            feature_config=(
                feature_engineering_result
                .resolved_config
            ),
            minimum_history_points=(
                feature_engineering_result
                .resolved_config
                .minimum_history_points
            ),
            training_config=(
                request.training
            ),
            metadata={
                "kpi_name": (
                    raw_kpi_name
                ),
                "part_c_output_directory": str(
                    feature_engineering_result
                    .output_directory
                ),
                "part_c_final_dataset": str(
                    feature_engineering_result
                    .final_dataset_file
                ),
                "hyperparameter_tuning_enabled": (
                    request
                    .training
                    .hyperparameter_tuning_enabled
                ),
                "hyperparameter_tuning_method": (
                    resolved_tuning_method
                ),
            },
        )

        kpi_workflow_results.append(
            KPIWorkflowResult(
                kpi_name=(
                    raw_kpi_name
                ),
                preprocessing_result=(
                    preprocessing_result
                ),
                feature_engineering_result=(
                    feature_engineering_result
                ),
                training_result=(
                    training_result
                ),
            )
        )

    kpi_results = tuple(
        kpi_workflow_results
    )

    inference_bootstrap = (
        _bootstrap_inference_state(
            request=request,
            output_root=(
                output_root
            ),
            eda_result=(
                eda_result
            ),
            feature_config=(
                feature_config
            ),
            kpi_results=(
                kpi_results
            ),
        )
    )

    result = TestPipelineWorkflowResult(
        output_directory=(
            output_root
        ),
        eda_result=(
            eda_result
        ),
        training_config=(
            request.training
        ),
        kpi_results=(
            kpi_results
        ),
        inference_bootstrap=(
            inference_bootstrap
        ),
    )

    logger.info(
        "Temporary Part A->D training completed and Part-E state prepared. "
        "run_id=%s, KPIs=%d, run_directory=%s.",
        result
        .inference_bootstrap
        .run_id,
        len(
            result.kpi_results
        ),
        result
        .inference_bootstrap
        .run_directory,
    )

    return result


def run_test_inference_workflow(
    *,
    request: TestInferenceRunRequest,
    logger,
) -> InferenceRunnerResult:
    """
    Run the real Part-E pipeline using an explicitly supplied saved-run folder.

    The later inference-suite scripts are responsible for locating this folder
    from the existing Part-D suite root.
    """

    run_directory = Path(
        request.run_directory
    ).expanduser().resolve()

    if not run_directory.is_dir():
        raise FileNotFoundError(
            "Temporary saved training run does not exist:\n"
            f"  {run_directory}"
        )

    config_file = (
        run_directory
        / INFERENCE_CONFIG_FILE_NAME
    )

    if not config_file.is_file():
        raise FileNotFoundError(
            "Temporary saved training run is not Part-E ready; "
            "inference_config.json is missing:\n"
            f"  {config_file}"
        )

    pure_request = (
        request
        .to_inference_request()
    )

    return run_inference_pipeline(
        request=(
            pure_request
        ),
        run_directory=(
            run_directory
        ),
        logger_override=(
            logger
        ),
    )


__all__ = [
    "InferenceBootstrapResult",
    "KPIWorkflowResult",
    "TestPipelineWorkflowResult",
    "run_test_inference_workflow",
    "run_test_pipeline_workflow",
]
