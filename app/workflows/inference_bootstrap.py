"""
Production inference-bootstrap workflow.

This stage converts a completed Part D training attempt into the self-contained
saved-run contract consumed by Part E.

Unlike the normal A->B->C->D handoff chain, bootstrap creates persistent state:

    D completed
        ↓
    inference_bootstrap
        ↓
    <bootstrap staging root>/
        ├── inference_config.json
        ├── history/
        ├── cache/
        ├── models/
        └── inference_results/
        ↓
    Run Management atomically publishes the staging root to:
        runs/<run_id>/inference/<attempt_id>/

The implementation is derived from the tested A->E integration workflow but is
adapted to the final Run Management architecture: all model/storage paths in
the saved contract are relative to the inference-state root, not to the wider
training attempt directory.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from enum import Enum
from logging import Logger
from math import ceil
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

import pandas as pd

from app.core.exceptions import RunManagementError
from app.run_management import (
    OutputRegistration,
    RunManager,
    get_successful_execution_id,
    resolve_stage_result,
    resolve_successful_execution_record,
)
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
from app.schemas.workflows import (
    FeatureEngineeringStageRequest,
    PreprocessingStageRequest,
    StageExecutionSummary,
    StandaloneStageResponse,
    TrainingStageRequest,
    WorkflowErrorDetail,
)
from app.services.file_service import (
    ensure_directory,
    read_csv_file,
    read_json_file,
    write_csv_file,
    write_json_file,
)
from app.services.run_manifest_service import (
    convert_exception_to_record,
    load_run_manifest,
)

from .common import (
    load_json_object,
    normalize_kpi_name,
    resolve_feature_engineering_config,
    safe_kpi_directory_name,
    to_json_safe,
)
from .handoffs import (
    PartBHandoff,
    PartCHandoff,
    load_part_b_handoff,
    load_part_c_handoff,
)
from .part_d import PART_D_SNAPSHOT_VERSION


BOOTSTRAP_RESULT_VERSION = "1.0"


def _error_detail(error: Exception) -> WorkflowErrorDetail:
    return WorkflowErrorDetail.model_validate(
        convert_exception_to_record(error)
    )


def _relative_posix(
    *,
    path: str | Path,
    root: str | Path,
) -> str:
    resolved_path = Path(
        path
    ).expanduser().resolve()
    resolved_root = Path(
        root
    ).expanduser().resolve()

    try:
        relative = resolved_path.relative_to(
            resolved_root
        )
    except ValueError as error:
        raise ValueError(
            "An inference-bootstrap artifact is outside the saved-state root. "
            f"path={resolved_path}, root={resolved_root}"
        ) from error

    return relative.as_posix()


def _portable_contract_value(
    value: Any,
) -> Any:
    """
    Convert a validated saved-inference contract into a portable JSON value.

    Pydantic correctly validates saved relative paths and stores them as Path
    objects for runtime use. On Windows, normal JSON serialization of Path can
    reintroduce backslashes. Bootstrap must persist the frozen Part-E contract
    with POSIX separators regardless of the host OS.
    """
    if isinstance(value, Path):
        return value.as_posix()

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Mapping):
        return {
            str(key): _portable_contract_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _portable_contract_value(item)
            for item in value
        ]

    if isinstance(value, (set, frozenset)):
        return [
            _portable_contract_value(item)
            for item in sorted(
                value,
                key=lambda item: str(item),
            )
        ]

    return value


def _portable_saved_contract_payload(
    contract: SavedInferenceContract,
) -> dict[str, Any]:
    """
    Serialize one validated Part-E contract without weakening its schema.

    The payload is round-trip validated through the existing inference schema
    before it is written. This keeps the schema as the source of truth while
    making persistence host-OS independent.
    """
    raw_payload = contract.model_dump(
        mode="python"
    )

    portable = _portable_contract_value(
        raw_payload
    )

    if not isinstance(
        portable,
        dict,
    ):
        raise TypeError(
            "Saved inference contract did not serialize to a JSON object."
        )

    # Critical persistence-boundary check:
    #
    # Require the payload to be fully JSON-native before write_json_file()
    # sees it. This prevents its generic default=str fallback from silently
    # converting values such as sets into strings.
    try:
        serialized = json.dumps(
            portable,
            ensure_ascii=False,
            allow_nan=False,
        )
        json_payload = json.loads(
            serialized
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            "Saved inference contract contains a value that cannot be "
            "persisted as strict portable JSON."
        ) from error

    if not isinstance(
        json_payload,
        dict,
    ):
        raise TypeError(
            "Saved inference contract JSON payload must be an object."
        )

    # Validate the exact JSON representation that will be persisted using the
    # existing unchanged Part-E schema.
    SavedInferenceContract.model_validate(
        json_payload
    )

    return json_payload


def _copy_csv(
    *,
    source: str | Path,
    destination: str | Path,
) -> pd.DataFrame:
    dataframe = read_csv_file(
        source
    )
    destination_path = Path(
        destination
    ).expanduser().resolve()

    ensure_directory(
        destination_path.parent
    )
    write_csv_file(
        dataframe,
        destination_path,
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
    """
    Create the initial Z-day cache exactly as the tested A->E workflow did.
    """
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
    preprocessing_config: Any,
) -> int:
    """Return the maximum day-based history span Part B can consult."""
    return max(
        1,
        int(
            preprocessing_config
            .phase_2
            .recent_same_type_lookback_days
        ),
        int(
            preprocessing_config
            .phase_2
            .last_week_lookback_days
        ),
        int(
            preprocessing_config
            .phase_3
            .maximum_lookback_days
        ),
        (
            int(
                preprocessing_config
                .phase_4
                .minimum_history_days
            )
            if (
                preprocessing_config
                .phase_4
                .enabled
            )
            else 0
        ),
    )


def _same_resolved_contracts(
    *,
    part_b: PartBHandoff,
    part_c: PartCHandoff,
) -> tuple[Any, Any]:
    """Require one common resolved B/C contract across all trained KPIs."""
    if not part_b.preprocessing_results:
        raise ValueError(
            "Cannot bootstrap inference state without Part B KPI results."
        )

    if not part_c.feature_engineering_results:
        raise ValueError(
            "Cannot bootstrap inference state without Part C KPI results."
        )

    b_by_kpi = {
        normalize_kpi_name(
            result.kpi_name
        ): result
        for result in part_b.preprocessing_results
    }
    c_by_kpi = {
        normalize_kpi_name(
            result.kpi_name
        ): result
        for result in part_c.feature_engineering_results
    }

    if set(
        b_by_kpi
    ) != set(
        c_by_kpi
    ):
        raise ValueError(
            "Part B and Part C contain different KPI identities."
        )

    ordered_kpis = tuple(
        b_by_kpi
    )

    first_b = (
        b_by_kpi[
            ordered_kpis[0]
        ]
        .resolved_frequency
    )
    first_c = (
        c_by_kpi[
            ordered_kpis[0]
        ]
        .resolved_config
    )

    for kpi_name in ordered_kpis[1:]:
        if (
            b_by_kpi[
                kpi_name
            ]
            .resolved_frequency
            != first_b
        ):
            raise ValueError(
                "All KPI Part B resolved contracts must match before one "
                "saved inference run can be created."
            )

        if (
            c_by_kpi[
                kpi_name
            ]
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


def _build_history_contract(
    *,
    preprocessing_config: Any,
    resolved_b: Any,
    resolved_c: Any,
    bootstrap_config: Any,
) -> SavedHistoryContract:
    """Freeze the final B+C history/cache requirement for Part E."""
    points_per_day = int(
        resolved_c.points_per_day
    )

    if points_per_day <= 0:
        raise ValueError(
            "Part C resolved points_per_day must be positive."
        )

    part_b_days = _part_b_history_days(
        preprocessing_config
    )

    part_b_points = max(
        1,
        part_b_days
        * points_per_day,
    )

    if (
        preprocessing_config
        .phase_4
        .enabled
    ):
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

    if part_c_points <= 0:
        raise ValueError(
            "Part C minimum_history_points must be positive."
        )

    if maximum_lookback_points < 0:
        raise ValueError(
            "Part C maximum_lookback_points cannot be negative."
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

    max_gap_days = int(
        bootstrap_config.max_gap_days
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
            final_required_days
            + max_gap_days
        ),
        max_gap_days=(
            max_gap_days
        ),
        max_gap_ratio=float(
            bootstrap_config.max_gap_ratio
        ),
        late_data_reconstruction_enabled=bool(
            bootstrap_config
            .late_data_reconstruction_enabled
        ),
    )


def _load_stage_request(
    *,
    manifest: Mapping[str, Any],
    attempt_id: str,
    stage_name: str,
    request_type: type,
) -> Any:
    """Reload the exact request that produced a successful stage execution."""
    execution = resolve_successful_execution_record(
        manifest,
        attempt_id,
        stage_name,
    )

    request_file = execution.get(
        "request_file"
    )

    if not request_file:
        raise ValueError(
            f"Successful {stage_name} execution does not register request_file."
        )

    payload = load_json_object(
        request_file
    )

    return request_type.model_validate(
        payload
    )


def _load_part_d_snapshot(
    result_file: str | Path,
) -> dict[str, Any]:
    payload = load_json_object(
        result_file
    )

    if (
        payload.get(
            "part_d_snapshot_version"
        )
        != PART_D_SNAPSHOT_VERSION
    ):
        raise ValueError(
            "Unsupported persisted Part D snapshot version."
        )

    if payload.get(
        "stage"
    ) != "part_d":
        raise ValueError(
            "The bootstrap prerequisite result is not a Part D snapshot."
        )

    records = payload.get(
        "training_records"
    )

    if (
        not isinstance(
            records,
            list,
        )
        or not records
    ):
        raise ValueError(
            "Part D snapshot contains no training records."
        )

    return payload


def _safe_model_relative_path(
    value: object,
) -> PurePosixPath:
    """
    Normalize a Part D internal model path without permitting package escape.
    """
    text = str(
        value
    ).strip()

    if not text:
        raise ValueError(
            "Part D model_file cannot be blank."
        )

    if (
        PureWindowsPath(
            text
        ).is_absolute()
        or PurePosixPath(
            text.replace(
                "\\",
                "/",
            )
        ).is_absolute()
    ):
        raise ValueError(
            "Part D metadata model_file must be relative to training_directory."
        )

    normalized = PurePosixPath(
        text.replace(
            "\\",
            "/",
        )
    )

    if (
        normalized
        == PurePosixPath(
            "."
        )
        or ".." in normalized.parts
    ):
        raise ValueError(
            "Part D metadata model_file cannot escape training_directory."
        )

    return normalized


def _require_run_owned_source(
    path: str | Path,
    *,
    run_directory: str | Path,
    role: str,
    must_be_file: bool | None,
) -> Path:
    """
    Require a persisted Part D source path to remain under the owning run.

    Part D created these paths originally, but bootstrap reloads them from a
    persisted JSON snapshot. Re-validating containment protects the
    cross-process persistence boundary from a corrupted snapshot.
    """
    resolved = Path(
        path
    ).expanduser().resolve()
    run_root = Path(
        run_directory
    ).expanduser().resolve()

    try:
        resolved.relative_to(
            run_root
        )
    except ValueError as error:
        raise RunManagementError(
            "A persisted Part D artifact is outside the owning run directory.",
            error_code="BOOTSTRAP_PART_D_SOURCE_OUTSIDE_RUN",
            details={
                "role": role,
                "path": str(
                    resolved
                ),
                "run_directory": str(
                    run_root
                ),
            },
        ) from error

    if not resolved.exists():
        raise RunManagementError(
            "A persisted Part D artifact required by bootstrap does not exist.",
            error_code="BOOTSTRAP_PART_D_SOURCE_MISSING",
            details={
                "role": role,
                "path": str(
                    resolved
                ),
            },
        )

    if (
        must_be_file is True
        and not resolved.is_file()
    ):
        raise RunManagementError(
            "Bootstrap expected a Part D file artifact.",
            error_code="BOOTSTRAP_PART_D_SOURCE_TYPE_INVALID",
            details={
                "role": role,
                "path": str(
                    resolved
                ),
            },
        )

    if (
        must_be_file is False
        and not resolved.is_dir()
    ):
        raise RunManagementError(
            "Bootstrap expected a Part D directory artifact.",
            error_code="BOOTSTRAP_PART_D_SOURCE_TYPE_INVALID",
            details={
                "role": role,
                "path": str(
                    resolved
                ),
            },
        )

    return resolved


def _copy_training_package(
    *,
    record: Mapping[str, Any],
    state_root: Path,
    safe_kpi_name: str,
    run_directory: Path,
) -> tuple[Path, Path]:
    """
    Copy one finalized Part D training package into bootstrap state.

    Part E receives only the published inference-state root. Therefore every
    model and model-side artifact it may need must physically live below that
    root rather than merely referencing the training-attempt directory.
    """
    source = _require_run_owned_source(
        record[
            "training_directory"
        ],
        run_directory=run_directory,
        role="Part D training directory",
        must_be_file=False,
    )

    destination = (
        state_root
        / "models"
        / safe_kpi_name
        / "training"
    ).resolve()

    ensure_directory(
        destination.parent
    )

    if destination.exists():
        raise FileExistsError(
            f"Bootstrap model package already exists: {destination}"
        )

    shutil.copytree(
        source,
        destination,
    )

    return (
        source,
        destination,
    )


def _copied_package_file(
    *,
    original_file: str | Path,
    source_training_directory: Path,
    copied_training_directory: Path,
    run_directory: Path,
    role: str,
) -> Path:
    """
    Map one Part D source artifact to the corresponding file in the copied
    self-contained training package.
    """
    source_file = _require_run_owned_source(
        original_file,
        run_directory=run_directory,
        role=role,
        must_be_file=True,
    )

    try:
        relative = source_file.relative_to(
            source_training_directory
        )
    except ValueError as error:
        raise RunManagementError(
            "A Part D artifact is outside its registered training package.",
            error_code="BOOTSTRAP_PART_D_ARTIFACT_OUTSIDE_TRAINING_PACKAGE",
            details={
                "role": role,
                "artifact": str(
                    source_file
                ),
                "training_directory": str(
                    source_training_directory
                ),
            },
        ) from error

    copied_file = (
        copied_training_directory
        / relative
    ).resolve()

    try:
        copied_file.relative_to(
            copied_training_directory
        )
    except ValueError as error:
        raise RunManagementError(
            "A copied Part D artifact escaped the bootstrap training package.",
            error_code="BOOTSTRAP_COPIED_ARTIFACT_OUTSIDE_PACKAGE",
            details={
                "role": role,
                "copied_file": str(
                    copied_file
                ),
            },
        ) from error

    if not copied_file.is_file():
        raise RunManagementError(
            "A required artifact is missing from the copied Part D package.",
            error_code="BOOTSTRAP_COPIED_ARTIFACT_MISSING",
            details={
                "role": role,
                "copied_file": str(
                    copied_file
                ),
            },
        )

    return copied_file


def _series_payloads(
    *,
    record: Mapping[str, Any],
    source_training_directory: Path,
    copied_training_directory: Path,
    state_root: Path,
    run_directory: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    tuple[str, ...],
    str | None,
]:
    """
    Reload copied Part D metadata/decision artifacts and build portable
    per-series inference payloads.

    Reading from the copied package proves that the self-contained state being
    published actually contains the artifacts from which its model contract was
    constructed.
    """
    metadata_file = _copied_package_file(
        original_file=record[
            "metadata_file"
        ],
        source_training_directory=(
            source_training_directory
        ),
        copied_training_directory=(
            copied_training_directory
        ),
        run_directory=run_directory,
        role="Part D metadata file",
    )
    decision_settings_file = (
        _copied_package_file(
            original_file=record[
                "decision_settings_file"
            ],
            source_training_directory=(
                source_training_directory
            ),
            copied_training_directory=(
                copied_training_directory
            ),
            run_directory=run_directory,
            role=(
                "Part D decision-settings file"
            ),
        )
    )

    metadata = read_json_file(
        metadata_file
    )
    decision_settings = read_json_file(
        decision_settings_file
    )

    if not isinstance(
        metadata,
        dict,
    ):
        raise ValueError(
            "Part D metadata artifact must contain a JSON object."
        )

    if not isinstance(
        decision_settings,
        dict,
    ):
        raise ValueError(
            "Part D decision-settings artifact must contain a JSON object."
        )

    raw_models = metadata.get(
        "series_models"
    )

    if (
        not isinstance(
            raw_models,
            dict,
        )
        or not raw_models
    ):
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

        decision_payload = raw_decision.get(
            "decision_settings"
        )

        if not isinstance(
            decision_payload,
            dict,
        ):
            raise ValueError(
                f"Saved series {series_key!r} does not contain valid "
                "decision_settings."
            )

        model_relative = _safe_model_relative_path(
            raw_model.get(
                "model_file"
            )
        )

        copied_model = (
            copied_training_directory
            / Path(
                *model_relative.parts
            )
        ).resolve()

        try:
            copied_model.relative_to(
                copied_training_directory
            )
        except ValueError as error:
            raise ValueError(
                "Copied Part D model path escapes its training package."
            ) from error

        if not copied_model.is_file():
            raise FileNotFoundError(
                "Copied Part D model file does not exist: "
                f"{copied_model}"
            )

        result[
            str(
                series_key
            )
        ] = {
            "identity": identity,
            "model_file": _relative_posix(
                path=copied_model,
                root=state_root,
            ),
            "decision_settings": (
                decision_payload
            ),
        }

    expected_model_files = record.get(
        "model_files"
    )

    if (
        not isinstance(
            expected_model_files,
            list,
        )
        or not expected_model_files
    ):
        raise ValueError(
            "Part D snapshot does not contain saved model_files."
        )

    expected_model_relatives: set[str] = set()

    for model_file in expected_model_files:
        source_model = _require_run_owned_source(
            model_file,
            run_directory=run_directory,
            role="Part D model file",
            must_be_file=True,
        )

        try:
            relative = source_model.relative_to(
                source_training_directory
            )
        except ValueError as error:
            raise RunManagementError(
                "A registered Part D model file is outside its training "
                "package.",
                error_code="BOOTSTRAP_PART_D_MODEL_OUTSIDE_TRAINING_PACKAGE",
                details={
                    "model_file": str(
                        source_model
                    ),
                    "training_directory": str(
                        source_training_directory
                    ),
                },
            ) from error

        expected_model_relatives.add(
            relative.as_posix()
        )

    metadata_model_relatives = {
        _safe_model_relative_path(
            raw_model.get(
                "model_file"
            )
        ).as_posix()
        for raw_model in raw_models.values()
    }

    if (
        metadata_model_relatives
        != expected_model_relatives
    ):
        raise RunManagementError(
            "Part D snapshot model_files do not match metadata series_models.",
            error_code="BOOTSTRAP_PART_D_MODEL_SET_MISMATCH",
            details={
                "snapshot_model_files": sorted(
                    expected_model_relatives
                ),
                "metadata_model_files": sorted(
                    metadata_model_relatives
                ),
            },
        )

    trained_series = int(
        record.get(
            "trained_series",
            -1,
        )
    )

    if (
        trained_series < 1
        or len(
            result
        )
        != trained_series
    ):
        raise RunManagementError(
            "Part D saved series count does not match trained_series.",
            error_code="BOOTSTRAP_PART_D_TRAINED_SERIES_MISMATCH",
            details={
                "trained_series": trained_series,
                "saved_series_count": len(
                    result
                ),
            },
        )

    model_features = tuple(
        str(
            value
        ).strip()
        for value in record.get(
            "model_feature_columns",
            (),
        )
    )

    if (
        not model_features
        or any(
            not value
            for value in model_features
        )
        or len(
            model_features
        )
        != len(
            set(
                model_features
            )
        )
    ):
        raise ValueError(
            "Part D snapshot contains an invalid model feature-column contract."
        )

    target_column = record.get(
        "target_column"
    )

    return (
        result,
        model_features,
        (
            str(
                target_column
            ).strip()
            if target_column is not None
            else None
        ),
    )


def _unique_safe_kpi_names(
    kpi_names: tuple[str, ...],
) -> dict[str, str]:
    """
    Build deterministic filesystem names and prevent sanitization collisions.
    """
    used: set[str] = set()
    mapping: dict[
        str,
        str,
    ] = {}

    for index, kpi_name in enumerate(
        kpi_names,
        start=1,
    ):
        base = safe_kpi_directory_name(
            kpi_name,
            fallback_index=index,
        )
        candidate = base

        if candidate in used:
            candidate = (
                f"{base}_{index}"
            )

        suffix = index

        while candidate in used:
            suffix += 1
            candidate = (
                f"{base}_{suffix}"
            )

        used.add(
            candidate
        )
        mapping[
            kpi_name
        ] = candidate

    return mapping


def _build_model_contract(
    *,
    model_type: str,
    threshold_direction: str,
    training_records: Mapping[
        str,
        Mapping[str, Any],
    ],
    state_root: Path,
    safe_names: Mapping[str, str],
    run_directory: Path,
) -> (
    XGBoostInferenceConfig
    | IsolationForestInferenceConfig
):
    if model_type == "xgboost":
        kpis: dict[
            str,
            XGBoostKPIInferenceConfig,
        ] = {}

        for kpi_name, record in training_records.items():
            (
                source_training,
                copied_training,
            ) = _copy_training_package(
                record=record,
                state_root=state_root,
                safe_kpi_name=safe_names[
                    kpi_name
                ],
                run_directory=run_directory,
            )

            (
                series_payloads,
                model_features,
                target_column,
            ) = _series_payloads(
                record=record,
                source_training_directory=(
                    source_training
                ),
                copied_training_directory=(
                    copied_training
                ),
                state_root=state_root,
                run_directory=run_directory,
            )

            if target_column is None:
                raise ValueError(
                    "Saved XGBoost training result has no target column."
                )

            kpis[
                kpi_name
            ] = XGBoostKPIInferenceConfig(
                model_feature_columns=(
                    model_features
                ),
                target_column=(
                    target_column
                ),
                series={
                    series_key: (
                        XGBoostSeriesInferenceConfig
                        .model_validate(
                            payload
                        )
                    )
                    for series_key, payload
                    in series_payloads.items()
                },
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

        for kpi_name, record in training_records.items():
            (
                source_training,
                copied_training,
            ) = _copy_training_package(
                record=record,
                state_root=state_root,
                safe_kpi_name=safe_names[
                    kpi_name
                ],
                run_directory=run_directory,
            )

            (
                series_payloads,
                model_features,
                _,
            ) = _series_payloads(
                record=record,
                source_training_directory=(
                    source_training
                ),
                copied_training_directory=(
                    copied_training
                ),
                state_root=state_root,
                run_directory=run_directory,
            )

            kpis_if[
                kpi_name
            ] = IsolationForestKPIInferenceConfig(
                model_feature_columns=(
                    model_features
                ),
                series={
                    series_key: (
                        IsolationForestSeriesInferenceConfig
                        .model_validate(
                            payload
                        )
                    )
                    for series_key, payload
                    in series_payloads.items()
                },
            )

        return IsolationForestInferenceConfig(
            threshold_direction=(
                threshold_direction
            ),
            kpis=kpis_if,
        )

    raise ValueError(
        "Unsupported model_type while building Part-E contract: "
        f"{model_type}"
    )


def _validate_training_config_unchanged(
    *,
    d_snapshot: Mapping[str, Any],
    request: TrainingStageRequest,
) -> None:
    """
    A completed D stage is immutable. Bootstrap retries may change only the
    bootstrap policy, not the completed D training configuration.
    """
    persisted = d_snapshot.get(
        "training_config"
    )
    supplied = to_json_safe(
        request.training
    )

    if persisted != supplied:
        raise RunManagementError(
            "The supplied training configuration does not match the completed "
            "Part D snapshot. Create a new attempt to change completed "
            "training settings.",
            error_code="COMPLETED_PART_D_TRAINING_CONFIG_MISMATCH",
            details={
                "run_id": request.run_id,
                "attempt_id": request.attempt_id,
            },
        )


def _bootstrap_state(
    *,
    request: TrainingStageRequest,
    state_root: Path,
    run_directory: Path,
    d_snapshot: Mapping[str, Any],
    part_b: PartBHandoff,
    part_c: PartCHandoff,
    preprocessing_request: PreprocessingStageRequest,
    feature_request: FeatureEngineeringStageRequest,
    raw_training_file: Path,
) -> tuple[
    SavedInferenceContract,
    tuple[OutputRegistration, ...],
    dict[str, Any],
]:
    """
    Create the full staged saved-run state before Run Management publishes it.
    """
    _validate_training_config_unchanged(
        d_snapshot=d_snapshot,
        request=request,
    )

    if (
        part_b.dataset_config
        != part_c.dataset_config
    ):
        raise ValueError(
            "Part B and Part C persisted different DatasetInputConfig values."
        )

    persisted_d_dataset = d_snapshot.get(
        "dataset_config"
    )

    if persisted_d_dataset != to_json_safe(
        part_c.dataset_config
    ):
        raise RunManagementError(
            "Part D snapshot dataset_config does not match the persisted B/C "
            "dataset contract.",
            error_code="BOOTSTRAP_PART_D_DATASET_CONTRACT_MISMATCH",
        )

    resolved_b, resolved_c = _same_resolved_contracts(
        part_b=part_b,
        part_c=part_c,
    )

    feature_config = resolve_feature_engineering_config(
        feature_request.feature_engineering,
        model_type=(
            feature_request.model_type_hint
            or request.training.model_type
        ),
    )

    history_contract = _build_history_contract(
        preprocessing_config=(
            preprocessing_request.preprocessing
        ),
        resolved_b=resolved_b,
        resolved_c=resolved_c,
        bootstrap_config=(
            request.inference_bootstrap
        ),
    )

    b_by_kpi = {
        normalize_kpi_name(
            result.kpi_name
        ): result
        for result in part_b.preprocessing_results
    }
    c_by_kpi = {
        normalize_kpi_name(
            result.kpi_name
        ): result
        for result in part_c.feature_engineering_results
    }

    raw_records = d_snapshot[
        "training_records"
    ]

    training_records: dict[
        str,
        Mapping[str, Any],
    ] = {}

    for record in raw_records:
        if not isinstance(
            record,
            Mapping,
        ):
            raise ValueError(
                "Part D training record must be a JSON object."
            )

        kpi_name = normalize_kpi_name(
            record.get(
                "kpi_name"
            )
        )

        if kpi_name in training_records:
            raise ValueError(
                f"Part D snapshot contains duplicate KPI {kpi_name!r}."
            )

        if str(
            record.get(
                "status"
            )
        ) != "completed":
            raise RunManagementError(
                "Part D snapshot contains a training record that is not "
                "completed.",
                error_code="BOOTSTRAP_PART_D_RECORD_STATUS_INVALID",
                details={
                    "kpi_name": kpi_name,
                    "status": str(
                        record.get(
                            "status"
                        )
                    ),
                },
            )

        if (
            str(
                record.get(
                    "model_type"
                )
            )
            != request.training.model_type
        ):
            raise ValueError(
                "Part D snapshot model_type does not match the training request."
            )

        training_records[
            kpi_name
        ] = record

    kpi_set = set(
        b_by_kpi
    )

    if (
        kpi_set
        != set(
            c_by_kpi
        )
        or kpi_set
        != set(
            training_records
        )
    ):
        raise ValueError(
            "Part B, Part C, and Part D contain different KPI identities."
        )

    kpi_names = tuple(
        b_by_kpi
    )
    safe_names = _unique_safe_kpi_names(
        kpi_names
    )

    history_directory = ensure_directory(
        state_root
        / "history"
    )
    cache_directory = ensure_directory(
        state_root
        / "cache"
    )
    processed_history_directory = ensure_directory(
        history_directory
        / "processed"
    )
    processed_cache_directory = ensure_directory(
        cache_directory
        / "processed"
    )
    ensure_directory(
        state_root
        / "models"
    )
    inference_results_directory = ensure_directory(
        state_root
        / "inference_results"
    )

    raw_history_file = (
        history_directory
        / "raw_history.csv"
    )
    raw_cache_file = (
        cache_directory
        / "raw_cache.csv"
    )

    raw_dataframe = _copy_csv(
        source=raw_training_file,
        destination=raw_history_file,
    )

    bucket_minutes = int(
        resolved_c.bucket_minutes
    )

    raw_cache_dataframe = _cache_dataframe(
        raw_dataframe,
        timestamp_column=(
            part_b
            .dataset_config
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
        role="initial raw history",
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

    for kpi_name in kpi_names:
        preprocessing_result = b_by_kpi[
            kpi_name
        ]
        safe_name = safe_names[
            kpi_name
        ]

        processed_history_file = (
            processed_history_directory
            / f"{safe_name}.csv"
        )
        processed_cache_file = (
            processed_cache_directory
            / f"{safe_name}.csv"
        )

        processed_dataframe = _copy_csv(
            source=(
                preprocessing_result
                .final_dataset_file
            ),
            destination=(
                processed_history_file
            ),
        )

        processed_cache_dataframe = _cache_dataframe(
            processed_dataframe,
            timestamp_column=(
                part_b
                .dataset_config
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
                f"for KPI {kpi_name!r}"
            ),
        )

        write_csv_file(
            processed_cache_dataframe,
            processed_cache_file,
        )

        processed_history_by_kpi[
            kpi_name
        ] = processed_history_file
        processed_cache_by_kpi[
            kpi_name
        ] = processed_cache_file

    model_contract = _build_model_contract(
        model_type=(
            request.training.model_type
        ),
        threshold_direction=(
            request
            .training
            .threshold_direction
        ),
        training_records=(
            training_records
        ),
        state_root=state_root,
        safe_names=safe_names,
        run_directory=run_directory,
    )

    saved_contract = SavedInferenceContract(
        run_id=request.run_id,
        created_at_utc=datetime.now(
            timezone.utc
        ),
        model_type=(
            request.training.model_type
        ),
        dataset=SavedDatasetContract(
            columns=(
                part_b
                .dataset_config
                .columns
            ),
            bucket=(
                part_b.bucket
            ),
            kpi_names=kpi_names,
            single_kpi_name=(
                part_b
                .dataset_config
                .single_kpi_name
            ),
        ),
        preprocessing=SavedPreprocessingContract(
            config=(
                preprocessing_request
                .preprocessing
            ),
            resolved=resolved_b,
        ),
        feature_engineering=SavedFeatureEngineeringContract(
            config=feature_config,
            resolved=resolved_c,
        ),
        history=history_contract,
        storage=SavedStorageContract(
            raw_history_file=_relative_posix(
                path=raw_history_file,
                root=state_root,
            ),
            raw_cache_file=_relative_posix(
                path=raw_cache_file,
                root=state_root,
            ),
            processed_history_by_kpi={
                kpi_name: _relative_posix(
                    path=path,
                    root=state_root,
                )
                for kpi_name, path
                in processed_history_by_kpi.items()
            },
            processed_cache_by_kpi={
                kpi_name: _relative_posix(
                    path=path,
                    root=state_root,
                )
                for kpi_name, path
                in processed_cache_by_kpi.items()
            },
            inference_results_directory=(
                _relative_posix(
                    path=(
                        inference_results_directory
                    ),
                    root=state_root,
                )
            ),
        ),
        model=model_contract,
    )

    config_file = (
        state_root
        / INFERENCE_CONFIG_FILE_NAME
    )

    saved_contract_payload = (
        _portable_saved_contract_payload(
            saved_contract
        )
    )

    write_json_file(
        data=saved_contract_payload,
        file_path=config_file,
        overwrite=False,
    )

    outputs: list[
        OutputRegistration
    ] = [
        OutputRegistration(
            file_path=config_file,
            output_type="inference_config",
            include_checksum=True,
        ),
        OutputRegistration(
            file_path=raw_history_file,
            output_type="raw_history",
        ),
        OutputRegistration(
            file_path=raw_cache_file,
            output_type="raw_cache",
        ),
    ]

    for kpi_name in kpi_names:
        outputs.extend(
            (
                OutputRegistration(
                    file_path=(
                        processed_history_by_kpi[
                            kpi_name
                        ]
                    ),
                    output_type=(
                        "processed_history"
                    ),
                ),
                OutputRegistration(
                    file_path=(
                        processed_cache_by_kpi[
                            kpi_name
                        ]
                    ),
                    output_type=(
                        "processed_cache"
                    ),
                ),
            )
        )

    summary = {
        "model_type": (
            request.training.model_type
        ),
        "kpi_count": len(
            kpi_names
        ),
        "cache_days": int(
            history_contract.cache_days
        ),
        "final_required_history_points": int(
            history_contract
            .final_required_history_points
        ),
        "final_required_history_days": int(
            history_contract
            .final_required_history_days
        ),
        "max_gap_days": int(
            history_contract.max_gap_days
        ),
        "max_gap_ratio": float(
            history_contract.max_gap_ratio
        ),
        "late_data_reconstruction_enabled": bool(
            history_contract
            .late_data_reconstruction_enabled
        ),
    }

    return (
        saved_contract,
        tuple(
            outputs
        ),
        summary,
    )



def _preflight_completed_part_d(
    *,
    request: TrainingStageRequest,
    manager: RunManager,
) -> tuple[
    dict[str, Any],
    Any,
    Any,
]:
    """
    Resolve immutable completed Part D state before creating a bootstrap
    execution.

    This is specifically important for bootstrap retries: changing the already
    completed Part D training configuration must be rejected without creating a
    new failed bootstrap execution.
    """
    run_context = manager.resolve_run(
        request.run_id
    )
    attempt_context = manager.resolve_attempt(
        request.run_id,
        request.attempt_id,
    )
    manifest = manager.get_manifest(
        request.run_id
    )

    d_result = resolve_stage_result(
        manifest,
        run_context.run_directory,
        request.attempt_id,
        "part_d",
    )

    d_snapshot = _load_part_d_snapshot(
        d_result.path
    )

    _validate_training_config_unchanged(
        d_snapshot=d_snapshot,
        request=request,
    )

    return (
        d_snapshot,
        run_context,
        attempt_context,
    )


def _failed_response(
    *,
    request: TrainingStageRequest,
    execution_id: str,
    error: Exception,
) -> StandaloneStageResponse:
    """Build the standard failed bootstrap response after failure persistence."""
    detail = _error_detail(
        error
    )

    return StandaloneStageResponse(
        status="failed",
        run_id=request.run_id,
        attempt_id=request.attempt_id,
        requested_stage=(
            "inference_bootstrap"
        ),
        executions=(
            StageExecutionSummary(
                stage=(
                    "inference_bootstrap"
                ),
                execution_id=(
                    execution_id
                ),
                status="failed",
                error=detail,
            ),
        ),
        message=(
            "Inference bootstrap failed. Part D remains completed and "
            "bootstrap may be retried in the same attempt."
        ),
        next_stage=(
            "inference_bootstrap"
        ),
        inference_ready=False,
        error=detail,
    )


def _persist_started_failure(
    *,
    manager: RunManager,
    execution: Any,
    error: Exception,
) -> bool:
    """
    Best-effort mark an already-started bootstrap execution failed.

    If persistence itself fails, the original exception is propagated instead
    of being masked by a secondary manifest error.
    """
    try:
        manager.fail_stage(
            execution,
            error,
        )
    except Exception:
        return False

    return True



def _bootstrap_execution_commit_state(
    execution: Any,
) -> bool | None:
    """
    Inspect Manifest V2 directly after ``complete_stage()`` raises.

    Returns:
        True:
            this exact bootstrap execution is already the successful completed
            execution for the stage;
        False:
            this execution is not committed as the successful bootstrap;
        None:
            commit state could not be determined safely.

    Direct manifest loading is deliberate here. ``RunManager.get_manifest()``
    resolves/synchronizes the registry first, and registry synchronization may
    itself be the reason ``complete_stage()`` raised after the technical
    bootstrap had already committed.
    """
    try:
        manifest = load_run_manifest(
            execution.run_directory
        )

        successful_execution_id = (
            get_successful_execution_id(
                manifest,
                execution.attempt_id,
                "inference_bootstrap",
            )
        )

    except Exception:
        return None

    return (
        successful_execution_id
        == execution.execution_id
    )


def _recover_committed_bootstrap_activation(
    *,
    manager: RunManager,
    request: TrainingStageRequest,
    execution: Any,
) -> bool:
    """
    Recover activation when bootstrap completion committed before an exception.

    ``RunManager.activate_inference_attempt()`` is the authoritative recovery
    operation for a completed/published bootstrap whose active inference pointer
    or registry/index synchronization did not finish cleanly.
    """
    commit_state = (
        _bootstrap_execution_commit_state(
            execution
        )
    )

    if commit_state is not True:
        return False

    # Never call fail_stage() once this exact execution is already the
    # successful completed bootstrap. Doing so would corrupt the invariant
    # between successful_execution_id and execution status.
    manager.activate_inference_attempt(
        request.run_id,
        request.attempt_id,
    )

    return True


def run_inference_bootstrap_workflow(
    *,
    request: TrainingStageRequest,
    manager: RunManager,
    logger: Logger,
) -> StandaloneStageResponse:
    """
    Build, publish, and activate the Part E saved state for one completed
    training attempt.

    Preflight errors happen before an inference_bootstrap execution exists and
    therefore propagate to the future API error handler. Once the execution
    starts, construction/publication failures are persisted as failed bootstrap
    executions and returned as structured workflow responses.
    """
    del logger  # Bootstrap currently uses shared services, not an ML logger.

    # ------------------------------------------------------------------
    # 1. Preflight immutable Part D before starting a bootstrap execution.
    # ------------------------------------------------------------------
    (
        preflight_d_snapshot,
        run_context,
        attempt_context,
    ) = _preflight_completed_part_d(
        request=request,
        manager=manager,
    )

    # ------------------------------------------------------------------
    # 2. Create/validate the bootstrap execution.
    #
    # Run Management supplies an execution-specific inference-state staging
    # root. On successful complete_stage(), Run Management atomically publishes
    # that root and activates the attempt for Part E.
    # ------------------------------------------------------------------
    execution = manager.prepare_stage_execution(
        request.run_id,
        request.attempt_id,
        "inference_bootstrap",
        request_payload=request,
        metadata={
            "workflow": (
                "inference_bootstrap"
            ),
        },
    )

    try:
        state_root = (
            execution
            .inference_state_directory
        )

        if state_root is None:
            raise RunManagementError(
                "Run Management did not provide bootstrap staging state.",
                error_code="BOOTSTRAP_STATE_DIRECTORY_MISSING",
            )

        if len(
            execution.prerequisite_results
        ) != 1:
            raise RunManagementError(
                "Inference bootstrap expected exactly one authoritative "
                "Part D result.",
                error_code="BOOTSTRAP_PART_D_PREREQUISITE_COUNT_INVALID",
                details={
                    "received_count": len(
                        execution.prerequisite_results
                    ),
                },
            )

        # Re-read the exact authoritative prerequisite selected for this
        # execution and ensure it is the same immutable D contract validated in
        # preflight.
        d_snapshot = _load_part_d_snapshot(
            execution
            .prerequisite_results[0]
            .path
        )

        if d_snapshot != preflight_d_snapshot:
            raise RunManagementError(
                "The completed Part D snapshot changed between bootstrap "
                "preflight and execution preparation.",
                error_code="BOOTSTRAP_PART_D_SNAPSHOT_CHANGED",
            )

        manifest = manager.get_manifest(
            request.run_id
        )

        b_result = resolve_stage_result(
            manifest,
            run_context.run_directory,
            request.attempt_id,
            "part_b",
        )
        c_result = resolve_stage_result(
            manifest,
            run_context.run_directory,
            request.attempt_id,
            "part_c",
        )

        part_b = load_part_b_handoff(
            result_file=b_result.path,
            run_directory=(
                run_context.run_directory
            ),
        )
        part_c = load_part_c_handoff(
            result_file=c_result.path,
            run_directory=(
                run_context.run_directory
            ),
        )

        preprocessing_request = _load_stage_request(
            manifest=manifest,
            attempt_id=request.attempt_id,
            stage_name="part_b",
            request_type=(
                PreprocessingStageRequest
            ),
        )
        feature_request = _load_stage_request(
            manifest=manifest,
            attempt_id=request.attempt_id,
            stage_name="part_c",
            request_type=(
                FeatureEngineeringStageRequest
            ),
        )

        (
            saved_contract,
            outputs,
            summary,
        ) = _bootstrap_state(
            request=request,
            state_root=state_root,
            run_directory=(
                run_context.run_directory
            ),
            d_snapshot=d_snapshot,
            part_b=part_b,
            part_c=part_c,
            preprocessing_request=(
                preprocessing_request
            ),
            feature_request=(
                feature_request
            ),
            raw_training_file=(
                attempt_context.input_file
            ),
        )

        result_payload = {
            "bootstrap_result_version": (
                BOOTSTRAP_RESULT_VERSION
            ),
            "stage": (
                "inference_bootstrap"
            ),
            "run_id": request.run_id,
            "attempt_id": request.attempt_id,
            "inference_config_file": str(
                state_root
                / INFERENCE_CONFIG_FILE_NAME
            ),
            "saved_contract": (
                _portable_saved_contract_payload(
                    saved_contract
                )
            ),
            "summary": summary,
        }

    except Exception as error:
        if not _persist_started_failure(
            manager=manager,
            execution=execution,
            error=error,
        ):
            raise

        return _failed_response(
            request=request,
            execution_id=(
                execution.execution_id
            ),
            error=error,
        )

    # ------------------------------------------------------------------
    # 3. Publish staged state through Run Management.
    #
    # Run Management validates registered outputs, atomically moves staging to
    # runs/<run_id>/inference/<attempt_id>/, remaps result/output paths, marks
    # bootstrap complete, and activates the inference attempt.
    # ------------------------------------------------------------------
    try:
        manager.complete_stage(
            execution,
            result_payload=result_payload,
            metrics=summary,
            outputs=outputs,
        )

    except Exception as error:
        # ``complete_stage()`` can raise after the technical bootstrap has
        # already been published and this exact execution marked successful
        # (for example during inference-attempt activation/registry sync).
        #
        # If that happened, the execution is immutable success and must never
        # be rewritten to failed. Run Management exposes
        # activate_inference_attempt() specifically as the deterministic
        # recovery operation for that state.
        commit_state = (
            _bootstrap_execution_commit_state(
                execution
            )
        )

        if commit_state is True:
            # If recovery succeeds, the bootstrap lifecycle is complete despite
            # the earlier completion exception.
            manager.activate_inference_attempt(
                request.run_id,
                request.attempt_id,
            )

        elif commit_state is False:
            # The execution did not commit. Normal failed-stage persistence is
            # still valid and bootstrap remains retryable.
            if not _persist_started_failure(
                manager=manager,
                execution=execution,
                error=error,
            ):
                raise

            return _failed_response(
                request=request,
                execution_id=(
                    execution.execution_id
                ),
                error=error,
            )

        else:
            # Commit state is unknown. Do not risk corrupting a possibly
            # completed successful execution by calling fail_stage().
            # Propagate the original complete_stage() exception instead.
            raise

    return StandaloneStageResponse(
        status="completed",
        run_id=request.run_id,
        attempt_id=request.attempt_id,
        requested_stage=(
            "inference_bootstrap"
        ),
        executions=(
            StageExecutionSummary(
                stage=(
                    "inference_bootstrap"
                ),
                execution_id=(
                    execution.execution_id
                ),
                status="completed",
                summary=summary,
            ),
        ),
        message=(
            "Inference bootstrap completed successfully. "
            "The training attempt is now inference-ready."
        ),
        next_stage="inference",
        inference_ready=True,
        summary=summary,
    )


__all__ = [
    "BOOTSTRAP_RESULT_VERSION",
    "run_inference_bootstrap_workflow",
]
