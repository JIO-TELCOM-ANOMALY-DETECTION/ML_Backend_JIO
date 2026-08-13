"""
Part D model-output persistence.

Intended project location::

    app/ml/training/model_output.py

This module is the single persistence boundary for Part D.

Only runner.py calls this module. Model-specific training/testing modules never
write model artifacts themselves.

Design
------
One independent model is trained per site/Composite_ID series. To avoid holding
hundreds of trained models in memory, runner.py saves each completed series
model immediately into a temporary Part D output session, retains only
lightweight summaries, and finalizes the complete training package after every
valid series succeeds.

Final structure::

    <runner output>/
    └── training/
        ├── model/
        │   ├── <series_key>/
        │   │   └── final_model.json       # XGBoost
        │   └── <series_key>/
        │       └── final_model.joblib     # Isolation Forest
        │
        ├── split_plan.json
        ├── training_config.json
        ├── feature_contract.json
        ├── final_hyperparameters.json
        ├── training_summary.json
        ├── decision_settings.json
        ├── usable_row_counts.json
        └── metadata.json

The source Part C engineered dataset is NOT duplicated. The split plan stores
the exact chronological ranges used.

This module does NOT validate, split, train, tune, calibrate, test, calculate
metrics, or make anomaly decisions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, TypeAlias

import numpy as np
from pydantic import BaseModel

from app.schemas.training.isolation_forest.isolation_forest import (
    IsolationForestTrainingConfig,
)
from app.schemas.training.xgboost.xgboost import (
    XGBoostTrainingConfig,
)

from .loader import (
    SeriesIdentity,
    TrainingSplitPlan,
)
from .validation import TrainingValidationResult


TrainingConfig: TypeAlias = (
    XGBoostTrainingConfig
    | IsolationForestTrainingConfig
)


@dataclass(
    frozen=True,
    slots=True,
)
class ModelOutputSession:
    """Temporary output session used while series models are being produced."""

    output_root: Path
    staging_directory: Path
    training_directory: Path
    model_directory: Path


@dataclass(
    frozen=True,
    slots=True,
)
class SavedSeriesModel:
    """Reference to one model already serialized inside the staging package."""

    identity: SeriesIdentity
    series_key: str
    relative_model_file: Path


@dataclass(
    frozen=True,
    slots=True,
)
class SeriesModelArtifact:
    """
    Lightweight result package for one completed series.

    The trained model object is intentionally absent because it has already been
    serialized through ``save_series_model``.
    """

    identity: SeriesIdentity
    series_key: str
    relative_model_file: Path

    selected_hyperparameters: Mapping[str, Any]
    final_model_parameters: Mapping[str, Any]

    tuning_summary: tuple[Mapping[str, Any], ...]
    training_summary: Mapping[str, Any]
    calibration_summary: Mapping[str, Any] | None
    testing_summary: Mapping[str, Any]

    decision_settings: Mapping[str, Any]
    usable_row_counts: Mapping[str, Any]


@dataclass(
    frozen=True,
    slots=True,
)
class ModelOutputPaths:
    """Final paths created by one successful Part D run."""

    training_directory: Path
    model_directory: Path

    model_files: tuple[Path, ...]

    split_plan_file: Path
    training_config_file: Path
    feature_contract_file: Path
    final_hyperparameters_file: Path
    training_summary_file: Path
    decision_settings_file: Path
    usable_row_counts_file: Path
    metadata_file: Path


def _json_safe(
    value: Any,
) -> Any:
    """Recursively convert project objects into strict JSON-safe values."""

    if value is None:
        return None

    if isinstance(
        value,
        BaseModel,
    ):
        return _json_safe(
            value.model_dump(
                mode="json"
            )
        )

    if is_dataclass(
        value
    ) and not isinstance(
        value,
        type,
    ):
        return _json_safe(
            asdict(
                value
            )
        )

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        (datetime, date),
    ):
        return value.isoformat()

    if isinstance(
        value,
        np.generic,
    ):
        return _json_safe(
            value.item()
        )

    if isinstance(
        value,
        np.ndarray,
    ):
        return _json_safe(
            value.tolist()
        )

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
        float,
    ):
        return (
            value
            if math.isfinite(value)
            else None
        )

    if isinstance(
        value,
        (
            str,
            int,
            bool,
        ),
    ):
        return value

    raise TypeError(
        "Object is not JSON serializable by model_output.py: "
        f"{type(value).__name__}"
    )


def _write_json(
    path: Path,
    payload: Any,
) -> None:
    """Write one readable UTF-8 JSON artifact."""

    path.write_text(
        json.dumps(
            _json_safe(payload),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _package_version(
    package_name: str,
) -> str | None:
    """Return an installed package version when available."""

    try:
        return importlib_metadata.version(
            package_name
        )
    except importlib_metadata.PackageNotFoundError:
        return None


def _slug(
    value: str,
) -> str:
    """Convert identity text into a safe readable folder component."""

    normalized = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        value.strip(),
    ).strip(
        "._-"
    )

    return (
        normalized[:80]
        if normalized
        else "series"
    )


def build_series_key(
    identity: SeriesIdentity,
) -> str:
    """Build a readable collision-resistant folder key for one series."""

    raw = "||".join(
        identity.as_tuple()
    )

    digest = hashlib.sha1(
        raw.encode(
            "utf-8"
        )
    ).hexdigest()[:10]

    readable = "__".join(
        _slug(part)
        for part in identity.as_tuple()
    )

    return (
        f"{readable}__{digest}"
    )


def prepare_model_output(
    output_directory: str | Path,
) -> ModelOutputSession:
    """
    Create an isolated staging area for one Part D run.

    Existing finalized/staging output is rejected rather than silently
    overwritten.
    """

    output_root = Path(
        output_directory
    ).expanduser()

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_directory = (
        output_root
        / "training"
    )

    staging_directory = (
        output_root
        / ".training_build"
    )

    if training_directory.exists():
        raise FileExistsError(
            "Part D training output already exists:\n"
            f"  {training_directory}"
        )

    if staging_directory.exists():
        raise FileExistsError(
            "A stale/in-progress Part D output staging directory exists:\n"
            f"  {staging_directory}"
        )

    model_directory = (
        staging_directory
        / "model"
    )

    model_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    return ModelOutputSession(
        output_root=output_root,
        staging_directory=staging_directory,
        training_directory=training_directory,
        model_directory=model_directory,
    )


def abort_model_output(
    session: ModelOutputSession,
) -> None:
    """Remove only the temporary staging output for a failed Part D run."""

    if session.staging_directory.exists():
        shutil.rmtree(
            session.staging_directory
        )


def _save_xgboost_model(
    *,
    trained_model: Any,
    series_directory: Path,
) -> Path:
    """Serialize one XGBoost model using XGBoost native JSON."""

    save_model = getattr(
        trained_model,
        "save_model",
        None,
    )

    if not callable(
        save_model
    ):
        raise TypeError(
            "Final XGBoost model must provide callable save_model()."
        )

    model_file = (
        series_directory
        / "final_model.json"
    )

    save_model(
        str(
            model_file
        )
    )

    if not model_file.is_file():
        raise RuntimeError(
            "XGBoost did not create the expected model file:\n"
            f"  {model_file}"
        )

    return model_file


def _save_isolation_forest_model(
    *,
    trained_model: Any,
    series_directory: Path,
) -> Path:
    """Serialize one scikit-learn Isolation Forest estimator with joblib."""

    try:
        import joblib
    except ImportError as error:
        raise RuntimeError(
            "joblib is required to save Isolation Forest models."
        ) from error

    model_file = (
        series_directory
        / "final_model.joblib"
    )

    joblib.dump(
        trained_model,
        model_file,
    )

    if not model_file.is_file():
        raise RuntimeError(
            "joblib did not create the expected model file:\n"
            f"  {model_file}"
        )

    return model_file


def save_series_model(
    *,
    session: ModelOutputSession,
    model_type: str,
    identity: SeriesIdentity,
    trained_model: Any,
) -> SavedSeriesModel:
    """
    Save one completed series model immediately.

    runner.py can then release the estimator from memory before training the
    next series.
    """

    series_key = build_series_key(
        identity
    )

    series_directory = (
        session.model_directory
        / series_key
    )

    if series_directory.exists():
        raise FileExistsError(
            "Series model output already exists:\n"
            f"  {series_directory}"
        )

    series_directory.mkdir(
        parents=False,
        exist_ok=False,
    )

    if model_type == "xgboost":
        model_file = _save_xgboost_model(
            trained_model=trained_model,
            series_directory=series_directory,
        )

    elif model_type == "isolation_forest":
        model_file = _save_isolation_forest_model(
            trained_model=trained_model,
            series_directory=series_directory,
        )

    else:
        raise ValueError(
            f"Unsupported training model_type: {model_type}"
        )

    relative_model_file = model_file.relative_to(
        session.staging_directory
    )

    return SavedSeriesModel(
        identity=identity,
        series_key=series_key,
        relative_model_file=relative_model_file,
    )


def _validate_output_inputs(
    *,
    validation_result: TrainingValidationResult,
    split_plan: TrainingSplitPlan,
    training_config: TrainingConfig,
    series_artifacts: tuple[SeriesModelArtifact, ...],
) -> None:
    """Ensure all final output inputs belong to one coherent Part D run."""

    if validation_result.model_type != split_plan.model_type:
        raise ValueError(
            "validation_result.model_type and split_plan.model_type differ."
        )

    if validation_result.input_file != split_plan.input_file:
        raise ValueError(
            "validation_result and split_plan reference different datasets."
        )

    model_type = validation_result.model_type

    if model_type == "xgboost":
        if not isinstance(
            training_config,
            XGBoostTrainingConfig,
        ):
            raise TypeError(
                "XGBoost output requires XGBoostTrainingConfig."
            )

    elif model_type == "isolation_forest":
        if not isinstance(
            training_config,
            IsolationForestTrainingConfig,
        ):
            raise TypeError(
                "Isolation Forest output requires "
                "IsolationForestTrainingConfig."
            )

    else:
        raise ValueError(
            f"Unsupported training model_type: {model_type}"
        )

    if not series_artifacts:
        raise ValueError(
            "Part D output requires at least one successfully trained series."
        )

    keys = [
        artifact.series_key
        for artifact in series_artifacts
    ]

    if len(keys) != len(set(keys)):
        raise ValueError(
            "Duplicate series_key values were supplied to model output."
        )


def _build_feature_contract(
    validation_result: TrainingValidationResult,
) -> dict[str, Any]:
    """Persist the exact validated model input/target contract."""

    return {
        "all_part_c_feature_columns": (
            validation_result.all_part_c_feature_columns
        ),
        "model_feature_columns": (
            validation_result.model_feature_columns
        ),
        "excluded_part_c_feature_columns": (
            validation_result.excluded_part_c_feature_columns
        ),
        "target_column": (
            validation_result.target_column
        ),
        "site_identity_columns": (
            validation_result.site_identity_columns
        ),
        "timestamp_column": (
            validation_result.timestamp_column
        ),
        "status_column": (
            validation_result.status_column
        ),
        "usability_column": (
            validation_result.usability_column
        ),
        "invalid_reason_column": (
            validation_result.invalid_reason_column
        ),
        "trusted_statuses": (
            validation_result.trusted_statuses
        ),
        "require_complete_feature_rows": (
            validation_result.require_complete_feature_rows
        ),
        "require_non_null_target": (
            validation_result.require_non_null_target
        ),
        "minimum_history_points": (
            validation_result.minimum_history_points
        ),
    }


def _identity_payload(
    identity: SeriesIdentity,
) -> dict[str, str]:
    return {
        "circle": identity.circle,
        "cluster": identity.cluster,
        "site_name": identity.site_name,
    }


def _artifact_map(
    series_artifacts: tuple[SeriesModelArtifact, ...],
    *,
    extractor: Any,
) -> dict[str, Any]:
    """Build a series-keyed aggregate JSON payload."""

    return {
        artifact.series_key: {
            "identity": _identity_payload(
                artifact.identity
            ),
            **extractor(
                artifact
            ),
        }
        for artifact in series_artifacts
    }


def _build_metadata(
    *,
    validation_result: TrainingValidationResult,
    split_plan: TrainingSplitPlan,
    series_artifacts: tuple[SeriesModelArtifact, ...],
    user_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Create run-level traceability metadata."""

    model_type = validation_result.model_type

    library_versions: dict[
        str,
        str | None,
    ] = {
        "numpy": _package_version(
            "numpy"
        ),
        "pandas": _package_version(
            "pandas"
        ),
    }

    if model_type == "xgboost":
        library_versions[
            "xgboost"
        ] = _package_version(
            "xgboost"
        )

    else:
        library_versions[
            "scikit-learn"
        ] = _package_version(
            "scikit-learn"
        )
        library_versions[
            "joblib"
        ] = _package_version(
            "joblib"
        )

    metadata: dict[str, Any] = {
        "model_type": (
            model_type
        ),
        "saved_at_utc": datetime.now(
            timezone.utc
        ),
        "source_part_c_dataset": (
            validation_result.input_file
        ),
        "source_dataset_name": (
            validation_result.input_file.name
        ),
        "total_source_rows": (
            validation_result.total_rows
        ),
        "split_plan_series_count": len(
            split_plan.series
        ),
        "trained_series_count": len(
            series_artifacts
        ),
        "feature_count": len(
            validation_result.model_feature_columns
        ),
        "target_column": (
            validation_result.target_column
        ),
        "hyperparameter_tuning_enabled": (
            split_plan.hyperparameter_tuning_enabled
        ),
        "series_models": {
            artifact.series_key: {
                "identity": _identity_payload(
                    artifact.identity
                ),
                "model_file": (
                    artifact.relative_model_file
                ),
            }
            for artifact in series_artifacts
        },
        "library_versions": (
            library_versions
        ),
    }

    if user_metadata:
        metadata[
            "run_metadata"
        ] = dict(
            user_metadata
        )

    return metadata


def finalize_model_output(
    *,
    session: ModelOutputSession,
    validation_result: TrainingValidationResult,
    split_plan: TrainingSplitPlan,
    training_config: TrainingConfig,
    series_artifacts: tuple[SeriesModelArtifact, ...],
    metadata: Mapping[str, Any] | None = None,
) -> ModelOutputPaths:
    """
    Write all aggregate JSON artifacts and atomically publish ``training/``.

    Every series model must already have been saved through
    ``save_series_model``.
    """

    _validate_output_inputs(
        validation_result=validation_result,
        split_plan=split_plan,
        training_config=training_config,
        series_artifacts=series_artifacts,
    )

    for artifact in series_artifacts:
        staged_model_file = (
            session.staging_directory
            / artifact.relative_model_file
        )

        if not staged_model_file.is_file():
            raise FileNotFoundError(
                "A series model referenced by final output is missing:\n"
                f"  {staged_model_file}"
            )

    split_plan_file = (
        session.staging_directory
        / "split_plan.json"
    )
    training_config_file = (
        session.staging_directory
        / "training_config.json"
    )
    feature_contract_file = (
        session.staging_directory
        / "feature_contract.json"
    )
    final_hyperparameters_file = (
        session.staging_directory
        / "final_hyperparameters.json"
    )
    training_summary_file = (
        session.staging_directory
        / "training_summary.json"
    )
    decision_settings_file = (
        session.staging_directory
        / "decision_settings.json"
    )
    usable_row_counts_file = (
        session.staging_directory
        / "usable_row_counts.json"
    )
    metadata_file = (
        session.staging_directory
        / "metadata.json"
    )

    _write_json(
        split_plan_file,
        split_plan,
    )

    _write_json(
        training_config_file,
        training_config,
    )

    _write_json(
        feature_contract_file,
        _build_feature_contract(
            validation_result
        ),
    )

    _write_json(
        final_hyperparameters_file,
        _artifact_map(
            series_artifacts,
            extractor=lambda artifact: {
                "selected_hyperparameters": (
                    artifact.selected_hyperparameters
                ),
                "final_model_parameters": (
                    artifact.final_model_parameters
                ),
            },
        ),
    )

    _write_json(
        training_summary_file,
        _artifact_map(
            series_artifacts,
            extractor=lambda artifact: {
                "tuning_summary": (
                    artifact.tuning_summary
                ),
                "training_summary": (
                    artifact.training_summary
                ),
                "calibration_summary": (
                    artifact.calibration_summary
                ),
                "testing_summary": (
                    artifact.testing_summary
                ),
            },
        ),
    )

    _write_json(
        decision_settings_file,
        _artifact_map(
            series_artifacts,
            extractor=lambda artifact: {
                "decision_settings": (
                    artifact.decision_settings
                ),
            },
        ),
    )

    _write_json(
        usable_row_counts_file,
        _artifact_map(
            series_artifacts,
            extractor=lambda artifact: {
                "usable_row_counts": (
                    artifact.usable_row_counts
                ),
            },
        ),
    )

    _write_json(
        metadata_file,
        _build_metadata(
            validation_result=validation_result,
            split_plan=split_plan,
            series_artifacts=series_artifacts,
            user_metadata=metadata,
        ),
    )

    if session.training_directory.exists():
        raise FileExistsError(
            "Cannot finalize because training output already exists:\n"
            f"  {session.training_directory}"
        )

    session.staging_directory.rename(
        session.training_directory
    )

    model_files = tuple(
        session.training_directory
        / artifact.relative_model_file
        for artifact in series_artifacts
    )

    return ModelOutputPaths(
        training_directory=(
            session.training_directory
        ),
        model_directory=(
            session.training_directory
            / "model"
        ),
        model_files=(
            model_files
        ),
        split_plan_file=(
            session.training_directory
            / split_plan_file.name
        ),
        training_config_file=(
            session.training_directory
            / training_config_file.name
        ),
        feature_contract_file=(
            session.training_directory
            / feature_contract_file.name
        ),
        final_hyperparameters_file=(
            session.training_directory
            / final_hyperparameters_file.name
        ),
        training_summary_file=(
            session.training_directory
            / training_summary_file.name
        ),
        decision_settings_file=(
            session.training_directory
            / decision_settings_file.name
        ),
        usable_row_counts_file=(
            session.training_directory
            / usable_row_counts_file.name
        ),
        metadata_file=(
            session.training_directory
            / metadata_file.name
        ),
    )


__all__ = [
    "ModelOutputPaths",
    "ModelOutputSession",
    "SavedSeriesModel",
    "SeriesModelArtifact",
    "abort_model_output",
    "build_series_key",
    "finalize_model_output",
    "prepare_model_output",
    "save_series_model",
]