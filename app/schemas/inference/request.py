"""
Common request and saved-run schemas for pure inference.

Intended project location::

    app/schemas/inference/request.py

Design
------
A successful training run is registered in a database under one unique run ID.
The database/service layer resolves that ID to a saved-run directory.

This module then:

1. validates the small runtime inference request;
2. loads and validates ``inference_config.json`` from the resolved run folder;
3. validates the saved Part A/B/C contracts and history/storage contract;
4. dispatches the model section to the XGBoost or Isolation Forest inference
   schema;
5. resolves portable run-relative storage paths safely; and
6. returns one structured object for ``app/ml/inference/runner.py``.

Database lookup itself does not belong here. The caller supplies the run
directory obtained from the database.

Parts A, B, and C are reused during inference. Their effective configuration
and resolved contracts are saved by the training pipeline so future inference
does not depend on new defaults.
"""

from __future__ import annotations

import json
from datetime import datetime
from math import ceil
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.schemas.common.dataset import (
    BucketConfig,
    DatasetColumnMapping,
    DatasetInputConfig,
    ExpectedTimelineConfig,
)
from app.schemas.feature_engineering import (
    FeatureEngineeringConfig,
    ResolvedFeatureEngineeringConfig,
)
from app.schemas.preprocessing import (
    PreprocessingConfig,
    ResolvedFrequencyConfig,
)

from .isolation_forest import IsolationForestInferenceConfig
from .xgboost import XGBoostInferenceConfig


INFERENCE_CONFIG_FILE_NAME = "inference_config.json"

InferenceModelConfig: TypeAlias = Annotated[
    XGBoostInferenceConfig | IsolationForestInferenceConfig,
    Field(discriminator="model_type"),
]


def _normalise_runtime_path(
    value: str | Path,
) -> Path:
    """Normalize a runtime path without requiring it to exist yet."""

    return Path(value).expanduser()


def _normalise_relative_path(
    value: str | Path,
    *,
    field_name: str,
) -> Path:
    """Validate one portable path relative to the saved-run root."""

    text = str(value).strip()

    if not text:
        raise ValueError(f"{field_name} must be a non-empty relative path.")

    if "\\" in text:
        raise ValueError(
            f"{field_name} must use portable forward-slash separators."
        )

    if (
        Path(text).is_absolute()
        or PureWindowsPath(text).is_absolute()
        or PurePosixPath(text).is_absolute()
    ):
        raise ValueError(
            f"{field_name} must be relative to the saved-run root."
        )

    pure_path = PurePosixPath(text)

    if ".." in pure_path.parts:
        raise ValueError(
            f"{field_name} cannot escape the saved-run root with '..'."
        )

    if pure_path == PurePosixPath("."):
        raise ValueError(
            f"{field_name} must identify a file or directory."
        )

    return Path(pure_path)


def _resolve_under_run_root(
    *,
    run_root: Path,
    relative_path: Path,
) -> Path:
    """Resolve one already-validated relative path under the run root."""

    root = run_root.expanduser().resolve()
    resolved = (
        root
        / relative_path
    ).resolve()

    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "A saved inference path resolves outside the saved-run root: "
            f"{relative_path}"
        ) from error

    return resolved


def _bucket_to_minutes(
    bucket: BucketConfig,
) -> int:
    """Convert the saved Part A bucket to whole minutes."""

    unit = bucket.bucket_type.strip().casefold()

    if unit in {"minute", "minutes", "min", "mins"}:
        minutes = bucket.bucket_size

    elif unit in {"hour", "hours", "hr", "hrs"}:
        minutes = bucket.bucket_size * 60

    elif unit in {"day", "days"}:
        minutes = bucket.bucket_size * 1_440

    elif unit in {"second", "seconds"}:
        if bucket.bucket_size % 60 != 0:
            raise ValueError(
                "The saved inference bucket must resolve to whole minutes."
            )
        minutes = bucket.bucket_size // 60

    else:
        raise ValueError(
            "Unsupported saved inference bucket type."
        )

    if minutes <= 0:
        raise ValueError(
            "The saved inference bucket must be positive."
        )

    return int(minutes)


def _ordered_subset(
    subset: tuple[str, ...],
    full: tuple[str, ...],
) -> bool:
    """Return True when subset preserves its relative order in full."""

    iterator = iter(full)

    for wanted in subset:
        for candidate in iterator:
            if candidate == wanted:
                break
        else:
            return False

    return True


class InferenceRequest(BaseModel):
    """Small runtime request for one pure-inference execution."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    run_id: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
        description=(
            "Unique completed training-run identifier resolved through the "
            "run database."
        ),
    )

    input_file: Path = Field(
        description=(
            "New raw input file containing the requested inference period."
        ),
    )

    inference_days: int = Field(
        ge=1,
        description=(
            "Number of new days represented by the requested inference run."
        ),
    )

    @field_validator(
        "input_file",
        mode="before",
    )
    @classmethod
    def normalise_input_file(
        cls,
        value: str | Path,
    ) -> Path:
        return _normalise_runtime_path(value)


class SavedDatasetContract(BaseModel):
    """Reusable Part A dataset contract stored after training."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    columns: DatasetColumnMapping
    bucket: BucketConfig

    kpi_names: tuple[str, ...] = Field(
        min_length=1,
        description=(
            "KPI names for which the completed run contains trained models."
        ),
    )

    single_kpi_name: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Saved single-KPI declaration when source files do not contain "
            "the configured KPI-name column."
        ),
    )

    @field_validator("kpi_names")
    @classmethod
    def validate_kpi_names(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        normalized = tuple(
            str(value).strip()
            for value in values
        )

        if any(not value for value in normalized):
            raise ValueError("kpi_names cannot contain blank values.")

        if len(normalized) != len(set(normalized)):
            raise ValueError("kpi_names cannot contain duplicates.")

        return normalized

    @model_validator(mode="after")
    def validate_single_kpi_contract(
        self,
    ) -> SavedDatasetContract:
        if (
            self.single_kpi_name is not None
            and (
                len(self.kpi_names) != 1
                or self.kpi_names[0] != self.single_kpi_name
            )
        ):
            raise ValueError(
                "single_kpi_name must exactly match the sole saved KPI."
            )

        return self

    def build_runtime_dataset_config(
        self,
        *,
        input_file: str | Path,
    ) -> DatasetInputConfig:
        """
        Build the Part A request for an inference working raw file.

        Old training date boundaries are deliberately not reused. The current
        working raw dataset defines its own minimum/maximum timeline.
        """

        return DatasetInputConfig(
            input_file=Path(input_file),
            columns=self.columns,
            bucket=self.bucket,
            single_kpi_name=self.single_kpi_name,
            expected_timeline=ExpectedTimelineConfig(
                timeline_mode="global_min_max",
            ),
        )


class SavedPreprocessingContract(BaseModel):
    """Effective Part B request plus its training-time resolved settings."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    config: PreprocessingConfig
    resolved: ResolvedFrequencyConfig


class SavedFeatureEngineeringContract(BaseModel):
    """Effective Part C request plus its exact resolved feature contract."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    config: FeatureEngineeringConfig
    resolved: ResolvedFeatureEngineeringConfig


class SavedHistoryContract(BaseModel):
    """Frozen history/cache policy used by the inference history subsystem."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    part_b_required_history_points: int = Field(gt=0)
    part_b_required_history_days: int = Field(gt=0)

    part_c_required_history_points: int = Field(gt=0)
    part_c_maximum_lookback_points: int = Field(ge=0)

    final_required_history_points: int = Field(gt=0)
    final_required_history_days: int = Field(gt=0)

    cache_days: int = Field(
        gt=0,
        description=(
            "Z-day rolling cache retained for normal inference."
        ),
    )

    max_gap_days: int = Field(
        ge=0,
        description=(
            "Maximum allowed whole-day bridge between stored history and "
            "incoming data."
        ),
    )

    max_gap_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Maximum allowed bridge ratio. Runtime history validation defines "
            "gap_ratio as missing bridge points divided by missing bridge "
            "points plus expected incoming points."
        ),
    )

    late_data_reconstruction_enabled: bool = True

    @model_validator(mode="after")
    def validate_history_relationships(
        self,
    ) -> SavedHistoryContract:
        required_points = max(
            self.part_b_required_history_points,
            self.part_c_required_history_points,
            self.part_c_maximum_lookback_points + 1,
        )

        if self.final_required_history_points < required_points:
            raise ValueError(
                "final_required_history_points must cover Part B history, "
                "Part C minimum history, and the complete Part C maximum "
                "historical lookback."
            )

        required_days = self.final_required_history_days

        if self.final_required_history_days < self.part_b_required_history_days:
            raise ValueError(
                "final_required_history_days cannot be below the Part B "
                "history span."
            )

        if self.cache_days < required_days + self.max_gap_days:
            raise ValueError(
                "cache_days must cover final_required_history_days plus "
                "max_gap_days because normal inference has no automatic "
                "full-history fallback."
            )

        return self


class SavedStorageContract(BaseModel):
    """Portable run-relative locations maintained for inference."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    raw_history_file: Path
    raw_cache_file: Path

    processed_history_by_kpi: dict[str, Path] = Field(
        min_length=1,
    )
    processed_cache_by_kpi: dict[str, Path] = Field(
        min_length=1,
    )

    inference_results_directory: Path

    @field_validator(
        "raw_history_file",
        "raw_cache_file",
        "inference_results_directory",
        mode="before",
    )
    @classmethod
    def validate_single_relative_path(
        cls,
        value: str | Path,
        info,
    ) -> Path:
        return _normalise_relative_path(
            value,
            field_name=info.field_name,
        )

    @field_validator(
        "processed_history_by_kpi",
        "processed_cache_by_kpi",
        mode="before",
    )
    @classmethod
    def validate_path_mapping(
        cls,
        value,
        info,
    ):
        if not isinstance(value, dict):
            return value

        return {
            str(key).strip(): _normalise_relative_path(
                path,
                field_name=f"{info.field_name}[{key!r}]",
            )
            for key, path in value.items()
        }

    @model_validator(mode="after")
    def validate_storage_mappings(
        self,
    ) -> SavedStorageContract:
        if any(
            not key
            for key in self.processed_history_by_kpi
        ):
            raise ValueError(
                "processed_history_by_kpi cannot contain blank KPI keys."
            )

        if any(
            not key
            for key in self.processed_cache_by_kpi
        ):
            raise ValueError(
                "processed_cache_by_kpi cannot contain blank KPI keys."
            )

        if (
            set(self.processed_history_by_kpi)
            != set(self.processed_cache_by_kpi)
        ):
            raise ValueError(
                "Processed full-history and cache mappings must contain the "
                "same KPI names."
            )

        return self


class SavedInferenceContract(BaseModel):
    """
    Canonical ``inference_config.json`` contract saved by training.

    This is the single frozen bridge between one completed A/B/C/D training run
    and all future pure-inference runs for that saved model set.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal["1.0"] = "1.0"

    run_id: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )

    created_at_utc: datetime

    model_type: Literal[
        "xgboost",
        "isolation_forest",
    ]

    dataset: SavedDatasetContract
    preprocessing: SavedPreprocessingContract
    feature_engineering: SavedFeatureEngineeringContract
    history: SavedHistoryContract
    storage: SavedStorageContract
    model: InferenceModelConfig

    @field_validator("created_at_utc")
    @classmethod
    def validate_utc_timestamp(
        cls,
        value: datetime,
    ) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                "created_at_utc must be timezone-aware."
            )
        return value

    @model_validator(mode="after")
    def validate_cross_contract_consistency(
        self,
    ) -> SavedInferenceContract:
        if self.model.model_type != self.model_type:
            raise ValueError(
                "model_type must match the model-specific inference contract."
            )

        bucket_minutes = _bucket_to_minutes(
            self.dataset.bucket
        )

        # Re-resolve the saved effective Part B and Part C requests using the
        # saved dataset bucket. A saved inference contract is only trustworthy
        # when the persisted resolved objects are exactly reproducible from
        # the persisted configs themselves.
        try:
            recomputed_preprocessing = (
                self.preprocessing
                .config
                .resolve_for_bucket(
                    self.dataset.bucket
                )
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Saved Part B config cannot be resolved against the saved "
                "dataset bucket."
            ) from error

        if recomputed_preprocessing != self.preprocessing.resolved:
            raise ValueError(
                "Saved Part B config does not reproduce the saved resolved "
                "Part B contract."
            )

        try:
            recomputed_features = (
                self.feature_engineering
                .config
                .resolve_for_bucket(
                    self.dataset.bucket
                )
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Saved Part C config cannot be resolved against the saved "
                "dataset bucket."
            ) from error

        if recomputed_features != self.feature_engineering.resolved:
            raise ValueError(
                "Saved Part C config does not reproduce the saved resolved "
                "Part C contract."
            )

        if self.preprocessing.resolved.bucket_minutes != bucket_minutes:
            raise ValueError(
                "Saved Part B resolved bucket does not match the saved "
                "dataset bucket."
            )

        resolved_features = self.feature_engineering.resolved

        if resolved_features.bucket_minutes != bucket_minutes:
            raise ValueError(
                "Saved Part C resolved bucket does not match the saved "
                "dataset bucket."
            )

        if 1_440 % bucket_minutes == 0:
            expected_points_per_day = 1_440 // bucket_minutes

            if (
                resolved_features.points_per_day
                != expected_points_per_day
            ):
                raise ValueError(
                    "Saved Part C points_per_day does not match the saved "
                    "dataset bucket."
                )

            preprocessing_ppd = (
                self.preprocessing.resolved.points_per_day
            )

            if (
                preprocessing_ppd is not None
                and preprocessing_ppd != expected_points_per_day
            ):
                raise ValueError(
                    "Saved Part B points_per_day does not match the saved "
                    "dataset bucket."
                )

            minimum_part_c_days = max(
                1,
                ceil(
                    resolved_features.minimum_history_points
                    / expected_points_per_day
                ),
            )

            maximum_part_c_lookback_days = max(
                1,
                ceil(
                    (
                        resolved_features.maximum_lookback_points
                        + 1
                    )
                    / expected_points_per_day
                ),
            )

            required_part_c_days = max(
                minimum_part_c_days,
                maximum_part_c_lookback_days,
            )

            if (
                self.history.final_required_history_days
                < required_part_c_days
            ):
                raise ValueError(
                    "final_required_history_days does not cover the saved "
                    "Part C minimum history and maximum historical lookback."
                )

        if (
            self.history.part_c_required_history_points
            != resolved_features.minimum_history_points
        ):
            raise ValueError(
                "part_c_required_history_points must equal the exact saved "
                "Part C minimum_history_points."
            )

        if (
            self.history.part_c_maximum_lookback_points
            != resolved_features.maximum_lookback_points
        ):
            raise ValueError(
                "part_c_maximum_lookback_points must equal the exact saved "
                "Part C maximum_lookback_points."
            )

        if self.preprocessing.config.phase_4.enabled:
            prophet_points = (
                self.preprocessing
                .resolved
                .prophet_minimum_history_points
            )

            if prophet_points is None:
                raise ValueError(
                    "Enabled Prophet Phase 4 requires saved resolved Prophet "
                    "minimum-history points."
                )

            if (
                self.history.part_b_required_history_points
                < prophet_points
            ):
                raise ValueError(
                    "Part B saved history requirement cannot be below the "
                    "resolved Prophet minimum-history points."
                )

        minimum_part_b_days = max(
            self.preprocessing.config.phase_2.recent_same_type_lookback_days,
            self.preprocessing.config.phase_2.last_week_lookback_days,
            self.preprocessing.config.phase_3.maximum_lookback_days,
            (
                self.preprocessing.config.phase_4.minimum_history_days
                if self.preprocessing.config.phase_4.enabled
                else 0
            ),
        )

        if (
            self.history.part_b_required_history_days
            < minimum_part_b_days
        ):
            raise ValueError(
                "part_b_required_history_days is below an enabled Part B "
                "historical lookback."
            )

        expected_kpis = set(
            self.dataset.kpi_names
        )

        if set(self.storage.processed_history_by_kpi) != expected_kpis:
            raise ValueError(
                "processed_history_by_kpi must contain exactly the trained "
                "KPI names."
            )

        if set(self.storage.processed_cache_by_kpi) != expected_kpis:
            raise ValueError(
                "processed_cache_by_kpi must contain exactly the trained "
                "KPI names."
            )

        if set(self.model.kpis) != expected_kpis:
            raise ValueError(
                "The model inference contract must contain exactly the "
                "trained KPI names."
            )

        part_c_columns = tuple(
            resolved_features.feature_columns
        )
        part_c_column_set = set(
            part_c_columns
        )

        if self.model_type == "xgboost":
            assert isinstance(
                self.model,
                XGBoostInferenceConfig,
            )

            for kpi_name, kpi_config in self.model.kpis.items():
                model_columns = tuple(
                    kpi_config.model_feature_columns
                )

                if not set(model_columns).issubset(part_c_column_set):
                    raise ValueError(
                        "XGBoost model_feature_columns must come from the "
                        f"saved Part C feature contract. KPI={kpi_name!r}."
                    )

                if not _ordered_subset(
                    model_columns,
                    part_c_columns,
                ):
                    raise ValueError(
                        "XGBoost model_feature_columns must preserve the "
                        f"saved Part C feature order. KPI={kpi_name!r}."
                    )

                if (
                    kpi_config.target_column
                    != self.dataset.columns.value_column
                ):
                    raise ValueError(
                        "XGBoost target_column must match the saved dataset "
                        f"value column. KPI={kpi_name!r}."
                    )

        else:
            assert isinstance(
                self.model,
                IsolationForestInferenceConfig,
            )

            for kpi_name, kpi_config in self.model.kpis.items():
                if (
                    tuple(kpi_config.model_feature_columns)
                    != part_c_columns
                ):
                    raise ValueError(
                        "Isolation Forest must use the exact ordered saved "
                        f"Part C feature contract. KPI={kpi_name!r}."
                    )

        return self


class ResolvedStoragePaths(BaseModel):
    """Absolute paths derived safely from one saved-run root."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    raw_history_file: Path
    raw_cache_file: Path

    processed_history_by_kpi: dict[str, Path]
    processed_cache_by_kpi: dict[str, Path]

    inference_results_directory: Path


class ResolvedInferenceRequest(BaseModel):
    """Fully validated request handed to the ML inference runner."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    request: InferenceRequest
    run_directory: Path

    config_file: Path
    saved: SavedInferenceContract

    storage: ResolvedStoragePaths


def load_saved_inference_request(
    *,
    request: InferenceRequest,
    run_directory: str | Path,
    config_filename: str = INFERENCE_CONFIG_FILE_NAME,
) -> ResolvedInferenceRequest:
    """
    Load one saved inference contract after the database resolves run_id.

    The function intentionally does not query the database. It validates the
    config file, verifies that the config belongs to the requested run, and
    resolves every common storage path under the supplied run root.

    Existence/contents of raw, processed, cache, and model files are validated
    later by the ML inference loader/history modules because those checks are
    runtime data responsibilities.
    """

    root = Path(
        run_directory
    ).expanduser().resolve()

    config_relative = _normalise_relative_path(
        config_filename,
        field_name="config_filename",
    )

    config_file = _resolve_under_run_root(
        run_root=root,
        relative_path=config_relative,
    )

    if not config_file.is_file():
        raise FileNotFoundError(
            "Saved inference configuration does not exist:\n"
            f"  {config_file}"
        )

    try:
        with config_file.open(
            "r",
            encoding="utf-8",
        ) as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(
            "Saved inference configuration contains invalid JSON."
        ) from error

    saved = SavedInferenceContract.model_validate(
        payload
    )

    if saved.run_id != request.run_id:
        raise ValueError(
            "Saved inference configuration belongs to a different run ID. "
            f"request={request.run_id!r}, saved={saved.run_id!r}."
        )

    storage = ResolvedStoragePaths(
        raw_history_file=_resolve_under_run_root(
            run_root=root,
            relative_path=saved.storage.raw_history_file,
        ),
        raw_cache_file=_resolve_under_run_root(
            run_root=root,
            relative_path=saved.storage.raw_cache_file,
        ),
        processed_history_by_kpi={
            kpi_name: _resolve_under_run_root(
                run_root=root,
                relative_path=relative_path,
            )
            for kpi_name, relative_path
            in saved.storage.processed_history_by_kpi.items()
        },
        processed_cache_by_kpi={
            kpi_name: _resolve_under_run_root(
                run_root=root,
                relative_path=relative_path,
            )
            for kpi_name, relative_path
            in saved.storage.processed_cache_by_kpi.items()
        },
        inference_results_directory=_resolve_under_run_root(
            run_root=root,
            relative_path=saved.storage.inference_results_directory,
        ),
    )

    return ResolvedInferenceRequest(
        request=request,
        run_directory=root,
        config_file=config_file,
        saved=saved,
        storage=storage,
    )


__all__ = [
    "INFERENCE_CONFIG_FILE_NAME",
    "InferenceModelConfig",
    "InferenceRequest",
    "ResolvedInferenceRequest",
    "ResolvedStoragePaths",
    "SavedDatasetContract",
    "SavedFeatureEngineeringContract",
    "SavedHistoryContract",
    "SavedInferenceContract",
    "SavedPreprocessingContract",
    "SavedStorageContract",
    "load_saved_inference_request",
]