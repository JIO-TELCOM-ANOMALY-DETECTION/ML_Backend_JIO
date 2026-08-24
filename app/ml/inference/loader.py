"""
Load and resolve saved inference artifacts.

Intended project location::

    app/ml/inference/loader.py

This module is the file/artifact-loading boundary for the parent inference
pipeline.

The parent inference runner is responsible for orchestration and memory
lifecycle. The loader therefore follows this design:

1. the parent runner/service resolves ``run_id -> run_directory``;
2. this loader calls the saved inference schema helper to load and validate
   ``inference_config.json``;
3. this loader resolves all saved model paths under that run directory and
   builds a lightweight path/config plan;
4. the parent runner passes that plan to ``inference/validation.py``;
5. only after validation succeeds, the parent runner requests ONE site model
   at a time with ``load_series_model(...)``;
6. the runner performs inference for that site and releases the estimator
   before loading the next site.

Important memory rule
---------------------
``load_inference_artifacts`` and ``build_model_load_plan`` NEVER deserialize
model estimators. They keep only schema objects, identities, feature contracts,
decision settings, and absolute model paths in memory.

There is deliberately no ``load_all_models`` function.

Model serialization follows the current Part D output contract:

- XGBoost: native ``final_model.json`` saved through ``save_model``;
- Isolation Forest: ``final_model.joblib`` saved through joblib.

The loader does not:

- query the run database;
- run history processing;
- validate Part C/model semantic compatibility;
- group inference rows by site;
- execute predictions;
- apply anomaly thresholds;
- save inference outputs; or
- commit history.

Those responsibilities belong to the parent runner, validation module,
model-specific inference modules, and history subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

from app.core.exceptions import (
    FileProcessingError,
    InputValidationError,
)
from app.schemas.inference.isolation_forest import (
    IsolationForestDecisionSettings,
    IsolationForestInferenceConfig,
)
from app.schemas.inference.request import (
    InferenceRequest,
    ResolvedInferenceRequest,
    ResolvedStoragePaths,
    SavedInferenceContract,
    load_saved_inference_request,
)
from app.schemas.inference.xgboost import (
    XGBoostDecisionSettings,
    XGBoostInferenceConfig,
)


InferenceModelType = Literal[
    "xgboost",
    "isolation_forest",
]

SeriesIdentityTuple = tuple[
    str,
    str,
    str,
]

DecisionSettings = (
    XGBoostDecisionSettings
    | IsolationForestDecisionSettings
)


@dataclass(frozen=True, slots=True)
class InferenceSeriesArtifact:
    """
    Lightweight saved artifact contract for ONE KPI/site series.

    No estimator object is loaded here. ``model_file`` is the absolute path
    safely resolved beneath the saved-run root.
    """

    model_type: InferenceModelType

    kpi_name: str
    series_key: str
    identity: SeriesIdentityTuple

    model_file: Path

    model_feature_columns: tuple[str, ...]
    target_column: str | None

    decision_settings: DecisionSettings


@dataclass(frozen=True, slots=True)
class InferenceKPIArtifactPlan:
    """Saved model-input and per-series artifact plan for one KPI."""

    model_type: InferenceModelType
    kpi_name: str

    model_feature_columns: tuple[str, ...]
    target_column: str | None

    series: tuple[
        InferenceSeriesArtifact,
        ...,
    ]

    def get_series(
        self,
        series_key: str,
    ) -> InferenceSeriesArtifact:
        """Return one saved series artifact by its schema series key."""

        for item in self.series:
            if item.series_key == series_key:
                return item

        raise KeyError(
            f"No saved inference series {series_key!r} exists for "
            f"KPI {self.kpi_name!r}."
        )

    def find_identity(
        self,
        identity: SeriesIdentityTuple,
    ) -> InferenceSeriesArtifact:
        """Return one saved artifact by its stable telecom site identity."""

        for item in self.series:
            if item.identity == identity:
                return item

        raise KeyError(
            "No saved inference model exists for site identity "
            f"{identity!r} in KPI {self.kpi_name!r}."
        )


@dataclass(frozen=True, slots=True)
class InferenceModelLoadPlan:
    """
    Complete lightweight model plan for one saved inference run.

    The plan may describe many sites, but contains no deserialized estimators.
    """

    run_directory: Path
    model_type: InferenceModelType

    kpis: tuple[
        InferenceKPIArtifactPlan,
        ...,
    ]

    @property
    def total_series(self) -> int:
        """Return the total number of saved site models."""

        return sum(
            len(kpi.series)
            for kpi in self.kpis
        )

    def get_kpi(
        self,
        kpi_name: str,
    ) -> InferenceKPIArtifactPlan:
        """Return one trained KPI's artifact plan."""

        for item in self.kpis:
            if item.kpi_name == kpi_name:
                return item

        raise KeyError(
            f"No saved inference model plan exists for KPI {kpi_name!r}."
        )

    def iter_series(
        self,
    ) -> Iterator[InferenceSeriesArtifact]:
        """
        Yield model artifacts in deterministic saved-KPI/series-key order.

        The parent runner should use this iterator and call
        ``load_series_model`` for exactly one yielded item at a time.
        """

        for kpi in self.kpis:
            yield from kpi.series


@dataclass(frozen=True, slots=True)
class InferenceLoadResult:
    """
    Parent-runner loading result.

    ``resolved_request`` contains the fully validated saved inference contract
    and resolved common storage paths.

    ``model_plan`` contains all model paths/settings without loading model
    objects.
    """

    resolved_request: ResolvedInferenceRequest
    model_plan: InferenceModelLoadPlan

    @property
    def request(self) -> InferenceRequest:
        return self.resolved_request.request

    @property
    def saved(self) -> SavedInferenceContract:
        return self.resolved_request.saved

    @property
    def storage(self) -> ResolvedStoragePaths:
        return self.resolved_request.storage

    @property
    def run_directory(self) -> Path:
        return self.resolved_request.run_directory


@dataclass(frozen=True, slots=True)
class LoadedSeriesModel:
    """
    Exactly one deserialized saved estimator plus its artifact contract.

    The parent runner should keep this object only for the duration of one
    site-series inference operation.
    """

    artifact: InferenceSeriesArtifact
    estimator: Any


def _resolve_under_run_root(
    *,
    run_directory: Path,
    relative_path: str | Path,
    role: str,
) -> Path:
    """
    Resolve a saved relative path while preventing saved-run escape.

    The schema already rejects absolute paths and ``..``. This runtime check is
    retained defensively because model files are opened by this module.
    """

    root = Path(
        run_directory
    ).expanduser().resolve()

    candidate = (
        root
        / Path(
            relative_path
        )
    ).resolve()

    try:
        candidate.relative_to(
            root
        )
    except ValueError as error:
        raise InputValidationError(
            "A saved inference model path escapes the resolved run directory.",
            error_code="INFERENCE_MODEL_PATH_ESCAPES_RUN_ROOT",
            details={
                "role": role,
                "run_directory": str(root),
                "resolved_path": str(candidate),
                "saved_relative_path": str(relative_path),
            },
        ) from error

    return candidate


def _expected_model_suffix(
    model_type: InferenceModelType,
) -> str:
    if model_type == "xgboost":
        return ".json"

    if model_type == "isolation_forest":
        return ".joblib"

    raise InputValidationError(
        "Unsupported inference model type.",
        error_code="UNSUPPORTED_INFERENCE_MODEL_TYPE",
        details={
            "model_type": model_type,
        },
    )


def _resolve_series_model_file(
    *,
    run_directory: Path,
    model_type: InferenceModelType,
    relative_path: str | Path,
    kpi_name: str,
    series_key: str,
) -> Path:
    """
    Resolve one schema model path without loading or requiring the file yet.

    File existence is intentionally left to validation and the final on-demand
    loader check. This keeps path discovery separate from semantic validation.
    """

    resolved = _resolve_under_run_root(
        run_directory=run_directory,
        relative_path=relative_path,
        role=(
            f"{model_type} model for KPI {kpi_name!r}, "
            f"series {series_key!r}"
        ),
    )

    expected_suffix = _expected_model_suffix(
        model_type
    )

    if resolved.suffix.lower() != expected_suffix:
        raise InputValidationError(
            "Saved model file extension does not match its model type.",
            error_code="INFERENCE_MODEL_FILE_EXTENSION_MISMATCH",
            details={
                "model_type": model_type,
                "kpi_name": kpi_name,
                "series_key": series_key,
                "model_file": str(resolved),
                "expected_suffix": expected_suffix,
            },
        )

    return resolved


def _build_xgboost_plan(
    *,
    resolved_request: ResolvedInferenceRequest,
    model_config: XGBoostInferenceConfig,
) -> InferenceModelLoadPlan:
    kpi_plans: list[
        InferenceKPIArtifactPlan
    ] = []

    expected_kpis = tuple(
        resolved_request.saved.dataset.kpi_names
    )

    if set(model_config.kpis) != set(expected_kpis):
        raise InputValidationError(
            "Saved XGBoost model mapping does not match the trained KPI set.",
            error_code="XGBOOST_INFERENCE_MODEL_KPI_SET_MISMATCH",
            details={
                "expected_kpis": sorted(expected_kpis),
                "model_kpis": sorted(model_config.kpis),
            },
        )

    for kpi_name in expected_kpis:
        kpi_config = model_config.kpis[
            kpi_name
        ]

        series_artifacts: list[
            InferenceSeriesArtifact
        ] = []

        for series_key in sorted(
            kpi_config.series
        ):
            series_config = (
                kpi_config.series[
                    series_key
                ]
            )

            series_artifacts.append(
                InferenceSeriesArtifact(
                    model_type="xgboost",
                    kpi_name=kpi_name,
                    series_key=series_key,
                    identity=(
                        series_config
                        .identity
                        .as_tuple()
                    ),
                    model_file=(
                        _resolve_series_model_file(
                            run_directory=(
                                resolved_request
                                .run_directory
                            ),
                            model_type="xgboost",
                            relative_path=(
                                series_config
                                .model_file
                            ),
                            kpi_name=kpi_name,
                            series_key=series_key,
                        )
                    ),
                    model_feature_columns=(
                        kpi_config
                        .model_feature_columns
                    ),
                    target_column=(
                        kpi_config
                        .target_column
                    ),
                    decision_settings=(
                        series_config
                        .decision_settings
                    ),
                )
            )

        kpi_plans.append(
            InferenceKPIArtifactPlan(
                model_type="xgboost",
                kpi_name=kpi_name,
                model_feature_columns=(
                    kpi_config
                    .model_feature_columns
                ),
                target_column=(
                    kpi_config
                    .target_column
                ),
                series=tuple(
                    series_artifacts
                ),
            )
        )

    return InferenceModelLoadPlan(
        run_directory=(
            resolved_request
            .run_directory
        ),
        model_type="xgboost",
        kpis=tuple(
            kpi_plans
        ),
    )


def _build_isolation_forest_plan(
    *,
    resolved_request: ResolvedInferenceRequest,
    model_config: IsolationForestInferenceConfig,
) -> InferenceModelLoadPlan:
    kpi_plans: list[
        InferenceKPIArtifactPlan
    ] = []

    expected_kpis = tuple(
        resolved_request.saved.dataset.kpi_names
    )

    if set(model_config.kpis) != set(expected_kpis):
        raise InputValidationError(
            "Saved Isolation Forest model mapping does not match the trained "
            "KPI set.",
            error_code="ISOLATION_FOREST_INFERENCE_MODEL_KPI_SET_MISMATCH",
            details={
                "expected_kpis": sorted(expected_kpis),
                "model_kpis": sorted(model_config.kpis),
            },
        )

    for kpi_name in expected_kpis:
        kpi_config = model_config.kpis[
            kpi_name
        ]

        series_artifacts: list[
            InferenceSeriesArtifact
        ] = []

        for series_key in sorted(
            kpi_config.series
        ):
            series_config = (
                kpi_config.series[
                    series_key
                ]
            )

            series_artifacts.append(
                InferenceSeriesArtifact(
                    model_type=(
                        "isolation_forest"
                    ),
                    kpi_name=kpi_name,
                    series_key=series_key,
                    identity=(
                        series_config
                        .identity
                        .as_tuple()
                    ),
                    model_file=(
                        _resolve_series_model_file(
                            run_directory=(
                                resolved_request
                                .run_directory
                            ),
                            model_type=(
                                "isolation_forest"
                            ),
                            relative_path=(
                                series_config
                                .model_file
                            ),
                            kpi_name=kpi_name,
                            series_key=series_key,
                        )
                    ),
                    model_feature_columns=(
                        kpi_config
                        .model_feature_columns
                    ),
                    target_column=None,
                    decision_settings=(
                        series_config
                        .decision_settings
                    ),
                )
            )

        kpi_plans.append(
            InferenceKPIArtifactPlan(
                model_type=(
                    "isolation_forest"
                ),
                kpi_name=kpi_name,
                model_feature_columns=(
                    kpi_config
                    .model_feature_columns
                ),
                target_column=None,
                series=tuple(
                    series_artifacts
                ),
            )
        )

    return InferenceModelLoadPlan(
        run_directory=(
            resolved_request
            .run_directory
        ),
        model_type="isolation_forest",
        kpis=tuple(
            kpi_plans
        ),
    )


def build_model_load_plan(
    resolved_request: ResolvedInferenceRequest,
) -> InferenceModelLoadPlan:
    """
    Resolve all saved model paths/settings without deserializing any model.

    This is the object the parent runner should pass to
    ``inference/validation.py``.
    """

    saved = resolved_request.saved

    if (
        saved.model_type
        != saved.model.model_type
    ):
        raise InputValidationError(
            "Saved run model_type does not match its model-specific contract.",
            error_code="INFERENCE_MODEL_TYPE_CONTRACT_MISMATCH",
            details={
                "saved_model_type": saved.model_type,
                "contract_model_type": (
                    saved.model.model_type
                ),
            },
        )

    if saved.model_type == "xgboost":
        if not isinstance(
            saved.model,
            XGBoostInferenceConfig,
        ):
            raise InputValidationError(
                "XGBoost saved run did not produce an XGBoost inference "
                "contract.",
                error_code="INVALID_XGBOOST_INFERENCE_CONTRACT",
            )

        return _build_xgboost_plan(
            resolved_request=resolved_request,
            model_config=saved.model,
        )

    if saved.model_type == "isolation_forest":
        if not isinstance(
            saved.model,
            IsolationForestInferenceConfig,
        ):
            raise InputValidationError(
                "Isolation Forest saved run did not produce an Isolation "
                "Forest inference contract.",
                error_code="INVALID_ISOLATION_FOREST_INFERENCE_CONTRACT",
            )

        return _build_isolation_forest_plan(
            resolved_request=resolved_request,
            model_config=saved.model,
        )

    raise InputValidationError(
        "Unsupported inference model type.",
        error_code="UNSUPPORTED_INFERENCE_MODEL_TYPE",
        details={
            "model_type": saved.model_type,
        },
    )


def load_inference_artifacts(
    *,
    request: InferenceRequest,
    run_directory: str | Path,
) -> InferenceLoadResult:
    """
    Load the saved inference contract and resolve all artifact paths.

    ``run_directory`` must already have been resolved by the parent
    runner/service from the request's ``run_id``.

    No trained estimator is deserialized by this function.
    """

    resolved_request = (
        load_saved_inference_request(
            request=request,
            run_directory=run_directory,
        )
    )

    model_plan = build_model_load_plan(
        resolved_request
    )

    return InferenceLoadResult(
        resolved_request=resolved_request,
        model_plan=model_plan,
    )


def _required_model_file(
    artifact: InferenceSeriesArtifact,
) -> Path:
    """
    Require one model file immediately before on-demand deserialization.

    Validation should normally have checked this already. The loader repeats
    the check so a file removed between validation and execution fails safely.
    """

    path = (
        artifact
        .model_file
        .expanduser()
        .resolve()
    )

    if not path.is_file():
        raise FileProcessingError(
            "Required saved inference model file does not exist.",
            error_code="MISSING_INFERENCE_MODEL_FILE",
            details={
                "model_type": artifact.model_type,
                "kpi_name": artifact.kpi_name,
                "series_key": artifact.series_key,
                "identity": artifact.identity,
                "model_file": str(path),
            },
        )

    expected_suffix = (
        _expected_model_suffix(
            artifact.model_type
        )
    )

    if path.suffix.lower() != expected_suffix:
        raise FileProcessingError(
            "Saved inference model file has an unexpected extension.",
            error_code="INVALID_INFERENCE_MODEL_FILE_EXTENSION",
            details={
                "model_type": artifact.model_type,
                "model_file": str(path),
                "expected_suffix": expected_suffix,
            },
        )

    return path


def _load_xgboost_model(
    model_file: Path,
) -> Any:
    """
    Load one XGBRegressor from the native JSON produced by Part D.
    """

    try:
        from xgboost import XGBRegressor
    except ImportError as error:
        raise FileProcessingError(
            "xgboost is required to load a saved XGBoost inference model.",
            error_code="XGBOOST_INFERENCE_DEPENDENCY_MISSING",
        ) from error

    model = XGBRegressor()

    try:
        model.load_model(
            str(
                model_file
            )
        )
    except Exception as error:
        raise FileProcessingError(
            "Saved XGBoost model could not be loaded.",
            error_code="XGBOOST_INFERENCE_MODEL_LOAD_FAILED",
            details={
                "model_file": str(model_file),
                "reason": str(error),
            },
        ) from error

    if not callable(
        getattr(
            model,
            "predict",
            None,
        )
    ):
        raise FileProcessingError(
            "Loaded XGBoost artifact does not expose predict().",
            error_code="INVALID_LOADED_XGBOOST_MODEL",
            details={
                "model_file": str(model_file),
            },
        )

    return model


def _load_isolation_forest_model(
    model_file: Path,
) -> Any:
    """
    Load one scikit-learn Isolation Forest estimator from joblib.
    """

    try:
        import joblib
    except ImportError as error:
        raise FileProcessingError(
            "joblib is required to load a saved Isolation Forest model.",
            error_code="ISOLATION_FOREST_INFERENCE_DEPENDENCY_MISSING",
        ) from error

    try:
        model = joblib.load(
            model_file
        )
    except Exception as error:
        raise FileProcessingError(
            "Saved Isolation Forest model could not be loaded.",
            error_code="ISOLATION_FOREST_INFERENCE_MODEL_LOAD_FAILED",
            details={
                "model_file": str(model_file),
                "reason": str(error),
            },
        ) from error

    required_methods = (
        "predict",
        "decision_function",
        "score_samples",
    )

    missing_methods = [
        method_name
        for method_name in required_methods
        if not callable(
            getattr(
                model,
                method_name,
                None,
            )
        )
    ]

    if missing_methods:
        raise FileProcessingError(
            "Loaded Isolation Forest artifact is missing required inference "
            "methods.",
            error_code="INVALID_LOADED_ISOLATION_FOREST_MODEL",
            details={
                "model_file": str(model_file),
                "missing_methods": missing_methods,
            },
        )

    return model


def load_series_model(
    artifact: InferenceSeriesArtifact,
) -> LoadedSeriesModel:
    """
    Deserialize exactly ONE saved site model.

    This is intentionally the only public estimator-loading function. The
    parent runner should call it inside its site loop, perform validation/model
    inference for that one site, release the returned object, and only then
    continue to the next artifact.
    """

    model_file = _required_model_file(
        artifact
    )

    if artifact.model_type == "xgboost":
        estimator = _load_xgboost_model(
            model_file
        )

    elif artifact.model_type == "isolation_forest":
        estimator = (
            _load_isolation_forest_model(
                model_file
            )
        )

    else:
        raise InputValidationError(
            "Unsupported inference model type.",
            error_code="UNSUPPORTED_INFERENCE_MODEL_TYPE",
            details={
                "model_type": artifact.model_type,
            },
        )

    return LoadedSeriesModel(
        artifact=artifact,
        estimator=estimator,
    )


__all__ = [
    "DecisionSettings",
    "InferenceKPIArtifactPlan",
    "InferenceLoadResult",
    "InferenceModelLoadPlan",
    "InferenceModelType",
    "InferenceSeriesArtifact",
    "LoadedSeriesModel",
    "SeriesIdentityTuple",
    "build_model_load_plan",
    "load_inference_artifacts",
    "load_series_model",
]