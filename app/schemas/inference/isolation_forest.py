"""
Saved Isolation Forest inference contract.

Intended project location::

    app/schemas/inference/isolation_forest.py

This module contains only information required to execute already-trained
Isolation Forest models during inference. Training, tuning, split, validation,
and test settings do not belong here.

Model paths are portable paths relative to the saved-run root.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.schemas.preprocessing import OutlierDirection


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
        raise ValueError(f"{field_name} must identify a file.")

    return Path(pure_path)


class IsolationForestSeriesIdentity(BaseModel):
    """Stable three-column telecom site identity used by Part D."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    circle: str = Field(min_length=1)
    cluster: str = Field(min_length=1)
    site_name: str = Field(min_length=1)

    def as_tuple(self) -> tuple[str, str, str]:
        return (
            self.circle,
            self.cluster,
            self.site_name,
        )


class IsolationForestDecisionSettings(BaseModel):
    """
    Persist the exact final decision contract produced by Part D.

    The estimator identifies model anomalies with predict == -1. For one-sided
    requests, the sign of direction_feature determines whether that anomaly
    belongs to the requested lower or upper side.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        str_strip_whitespace=True,
    )

    method: Literal["isolation_forest"] = "isolation_forest"

    model_prediction_rule: Literal[
        "predict == -1"
    ] = "predict == -1"

    decision_function_rule: Literal[
        "decision_function < 0 indicates model anomaly"
    ] = "decision_function < 0 indicates model anomaly"

    model_offset: float

    contamination: float | Literal["auto"]

    threshold_direction: OutlierDirection

    direction_feature: str = Field(
        min_length=1,
    )

    lower_direction_rule: str = Field(
        min_length=1,
    )

    upper_direction_rule: str = Field(
        min_length=1,
    )

    @field_validator("contamination")
    @classmethod
    def validate_contamination(
        cls,
        value: float | Literal["auto"],
    ) -> float | Literal["auto"]:
        if value == "auto":
            return value

        numeric = float(value)

        if not 0.0 < numeric <= 0.5:
            raise ValueError(
                "Isolation Forest contamination must be 'auto' or lie in "
                "(0, 0.5]."
            )

        return numeric

    @model_validator(mode="after")
    def validate_direction_rules(
        self,
    ) -> IsolationForestDecisionSettings:
        expected_lower = f"{self.direction_feature} < 0"
        expected_upper = f"{self.direction_feature} > 0"

        if self.lower_direction_rule != expected_lower:
            raise ValueError(
                "lower_direction_rule must match the saved direction_feature."
            )

        if self.upper_direction_rule != expected_upper:
            raise ValueError(
                "upper_direction_rule must match the saved direction_feature."
            )

        return self


class IsolationForestSeriesInferenceConfig(BaseModel):
    """Saved model and decision contract for one site/KPI series."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    identity: IsolationForestSeriesIdentity

    model_file: Path = Field(
        description=(
            "Portable model path relative to the saved-run root."
        ),
    )

    decision_settings: IsolationForestDecisionSettings

    @field_validator(
        "model_file",
        mode="before",
    )
    @classmethod
    def validate_model_file(
        cls,
        value: str | Path,
    ) -> Path:
        return _normalise_relative_path(
            value,
            field_name="model_file",
        )


class IsolationForestKPIInferenceConfig(BaseModel):
    """Exact Isolation Forest input contract for one KPI."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    model_feature_columns: tuple[str, ...] = Field(
        min_length=1,
        description=(
            "Exact ordered Part C feature matrix used by the estimator."
        ),
    )

    series: dict[str, IsolationForestSeriesInferenceConfig] = Field(
        min_length=1,
    )

    @field_validator("model_feature_columns")
    @classmethod
    def validate_feature_columns(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        normalized = tuple(
            str(value).strip()
            for value in values
        )

        if any(not value for value in normalized):
            raise ValueError(
                "model_feature_columns cannot contain blank names."
            )

        if len(normalized) != len(set(normalized)):
            raise ValueError(
                "model_feature_columns cannot contain duplicates."
            )

        return normalized

    @field_validator("series")
    @classmethod
    def validate_series_keys(
        cls,
        values: dict[str, IsolationForestSeriesInferenceConfig],
    ) -> dict[str, IsolationForestSeriesInferenceConfig]:
        if any(not str(key).strip() for key in values):
            raise ValueError("Isolation Forest series keys cannot be blank.")
        return values

    @model_validator(mode="after")
    def validate_series_contracts(
        self,
    ) -> IsolationForestKPIInferenceConfig:
        identities = [
            item.identity.as_tuple()
            for item in self.series.values()
        ]

        if len(identities) != len(set(identities)):
            raise ValueError(
                "A KPI cannot map the same site identity to multiple "
                "Isolation Forest series entries."
            )

        available_features = set(
            self.model_feature_columns
        )

        for series_key, series_config in self.series.items():
            direction = (
                series_config
                .decision_settings
                .threshold_direction
            )
            direction_feature = (
                series_config
                .decision_settings
                .direction_feature
            )

            if (
                direction in {"lower_only", "upper_only"}
                and direction_feature not in available_features
            ):
                raise ValueError(
                    "One-sided Isolation Forest inference requires its saved "
                    "direction_feature to be present in model_feature_columns. "
                    f"Series={series_key!r}, feature={direction_feature!r}."
                )

        return self


class IsolationForestInferenceConfig(BaseModel):
    """Complete model-specific saved contract for Isolation Forest inference."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    model_type: Literal["isolation_forest"] = "isolation_forest"

    threshold_direction: OutlierDirection

    kpis: dict[str, IsolationForestKPIInferenceConfig] = Field(
        min_length=1,
    )

    @field_validator("kpis")
    @classmethod
    def validate_kpi_names(
        cls,
        values: dict[str, IsolationForestKPIInferenceConfig],
    ) -> dict[str, IsolationForestKPIInferenceConfig]:
        if any(not str(key).strip() for key in values):
            raise ValueError("Isolation Forest KPI names cannot be blank.")
        return values

    @model_validator(mode="after")
    def validate_shared_threshold_direction(
        self,
    ) -> IsolationForestInferenceConfig:
        model_file_owners: dict[
            str,
            tuple[str, str],
        ] = {}

        for kpi_name, kpi_config in self.kpis.items():
            for series_key, series_config in kpi_config.series.items():
                if (
                    series_config.decision_settings.threshold_direction
                    != self.threshold_direction
                ):
                    raise ValueError(
                        "Isolation Forest decision settings must match the "
                        "saved run-level threshold_direction. "
                        f"KPI={kpi_name!r}, series={series_key!r}."
                    )

                model_file_key = (
                    series_config
                    .model_file
                    .as_posix()
                )

                previous_owner = model_file_owners.get(
                    model_file_key
                )

                if previous_owner is not None:
                    raise ValueError(
                        "Each saved Isolation Forest site series must "
                        "reference a unique model_file. "
                        f"File={model_file_key!r}, "
                        f"first=({previous_owner[0]!r}, "
                        f"{previous_owner[1]!r}), "
                        f"duplicate=({kpi_name!r}, {series_key!r})."
                    )

                model_file_owners[
                    model_file_key
                ] = (
                    kpi_name,
                    series_key,
                )

        return self


__all__ = [
    "IsolationForestDecisionSettings",
    "IsolationForestInferenceConfig",
    "IsolationForestKPIInferenceConfig",
    "IsolationForestSeriesIdentity",
    "IsolationForestSeriesInferenceConfig",
]