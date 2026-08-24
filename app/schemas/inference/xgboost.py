"""
Saved XGBoost inference contract.

Intended project location::

    app/schemas/inference/xgboost.py

This module contains only information required to execute already-trained
XGBoost models during inference. Training, tuning, splitting, calibration,
and testing settings do not belong here.

Model paths stored by this schema are portable paths relative to the saved-run
root. The inference loader resolves them against the run directory obtained
from the run database.
"""

from __future__ import annotations

from math import isclose
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


class XGBoostSeriesIdentity(BaseModel):
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
        """Return the identity in the same logical order used by Part D."""

        return (
            self.circle,
            self.cluster,
            self.site_name,
        )


class XGBoostDecisionSettings(BaseModel):
    """
    Persist the residual-calibration contract used by XGBoost inference.

    The current Part D calibration contract defines::

        residual = actual_target - predicted_target

    and applies the saved lower and/or upper quantile threshold according to
    threshold_direction.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    method: Literal["residual_quantile"] = "residual_quantile"
    residual_definition: Literal["actual_minus_prediction"] = (
        "actual_minus_prediction"
    )

    threshold_direction: OutlierDirection

    residual_quantile: float = Field(
        gt=0.5,
        lt=1.0,
    )

    lower_quantile: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    upper_quantile: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    lower_threshold: float | None = None
    upper_threshold: float | None = None

    quantile_method: Literal["linear"] = "linear"

    lower_comparison: Literal[
        "residual <= lower_threshold"
    ] | None = None

    upper_comparison: Literal[
        "residual >= upper_threshold"
    ] | None = None

    @model_validator(mode="after")
    def validate_direction_contract(
        self,
    ) -> XGBoostDecisionSettings:
        """Require exactly the thresholds used by the saved direction."""

        expected_lower_quantile = 1.0 - self.residual_quantile
        expected_upper_quantile = self.residual_quantile

        if self.threshold_direction in {
            "lower_only",
            "both",
        }:
            if self.lower_quantile is None or self.lower_threshold is None:
                raise ValueError(
                    "Lower-side XGBoost inference requires lower_quantile "
                    "and lower_threshold."
                )

            if not isclose(
                self.lower_quantile,
                expected_lower_quantile,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "lower_quantile must equal 1 - residual_quantile."
                )

            if self.lower_comparison != "residual <= lower_threshold":
                raise ValueError(
                    "Lower-side XGBoost inference requires the saved "
                    "comparison 'residual <= lower_threshold'."
                )

        else:
            if (
                self.lower_quantile is not None
                or self.lower_threshold is not None
                or self.lower_comparison is not None
            ):
                raise ValueError(
                    "upper_only XGBoost inference cannot contain lower-side "
                    "decision settings."
                )

        if self.threshold_direction in {
            "upper_only",
            "both",
        }:
            if self.upper_quantile is None or self.upper_threshold is None:
                raise ValueError(
                    "Upper-side XGBoost inference requires upper_quantile "
                    "and upper_threshold."
                )

            if not isclose(
                self.upper_quantile,
                expected_upper_quantile,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "upper_quantile must equal residual_quantile."
                )

            if self.upper_comparison != "residual >= upper_threshold":
                raise ValueError(
                    "Upper-side XGBoost inference requires the saved "
                    "comparison 'residual >= upper_threshold'."
                )

        else:
            if (
                self.upper_quantile is not None
                or self.upper_threshold is not None
                or self.upper_comparison is not None
            ):
                raise ValueError(
                    "lower_only XGBoost inference cannot contain upper-side "
                    "decision settings."
                )

        if (
            self.lower_threshold is not None
            and self.upper_threshold is not None
            and self.lower_threshold > self.upper_threshold
        ):
            raise ValueError(
                "lower_threshold cannot be greater than upper_threshold."
            )

        return self


class XGBoostSeriesInferenceConfig(BaseModel):
    """Saved model and decision contract for one site/KPI series."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    identity: XGBoostSeriesIdentity

    model_file: Path = Field(
        description=(
            "Portable model path relative to the saved-run root."
        ),
    )

    decision_settings: XGBoostDecisionSettings

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


class XGBoostKPIInferenceConfig(BaseModel):
    """Exact XGBoost model-input contract for one KPI."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    model_feature_columns: tuple[str, ...] = Field(
        min_length=1,
        description=(
            "Exact ordered feature matrix used to train this KPI's models."
        ),
    )

    target_column: str = Field(
        min_length=1,
        description=(
            "Observed KPI column used to calculate actual-minus-prediction "
            "residuals during inference."
        ),
    )

    series: dict[str, XGBoostSeriesInferenceConfig] = Field(
        min_length=1,
        description=(
            "Series-keyed model and decision settings."
        ),
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
        values: dict[str, XGBoostSeriesInferenceConfig],
    ) -> dict[str, XGBoostSeriesInferenceConfig]:
        if any(not str(key).strip() for key in values):
            raise ValueError("XGBoost series keys cannot be blank.")
        return values

    @model_validator(mode="after")
    def validate_unique_series_identities(
        self,
    ) -> XGBoostKPIInferenceConfig:
        identities = [
            item.identity.as_tuple()
            for item in self.series.values()
        ]

        if len(identities) != len(set(identities)):
            raise ValueError(
                "A KPI cannot map the same site identity to multiple "
                "XGBoost series entries."
            )

        return self


class XGBoostInferenceConfig(BaseModel):
    """Complete model-specific saved contract for XGBoost inference."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    model_type: Literal["xgboost"] = "xgboost"

    threshold_direction: OutlierDirection

    kpis: dict[str, XGBoostKPIInferenceConfig] = Field(
        min_length=1,
    )

    @field_validator("kpis")
    @classmethod
    def validate_kpi_names(
        cls,
        values: dict[str, XGBoostKPIInferenceConfig],
    ) -> dict[str, XGBoostKPIInferenceConfig]:
        if any(not str(key).strip() for key in values):
            raise ValueError("XGBoost KPI names cannot be blank.")
        return values

    @model_validator(mode="after")
    def validate_shared_threshold_direction(
        self,
    ) -> XGBoostInferenceConfig:
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
                        "XGBoost decision settings must match the saved "
                        "run-level threshold_direction. "
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
                        "Each saved XGBoost site series must reference a "
                        "unique model_file. "
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
    "XGBoostDecisionSettings",
    "XGBoostInferenceConfig",
    "XGBoostKPIInferenceConfig",
    "XGBoostSeriesIdentity",
    "XGBoostSeriesInferenceConfig",
]