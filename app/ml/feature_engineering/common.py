"""Shared utilities for Part C feature engineering.

Intended project location::

    app/ml/feature_engineering/common.py

This module contains reusable dataframe and site-series helpers only. It does
not decide which features to create and does not orchestrate feature modules.
Those responsibilities belong to ``runner.py``.

The main rules enforced here are:

- physical dataset columns are resolved through ``DatasetInputConfig``;
- each KPI dataset is processed independently by site;
- site rows are stably ordered by timestamp while preserving their original
  dataframe indices;
- sites marked ``kpi_usability == 'invalid'`` can be identified and skipped by
  the runner;
- value-based feature functions receive either ``raw_value`` or
  ``corrected_history`` explicitly; and
- generated feature columns must preserve the source row count and index and
  must never overwrite existing columns.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import pandas as pd

from app.schemas.common.dataset import DatasetInputConfig
from app.schemas.feature_engineering import FeatureSource


# ---------------------------------------------------------------------------
# Shared runtime models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FeatureInputColumns:
    """Resolved physical columns needed by the Part C runner."""

    site_identity_columns: tuple[str, ...]
    timestamp_column: str
    value_column: str

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Return all required physical columns without duplicates."""

        return tuple(
            dict.fromkeys(
                (
                    *self.site_identity_columns,
                    self.timestamp_column,
                    self.value_column,
                )
            )
        )


@dataclass(frozen=True, slots=True)
class SiteGroup:
    """One timestamp-ordered site series from a single KPI dataset."""

    key: tuple[Any, ...]
    composite_id: str
    dataframe: pd.DataFrame


@dataclass(frozen=True, slots=True)
class FeatureSourceBundle:
    """Available value sources for one timestamp-ordered site.

    ``raw_value`` contains the original Part B KPI value column converted to
    ``float64``. ``corrected_history`` is supplied by
    ``source_preparation.py`` only when at least one requested feature needs
    it.

    Feature generators must treat the returned Series as read-only.
    """

    raw_value: pd.Series
    corrected_history: pd.Series | None = None

    def __post_init__(self) -> None:
        """Require complete alignment between available source Series."""

        if self.raw_value.ndim != 1:
            raise ValueError(
                "raw_value must be a one-dimensional pandas Series."
            )

        if self.corrected_history is None:
            return

        if self.corrected_history.ndim != 1:
            raise ValueError(
                "corrected_history must be a one-dimensional pandas Series."
            )

        if len(self.raw_value) != len(self.corrected_history):
            raise ValueError(
                "raw_value and corrected_history must contain the same number "
                "of rows."
            )

        if not self.raw_value.index.equals(self.corrected_history.index):
            raise ValueError(
                "raw_value and corrected_history must use the same index."
            )

    def select(
        self,
        source: FeatureSource,
    ) -> pd.Series:
        """Return the explicitly requested feature source."""

        if source == "raw_value":
            return self.raw_value

        if source == "corrected_history":
            if self.corrected_history is None:
                raise ValueError(
                    "A feature requested source='corrected_history', but the "
                    "runner has not prepared the corrected-history Series."
                )

            return self.corrected_history

        raise ValueError(
            f"Unsupported feature source: {source!r}"
        )


FeatureOutput: TypeAlias = pd.Series | pd.DataFrame


# ---------------------------------------------------------------------------
# Column resolution and basic dataframe helpers
# ---------------------------------------------------------------------------


def resolve_feature_input_columns(
    dataset_config: DatasetInputConfig,
) -> FeatureInputColumns:
    """Resolve physical site, timestamp, and KPI-value columns.

    ``DatasetInputConfig`` owns the logical-to-physical dataset mapping.
    This helper converts that larger schema object into the smaller immutable
    runtime object used throughout Part C.
    """

    if not isinstance(dataset_config, DatasetInputConfig):
        raise TypeError(
            "dataset_config must be an instance of DatasetInputConfig."
        )

    site_identity_columns = tuple(
        dataset_config.columns.get_site_identity_columns()
    )

    if not site_identity_columns:
        raise ValueError(
            "Dataset column mapping must provide at least one site identity "
            "column."
        )

    return FeatureInputColumns(
        site_identity_columns=site_identity_columns,
        timestamp_column=dataset_config.columns.timestamp_column,
        value_column=dataset_config.columns.value_column,
    )


def require_columns(
    dataframe: pd.DataFrame,
    required_columns: Sequence[str],
    *,
    context: str,
) -> None:
    """Raise a clear error when required dataframe columns are absent."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame for {context}; received "
            f"{type(dataframe).__name__}."
        )

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns for {context}: {missing}. "
            f"Available columns: {dataframe.columns.tolist()}"
        )


def sort_site_dataframe(
    site_dataframe: pd.DataFrame,
    *,
    timestamp_column: str,
) -> pd.DataFrame:
    """Return one copied site dataframe in stable chronological order.

    Part C assumes Parts A and B already corrected and parsed the timeline.
    This helper only orders rows. It deliberately preserves original dataframe
    indices so generated feature values can be attached safely to the original
    Part B rows.
    """

    require_columns(
        site_dataframe,
        [timestamp_column],
        context="site sorting",
    )

    return (
        site_dataframe
        .sort_values(
            timestamp_column,
            kind="mergesort",
        )
        .copy()
    )


# ---------------------------------------------------------------------------
# Site grouping and usability
# ---------------------------------------------------------------------------


def normalize_site_key(
    group_key: Any,
) -> tuple[Any, ...]:
    """Normalize a pandas group key to one tuple representation."""

    if isinstance(group_key, tuple):
        return group_key

    return (group_key,)


def make_composite_id(
    group_key: Any,
    *,
    separator: str = " | ",
    missing_token: str = "<MISSING_ID>",
) -> str:
    """Build a readable site identifier from one pandas group key."""

    normalized_key = normalize_site_key(
        group_key
    )

    return separator.join(
        missing_token if pd.isna(value) else str(value)
        for value in normalized_key
    )


def iter_site_groups(
    dataframe: pd.DataFrame,
    *,
    site_identity_columns: Sequence[str],
    timestamp_column: str,
) -> Iterator[SiteGroup]:
    """Yield every site independently in stable timestamp order.

    This helper does not skip invalid sites. The runner should call
    :func:`site_is_invalid` for each yielded group and then apply the selected
    invalid-site output policy.
    """

    site_columns = tuple(
        site_identity_columns
    )

    if not site_columns:
        raise ValueError(
            "At least one site identity column is required for grouping."
        )

    require_columns(
        dataframe,
        [*site_columns, timestamp_column],
        context="site grouping",
    )

    grouper: str | list[str]

    if len(site_columns) == 1:
        grouper = site_columns[0]
    else:
        grouper = list(site_columns)

    grouped = dataframe.groupby(
        grouper,
        sort=False,
        dropna=False,
        observed=True,
    )

    for group_key, site_dataframe in grouped:
        normalized_key = normalize_site_key(
            group_key
        )

        yield SiteGroup(
            key=normalized_key,
            composite_id=make_composite_id(
                normalized_key
            ),
            dataframe=sort_site_dataframe(
                site_dataframe,
                timestamp_column=timestamp_column,
            ),
        )


def resolve_site_usability(
    site_dataframe: pd.DataFrame,
    *,
    usability_column: str = "kpi_usability",
) -> str:
    """Return one normalized usability value for a site.

    Parts A and B should already assign one consistent value per site. Part C
    reports inconsistent labels instead of attempting to repair them.
    """

    require_columns(
        site_dataframe,
        [usability_column],
        context="site usability resolution",
    )

    usability_values = (
        site_dataframe[usability_column]
        .dropna()
        .astype("string")
        .str.strip()
        .str.lower()
        .drop_duplicates()
        .tolist()
    )

    if not usability_values:
        raise ValueError(
            f"Site contains no usable value in '{usability_column}'."
        )

    if len(usability_values) != 1:
        raise ValueError(
            f"Site contains inconsistent '{usability_column}' values: "
            f"{usability_values}"
        )

    return str(
        usability_values[0]
    )


def site_is_invalid(
    site_dataframe: pd.DataFrame,
    *,
    usability_column: str = "kpi_usability",
    invalid_value: str = "invalid",
) -> bool:
    """Return whether the runner should skip feature calculation for a site."""

    normalized_invalid_value = invalid_value.strip().lower()

    if not normalized_invalid_value:
        raise ValueError(
            "invalid_value must be a non-empty string."
        )

    return (
        resolve_site_usability(
            site_dataframe,
            usability_column=usability_column,
        )
        == normalized_invalid_value
    )


# ---------------------------------------------------------------------------
# Feature source preparation and selection
# ---------------------------------------------------------------------------


def build_raw_value_series(
    site_dataframe: pd.DataFrame,
    *,
    value_column: str,
) -> pd.Series:
    """Return the original Part B KPI values as a numeric feature source.

    Non-numeric values become ``NaN``. Formal input validation and reporting
    belong to ``feature_validation.py``; this helper performs no filling,
    clipping, outlier removal, or timeline correction.
    """

    require_columns(
        site_dataframe,
        [value_column],
        context="raw feature-source creation",
    )

    raw_value = pd.to_numeric(
        site_dataframe[value_column],
        errors="coerce",
    ).astype("float64")

    raw_value.name = "raw_value"

    return raw_value


def build_feature_source_bundle(
    site_dataframe: pd.DataFrame,
    *,
    value_column: str,
    corrected_history: pd.Series | None = None,
) -> FeatureSourceBundle:
    """Create the aligned source bundle passed to value-based generators."""

    raw_value = build_raw_value_series(
        site_dataframe,
        value_column=value_column,
    )

    normalized_corrected_history: pd.Series | None = None

    if corrected_history is not None:
        normalized_corrected_history = pd.to_numeric(
            corrected_history,
            errors="coerce",
        ).astype("float64")

        normalized_corrected_history.name = "corrected_history"

    return FeatureSourceBundle(
        raw_value=raw_value,
        corrected_history=normalized_corrected_history,
    )


# ---------------------------------------------------------------------------
# Safe attachment of generated feature columns
# ---------------------------------------------------------------------------


def normalize_feature_output(
    feature_output: FeatureOutput,
    *,
    expected_index: pd.Index,
    context: str,
) -> pd.DataFrame:
    """Normalize and validate a feature generator's Series/DataFrame output."""

    if isinstance(feature_output, pd.Series):
        if feature_output.name is None:
            raise ValueError(
                f"Feature output for {context} is an unnamed Series."
            )

        feature_frame = feature_output.to_frame()

    elif isinstance(feature_output, pd.DataFrame):
        feature_frame = feature_output.copy()

    else:
        raise TypeError(
            f"Feature output for {context} must be a pandas Series or "
            f"DataFrame; received {type(feature_output).__name__}."
        )

    if len(feature_frame) == 0 and len(expected_index) > 0:
        raise ValueError(
            f"Feature output for {context} contains no rows."
        )

    if feature_frame.columns.empty:
        raise ValueError(
            f"Feature output for {context} contains no feature columns."
        )

    if feature_frame.columns.has_duplicates:
        duplicates = (
            feature_frame.columns[
                feature_frame.columns.duplicated()
            ]
            .tolist()
        )

        raise ValueError(
            f"Feature output for {context} contains duplicate columns: "
            f"{duplicates}"
        )

    invalid_names = [
        column
        for column in feature_frame.columns
        if not isinstance(column, str) or not column.strip()
    ]

    if invalid_names:
        raise ValueError(
            f"Feature output for {context} contains invalid column names: "
            f"{invalid_names}"
        )

    if len(feature_frame) != len(expected_index):
        raise ValueError(
            f"Feature output for {context} changed the row count: expected "
            f"{len(expected_index):,}, received {len(feature_frame):,}."
        )

    if not feature_frame.index.equals(expected_index):
        raise ValueError(
            f"Feature output for {context} is not aligned with the source "
            "site index."
        )

    return feature_frame


def append_feature_output(
    base_dataframe: pd.DataFrame,
    feature_output: FeatureOutput,
    *,
    context: str,
) -> pd.DataFrame:
    """Append validated feature columns without overwriting existing columns."""

    if not isinstance(base_dataframe, pd.DataFrame):
        raise TypeError(
            "base_dataframe must be a pandas DataFrame."
        )

    feature_frame = normalize_feature_output(
        feature_output,
        expected_index=base_dataframe.index,
        context=context,
    )

    overlapping_columns = sorted(
        set(base_dataframe.columns)
        & set(feature_frame.columns)
    )

    if overlapping_columns:
        raise ValueError(
            f"Feature output for {context} would overwrite existing columns: "
            f"{overlapping_columns}"
        )

    output = pd.concat(
        [base_dataframe, feature_frame],
        axis=1,
        copy=False,
    )

    if len(output) != len(base_dataframe):
        raise RuntimeError(
            f"Appending feature output for {context} unexpectedly changed the "
            "row count."
        )

    if not output.index.equals(base_dataframe.index):
        raise RuntimeError(
            f"Appending feature output for {context} unexpectedly changed the "
            "source index."
        )

    return output


def append_feature_outputs(
    base_dataframe: pd.DataFrame,
    feature_outputs: Mapping[str, FeatureOutput],
) -> pd.DataFrame:
    """Append multiple named feature-generator outputs in request order."""

    output = base_dataframe

    for context, feature_output in feature_outputs.items():
        output = append_feature_output(
            output,
            feature_output,
            context=context,
        )

    return output


__all__ = [
    "FeatureInputColumns",
    "FeatureOutput",
    "FeatureSourceBundle",
    "SiteGroup",
    "append_feature_output",
    "append_feature_outputs",
    "build_feature_source_bundle",
    "build_raw_value_series",
    "iter_site_groups",
    "make_composite_id",
    "normalize_feature_output",
    "normalize_site_key",
    "require_columns",
    "resolve_feature_input_columns",
    "resolve_site_usability",
    "site_is_invalid",
    "sort_site_dataframe",
]