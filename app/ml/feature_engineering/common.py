"""Shared utilities for Part C feature engineering.

Intended project location::

    app/ml/feature_engineering/common.py

This module contains reusable dataframe and site-series helpers only. It does
not decide which features to create and does not orchestrate feature modules.
Those responsibilities belong to ``runner.py``.

The main rules enforced here are:

- physical dataset columns are resolved through ``DatasetInputConfig``;
- each KPI dataset is processed independently by site;
- site rows are stably ordered by timestamp;
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
from typing import Any, Literal, Protocol, TypeAlias

import pandas as pd

from app.schemas.common.dataset import DatasetInputConfig

FeatureSourceName: TypeAlias = Literal[
    "raw_value",
    "corrected_history",
]


# ---------------------------------------------------------------------------
# NOT NEEDED AS DATASET-CONFIGS IS NOW CALLED FROM APP/SCHEMA/COMMON/DATASET
# ---------------------------------------------------------------------------

# class DatasetColumnMappingProtocol(Protocol):
#     """Minimum column-mapping interface required from DatasetInputConfig."""

#     timestamp_column: str
#     value_column: str

#     def get_site_identity_columns(self) -> Sequence[str]:
#         """Return the physical columns that uniquely identify one site."""


# class DatasetConfigProtocol(Protocol):
#     """Minimum dataset configuration interface required by this module."""

#     columns: DatasetColumnMappingProtocol




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
    """Available value series for one timestamp-ordered site.

    ``raw_value`` always contains the original Part B KPI value column converted
    to ``float64``. ``corrected_history`` is supplied later by
    ``source_preparation.py`` only when at least one requested feature needs it.

    Feature functions must treat returned series as read-only.
    """

    raw_value: pd.Series
    corrected_history: pd.Series | None = None

    def __post_init__(self) -> None:
        """Require source-series alignment."""

        if self.raw_value.ndim != 1:
            raise ValueError("raw_value must be a one-dimensional Series.")

        if self.corrected_history is None:
            return

        if self.corrected_history.ndim != 1:
            raise ValueError(
                "corrected_history must be a one-dimensional Series."
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

    def select(self, source: FeatureSourceName) -> pd.Series:
        """Return the explicitly requested feature source."""

        if source == "raw_value":
            return self.raw_value

        if source == "corrected_history":
            if self.corrected_history is None:
                raise ValueError(
                    "A feature requested source='corrected_history', but the "
                    "runner has not prepared the corrected-history series."
                )
            return self.corrected_history

        raise ValueError(f"Unsupported feature source: {source!r}")


FeatureOutput: TypeAlias = pd.Series | pd.DataFrame


# ---------------------------------------------------------------------------
# Column resolution and basic dataframe helpers
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# NOT NEEDED AS DATASET-CONFIGS IS NOW CALLED FROM APP/SCHEMA/COMMON/DATASET
# ---------------------------------------------------------------------------

# def resolve_feature_input_columns(
#     dataset_config: DatasetInputConfig,
# ) -> FeatureInputColumns:
#     """Resolve physical site, timestamp, and KPI-value columns.

#     The function depends only on the public column-mapping interface already
#     used by Parts A and B. It does not hardcode Jio dataset column names.
#     """

#     site_columns = tuple(
#         dataset_config.columns.get_site_identity_columns()
#     )
#     timestamp_column = dataset_config.columns.timestamp_column
#     value_column = dataset_config.columns.value_column

#     if not site_columns:
#         raise ValueError(
#             "Dataset column mapping must provide at least one site identity "
#             "column."
#         )

#     all_columns = (
#         *site_columns,
#         timestamp_column,
#         value_column,
#     )

#     empty_names = [
#         column
#         for column in all_columns
#         if not isinstance(column, str) or not column.strip()
#     ]
#     if empty_names:
#         raise ValueError(
#             "Resolved feature-engineering column names must be non-empty "
#             "strings."
#         )

#     if len(all_columns) != len(set(all_columns)):
#         raise ValueError(
#             "Site identity, timestamp, and value columns must refer to "
#             "different physical columns."
#         )

#     return FeatureInputColumns(
#         site_identity_columns=site_columns,
#         timestamp_column=timestamp_column,
#         value_column=value_column,
#     )





def require_columns(
    dataframe: pd.DataFrame,
    required_columns: Sequence[str],
    *,
    context: str,
) -> None:
    """Raise a clear error when required dataframe columns are absent."""

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
    This helper only orders rows; timeline validity belongs to
    ``feature_validation.py``.
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
        .reset_index(drop=True)
        .copy()
    )


# ---------------------------------------------------------------------------
# Site grouping and usability
# ---------------------------------------------------------------------------


def normalize_site_key(
    group_key: Any,
) -> tuple[Any, ...]:
    """Normalize pandas group keys to one tuple representation."""

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

    normalized_key = normalize_site_key(group_key)

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
    :func:`site_is_invalid` for each yielded group and apply the selected output
    policy.
    """

    site_columns = tuple(site_identity_columns)

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
        normalized_key = normalize_site_key(group_key)

        yield SiteGroup(
            key=normalized_key,
            composite_id=make_composite_id(normalized_key),
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
    does not repair inconsistent usability labels.
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

    return str(usability_values[0])


def site_is_invalid(
    site_dataframe: pd.DataFrame,
    *,
    usability_column: str = "kpi_usability",
    invalid_value: str = "invalid",
) -> bool:
    """Return whether the site should be skipped by the runner."""

    normalized_invalid_value = invalid_value.strip().lower()

    if not normalized_invalid_value:
        raise ValueError("invalid_value must be a non-empty string.")

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
    """Return the unchanged Part B KPI values as a numeric feature source.

    Non-numeric values become ``NaN``. Their formal rejection or reporting is a
    responsibility of ``feature_validation.py``; this helper performs no
    missing-value filling or outlier correction.
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
    """Create the aligned source bundle passed to value-based features."""

    raw_value = build_raw_value_series(
        site_dataframe,
        value_column=value_column,
    )

    if corrected_history is not None:
        corrected_history = pd.to_numeric(
            corrected_history,
            errors="coerce",
        ).astype("float64")
        corrected_history.name = "corrected_history"

    return FeatureSourceBundle(
        raw_value=raw_value,
        corrected_history=corrected_history,
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
    """Normalize and validate a feature module's Series/DataFrame output."""

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
            f"DataFrame, received {type(feature_output).__name__}."
        )

    if feature_frame.empty and len(expected_index) > 0:
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

    return output


def append_feature_outputs(
    base_dataframe: pd.DataFrame,
    feature_outputs: Mapping[str, FeatureOutput],
) -> pd.DataFrame:
    """Append multiple named feature-module outputs in request order."""

    output = base_dataframe

    for context, feature_output in feature_outputs.items():
        output = append_feature_output(
            output,
            feature_output,
            context=context,
        )

    return output


__all__ = [
    "DatasetColumnMappingProtocol",
    "DatasetConfigProtocol",
    "FeatureInputColumns",
    "FeatureOutput",
    "FeatureSourceBundle",
    "FeatureSourceName",
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