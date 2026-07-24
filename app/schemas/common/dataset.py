"""
Shared dataset schemas for the telecom anomaly-detection pipeline.

This module defines the common dataset input contract used by:

- EDA
- preprocessing
- feature engineering
- training and inference

The schemas describe request-level information such as source column
names, bucket configuration, KPI identity, and expected timeline
boundaries.

This module does not load, inspect, split, or modify CSV files. Those
operations are handled later by dataset-preparation and workflow
modules.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class DatasetColumnMapping(BaseModel):
    """
    Define the physical CSV columns used by the backend pipeline.

    The default values correspond to the current Reliance Jio AMF
    dataset structure.

    Allowing the mappings to be configured means future datasets can
    use different physical column names without requiring changes to
    EDA, preprocessing, feature-engineering, or model code.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    circle_column: str = Field(
        default="Jio_Circles",
        min_length=1,
        description=(
            "Column containing the Jio circle associated with each row."
        ),
    )

    cluster_column: str = Field(
        default="Jio_Cluster",
        min_length=1,
        description=(
            "Column containing the Jio cluster associated with each "
            "row."
        ),
    )

    site_name_column: str = Field(
        default="AMF_Site_Name",
        min_length=1,
        description=(
            "Column containing the AMF site name associated with each "
            "row."
        ),
    )

    timestamp_column: str = Field(
        default="time",
        min_length=1,
        description="Column containing the observation timestamp.",
    )

    value_column: str = Field(
        default="computedValue",
        min_length=1,
        description="Column containing the KPI observation value.",
    )

    kpi_name_column: str = Field(
        default="kpiName",
        min_length=1,
        description=(
            "Column containing the KPI name when the input file "
            "contains one or more named KPIs."
        ),
    )

    server_type_column: str = Field(
        default="JIO_AMF_Server_Type",
        min_length=1,
        description=(
            "Optional column containing the server type associated "
            "with each site."
        ),
    )

    consequent_value_column: str = Field(
        default="consequentValue",
        min_length=1,
        description=(
            "Optional column whose non-zero values are recorded as "
            "informational warnings."
        ),
    )

    bucket_type_column: str = Field(
        default="bucketType",
        min_length=1,
        description=(
            "Column containing the timeline bucket unit when bucket "
            "information is supplied through the CSV."
        ),
    )

    bucket_size_column: str = Field(
        default="bucketSize",
        min_length=1,
        description=(
            "Column containing the timeline bucket size when bucket "
            "information is supplied through the CSV."
        ),
    )

    @model_validator(mode="after")
    def validate_unique_column_mappings(
        self,
    ) -> DatasetColumnMapping:
        """
        Validate that logical fields use distinct physical columns.

        Mapping multiple logical fields to the same source column would
        make that source column's meaning ambiguous during validation
        and processing.

        Args:
            None.

        Returns:
            DatasetColumnMapping:
                The validated column-mapping instance.

        Raises:
            ValueError:
                If two or more logical fields map to the same physical
                CSV column.
        """

        configured_columns = {
            "circle_column": self.circle_column,
            "cluster_column": self.cluster_column,
            "site_name_column": self.site_name_column,
            "timestamp_column": self.timestamp_column,
            "value_column": self.value_column,
            "kpi_name_column": self.kpi_name_column,
            "server_type_column": self.server_type_column,
            "consequent_value_column": (
                self.consequent_value_column
            ),
            "bucket_type_column": self.bucket_type_column,
            "bucket_size_column": self.bucket_size_column,
        }

        mappings_by_column: dict[str, list[str]] = {}

        for logical_name, physical_name in configured_columns.items():
            mappings_by_column.setdefault(
                physical_name,
                [],
            ).append(
                logical_name
            )

        duplicate_mappings = {
            physical_name: logical_names
            for physical_name, logical_names
            in mappings_by_column.items()
            if len(logical_names) > 1
        }

        if duplicate_mappings:
            raise ValueError(
                "Multiple logical fields cannot map to the same "
                "source column. Duplicate mappings: "
                f"{duplicate_mappings}"
            )

        return self

    def get_base_required_columns(
        self,
    ) -> tuple[str, ...]:
        """
        Return columns required in every accepted source dataset.

        KPI and bucket columns are not included because their values
        may be supplied through the API request instead of the CSV.

        Args:
            None.

        Returns:
            tuple[str, ...]:
                Physical column names that must always exist in the
                source dataset.

        Raises:
            None.
        """

        return (
            self.circle_column,
            self.cluster_column,
            self.site_name_column,
            self.timestamp_column,
            self.value_column,
        )

    def get_site_identity_columns(
        self,
    ) -> tuple[str, ...]:
        """
        Return columns used to identify one unique telecom site.

        The backend defines a unique site using the combination of
        Jio circle, Jio cluster, and AMF site name.

        Args:
            None.

        Returns:
            tuple[str, ...]:
                Physical columns used to construct or identify a
                unique site.

        Raises:
            None.
        """

        return (
            self.circle_column,
            self.cluster_column,
            self.site_name_column,
        )

    def get_optional_analysis_columns(
        self,
    ) -> tuple[str, ...]:
        """
        Return optional columns used for informational checks.

        Missing optional analysis columns must not prevent the pipeline
        from continuing.

        Args:
            None.

        Returns:
            tuple[str, ...]:
                Optional physical column names used during EDA.

        Raises:
            None.
        """

        return (
            self.server_type_column,
            self.consequent_value_column,
        )

    def get_bucket_columns(
        self,
    ) -> tuple[str, str]:
        """
        Return the physical bucket type and bucket size columns.

        These columns are required from the CSV only when equivalent
        bucket information is not supplied through the request.

        Args:
            None.

        Returns:
            tuple[str, str]:
                The bucket type column followed by the bucket size
                column.

        Raises:
            None.
        """

        return (
            self.bucket_type_column,
            self.bucket_size_column,
        )


class BucketConfig(BaseModel):
    """
    Define request-level timeline bucket information.

    This configuration is optional when equivalent bucket information
    is already present in the source CSV.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    bucket_type: str = Field(
        min_length=1,
        description=(
            "Timeline bucket unit, such as Minutes, Hours, or Days."
        ),
    )

    bucket_size: int = Field(
        gt=0,
        description=(
            "Positive number of bucket units between expected "
            "timestamps."
        ),
    )


class ExpectedTimelineConfig(BaseModel):
    """
    Define optional expected boundaries for timeline analysis.

    The supplied boundaries help the EDA stage identify whether
    timestamps are missing:

    - before the first available observation;
    - after the final available observation;
    - within the available dataset.

    Either boundary may be supplied independently.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    expected_start: datetime | None = Field(
        default=None,
        description=(
            "Optional expected first timestamp of the input dataset."
        ),
    )

    expected_end: datetime | None = Field(
        default=None,
        description=(
            "Optional expected final timestamp of the input dataset."
        ),
    )

    @model_validator(mode="after")
    def validate_expected_timeline_order(
        self,
    ) -> ExpectedTimelineConfig:
        """
        Validate the order of the expected timeline boundaries.

        Args:
            None.

        Returns:
            ExpectedTimelineConfig:
                The validated expected-timeline configuration.

        Raises:
            ValueError:
                If both boundaries are provided and expected_end is
                earlier than or equal to expected_start.
        """

        if (
            self.expected_start is not None
            and self.expected_end is not None
            and self.expected_end <= self.expected_start
        ):
            raise ValueError(
                "expected_end must be later than expected_start."
            )

        return self


class DatasetInputConfig(BaseModel):
    """
    Define the common input contract for a pipeline dataset.

    This schema records:

    - source CSV path;
    - physical column mappings;
    - optional request-level bucket information;
    - optional single-KPI declaration;
    - optional expected timeline boundaries.

    The CSV contents are validated separately after the file is loaded.

    Multi-KPI inputs are also handled after loading. When multiple KPI
    names are found, the dataset-preparation stage creates a separate
    file for each KPI before downstream ML processing begins.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    input_file: Path = Field(
        description="Path to the source CSV supplied to the pipeline.",
    )

    columns: DatasetColumnMapping = Field(
        default_factory=DatasetColumnMapping,
        description=(
            "Mapping between backend field meanings and physical "
            "source CSV column names."
        ),
    )

    bucket: BucketConfig | None = Field(
        default=None,
        description=(
            "Request-level bucket information. It is required only "
            "when bucketType and bucketSize are unavailable in the CSV."
        ),
    )

    single_kpi_name: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "KPI name supplied when the entire source file contains "
            "one KPI and no kpiName column."
        ),
    )

    expected_timeline: ExpectedTimelineConfig | None = Field(
        default=None,
        description=(
            "Optional expected timeline boundaries used during data "
            "integrity analysis."
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
        """
        Convert the supplied input location into a normalized Path.

        This validator expands a leading user-home marker such as `~`.
        It does not verify that the file exists or has an accepted
        extension. Those checks remain the responsibility of the file
        service and dataset-validation stage.

        Args:
            value:
                Raw source-file location supplied as a string or Path.

        Returns:
            Path:
                The normalized source-file path.

        Raises:
            TypeError:
                If the value is not path-compatible.
            ValueError:
                If the supplied path contains invalid data.
        """

        return Path(
            value
        ).expanduser()

    def get_required_source_columns(
        self,
    ) -> tuple[str, ...]:
        """
        Return columns required from the CSV for this request.

        The KPI-name column is required only when single_kpi_name is
        not supplied.

        The bucket columns are required only when request-level bucket
        information is not supplied.

        Args:
            None.

        Returns:
            tuple[str, ...]:
                Physical source columns required for this particular
                request.

        Raises:
            None.
        """

        required_columns = list(
            self.columns.get_base_required_columns()
        )

        if self.single_kpi_name is None:
            required_columns.append(
                self.columns.kpi_name_column
            )

        if self.bucket is None:
            required_columns.extend(
                self.columns.get_bucket_columns()
            )

        return tuple(
            required_columns
        )

    def requires_kpi_column(
        self,
    ) -> bool:
        """
        Determine whether the CSV must contain a KPI-name column.

        The KPI-name column is not required when the caller explicitly
        declares that the complete source dataset belongs to one KPI.

        Args:
            None.

        Returns:
            bool:
                True when the KPI-name column must exist in the CSV;
                otherwise False.

        Raises:
            None.
        """

        return self.single_kpi_name is None

    def requires_bucket_columns(
        self,
    ) -> bool:
        """
        Determine whether bucket information must come from the CSV.

        Args:
            None.

        Returns:
            bool:
                True when request-level bucket information was not
                supplied and the CSV must contain bucketType and
                bucketSize; otherwise False.

        Raises:
            None.
        """

        return self.bucket is None