"""
Connect the completed Part A EDA runner, Part B preprocessing runner,
and Part C feature-engineering runner for temporary server testing.

This workflow:

1. receives the validated pipeline request;
2. creates separate EDA, preprocessing, and feature-engineering output roots;
3. runs Part A EDA once;
4. loops through every KPI result returned by Part A;
5. runs Part B preprocessing once for each KPI;
6. passes the exact Part B result to Part C for that KPI; and
7. returns all structured runner results.
"""

from __future__ import annotations

from dataclasses import dataclass
from logging import Logger
from pathlib import Path

from app.ml.eda.runner import (
    EDARunnerResult,
    run_eda_pipeline,
)
from app.ml.feature_engineering import (
    FeatureEngineeringRunnerResult,
    run_feature_engineering_pipeline,
)
from app.ml.preprocessing.runner import (
    PreprocessingRunnerResult,
    run_preprocessing_pipeline,
)
from app.schemas.test_pipeline_till_part_c import PipelineRunRequest
from app.services.file_service import (
    ensure_directory,
    sanitise_filename,
)


@dataclass(frozen=True, slots=True)
class KPIWorkflowResult:
    """
    Store the Part B and Part C results for one KPI.

    Attributes:
        kpi_name:
            Original KPI name returned by Part A.

        preprocessing_result:
            Complete structured result returned by the Part B runner.

        feature_engineering_result:
            Complete structured result returned by the Part C runner.
    """

    kpi_name: str
    preprocessing_result: PreprocessingRunnerResult
    feature_engineering_result: FeatureEngineeringRunnerResult


@dataclass(frozen=True, slots=True)
class TestPipelineWorkflowResult:
    """
    Store the complete result of the temporary pipeline workflow.

    Attributes:
        output_directory:
            Root output directory supplied through the request.

        eda_result:
            Complete structured result returned by Part A.

        kpi_results:
            Part B and Part C results for every KPI returned by Part A.
    """

    output_directory: Path
    eda_result: EDARunnerResult
    kpi_results: tuple[KPIWorkflowResult, ...]


def run_test_pipeline_workflow(
    *,
    request: PipelineRunRequest,
    logger: Logger,
) -> TestPipelineWorkflowResult:
    """
    Run Part A, then Part B and Part C for every KPI.

    Args:
        request:
            Validated dataset, preprocessing, feature-engineering, and
            output-directory settings.

        logger:
            Logger shared across the complete test pipeline.

    Returns:
        TestPipelineWorkflowResult:
            Structured EDA, preprocessing, and feature-engineering results.

    Raises:
        InputValidationError:
            If dataset, preprocessing, or feature-engineering validation fails.

        FileProcessingError:
            If an input or output file cannot be processed.

        Exception:
            Any unexpected error raised by the existing runners.
    """

    output_root = ensure_directory(
        Path(
            request.output_directory
        ).expanduser().resolve()
    )

    eda_root = ensure_directory(
        output_root / "eda"
    )

    preprocessing_root = ensure_directory(
        output_root / "preprocessing"
    )

    feature_engineering_root = ensure_directory(
        output_root / "feature_engineering"
    )

    logger.info(
        "Starting temporary EDA, preprocessing, and feature-engineering "
        "pipeline test. Output root: %s.",
        output_root,
    )

    eda_result = run_eda_pipeline(
        config=request.dataset,
        logger=logger,
        output_directory=eda_root,
    )

    kpi_workflow_results: list[
        KPIWorkflowResult
    ] = []

    for index, kpi_result in enumerate(
        eda_result.kpi_results,
        start=1,
    ):
        raw_kpi_name = str(
            kpi_result.prepared_dataset.kpi_name
        ).strip()

        safe_kpi_name = sanitise_filename(
            raw_kpi_name
        )

        if not safe_kpi_name:
            safe_kpi_name = f"kpi_{index}"

        kpi_preprocessing_output_directory = ensure_directory(
            preprocessing_root
            / safe_kpi_name
        )

        logger.info(
            "Starting preprocessing for KPI '%s'. "
            "Output directory: %s.",
            raw_kpi_name,
            kpi_preprocessing_output_directory,
        )

        preprocessing_result = (
            run_preprocessing_pipeline(
                kpi_result=kpi_result,
                output_directory=(
                    kpi_preprocessing_output_directory
                ),
                dataset_config=request.dataset,
                preprocessing_config=(
                    request.preprocessing
                ),
                bucket=(
                    eda_result.preparation.bucket
                ),
                logger=logger,
            )
        )

        kpi_feature_engineering_output_directory = ensure_directory(
            feature_engineering_root
            / safe_kpi_name
        )

        logger.info(
            "Starting feature engineering for KPI '%s'. "
            "Output directory: %s.",
            raw_kpi_name,
            kpi_feature_engineering_output_directory,
        )

        feature_engineering_result = (
            run_feature_engineering_pipeline(
                preprocessing_result=preprocessing_result,
                output_directory=(
                    kpi_feature_engineering_output_directory
                ),
                dataset_config=request.dataset,
                feature_config=request.feature_engineering,
                bucket=(
                    eda_result.preparation.bucket
                ),
                logger=logger,
            )
        )

        kpi_workflow_results.append(
            KPIWorkflowResult(
                kpi_name=raw_kpi_name,
                preprocessing_result=(
                    preprocessing_result
                ),
                feature_engineering_result=(
                    feature_engineering_result
                ),
            )
        )

    result = TestPipelineWorkflowResult(
        output_directory=output_root,
        eda_result=eda_result,
        kpi_results=tuple(
            kpi_workflow_results
        ),
    )

    logger.info(
        "Temporary pipeline test completed through Part C for %d KPI "
        "dataset(s). Output root: %s.",
        len(result.kpi_results),
        result.output_directory,
    )

    return result


__all__ = [
    "KPIWorkflowResult",
    "TestPipelineWorkflowResult",
    "run_test_pipeline_workflow",
]