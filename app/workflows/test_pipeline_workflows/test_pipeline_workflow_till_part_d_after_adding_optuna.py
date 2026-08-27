"""
Connect the completed Part A EDA runner, Part B preprocessing runner,
Part C feature-engineering runner, and Part D training runner for temporary
server testing, including fixed and Optuna tuning modes.

This workflow:

1. receives the validated pipeline request;
2. creates separate EDA, preprocessing, feature-engineering, and Part D roots;
3. runs Part A EDA once;
4. loops through every KPI result returned by Part A;
5. runs Part B preprocessing once for each KPI;
6. passes the exact Part B result to Part C for that KPI;
7. passes the exact Part C engineered dataset and resolved feature contract to
   Part D;
8. trains the single model selected by the request independently for every
   valid site series inside that KPI; and
9. returns all structured runner results.
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
from app.ml.training import (
    PartDTrainingResult,
    run_training,
)
from app.schemas.feature_engineering import FeatureEngineeringConfig
from app.schemas.test_pipeline_till_part_d import (
    PipelineRunRequest,
    TrainingConfig,
)
from app.services.file_service import (
    ensure_directory,
    sanitise_filename,
)


@dataclass(frozen=True, slots=True)
class KPIWorkflowResult:
    """
    Store the Part B, Part C, and Part D results for one KPI.

    Attributes:
        kpi_name:
            Original KPI name returned by Part A.

        preprocessing_result:
            Complete structured result returned by the Part B runner.

        feature_engineering_result:
            Complete structured result returned by the Part C runner.

        training_result:
            Complete structured result returned by the Part D runner.
    """

    kpi_name: str
    preprocessing_result: PreprocessingRunnerResult
    feature_engineering_result: FeatureEngineeringRunnerResult
    training_result: PartDTrainingResult


@dataclass(frozen=True, slots=True)
class TestPipelineWorkflowResult:
    """
    Store the complete result of the temporary pipeline workflow.

    Attributes:
        output_directory:
            Root output directory supplied through the request.

        eda_result:
            Complete structured result returned by Part A.

        training_config:
            Exact validated Part D training configuration used for every
            KPI, including fixed/Optuna tuning settings.

        kpi_results:
            Part B, Part C, and Part D results for every KPI returned by
            Part A.
    """

    output_directory: Path
    eda_result: EDARunnerResult
    training_config: TrainingConfig
    kpi_results: tuple[KPIWorkflowResult, ...]


def run_test_pipeline_workflow(
    *,
    request: PipelineRunRequest,
    logger: Logger,
) -> TestPipelineWorkflowResult:
    """
    Run Part A, then Part B, Part C, and Part D for every KPI.

    Part D trains exactly one model type per request. The selected training
    schema's ``model_type`` is used as the explicit runner dispatch switch.

    When the request omits ``feature_engineering``, the workflow creates a
    default ``FeatureEngineeringConfig`` whose profile matches that selected
    model type.

    Args:
        request:
            Validated dataset, preprocessing, feature-engineering, training,
            and output-directory settings.

        logger:
            Logger shared across the complete test pipeline.

    Returns:
        TestPipelineWorkflowResult:
            Structured EDA, preprocessing, feature-engineering, and training
            results.

    Raises:
        InputValidationError:
            If dataset, preprocessing, feature-engineering, or training
            validation fails.

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

    # Part D's model_output.py creates its own final ``training`` directory
    # inside the directory passed to run_training(). Using a separate Part D
    # root keeps that package boundary clear and avoids mixing it with Part C.
    part_d_root = ensure_directory(
        output_root / "part_d"
    )

    feature_config = (
        request.feature_engineering
        if request.feature_engineering is not None
        else FeatureEngineeringConfig(
            default_profile=request.training.model_type
        )
    )

    resolved_tuning_method = (
        request.training.hyperparameter_tuning_method
        if request.training.hyperparameter_tuning_enabled
        else None
    )

    logger.info(
        "Starting temporary EDA, preprocessing, feature-engineering, and "
        "training pipeline test. Model: %s. Tuning enabled: %s. "
        "Tuning method: %s. Output root: %s.",
        request.training.model_type,
        request.training.hyperparameter_tuning_enabled,
        resolved_tuning_method,
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
                feature_config=feature_config,
                bucket=(
                    eda_result.preparation.bucket
                ),
                logger=logger,
            )
        )

        kpi_part_d_output_directory = ensure_directory(
            part_d_root
            / safe_kpi_name
        )

        logger.info(
            "Starting Part D %s training for KPI '%s'. "
            "Output directory: %s.",
            request.training.model_type,
            raw_kpi_name,
            kpi_part_d_output_directory,
        )

        training_result = run_training(
            model_type=request.training.model_type,
            input_file=(
                feature_engineering_result.final_dataset_file
            ),
            output_directory=(
                kpi_part_d_output_directory
            ),
            dataset_columns=request.dataset.columns,
            feature_config=(
                feature_engineering_result.resolved_config
            ),
            minimum_history_points=(
                feature_engineering_result
                .resolved_config
                .minimum_history_points
            ),
            training_config=request.training,
            metadata={
                "kpi_name": raw_kpi_name,
                "part_c_output_directory": str(
                    feature_engineering_result.output_directory
                ),
                "part_c_final_dataset": str(
                    feature_engineering_result.final_dataset_file
                ),
                "hyperparameter_tuning_enabled": (
                    request.training.hyperparameter_tuning_enabled
                ),
                "hyperparameter_tuning_method": (
                    resolved_tuning_method
                ),
            },
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
                training_result=(
                    training_result
                ),
            )
        )

    result = TestPipelineWorkflowResult(
        output_directory=output_root,
        eda_result=eda_result,
        training_config=request.training,
        kpi_results=tuple(
            kpi_workflow_results
        ),
    )

    logger.info(
        "Temporary pipeline test completed through Part D for %d KPI "
        "dataset(s). Model: %s. Tuning method: %s. Output root: %s.",
        len(result.kpi_results),
        request.training.model_type,
        resolved_tuning_method,
        result.output_directory,
    )

    return result


__all__ = [
    "KPIWorkflowResult",
    "TestPipelineWorkflowResult",
    "run_test_pipeline_workflow",
]
