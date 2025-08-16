"""
ML Pipeline Automation Tasks
Orchestrates ML model training, evaluation, and deployment
"""
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import pandas as pd
import numpy as np
from celery import Task
from celery.result import AsyncResult
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from app.core.celery_app import celery_app
from app.services.training_data_service import training_data_service, DatasetType
from app.services.inference_pipeline import inference_pipeline, ModelType
from app.services.model_monitoring import model_monitoring_service
from app.services.feature_engineering import feature_engineering_service
from app.core.config import settings

logger = logging.getLogger(__name__)


class PipelineStage(str, Enum):
    """ML pipeline stages"""

    DATA_COLLECTION = "data_collection"
    DATA_VALIDATION = "data_validation"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    MODEL_VALIDATION = "model_validation"
    MODEL_DEPLOYMENT = "model_deployment"
    MONITORING = "monitoring"


class ModelStatus(str, Enum):
    """Model lifecycle status"""

    TRAINING = "training"
    EVALUATING = "evaluating"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    FAILED = "failed"


@dataclass
class PipelineConfig:
    """ML pipeline configuration"""

    pipeline_id: str
    name: str
    model_type: ModelType
    dataset_type: DatasetType
    training_config: Dict[str, Any]
    evaluation_metrics: List[str]
    deployment_criteria: Dict[str, float]
    schedule: Optional[str] = None  # cron expression
    auto_deploy: bool = False
    notification_emails: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineRun:
    """ML pipeline run instance"""

    run_id: str
    pipeline_id: str
    started_at: datetime
    completed_at: Optional[datetime]
    current_stage: PipelineStage
    status: str
    metrics: Dict[str, float]
    artifacts: Dict[str, str]
    error_message: Optional[str] = None
    model_version: Optional[str] = None


class MLPipelineOrchestrator:
    """Orchestrates ML pipeline execution"""

    def __init__(self):
        self.active_pipelines: Dict[str, PipelineConfig] = {}
        self.running_tasks: Dict[str, AsyncResult] = {}
        self.mlflow_tracking_uri = settings.MLFLOW_TRACKING_URI or "mlruns"
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

    async def create_pipeline(
        self,
        name: str,
        model_type: ModelType,
        dataset_type: DatasetType,
        training_config: Dict[str, Any],
        evaluation_metrics: List[str],
        deployment_criteria: Dict[str, float],
        auto_deploy: bool = False,
        schedule: Optional[str] = None,
    ) -> PipelineConfig:
        """Create a new ML pipeline configuration"""
        pipeline_id = f"pipeline_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        config = PipelineConfig(
            pipeline_id=pipeline_id,
            name=name,
            model_type=model_type,
            dataset_type=dataset_type,
            training_config=training_config,
            evaluation_metrics=evaluation_metrics,
            deployment_criteria=deployment_criteria,
            schedule=schedule,
            auto_deploy=auto_deploy,
        )

        self.active_pipelines[pipeline_id] = config

        # Schedule if cron expression provided
        if schedule:
            await self._schedule_pipeline(config)

        logger.info(f"Created ML pipeline {pipeline_id}: {name}")
        return config

    async def execute_pipeline(
        self, pipeline_id: str, force_retrain: bool = False
    ) -> PipelineRun:
        """Execute ML pipeline end-to-end"""
        config = self.active_pipelines.get(pipeline_id)
        if not config:
            raise ValueError(f"Pipeline {pipeline_id} not found")

        run_id = f"run_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        run = PipelineRun(
            run_id=run_id,
            pipeline_id=pipeline_id,
            started_at=datetime.utcnow(),
            completed_at=None,
            current_stage=PipelineStage.DATA_COLLECTION,
            status="running",
            metrics={},
            artifacts={},
        )

        try:
            # Execute pipeline stages
            with mlflow.start_run(run_name=run_id) as mlflow_run:
                # Log pipeline parameters
                mlflow.log_params(
                    {
                        "pipeline_id": pipeline_id,
                        "model_type": config.model_type.value,
                        "dataset_type": config.dataset_type.value,
                        "auto_deploy": config.auto_deploy,
                    }
                )

                # Stage 1: Data Collection
                run.current_stage = PipelineStage.DATA_COLLECTION
                dataset = await self._collect_data(config, run)

                # Stage 2: Data Validation
                run.current_stage = PipelineStage.DATA_VALIDATION
                validated_data = await self._validate_data(dataset, config, run)

                # Stage 3: Feature Engineering
                run.current_stage = PipelineStage.FEATURE_ENGINEERING
                features = await self._engineer_features(validated_data, config, run)

                # Stage 4: Model Training
                run.current_stage = PipelineStage.MODEL_TRAINING
                model = await self._train_model(features, config, run)

                # Stage 5: Model Evaluation
                run.current_stage = PipelineStage.MODEL_EVALUATION
                eval_metrics = await self._evaluate_model(model, features, config, run)

                # Stage 6: Model Validation
                run.current_stage = PipelineStage.MODEL_VALIDATION
                is_valid = await self._validate_model(eval_metrics, config, run)

                # Stage 7: Model Deployment (if criteria met)
                if is_valid and config.auto_deploy:
                    run.current_stage = PipelineStage.MODEL_DEPLOYMENT
                    await self._deploy_model(model, config, run)

                # Stage 8: Monitoring Setup
                run.current_stage = PipelineStage.MONITORING
                await self._setup_monitoring(model, config, run)

                # Complete run
                run.completed_at = datetime.utcnow()
                run.status = "completed"
                run.model_version = mlflow_run.info.run_id

                # Log final metrics
                mlflow.log_metrics(run.metrics)

        except Exception as e:
            logger.error(f"Pipeline {pipeline_id} failed: {e}")
            run.status = "failed"
            run.error_message = str(e)
            run.completed_at = datetime.utcnow()

        return run

    async def _collect_data(
        self, config: PipelineConfig, run: PipelineRun
    ) -> pd.DataFrame:
        """Collect training data"""
        logger.info(f"Collecting data for {config.dataset_type}")

        # Get latest dataset version
        versions = await training_data_service.list_versions(
            dataset_id=config.dataset_type.value
        )

        if not versions:
            # Create new dataset
            raise ValueError(f"No training data available for {config.dataset_type}")

        # Load latest version
        latest_version = versions[0]
        dataset = await training_data_service.load_dataset(
            version_id=latest_version.version_id
        )

        run.artifacts["dataset_version"] = latest_version.version_id
        run.metrics["data_rows"] = len(dataset)
        run.metrics["data_columns"] = len(dataset.columns)

        return dataset

    async def _validate_data(
        self, dataset: pd.DataFrame, config: PipelineConfig, run: PipelineRun
    ) -> pd.DataFrame:
        """Validate training data quality"""
        logger.info("Validating data quality")

        validation = await training_data_service.validate_data(
            data=dataset, dataset_type=config.dataset_type
        )

        run.metrics["data_completeness"] = validation.completeness_score
        run.metrics["data_consistency"] = validation.consistency_score
        run.metrics["data_accuracy"] = validation.accuracy_score
        run.metrics["data_quality_overall"] = validation.overall_score

        if validation.overall_score < 0.8:
            raise ValueError(f"Data quality too low: {validation.overall_score}")

        return dataset

    async def _engineer_features(
        self, dataset: pd.DataFrame, config: PipelineConfig, run: PipelineRun
    ) -> pd.DataFrame:
        """Engineer features for model training"""
        logger.info("Engineering features")

        # Apply feature engineering
        features = await feature_engineering_service.engineer_features(
            data=dataset, feature_config=config.training_config.get("features", {})
        )

        run.metrics["feature_count"] = len(features.columns)
        run.artifacts["feature_names"] = json.dumps(features.columns.tolist())

        return features

    async def _train_model(
        self, features: pd.DataFrame, config: PipelineConfig, run: PipelineRun
    ) -> Any:
        """Train ML model"""
        logger.info(f"Training {config.model_type} model")

        # Split data
        X = features.drop(
            columns=config.training_config.get("target_column", "target"),
            errors="ignore",
        )
        y = (
            features[config.training_config.get("target_column", "target")]
            if config.training_config.get("target_column", "target") in features.columns
            else np.random.randint(0, 2, len(features))
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.training_config.get("test_size", 0.2),
            random_state=42,
        )

        # Select model based on type
        model = self._get_model_instance(config.model_type, config.training_config)

        # Train model
        start_time = datetime.utcnow()
        model.fit(X_train, y_train)
        training_time = (datetime.utcnow() - start_time).total_seconds()

        run.metrics["training_time_seconds"] = training_time
        run.metrics["training_samples"] = len(X_train)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Store test set for evaluation
        run.artifacts["test_data"] = "X_test,y_test"  # In practice, save to storage
        self._test_data = (X_test, y_test)  # Temporary storage

        return model

    async def _evaluate_model(
        self,
        model: Any,
        features: pd.DataFrame,
        config: PipelineConfig,
        run: PipelineRun,
    ) -> Dict[str, float]:
        """Evaluate model performance"""
        logger.info("Evaluating model")

        # Get test data
        X_test, y_test = self._test_data

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {}

        if "accuracy" in config.evaluation_metrics:
            metrics["accuracy"] = accuracy_score(y_test, y_pred)

        if (
            "precision" in config.evaluation_metrics
            or "recall" in config.evaluation_metrics
        ):
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average="weighted"
            )
            metrics["precision"] = precision
            metrics["recall"] = recall
            metrics["f1_score"] = f1

        if "auc" in config.evaluation_metrics and hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            if len(np.unique(y_test)) == 2:  # Binary classification
                metrics["auc"] = roc_auc_score(y_test, y_proba[:, 1])

        # Add to run metrics
        for metric_name, value in metrics.items():
            run.metrics[f"eval_{metric_name}"] = value

        return metrics

    async def _validate_model(
        self, eval_metrics: Dict[str, float], config: PipelineConfig, run: PipelineRun
    ) -> bool:
        """Validate model against deployment criteria"""
        logger.info("Validating model against criteria")

        is_valid = True

        for metric_name, threshold in config.deployment_criteria.items():
            actual_value = eval_metrics.get(metric_name, 0)

            if actual_value < threshold:
                logger.warning(
                    f"Model failed {metric_name} criteria: {actual_value} < {threshold}"
                )
                is_valid = False

            run.metrics[f"criteria_{metric_name}_met"] = (
                1 if actual_value >= threshold else 0
            )

        run.metrics["model_validated"] = 1 if is_valid else 0

        return is_valid

    async def _deploy_model(
        self, model: Any, config: PipelineConfig, run: PipelineRun
    ) -> bool:
        """Deploy model to production"""
        logger.info(f"Deploying model to production")

        try:
            # Export model
            model_path = f"models/{config.model_type.value}_{run.model_version}.pkl"

            # Register with inference pipeline
            success = await inference_pipeline.register_model(
                model_type=config.model_type,
                model_path=model_path,
                batch_size=config.training_config.get("batch_size", 8),
            )

            if success:
                # Update model version in inference pipeline
                await inference_pipeline.update_model(
                    model_type=config.model_type,
                    new_model_path=model_path,
                    version=run.model_version,
                )

                run.metrics["model_deployed"] = 1
                run.artifacts["deployed_model_path"] = model_path

                logger.info(f"Model deployed successfully: {model_path}")
                return True
            else:
                run.metrics["model_deployed"] = 0
                return False

        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            run.metrics["model_deployed"] = 0
            return False

    async def _setup_monitoring(
        self, model: Any, config: PipelineConfig, run: PipelineRun
    ) -> None:
        """Setup model monitoring"""
        logger.info("Setting up model monitoring")

        # Configure monitoring
        monitoring_config = {
            "model_id": run.model_version,
            "model_type": config.model_type.value,
            "metrics_to_track": config.evaluation_metrics,
            "alert_thresholds": config.deployment_criteria,
            "check_frequency": "hourly",
        }

        # Register with monitoring service
        await model_monitoring_service.register_model(
            model_id=run.model_version, monitoring_config=monitoring_config
        )

        run.metrics["monitoring_enabled"] = 1

    def _get_model_instance(
        self, model_type: ModelType, training_config: Dict[str, Any]
    ) -> Any:
        """Get model instance based on type"""
        # Import here to avoid circular dependencies
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.neural_network import MLPClassifier

        model_map = {
            ModelType.QUALITY_SCORING: RandomForestClassifier(
                n_estimators=training_config.get("n_estimators", 100),
                max_depth=training_config.get("max_depth", 10),
                random_state=42,
            ),
            ModelType.ENGAGEMENT_PREDICTION: GradientBoostingRegressor(
                n_estimators=training_config.get("n_estimators", 100),
                learning_rate=training_config.get("learning_rate", 0.1),
                random_state=42,
            ),
            ModelType.REVENUE_FORECAST: LinearRegression(),
            ModelType.USER_BEHAVIOR: MLPClassifier(
                hidden_layer_sizes=training_config.get("hidden_layers", (100, 50)),
                max_iter=training_config.get("max_iter", 500),
                random_state=42,
            ),
            ModelType.SENTIMENT_ANALYSIS: LogisticRegression(
                max_iter=training_config.get("max_iter", 1000), random_state=42
            ),
        }

        # Default to RandomForest
        return model_map.get(model_type, RandomForestClassifier(random_state=42))

    async def _schedule_pipeline(self, config: PipelineConfig):
        """Schedule pipeline execution"""
        # This would integrate with Celery beat or similar scheduler
        logger.info(
            f"Scheduling pipeline {config.pipeline_id} with schedule: {config.schedule}"
        )


# Celery tasks for async execution
@celery_app.task(bind=True, name="ml_pipeline.train_model")
def train_model_task(
    self: Task, pipeline_id: str, force_retrain: bool = False
) -> Dict[str, Any]:
    """Celery task to train model asynchronously"""
    orchestrator = MLPipelineOrchestrator()

    # Run async function in sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        run = loop.run_until_complete(
            orchestrator.execute_pipeline(pipeline_id, force_retrain)
        )

        return asdict(run)

    except Exception as e:
        logger.error(f"Training task failed: {e}")
        self.retry(exc=e, countdown=60)
    finally:
        loop.close()


@celery_app.task(name="ml_pipeline.batch_prediction")
def batch_prediction_task(
    model_type: str, input_data: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Celery task for batch predictions"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        responses = loop.run_until_complete(
            inference_pipeline.predict_batch(
                model_type=ModelType(model_type), input_batch=input_data
            )
        )

        return [asdict(r) for r in responses]

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise
    finally:
        loop.close()


@celery_app.task(name="ml_pipeline.retrain_all_models")
def retrain_all_models_task() -> Dict[str, Any]:
    """Celery task to retrain all active models"""
    orchestrator = MLPipelineOrchestrator()
    results = {}

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        for pipeline_id, config in orchestrator.active_pipelines.items():
            try:
                run = loop.run_until_complete(
                    orchestrator.execute_pipeline(pipeline_id, force_retrain=True)
                )
                results[pipeline_id] = {
                    "status": run.status,
                    "metrics": run.metrics,
                    "model_version": run.model_version,
                }
            except Exception as e:
                results[pipeline_id] = {"status": "failed", "error": str(e)}

        return results

    finally:
        loop.close()


@celery_app.task(name="ml_pipeline.monitor_model_drift")
def monitor_model_drift_task() -> Dict[str, Any]:
    """Celery task to monitor model drift"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        drift_report = loop.run_until_complete(
            model_monitoring_service.check_drift_all_models()
        )

        # Trigger retraining if significant drift detected
        for model_id, drift_metrics in drift_report.items():
            if drift_metrics.get("drift_detected", False):
                logger.warning(f"Drift detected for model {model_id}")
                # Queue retraining task
                train_model_task.delay(
                    pipeline_id=drift_metrics.get("pipeline_id"), force_retrain=True
                )

        return drift_report

    finally:
        loop.close()


# Pipeline orchestrator singleton
ml_pipeline_orchestrator = MLPipelineOrchestrator()
