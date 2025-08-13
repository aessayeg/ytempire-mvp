"""
ML Training Pipeline for YTEmpire
Complete training pipeline with data preparation, model training, and deployment
"""

import asyncio
import json
import logging
import pickle
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from enum import Enum

# Core ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
import joblib

# Database and storage
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import redis.asyncio as redis

# ML tracking and deployment
try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available, using local tracking")

# Monitoring
from prometheus_client import Counter, Histogram, Gauge
import yaml

logger = logging.getLogger(__name__)

# Metrics
training_counter = Counter('ml_training_jobs_total', 'Total ML training jobs', ['model_type', 'status'])
training_duration = Histogram('ml_training_duration_seconds', 'Training duration', ['model_type'])
model_accuracy = Gauge('ml_model_accuracy', 'Model accuracy score', ['model_name'])
data_pipeline_duration = Histogram('ml_data_pipeline_seconds', 'Data pipeline duration', ['pipeline'])


class ModelType(Enum):
    """Supported model types"""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"
    NLP = "nlp"
    RECOMMENDATION = "recommendation"


class DeploymentTarget(Enum):
    """Deployment targets"""
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"
    CANARY = "canary"


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    model_name: str
    model_type: ModelType
    algorithm: str  # specific algorithm to use
    data_source: Dict[str, Any]
    feature_columns: List[str]
    target_column: str
    
    # Data splitting
    test_size: float = 0.2
    validation_size: float = 0.1
    stratify: bool = False
    random_state: int = 42
    
    # Training parameters
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    cross_validation: bool = True
    cv_folds: int = 5
    scoring_metric: str = "accuracy"
    
    # Optimization
    hyperparameter_tuning: bool = True
    param_grid: Dict[str, List[Any]] = field(default_factory=dict)
    n_iter: int = 20  # for RandomizedSearchCV
    
    # Deployment
    auto_deploy: bool = True
    deployment_target: DeploymentTarget = DeploymentTarget.STAGING
    min_performance_threshold: float = 0.8
    
    # Tracking
    experiment_name: str = "default"
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Storage
    model_storage_path: str = "models"
    artifact_storage: str = "artifacts"


@dataclass
class TrainingResult:
    """Result from training pipeline"""
    success: bool
    model_id: str
    model_version: str
    run_id: Optional[str]
    
    # Metrics
    train_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    
    # Timing
    training_duration: float
    data_prep_duration: float
    
    # Artifacts
    model_path: str
    scaler_path: Optional[str]
    encoder_paths: Dict[str, str]
    
    # Deployment
    deployed: bool = False
    deployment_endpoint: Optional[str] = None
    deployment_timestamp: Optional[datetime] = None
    
    # Metadata
    feature_importance: Optional[Dict[str, float]] = None
    best_hyperparameters: Optional[Dict[str, Any]] = None
    cross_validation_scores: Optional[List[float]] = None


class MLTrainingPipeline:
    """
    Complete ML training pipeline with data preparation, training, and deployment
    """
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        redis_url: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        storage_path: str = "models"
    ):
        self.database_url = database_url
        self.redis_url = redis_url
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.db_engine = None
        self.redis_client = None
        self.mlflow_client = None
        
        if database_url:
            self.db_engine = create_async_engine(database_url)
        
        if MLFLOW_AVAILABLE and mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            self.mlflow_client = MlflowClient()
    
    async def initialize(self):
        """Initialize async components"""
        if self.redis_url:
            self.redis_client = await redis.from_url(self.redis_url)
    
    async def train_model(self, config: TrainingConfig) -> TrainingResult:
        """
        Execute complete training pipeline
        """
        logger.info(f"Starting training pipeline for {config.model_name}")
        training_counter.labels(model_type=config.model_type.value, status="started").inc()
        
        start_time = datetime.now()
        run_id = None
        
        try:
            # Start MLflow run if available
            if MLFLOW_AVAILABLE and self.mlflow_client:
                experiment = mlflow.set_experiment(config.experiment_name)
                run = mlflow.start_run(tags=config.tags)
                run_id = run.info.run_id
                
                # Log configuration
                mlflow.log_params({
                    "model_type": config.model_type.value,
                    "algorithm": config.algorithm,
                    "test_size": config.test_size,
                    "cv_folds": config.cv_folds
                })
            
            # Step 1: Load and prepare data
            logger.info("Loading and preparing data...")
            data_start = datetime.now()
            
            X_train, X_val, X_test, y_train, y_val, y_test, preprocessors = \
                await self._prepare_data(config)
            
            data_prep_duration = (datetime.now() - data_start).total_seconds()
            data_pipeline_duration.labels(pipeline="data_preparation").observe(data_prep_duration)
            
            # Log data statistics
            if MLFLOW_AVAILABLE:
                mlflow.log_metrics({
                    "train_samples": len(X_train),
                    "val_samples": len(X_val),
                    "test_samples": len(X_test),
                    "n_features": X_train.shape[1]
                })
            
            # Step 2: Hyperparameter tuning
            best_params = config.hyperparameters.copy()
            
            if config.hyperparameter_tuning and config.param_grid:
                logger.info("Performing hyperparameter tuning...")
                best_params = await self._tune_hyperparameters(
                    config, X_train, y_train, X_val, y_val
                )
                
                if MLFLOW_AVAILABLE:
                    mlflow.log_params(best_params)
            
            # Step 3: Train model
            logger.info(f"Training {config.algorithm} model...")
            training_start = datetime.now()
            
            model = self._create_model(config.algorithm, best_params)
            model.fit(X_train, y_train)
            
            training_time = (datetime.now() - training_start).total_seconds()
            training_duration.labels(model_type=config.model_type.value).observe(training_time)
            
            # Step 4: Evaluate model
            logger.info("Evaluating model...")
            
            train_metrics = self._evaluate_model(model, X_train, y_train, config.model_type)
            val_metrics = self._evaluate_model(model, X_val, y_val, config.model_type)
            test_metrics = self._evaluate_model(model, X_test, y_test, config.model_type)
            
            # Log metrics
            if MLFLOW_AVAILABLE:
                for prefix, metrics in [
                    ("train", train_metrics),
                    ("val", val_metrics),
                    ("test", test_metrics)
                ]:
                    for name, value in metrics.items():
                        mlflow.log_metric(f"{prefix}_{name}", value)
            
            # Update Prometheus metrics
            if config.model_type == ModelType.CLASSIFICATION:
                model_accuracy.labels(model_name=config.model_name).set(
                    test_metrics.get("accuracy", 0)
                )
            
            # Step 5: Cross-validation
            cv_scores = None
            if config.cross_validation:
                logger.info("Performing cross-validation...")
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=config.cv_folds,
                    scoring=config.scoring_metric
                )
                
                if MLFLOW_AVAILABLE:
                    mlflow.log_metric("cv_mean_score", cv_scores.mean())
                    mlflow.log_metric("cv_std_score", cv_scores.std())
            
            # Step 6: Extract feature importance
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(
                    config.feature_columns,
                    model.feature_importances_
                ))
            
            # Step 7: Save model and artifacts
            logger.info("Saving model and artifacts...")
            
            model_id = self._generate_model_id(config.model_name)
            model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            model_path, artifact_paths = await self._save_model_artifacts(
                model, preprocessors, config, model_id, model_version
            )
            
            # Log model to MLflow
            if MLFLOW_AVAILABLE:
                mlflow.sklearn.log_model(
                    model, "model",
                    registered_model_name=config.model_name
                )
            
            # Step 8: Deploy if criteria met
            deployed = False
            deployment_endpoint = None
            deployment_timestamp = None
            
            if config.auto_deploy and self._check_deployment_criteria(
                test_metrics, config
            ):
                logger.info("Deploying model...")
                deployment_endpoint = await self._deploy_model(
                    model, config, model_id, model_version
                )
                deployed = True
                deployment_timestamp = datetime.now()
            
            # Create result
            result = TrainingResult(
                success=True,
                model_id=model_id,
                model_version=model_version,
                run_id=run_id,
                train_metrics=train_metrics,
                validation_metrics=val_metrics,
                test_metrics=test_metrics,
                training_duration=training_time,
                data_prep_duration=data_prep_duration,
                model_path=str(model_path),
                scaler_path=str(artifact_paths.get("scaler")),
                encoder_paths=artifact_paths.get("encoders", {}),
                deployed=deployed,
                deployment_endpoint=deployment_endpoint,
                deployment_timestamp=deployment_timestamp,
                feature_importance=feature_importance,
                best_hyperparameters=best_params,
                cross_validation_scores=cv_scores.tolist() if cv_scores is not None else None
            )
            
            # Store result in Redis if available
            if self.redis_client:
                await self._cache_training_result(result)
            
            training_counter.labels(
                model_type=config.model_type.value,
                status="success"
            ).inc()
            
            logger.info(f"Training pipeline completed successfully for {config.model_name}")
            
            return result
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            training_counter.labels(
                model_type=config.model_type.value,
                status="failed"
            ).inc()
            
            return TrainingResult(
                success=False,
                model_id="",
                model_version="",
                run_id=run_id,
                train_metrics={},
                validation_metrics={},
                test_metrics={},
                training_duration=0,
                data_prep_duration=0,
                model_path="",
                scaler_path=None,
                encoder_paths={}
            )
        
        finally:
            if MLFLOW_AVAILABLE and run_id:
                mlflow.end_run()
    
    async def _prepare_data(
        self,
        config: TrainingConfig
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Load and prepare data for training
        """
        # Load data based on source type
        if config.data_source["type"] == "database":
            data = await self._load_from_database(config.data_source)
        elif config.data_source["type"] == "csv":
            data = pd.read_csv(config.data_source["path"])
        elif config.data_source["type"] == "parquet":
            data = pd.read_parquet(config.data_source["path"])
        else:
            raise ValueError(f"Unsupported data source: {config.data_source['type']}")
        
        # Separate features and target
        X = data[config.feature_columns]
        y = data[config.target_column]
        
        # Handle missing values
        X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
        
        # Encode categorical variables
        encoders = {}
        categorical_columns = X.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=y if config.stratify and config.model_type == ModelType.CLASSIFICATION else None
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=config.validation_size / (1 - config.test_size),
            random_state=config.random_state,
            stratify=y_temp if config.stratify and config.model_type == ModelType.CLASSIFICATION else None
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        preprocessors = {
            "scaler": scaler,
            "encoders": encoders
        }
        
        return X_train, X_val, X_test, y_train, y_val, y_test, preprocessors
    
    async def _load_from_database(self, data_source: Dict[str, Any]) -> pd.DataFrame:
        """Load data from database"""
        if not self.db_engine:
            raise ValueError("Database engine not configured")
        
        query = data_source.get("query")
        table = data_source.get("table")
        
        async with AsyncSession(self.db_engine) as session:
            if query:
                result = await session.execute(text(query))
            elif table:
                result = await session.execute(text(f"SELECT * FROM {table}"))
            else:
                raise ValueError("Either 'query' or 'table' must be specified")
            
            rows = result.fetchall()
            columns = result.keys()
            
        return pd.DataFrame(rows, columns=columns)
    
    def _create_model(self, algorithm: str, params: Dict[str, Any]):
        """Create model instance based on algorithm"""
        models = {
            "random_forest_regressor": RandomForestRegressor,
            "gradient_boosting_regressor": GradientBoostingRegressor,
            "linear_regression": LinearRegression,
            "logistic_regression": LogisticRegression,
            "random_forest_classifier": RandomForestRegressor,
            "gradient_boosting_classifier": GradientBoostingRegressor,
        }
        
        model_class = models.get(algorithm)
        if not model_class:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        return model_class(**params)
    
    def _evaluate_model(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        model_type: ModelType
    ) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = model.predict(X)
        metrics = {}
        
        if model_type == ModelType.REGRESSION:
            metrics["mse"] = mean_squared_error(y, predictions)
            metrics["mae"] = mean_absolute_error(y, predictions)
            metrics["r2"] = r2_score(y, predictions)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            
        elif model_type == ModelType.CLASSIFICATION:
            metrics["accuracy"] = accuracy_score(y, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y, predictions, average='weighted', zero_division=0
            )
            metrics["precision"] = precision
            metrics["recall"] = recall
            metrics["f1"] = f1
        
        return metrics
    
    async def _tune_hyperparameters(
        self,
        config: TrainingConfig,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """Perform hyperparameter tuning"""
        model = self._create_model(config.algorithm, {})
        
        grid_search = GridSearchCV(
            model,
            config.param_grid,
            cv=config.cv_folds,
            scoring=config.scoring_metric,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_params_
    
    def _check_deployment_criteria(
        self,
        metrics: Dict[str, float],
        config: TrainingConfig
    ) -> bool:
        """Check if model meets deployment criteria"""
        if config.model_type == ModelType.CLASSIFICATION:
            score = metrics.get("accuracy", 0)
        elif config.model_type == ModelType.REGRESSION:
            score = metrics.get("r2", 0)
        else:
            score = 0
        
        return score >= config.min_performance_threshold
    
    async def _deploy_model(
        self,
        model,
        config: TrainingConfig,
        model_id: str,
        model_version: str
    ) -> str:
        """Deploy model to target environment"""
        # This would integrate with your deployment infrastructure
        # For now, we'll create a simple endpoint reference
        
        endpoint = f"models/{config.model_name}/{model_version}"
        
        # Store deployment info in Redis if available
        if self.redis_client:
            deployment_info = {
                "model_id": model_id,
                "model_version": model_version,
                "endpoint": endpoint,
                "deployment_target": config.deployment_target.value,
                "deployed_at": datetime.now().isoformat()
            }
            
            await self.redis_client.set(
                f"deployment:{model_id}",
                json.dumps(deployment_info),
                ex=86400 * 30  # 30 days TTL
            )
        
        logger.info(f"Model deployed to {endpoint}")
        
        return endpoint
    
    async def _save_model_artifacts(
        self,
        model,
        preprocessors: Dict,
        config: TrainingConfig,
        model_id: str,
        model_version: str
    ) -> Tuple[Path, Dict[str, Path]]:
        """Save model and preprocessing artifacts"""
        model_dir = self.storage_path / config.model_name / model_version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.pkl"
        joblib.dump(model, model_path)
        
        # Save preprocessors
        artifact_paths = {}
        
        if preprocessors.get("scaler"):
            scaler_path = model_dir / "scaler.pkl"
            joblib.dump(preprocessors["scaler"], scaler_path)
            artifact_paths["scaler"] = scaler_path
        
        if preprocessors.get("encoders"):
            encoder_dir = model_dir / "encoders"
            encoder_dir.mkdir(exist_ok=True)
            artifact_paths["encoders"] = {}
            
            for name, encoder in preprocessors["encoders"].items():
                encoder_path = encoder_dir / f"{name}_encoder.pkl"
                joblib.dump(encoder, encoder_path)
                artifact_paths["encoders"][name] = str(encoder_path)
        
        # Save config
        config_path = model_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(asdict(config), f)
        
        # Save metadata
        metadata = {
            "model_id": model_id,
            "model_version": model_version,
            "created_at": datetime.now().isoformat(),
            "algorithm": config.algorithm,
            "model_type": config.model_type.value,
            "features": config.feature_columns,
            "target": config.target_column
        }
        
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return model_path, artifact_paths
    
    def _generate_model_id(self, model_name: str) -> str:
        """Generate unique model ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        hash_input = f"{model_name}_{timestamp}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    async def _cache_training_result(self, result: TrainingResult):
        """Cache training result in Redis"""
        if not self.redis_client:
            return
        
        key = f"training_result:{result.model_id}"
        value = json.dumps(asdict(result), default=str)
        
        await self.redis_client.set(key, value, ex=86400 * 7)  # 7 days TTL
    
    async def load_model(
        self,
        model_name: str,
        version: Optional[str] = None
    ) -> Tuple[Any, Dict]:
        """Load a trained model and its artifacts"""
        model_dir = self.storage_path / model_name
        
        if version is None:
            # Get latest version
            versions = sorted([d.name for d in model_dir.iterdir() if d.is_dir()])
            if not versions:
                raise ValueError(f"No versions found for model {model_name}")
            version = versions[-1]
        
        version_dir = model_dir / version
        
        # Load model
        model_path = version_dir / "model.pkl"
        model = joblib.load(model_path)
        
        # Load preprocessors
        artifacts = {}
        
        scaler_path = version_dir / "scaler.pkl"
        if scaler_path.exists():
            artifacts["scaler"] = joblib.load(scaler_path)
        
        encoder_dir = version_dir / "encoders"
        if encoder_dir.exists():
            artifacts["encoders"] = {}
            for encoder_path in encoder_dir.glob("*_encoder.pkl"):
                name = encoder_path.stem.replace("_encoder", "")
                artifacts["encoders"][name] = joblib.load(encoder_path)
        
        # Load metadata
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        return model, {"artifacts": artifacts, "metadata": metadata}
    
    async def get_training_history(
        self,
        model_name: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """Get training history from cache"""
        if not self.redis_client:
            return []
        
        pattern = f"training_result:*"
        if model_name:
            pattern = f"training_result:{model_name}*"
        
        keys = await self.redis_client.keys(pattern)
        results = []
        
        for key in keys[:limit]:
            value = await self.redis_client.get(key)
            if value:
                results.append(json.loads(value))
        
        # Sort by timestamp
        results.sort(
            key=lambda x: x.get("deployment_timestamp", ""),
            reverse=True
        )
        
        return results
    
    async def schedule_training(
        self,
        config: TrainingConfig,
        cron_expression: str
    ):
        """Schedule periodic model training"""
        if self.redis_client:
            # Convert config to dict and handle enums
            config_dict = asdict(config)
            config_dict["model_type"] = config.model_type.value
            config_dict["deployment_target"] = config.deployment_target.value
            
            schedule_data = {
                "config": config_dict,
                "cron": cron_expression,
                "next_run": datetime.now().isoformat(),
                "created_at": datetime.now().isoformat()
            }
            
            await self.redis_client.set(
                f"training_schedule:{config.model_name}",
                json.dumps(schedule_data),
                ex=86400 * 30  # 30 days TTL
            )
            
            logger.info(f"Scheduled training for {config.model_name}: {cron_expression}")
    
    async def get_scheduled_trainings(self) -> List[Dict]:
        """Get all scheduled training jobs"""
        if not self.redis_client:
            return []
        
        schedules = []
        keys = await self.redis_client.keys("training_schedule:*")
        
        for key in keys:
            value = await self.redis_client.get(key)
            if value:
                schedules.append(json.loads(value))
        
        return schedules
    
    async def trigger_retraining(
        self,
        model_name: str,
        reason: str,
        priority: str = "normal"
    ):
        """Trigger model retraining"""
        if self.redis_client:
            trigger_data = {
                "model_name": model_name,
                "reason": reason,
                "priority": priority,
                "triggered_at": datetime.now().isoformat()
            }
            
            # Add to retraining queue
            await self.redis_client.lpush(
                "retraining_queue",
                json.dumps(trigger_data)
            )
            
            # Publish notification
            await self.redis_client.publish(
                "model_retraining",
                json.dumps(trigger_data)
            )
            
            logger.info(f"Triggered retraining for {model_name}: {reason}")
    
    async def monitor_model_performance(
        self,
        model_name: str,
        threshold: float = 0.8
    ) -> bool:
        """Monitor model performance and trigger retraining if needed"""
        # Get latest model metrics
        history = await self.get_training_history(model_name, limit=1)
        
        if not history:
            return False
        
        latest = history[0]
        test_metrics = latest.get("test_metrics", {})
        
        # Check performance threshold
        if "accuracy" in test_metrics:
            current_score = test_metrics["accuracy"]
        elif "r2" in test_metrics:
            current_score = test_metrics["r2"]
        else:
            current_score = 0
        
        if current_score < threshold:
            await self.trigger_retraining(
                model_name,
                f"Performance dropped below threshold: {current_score:.2f} < {threshold:.2f}",
                priority="high"
            )
            return True
        
        return False


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Initialize pipeline
        pipeline = MLTrainingPipeline(
            database_url="postgresql+asyncpg://user:pass@localhost/db",
            redis_url="redis://localhost:6379",
            mlflow_tracking_uri="http://localhost:5000"
        )
        
        await pipeline.initialize()
        
        # Configure training
        config = TrainingConfig(
            model_name="video_performance_predictor",
            model_type=ModelType.REGRESSION,
            algorithm="gradient_boosting_regressor",
            data_source={
                "type": "csv",
                "path": "data/training_data.csv"
            },
            feature_columns=[
                "title_length", "description_length", "tags_count",
                "publish_hour", "channel_subscribers", "category"
            ],
            target_column="views",
            hyperparameters={
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 5
            },
            param_grid={
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7]
            },
            experiment_name="video_performance",
            tags={"team": "data", "version": "v1"}
        )
        
        # Train model
        result = await pipeline.train_model(config)
        
        print(f"Training completed: {result.success}")
        print(f"Model ID: {result.model_id}")
        print(f"Test R2 Score: {result.test_metrics.get('r2', 0):.4f}")
        
        if result.deployed:
            print(f"Deployed to: {result.deployment_endpoint}")
        
        # Schedule periodic retraining
        await pipeline.schedule_training(config, "0 0 * * 0")  # Weekly
        
        # Monitor performance
        needs_retraining = await pipeline.monitor_model_performance(
            "video_performance_predictor",
            threshold=0.75
        )
        
        if needs_retraining:
            print("Model needs retraining")
    
    asyncio.run(main())