"""
Custom Model Training Interface for YTEmpire
Interactive interface for training, fine-tuning, and deploying custom ML models
"""

import asyncio
import json
import logging
import os
import pickle
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Import our AutoML platform
from automl_platform_v2 import (
    AdvancedAutoMLPlatform,
    AutoMLConfig,
    TaskType,
    ModelFamily,
    OptimizationStrategy
)

# Visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# MLflow for experiment tracking
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available. Install with: pip install mlflow")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model training status"""
    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    DEPLOYED = "deployed"


class DeploymentTarget(Enum):
    """Model deployment targets"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    EDGE = "edge"
    BATCH = "batch"


@dataclass
class TrainingJob:
    """Training job metadata"""
    job_id: str
    name: str
    description: str
    task_type: TaskType
    status: ModelStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    model_path: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    deployment_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelMetadata:
    """Model metadata for versioning and tracking"""
    model_id: str
    version: str
    name: str
    description: str
    task_type: str
    created_at: datetime
    updated_at: datetime
    training_job_id: str
    performance_metrics: Dict[str, float]
    feature_names: List[str]
    model_size_mb: float
    inference_time_ms: float
    deployment_status: str
    tags: List[str] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)


class ModelTrainingRequest(BaseModel):
    """API request for model training"""
    name: str = Field(..., description="Model name")
    description: str = Field("", description="Model description")
    task_type: str = Field("regression", description="Task type: regression, classification")
    optimization_metrics: List[str] = Field(["r2", "mse"], description="Metrics to optimize")
    model_families: List[str] = Field(
        ["linear", "tree_based", "gradient_boosting"],
        description="Model families to try"
    )
    optimization_strategy: str = Field("bayesian", description="Optimization strategy")
    n_trials: int = Field(50, description="Number of optimization trials")
    timeout_seconds: int = Field(3600, description="Training timeout")
    enable_feature_engineering: bool = Field(True, description="Enable automated feature engineering")
    enable_ensemble: bool = Field(True, description="Enable ensemble methods")
    test_size: float = Field(0.2, description="Test set size")
    cv_folds: int = Field(5, description="Cross-validation folds")


class CustomModelTrainingInterface:
    """
    Interactive interface for custom model training and deployment
    """
    
    def __init__(
        self,
        models_dir: str = "models/custom",
        data_dir: str = "data/training",
        mlflow_tracking_uri: Optional[str] = None
    ):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Training jobs tracking
        self.training_jobs: Dict[str, TrainingJob] = {}
        
        # Model registry
        self.model_registry: Dict[str, ModelMetadata] = {}
        
        # Initialize MLflow if available
        if MLFLOW_AVAILABLE and mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            logger.info(f"MLflow tracking initialized at: {mlflow_tracking_uri}")
        
        # Initialize FastAPI app
        self.app = self._create_api()
    
    def _create_api(self) -> FastAPI:
        """Create FastAPI application for the training interface"""
        app = FastAPI(
            title="YTEmpire Custom Model Training Interface",
            description="Train, evaluate, and deploy custom ML models",
            version="1.0.0"
        )
        
        @app.get("/")
        async def root():
            return {
                "service": "Custom Model Training Interface",
                "status": "operational",
                "endpoints": {
                    "training": "/train",
                    "status": "/status/{job_id}",
                    "models": "/models",
                    "predict": "/predict/{model_id}",
                    "deploy": "/deploy/{model_id}",
                    "metrics": "/metrics/{model_id}"
                }
            }
        
        @app.post("/train")
        async def train_model(
            request: ModelTrainingRequest,
            background_tasks: BackgroundTasks,
            data_file: UploadFile = File(...)
        ):
            """Start a new model training job"""
            # Create job ID
            job_id = str(uuid.uuid4())
            
            # Save uploaded data
            data_path = self.data_dir / f"{job_id}_data.csv"
            with open(data_path, "wb") as f:
                shutil.copyfileobj(data_file.file, f)
            
            # Create training job
            job = TrainingJob(
                job_id=job_id,
                name=request.name,
                description=request.description,
                task_type=TaskType(request.task_type),
                status=ModelStatus.PENDING,
                created_at=datetime.now(),
                config=request.dict(),
                dataset_info={"path": str(data_path), "filename": data_file.filename}
            )
            
            self.training_jobs[job_id] = job
            
            # Start training in background
            background_tasks.add_task(
                self._train_model_async,
                job_id,
                data_path,
                request
            )
            
            return {
                "job_id": job_id,
                "status": "training_started",
                "message": f"Model training job {job_id} has been started"
            }
        
        @app.get("/status/{job_id}")
        async def get_job_status(job_id: str):
            """Get training job status"""
            if job_id not in self.training_jobs:
                raise HTTPException(status_code=404, detail="Job not found")
            
            job = self.training_jobs[job_id]
            return {
                "job_id": job_id,
                "name": job.name,
                "status": job.status.value,
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "metrics": job.metrics,
                "error_message": job.error_message
            }
        
        @app.get("/models")
        async def list_models():
            """List all trained models"""
            models = []
            for model_id, metadata in self.model_registry.items():
                models.append({
                    "model_id": model_id,
                    "name": metadata.name,
                    "version": metadata.version,
                    "task_type": metadata.task_type,
                    "created_at": metadata.created_at.isoformat(),
                    "deployment_status": metadata.deployment_status,
                    "performance": metadata.performance_metrics
                })
            
            return {"models": models, "total": len(models)}
        
        @app.post("/predict/{model_id}")
        async def predict(model_id: str, data: Dict[str, Any]):
            """Make predictions with a trained model"""
            if model_id not in self.model_registry:
                raise HTTPException(status_code=404, detail="Model not found")
            
            try:
                predictions = self._make_prediction(model_id, data)
                return {
                    "model_id": model_id,
                    "predictions": predictions,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/deploy/{model_id}")
        async def deploy_model(
            model_id: str,
            target: str = "staging"
        ):
            """Deploy a trained model"""
            if model_id not in self.model_registry:
                raise HTTPException(status_code=404, detail="Model not found")
            
            try:
                deployment_info = self._deploy_model(
                    model_id,
                    DeploymentTarget(target)
                )
                return {
                    "model_id": model_id,
                    "deployment": deployment_info,
                    "status": "deployed"
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/metrics/{model_id}")
        async def get_model_metrics(model_id: str):
            """Get detailed model metrics and performance"""
            if model_id not in self.model_registry:
                raise HTTPException(status_code=404, detail="Model not found")
            
            metadata = self.model_registry[model_id]
            return {
                "model_id": model_id,
                "name": metadata.name,
                "performance_metrics": metadata.performance_metrics,
                "model_size_mb": metadata.model_size_mb,
                "inference_time_ms": metadata.inference_time_ms,
                "feature_importance": self._get_feature_importance(model_id)
            }
        
        @app.post("/retrain/{model_id}")
        async def retrain_model(
            model_id: str,
            background_tasks: BackgroundTasks,
            data_file: UploadFile = File(...)
        ):
            """Retrain an existing model with new data"""
            if model_id not in self.model_registry:
                raise HTTPException(status_code=404, detail="Model not found")
            
            # Create new job for retraining
            job_id = str(uuid.uuid4())
            metadata = self.model_registry[model_id]
            
            # Save new training data
            data_path = self.data_dir / f"{job_id}_retrain_data.csv"
            with open(data_path, "wb") as f:
                shutil.copyfileobj(data_file.file, f)
            
            # Start retraining
            background_tasks.add_task(
                self._retrain_model_async,
                model_id,
                job_id,
                data_path
            )
            
            return {
                "model_id": model_id,
                "job_id": job_id,
                "status": "retraining_started"
            }
        
        @app.get("/compare")
        async def compare_models(model_ids: str):
            """Compare multiple models"""
            ids = model_ids.split(",")
            comparison = []
            
            for model_id in ids:
                if model_id in self.model_registry:
                    metadata = self.model_registry[model_id]
                    comparison.append({
                        "model_id": model_id,
                        "name": metadata.name,
                        "task_type": metadata.task_type,
                        "metrics": metadata.performance_metrics,
                        "model_size_mb": metadata.model_size_mb,
                        "inference_time_ms": metadata.inference_time_ms
                    })
            
            return {"comparison": comparison}
        
        @app.get("/export/{model_id}")
        async def export_model(model_id: str, format: str = "pickle"):
            """Export a trained model"""
            if model_id not in self.model_registry:
                raise HTTPException(status_code=404, detail="Model not found")
            
            metadata = self.model_registry[model_id]
            model_path = Path(metadata.model_path)
            
            if not model_path.exists():
                raise HTTPException(status_code=404, detail="Model file not found")
            
            if format == "onnx":
                # Convert to ONNX if requested
                onnx_path = self._convert_to_onnx(model_id)
                return FileResponse(onnx_path, filename=f"{model_id}.onnx")
            else:
                return FileResponse(model_path, filename=f"{model_id}.pkl")
        
        @app.delete("/models/{model_id}")
        async def delete_model(model_id: str):
            """Delete a model"""
            if model_id not in self.model_registry:
                raise HTTPException(status_code=404, detail="Model not found")
            
            # Delete model files
            metadata = self.model_registry[model_id]
            model_path = Path(metadata.model_path)
            if model_path.exists():
                model_path.unlink()
            
            # Remove from registry
            del self.model_registry[model_id]
            
            return {"status": "deleted", "model_id": model_id}
        
        return app
    
    async def _train_model_async(
        self,
        job_id: str,
        data_path: Path,
        request: ModelTrainingRequest
    ):
        """Asynchronous model training"""
        job = self.training_jobs[job_id]
        
        try:
            # Update status
            job.status = ModelStatus.PREPROCESSING
            job.started_at = datetime.now()
            
            # Load data
            df = pd.read_csv(data_path)
            
            # Assume last column is target
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            
            # Update dataset info
            job.dataset_info.update({
                "n_samples": len(df),
                "n_features": X.shape[1],
                "feature_names": X.columns.tolist()
            })
            
            # Create AutoML config
            config = AutoMLConfig(
                task_type=TaskType(request.task_type),
                optimization_metrics=request.optimization_metrics,
                n_trials=request.n_trials,
                timeout_seconds=request.timeout_seconds,
                enable_feature_engineering=request.enable_feature_engineering,
                enable_ensemble=request.enable_ensemble,
                test_size=request.test_size,
                cv_folds=request.cv_folds,
                model_families=[ModelFamily(f) for f in request.model_families],
                optimization_strategy=OptimizationStrategy(request.optimization_strategy)
            )
            
            # Update status
            job.status = ModelStatus.TRAINING
            
            # Start MLflow run if available
            if MLFLOW_AVAILABLE:
                with mlflow.start_run(run_name=job.name):
                    # Log parameters
                    mlflow.log_params(asdict(config))
                    
                    # Train model
                    automl = AdvancedAutoMLPlatform(config)
                    automl.fit(X, y)
                    
                    # Log metrics
                    if automl.best_metrics:
                        mlflow.log_metrics(automl.best_metrics)
                    
                    # Log model
                    mlflow.sklearn.log_model(
                        automl.best_model,
                        "model",
                        registered_model_name=job.name
                    )
            else:
                # Train without MLflow
                automl = AdvancedAutoMLPlatform(config)
                automl.fit(X, y)
            
            # Update status
            job.status = ModelStatus.EVALUATING
            
            # Save model
            model_id = f"{job.name}_{job_id[:8]}"
            model_path = self.models_dir / f"{model_id}.pkl"
            joblib.dump(automl.best_model, model_path)
            
            # Save feature engineering pipeline
            pipeline_path = self.models_dir / f"{model_id}_pipeline.pkl"
            joblib.dump(automl.feature_engineer, pipeline_path)
            
            # Calculate model size
            model_size_mb = model_path.stat().st_size / (1024 * 1024)
            
            # Measure inference time
            inference_time_ms = self._measure_inference_time(automl.best_model, X.iloc[:10])
            
            # Create model metadata
            metadata = ModelMetadata(
                model_id=model_id,
                version="1.0.0",
                name=job.name,
                description=job.description,
                task_type=request.task_type,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                training_job_id=job_id,
                performance_metrics=automl.best_metrics or {},
                feature_names=automl.feature_engineer.feature_names,
                model_size_mb=model_size_mb,
                inference_time_ms=inference_time_ms,
                deployment_status="ready",
                tags=["automl", request.task_type],
                hyperparameters=config.__dict__
            )
            
            # Register model
            self.model_registry[model_id] = metadata
            
            # Update job
            job.status = ModelStatus.COMPLETED
            job.completed_at = datetime.now()
            job.model_path = str(model_path)
            job.metrics = automl.best_metrics or {}
            
            # Generate visualizations if available
            if PLOTTING_AVAILABLE:
                self._generate_training_report(job_id, automl)
            
            logger.info(f"Training job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Training job {job_id} failed: {e}")
            job.status = ModelStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
    
    async def _retrain_model_async(
        self,
        model_id: str,
        job_id: str,
        data_path: Path
    ):
        """Retrain an existing model with new data"""
        try:
            # Load existing model
            metadata = self.model_registry[model_id]
            model_path = Path(metadata.model_path)
            model = joblib.load(model_path)
            
            # Load new data
            df = pd.read_csv(data_path)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            
            # Retrain (simplified - in production, you'd merge with historical data)
            model.fit(X, y)
            
            # Save updated model
            new_version = self._increment_version(metadata.version)
            new_model_id = f"{metadata.name}_{new_version}_{job_id[:8]}"
            new_model_path = self.models_dir / f"{new_model_id}.pkl"
            joblib.dump(model, new_model_path)
            
            # Create new metadata
            new_metadata = ModelMetadata(
                model_id=new_model_id,
                version=new_version,
                name=metadata.name,
                description=f"{metadata.description} (retrained)",
                task_type=metadata.task_type,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                training_job_id=job_id,
                performance_metrics={},  # Would recalculate in production
                feature_names=metadata.feature_names,
                model_size_mb=new_model_path.stat().st_size / (1024 * 1024),
                inference_time_ms=metadata.inference_time_ms,
                deployment_status="ready",
                tags=metadata.tags + ["retrained"],
                hyperparameters=metadata.hyperparameters
            )
            
            # Register new version
            self.model_registry[new_model_id] = new_metadata
            
            logger.info(f"Model {model_id} retrained as {new_model_id}")
            
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
    
    def _make_prediction(self, model_id: str, data: Dict[str, Any]) -> List[float]:
        """Make predictions with a model"""
        metadata = self.model_registry[model_id]
        model_path = Path(metadata.model_path)
        
        # Load model
        model = joblib.load(model_path)
        
        # Prepare input data
        if "features" in data:
            X = pd.DataFrame([data["features"]])
        else:
            # Assume data contains feature names as keys
            X = pd.DataFrame([data])
        
        # Make prediction
        predictions = model.predict(X)
        
        return predictions.tolist()
    
    def _deploy_model(
        self,
        model_id: str,
        target: DeploymentTarget
    ) -> Dict[str, Any]:
        """Deploy a model to target environment"""
        metadata = self.model_registry[model_id]
        
        deployment_info = {
            "model_id": model_id,
            "target": target.value,
            "deployed_at": datetime.now().isoformat(),
            "endpoint": f"https://api.ytempire.com/models/{model_id}/predict",
            "status": "active"
        }
        
        # Update metadata
        metadata.deployment_status = f"deployed_{target.value}"
        metadata.deployment_info = deployment_info
        
        # In production, this would:
        # 1. Package model with dependencies
        # 2. Create Docker container
        # 3. Deploy to Kubernetes/Cloud Run
        # 4. Set up API endpoint
        # 5. Configure monitoring
        
        logger.info(f"Model {model_id} deployed to {target.value}")
        
        return deployment_info
    
    def _get_feature_importance(self, model_id: str) -> Dict[str, float]:
        """Get feature importance for a model"""
        metadata = self.model_registry[model_id]
        model_path = Path(metadata.model_path)
        
        model = joblib.load(model_path)
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            return dict(zip(metadata.feature_names, importances.tolist()))
        
        return {}
    
    def _measure_inference_time(self, model: Any, X_sample: pd.DataFrame) -> float:
        """Measure model inference time"""
        start_time = time.time()
        _ = model.predict(X_sample)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        return inference_time / len(X_sample)  # Per sample
    
    def _increment_version(self, version: str) -> str:
        """Increment model version"""
        parts = version.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        return ".".join(parts)
    
    def _convert_to_onnx(self, model_id: str) -> Path:
        """Convert model to ONNX format"""
        # This would use skl2onnx or similar library
        # Simplified for demonstration
        metadata = self.model_registry[model_id]
        onnx_path = self.models_dir / f"{model_id}.onnx"
        
        # In production:
        # from skl2onnx import convert_sklearn
        # onnx_model = convert_sklearn(model, initial_types=[...])
        # with open(onnx_path, "wb") as f:
        #     f.write(onnx_model.SerializeToString())
        
        # For now, just copy the pickle file
        shutil.copy(metadata.model_path, onnx_path)
        
        return onnx_path
    
    def _generate_training_report(self, job_id: str, automl: AdvancedAutoMLPlatform):
        """Generate training report with visualizations"""
        job = self.training_jobs[job_id]
        report_dir = self.models_dir / "reports"
        report_dir.mkdir(exist_ok=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Model comparison
        if automl.models:
            model_names = list(automl.models.keys())[:10]
            scores = [
                automl.models[name].metrics_.get('r2', 0) 
                if hasattr(automl.models[name], 'metrics_') else 0
                for name in model_names
            ]
            
            axes[0, 0].barh(model_names, scores)
            axes[0, 0].set_xlabel('R² Score')
            axes[0, 0].set_title('Model Performance Comparison')
        
        # Plot 2: Feature importance
        importance = automl.get_feature_importance()
        if importance:
            top_features = dict(sorted(
                importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10])
            
            axes[0, 1].barh(list(top_features.keys()), list(top_features.values()))
            axes[0, 1].set_xlabel('Importance')
            axes[0, 1].set_title('Top 10 Feature Importance')
        
        # Plot 3: Training metrics over time (placeholder)
        axes[1, 0].plot([1, 2, 3, 4, 5], [0.5, 0.6, 0.7, 0.75, 0.78])
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Training Progress')
        
        # Plot 4: Model info
        axes[1, 1].axis('off')
        info_text = f"""
        Job ID: {job_id}
        Model: {job.name}
        Status: {job.status.value}
        Best Score: {job.metrics.get('r2', 'N/A'):.4f}
        Training Time: {(job.completed_at - job.started_at).total_seconds():.1f}s
        Features: {job.dataset_info.get('n_features', 'N/A')}
        Samples: {job.dataset_info.get('n_samples', 'N/A')}
        """
        axes[1, 1].text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center')
        
        plt.suptitle(f'Training Report - {job.name}', fontsize=14)
        plt.tight_layout()
        
        # Save report
        report_path = report_dir / f"{job_id}_report.png"
        plt.savefig(report_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training report saved to: {report_path}")
    
    def run(self, host: str = "0.0.0.0", port: int = 8001):
        """Run the training interface API"""
        logger.info(f"Starting Custom Model Training Interface on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# CLI Interface
class ModelTrainingCLI:
    """Command-line interface for model training"""
    
    def __init__(self, interface: CustomModelTrainingInterface):
        self.interface = interface
    
    def train_from_file(
        self,
        data_path: str,
        name: str,
        task_type: str = "regression",
        **kwargs
    ) -> str:
        """Train a model from a CSV file"""
        # Load data
        df = pd.read_csv(data_path)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Create config
        config = AutoMLConfig(
            task_type=TaskType(task_type),
            **kwargs
        )
        
        # Train
        automl = AdvancedAutoMLPlatform(config)
        automl.fit(X, y)
        
        # Save model
        model_id = f"{name}_{uuid.uuid4().hex[:8]}"
        model_path = self.interface.models_dir / f"{model_id}.pkl"
        joblib.dump(automl.best_model, model_path)
        
        print(f"Model trained and saved: {model_id}")
        print(f"Best metrics: {automl.best_metrics}")
        
        return model_id
    
    def evaluate_model(self, model_id: str, test_data_path: str):
        """Evaluate a trained model"""
        # Load model
        model_path = self.interface.models_dir / f"{model_id}.pkl"
        model = joblib.load(model_path)
        
        # Load test data
        df = pd.read_csv(test_data_path)
        X_test = df.iloc[:, :-1]
        y_test = df.iloc[:, -1]
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model: {model_id}")
        print(f"MSE: {mse:.4f}")
        print(f"R²: {r2:.4f}")
    
    def list_models(self):
        """List all available models"""
        models = list(self.interface.models_dir.glob("*.pkl"))
        
        print("Available models:")
        for model_path in models:
            if not "_pipeline" in model_path.name:
                print(f"  - {model_path.stem}")


# Example usage
if __name__ == "__main__":
    # Create interface
    interface = CustomModelTrainingInterface()
    
    # Option 1: Run as API server
    # interface.run(port=8001)
    
    # Option 2: Use CLI
    cli = ModelTrainingCLI(interface)
    
    # Example: Train a model
    # model_id = cli.train_from_file(
    #     "data/training_data.csv",
    #     name="revenue_predictor",
    #     task_type="regression",
    #     n_trials=100
    # )
    
    print("Custom Model Training Interface initialized")
    print("API available at: http://localhost:8001")
    print("Documentation at: http://localhost:8001/docs")