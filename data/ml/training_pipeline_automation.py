"""
Training Pipeline Automation
End-to-end ML training pipeline with automation, versioning, and deployment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import json
import yaml
import pickle
import joblib
from pathlib import Path
import hashlib
import shutil

import torch
import tensorflow as tf
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error

import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import redis.asyncio as redis
from sqlalchemy import create_engine
import boto3
from google.cloud import storage as gcs
import wandb

from prometheus_client import Counter, Histogram, Gauge
import logging

# Metrics
training_jobs_started = Counter('training_jobs_started', 'Training jobs started', ['model_type'])
training_jobs_completed = Counter('training_jobs_completed', 'Training jobs completed', ['model_type', 'status'])
training_duration = Histogram('training_duration_seconds', 'Training duration', ['model_type'])
model_performance = Gauge('model_performance_score', 'Model performance score', ['model_name', 'metric'])
data_pipeline_duration = Histogram('data_pipeline_duration', 'Data pipeline duration', ['pipeline_name'])

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training pipeline configuration"""
    model_name: str
    model_type: str  # 'classification', 'regression', 'nlp', 'vision'
    framework: str  # 'sklearn', 'pytorch', 'tensorflow'
    data_source: Dict[str, Any]
    features: List[str]
    target: str
    hyperparameters: Dict[str, Any]
    validation_split: float = 0.2
    test_split: float = 0.1
    cross_validation: bool = True
    cv_folds: int = 5
    auto_tune: bool = True
    early_stopping: bool = True
    max_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = 'adam'
    loss_function: str = 'cross_entropy'
    metrics: List[str] = field(default_factory=lambda: ['accuracy'])
    save_path: str = '/models'
    deploy_on_success: bool = True
    deployment_target: str = 'production'
    min_performance_threshold: float = 0.8
    experiment_name: str = 'default'
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class TrainingResult:
    """Training pipeline result"""
    run_id: str
    model_name: str
    model_version: str
    metrics: Dict[str, float]
    best_params: Dict[str, Any]
    training_time: float
    model_path: str
    deployed: bool
    deployment_endpoint: Optional[str] = None
    artifacts: Dict[str, str] = field(default_factory=dict)

class TrainingPipeline:
    """Automated ML training pipeline"""
    
    def __init__(self,
                 mlflow_uri: str = 'http://localhost:5000',
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 storage_backend: str = 's3',  # 's3', 'gcs', 'local'
                 storage_config: Dict[str, Any] = None):
        
        # MLflow setup
        mlflow.set_tracking_uri(mlflow_uri)
        self.mlflow_client = MlflowClient()
        
        # Redis for caching and coordination
        self.redis_client = None
        self.redis_host = redis_host
        self.redis_port = redis_port
        
        # Storage backend
        self.storage_backend = storage_backend
        self.storage_config = storage_config or {}
        
        # Initialize storage clients
        if storage_backend == 's3':
            self.s3_client = boto3.client('s3')
            self.bucket_name = storage_config.get('bucket', 'ytempire-models')
        elif storage_backend == 'gcs':
            self.gcs_client = gcs.Client()
            self.bucket_name = storage_config.get('bucket', 'ytempire-models')
        
        # Ray for distributed training
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # Training configurations
        self.active_trainings = {}
        self.scheduled_trainings = []
    
    async def initialize(self):
        """Initialize async components"""
        self.redis_client = await redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            decode_responses=False
        )
    
    async def train_model(self, config: TrainingConfig) -> TrainingResult:
        """
        Execute training pipeline
        
        Args:
            config: Training configuration
            
        Returns:
            Training result with metrics and artifacts
        """
        training_jobs_started.labels(model_type=config.model_type).inc()
        start_time = datetime.now()
        
        try:
            # Create MLflow experiment
            experiment = mlflow.set_experiment(config.experiment_name)
            
            with mlflow.start_run(tags=config.tags) as run:
                run_id = run.info.run_id
                
                # Log configuration
                mlflow.log_params(config.hyperparameters)
                mlflow.log_param("model_type", config.model_type)
                mlflow.log_param("framework", config.framework)
                
                # Load and prepare data
                logger.info(f"Loading data for {config.model_name}")
                X_train, X_val, X_test, y_train, y_val, y_test = await self._prepare_data(config)
                
                # Log data statistics
                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("val_samples", len(X_val))
                mlflow.log_param("test_samples", len(X_test))
                
                # Hyperparameter tuning
                if config.auto_tune:
                    logger.info("Starting hyperparameter tuning")
                    best_params = await self._tune_hyperparameters(
                        config, X_train, y_train, X_val, y_val
                    )
                    config.hyperparameters.update(best_params)
                    mlflow.log_params(best_params)
                
                # Train model
                logger.info(f"Training {config.model_name}")
                model = await self._train_model(config, X_train, y_train, X_val, y_val)
                
                # Evaluate model
                logger.info("Evaluating model")
                metrics = await self._evaluate_model(model, config, X_test, y_test)
                
                # Log metrics
                for metric_name, value in metrics.items():
                    mlflow.log_metric(metric_name, value)
                    model_performance.labels(
                        model_name=config.model_name,
                        metric=metric_name
                    ).set(value)
                
                # Save model
                model_path = await self._save_model(model, config, run_id)
                
                # Log model to MLflow
                signature = infer_signature(X_train, y_train)
                
                if config.framework == 'sklearn':
                    mlflow.sklearn.log_model(model, "model", signature=signature)
                elif config.framework == 'pytorch':
                    mlflow.pytorch.log_model(model, "model", signature=signature)
                elif config.framework == 'tensorflow':
                    mlflow.tensorflow.log_model(model, "model", signature=signature)
                
                # Register model
                model_version = await self._register_model(config.model_name, run_id)
                
                # Deploy if criteria met
                deployed = False
                deployment_endpoint = None
                
                if config.deploy_on_success and self._check_deployment_criteria(metrics, config):
                    logger.info("Deploying model")
                    deployment_endpoint = await self._deploy_model(
                        model, config, model_version
                    )
                    deployed = True
                
                # Calculate training duration
                training_time = (datetime.now() - start_time).total_seconds()
                training_duration.labels(model_type=config.model_type).observe(training_time)
                
                # Create result
                result = TrainingResult(
                    run_id=run_id,
                    model_name=config.model_name,
                    model_version=model_version,
                    metrics=metrics,
                    best_params=config.hyperparameters,
                    training_time=training_time,
                    model_path=model_path,
                    deployed=deployed,
                    deployment_endpoint=deployment_endpoint,
                    artifacts={
                        'model_path': model_path,
                        'mlflow_run_id': run_id,
                        'experiment_id': experiment.experiment_id
                    }
                )
                
                # Update training status
                training_jobs_completed.labels(
                    model_type=config.model_type,
                    status='success'
                ).inc()
                
                return result
                
        except Exception as e:
            logger.error(f"Training failed: {e}")
            training_jobs_completed.labels(
                model_type=config.model_type,
                status='failed'
            ).inc()
            raise
    
    async def _prepare_data(self, config: TrainingConfig) -> Tuple:
        """Prepare training data"""
        start_time = datetime.now()
        
        # Load data based on source type
        if config.data_source['type'] == 'database':
            data = await self._load_from_database(config.data_source)
        elif config.data_source['type'] == 'file':
            data = await self._load_from_file(config.data_source)
        elif config.data_source['type'] == 'feature_store':
            data = await self._load_from_feature_store(config.data_source)
        else:
            raise ValueError(f"Unknown data source type: {config.data_source['type']}")
        
        # Extract features and target
        X = data[config.features]
        y = data[config.target]
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Encode categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=config.test_split, random_state=42, stratify=y if config.model_type == 'classification' else None
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=config.validation_split, random_state=42, stratify=y_temp if config.model_type == 'classification' else None
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        # Save scaler
        scaler_path = f"{config.save_path}/scaler_{config.model_name}.pkl"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path)
        
        # Log data pipeline duration
        duration = (datetime.now() - start_time).total_seconds()
        data_pipeline_duration.labels(pipeline_name='data_preparation').observe(duration)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    async def _load_from_database(self, data_source: Dict) -> pd.DataFrame:
        """Load data from database"""
        engine = create_engine(data_source['connection_string'])
        query = data_source.get('query', f"SELECT * FROM {data_source['table']}")
        return pd.read_sql(query, engine)
    
    async def _load_from_file(self, data_source: Dict) -> pd.DataFrame:
        """Load data from file"""
        file_path = data_source['path']
        
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            return pd.read_parquet(file_path)
        elif file_path.endswith('.json'):
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    async def _load_from_feature_store(self, data_source: Dict) -> pd.DataFrame:
        """Load data from feature store"""
        # This would integrate with the feature store implementation
        # Placeholder implementation
        return pd.DataFrame()
    
    async def _tune_hyperparameters(self,
                                   config: TrainingConfig,
                                   X_train: np.ndarray,
                                   y_train: np.ndarray,
                                   X_val: np.ndarray,
                                   y_val: np.ndarray) -> Dict[str, Any]:
        """Tune hyperparameters using Ray Tune"""
        
        if config.framework == 'sklearn':
            return await self._tune_sklearn(config, X_train, y_train)
        elif config.framework == 'pytorch':
            return await self._tune_pytorch(config, X_train, y_train, X_val, y_val)
        elif config.framework == 'tensorflow':
            return await self._tune_tensorflow(config, X_train, y_train, X_val, y_val)
        else:
            return config.hyperparameters
    
    async def _tune_sklearn(self,
                           config: TrainingConfig,
                           X_train: np.ndarray,
                           y_train: np.ndarray) -> Dict[str, Any]:
        """Tune sklearn model hyperparameters"""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Select model
        if config.model_type == 'classification':
            model = RandomForestClassifier(random_state=42)
        else:
            model = RandomForestRegressor(random_state=42)
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=config.cv_folds,
            scoring='accuracy' if config.model_type == 'classification' else 'neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_params_
    
    async def _tune_pytorch(self,
                          config: TrainingConfig,
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_val: np.ndarray,
                          y_val: np.ndarray) -> Dict[str, Any]:
        """Tune PyTorch model hyperparameters using Ray Tune"""
        
        def train_torch_model(config_dict):
            """Training function for Ray Tune"""
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
            
            # Create model
            model = nn.Sequential(
                nn.Linear(X_train.shape[1], config_dict['hidden_size']),
                nn.ReLU(),
                nn.Dropout(config_dict['dropout']),
                nn.Linear(config_dict['hidden_size'], config_dict['hidden_size'] // 2),
                nn.ReLU(),
                nn.Linear(config_dict['hidden_size'] // 2, 1)
            )
            
            # Create data loaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train)
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=config_dict['batch_size'],
                shuffle=True
            )
            
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val)
            )
            val_loader = DataLoader(val_dataset, batch_size=config_dict['batch_size'])
            
            # Training
            optimizer = optim.Adam(model.parameters(), lr=config_dict['lr'])
            criterion = nn.MSELoss()
            
            for epoch in range(10):  # Quick tuning
                model.train()
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                
                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        outputs = model(batch_x)
                        val_loss += criterion(outputs.squeeze(), batch_y).item()
                
                tune.report(loss=val_loss / len(val_loader))
        
        # Define search space
        search_space = {
            'hidden_size': tune.choice([64, 128, 256]),
            'dropout': tune.uniform(0.1, 0.5),
            'lr': tune.loguniform(1e-4, 1e-2),
            'batch_size': tune.choice([16, 32, 64])
        }
        
        # Run tuning
        analysis = tune.run(
            train_torch_model,
            config=search_space,
            num_samples=10,
            scheduler=ASHAScheduler(metric='loss', mode='min'),
            progress_reporter=CLIReporter(metric_columns=['loss'])
        )
        
        return analysis.best_config
    
    async def _tune_tensorflow(self,
                             config: TrainingConfig,
                             X_train: np.ndarray,
                             y_train: np.ndarray,
                             X_val: np.ndarray,
                             y_val: np.ndarray) -> Dict[str, Any]:
        """Tune TensorFlow model hyperparameters"""
        import keras_tuner as kt
        
        def build_model(hp):
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    hp.Int('units_1', 32, 256, step=32),
                    activation='relu',
                    input_shape=(X_train.shape[1],)
                ),
                tf.keras.layers.Dropout(hp.Float('dropout_1', 0, 0.5, step=0.1)),
                tf.keras.layers.Dense(
                    hp.Int('units_2', 32, 128, step=32),
                    activation='relu'
                ),
                tf.keras.layers.Dense(1)
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
                loss='mse',
                metrics=['mae']
            )
            
            return model
        
        tuner = kt.RandomSearch(
            build_model,
            objective='val_loss',
            max_trials=10,
            directory='tuning',
            project_name=config.model_name
        )
        
        tuner.search(
            X_train, y_train,
            epochs=10,
            validation_data=(X_val, y_val),
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)]
        )
        
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        
        return {
            'units_1': best_hps.get('units_1'),
            'units_2': best_hps.get('units_2'),
            'dropout_1': best_hps.get('dropout_1'),
            'learning_rate': best_hps.get('learning_rate')
        }
    
    async def _train_model(self,
                         config: TrainingConfig,
                         X_train: np.ndarray,
                         y_train: np.ndarray,
                         X_val: np.ndarray,
                         y_val: np.ndarray) -> Any:
        """Train the model"""
        
        if config.framework == 'sklearn':
            return self._train_sklearn_model(config, X_train, y_train)
        elif config.framework == 'pytorch':
            return await self._train_pytorch_model(config, X_train, y_train, X_val, y_val)
        elif config.framework == 'tensorflow':
            return await self._train_tensorflow_model(config, X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Unknown framework: {config.framework}")
    
    def _train_sklearn_model(self,
                           config: TrainingConfig,
                           X_train: np.ndarray,
                           y_train: np.ndarray) -> Any:
        """Train sklearn model"""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        if config.model_type == 'classification':
            model = RandomForestClassifier(**config.hyperparameters, random_state=42)
        else:
            model = RandomForestRegressor(**config.hyperparameters, random_state=42)
        
        model.fit(X_train, y_train)
        
        # Cross-validation
        if config.cross_validation:
            cv_scores = cross_val_score(model, X_train, y_train, cv=config.cv_folds)
            mlflow.log_metric("cv_score_mean", cv_scores.mean())
            mlflow.log_metric("cv_score_std", cv_scores.std())
        
        return model
    
    async def _train_pytorch_model(self,
                                 config: TrainingConfig,
                                 X_train: np.ndarray,
                                 y_train: np.ndarray,
                                 X_val: np.ndarray,
                                 y_val: np.ndarray) -> Any:
        """Train PyTorch model"""
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create model
        class NeuralNet(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)
                self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
                self.fc3 = nn.Linear(hidden_size // 2, output_size)
            
            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.fc2(x)
                x = self.relu(x)
                x = self.fc3(x)
                return x
        
        model = NeuralNet(
            X_train.shape[1],
            config.hyperparameters.get('hidden_size', 128),
            1 if config.model_type == 'regression' else len(np.unique(y_train))
        )
        
        # Prepare data
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True
        )
        
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
        
        # Training
        criterion = nn.MSELoss() if config.model_type == 'regression' else nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(config.max_epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = model(batch_x)
                    val_loss += criterion(outputs.squeeze(), batch_y).item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # Log metrics
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            
            # Early stopping
            if config.early_stopping:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
        
        return model
    
    async def _train_tensorflow_model(self,
                                    config: TrainingConfig,
                                    X_train: np.ndarray,
                                    y_train: np.ndarray,
                                    X_val: np.ndarray,
                                    y_val: np.ndarray) -> Any:
        """Train TensorFlow model"""
        
        # Create model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                config.hyperparameters.get('units_1', 128),
                activation='relu',
                input_shape=(X_train.shape[1],)
            ),
            tf.keras.layers.Dropout(config.hyperparameters.get('dropout_1', 0.2)),
            tf.keras.layers.Dense(
                config.hyperparameters.get('units_2', 64),
                activation='relu'
            ),
            tf.keras.layers.Dense(
                1 if config.model_type == 'regression' else len(np.unique(y_train)),
                activation=None if config.model_type == 'regression' else 'softmax'
            )
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(config.learning_rate),
            loss='mse' if config.model_type == 'regression' else 'sparse_categorical_crossentropy',
            metrics=['mae'] if config.model_type == 'regression' else ['accuracy']
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                f"{config.save_path}/{config.model_name}_best.h5",
                save_best_only=True
            )
        ]
        
        if config.early_stopping:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            )
        
        # MLflow callback
        callbacks.append(
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: [
                    mlflow.log_metric(key, value, step=epoch)
                    for key, value in logs.items()
                ]
            )
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.max_epochs,
            batch_size=config.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return model
    
    async def _evaluate_model(self,
                            model: Any,
                            config: TrainingConfig,
                            X_test: np.ndarray,
                            y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        metrics = {}
        
        # Get predictions
        if config.framework == 'sklearn':
            y_pred = model.predict(X_test)
        elif config.framework == 'pytorch':
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test)
                y_pred = model(X_test_tensor).numpy()
                if config.model_type == 'classification':
                    y_pred = np.argmax(y_pred, axis=1)
                else:
                    y_pred = y_pred.squeeze()
        elif config.framework == 'tensorflow':
            y_pred = model.predict(X_test)
            if config.model_type == 'classification':
                y_pred = np.argmax(y_pred, axis=1)
            else:
                y_pred = y_pred.squeeze()
        
        # Calculate metrics
        if config.model_type == 'classification':
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted'
            )
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1_score'] = f1
        else:
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = np.mean(np.abs(y_test - y_pred))
            metrics['r2'] = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
        
        return metrics
    
    async def _save_model(self, model: Any, config: TrainingConfig, run_id: str) -> str:
        """Save model to storage"""
        model_path = f"{config.save_path}/{config.model_name}_{run_id}"
        
        # Save based on framework
        if config.framework == 'sklearn':
            joblib.dump(model, f"{model_path}.pkl")
        elif config.framework == 'pytorch':
            torch.save(model.state_dict(), f"{model_path}.pth")
        elif config.framework == 'tensorflow':
            model.save(f"{model_path}.h5")
        
        # Upload to cloud storage
        if self.storage_backend == 's3':
            self.s3_client.upload_file(
                model_path,
                self.bucket_name,
                f"models/{config.model_name}/{run_id}/model"
            )
        elif self.storage_backend == 'gcs':
            bucket = self.gcs_client.bucket(self.bucket_name)
            blob = bucket.blob(f"models/{config.model_name}/{run_id}/model")
            blob.upload_from_filename(model_path)
        
        return model_path
    
    async def _register_model(self, model_name: str, run_id: str) -> str:
        """Register model in MLflow"""
        model_uri = f"runs:/{run_id}/model"
        
        # Register model
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Transition to staging
        self.mlflow_client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        
        return model_version.version
    
    def _check_deployment_criteria(self, metrics: Dict[str, float], config: TrainingConfig) -> bool:
        """Check if model meets deployment criteria"""
        # Check performance threshold
        if config.model_type == 'classification':
            return metrics.get('accuracy', 0) >= config.min_performance_threshold
        else:
            return metrics.get('r2', 0) >= config.min_performance_threshold
    
    async def _deploy_model(self, model: Any, config: TrainingConfig, version: str) -> str:
        """Deploy model to production"""
        # Transition model to production in MLflow
        self.mlflow_client.transition_model_version_stage(
            name=config.model_name,
            version=version,
            stage="Production"
        )
        
        # Deploy to serving infrastructure
        # This would integrate with your serving infrastructure (e.g., SageMaker, Vertex AI, etc.)
        deployment_endpoint = f"https://api.ytempire.com/models/{config.model_name}/v{version}"
        
        logger.info(f"Model deployed to {deployment_endpoint}")
        
        return deployment_endpoint
    
    async def schedule_training(self, config: TrainingConfig, schedule: str):
        """Schedule periodic training"""
        # This would integrate with a scheduler like Airflow or Prefect
        self.scheduled_trainings.append({
            'config': config,
            'schedule': schedule,
            'next_run': datetime.now()
        })
        
        logger.info(f"Scheduled training for {config.model_name} with schedule {schedule}")
    
    async def trigger_retraining(self, model_name: str, reason: str):
        """Trigger model retraining"""
        logger.info(f"Triggering retraining for {model_name}: {reason}")
        
        # Find configuration for model
        # This would load the saved configuration for the model
        # and trigger a new training job
        
        # Send notification
        if self.redis_client:
            await self.redis_client.publish('model_retraining', json.dumps({
                'model': model_name,
                'reason': reason,
                'timestamp': datetime.now().isoformat()
            }))

# Example usage
async def main():
    # Initialize training pipeline
    pipeline = TrainingPipeline(
        mlflow_uri='http://localhost:5000',
        storage_backend='s3',
        storage_config={'bucket': 'ytempire-models'}
    )
    
    await pipeline.initialize()
    
    # Define training configuration
    config = TrainingConfig(
        model_name='content_quality_predictor',
        model_type='regression',
        framework='sklearn',
        data_source={
            'type': 'database',
            'connection_string': 'postgresql://user:pass@localhost/ytempire',
            'query': 'SELECT * FROM video_features WHERE created_at >= NOW() - INTERVAL 30 DAY'
        },
        features=['title_length', 'description_length', 'tags_count', 'thumbnail_quality'],
        target='engagement_score',
        hyperparameters={
            'n_estimators': 100,
            'max_depth': 10
        },
        validation_split=0.2,
        test_split=0.1,
        auto_tune=True,
        deploy_on_success=True,
        min_performance_threshold=0.75,
        experiment_name='content_quality',
        tags={'team': 'data_science', 'project': 'youtube_automation'}
    )
    
    # Train model
    result = await pipeline.train_model(config)
    
    print(f"Training completed: {result.model_name}")
    print(f"Metrics: {result.metrics}")
    print(f"Model version: {result.model_version}")
    print(f"Deployed: {result.deployed}")
    
    # Schedule periodic retraining
    await pipeline.schedule_training(config, '0 0 * * 0')  # Weekly

if __name__ == "__main__":
    asyncio.run(main())