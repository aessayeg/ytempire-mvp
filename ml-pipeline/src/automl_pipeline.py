"""
AutoML Pipeline Implementation for YTEmpire MVP
Handles automated model training, hyperparameter tuning, and retraining
"""

import asyncio
import json
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum

# ML Libraries
from sklearn.model_selection import (
    GridSearchCV, 
    RandomizedSearchCV,
    cross_val_score,
    train_test_split
)
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import joblib

# Optional imports for advanced ML
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported model types for AutoML"""
    LINEAR_REGRESSION = "linear_regression"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    ADABOOST = "adaboost"


class OptimizationMetric(Enum):
    """Metrics for model optimization"""
    MSE = "mse"
    MAE = "mae"
    R2 = "r2"
    EXPLAINED_VARIANCE = "explained_variance"


@dataclass
class AutoMLConfig:
    """Configuration for AutoML pipeline"""
    task_type: str = "regression"  # or "classification"
    optimization_metric: OptimizationMetric = OptimizationMetric.R2
    test_size: float = 0.2
    cv_folds: int = 5
    n_trials: int = 100  # For Optuna
    timeout_seconds: int = 3600  # 1 hour
    enable_feature_engineering: bool = True
    enable_ensemble: bool = True
    auto_retrain_days: int = 7
    min_performance_threshold: float = 0.8
    max_models_to_evaluate: int = 10
    use_gpu: bool = False
    save_path: Path = Path("models/automl")
    

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_name: str
    model_type: ModelType
    train_score: float
    validation_score: float
    test_score: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    training_time: float = 0.0
    inference_time: float = 0.0
    best_params: Dict[str, Any] = field(default_factory=dict)
    feature_importance: Optional[Dict[str, float]] = None
    timestamp: datetime = field(default_factory=datetime.now)


class AutoMLPipeline:
    """
    Automated Machine Learning Pipeline
    Handles model selection, hyperparameter tuning, and automated retraining
    """
    
    def __init__(self, config: Optional[AutoMLConfig] = None):
        self.config = config or AutoMLConfig()
        self.best_model = None
        self.best_score = -np.inf
        self.model_registry: List[ModelPerformance] = []
        self.feature_names: Optional[List[str]] = None
        self.scaler = None
        self.models_path = self.config.save_path
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize model search space
        self._init_search_spaces()
        
    def _init_search_spaces(self):
        """Initialize hyperparameter search spaces for different models"""
        self.search_spaces = {
            ModelType.LINEAR_REGRESSION: {},
            
            ModelType.RIDGE: {
                'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr']
            },
            
            ModelType.LASSO: {
                'alpha': [0.001, 0.01, 0.1, 1, 10],
                'selection': ['cyclic', 'random']
            },
            
            ModelType.ELASTIC_NET: {
                'alpha': [0.001, 0.01, 0.1, 1],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            },
            
            ModelType.RANDOM_FOREST: {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            },
            
            ModelType.GRADIENT_BOOSTING: {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'subsample': [0.7, 0.8, 0.9, 1.0]
            },
            
            ModelType.ADABOOST: {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.5, 1.0]
            }
        }
        
        if XGBOOST_AVAILABLE:
            self.search_spaces[ModelType.XGBOOST] = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
            
        if LIGHTGBM_AVAILABLE:
            self.search_spaces[ModelType.LIGHTGBM] = {
                'n_estimators': [100, 200, 300],
                'max_depth': [-1, 5, 10, 20],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'num_leaves': [31, 50, 100, 200],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
    
    def _get_model(self, model_type: ModelType) -> Any:
        """Get model instance based on type"""
        model_map = {
            ModelType.LINEAR_REGRESSION: LinearRegression(),
            ModelType.RIDGE: Ridge(),
            ModelType.LASSO: Lasso(),
            ModelType.ELASTIC_NET: ElasticNet(),
            ModelType.RANDOM_FOREST: RandomForestRegressor(random_state=42, n_jobs=-1),
            ModelType.GRADIENT_BOOSTING: GradientBoostingRegressor(random_state=42),
            ModelType.ADABOOST: AdaBoostRegressor(random_state=42)
        }
        
        if XGBOOST_AVAILABLE and model_type == ModelType.XGBOOST:
            model_map[ModelType.XGBOOST] = xgb.XGBRegressor(
                random_state=42,
                n_jobs=-1,
                tree_method='gpu_hist' if self.config.use_gpu else 'auto'
            )
            
        if LIGHTGBM_AVAILABLE and model_type == ModelType.LIGHTGBM:
            model_map[ModelType.LIGHTGBM] = lgb.LGBMRegressor(
                random_state=42,
                n_jobs=-1,
                device='gpu' if self.config.use_gpu else 'cpu'
            )
            
        return model_map.get(model_type)
    
    def engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Automated feature engineering
        Creates polynomial features, interactions, and aggregations
        """
        if not self.config.enable_feature_engineering:
            return X
            
        X_engineered = X.copy()
        
        # Add polynomial features for numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:5]:  # Limit to prevent explosion
            X_engineered[f'{col}_squared'] = X[col] ** 2
            X_engineered[f'{col}_cubed'] = X[col] ** 3
            X_engineered[f'{col}_sqrt'] = np.sqrt(np.abs(X[col]))
            X_engineered[f'{col}_log'] = np.log1p(np.abs(X[col]))
        
        # Add interaction features
        for i, col1 in enumerate(numeric_cols[:3]):
            for col2 in numeric_cols[i+1:4]:
                X_engineered[f'{col1}_x_{col2}'] = X[col1] * X[col2]
                X_engineered[f'{col1}_div_{col2}'] = X[col1] / (X[col2] + 1e-10)
        
        # Add rolling statistics if time-based
        if 'timestamp' in X.columns or 'date' in X.columns:
            time_col = 'timestamp' if 'timestamp' in X.columns else 'date'
            X_sorted = X.sort_values(time_col)
            
            for col in numeric_cols[:3]:
                X_engineered[f'{col}_rolling_mean_7'] = X_sorted[col].rolling(7, min_periods=1).mean()
                X_engineered[f'{col}_rolling_std_7'] = X_sorted[col].rolling(7, min_periods=1).std()
                X_engineered[f'{col}_ewm_mean'] = X_sorted[col].ewm(span=7, min_periods=1).mean()
        
        return X_engineered
    
    def _evaluate_model(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_type: ModelType,
        search_space: Dict[str, List]
    ) -> ModelPerformance:
        """
        Evaluate a single model with hyperparameter tuning
        """
        import time
        start_time = time.time()
        
        # Perform hyperparameter tuning
        if search_space:
            if len(search_space) > 5 or sum(len(v) for v in search_space.values()) > 50:
                # Use RandomizedSearchCV for large search spaces
                searcher = RandomizedSearchCV(
                    model,
                    search_space,
                    n_iter=min(50, self.config.n_trials),
                    cv=self.config.cv_folds,
                    scoring=self._get_sklearn_scorer(),
                    n_jobs=-1,
                    random_state=42
                )
            else:
                # Use GridSearchCV for small search spaces
                searcher = GridSearchCV(
                    model,
                    search_space,
                    cv=self.config.cv_folds,
                    scoring=self._get_sklearn_scorer(),
                    n_jobs=-1
                )
            
            searcher.fit(X_train, y_train)
            best_model = searcher.best_estimator_
            best_params = searcher.best_params_
        else:
            best_model = model
            best_model.fit(X_train, y_train)
            best_params = {}
        
        training_time = time.time() - start_time
        
        # Evaluate on validation set
        val_pred = best_model.predict(X_val)
        train_pred = best_model.predict(X_train)
        
        # Calculate metrics
        val_score = self._calculate_metric(y_val, val_pred)
        train_score = self._calculate_metric(y_train, train_pred)
        
        # Calculate additional metrics
        mse = mean_squared_error(y_val, val_pred)
        mae = mean_absolute_error(y_val, val_pred)
        r2 = r2_score(y_val, val_pred)
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(best_model, 'feature_importances_'):
            if self.feature_names:
                feature_importance = dict(zip(
                    self.feature_names,
                    best_model.feature_importances_
                ))
        
        # Measure inference time
        inference_start = time.time()
        _ = best_model.predict(X_val[:100])  # Predict on subset
        inference_time = (time.time() - inference_start) / 100
        
        return ModelPerformance(
            model_name=f"{model_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model_type=model_type,
            train_score=train_score,
            validation_score=val_score,
            mse=mse,
            mae=mae,
            r2=r2,
            training_time=training_time,
            inference_time=inference_time,
            best_params=best_params,
            feature_importance=feature_importance
        )
    
    def _calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the optimization metric"""
        if self.config.optimization_metric == OptimizationMetric.MSE:
            return -mean_squared_error(y_true, y_pred)
        elif self.config.optimization_metric == OptimizationMetric.MAE:
            return -mean_absolute_error(y_true, y_pred)
        elif self.config.optimization_metric == OptimizationMetric.R2:
            return r2_score(y_true, y_pred)
        elif self.config.optimization_metric == OptimizationMetric.EXPLAINED_VARIANCE:
            return explained_variance_score(y_true, y_pred)
        else:
            return r2_score(y_true, y_pred)
    
    def _get_sklearn_scorer(self) -> str:
        """Get sklearn scorer string for the optimization metric"""
        scorer_map = {
            OptimizationMetric.MSE: 'neg_mean_squared_error',
            OptimizationMetric.MAE: 'neg_mean_absolute_error',
            OptimizationMetric.R2: 'r2',
            OptimizationMetric.EXPLAINED_VARIANCE: 'explained_variance'
        }
        return scorer_map.get(self.config.optimization_metric, 'r2')
    
    async def train_async(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Async wrapper for training"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.train, X, y, feature_names)
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Main training method that orchestrates the AutoML pipeline
        """
        logger.info("Starting AutoML pipeline training")
        
        # Store feature names
        self.feature_names = feature_names or list(X.columns)
        
        # Feature engineering
        if self.config.enable_feature_engineering:
            X = self.engineer_features(X)
            self.feature_names = list(X.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=42
        )
        
        # Further split train into train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Evaluate models
        model_performances = []
        models_to_try = list(ModelType)[:self.config.max_models_to_evaluate]
        
        for model_type in models_to_try:
            try:
                logger.info(f"Evaluating {model_type.value}")
                
                model = self._get_model(model_type)
                if model is None:
                    continue
                
                search_space = self.search_spaces.get(model_type, {})
                
                performance = self._evaluate_model(
                    model,
                    X_train_scaled,
                    y_train,
                    X_val_scaled,
                    y_val,
                    model_type,
                    search_space
                )
                
                # Test on holdout set
                if performance.validation_score > self.best_score:
                    test_model = self._get_model(model_type)
                    test_model.set_params(**performance.best_params)
                    test_model.fit(X_train_scaled, y_train)
                    test_pred = test_model.predict(X_test_scaled)
                    performance.test_score = self._calculate_metric(y_test, test_pred)
                    
                    self.best_model = test_model
                    self.best_score = performance.validation_score
                
                model_performances.append(performance)
                self.model_registry.append(performance)
                
            except Exception as e:
                logger.error(f"Error evaluating {model_type.value}: {str(e)}")
                continue
        
        # Ensemble best models if enabled
        if self.config.enable_ensemble and len(model_performances) > 2:
            ensemble_model = self._create_ensemble(
                model_performances[:3],
                X_train_scaled,
                y_train,
                X_val_scaled,
                y_val
            )
            if ensemble_model:
                self.best_model = ensemble_model
        
        # Save best model
        self.save_model()
        
        # Prepare results
        results = {
            'best_model_type': max(model_performances, key=lambda x: x.validation_score).model_type.value,
            'best_score': self.best_score,
            'all_performances': [
                {
                    'model': p.model_type.value,
                    'val_score': p.validation_score,
                    'train_score': p.train_score,
                    'test_score': p.test_score,
                    'training_time': p.training_time
                }
                for p in model_performances
            ],
            'feature_importance': max(model_performances, key=lambda x: x.validation_score).feature_importance,
            'training_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"AutoML training complete. Best score: {self.best_score:.4f}")
        
        return results
    
    def _create_ensemble(
        self,
        top_models: List[ModelPerformance],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Optional[Any]:
        """
        Create an ensemble of the top performing models
        """
        try:
            from sklearn.ensemble import VotingRegressor
            
            estimators = []
            for i, perf in enumerate(top_models):
                model = self._get_model(perf.model_type)
                model.set_params(**perf.best_params)
                estimators.append((f'model_{i}', model))
            
            ensemble = VotingRegressor(estimators=estimators)
            ensemble.fit(X_train, y_train)
            
            # Check if ensemble performs better
            ensemble_pred = ensemble.predict(X_val)
            ensemble_score = self._calculate_metric(y_val, ensemble_pred)
            
            if ensemble_score > self.best_score:
                logger.info(f"Ensemble model performs better: {ensemble_score:.4f}")
                return ensemble
                
        except Exception as e:
            logger.error(f"Error creating ensemble: {str(e)}")
        
        return None
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the best model"""
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        
        # Apply same feature engineering
        if self.config.enable_feature_engineering:
            X = self.engineer_features(X)
        
        # Scale features
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.best_model.predict(X_scaled)
    
    def save_model(self, path: Optional[Path] = None) -> None:
        """Save the best model and associated artifacts"""
        save_path = path or self.models_path / f"automl_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        artifacts = {
            'model': self.best_model,
            'scaler': self.scaler,
            'config': self.config,
            'best_score': self.best_score,
            'feature_names': self.feature_names,
            'model_registry': self.model_registry[-5:]  # Keep last 5 models
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(artifacts, f)
        
        # Also save as joblib for sklearn models
        if self.best_model:
            joblib.dump(self.best_model, save_path.with_suffix('.joblib'))
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, path: Path) -> None:
        """Load a saved model and artifacts"""
        with open(path, 'rb') as f:
            artifacts = pickle.load(f)
        
        self.best_model = artifacts['model']
        self.scaler = artifacts['scaler']
        self.config = artifacts['config']
        self.best_score = artifacts['best_score']
        self.feature_names = artifacts['feature_names']
        self.model_registry = artifacts.get('model_registry', [])
        
        logger.info(f"Model loaded from {path}")
    
    def should_retrain(self) -> bool:
        """
        Check if model should be retrained based on performance or time
        """
        if not self.model_registry:
            return True
        
        latest_model = self.model_registry[-1]
        
        # Check time since last training
        days_since_training = (datetime.now() - latest_model.timestamp).days
        if days_since_training >= self.config.auto_retrain_days:
            logger.info(f"Model is {days_since_training} days old. Retraining needed.")
            return True
        
        # Check performance degradation
        if latest_model.validation_score < self.config.min_performance_threshold:
            logger.info(f"Model performance {latest_model.validation_score:.4f} below threshold. Retraining needed.")
            return True
        
        return False
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of the current model and training history"""
        if not self.model_registry:
            return {"status": "No models trained yet"}
        
        latest = self.model_registry[-1]
        
        return {
            "current_model": {
                "type": latest.model_type.value,
                "validation_score": latest.validation_score,
                "test_score": latest.test_score,
                "training_time": latest.training_time,
                "inference_time_ms": latest.inference_time * 1000,
                "trained_at": latest.timestamp.isoformat()
            },
            "best_parameters": latest.best_params,
            "top_features": dict(sorted(
                latest.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]) if latest.feature_importance else None,
            "model_history": [
                {
                    "model": m.model_type.value,
                    "score": m.validation_score,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in self.model_registry[-5:]
            ],
            "retrain_needed": self.should_retrain()
        }


class AutoMLOptunaTuner:
    """
    Advanced hyperparameter tuning using Optuna
    Provides more sophisticated optimization than GridSearch
    """
    
    def __init__(self, model_type: ModelType, config: AutoMLConfig):
        self.model_type = model_type
        self.config = config
        
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for advanced tuning. Install with: pip install optuna")
    
    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Optimize hyperparameters using Optuna
        """
        import optuna
        
        def objective(trial):
            # Define hyperparameter search space based on model type
            params = self._get_optuna_params(trial)
            
            # Create and train model
            if self.model_type == ModelType.XGBOOST and XGBOOST_AVAILABLE:
                model = xgb.XGBRegressor(**params, random_state=42)
            elif self.model_type == ModelType.LIGHTGBM and LIGHTGBM_AVAILABLE:
                model = lgb.LGBMRegressor(**params, random_state=42)
            elif self.model_type == ModelType.RANDOM_FOREST:
                model = RandomForestRegressor(**params, random_state=42)
            elif self.model_type == ModelType.GRADIENT_BOOSTING:
                model = GradientBoostingRegressor(**params, random_state=42)
            else:
                return -np.inf
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_val)
            
            if self.config.optimization_metric == OptimizationMetric.MSE:
                return mean_squared_error(y_val, y_pred)
            elif self.config.optimization_metric == OptimizationMetric.MAE:
                return mean_absolute_error(y_val, y_pred)
            elif self.config.optimization_metric == OptimizationMetric.R2:
                return -r2_score(y_val, y_pred)  # Minimize negative R2
            else:
                return -r2_score(y_val, y_pred)
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_seconds,
            n_jobs=1  # Set to 1 to avoid issues with parallel execution
        )
        
        # Get best model
        best_params = study.best_params
        
        if self.model_type == ModelType.XGBOOST and XGBOOST_AVAILABLE:
            best_model = xgb.XGBRegressor(**best_params, random_state=42)
        elif self.model_type == ModelType.LIGHTGBM and LIGHTGBM_AVAILABLE:
            best_model = lgb.LGBMRegressor(**best_params, random_state=42)
        elif self.model_type == ModelType.RANDOM_FOREST:
            best_model = RandomForestRegressor(**best_params, random_state=42)
        elif self.model_type == ModelType.GRADIENT_BOOSTING:
            best_model = GradientBoostingRegressor(**best_params, random_state=42)
        else:
            raise ValueError(f"Unsupported model type for Optuna: {self.model_type}")
        
        best_model.fit(X_train, y_train)
        
        return best_model, best_params
    
    def _get_optuna_params(self, trial) -> Dict[str, Any]:
        """Define Optuna search space for different models"""
        if self.model_type == ModelType.XGBOOST:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
            }
        
        elif self.model_type == ModelType.LIGHTGBM:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', -1, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
            }
        
        elif self.model_type == ModelType.RANDOM_FOREST:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
            }
        
        elif self.model_type == ModelType.GRADIENT_BOOSTING:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0)
            }
        
        else:
            return {}


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create target with some relationship to features
    y = pd.Series(
        X['feature_0'] * 2 + X['feature_1'] * 1.5 - X['feature_2'] * 0.5 + 
        np.random.randn(n_samples) * 0.1
    )
    
    # Initialize AutoML pipeline
    config = AutoMLConfig(
        optimization_metric=OptimizationMetric.R2,
        max_models_to_evaluate=5,
        enable_feature_engineering=True,
        enable_ensemble=True
    )
    
    automl = AutoMLPipeline(config)
    
    # Train models
    results = automl.train(X, y)
    
    # Print results
    print(f"Best model type: {results['best_model_type']}")
    print(f"Best score: {results['best_score']:.4f}")
    print("\nModel performances:")
    for perf in results['all_performances']:
        print(f"  {perf['model']}: {perf['val_score']:.4f} (train: {perf['train_score']:.4f})")
    
    # Get model summary
    summary = automl.get_model_summary()
    print(f"\nModel summary: {json.dumps(summary, indent=2, default=str)}")
    
    # Make predictions
    X_test = pd.DataFrame(
        np.random.randn(10, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    predictions = automl.predict(X_test)
    print(f"\nPredictions: {predictions[:5]}")