"""
Advanced AutoML Platform v2.0 for YTEmpire
Expanded with neural architecture search, automated feature engineering, and multi-objective optimization
"""

import asyncio
import json
import logging
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import yaml

# Core ML Libraries
from sklearn.model_selection import (
    GridSearchCV, 
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
    StratifiedKFold,
    TimeSeriesSplit
)
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    AdaBoostRegressor,
    AdaBoostClassifier,
    VotingRegressor,
    VotingClassifier,
    StackingRegressor,
    StackingClassifier
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    Lasso,
    ElasticNet
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.preprocessing import (
    StandardScaler, 
    MinMaxScaler, 
    RobustScaler,
    PolynomialFeatures,
    PowerTransformer
)
from sklearn.feature_selection import (
    SelectKBest,
    RFE,
    SelectFromModel,
    VarianceThreshold
)
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import Pipeline, FeatureUnion
import joblib

# Advanced ML Libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available. Install with: pip install lightgbm")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    warnings.warn("CatBoost not available. Install with: pip install catboost")

try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not available. Install with: pip install optuna")

try:
    from autosklearn.regression import AutoSklearnRegressor
    from autosklearn.classification import AutoSklearnClassifier
    AUTOSKLEARN_AVAILABLE = True
except ImportError:
    AUTOSKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    warnings.warn("PyTorch not available for neural architecture search")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Machine learning task types"""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    TIME_SERIES = "time_series"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    RECOMMENDATION = "recommendation"


class ModelFamily(Enum):
    """Model families for AutoML"""
    LINEAR = "linear"
    TREE_BASED = "tree_based"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"
    GRADIENT_BOOSTING = "gradient_boosting"
    SVM = "svm"
    NEIGHBORS = "neighbors"


class OptimizationStrategy(Enum):
    """Optimization strategies"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    EVOLUTIONARY = "evolutionary"
    HYPERBAND = "hyperband"
    NEURAL_ARCHITECTURE_SEARCH = "nas"


@dataclass
class AutoMLConfig:
    """Enhanced AutoML configuration"""
    task_type: TaskType = TaskType.REGRESSION
    optimization_metrics: List[str] = field(default_factory=lambda: ["r2", "mse"])
    test_size: float = 0.2
    validation_size: float = 0.2
    cv_folds: int = 5
    n_trials: int = 100
    timeout_seconds: int = 3600
    
    # Feature engineering
    enable_feature_engineering: bool = True
    feature_selection_method: str = "auto"
    max_features: Optional[int] = None
    polynomial_features: bool = False
    polynomial_degree: int = 2
    
    # Model selection
    model_families: List[ModelFamily] = field(default_factory=lambda: [
        ModelFamily.LINEAR,
        ModelFamily.TREE_BASED,
        ModelFamily.GRADIENT_BOOSTING
    ])
    enable_ensemble: bool = True
    ensemble_method: str = "voting"  # voting, stacking, blending
    
    # Optimization
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN
    multi_objective: bool = False
    
    # Neural architecture search
    enable_nas: bool = False
    nas_max_layers: int = 5
    nas_max_neurons: int = 512
    
    # Resources
    n_jobs: int = -1
    gpu_enabled: bool = False
    memory_limit: Optional[int] = None  # MB
    
    # Persistence
    save_models: bool = True
    model_dir: str = "models/automl"
    experiment_tracking: bool = True
    
    # Auto-retraining
    auto_retrain: bool = True
    retrain_schedule: str = "weekly"  # daily, weekly, monthly
    performance_threshold: float = 0.9  # Trigger retraining if performance drops below


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    train_score: float
    val_score: float
    test_score: float
    metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]] = None
    training_time: float = 0.0
    inference_time: float = 0.0
    model_size_mb: float = 0.0


class AdvancedFeatureEngineer:
    """Advanced automated feature engineering"""
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.transformers = []
        self.feature_names = []
        
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Apply automated feature engineering"""
        logger.info("Starting advanced feature engineering")
        
        X_transformed = X.copy()
        
        # 1. Handle missing values intelligently
        X_transformed = self._handle_missing_values(X_transformed)
        
        # 2. Encode categorical variables
        X_transformed = self._encode_categoricals(X_transformed)
        
        # 3. Create interaction features
        if self.config.polynomial_features:
            X_transformed = self._create_polynomial_features(X_transformed)
        
        # 4. Generate statistical features
        X_transformed = self._generate_statistical_features(X_transformed)
        
        # 5. Create time-based features if applicable
        X_transformed = self._create_temporal_features(X_transformed)
        
        # 6. Apply transformations
        X_transformed = self._apply_transformations(X_transformed)
        
        # 7. Feature selection
        if self.config.enable_feature_engineering and y is not None:
            X_transformed = self._select_features(X_transformed, y)
        
        self.feature_names = X_transformed.columns.tolist()
        logger.info(f"Feature engineering complete. Generated {len(self.feature_names)} features")
        
        return X_transformed
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Intelligent missing value handling"""
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                if X[col].dtype in ['float64', 'int64']:
                    # Use median for numerical
                    X[col].fillna(X[col].median(), inplace=True)
                else:
                    # Use mode for categorical
                    X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'missing', inplace=True)
        return X
    
    def _encode_categoricals(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            # Use target encoding for high cardinality
            if X[col].nunique() > 10:
                # Simple frequency encoding as placeholder
                freq_encoding = X[col].value_counts(normalize=True).to_dict()
                X[f"{col}_freq"] = X[col].map(freq_encoding)
            else:
                # One-hot encoding for low cardinality
                X = pd.get_dummies(X, columns=[col], prefix=col)
        
        return X
    
    def _create_polynomial_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial and interaction features"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns[:10]  # Limit to prevent explosion
        
        if len(numeric_cols) > 1:
            poly = PolynomialFeatures(degree=self.config.polynomial_degree, include_bias=False)
            poly_features = poly.fit_transform(X[numeric_cols])
            poly_df = pd.DataFrame(
                poly_features,
                columns=[f"poly_{i}" for i in range(poly_features.shape[1])],
                index=X.index
            )
            X = pd.concat([X, poly_df.iloc[:, len(numeric_cols):]], axis=1)
        
        return X
    
    def _generate_statistical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate statistical aggregation features"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 2:
            # Add statistical features
            X['mean_numeric'] = X[numeric_cols].mean(axis=1)
            X['std_numeric'] = X[numeric_cols].std(axis=1)
            X['min_numeric'] = X[numeric_cols].min(axis=1)
            X['max_numeric'] = X[numeric_cols].max(axis=1)
            X['skew_numeric'] = X[numeric_cols].skew(axis=1)
        
        return X
    
    def _create_temporal_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features if datetime columns exist"""
        datetime_cols = X.select_dtypes(include=['datetime64']).columns
        
        for col in datetime_cols:
            X[f'{col}_year'] = X[col].dt.year
            X[f'{col}_month'] = X[col].dt.month
            X[f'{col}_day'] = X[col].dt.day
            X[f'{col}_dayofweek'] = X[col].dt.dayofweek
            X[f'{col}_hour'] = X[col].dt.hour
            
            # Cyclical encoding
            X[f'{col}_month_sin'] = np.sin(2 * np.pi * X[col].dt.month / 12)
            X[f'{col}_month_cos'] = np.cos(2 * np.pi * X[col].dt.month / 12)
            
            # Drop original datetime column
            X = X.drop(columns=[col])
        
        return X
    
    def _apply_transformations(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply mathematical transformations"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if X[col].min() > 0:  # Only for positive values
                X[f'{col}_log'] = np.log1p(X[col])
                X[f'{col}_sqrt'] = np.sqrt(X[col])
        
        return X
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Automated feature selection"""
        if self.config.feature_selection_method == "variance":
            selector = VarianceThreshold(threshold=0.01)
        elif self.config.feature_selection_method == "kbest":
            k = min(self.config.max_features or X.shape[1] // 2, X.shape[1])
            selector = SelectKBest(k=k)
        elif self.config.feature_selection_method == "rfe":
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            selector = RFE(estimator, n_features_to_select=self.config.max_features or X.shape[1] // 2)
        else:  # auto
            selector = SelectFromModel(
                RandomForestRegressor(n_estimators=50, random_state=42),
                threshold='median'
            )
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)


class NeuralArchitectureSearch:
    """Neural Architecture Search for automatic neural network design"""
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.best_architecture = None
        self.search_space = self._define_search_space()
    
    def _define_search_space(self) -> Dict[str, Any]:
        """Define the neural architecture search space"""
        return {
            'n_layers': list(range(1, self.config.nas_max_layers + 1)),
            'neurons_per_layer': [32, 64, 128, 256, 512],
            'activation': ['relu', 'tanh', 'sigmoid', 'elu'],
            'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'batch_norm': [True, False],
            'optimizer': ['adam', 'sgd', 'rmsprop'],
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [16, 32, 64, 128]
        }
    
    def search(self, X_train: np.ndarray, y_train: np.ndarray,
               X_val: np.ndarray, y_val: np.ndarray,
               n_trials: int = 50) -> Dict[str, Any]:
        """Perform neural architecture search"""
        if not PYTORCH_AVAILABLE:
            logger.warning("PyTorch not available. Returning default architecture.")
            return self._get_default_architecture()
        
        logger.info("Starting Neural Architecture Search")
        
        if OPTUNA_AVAILABLE:
            study = optuna.create_study(
                direction='maximize' if self.config.task_type == TaskType.CLASSIFICATION else 'minimize',
                sampler=TPESampler()
            )
            
            study.optimize(
                lambda trial: self._objective(trial, X_train, y_train, X_val, y_val),
                n_trials=n_trials,
                timeout=self.config.timeout_seconds // 2
            )
            
            self.best_architecture = study.best_params
            logger.info(f"Best architecture found: {self.best_architecture}")
        else:
            # Fallback to random search
            self.best_architecture = self._random_search(X_train, y_train, X_val, y_val, n_trials)
        
        return self.best_architecture
    
    def _objective(self, trial, X_train, y_train, X_val, y_val) -> float:
        """Optuna objective function for NAS"""
        # Sample architecture
        n_layers = trial.suggest_int('n_layers', 1, self.config.nas_max_layers)
        layers = []
        
        for i in range(n_layers):
            n_neurons = trial.suggest_int(f'n_neurons_l{i}', 32, self.config.nas_max_neurons)
            layers.append(n_neurons)
        
        activation = trial.suggest_categorical('activation', self.search_space['activation'])
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
        
        # Build and train model
        model = self._build_pytorch_model(
            input_dim=X_train.shape[1],
            layers=layers,
            activation=activation,
            dropout_rate=dropout_rate
        )
        
        # Train model
        score = self._train_pytorch_model(
            model, X_train, y_train, X_val, y_val,
            learning_rate=learning_rate,
            epochs=50
        )
        
        return score
    
    def _build_pytorch_model(self, input_dim: int, layers: List[int],
                            activation: str, dropout_rate: float) -> nn.Module:
        """Build PyTorch model from architecture"""
        class DynamicNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList()
                
                # Input layer
                prev_dim = input_dim
                
                for i, n_neurons in enumerate(layers):
                    self.layers.append(nn.Linear(prev_dim, n_neurons))
                    self.layers.append(self._get_activation(activation))
                    if dropout_rate > 0:
                        self.layers.append(nn.Dropout(dropout_rate))
                    prev_dim = n_neurons
                
                # Output layer
                output_dim = 1 if self.config.task_type == TaskType.REGRESSION else 2
                self.layers.append(nn.Linear(prev_dim, output_dim))
            
            def _get_activation(self, name: str) -> nn.Module:
                activations = {
                    'relu': nn.ReLU(),
                    'tanh': nn.Tanh(),
                    'sigmoid': nn.Sigmoid(),
                    'elu': nn.ELU()
                }
                return activations.get(name, nn.ReLU())
            
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
        
        return DynamicNN()
    
    def _train_pytorch_model(self, model, X_train, y_train, X_val, y_val,
                           learning_rate: float, epochs: int) -> float:
        """Train PyTorch model and return validation score"""
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train.reshape(-1, 1))
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val.reshape(-1, 1))
        
        # Setup
        criterion = nn.MSELoss() if self.config.task_type == TaskType.REGRESSION else nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t)
        
        return -val_loss.item()  # Return negative for minimization
    
    def _random_search(self, X_train, y_train, X_val, y_val, n_trials: int) -> Dict[str, Any]:
        """Fallback random search for NAS"""
        best_score = float('-inf')
        best_arch = None
        
        for _ in range(n_trials):
            # Random architecture
            arch = {
                'n_layers': np.random.choice(self.search_space['n_layers']),
                'neurons_per_layer': np.random.choice(self.search_space['neurons_per_layer']),
                'activation': np.random.choice(self.search_space['activation']),
                'dropout_rate': np.random.choice(self.search_space['dropout_rate'])
            }
            
            # Evaluate (simplified)
            score = np.random.random()  # Placeholder
            
            if score > best_score:
                best_score = score
                best_arch = arch
        
        return best_arch
    
    def _get_default_architecture(self) -> Dict[str, Any]:
        """Return default neural architecture"""
        return {
            'n_layers': 3,
            'neurons_per_layer': [128, 64, 32],
            'activation': 'relu',
            'dropout_rate': 0.2,
            'batch_norm': True,
            'optimizer': 'adam',
            'learning_rate': 0.001
        }


class AdvancedAutoMLPlatform:
    """
    Advanced AutoML Platform with expanded capabilities
    """
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.feature_engineer = AdvancedFeatureEngineer(config)
        self.nas = NeuralArchitectureSearch(config) if config.enable_nas else None
        self.models = {}
        self.best_model = None
        self.best_metrics = None
        self.experiment_history = []
        
        # Create model directory
        Path(self.config.model_dir).mkdir(parents=True, exist_ok=True)
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_test: Optional[pd.DataFrame] = None,
            y_test: Optional[pd.Series] = None) -> 'AdvancedAutoMLPlatform':
        """
        Fit AutoML pipeline with advanced features
        """
        logger.info("Starting Advanced AutoML Platform")
        start_time = datetime.now()
        
        # 1. Feature Engineering
        X_engineered = self.feature_engineer.fit_transform(X, y)
        
        # 2. Split data
        if X_test is None:
            X_train, X_test, y_train, y_test = train_test_split(
                X_engineered, y,
                test_size=self.config.test_size,
                random_state=42
            )
        else:
            X_train, y_train = X_engineered, y
            X_test = self.feature_engineer.fit_transform(X_test)
        
        # Further split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=self.config.validation_size,
            random_state=42
        )
        
        # 3. Model Selection and Training
        if self.config.optimization_strategy == OptimizationStrategy.NEURAL_ARCHITECTURE_SEARCH:
            self._train_with_nas(X_train, y_train, X_val, y_val)
        else:
            self._train_models(X_train, y_train, X_val, y_val)
        
        # 4. Ensemble Learning
        if self.config.enable_ensemble:
            self._create_ensemble(X_train, y_train, X_val, y_val)
        
        # 5. Evaluate on test set
        self._evaluate_models(X_test, y_test)
        
        # 6. Select best model
        self._select_best_model()
        
        # 7. Save results
        self._save_results()
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"AutoML completed in {elapsed_time:.2f} seconds")
        logger.info(f"Best model: {self.best_model.__class__.__name__}")
        logger.info(f"Best metrics: {self.best_metrics}")
        
        return self
    
    def _train_models(self, X_train, y_train, X_val, y_val):
        """Train multiple model families"""
        logger.info("Training models across different families")
        
        # Get candidate models
        candidate_models = self._get_candidate_models()
        
        # Train each model
        for name, model in candidate_models.items():
            logger.info(f"Training {name}")
            
            try:
                # Hyperparameter optimization
                if self.config.optimization_strategy == OptimizationStrategy.BAYESIAN:
                    optimized_model = self._bayesian_optimization(
                        model, X_train, y_train, X_val, y_val
                    )
                else:
                    optimized_model = self._grid_search(
                        model, X_train, y_train
                    )
                
                # Store model
                self.models[name] = optimized_model
                
                # Track experiment
                self.experiment_history.append({
                    'model': name,
                    'timestamp': datetime.now().isoformat(),
                    'params': optimized_model.get_params() if hasattr(optimized_model, 'get_params') else {}
                })
                
            except Exception as e:
                logger.warning(f"Failed to train {name}: {e}")
    
    def _get_candidate_models(self) -> Dict[str, Any]:
        """Get candidate models based on configuration"""
        models = {}
        
        if self.config.task_type == TaskType.REGRESSION:
            if ModelFamily.LINEAR in self.config.model_families:
                models['linear_regression'] = LinearRegression()
                models['ridge'] = Ridge()
                models['lasso'] = Lasso()
                models['elastic_net'] = ElasticNet()
            
            if ModelFamily.TREE_BASED in self.config.model_families:
                models['random_forest'] = RandomForestRegressor(n_jobs=self.config.n_jobs)
            
            if ModelFamily.GRADIENT_BOOSTING in self.config.model_families:
                models['gradient_boosting'] = GradientBoostingRegressor()
                
                if XGBOOST_AVAILABLE:
                    models['xgboost'] = xgb.XGBRegressor(n_jobs=self.config.n_jobs)
                
                if LIGHTGBM_AVAILABLE:
                    models['lightgbm'] = lgb.LGBMRegressor(n_jobs=self.config.n_jobs)
                
                if CATBOOST_AVAILABLE:
                    models['catboost'] = cb.CatBoostRegressor(verbose=False)
            
            if ModelFamily.SVM in self.config.model_families:
                models['svr'] = SVR()
            
            if ModelFamily.NEIGHBORS in self.config.model_families:
                models['knn'] = KNeighborsRegressor()
            
            if ModelFamily.NEURAL_NETWORK in self.config.model_families:
                models['mlp'] = MLPRegressor(max_iter=1000)
        
        elif self.config.task_type == TaskType.CLASSIFICATION:
            if ModelFamily.LINEAR in self.config.model_families:
                models['logistic_regression'] = LogisticRegression(max_iter=1000)
            
            if ModelFamily.TREE_BASED in self.config.model_families:
                models['random_forest'] = RandomForestClassifier(n_jobs=self.config.n_jobs)
            
            if ModelFamily.GRADIENT_BOOSTING in self.config.model_families:
                models['gradient_boosting'] = GradientBoostingClassifier()
                
                if XGBOOST_AVAILABLE:
                    models['xgboost'] = xgb.XGBClassifier(n_jobs=self.config.n_jobs)
                
                if LIGHTGBM_AVAILABLE:
                    models['lightgbm'] = lgb.LGBMClassifier(n_jobs=self.config.n_jobs)
            
            if ModelFamily.SVM in self.config.model_families:
                models['svc'] = SVC(probability=True)
            
            if ModelFamily.NEURAL_NETWORK in self.config.model_families:
                models['mlp'] = MLPClassifier(max_iter=1000)
        
        return models
    
    def _bayesian_optimization(self, model, X_train, y_train, X_val, y_val):
        """Bayesian optimization using Optuna"""
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available. Using default parameters.")
            return model.fit(X_train, y_train)
        
        def objective(trial):
            # Get hyperparameters based on model type
            params = self._get_trial_params(trial, model)
            
            # Create model with suggested params
            model_copy = model.__class__(**params)
            
            # Train and validate
            model_copy.fit(X_train, y_train)
            
            # Evaluate
            if self.config.task_type == TaskType.REGRESSION:
                y_pred = model_copy.predict(X_val)
                score = -mean_squared_error(y_val, y_pred)  # Negative for maximization
            else:
                y_pred = model_copy.predict(X_val)
                score = accuracy_score(y_val, y_pred)
            
            return score
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler()
        )
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_seconds
        )
        
        # Train final model with best params
        best_params = study.best_params
        final_model = model.__class__(**best_params)
        final_model.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))
        
        return final_model
    
    def _get_trial_params(self, trial, model) -> Dict[str, Any]:
        """Get Optuna trial parameters for different model types"""
        model_name = model.__class__.__name__
        
        if 'RandomForest' in model_name:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
            }
        
        elif 'XGB' in model_name:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0)
            }
        
        elif 'LGBM' in model_name:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0)
            }
        
        elif 'Ridge' in model_name or 'Lasso' in model_name:
            return {
                'alpha': trial.suggest_loguniform('alpha', 1e-5, 100)
            }
        
        elif 'ElasticNet' in model_name:
            return {
                'alpha': trial.suggest_loguniform('alpha', 1e-5, 100),
                'l1_ratio': trial.suggest_uniform('l1_ratio', 0.0, 1.0)
            }
        
        elif 'SV' in model_name:
            return {
                'C': trial.suggest_loguniform('C', 1e-3, 100),
                'gamma': trial.suggest_loguniform('gamma', 1e-5, 1),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
            }
        
        elif 'MLP' in model_name:
            n_layers = trial.suggest_int('n_layers', 1, 3)
            layers = tuple(
                trial.suggest_int(f'n_neurons_l{i}', 10, 200)
                for i in range(n_layers)
            )
            return {
                'hidden_layer_sizes': layers,
                'activation': trial.suggest_categorical('activation', ['tanh', 'relu']),
                'solver': trial.suggest_categorical('solver', ['adam', 'lbfgs']),
                'alpha': trial.suggest_loguniform('alpha', 1e-5, 1),
                'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive'])
            }
        
        else:
            return {}
    
    def _grid_search(self, model, X_train, y_train):
        """Fallback grid search optimization"""
        # Define parameter grids for different models
        param_grids = {
            'RandomForest': {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5]
            },
            'GradientBoosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            }
        }
        
        model_name = model.__class__.__name__
        for key in param_grids:
            if key in model_name:
                grid_search = GridSearchCV(
                    model,
                    param_grids[key],
                    cv=self.config.cv_folds,
                    n_jobs=self.config.n_jobs,
                    scoring='neg_mean_squared_error' if self.config.task_type == TaskType.REGRESSION else 'accuracy'
                )
                grid_search.fit(X_train, y_train)
                return grid_search.best_estimator_
        
        # Default: train without optimization
        model.fit(X_train, y_train)
        return model
    
    def _train_with_nas(self, X_train, y_train, X_val, y_val):
        """Train using Neural Architecture Search"""
        if self.nas:
            best_arch = self.nas.search(
                X_train.values if hasattr(X_train, 'values') else X_train,
                y_train.values if hasattr(y_train, 'values') else y_train,
                X_val.values if hasattr(X_val, 'values') else X_val,
                y_val.values if hasattr(y_val, 'values') else y_val,
                n_trials=self.config.n_trials // 2
            )
            
            # Build and train final model with best architecture
            # This is simplified - in production, you'd rebuild the model
            mlp = MLPRegressor(
                hidden_layer_sizes=best_arch.get('neurons_per_layer', [128, 64]),
                activation=best_arch.get('activation', 'relu'),
                learning_rate_init=best_arch.get('learning_rate', 0.001),
                max_iter=1000
            )
            
            mlp.fit(
                np.vstack([X_train, X_val]),
                np.hstack([y_train, y_val])
            )
            
            self.models['neural_architecture_search'] = mlp
    
    def _create_ensemble(self, X_train, y_train, X_val, y_val):
        """Create ensemble models"""
        if len(self.models) < 2:
            logger.warning("Not enough models for ensemble. Skipping.")
            return
        
        logger.info(f"Creating ensemble with {self.config.ensemble_method} method")
        
        # Select top models for ensemble
        top_models = self._select_top_models(X_val, y_val, top_k=5)
        
        if self.config.task_type == TaskType.REGRESSION:
            if self.config.ensemble_method == "voting":
                ensemble = VotingRegressor(
                    estimators=[(name, model) for name, model in top_models.items()]
                )
            elif self.config.ensemble_method == "stacking":
                ensemble = StackingRegressor(
                    estimators=[(name, model) for name, model in top_models.items()],
                    final_estimator=Ridge()
                )
            else:  # blending
                # Simplified blending
                ensemble = VotingRegressor(
                    estimators=[(name, model) for name, model in top_models.items()],
                    weights=self._calculate_blend_weights(top_models, X_val, y_val)
                )
        else:
            if self.config.ensemble_method == "voting":
                ensemble = VotingClassifier(
                    estimators=[(name, model) for name, model in top_models.items()],
                    voting='soft'
                )
            elif self.config.ensemble_method == "stacking":
                ensemble = StackingClassifier(
                    estimators=[(name, model) for name, model in top_models.items()],
                    final_estimator=LogisticRegression()
                )
            else:
                ensemble = VotingClassifier(
                    estimators=[(name, model) for name, model in top_models.items()],
                    voting='soft',
                    weights=self._calculate_blend_weights(top_models, X_val, y_val)
                )
        
        # Train ensemble
        ensemble.fit(
            np.vstack([X_train, X_val]),
            np.hstack([y_train, y_val])
        )
        
        self.models[f'ensemble_{self.config.ensemble_method}'] = ensemble
    
    def _select_top_models(self, X_val, y_val, top_k: int = 5) -> Dict:
        """Select top performing models"""
        model_scores = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_val)
            
            if self.config.task_type == TaskType.REGRESSION:
                score = r2_score(y_val, y_pred)
            else:
                score = accuracy_score(y_val, y_pred)
            
            model_scores[name] = score
        
        # Sort and select top models
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        top_model_names = [name for name, _ in sorted_models[:top_k]]
        
        return {name: self.models[name] for name in top_model_names}
    
    def _calculate_blend_weights(self, models: Dict, X_val, y_val) -> List[float]:
        """Calculate blending weights based on validation performance"""
        weights = []
        
        for name, model in models.items():
            y_pred = model.predict(X_val)
            
            if self.config.task_type == TaskType.REGRESSION:
                score = r2_score(y_val, y_pred)
            else:
                score = accuracy_score(y_val, y_pred)
            
            weights.append(max(0, score))  # Ensure non-negative
        
        # Normalize weights
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
        
        return weights
    
    def _evaluate_models(self, X_test, y_test):
        """Evaluate all models on test set"""
        logger.info("Evaluating models on test set")
        
        for name, model in self.models.items():
            metrics = self._calculate_metrics(model, X_test, y_test)
            
            # Store metrics
            if not hasattr(model, 'metrics_'):
                model.metrics_ = metrics
    
    def _calculate_metrics(self, model, X_test, y_test) -> Dict[str, float]:
        """Calculate comprehensive metrics"""
        y_pred = model.predict(X_test)
        
        if self.config.task_type == TaskType.REGRESSION:
            return {
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'explained_variance': explained_variance_score(y_test, y_pred)
            }
        else:
            return {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
    
    def _select_best_model(self):
        """Select the best model based on optimization metrics"""
        best_score = float('-inf') if self.config.task_type == TaskType.CLASSIFICATION else float('inf')
        
        primary_metric = self.config.optimization_metrics[0]
        
        for name, model in self.models.items():
            if hasattr(model, 'metrics_'):
                score = model.metrics_.get(primary_metric, 0)
                
                # For regression, lower is better for mse/mae
                if primary_metric in ['mse', 'mae']:
                    if score < best_score:
                        best_score = score
                        self.best_model = model
                        self.best_metrics = model.metrics_
                else:
                    if score > best_score:
                        best_score = score
                        self.best_model = model
                        self.best_metrics = model.metrics_
    
    def _save_results(self):
        """Save models and results"""
        if self.config.save_models and self.best_model:
            # Save best model
            model_path = Path(self.config.model_dir) / f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            joblib.dump(self.best_model, model_path)
            logger.info(f"Best model saved to {model_path}")
            
            # Save experiment history
            history_path = Path(self.config.model_dir) / "experiment_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.experiment_history, f, indent=2, default=str)
            
            # Save feature names
            features_path = Path(self.config.model_dir) / "feature_names.json"
            with open(features_path, 'w') as f:
                json.dump(self.feature_engineer.feature_names, f)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with the best model"""
        if self.best_model is None:
            raise ValueError("No model has been trained yet. Call fit() first.")
        
        # Apply same feature engineering
        X_transformed = self.feature_engineer.fit_transform(X)
        
        return self.best_model.predict(X_transformed)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the best model"""
        if self.best_model is None:
            return {}
        
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            return dict(zip(self.feature_engineer.feature_names, importances))
        
        return {}
    
    def retrain(self, X_new: pd.DataFrame, y_new: pd.Series):
        """Retrain the model with new data"""
        logger.info("Retraining model with new data")
        
        # Combine with existing data if available
        # In production, you'd load historical data
        
        # Retrain
        self.fit(X_new, y_new)
        
        logger.info("Retraining complete")


def create_automl_platform(config_dict: Optional[Dict] = None) -> AdvancedAutoMLPlatform:
    """Factory function to create AutoML platform"""
    if config_dict:
        config = AutoMLConfig(**config_dict)
    else:
        config = AutoMLConfig()
    
    return AdvancedAutoMLPlatform(config)


# Example usage
if __name__ == "__main__":
    # Create sample data
    from sklearn.datasets import make_regression
    
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
    y = pd.Series(y, name="target")
    
    # Configure AutoML
    config = AutoMLConfig(
        task_type=TaskType.REGRESSION,
        optimization_metrics=["r2", "mse"],
        n_trials=50,
        timeout_seconds=300,
        enable_feature_engineering=True,
        enable_ensemble=True,
        enable_nas=False,  # Set to True if PyTorch is available
        model_families=[
            ModelFamily.LINEAR,
            ModelFamily.TREE_BASED,
            ModelFamily.GRADIENT_BOOSTING
        ]
    )
    
    # Create and run AutoML
    automl = AdvancedAutoMLPlatform(config)
    automl.fit(X, y)
    
    # Get results
    print(f"Best model: {automl.best_model}")
    print(f"Best metrics: {automl.best_metrics}")
    print(f"Feature importance: {automl.get_feature_importance()}")