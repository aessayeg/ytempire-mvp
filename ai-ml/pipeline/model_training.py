"""
ML Model Training Pipeline
Owner: AI/ML Team Lead
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
from prophet import Prophet
import mlflow
import mlflow.sklearn
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model training configuration"""
    model_type: str
    version: str
    params: Dict[str, Any]
    metrics_threshold: Dict[str, float]
    

class TrendPredictionModel:
    """
    Trend prediction model for YouTube content
    Uses Prophet for time series forecasting
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.metrics = {}
        self.feature_importance = {}
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for Prophet model"""
        # Prophet requires 'ds' and 'y' columns
        df_prophet = df.rename(columns={
            'date': 'ds',
            'views': 'y'
        })
        
        # Add additional regressors
        if 'subscribers' in df.columns:
            df_prophet['subscribers_scaled'] = df['subscribers'] / df['subscribers'].max()
        
        if 'videos_published' in df.columns:
            df_prophet['videos_published'] = df['videos_published']
        
        return df_prophet
    
    def train(self, train_data: pd.DataFrame) -> None:
        """Train the trend prediction model"""
        logger.info("Starting model training...")
        
        # Initialize Prophet with custom parameters
        self.model = Prophet(
            changepoint_prior_scale=self.config.params.get('changepoint_prior_scale', 0.05),
            seasonality_prior_scale=self.config.params.get('seasonality_prior_scale', 10),
            seasonality_mode=self.config.params.get('seasonality_mode', 'additive'),
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        
        # Add custom seasonalities
        self.model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=5
        )
        
        # Add regressors if available
        if 'subscribers_scaled' in train_data.columns:
            self.model.add_regressor('subscribers_scaled')
        
        if 'videos_published' in train_data.columns:
            self.model.add_regressor('videos_published')
        
        # Fit the model
        self.model.fit(train_data)
        
        logger.info("Model training completed")
    
    def predict(self, periods: int = 30) -> pd.DataFrame:
        """Make predictions for future periods"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods)
        
        # Add regressor values for future (using last known values)
        # In production, these would be more sophisticated
        if 'subscribers_scaled' in self.model.history.columns:
            future['subscribers_scaled'] = self.model.history['subscribers_scaled'].iloc[-1]
        
        if 'videos_published' in self.model.history.columns:
            future['videos_published'] = 1  # Assume 1 video per day
        
        # Make predictions
        forecast = self.model.predict(future)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']]
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = self.predict(len(test_data))
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        y_true = test_data['y'].values
        y_pred = predictions['yhat'].iloc[-len(test_data):].values
        
        self.metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        
        return self.metrics
    
    def save_model(self, path: str) -> None:
        """Save trained model"""
        model_data = {
            'model': self.model,
            'config': self.config,
            'metrics': self.metrics,
            'trained_at': datetime.utcnow().isoformat()
        }
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load trained model"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.config = model_data['config']
        self.metrics = model_data['metrics']
        logger.info(f"Model loaded from {path}")


class ContentQualityModel:
    """
    Model to predict content quality and performance
    """
    
    def __init__(self):
        self.model = None
        self.feature_columns = []
        self.scaler = None
        
    def extract_features(self, content_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from content data"""
        features = []
        
        # Title features
        title = content_data.get('title', '')
        features.extend([
            len(title),  # Title length
            title.count(' '),  # Word count
            1 if '?' in title else 0,  # Has question
            1 if any(char.isdigit() for char in title) else 0,  # Has number
            1 if title.isupper() else 0  # All caps
        ])
        
        # Description features
        description = content_data.get('description', '')
        features.extend([
            len(description),  # Description length
            description.count('\n'),  # Line breaks
            description.count('#'),  # Hashtag count
            description.count('http'),  # Link count
        ])
        
        # Tags features
        tags = content_data.get('tags', [])
        features.extend([
            len(tags),  # Number of tags
            sum(len(tag) for tag in tags),  # Total tag length
        ])
        
        # Video properties
        features.extend([
            content_data.get('duration', 600),  # Video duration
            content_data.get('hour_of_day', 12),  # Publishing hour
            content_data.get('day_of_week', 3),  # Publishing day
        ])
        
        return np.array(features).reshape(1, -1)
    
    def train(self, training_data: pd.DataFrame) -> None:
        """Train content quality model"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        
        # Prepare features and target
        X = training_data.drop(['performance_score'], axis=1)
        y = training_data['performance_score']
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_scaled, y)
        
        # Store feature importance
        self.feature_columns = X.columns.tolist()
        self.feature_importance = dict(zip(
            self.feature_columns,
            self.model.feature_importances_
        ))
    
    def predict_quality(self, content_data: Dict[str, Any]) -> float:
        """Predict content quality score"""
        if self.model is None:
            # Return default score if model not trained
            return 0.7
        
        features = self.extract_features(content_data)
        features_scaled = self.scaler.transform(features)
        score = self.model.predict(features_scaled)[0]
        
        return max(0.0, min(1.0, score))  # Clip to [0, 1]


class MLPipeline:
    """
    Main ML pipeline orchestrator
    """
    
    def __init__(self):
        self.trend_model = None
        self.quality_model = None
        self.experiment_name = "ytempire_models"
        
        # Initialize MLflow
        mlflow.set_experiment(self.experiment_name)
        
    def train_all_models(self, data_path: str) -> Dict[str, Any]:
        """Train all models in the pipeline"""
        results = {}
        
        # Train trend prediction model
        with mlflow.start_run(run_name="trend_prediction"):
            logger.info("Training trend prediction model...")
            
            # Load and prepare data
            df = pd.read_csv(f"{data_path}/trend_data.csv")
            df['date'] = pd.to_datetime(df['date'])
            
            # Split data
            train_size = int(len(df) * 0.8)
            train_data = df[:train_size]
            test_data = df[train_size:]
            
            # Train model
            config = ModelConfig(
                model_type="prophet",
                version="1.0.0",
                params={
                    'changepoint_prior_scale': 0.05,
                    'seasonality_prior_scale': 10
                },
                metrics_threshold={
                    'mape': 20.0,
                    'r2': 0.7
                }
            )
            
            self.trend_model = TrendPredictionModel(config)
            prepared_train = self.trend_model.prepare_data(train_data)
            self.trend_model.train(prepared_train)
            
            # Evaluate
            prepared_test = self.trend_model.prepare_data(test_data)
            metrics = self.trend_model.evaluate(prepared_test)
            
            # Log to MLflow
            mlflow.log_params(config.params)
            mlflow.log_metrics(metrics)
            
            # Save model
            self.trend_model.save_model("models/trend_model.pkl")
            mlflow.log_artifact("models/trend_model.pkl")
            
            results['trend_model'] = metrics
        
        # Train content quality model
        with mlflow.start_run(run_name="content_quality"):
            logger.info("Training content quality model...")
            
            # Load training data
            quality_data = pd.read_csv(f"{data_path}/quality_data.csv")
            
            self.quality_model = ContentQualityModel()
            self.quality_model.train(quality_data)
            
            # Save model
            joblib.dump(self.quality_model, "models/quality_model.pkl")
            mlflow.log_artifact("models/quality_model.pkl")
            
            # Log feature importance
            mlflow.log_dict(self.quality_model.feature_importance, "feature_importance.json")
            
            results['quality_model'] = {
                'feature_importance': self.quality_model.feature_importance
            }
        
        return results
    
    def predict_trend(self, historical_data: pd.DataFrame, days_ahead: int = 30) -> pd.DataFrame:
        """Predict future trends"""
        if self.trend_model is None:
            self.trend_model = TrendPredictionModel(ModelConfig(
                model_type="prophet",
                version="1.0.0",
                params={},
                metrics_threshold={}
            ))
            self.trend_model.load_model("models/trend_model.pkl")
        
        prepared_data = self.trend_model.prepare_data(historical_data)
        self.trend_model.train(prepared_data)
        predictions = self.trend_model.predict(days_ahead)
        
        return predictions
    
    def score_content(self, content_data: Dict[str, Any]) -> float:
        """Score content quality"""
        if self.quality_model is None:
            self.quality_model = joblib.load("models/quality_model.pkl")
        
        return self.quality_model.predict_quality(content_data)