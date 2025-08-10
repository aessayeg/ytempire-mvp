"""
Trend Prediction Model
Owner: ML Engineer

Prophet-based time series forecasting for YouTube trends.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import pickle
import os

from app.core.config import settings

logger = logging.getLogger(__name__)


class TrendPredictor:
    """YouTube trend prediction using Prophet time series forecasting."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path or f"{settings.MODEL_STORAGE_PATH}/trend_predictor.pkl"
        self.is_trained = False
        self.feature_columns = ['search_volume', 'video_count', 'avg_views']
        
    def prepare_data(self, data: pd.DataFrame, target_column: str = 'search_volume') -> pd.DataFrame:
        """Prepare data for Prophet model training."""
        try:
            # Ensure datetime index
            if 'date' in data.columns:
                data['ds'] = pd.to_datetime(data['date'])
            elif 'timestamp' in data.columns:
                data['ds'] = pd.to_datetime(data['timestamp'])
            else:
                data['ds'] = data.index
            
            # Set target variable
            data['y'] = data[target_column]
            
            # Add additional regressors
            prophet_data = data[['ds', 'y']].copy()
            
            # Add seasonality components
            prophet_data['month'] = prophet_data['ds'].dt.month
            prophet_data['day_of_week'] = prophet_data['ds'].dt.dayofweek
            prophet_data['is_weekend'] = (prophet_data['day_of_week'] >= 5).astype(int)
            
            # Add trend indicators
            if 'video_count' in data.columns:
                prophet_data['video_count'] = data['video_count']
            if 'avg_views' in data.columns:
                prophet_data['avg_views'] = data['avg_views']
            
            return prophet_data
            
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise
    
    def train(self, data: pd.DataFrame, target_column: str = 'search_volume') -> Dict:
        """Train the trend prediction model."""
        try:
            logger.info("Starting trend prediction model training")
            
            # Prepare data
            prophet_data = self.prepare_data(data, target_column)
            
            # Initialize Prophet model
            self.model = Prophet(
                growth='linear',
                seasonality_mode='multiplicative',
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                seasonality_prior_scale=0.1,
                holidays_prior_scale=0.1,
                changepoint_prior_scale=0.05,
                interval_width=0.95
            )
            
            # Add custom seasonalities
            self.model.add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=5
            )
            
            # Add additional regressors
            if 'video_count' in prophet_data.columns:
                self.model.add_regressor('video_count')
            if 'avg_views' in prophet_data.columns:
                self.model.add_regressor('avg_views')
            
            # Train model
            self.model.fit(prophet_data)
            self.is_trained = True
            
            # Calculate training metrics
            in_sample_forecast = self.model.predict(prophet_data)
            mae = np.mean(np.abs(prophet_data['y'] - in_sample_forecast['yhat']))
            mape = np.mean(np.abs((prophet_data['y'] - in_sample_forecast['yhat']) / prophet_data['y'])) * 100
            
            # Save model
            self._save_model()
            
            training_results = {
                'model_trained': True,
                'training_samples': len(prophet_data),
                'mae': mae,
                'mape': mape,
                'trained_at': datetime.now().isoformat()
            }
            
            logger.info(f"Model training completed. MAE: {mae:.2f}, MAPE: {mape:.2f}%")
            return training_results
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise
    
    def predict(self, periods: int = 30, include_history: bool = False) -> pd.DataFrame:
        """Generate trend predictions."""
        try:
            if not self.is_trained:
                self._load_model()
            
            if not self.model:
                raise ValueError("Model not trained or loaded")
            
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=periods, include_history=include_history)
            
            # Add regressors to future dataframe (use historical means as default)
            if 'video_count' in self.model.extra_regressors:
                future['video_count'] = future['video_count'].fillna(future['video_count'].mean())
            if 'avg_views' in self.model.extra_regressors:
                future['avg_views'] = future['avg_views'].fillna(future['avg_views'].mean())
            
            # Generate forecast
            forecast = self.model.predict(future)
            
            # Format results
            result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']].copy()
            result.columns = ['date', 'predicted', 'lower_bound', 'upper_bound', 'trend']
            
            # Add confidence scores
            result['confidence'] = 1 - (result['upper_bound'] - result['lower_bound']) / result['predicted']
            result['confidence'] = result['confidence'].clip(0, 1)
            
            logger.info(f"Generated predictions for {periods} periods")
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def predict_keyword_trends(self, keywords: List[str], periods: int = 7) -> Dict[str, Dict]:
        """Predict trends for multiple keywords."""
        try:
            predictions = {}
            
            for keyword in keywords:
                # This would typically fetch historical data for the keyword
                # For now, we'll generate sample data
                historical_data = self._get_keyword_data(keyword)
                
                if len(historical_data) < 10:  # Minimum data points required
                    logger.warning(f"Insufficient data for keyword: {keyword}")
                    continue
                
                # Train keyword-specific model
                keyword_model = TrendPredictor()
                keyword_model.train(historical_data)
                
                # Generate predictions
                keyword_forecast = keyword_model.predict(periods=periods)
                
                # Calculate trend direction
                recent_trend = keyword_forecast['trend'].tail(7).mean()
                historical_trend = keyword_forecast['trend'].head(7).mean()
                trend_direction = 'up' if recent_trend > historical_trend else 'down'
                
                predictions[keyword] = {
                    'forecast': keyword_forecast.to_dict('records'),
                    'trend_direction': trend_direction,
                    'confidence': keyword_forecast['confidence'].mean(),
                    'predicted_peak': keyword_forecast['predicted'].max(),
                    'predicted_at': datetime.now().isoformat()
                }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Keyword trend prediction failed: {str(e)}")
            raise
    
    def _get_keyword_data(self, keyword: str) -> pd.DataFrame:
        """Fetch historical data for keyword (mock implementation)."""
        # In production, this would fetch from YouTube API, Google Trends, etc.
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        np.random.seed(hash(keyword) % 2**32)  # Consistent random data for same keyword
        
        # Generate synthetic trend data
        base_trend = np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 50 + 100
        noise = np.random.normal(0, 20, len(dates))
        search_volume = base_trend + noise
        search_volume = np.maximum(search_volume, 0)  # No negative values
        
        video_count = search_volume * 0.1 + np.random.normal(0, 5, len(dates))
        avg_views = search_volume * 100 + np.random.normal(0, 1000, len(dates))
        
        return pd.DataFrame({
            'date': dates,
            'search_volume': search_volume,
            'video_count': np.maximum(video_count, 0),
            'avg_views': np.maximum(avg_views, 0)
        })
    
    def evaluate_model(self, test_data: pd.DataFrame) -> Dict:
        """Evaluate model performance on test data."""
        try:
            if not self.is_trained:
                raise ValueError("Model not trained")
            
            test_prepared = self.prepare_data(test_data)
            forecast = self.model.predict(test_prepared)
            
            # Calculate metrics
            mae = np.mean(np.abs(test_prepared['y'] - forecast['yhat']))
            mse = np.mean((test_prepared['y'] - forecast['yhat']) ** 2)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((test_prepared['y'] - forecast['yhat']) / test_prepared['y'])) * 100
            
            # Calculate R²
            ss_res = np.sum((test_prepared['y'] - forecast['yhat']) ** 2)
            ss_tot = np.sum((test_prepared['y'] - np.mean(test_prepared['y'])) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            return {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape,
                'r2': r2,
                'test_samples': len(test_prepared)
            }
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise
    
    def _save_model(self):
        """Save trained model to disk."""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'is_trained': self.is_trained,
                    'saved_at': datetime.now().isoformat()
                }, f)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
    
    def _load_model(self):
        """Load trained model from disk."""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.model = saved_data['model']
                    self.is_trained = saved_data['is_trained']
                logger.info(f"Model loaded from {self.model_path}")
            else:
                logger.warning("No saved model found")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")


# Utility functions for trend analysis
def analyze_trend_opportunities(predictions: Dict[str, Dict]) -> List[Dict]:
    """Analyze predictions to identify trending opportunities."""
    opportunities = []
    
    for keyword, data in predictions.items():
        if data['trend_direction'] == 'up' and data['confidence'] > 0.7:
            opportunities.append({
                'keyword': keyword,
                'opportunity_score': data['confidence'] * (data['predicted_peak'] / 100),
                'trend_direction': data['trend_direction'],
                'confidence': data['confidence'],
                'predicted_peak': data['predicted_peak'],
                'recommendation': 'high' if data['confidence'] > 0.8 else 'medium'
            })
    
    # Sort by opportunity score
    opportunities.sort(key=lambda x: x['opportunity_score'], reverse=True)
    return opportunities