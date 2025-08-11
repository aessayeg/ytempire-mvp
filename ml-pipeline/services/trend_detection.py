"""
Trend Detection Model Integration
Prophet and LSTM implementation for trend analysis
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import json

# Prophet for time series forecasting
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# TensorFlow/Keras for LSTM
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

@dataclass
class TrendPrediction:
    """Trend prediction result"""
    trend_id: str
    keyword: str
    current_score: float
    predicted_score: float
    confidence: float
    trend_direction: str  # rising, falling, stable
    velocity: float  # rate of change
    peak_time: Optional[datetime]
    recommended_action: str
    metadata: Dict[str, Any]


class ProphetTrendModel:
    """Prophet-based trend detection and forecasting"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    def train(
        self,
        data: pd.DataFrame,
        keyword: str,
        seasonality_mode: str = 'multiplicative'
    ) -> Dict[str, Any]:
        """
        Train Prophet model on historical trend data
        
        Args:
            data: DataFrame with 'ds' (date) and 'y' (metric) columns
            keyword: Keyword/topic being modeled
            seasonality_mode: 'additive' or 'multiplicative'
        """
        try:
            # Initialize Prophet model
            model = Prophet(
                seasonality_mode=seasonality_mode,
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05,
                interval_width=0.95
            )
            
            # Add custom seasonalities if needed
            model.add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=5
            )
            
            # Fit model
            model.fit(data)
            
            # Store model
            self.models[keyword] = model
            
            # Cross-validation
            df_cv = cross_validation(
                model,
                initial='30 days',
                period='7 days',
                horizon='14 days'
            )
            
            # Calculate metrics
            df_metrics = performance_metrics(df_cv)
            
            return {
                'keyword': keyword,
                'trained': True,
                'mape': df_metrics['mape'].mean(),
                'rmse': df_metrics['rmse'].mean(),
                'coverage': df_metrics['coverage'].mean()
            }
            
        except Exception as e:
            logger.error(f"Prophet training failed for {keyword}: {e}")
            raise
    
    def predict(
        self,
        keyword: str,
        periods: int = 7,
        freq: str = 'D'
    ) -> pd.DataFrame:
        """
        Make trend predictions
        
        Args:
            keyword: Keyword to predict
            periods: Number of periods to forecast
            freq: Frequency ('D' for daily, 'H' for hourly)
        """
        if keyword not in self.models:
            raise ValueError(f"No model found for keyword: {keyword}")
        
        model = self.models[keyword]
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, freq=freq)
        
        # Make predictions
        forecast = model.predict(future)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']]
    
    def detect_changepoints(self, keyword: str) -> List[datetime]:
        """Detect significant trend changepoints"""
        if keyword not in self.models:
            return []
        
        model = self.models[keyword]
        
        # Get changepoints
        changepoints = model.changepoints
        
        # Filter significant changepoints
        significant_changepoints = []
        
        # Add logic to filter based on magnitude of change
        # This is simplified - you'd want more sophisticated filtering
        
        return changepoints.tolist()


class LSTMTrendModel:
    """LSTM-based trend detection for complex patterns"""
    
    def __init__(self, sequence_length: int = 30):
        self.sequence_length = sequence_length
        self.models = {}
        self.scalers = {}
        
    def prepare_sequences(
        self,
        data: np.ndarray,
        sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        
        return np.array(X), np.array(y)
    
    def build_model(
        self,
        input_shape: Tuple[int, int],
        units: List[int] = [128, 64, 32]
    ) -> keras.Model:
        """Build LSTM model architecture"""
        model = models.Sequential()
        
        # First LSTM layer with return sequences
        model.add(layers.LSTM(
            units[0],
            return_sequences=True,
            input_shape=input_shape
        ))
        model.add(layers.Dropout(0.2))
        
        # Additional LSTM layers
        for i, unit_count in enumerate(units[1:-1]):
            model.add(layers.LSTM(unit_count, return_sequences=True))
            model.add(layers.Dropout(0.2))
        
        # Final LSTM layer
        model.add(layers.LSTM(units[-1], return_sequences=False))
        model.add(layers.Dropout(0.2))
        
        # Dense layers
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(
        self,
        data: pd.DataFrame,
        keyword: str,
        target_column: str = 'search_volume',
        epochs: int = 50,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """Train LSTM model on trend data"""
        try:
            # Extract values
            values = data[target_column].values.reshape(-1, 1)
            
            # Scale data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(values)
            
            # Store scaler
            self.scalers[keyword] = scaler
            
            # Prepare sequences
            X, y = self.prepare_sequences(scaled_data, self.sequence_length)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Build model
            model = self.build_model((self.sequence_length, 1))
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        patience=10,
                        restore_best_weights=True
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        patience=5,
                        factor=0.5
                    )
                ]
            )
            
            # Store model
            self.models[keyword] = model
            
            # Evaluate
            val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
            
            return {
                'keyword': keyword,
                'trained': True,
                'val_loss': float(val_loss),
                'val_mae': float(val_mae),
                'epochs_trained': len(history.history['loss'])
            }
            
        except Exception as e:
            logger.error(f"LSTM training failed for {keyword}: {e}")
            raise
    
    def predict(
        self,
        keyword: str,
        recent_data: np.ndarray,
        steps: int = 7
    ) -> np.ndarray:
        """Make multi-step predictions"""
        if keyword not in self.models:
            raise ValueError(f"No model found for keyword: {keyword}")
        
        model = self.models[keyword]
        scaler = self.scalers[keyword]
        
        # Scale input data
        scaled_input = scaler.transform(recent_data.reshape(-1, 1))
        
        # Prepare sequence
        if len(scaled_input) < self.sequence_length:
            # Pad with zeros if needed
            padding = np.zeros((self.sequence_length - len(scaled_input), 1))
            scaled_input = np.vstack([padding, scaled_input])
        
        current_sequence = scaled_input[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        
        predictions = []
        
        for _ in range(steps):
            # Make prediction
            pred = model.predict(current_sequence, verbose=0)
            predictions.append(pred[0, 0])
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = pred[0, 0]
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions)
        
        return predictions.flatten()


class TrendDetectionSystem:
    """Main trend detection system combining Prophet and LSTM"""
    
    def __init__(self):
        self.prophet_model = ProphetTrendModel()
        self.lstm_model = LSTMTrendModel()
        self.trend_cache = {}
        
    async def analyze_trend(
        self,
        keyword: str,
        historical_data: pd.DataFrame,
        use_ensemble: bool = True
    ) -> TrendPrediction:
        """
        Analyze trend using ensemble of models
        
        Args:
            keyword: Trend keyword/topic
            historical_data: Historical trend data
            use_ensemble: Whether to use both models
        """
        try:
            predictions = {}
            
            # Prophet prediction
            if 'ds' in historical_data.columns and 'y' in historical_data.columns:
                # Train if needed
                if keyword not in self.prophet_model.models:
                    self.prophet_model.train(historical_data, keyword)
                
                # Get predictions
                prophet_forecast = self.prophet_model.predict(keyword, periods=7)
                predictions['prophet'] = prophet_forecast
            
            # LSTM prediction
            if use_ensemble and len(historical_data) > self.lstm_model.sequence_length:
                # Train if needed
                if keyword not in self.lstm_model.models:
                    self.lstm_model.train(historical_data, keyword)
                
                # Get predictions
                recent_values = historical_data['y'].values[-30:]
                lstm_forecast = self.lstm_model.predict(keyword, recent_values, steps=7)
                predictions['lstm'] = lstm_forecast
            
            # Combine predictions
            trend_prediction = self._combine_predictions(
                keyword,
                historical_data,
                predictions
            )
            
            # Cache result
            self.trend_cache[keyword] = {
                'prediction': trend_prediction,
                'timestamp': datetime.utcnow()
            }
            
            return trend_prediction
            
        except Exception as e:
            logger.error(f"Trend analysis failed for {keyword}: {e}")
            raise
    
    def _combine_predictions(
        self,
        keyword: str,
        historical_data: pd.DataFrame,
        predictions: Dict[str, Any]
    ) -> TrendPrediction:
        """Combine predictions from multiple models"""
        
        # Get current value
        current_value = historical_data['y'].iloc[-1]
        
        # Calculate predicted value (ensemble average)
        predicted_values = []
        
        if 'prophet' in predictions:
            prophet_pred = predictions['prophet']['yhat'].iloc[-1]
            predicted_values.append(prophet_pred)
        
        if 'lstm' in predictions:
            lstm_pred = predictions['lstm'][-1]
            predicted_values.append(lstm_pred)
        
        if predicted_values:
            predicted_value = np.mean(predicted_values)
        else:
            predicted_value = current_value
        
        # Calculate trend metrics
        change_rate = (predicted_value - current_value) / current_value if current_value != 0 else 0
        
        # Determine trend direction
        if change_rate > 0.1:
            direction = "rising"
        elif change_rate < -0.1:
            direction = "falling"
        else:
            direction = "stable"
        
        # Calculate velocity (rate of change)
        recent_changes = historical_data['y'].pct_change().dropna().tail(7)
        velocity = recent_changes.mean() if len(recent_changes) > 0 else 0
        
        # Determine recommended action
        if direction == "rising" and velocity > 0.05:
            action = "capitalize_immediately"
        elif direction == "rising":
            action = "prepare_content"
        elif direction == "falling":
            action = "pivot_strategy"
        else:
            action = "monitor"
        
        # Calculate confidence
        confidence = self._calculate_confidence(historical_data, predictions)
        
        return TrendPrediction(
            trend_id=f"trend_{keyword}_{datetime.utcnow().strftime('%Y%m%d')}",
            keyword=keyword,
            current_score=float(current_value),
            predicted_score=float(predicted_value),
            confidence=confidence,
            trend_direction=direction,
            velocity=float(velocity),
            peak_time=self._estimate_peak_time(predictions),
            recommended_action=action,
            metadata={
                'models_used': list(predictions.keys()),
                'forecast_horizon': 7,
                'last_updated': datetime.utcnow().isoformat()
            }
        )
    
    def _calculate_confidence(
        self,
        historical_data: pd.DataFrame,
        predictions: Dict[str, Any]
    ) -> float:
        """Calculate prediction confidence score"""
        
        # Base confidence on data quality and model agreement
        confidence_factors = []
        
        # Data completeness
        data_completeness = 1.0 - (historical_data.isnull().sum().sum() / historical_data.size)
        confidence_factors.append(data_completeness)
        
        # Historical volatility (lower volatility = higher confidence)
        volatility = historical_data['y'].pct_change().std()
        volatility_score = max(0, 1 - volatility)
        confidence_factors.append(volatility_score)
        
        # Model agreement (if using ensemble)
        if len(predictions) > 1:
            # Calculate variance between model predictions
            pred_values = []
            if 'prophet' in predictions:
                pred_values.append(predictions['prophet']['yhat'].iloc[-1])
            if 'lstm' in predictions:
                pred_values.append(predictions['lstm'][-1])
            
            if len(pred_values) > 1:
                agreement = 1 - (np.std(pred_values) / np.mean(pred_values))
                confidence_factors.append(max(0, agreement))
        
        return float(np.mean(confidence_factors))
    
    def _estimate_peak_time(self, predictions: Dict[str, Any]) -> Optional[datetime]:
        """Estimate when trend will peak"""
        
        if 'prophet' in predictions:
            forecast = predictions['prophet']
            peak_idx = forecast['yhat'].idxmax()
            if peak_idx < len(forecast) - 1:  # Peak is within forecast
                return forecast.loc[peak_idx, 'ds']
        
        return None
    
    async def get_trending_topics(
        self,
        min_score: float = 0.7,
        limit: int = 10
    ) -> List[TrendPrediction]:
        """Get currently trending topics"""
        
        # Filter cached trends by score
        trending = []
        
        for keyword, cache_data in self.trend_cache.items():
            prediction = cache_data['prediction']
            
            # Check cache age
            cache_age = datetime.utcnow() - cache_data['timestamp']
            if cache_age > timedelta(hours=6):
                continue  # Skip old cache entries
            
            # Check trend score
            trend_score = prediction.predicted_score / prediction.current_score if prediction.current_score > 0 else 0
            
            if trend_score >= min_score:
                trending.append(prediction)
        
        # Sort by velocity and predicted score
        trending.sort(key=lambda x: (x.velocity, x.predicted_score), reverse=True)
        
        return trending[:limit]


# Global instance
trend_detection = TrendDetectionSystem()