"""
Advanced Forecasting Models Service
Provides sophisticated time series forecasting and predictive analytics
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Optional imports for advanced forecasting
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.config import settings
from app.models.video import Video
from app.models.channel import Channel
from app.models.analytics import Analytics

logger = logging.getLogger(__name__)


class ForecastModel(Enum):
    """Types of forecasting models available"""
    ARIMA = "arima"
    SARIMA = "sarima"
    PROPHET = "prophet"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    LSTM = "lstm"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    LINEAR_REGRESSION = "linear_regression"
    ENSEMBLE = "ensemble"
    NEURAL_PROPHET = "neural_prophet"
    XGBOOST = "xgboost"
    VAR = "vector_autoregression"


class ForecastMetric(Enum):
    """Metrics that can be forecasted"""
    REVENUE = "revenue"
    VIEWS = "views"
    SUBSCRIBERS = "subscribers"
    ENGAGEMENT = "engagement"
    WATCH_TIME = "watch_time"
    CPM = "cpm"
    CTR = "click_through_rate"
    CONVERSIONS = "conversions"
    CHANNEL_GROWTH = "channel_growth"
    VIDEO_PERFORMANCE = "video_performance"


@dataclass
class ForecastConfig:
    """Configuration for a forecast model"""
    model_type: ForecastModel
    metric: ForecastMetric
    horizon: int  # Forecast horizon in days
    confidence_level: float = 0.95
    seasonality: Optional[str] = None  # 'daily', 'weekly', 'monthly', 'yearly'
    include_holidays: bool = False
    include_external_regressors: bool = False
    hyperparameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ForecastResult:
    """Result from a forecasting model"""
    model_type: ForecastModel
    metric: ForecastMetric
    predictions: pd.DataFrame
    confidence_intervals: Optional[pd.DataFrame] = None
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)
    model_params: Dict[str, Any] = field(default_factory=dict)
    feature_importance: Optional[Dict[str, float]] = None
    residuals: Optional[pd.Series] = None
    forecast_date: datetime = field(default_factory=datetime.now)


class AdvancedForecastingModels:
    """Service for advanced forecasting and predictive analytics"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.forecast_cache: Dict[str, Tuple[ForecastResult, datetime]] = {}
        self.model_registry: Dict[str, Dict[str, Any]] = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize available forecasting models"""
        self.model_registry = {
            ForecastModel.ARIMA: {
                'name': 'ARIMA',
                'description': 'AutoRegressive Integrated Moving Average',
                'suitable_for': ['short-term', 'univariate', 'stationary'],
                'requires': ['statsmodels']
            },
            ForecastModel.SARIMA: {
                'name': 'SARIMA',
                'description': 'Seasonal ARIMA',
                'suitable_for': ['seasonal', 'medium-term', 'univariate'],
                'requires': ['statsmodels']
            },
            ForecastModel.PROPHET: {
                'name': 'Prophet',
                'description': 'Facebook Prophet for time series',
                'suitable_for': ['long-term', 'seasonal', 'holidays', 'changepoints'],
                'requires': ['prophet']
            },
            ForecastModel.EXPONENTIAL_SMOOTHING: {
                'name': 'Exponential Smoothing',
                'description': 'Holt-Winters exponential smoothing',
                'suitable_for': ['short-term', 'seasonal', 'trend'],
                'requires': ['statsmodels']
            },
            ForecastModel.RANDOM_FOREST: {
                'name': 'Random Forest',
                'description': 'Random Forest regression for time series',
                'suitable_for': ['non-linear', 'feature-rich', 'robust'],
                'requires': ['sklearn']
            },
            ForecastModel.GRADIENT_BOOSTING: {
                'name': 'Gradient Boosting',
                'description': 'Gradient Boosting for time series',
                'suitable_for': ['non-linear', 'complex-patterns', 'high-accuracy'],
                'requires': ['sklearn']
            },
            ForecastModel.ENSEMBLE: {
                'name': 'Ensemble',
                'description': 'Combination of multiple models',
                'suitable_for': ['high-accuracy', 'robust', 'all-purpose'],
                'requires': ['multiple']
            }
        }
    
    async def create_forecast(
        self,
        config: ForecastConfig,
        historical_data: pd.DataFrame,
        db: Optional[AsyncSession] = None
    ) -> ForecastResult:
        """Create a forecast using specified model and configuration"""
        # Validate data
        if historical_data.empty:
            raise ValueError("Historical data is empty")
        
        # Prepare data
        prepared_data = self._prepare_data(historical_data, config)
        
        # Check cache
        cache_key = self._generate_cache_key(config, prepared_data)
        cached_result = self._get_cached_forecast(cache_key)
        if cached_result:
            return cached_result
        
        # Select and train model
        if config.model_type == ForecastModel.ARIMA:
            result = await self._forecast_arima(prepared_data, config)
        elif config.model_type == ForecastModel.SARIMA:
            result = await self._forecast_sarima(prepared_data, config)
        elif config.model_type == ForecastModel.PROPHET:
            result = await self._forecast_prophet(prepared_data, config)
        elif config.model_type == ForecastModel.EXPONENTIAL_SMOOTHING:
            result = await self._forecast_exponential_smoothing(prepared_data, config)
        elif config.model_type == ForecastModel.RANDOM_FOREST:
            result = await self._forecast_random_forest(prepared_data, config)
        elif config.model_type == ForecastModel.GRADIENT_BOOSTING:
            result = await self._forecast_gradient_boosting(prepared_data, config)
        elif config.model_type == ForecastModel.ENSEMBLE:
            result = await self._forecast_ensemble(prepared_data, config)
        else:
            result = await self._forecast_baseline(prepared_data, config)
        
        # Cache result
        self._cache_forecast(cache_key, result)
        
        return result
    
    def _prepare_data(self, data: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
        """Prepare data for forecasting"""
        # Ensure datetime index
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data = data.set_index('date')
        elif not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Sort by date
        data = data.sort_index()
        
        # Handle missing values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Ensure numeric data
        if config.metric.value in data.columns:
            data = data[[config.metric.value]]
        else:
            # Use first numeric column
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                data = data[[numeric_cols[0]]]
        
        return data
    
    async def _forecast_arima(self, data: pd.DataFrame, config: ForecastConfig) -> ForecastResult:
        """ARIMA forecasting"""
        if not STATSMODELS_AVAILABLE:
            return await self._forecast_baseline(data, config)
        
        try:
            # Determine ARIMA parameters using auto-selection
            p, d, q = self._select_arima_params(data)
            
            # Fit ARIMA model
            model = ARIMA(data.iloc[:, 0], order=(p, d, q))
            fitted_model = model.fit()
            
            # Generate forecasts
            forecast = fitted_model.forecast(steps=config.horizon)
            
            # Get confidence intervals
            forecast_df = fitted_model.get_forecast(steps=config.horizon)
            confidence_intervals = forecast_df.conf_int(alpha=1-config.confidence_level)
            
            # Create predictions dataframe
            future_dates = pd.date_range(
                start=data.index[-1] + timedelta(days=1),
                periods=config.horizon,
                freq='D'
            )
            
            predictions_df = pd.DataFrame({
                'forecast': forecast,
                'date': future_dates
            }).set_index('date')
            
            # Calculate accuracy metrics on training data
            in_sample_pred = fitted_model.fittedvalues
            accuracy_metrics = self._calculate_accuracy_metrics(
                data.iloc[:, 0].values[-len(in_sample_pred):],
                in_sample_pred
            )
            
            return ForecastResult(
                model_type=ForecastModel.ARIMA,
                metric=config.metric,
                predictions=predictions_df,
                confidence_intervals=confidence_intervals,
                accuracy_metrics=accuracy_metrics,
                model_params={'p': p, 'd': d, 'q': q, 'aic': fitted_model.aic},
                residuals=fitted_model.resid
            )
            
        except Exception as e:
            logger.error(f"ARIMA forecast failed: {e}")
            return await self._forecast_baseline(data, config)
    
    async def _forecast_sarima(self, data: pd.DataFrame, config: ForecastConfig) -> ForecastResult:
        """SARIMA forecasting with seasonality"""
        if not STATSMODELS_AVAILABLE:
            return await self._forecast_baseline(data, config)
        
        try:
            # Determine seasonal period
            seasonal_period = self._detect_seasonality(data)
            
            # Fit SARIMA model
            model = SARIMAX(
                data.iloc[:, 0],
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, seasonal_period),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted_model = model.fit(disp=False)
            
            # Generate forecasts
            forecast = fitted_model.forecast(steps=config.horizon)
            
            # Get confidence intervals
            forecast_df = fitted_model.get_forecast(steps=config.horizon)
            confidence_intervals = forecast_df.conf_int(alpha=1-config.confidence_level)
            
            # Create predictions dataframe
            future_dates = pd.date_range(
                start=data.index[-1] + timedelta(days=1),
                periods=config.horizon,
                freq='D'
            )
            
            predictions_df = pd.DataFrame({
                'forecast': forecast,
                'date': future_dates
            }).set_index('date')
            
            # Calculate accuracy metrics
            in_sample_pred = fitted_model.fittedvalues
            accuracy_metrics = self._calculate_accuracy_metrics(
                data.iloc[:, 0].values[-len(in_sample_pred):],
                in_sample_pred
            )
            
            return ForecastResult(
                model_type=ForecastModel.SARIMA,
                metric=config.metric,
                predictions=predictions_df,
                confidence_intervals=confidence_intervals,
                accuracy_metrics=accuracy_metrics,
                model_params={
                    'order': (1, 1, 1),
                    'seasonal_order': (1, 1, 1, seasonal_period),
                    'aic': fitted_model.aic
                }
            )
            
        except Exception as e:
            logger.error(f"SARIMA forecast failed: {e}")
            return await self._forecast_baseline(data, config)
    
    async def _forecast_prophet(self, data: pd.DataFrame, config: ForecastConfig) -> ForecastResult:
        """Prophet forecasting"""
        if not PROPHET_AVAILABLE:
            return await self._forecast_baseline(data, config)
        
        try:
            # Prepare data for Prophet
            prophet_data = data.reset_index()
            prophet_data.columns = ['ds', 'y']
            
            # Initialize Prophet model
            model = Prophet(
                daily_seasonality=config.seasonality == 'daily',
                weekly_seasonality=config.seasonality == 'weekly',
                yearly_seasonality=config.seasonality == 'yearly',
                interval_width=config.confidence_level
            )
            
            # Add holidays if requested
            if config.include_holidays:
                model.add_country_holidays(country_name='US')
            
            # Fit model
            model.fit(prophet_data)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=config.horizon)
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Extract predictions
            predictions_df = forecast[['ds', 'yhat']].tail(config.horizon)
            predictions_df = predictions_df.rename(columns={'ds': 'date', 'yhat': 'forecast'})
            predictions_df = predictions_df.set_index('date')
            
            # Extract confidence intervals
            confidence_intervals = pd.DataFrame({
                'lower': forecast['yhat_lower'].tail(config.horizon).values,
                'upper': forecast['yhat_upper'].tail(config.horizon).values
            }, index=predictions_df.index)
            
            # Calculate accuracy metrics
            historical_forecast = forecast[forecast['ds'] <= data.index[-1]]
            if len(historical_forecast) > 0:
                accuracy_metrics = self._calculate_accuracy_metrics(
                    prophet_data['y'].values[-len(historical_forecast):],
                    historical_forecast['yhat'].values
                )
            else:
                accuracy_metrics = {}
            
            return ForecastResult(
                model_type=ForecastModel.PROPHET,
                metric=config.metric,
                predictions=predictions_df,
                confidence_intervals=confidence_intervals,
                accuracy_metrics=accuracy_metrics,
                model_params={
                    'changepoint_prior_scale': model.changepoint_prior_scale,
                    'seasonality_prior_scale': model.seasonality_prior_scale
                }
            )
            
        except Exception as e:
            logger.error(f"Prophet forecast failed: {e}")
            return await self._forecast_baseline(data, config)
    
    async def _forecast_exponential_smoothing(self, data: pd.DataFrame, config: ForecastConfig) -> ForecastResult:
        """Exponential Smoothing forecasting"""
        if not STATSMODELS_AVAILABLE:
            return await self._forecast_baseline(data, config)
        
        try:
            # Detect trend and seasonal components
            seasonal_period = self._detect_seasonality(data)
            
            # Fit Exponential Smoothing model
            model = ExponentialSmoothing(
                data.iloc[:, 0],
                seasonal_periods=seasonal_period if seasonal_period > 1 else None,
                trend='add',
                seasonal='add' if seasonal_period > 1 else None
            )
            fitted_model = model.fit()
            
            # Generate forecasts
            forecast = fitted_model.forecast(steps=config.horizon)
            
            # Create predictions dataframe
            future_dates = pd.date_range(
                start=data.index[-1] + timedelta(days=1),
                periods=config.horizon,
                freq='D'
            )
            
            predictions_df = pd.DataFrame({
                'forecast': forecast,
                'date': future_dates
            }).set_index('date')
            
            # Calculate confidence intervals (approximate)
            std_error = np.std(fitted_model.resid)
            z_score = stats.norm.ppf((1 + config.confidence_level) / 2)
            confidence_intervals = pd.DataFrame({
                'lower': forecast - z_score * std_error,
                'upper': forecast + z_score * std_error
            }, index=predictions_df.index)
            
            # Calculate accuracy metrics
            accuracy_metrics = self._calculate_accuracy_metrics(
                data.iloc[:, 0].values,
                fitted_model.fittedvalues
            )
            
            return ForecastResult(
                model_type=ForecastModel.EXPONENTIAL_SMOOTHING,
                metric=config.metric,
                predictions=predictions_df,
                confidence_intervals=confidence_intervals,
                accuracy_metrics=accuracy_metrics,
                model_params={
                    'smoothing_level': fitted_model.params.get('smoothing_level', None),
                    'smoothing_trend': fitted_model.params.get('smoothing_trend', None),
                    'smoothing_seasonal': fitted_model.params.get('smoothing_seasonal', None)
                },
                residuals=fitted_model.resid
            )
            
        except Exception as e:
            logger.error(f"Exponential Smoothing forecast failed: {e}")
            return await self._forecast_baseline(data, config)
    
    async def _forecast_random_forest(self, data: pd.DataFrame, config: ForecastConfig) -> ForecastResult:
        """Random Forest forecasting"""
        if not SKLEARN_AVAILABLE:
            return await self._forecast_baseline(data, config)
        
        try:
            # Create features for time series
            X, y = self._create_time_features(data)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            
            # Train Random Forest
            model = RandomForestRegressor(
                n_estimators=config.hyperparameters.get('n_estimators', 100),
                max_depth=config.hyperparameters.get('max_depth', None),
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Generate future features
            future_X = self._create_future_features(data, config.horizon)
            
            # Make predictions
            predictions = model.predict(future_X)
            
            # Create predictions dataframe
            future_dates = pd.date_range(
                start=data.index[-1] + timedelta(days=1),
                periods=config.horizon,
                freq='D'
            )
            
            predictions_df = pd.DataFrame({
                'forecast': predictions,
                'date': future_dates
            }).set_index('date')
            
            # Calculate confidence intervals using prediction intervals from trees
            tree_predictions = np.array([tree.predict(future_X) for tree in model.estimators_])
            lower = np.percentile(tree_predictions, (1 - config.confidence_level) / 2 * 100, axis=0)
            upper = np.percentile(tree_predictions, (1 + config.confidence_level) / 2 * 100, axis=0)
            
            confidence_intervals = pd.DataFrame({
                'lower': lower,
                'upper': upper
            }, index=predictions_df.index)
            
            # Calculate accuracy metrics
            test_predictions = model.predict(X_test)
            accuracy_metrics = self._calculate_accuracy_metrics(y_test, test_predictions)
            
            # Feature importance
            feature_importance = dict(zip(
                [f'lag_{i}' for i in range(1, X.shape[1] + 1)],
                model.feature_importances_
            ))
            
            return ForecastResult(
                model_type=ForecastModel.RANDOM_FOREST,
                metric=config.metric,
                predictions=predictions_df,
                confidence_intervals=confidence_intervals,
                accuracy_metrics=accuracy_metrics,
                model_params={'n_estimators': model.n_estimators, 'max_depth': model.max_depth},
                feature_importance=feature_importance
            )
            
        except Exception as e:
            logger.error(f"Random Forest forecast failed: {e}")
            return await self._forecast_baseline(data, config)
    
    async def _forecast_gradient_boosting(self, data: pd.DataFrame, config: ForecastConfig) -> ForecastResult:
        """Gradient Boosting forecasting"""
        if not SKLEARN_AVAILABLE:
            return await self._forecast_baseline(data, config)
        
        try:
            # Create features for time series
            X, y = self._create_time_features(data)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            
            # Train Gradient Boosting
            model = GradientBoostingRegressor(
                n_estimators=config.hyperparameters.get('n_estimators', 100),
                learning_rate=config.hyperparameters.get('learning_rate', 0.1),
                max_depth=config.hyperparameters.get('max_depth', 3),
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Generate future features
            future_X = self._create_future_features(data, config.horizon)
            
            # Make predictions
            predictions = model.predict(future_X)
            
            # Create predictions dataframe
            future_dates = pd.date_range(
                start=data.index[-1] + timedelta(days=1),
                periods=config.horizon,
                freq='D'
            )
            
            predictions_df = pd.DataFrame({
                'forecast': predictions,
                'date': future_dates
            }).set_index('date')
            
            # Calculate confidence intervals (approximate)
            std_error = np.std(y_train - model.predict(X_train))
            z_score = stats.norm.ppf((1 + config.confidence_level) / 2)
            
            confidence_intervals = pd.DataFrame({
                'lower': predictions - z_score * std_error,
                'upper': predictions + z_score * std_error
            }, index=predictions_df.index)
            
            # Calculate accuracy metrics
            test_predictions = model.predict(X_test)
            accuracy_metrics = self._calculate_accuracy_metrics(y_test, test_predictions)
            
            # Feature importance
            feature_importance = dict(zip(
                [f'lag_{i}' for i in range(1, X.shape[1] + 1)],
                model.feature_importances_
            ))
            
            return ForecastResult(
                model_type=ForecastModel.GRADIENT_BOOSTING,
                metric=config.metric,
                predictions=predictions_df,
                confidence_intervals=confidence_intervals,
                accuracy_metrics=accuracy_metrics,
                model_params={
                    'n_estimators': model.n_estimators,
                    'learning_rate': model.learning_rate,
                    'max_depth': model.max_depth
                },
                feature_importance=feature_importance
            )
            
        except Exception as e:
            logger.error(f"Gradient Boosting forecast failed: {e}")
            return await self._forecast_baseline(data, config)
    
    async def _forecast_ensemble(self, data: pd.DataFrame, config: ForecastConfig) -> ForecastResult:
        """Ensemble forecasting combining multiple models"""
        models_to_use = [
            ForecastModel.EXPONENTIAL_SMOOTHING,
            ForecastModel.RANDOM_FOREST,
            ForecastModel.GRADIENT_BOOSTING
        ]
        
        predictions = []
        weights = []
        
        for model_type in models_to_use:
            try:
                # Create config for individual model
                model_config = ForecastConfig(
                    model_type=model_type,
                    metric=config.metric,
                    horizon=config.horizon,
                    confidence_level=config.confidence_level
                )
                
                # Get forecast
                if model_type == ForecastModel.EXPONENTIAL_SMOOTHING:
                    result = await self._forecast_exponential_smoothing(data, model_config)
                elif model_type == ForecastModel.RANDOM_FOREST:
                    result = await self._forecast_random_forest(data, model_config)
                elif model_type == ForecastModel.GRADIENT_BOOSTING:
                    result = await self._forecast_gradient_boosting(data, model_config)
                else:
                    continue
                
                # Add to ensemble
                predictions.append(result.predictions['forecast'].values)
                
                # Weight based on accuracy (if available)
                if result.accuracy_metrics.get('r2'):
                    weights.append(max(0, result.accuracy_metrics['r2']))
                else:
                    weights.append(1.0)
                    
            except Exception as e:
                logger.warning(f"Model {model_type} failed in ensemble: {e}")
                continue
        
        if not predictions:
            return await self._forecast_baseline(data, config)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted average of predictions
        ensemble_predictions = np.average(predictions, axis=0, weights=weights)
        
        # Create predictions dataframe
        future_dates = pd.date_range(
            start=data.index[-1] + timedelta(days=1),
            periods=config.horizon,
            freq='D'
        )
        
        predictions_df = pd.DataFrame({
            'forecast': ensemble_predictions,
            'date': future_dates
        }).set_index('date')
        
        # Calculate confidence intervals (wider for ensemble)
        std_predictions = np.std(predictions, axis=0)
        z_score = stats.norm.ppf((1 + config.confidence_level) / 2)
        
        confidence_intervals = pd.DataFrame({
            'lower': ensemble_predictions - z_score * std_predictions,
            'upper': ensemble_predictions + z_score * std_predictions
        }, index=predictions_df.index)
        
        return ForecastResult(
            model_type=ForecastModel.ENSEMBLE,
            metric=config.metric,
            predictions=predictions_df,
            confidence_intervals=confidence_intervals,
            accuracy_metrics={'ensemble_models': len(predictions)},
            model_params={'weights': weights.tolist(), 'models': [m.value for m in models_to_use]}
        )
    
    async def _forecast_baseline(self, data: pd.DataFrame, config: ForecastConfig) -> ForecastResult:
        """Baseline forecasting using simple methods"""
        # Use simple moving average
        window = min(7, len(data) // 4)
        rolling_mean = data.iloc[:, 0].rolling(window=window).mean()
        last_mean = rolling_mean.iloc[-1]
        
        # Calculate trend
        if len(data) > 14:
            recent_data = data.iloc[-14:, 0]
            x = np.arange(len(recent_data))
            slope, intercept = np.polyfit(x, recent_data.values, 1)
        else:
            slope = 0
        
        # Generate predictions
        predictions = []
        for i in range(config.horizon):
            pred = last_mean + slope * i
            predictions.append(pred)
        
        # Create predictions dataframe
        future_dates = pd.date_range(
            start=data.index[-1] + timedelta(days=1),
            periods=config.horizon,
            freq='D'
        )
        
        predictions_df = pd.DataFrame({
            'forecast': predictions,
            'date': future_dates
        }).set_index('date')
        
        # Simple confidence intervals
        std = data.iloc[:, 0].std()
        z_score = stats.norm.ppf((1 + config.confidence_level) / 2)
        
        confidence_intervals = pd.DataFrame({
            'lower': np.array(predictions) - z_score * std,
            'upper': np.array(predictions) + z_score * std
        }, index=predictions_df.index)
        
        return ForecastResult(
            model_type=ForecastModel.LINEAR_REGRESSION,
            metric=config.metric,
            predictions=predictions_df,
            confidence_intervals=confidence_intervals,
            accuracy_metrics={'method': 'moving_average_with_trend'},
            model_params={'window': window, 'slope': slope}
        )
    
    def _select_arima_params(self, data: pd.DataFrame) -> Tuple[int, int, int]:
        """Select ARIMA parameters automatically"""
        if not STATSMODELS_AVAILABLE:
            return (1, 1, 1)
        
        try:
            # Test for stationarity
            adf_result = adfuller(data.iloc[:, 0])
            d = 0 if adf_result[1] < 0.05 else 1
            
            # Determine p and q using ACF and PACF
            if len(data) > 40:
                acf_values = acf(data.iloc[:, 0].diff().dropna(), nlags=20)
                pacf_values = pacf(data.iloc[:, 0].diff().dropna(), nlags=20)
                
                # Simple heuristic for p and q
                p = np.where(np.abs(pacf_values) < 0.1)[0][0] if np.any(np.abs(pacf_values) < 0.1) else 1
                q = np.where(np.abs(acf_values) < 0.1)[0][0] if np.any(np.abs(acf_values) < 0.1) else 1
                
                p = min(max(p, 1), 3)
                q = min(max(q, 1), 3)
            else:
                p, q = 1, 1
            
            return (p, d, q)
            
        except Exception as e:
            logger.warning(f"Failed to auto-select ARIMA params: {e}")
            return (1, 1, 1)
    
    def _detect_seasonality(self, data: pd.DataFrame) -> int:
        """Detect seasonal period in data"""
        if len(data) < 14:
            return 1
        
        # Simple approach: check for weekly pattern
        if len(data) >= 14:
            return 7
        
        return 1
    
    def _create_time_features(self, data: pd.DataFrame, lags: int = 7) -> Tuple[np.ndarray, np.ndarray]:
        """Create lagged features for ML models"""
        series = data.iloc[:, 0].values
        
        X = []
        y = []
        
        for i in range(lags, len(series)):
            X.append(series[i-lags:i])
            y.append(series[i])
        
        return np.array(X), np.array(y)
    
    def _create_future_features(self, data: pd.DataFrame, horizon: int, lags: int = 7) -> np.ndarray:
        """Create features for future predictions"""
        series = data.iloc[:, 0].values
        
        future_X = []
        last_values = list(series[-lags:])
        
        for _ in range(horizon):
            future_X.append(last_values.copy())
            # Use average of last values as next prediction (simplified)
            next_val = np.mean(last_values)
            last_values.pop(0)
            last_values.append(next_val)
        
        return np.array(future_X)
    
    def _calculate_accuracy_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Calculate forecast accuracy metrics"""
        metrics = {}
        
        # Ensure same length
        min_len = min(len(actual), len(predicted))
        actual = actual[-min_len:]
        predicted = predicted[-min_len:]
        
        # MSE
        metrics['mse'] = mean_squared_error(actual, predicted) if SKLEARN_AVAILABLE else np.mean((actual - predicted) ** 2)
        
        # RMSE
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # MAE
        metrics['mae'] = mean_absolute_error(actual, predicted) if SKLEARN_AVAILABLE else np.mean(np.abs(actual - predicted))
        
        # MAPE
        non_zero = actual != 0
        if np.any(non_zero):
            metrics['mape'] = np.mean(np.abs((actual[non_zero] - predicted[non_zero]) / actual[non_zero])) * 100
        
        # R-squared
        if SKLEARN_AVAILABLE:
            try:
                metrics['r2'] = r2_score(actual, predicted)
            except:
                pass
        
        return metrics
    
    def _generate_cache_key(self, config: ForecastConfig, data: pd.DataFrame) -> str:
        """Generate cache key for forecast"""
        key_parts = [
            config.model_type.value,
            config.metric.value,
            str(config.horizon),
            str(len(data)),
            str(data.iloc[-1, 0])  # Last value
        ]
        return '_'.join(key_parts)
    
    def _get_cached_forecast(self, cache_key: str) -> Optional[ForecastResult]:
        """Get cached forecast if available"""
        if cache_key in self.forecast_cache:
            result, cached_time = self.forecast_cache[cache_key]
            # Cache valid for 1 hour
            if datetime.now() - cached_time < timedelta(hours=1):
                return result
        return None
    
    def _cache_forecast(self, cache_key: str, result: ForecastResult):
        """Cache forecast result"""
        self.forecast_cache[cache_key] = (result, datetime.now())
        
        # Clean old cache entries
        cutoff_time = datetime.now() - timedelta(hours=2)
        self.forecast_cache = {
            k: v for k, v in self.forecast_cache.items()
            if v[1] > cutoff_time
        }
    
    async def compare_models(
        self,
        historical_data: pd.DataFrame,
        metric: ForecastMetric,
        horizon: int = 30,
        models: Optional[List[ForecastModel]] = None
    ) -> Dict[str, Any]:
        """Compare multiple forecasting models"""
        if models is None:
            models = [
                ForecastModel.EXPONENTIAL_SMOOTHING,
                ForecastModel.RANDOM_FOREST,
                ForecastModel.GRADIENT_BOOSTING
            ]
        
        results = {}
        
        for model_type in models:
            try:
                config = ForecastConfig(
                    model_type=model_type,
                    metric=metric,
                    horizon=horizon
                )
                
                result = await self.create_forecast(config, historical_data)
                
                results[model_type.value] = {
                    'predictions': result.predictions['forecast'].tolist(),
                    'accuracy': result.accuracy_metrics,
                    'params': result.model_params
                }
            except Exception as e:
                logger.error(f"Model {model_type} comparison failed: {e}")
                continue
        
        # Determine best model
        best_model = None
        best_score = float('inf')
        
        for model_name, model_result in results.items():
            if 'accuracy' in model_result and 'rmse' in model_result['accuracy']:
                if model_result['accuracy']['rmse'] < best_score:
                    best_score = model_result['accuracy']['rmse']
                    best_model = model_name
        
        return {
            'models': results,
            'best_model': best_model,
            'best_score': best_score,
            'comparison_date': datetime.now().isoformat()
        }
    
    def get_model_recommendations(
        self,
        data_characteristics: Dict[str, Any]
    ) -> List[ForecastModel]:
        """Recommend models based on data characteristics"""
        recommendations = []
        
        data_length = data_characteristics.get('length', 0)
        has_seasonality = data_characteristics.get('seasonality', False)
        has_trend = data_characteristics.get('trend', False)
        is_stationary = data_characteristics.get('stationary', False)
        has_outliers = data_characteristics.get('outliers', False)
        
        # ARIMA for stationary or small datasets
        if is_stationary or data_length < 100:
            recommendations.append(ForecastModel.ARIMA)
        
        # SARIMA for seasonal data
        if has_seasonality and data_length > 50:
            recommendations.append(ForecastModel.SARIMA)
        
        # Prophet for longer time series with seasonality
        if data_length > 365 and (has_seasonality or has_trend):
            recommendations.append(ForecastModel.PROPHET)
        
        # Exponential Smoothing for trend and seasonality
        if has_trend or has_seasonality:
            recommendations.append(ForecastModel.EXPONENTIAL_SMOOTHING)
        
        # ML models for complex patterns
        if data_length > 100:
            recommendations.append(ForecastModel.RANDOM_FOREST)
            if not has_outliers:
                recommendations.append(ForecastModel.GRADIENT_BOOSTING)
        
        # Always include ensemble for best results
        if len(recommendations) > 2:
            recommendations.append(ForecastModel.ENSEMBLE)
        
        return recommendations[:5]  # Return top 5 recommendations


# Singleton instance
advanced_forecasting = AdvancedForecastingModels()