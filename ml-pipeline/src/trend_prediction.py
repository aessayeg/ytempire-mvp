"""
Trend Prediction System using Prophet for YTEmpire
Predicts video performance and trending topics
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import logging
import json
import pickle
from dataclasses import dataclass
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrendPrediction:
    """Container for trend predictions"""
    topic: str
    predicted_views: float
    confidence_lower: float
    confidence_upper: float
    trend_direction: str  # 'rising', 'stable', 'declining'
    trend_strength: float  # 0-1 score
    optimal_publish_time: datetime
    metadata: Dict[str, Any]


class TrendPredictor:
    """Prophet-based trend prediction system"""
    
    def __init__(self, db_session: Optional[AsyncSession] = None):
        self.db_session = db_session
        self.models = {}
        self.scaler = StandardScaler()
        
    async def train_channel_model(self, channel_id: str, historical_data: pd.DataFrame) -> Prophet:
        """
        Train a Prophet model for a specific channel
        
        Args:
            channel_id: Channel identifier
            historical_data: DataFrame with columns ['ds', 'y'] for time and views
        
        Returns:
            Trained Prophet model
        """
        # Prepare data
        df = historical_data[['ds', 'y']].copy()
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Add additional regressors if available
        if 'day_of_week' in historical_data.columns:
            df['day_of_week'] = historical_data['day_of_week']
        if 'hour_of_day' in historical_data.columns:
            df['hour_of_day'] = historical_data['hour_of_day']
        
        # Initialize Prophet with custom parameters
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=0.05,  # Flexibility of trend
            seasonality_prior_scale=10.0,  # Strength of seasonality
            interval_width=0.95  # 95% confidence interval
        )
        
        # Add custom seasonalities
        model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=5
        )
        
        # Add country holidays (US by default)
        model.add_country_holidays(country_name='US')
        
        # Add regressors if present
        if 'day_of_week' in df.columns:
            model.add_regressor('day_of_week')
        if 'hour_of_day' in df.columns:
            model.add_regressor('hour_of_day')
        
        # Fit model
        model.fit(df)
        
        # Store model
        self.models[channel_id] = model
        
        # Perform cross-validation
        cv_results = cross_validation(
            model,
            initial='30 days',
            period='7 days',
            horizon='14 days'
        )
        
        # Calculate performance metrics
        metrics = performance_metrics(cv_results)
        logger.info(f"Model trained for channel {channel_id}")
        logger.info(f"MAPE: {metrics['mape'].mean():.2%}")
        logger.info(f"RMSE: {metrics['rmse'].mean():.2f}")
        
        return model
    
    async def predict_topic_trend(
        self,
        topic: str,
        channel_id: str,
        days_ahead: int = 30
    ) -> TrendPrediction:
        """
        Predict trend for a specific topic
        
        Args:
            topic: Topic to predict
            channel_id: Channel context
            days_ahead: Number of days to predict
        
        Returns:
            TrendPrediction object
        """
        # Get or train model for channel
        if channel_id not in self.models:
            # Load historical data from database
            historical_data = await self._load_channel_history(channel_id)
            await self.train_channel_model(channel_id, historical_data)
        
        model = self.models[channel_id]
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=days_ahead)
        
        # Add regressors for future dates
        future['day_of_week'] = future['ds'].dt.dayofweek
        future['hour_of_day'] = future['ds'].dt.hour
        
        # Make prediction
        forecast = model.predict(future)
        
        # Get last N days of forecast
        future_forecast = forecast.tail(days_ahead)
        
        # Calculate trend metrics
        predicted_views = future_forecast['yhat'].mean()
        confidence_lower = future_forecast['yhat_lower'].mean()
        confidence_upper = future_forecast['yhat_upper'].mean()
        
        # Determine trend direction
        trend_slope = np.polyfit(range(len(future_forecast)), future_forecast['yhat'].values, 1)[0]
        
        if trend_slope > 0.1:
            trend_direction = 'rising'
        elif trend_slope < -0.1:
            trend_direction = 'declining'
        else:
            trend_direction = 'stable'
        
        # Calculate trend strength (0-1)
        trend_strength = min(abs(trend_slope) / future_forecast['yhat'].mean(), 1.0)
        
        # Find optimal publish time
        best_day = future_forecast.loc[future_forecast['yhat'].idxmax()]
        optimal_time = best_day['ds'].to_pydatetime()
        
        # Add topic-specific adjustments
        topic_multiplier = await self._get_topic_multiplier(topic)
        predicted_views *= topic_multiplier
        confidence_lower *= topic_multiplier
        confidence_upper *= topic_multiplier
        
        return TrendPrediction(
            topic=topic,
            predicted_views=predicted_views,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            optimal_publish_time=optimal_time,
            metadata={
                'channel_id': channel_id,
                'forecast_days': days_ahead,
                'model_components': model.component_modes
            }
        )
    
    async def predict_viral_probability(
        self,
        video_features: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Predict probability of video going viral
        
        Args:
            video_features: Video characteristics
        
        Returns:
            Viral probability and feature importance
        """
        # Extract features
        features = np.array([
            video_features.get('title_length', 50),
            video_features.get('description_length', 200),
            video_features.get('tags_count', 10),
            video_features.get('thumbnail_quality_score', 0.5),
            video_features.get('topic_trending_score', 0.5),
            video_features.get('channel_subscriber_count', 1000),
            video_features.get('upload_hour', 14),
            video_features.get('duration_seconds', 600),
        ])
        
        # Normalize features
        features_normalized = self.scaler.fit_transform(features.reshape(1, -1))
        
        # Simple viral probability model (would be ML model in production)
        weights = np.array([0.1, 0.05, 0.1, 0.2, 0.3, 0.1, 0.05, 0.1])
        viral_score = np.dot(features_normalized[0], weights)
        
        # Convert to probability
        viral_probability = 1 / (1 + np.exp(-viral_score))
        
        # Calculate feature importance
        feature_importance = {
            'title_length': abs(weights[0] * features_normalized[0][0]),
            'description_length': abs(weights[1] * features_normalized[0][1]),
            'tags_count': abs(weights[2] * features_normalized[0][2]),
            'thumbnail_quality': abs(weights[3] * features_normalized[0][3]),
            'topic_trending': abs(weights[4] * features_normalized[0][4]),
            'channel_size': abs(weights[5] * features_normalized[0][5]),
            'upload_time': abs(weights[6] * features_normalized[0][6]),
            'video_duration': abs(weights[7] * features_normalized[0][7]),
        }
        
        return viral_probability, feature_importance
    
    async def identify_trending_topics(
        self,
        category: str,
        region: str = 'US',
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Identify currently trending topics
        
        Args:
            category: Video category
            region: Geographic region
            limit: Number of topics to return
        
        Returns:
            List of trending topics with scores
        """
        # In production, this would fetch from YouTube Trends API
        # For now, return mock trending topics
        trending_topics = [
            {'topic': 'AI Tools 2024', 'score': 0.95, 'growth_rate': 0.25},
            {'topic': 'Productivity Hacks', 'score': 0.88, 'growth_rate': 0.15},
            {'topic': 'Remote Work Setup', 'score': 0.82, 'growth_rate': 0.10},
            {'topic': 'ChatGPT Tips', 'score': 0.79, 'growth_rate': 0.20},
            {'topic': 'Python Tutorial', 'score': 0.75, 'growth_rate': 0.08},
            {'topic': 'Web3 Development', 'score': 0.72, 'growth_rate': 0.12},
            {'topic': 'Machine Learning', 'score': 0.70, 'growth_rate': 0.05},
            {'topic': 'Content Creation', 'score': 0.68, 'growth_rate': 0.18},
            {'topic': 'Startup Ideas', 'score': 0.65, 'growth_rate': 0.22},
            {'topic': 'Investing Tips', 'score': 0.63, 'growth_rate': 0.07},
        ]
        
        # Filter by category if needed
        if category:
            # Apply category-specific filtering logic
            pass
        
        # Sort by score and limit
        trending_topics.sort(key=lambda x: x['score'], reverse=True)
        
        return trending_topics[:limit]
    
    async def generate_content_calendar(
        self,
        channel_id: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Generate optimal content calendar based on predictions
        
        Args:
            channel_id: Channel identifier
            days: Number of days to plan
        
        Returns:
            Content calendar with recommended publish times and topics
        """
        calendar = []
        
        # Get trending topics
        trending = await self.identify_trending_topics('technology')
        
        # Generate calendar entries
        current_date = datetime.now()
        for day in range(days):
            date = current_date + timedelta(days=day)
            
            # Determine optimal posting time for this day
            day_of_week = date.weekday()
            
            # Best times by day of week (based on typical patterns)
            optimal_hours = {
                0: 14,  # Monday 2 PM
                1: 10,  # Tuesday 10 AM
                2: 15,  # Wednesday 3 PM
                3: 17,  # Thursday 5 PM
                4: 12,  # Friday 12 PM
                5: 10,  # Saturday 10 AM
                6: 20,  # Sunday 8 PM
            }
            
            optimal_hour = optimal_hours.get(day_of_week, 14)
            publish_time = date.replace(hour=optimal_hour, minute=0, second=0)
            
            # Select topic for this day
            topic_index = day % len(trending)
            topic = trending[topic_index]
            
            # Predict performance
            prediction = await self.predict_topic_trend(
                topic['topic'],
                channel_id,
                1
            )
            
            calendar.append({
                'date': publish_time.isoformat(),
                'topic': topic['topic'],
                'predicted_views': prediction.predicted_views,
                'confidence_range': [
                    prediction.confidence_lower,
                    prediction.confidence_upper
                ],
                'trend_direction': prediction.trend_direction,
                'priority': topic['score'],
            })
        
        return calendar
    
    def visualize_forecast(
        self,
        forecast: pd.DataFrame,
        title: str = "View Count Forecast"
    ) -> go.Figure:
        """
        Create interactive visualization of forecast
        
        Args:
            forecast: Prophet forecast DataFrame
            title: Chart title
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add actual data if present
        if 'y' in forecast.columns:
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['y'],
                mode='markers',
                name='Actual',
                marker=dict(color='blue', size=6)
            ))
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='red', width=2)
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
            y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence',
            showlegend=True
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="View Count",
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    async def _load_channel_history(self, channel_id: str) -> pd.DataFrame:
        """Load historical data for a channel"""
        # In production, load from database
        # For now, generate synthetic data
        dates = pd.date_range(end=datetime.now(), periods=365, freq='D')
        
        # Generate realistic view counts with seasonality
        base_views = 10000
        trend = np.linspace(0, 5000, len(dates))
        weekly_season = 2000 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
        noise = np.random.normal(0, 1000, len(dates))
        
        views = base_views + trend + weekly_season + noise
        views = np.maximum(views, 0)  # Ensure non-negative
        
        df = pd.DataFrame({
            'ds': dates,
            'y': views,
            'day_of_week': dates.dayofweek,
            'hour_of_day': 14  # Assume afternoon posting
        })
        
        return df
    
    async def _get_topic_multiplier(self, topic: str) -> float:
        """Get performance multiplier for a topic"""
        # Topic-specific performance adjustments
        topic_multipliers = {
            'AI Tools': 1.5,
            'ChatGPT': 1.4,
            'Productivity': 1.2,
            'Tutorial': 1.1,
            'Review': 1.0,
        }
        
        # Check if any keyword matches
        for keyword, multiplier in topic_multipliers.items():
            if keyword.lower() in topic.lower():
                return multiplier
        
        return 1.0
    
    def save_model(self, channel_id: str, filepath: str):
        """Save trained model to disk"""
        if channel_id in self.models:
            with open(filepath, 'wb') as f:
                pickle.dump(self.models[channel_id], f)
            logger.info(f"Model saved for channel {channel_id}")
    
    def load_model(self, channel_id: str, filepath: str):
        """Load trained model from disk"""
        with open(filepath, 'rb') as f:
            self.models[channel_id] = pickle.load(f)
        logger.info(f"Model loaded for channel {channel_id}")


# Example usage
async def main():
    """Example usage of trend prediction system"""
    predictor = TrendPredictor()
    
    # Identify trending topics
    trending = await predictor.identify_trending_topics('technology')
    print("Trending Topics:")
    for topic in trending[:5]:
        print(f"  - {topic['topic']}: Score {topic['score']:.2f}")
    
    # Predict trend for a topic
    prediction = await predictor.predict_topic_trend(
        "AI Tools 2024",
        "channel_123",
        days_ahead=30
    )
    print(f"\nTrend Prediction for 'AI Tools 2024':")
    print(f"  Predicted Views: {prediction.predicted_views:,.0f}")
    print(f"  Confidence Range: {prediction.confidence_lower:,.0f} - {prediction.confidence_upper:,.0f}")
    print(f"  Trend Direction: {prediction.trend_direction}")
    print(f"  Optimal Publish Time: {prediction.optimal_publish_time}")
    
    # Test viral probability
    video_features = {
        'title_length': 45,
        'description_length': 250,
        'tags_count': 15,
        'thumbnail_quality_score': 0.8,
        'topic_trending_score': 0.9,
        'channel_subscriber_count': 5000,
        'upload_hour': 14,
        'duration_seconds': 480,
    }
    
    viral_prob, importance = await predictor.predict_viral_probability(video_features)
    print(f"\nViral Probability: {viral_prob:.2%}")
    print("Feature Importance:")
    for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {feature}: {score:.3f}")


if __name__ == "__main__":
    asyncio.run(main())