"""
Feature Engineering Pipeline for YTEmpire
Extracts and transforms features for ML models
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from textblob import TextBlob
import re
from collections import Counter
import asyncio
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract features from raw data"""
    
    def __init__(self):
        self.text_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    async def extract_video_features(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from video metadata"""
        features = {}
        
        # Basic metadata features
        features['title_length'] = len(video_data.get('title', ''))
        features['title_word_count'] = len(video_data.get('title', '').split())
        features['description_length'] = len(video_data.get('description', ''))
        features['description_word_count'] = len(video_data.get('description', '').split())
        
        # Title features
        title = video_data.get('title', '')
        features['has_numbers'] = int(bool(re.search(r'\d', title)))
        features['has_capitals'] = int(sum(1 for c in title if c.isupper()) > len(title) * 0.3)
        features['has_emoji'] = int(bool(re.search(r'[^\w\s,]', title)))
        features['has_question'] = int('?' in title)
        features['has_exclamation'] = int('!' in title)
        
        # Engagement bait keywords
        clickbait_words = ['shocking', 'amazing', 'unbelievable', 'secret', 'hack', 
                          'trick', 'easy', 'simple', 'fast', 'free', 'best', 'top']
        features['clickbait_score'] = sum(1 for word in clickbait_words if word.lower() in title.lower())
        
        # Tags features
        tags = video_data.get('tags', [])
        features['tag_count'] = len(tags)
        features['avg_tag_length'] = np.mean([len(tag) for tag in tags]) if tags else 0
        
        # Temporal features
        if 'scheduled_date' in video_data:
            scheduled = pd.to_datetime(video_data['scheduled_date'])
            features['publish_hour'] = scheduled.hour
            features['publish_day'] = scheduled.dayofweek
            features['publish_month'] = scheduled.month
            features['is_weekend'] = int(scheduled.dayofweek >= 5)
            features['is_prime_time'] = int(18 <= scheduled.hour <= 22)
        
        # Duration features
        duration = video_data.get('duration_seconds', 0)
        features['duration_seconds'] = duration
        features['duration_category'] = self._categorize_duration(duration)
        features['is_short'] = int(duration <= 60)
        features['is_long'] = int(duration >= 600)
        
        # Category features
        category = video_data.get('category', 'unknown')
        features['category_encoded'] = self._encode_category(category)
        
        # Language features
        features['language'] = video_data.get('language', 'en')
        features['is_english'] = int(features['language'] == 'en')
        
        # Sentiment features
        sentiment_features = await self._extract_sentiment_features(
            video_data.get('title', ''),
            video_data.get('description', '')
        )
        features.update(sentiment_features)
        
        return features
    
    async def extract_channel_features(self, channel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from channel data"""
        features = {}
        
        # Channel metrics
        features['subscriber_count'] = channel_data.get('subscriber_count', 0)
        features['video_count'] = channel_data.get('video_count', 0)
        features['total_views'] = channel_data.get('view_count', 0)
        
        # Derived metrics
        if features['video_count'] > 0:
            features['avg_views_per_video'] = features['total_views'] / features['video_count']
        else:
            features['avg_views_per_video'] = 0
            
        if features['subscriber_count'] > 0:
            features['engagement_ratio'] = features['total_views'] / features['subscriber_count']
        else:
            features['engagement_ratio'] = 0
        
        # Channel age
        created_date = pd.to_datetime(channel_data.get('created_at', datetime.now()))
        features['channel_age_days'] = (datetime.now() - created_date).days
        
        # Monetization
        features['is_monetized'] = int(channel_data.get('is_monetized', False))
        
        # Upload frequency
        features['uploads_per_week'] = self._calculate_upload_frequency(channel_data)
        
        # Channel category
        category = channel_data.get('category', 'unknown')
        features['channel_category'] = self._encode_category(category)
        
        # Growth metrics
        growth_features = await self._extract_growth_features(channel_data)
        features.update(growth_features)
        
        return features
    
    async def extract_trend_features(self, trend_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract features from trend data"""
        features = {}
        
        if not trend_data:
            return features
        
        # Aggregate trend metrics
        df = pd.DataFrame(trend_data)
        
        # Time series features
        if 'search_volume' in df.columns:
            features['trend_mean'] = df['search_volume'].mean()
            features['trend_std'] = df['search_volume'].std()
            features['trend_min'] = df['search_volume'].min()
            features['trend_max'] = df['search_volume'].max()
            features['trend_range'] = features['trend_max'] - features['trend_min']
            
            # Trend direction
            if len(df) > 1:
                features['trend_slope'] = np.polyfit(range(len(df)), df['search_volume'], 1)[0]
                features['is_trending_up'] = int(features['trend_slope'] > 0)
            else:
                features['trend_slope'] = 0
                features['is_trending_up'] = 0
        
        # Seasonality features
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['month'] = df['date'].dt.month
            monthly_avg = df.groupby('month')['search_volume'].mean()
            features['seasonal_variance'] = monthly_avg.var()
        
        # Competition features
        if 'competition' in df.columns:
            features['avg_competition'] = df['competition'].mean()
            features['max_competition'] = df['competition'].max()
        
        return features
    
    async def extract_cost_features(self, cost_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract features from cost data"""
        features = {}
        
        if not cost_data:
            return features
        
        df = pd.DataFrame(cost_data)
        
        # Cost aggregations
        features['total_cost'] = df['amount'].sum()
        features['avg_cost'] = df['amount'].mean()
        features['cost_std'] = df['amount'].std()
        
        # Cost by service
        if 'service_type' in df.columns:
            service_costs = df.groupby('service_type')['amount'].sum()
            for service, cost in service_costs.items():
                features[f'cost_{service}'] = cost
        
        # Cost trend
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
            df = df.sort_values('created_at')
            daily_costs = df.groupby(df['created_at'].dt.date)['amount'].sum()
            
            if len(daily_costs) > 1:
                features['cost_trend'] = np.polyfit(range(len(daily_costs)), daily_costs.values, 1)[0]
                features['cost_increasing'] = int(features['cost_trend'] > 0)
            else:
                features['cost_trend'] = 0
                features['cost_increasing'] = 0
        
        # API usage features
        if 'tokens_used' in df.columns:
            features['total_tokens'] = df['tokens_used'].apply(
                lambda x: x.get('input', 0) + x.get('output', 0) if isinstance(x, dict) else 0
            ).sum()
        
        return features
    
    async def extract_performance_features(self, analytics_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract features from performance analytics"""
        features = {}
        
        if not analytics_data:
            return features
        
        df = pd.DataFrame(analytics_data)
        
        # View metrics
        if 'views' in df.columns:
            features['total_views'] = df['views'].sum()
            features['avg_daily_views'] = df['views'].mean()
            features['peak_views'] = df['views'].max()
            features['views_std'] = df['views'].std()
            
            # View velocity
            if len(df) > 1:
                features['view_acceleration'] = df['views'].diff().mean()
        
        # Engagement metrics
        if 'likes' in df.columns and 'views' in df.columns:
            features['like_rate'] = (df['likes'].sum() / df['views'].sum()) if df['views'].sum() > 0 else 0
        
        if 'comments' in df.columns and 'views' in df.columns:
            features['comment_rate'] = (df['comments'].sum() / df['views'].sum()) if df['views'].sum() > 0 else 0
        
        # Watch time
        if 'watch_time_minutes' in df.columns:
            features['total_watch_time'] = df['watch_time_minutes'].sum()
            features['avg_watch_time'] = df['watch_time_minutes'].mean()
        
        # Revenue metrics
        if 'estimated_revenue' in df.columns:
            features['total_revenue'] = df['estimated_revenue'].sum()
            features['avg_daily_revenue'] = df['estimated_revenue'].mean()
            features['revenue_per_view'] = features['total_revenue'] / features['total_views'] if features.get('total_views', 0) > 0 else 0
        
        return features
    
    async def _extract_sentiment_features(self, title: str, description: str) -> Dict[str, float]:
        """Extract sentiment features from text"""
        features = {}
        
        # Title sentiment
        title_blob = TextBlob(title)
        features['title_polarity'] = title_blob.sentiment.polarity
        features['title_subjectivity'] = title_blob.sentiment.subjectivity
        
        # Description sentiment
        if description:
            desc_blob = TextBlob(description)
            features['desc_polarity'] = desc_blob.sentiment.polarity
            features['desc_subjectivity'] = desc_blob.sentiment.subjectivity
        else:
            features['desc_polarity'] = 0
            features['desc_subjectivity'] = 0
        
        # Combined sentiment
        features['overall_sentiment'] = (features['title_polarity'] + features['desc_polarity']) / 2
        
        return features
    
    async def _extract_growth_features(self, channel_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract growth-related features"""
        features = {}
        
        # Growth rate calculation would require historical data
        # For now, we'll use proxy metrics
        
        subscriber_count = channel_data.get('subscriber_count', 0)
        video_count = channel_data.get('video_count', 0)
        channel_age_days = (datetime.now() - pd.to_datetime(channel_data.get('created_at', datetime.now()))).days
        
        if channel_age_days > 0:
            features['subscriber_growth_rate'] = subscriber_count / channel_age_days
            features['video_production_rate'] = video_count / channel_age_days
        else:
            features['subscriber_growth_rate'] = 0
            features['video_production_rate'] = 0
        
        # Channel tier
        if subscriber_count < 1000:
            features['channel_tier'] = 0  # Starter
        elif subscriber_count < 10000:
            features['channel_tier'] = 1  # Growing
        elif subscriber_count < 100000:
            features['channel_tier'] = 2  # Established
        elif subscriber_count < 1000000:
            features['channel_tier'] = 3  # Popular
        else:
            features['channel_tier'] = 4  # Mega
        
        return features
    
    def _categorize_duration(self, seconds: int) -> int:
        """Categorize video duration"""
        if seconds <= 60:
            return 0  # Shorts
        elif seconds <= 300:
            return 1  # Short
        elif seconds <= 600:
            return 2  # Medium
        elif seconds <= 1200:
            return 3  # Long
        else:
            return 4  # Very long
    
    def _encode_category(self, category: str) -> int:
        """Encode category to numerical value"""
        if 'category' not in self.label_encoders:
            self.label_encoders['category'] = LabelEncoder()
            # Pre-fit with known categories
            known_categories = [
                'Technology', 'Gaming', 'Education', 'Entertainment',
                'Music', 'Sports', 'News', 'Comedy', 'Science', 
                'Travel', 'Food', 'Fashion', 'Fitness', 'Other'
            ]
            self.label_encoders['category'].fit(known_categories)
        
        try:
            return self.label_encoders['category'].transform([category])[0]
        except:
            return self.label_encoders['category'].transform(['Other'])[0]
    
    def _calculate_upload_frequency(self, channel_data: Dict[str, Any]) -> float:
        """Calculate upload frequency per week"""
        video_count = channel_data.get('video_count', 0)
        channel_age_days = (datetime.now() - pd.to_datetime(channel_data.get('created_at', datetime.now()))).days
        
        if channel_age_days > 0:
            weeks = channel_age_days / 7
            return video_count / weeks if weeks > 0 else 0
        return 0


class FeatureTransformer:
    """Transform features for ML models"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        
    def fit(self, features_df: pd.DataFrame):
        """Fit transformers on training data"""
        self.feature_columns = features_df.columns.tolist()
        
        # Identify numerical and categorical columns
        numerical_cols = features_df.select_dtypes(include=[np.number]).columns
        categorical_cols = features_df.select_dtypes(include=['object']).columns
        
        # Fit scalers for numerical features
        if len(numerical_cols) > 0:
            self.scalers['standard'] = StandardScaler()
            self.scalers['standard'].fit(features_df[numerical_cols])
            
            self.scalers['minmax'] = MinMaxScaler()
            self.scalers['minmax'].fit(features_df[numerical_cols])
        
        # Fit encoders for categorical features
        for col in categorical_cols:
            self.encoders[col] = LabelEncoder()
            self.encoders[col].fit(features_df[col])
    
    def transform(self, features_df: pd.DataFrame, scaler_type: str = 'standard') -> np.ndarray:
        """Transform features"""
        df = features_df.copy()
        
        # Encode categorical features
        for col, encoder in self.encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col])
        
        # Scale numerical features
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0 and scaler_type in self.scalers:
            df[numerical_cols] = self.scalers[scaler_type].transform(df[numerical_cols])
        
        return df.values
    
    def fit_transform(self, features_df: pd.DataFrame, scaler_type: str = 'standard') -> np.ndarray:
        """Fit and transform features"""
        self.fit(features_df)
        return self.transform(features_df, scaler_type)
    
    def inverse_transform(self, transformed_data: np.ndarray, scaler_type: str = 'standard') -> pd.DataFrame:
        """Inverse transform features"""
        df = pd.DataFrame(transformed_data, columns=self.feature_columns)
        
        # Inverse scale numerical features
        numerical_cols = [col for col in self.feature_columns if col not in self.encoders]
        if len(numerical_cols) > 0 and scaler_type in self.scalers:
            df[numerical_cols] = self.scalers[scaler_type].inverse_transform(df[numerical_cols])
        
        # Inverse encode categorical features
        for col, encoder in self.encoders.items():
            if col in df.columns:
                df[col] = encoder.inverse_transform(df[col].astype(int))
        
        return df


class FeatureStore:
    """Store and retrieve features"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.cache = {}
        
    async def save_features(self, entity_id: str, entity_type: str, features: Dict[str, Any]):
        """Save features to store"""
        key = f"features:{entity_type}:{entity_id}"
        
        # Add metadata
        features['_entity_id'] = entity_id
        features['_entity_type'] = entity_type
        features['_timestamp'] = datetime.utcnow().isoformat()
        features['_version'] = self._calculate_version(features)
        
        # Save to cache
        self.cache[key] = features
        
        # Save to Redis if available
        if self.redis_client:
            await self.redis_client.set(
                key,
                json.dumps(features, default=str),
                ex=86400  # 24 hour expiry
            )
    
    async def get_features(self, entity_id: str, entity_type: str) -> Optional[Dict[str, Any]]:
        """Retrieve features from store"""
        key = f"features:{entity_type}:{entity_id}"
        
        # Check cache first
        if key in self.cache:
            return self.cache[key]
        
        # Check Redis
        if self.redis_client:
            data = await self.redis_client.get(key)
            if data:
                features = json.loads(data)
                self.cache[key] = features
                return features
        
        return None
    
    async def get_feature_batch(self, entity_ids: List[str], entity_type: str) -> pd.DataFrame:
        """Retrieve batch of features"""
        features_list = []
        
        for entity_id in entity_ids:
            features = await self.get_features(entity_id, entity_type)
            if features:
                features_list.append(features)
        
        if features_list:
            return pd.DataFrame(features_list)
        return pd.DataFrame()
    
    def _calculate_version(self, features: Dict[str, Any]) -> str:
        """Calculate version hash for features"""
        feature_str = json.dumps(features, sort_keys=True, default=str)
        return hashlib.md5(feature_str.encode()).hexdigest()[:8]


# Global instances
feature_extractor = FeatureExtractor()
feature_transformer = FeatureTransformer()
feature_store = FeatureStore()