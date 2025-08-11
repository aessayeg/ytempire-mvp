"""
Feature Engineering Pipeline
Transforms raw data into features for ML models
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    name: str
    type: str  # numeric, categorical, text, temporal
    transform: str  # scale, normalize, encode, vectorize
    params: Dict[str, Any]


class FeatureEngineer:
    """Main feature engineering pipeline"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.vectorizers = {}
        self.feature_cache = {}
        
    def create_video_features(self, video_data: Dict[str, Any]) -> pd.DataFrame:
        """Create features for video generation models"""
        features = {}
        
        # Temporal features
        features.update(self._extract_temporal_features(video_data))
        
        # Content features
        features.update(self._extract_content_features(video_data))
        
        # Engagement features
        features.update(self._extract_engagement_features(video_data))
        
        # Trend features
        features.update(self._extract_trend_features(video_data))
        
        return pd.DataFrame([features])
    
    def create_channel_features(self, channel_data: Dict[str, Any]) -> pd.DataFrame:
        """Create features for channel analysis"""
        features = {}
        
        # Performance metrics
        features['subscriber_count'] = channel_data.get('subscribers', 0)
        features['video_count'] = channel_data.get('video_count', 0)
        features['avg_views'] = channel_data.get('avg_views', 0)
        features['growth_rate'] = self._calculate_growth_rate(channel_data)
        
        # Health indicators
        features['posting_frequency'] = channel_data.get('posts_per_week', 0)
        features['engagement_rate'] = channel_data.get('engagement_rate', 0)
        features['monetization_enabled'] = int(channel_data.get('monetized', False))
        
        return pd.DataFrame([features])
    
    def create_trend_features(self, trend_data: Dict[str, Any]) -> pd.DataFrame:
        """Create features for trend prediction"""
        features = {}
        
        # Trend metrics
        features['search_volume'] = trend_data.get('search_volume', 0)
        features['growth_velocity'] = trend_data.get('growth_rate', 0)
        features['competition_level'] = trend_data.get('competition', 0)
        features['seasonal_factor'] = self._get_seasonal_factor(trend_data)
        
        # Topic features
        if 'keywords' in trend_data:
            keyword_features = self._vectorize_keywords(trend_data['keywords'])
            features.update(keyword_features)
        
        return pd.DataFrame([features])
    
    def _extract_temporal_features(self, data: Dict) -> Dict[str, float]:
        """Extract time-based features"""
        features = {}
        current_time = datetime.utcnow()
        
        # Time of day
        features['hour_of_day'] = current_time.hour
        features['day_of_week'] = current_time.weekday()
        features['is_weekend'] = int(current_time.weekday() >= 5)
        
        # Optimal posting time
        optimal_hours = [9, 12, 15, 18, 21]  # Peak engagement hours
        features['is_optimal_hour'] = int(current_time.hour in optimal_hours)
        
        return features
    
    def _extract_content_features(self, data: Dict) -> Dict[str, float]:
        """Extract content-related features"""
        features = {}
        
        # Title features
        title = data.get('title', '')
        features['title_length'] = len(title)
        features['title_word_count'] = len(title.split())
        features['has_number'] = int(any(char.isdigit() for char in title))
        features['has_question'] = int('?' in title)
        
        # Description features
        description = data.get('description', '')
        features['desc_length'] = len(description)
        features['desc_word_count'] = len(description.split())
        
        # Tags
        tags = data.get('tags', [])
        features['tag_count'] = len(tags)
        
        return features
    
    def _extract_engagement_features(self, data: Dict) -> Dict[str, float]:
        """Extract engagement prediction features"""
        features = {}
        
        # Historical performance
        features['avg_ctr'] = data.get('historical_ctr', 0.05)
        features['avg_watch_time'] = data.get('avg_watch_time', 300)
        features['avg_retention'] = data.get('avg_retention', 0.5)
        
        # Content quality scores
        features['thumbnail_score'] = data.get('thumbnail_quality', 0.7)
        features['script_score'] = data.get('script_quality', 0.7)
        
        return features
    
    def _extract_trend_features(self, data: Dict) -> Dict[str, float]:
        """Extract trend-related features"""
        features = {}
        
        features['trend_score'] = data.get('trend_alignment', 0.5)
        features['competition_index'] = data.get('competition', 0.5)
        features['virality_potential'] = data.get('virality_score', 0.5)
        
        return features
    
    def _calculate_growth_rate(self, data: Dict) -> float:
        """Calculate channel growth rate"""
        current = data.get('subscribers', 0)
        previous = data.get('subscribers_30d_ago', current)
        
        if previous == 0:
            return 0.0
        
        return (current - previous) / previous
    
    def _get_seasonal_factor(self, data: Dict) -> float:
        """Calculate seasonal adjustment factor"""
        month = datetime.utcnow().month
        
        # Seasonal weights (example)
        seasonal_weights = {
            1: 0.9,   # January
            2: 0.85,  # February
            3: 0.9,   # March
            4: 0.95,  # April
            5: 1.0,   # May
            6: 1.0,   # June
            7: 0.95,  # July
            8: 0.9,   # August
            9: 1.05,  # September
            10: 1.1,  # October
            11: 1.15, # November
            12: 1.2   # December
        }
        
        return seasonal_weights.get(month, 1.0)
    
    def _vectorize_keywords(self, keywords: List[str]) -> Dict[str, float]:
        """Convert keywords to features"""
        features = {}
        
        # Simple keyword presence features
        important_keywords = ['tutorial', 'review', 'best', 'how to', 'guide']
        
        for keyword in important_keywords:
            features[f'has_{keyword.replace(" ", "_")}'] = int(
                any(keyword in kw.lower() for kw in keywords)
            )
        
        return features
    
    def transform_features(
        self,
        df: pd.DataFrame,
        feature_configs: List[FeatureConfig]
    ) -> pd.DataFrame:
        """Apply transformations to features"""
        transformed_df = df.copy()
        
        for config in feature_configs:
            if config.type == 'numeric':
                transformed_df = self._transform_numeric(
                    transformed_df, config.name, config.transform, config.params
                )
            elif config.type == 'categorical':
                transformed_df = self._transform_categorical(
                    transformed_df, config.name, config.transform, config.params
                )
            elif config.type == 'text':
                transformed_df = self._transform_text(
                    transformed_df, config.name, config.transform, config.params
                )
        
        return transformed_df
    
    def _transform_numeric(
        self,
        df: pd.DataFrame,
        column: str,
        transform: str,
        params: Dict
    ) -> pd.DataFrame:
        """Transform numeric features"""
        if column not in df.columns:
            return df
        
        if transform == 'scale':
            if column not in self.scalers:
                self.scalers[column] = StandardScaler()
                df[column] = self.scalers[column].fit_transform(df[[column]])
            else:
                df[column] = self.scalers[column].transform(df[[column]])
        
        elif transform == 'normalize':
            if column not in self.scalers:
                self.scalers[column] = MinMaxScaler()
                df[column] = self.scalers[column].fit_transform(df[[column]])
            else:
                df[column] = self.scalers[column].transform(df[[column]])
        
        elif transform == 'log':
            df[column] = np.log1p(df[column])
        
        return df
    
    def _transform_categorical(
        self,
        df: pd.DataFrame,
        column: str,
        transform: str,
        params: Dict
    ) -> pd.DataFrame:
        """Transform categorical features"""
        if column not in df.columns:
            return df
        
        if transform == 'encode':
            if column not in self.encoders:
                self.encoders[column] = LabelEncoder()
                df[column] = self.encoders[column].fit_transform(df[column])
            else:
                df[column] = self.encoders[column].transform(df[column])
        
        elif transform == 'onehot':
            dummies = pd.get_dummies(df[column], prefix=column)
            df = pd.concat([df.drop(column, axis=1), dummies], axis=1)
        
        return df
    
    def _transform_text(
        self,
        df: pd.DataFrame,
        column: str,
        transform: str,
        params: Dict
    ) -> pd.DataFrame:
        """Transform text features"""
        if column not in df.columns:
            return df
        
        if transform == 'tfidf':
            if column not in self.vectorizers:
                self.vectorizers[column] = TfidfVectorizer(**params)
                vectors = self.vectorizers[column].fit_transform(df[column])
            else:
                vectors = self.vectorizers[column].transform(df[column])
            
            # Add vectorized features
            feature_names = [f"{column}_tfidf_{i}" for i in range(vectors.shape[1])]
            vector_df = pd.DataFrame(vectors.toarray(), columns=feature_names, index=df.index)
            df = pd.concat([df.drop(column, axis=1), vector_df], axis=1)
        
        return df
    
    def create_interaction_features(
        self,
        df: pd.DataFrame,
        interactions: List[Tuple[str, str]]
    ) -> pd.DataFrame:
        """Create interaction features between columns"""
        for col1, col2 in interactions:
            if col1 in df.columns and col2 in df.columns:
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        
        return df
    
    def create_aggregate_features(
        self,
        df: pd.DataFrame,
        group_by: str,
        agg_columns: List[str],
        agg_funcs: List[str]
    ) -> pd.DataFrame:
        """Create aggregate features"""
        for col in agg_columns:
            for func in agg_funcs:
                agg_name = f'{col}_{func}_by_{group_by}'
                df[agg_name] = df.groupby(group_by)[col].transform(func)
        
        return df


class FeaturePipeline:
    """End-to-end feature engineering pipeline"""
    
    def __init__(self):
        self.engineer = FeatureEngineer()
        self.feature_configs = self._load_feature_configs()
    
    def _load_feature_configs(self) -> List[FeatureConfig]:
        """Load feature configuration"""
        return [
            FeatureConfig('subscriber_count', 'numeric', 'log', {}),
            FeatureConfig('video_count', 'numeric', 'scale', {}),
            FeatureConfig('avg_views', 'numeric', 'log', {}),
            FeatureConfig('growth_rate', 'numeric', 'normalize', {}),
            FeatureConfig('engagement_rate', 'numeric', 'normalize', {}),
            FeatureConfig('search_volume', 'numeric', 'log', {}),
            FeatureConfig('competition_level', 'numeric', 'normalize', {}),
        ]
    
    def process(self, data: Dict[str, Any], feature_type: str) -> pd.DataFrame:
        """Process data through feature pipeline"""
        # Create base features
        if feature_type == 'video':
            df = self.engineer.create_video_features(data)
        elif feature_type == 'channel':
            df = self.engineer.create_channel_features(data)
        elif feature_type == 'trend':
            df = self.engineer.create_trend_features(data)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        # Apply transformations
        df = self.engineer.transform_features(df, self.feature_configs)
        
        # Create interaction features
        interactions = [
            ('subscriber_count', 'engagement_rate'),
            ('video_count', 'avg_views'),
            ('trend_score', 'competition_index')
        ]
        df = self.engineer.create_interaction_features(df, interactions)
        
        return df


# Global pipeline instance
feature_pipeline = FeaturePipeline()