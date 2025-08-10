"""
Feature Engineering for YTEmpire ML Models
Owner: AI/ML Engineer

This module handles feature extraction and engineering for:
- Video performance prediction
- Trend analysis  
- Content optimization
- Cost prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import re
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import logging

logger = logging.getLogger(__name__)


class VideoFeatureEngineer:
    """Extract and engineer features from video metadata and content."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.tfidf_vectorizers = {}
        
    def extract_title_features(self, title: str) -> Dict[str, float]:
        """Extract features from video title."""
        features = {}
        
        # Basic length features
        features['title_length'] = len(title)
        features['title_word_count'] = len(title.split())
        features['title_char_count'] = len(title)
        
        # Capitalization features
        features['title_caps_ratio'] = sum(1 for c in title if c.isupper()) / len(title) if title else 0
        features['title_has_caps'] = int(any(c.isupper() for c in title))
        features['title_all_caps_words'] = sum(1 for word in title.split() if word.isupper())
        
        # Special characters
        features['title_exclamation_count'] = title.count('!')
        features['title_question_count'] = title.count('?')
        features['title_number_count'] = sum(1 for char in title if char.isdigit())
        features['title_emoji_count'] = len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', title))
        
        # Common words that drive engagement
        clickbait_words = ['amazing', 'shocking', 'incredible', 'unbelievable', 'secret', 'hack', 'trick', 'easy', 'simple', 'quick', 'fast', 'instant', 'ultimate', 'best', 'worst', 'epic', 'insane', 'crazy', 'weird', 'strange']
        features['title_clickbait_words'] = sum(1 for word in clickbait_words if word.lower() in title.lower())
        
        # Sentiment analysis
        blob = TextBlob(title)
        features['title_sentiment_polarity'] = blob.sentiment.polarity
        features['title_sentiment_subjectivity'] = blob.sentiment.subjectivity
        
        return features
    
    def extract_description_features(self, description: str) -> Dict[str, float]:
        """Extract features from video description."""
        if not description:
            description = ""
            
        features = {}
        
        # Length features
        features['desc_length'] = len(description)
        features['desc_word_count'] = len(description.split())
        features['desc_line_count'] = description.count('\n') + 1
        features['desc_paragraph_count'] = len([p for p in description.split('\n\n') if p.strip()])
        
        # URL and hashtag features
        features['desc_url_count'] = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', description))
        features['desc_hashtag_count'] = len(re.findall(r'#\w+', description))
        features['desc_mention_count'] = len(re.findall(r'@\w+', description))
        
        # Call-to-action features
        cta_words = ['subscribe', 'like', 'comment', 'share', 'follow', 'click', 'watch', 'check out', 'visit', 'download']
        features['desc_cta_count'] = sum(1 for cta in cta_words if cta.lower() in description.lower())
        
        # Sentiment
        blob = TextBlob(description)
        features['desc_sentiment_polarity'] = blob.sentiment.polarity
        features['desc_sentiment_subjectivity'] = blob.sentiment.subjectivity
        
        return features
    
    def extract_temporal_features(self, publish_time: datetime) -> Dict[str, float]:
        """Extract time-based features."""
        features = {}
        
        # Day of week (0 = Monday)
        features['publish_day_of_week'] = publish_time.weekday()
        features['is_weekend'] = int(publish_time.weekday() >= 5)
        
        # Hour of day
        features['publish_hour'] = publish_time.hour
        features['is_prime_time'] = int(18 <= publish_time.hour <= 22)  # 6-10 PM
        features['is_morning'] = int(6 <= publish_time.hour < 12)
        features['is_afternoon'] = int(12 <= publish_time.hour < 18)
        features['is_evening'] = int(18 <= publish_time.hour <= 22)
        features['is_night'] = int(publish_time.hour < 6 or publish_time.hour > 22)
        
        # Month and season
        features['publish_month'] = publish_time.month
        features['publish_quarter'] = (publish_time.month - 1) // 3 + 1
        
        # Is it a special day/period?
        features['is_january'] = int(publish_time.month == 1)  # New Year effect
        features['is_december'] = int(publish_time.month == 12)  # Holiday season
        features['is_summer'] = int(publish_time.month in [6, 7, 8])
        
        return features
    
    def extract_channel_features(self, channel_data: Dict) -> Dict[str, float]:
        """Extract features from channel metadata."""
        features = {}
        
        # Channel age and activity
        if 'created_date' in channel_data:
            created = pd.to_datetime(channel_data['created_date'])
            features['channel_age_days'] = (datetime.now() - created).days
            features['channel_age_years'] = features['channel_age_days'] / 365.25
        
        # Channel size and engagement
        features['channel_subscriber_count'] = channel_data.get('subscriber_count', 0)
        features['channel_video_count'] = channel_data.get('video_count', 0)
        features['channel_view_count'] = channel_data.get('total_view_count', 0)
        
        # Engagement rates
        if features['channel_video_count'] > 0:
            features['channel_avg_views_per_video'] = features['channel_view_count'] / features['channel_video_count']
        else:
            features['channel_avg_views_per_video'] = 0
            
        # Channel category
        if 'category' in channel_data:
            features['channel_category'] = channel_data['category']
        
        return features
    
    def extract_competition_features(self, video_data: Dict, market_data: Optional[Dict] = None) -> Dict[str, float]:
        """Extract features related to competition and market conditions."""
        features = {}
        
        # Market saturation features (if market data available)
        if market_data:
            features['market_video_count_last_7d'] = market_data.get('video_count_7d', 0)
            features['market_avg_views_last_7d'] = market_data.get('avg_views_7d', 0)
            features['market_competition_score'] = market_data.get('competition_score', 0.5)
        
        # Video uniqueness features
        title_words = set(video_data.get('title', '').lower().split())
        features['title_uniqueness_score'] = len(title_words) / max(len(video_data.get('title', '').split()), 1)
        
        return features
    
    def extract_cost_features(self, cost_data: Dict) -> Dict[str, float]:
        """Extract features related to production costs."""
        features = {}
        
        # Direct cost features
        features['script_cost'] = cost_data.get('script_cost', 0)
        features['voice_cost'] = cost_data.get('voice_cost', 0) 
        features['image_cost'] = cost_data.get('image_cost', 0)
        features['total_ai_cost'] = sum([
            features['script_cost'],
            features['voice_cost'], 
            features['image_cost']
        ])
        
        # Cost efficiency features
        features['cost_per_minute'] = features['total_ai_cost'] / max(cost_data.get('duration_minutes', 1), 0.1)
        features['is_under_budget'] = int(features['total_ai_cost'] <= 3.0)  # $3 target
        features['budget_utilization'] = features['total_ai_cost'] / 3.0
        
        return features
    
    def create_tfidf_features(self, texts: List[str], feature_name: str, max_features: int = 100) -> np.ndarray:
        """Create TF-IDF features from text data."""
        if feature_name not in self.tfidf_vectorizers:
            self.tfidf_vectorizers[feature_name] = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2),
                lowercase=True
            )
            tfidf_features = self.tfidf_vectorizers[feature_name].fit_transform(texts)
        else:
            tfidf_features = self.tfidf_vectorizers[feature_name].transform(texts)
            
        return tfidf_features.toarray()
    
    def engineer_features(self, video_data: List[Dict]) -> pd.DataFrame:
        """
        Engineer all features for a list of videos.
        
        Args:
            video_data: List of video dictionaries containing metadata
            
        Returns:
            DataFrame with engineered features
        """
        feature_rows = []
        
        for video in video_data:
            features = {}
            
            # Extract all feature types
            features.update(self.extract_title_features(video.get('title', '')))
            features.update(self.extract_description_features(video.get('description', '')))
            
            if 'publish_time' in video:
                features.update(self.extract_temporal_features(pd.to_datetime(video['publish_time'])))
            
            if 'channel_data' in video:
                features.update(self.extract_channel_features(video['channel_data']))
                
            if 'cost_data' in video:
                features.update(self.extract_cost_features(video['cost_data']))
                
            # Add video ID for tracking
            features['video_id'] = video.get('id', '')
            
            feature_rows.append(features)
        
        df = pd.DataFrame(feature_rows)
        
        # Handle missing values
        df = df.fillna(0)
        
        return df
    
    def scale_features(self, df: pd.DataFrame, feature_columns: List[str], fit: bool = True) -> pd.DataFrame:
        """Scale numerical features."""
        df_scaled = df.copy()
        
        for col in feature_columns:
            if col in df.columns:
                if fit:
                    if col not in self.scalers:
                        self.scalers[col] = StandardScaler()
                    df_scaled[col] = self.scalers[col].fit_transform(df[[col]])
                else:
                    if col in self.scalers:
                        df_scaled[col] = self.scalers[col].transform(df[[col]])
        
        return df_scaled
    
    def encode_categorical_features(self, df: pd.DataFrame, categorical_columns: List[str], fit: bool = True) -> pd.DataFrame:
        """Encode categorical features."""
        df_encoded = df.copy()
        
        for col in categorical_columns:
            if col in df.columns:
                if fit:
                    if col not in self.encoders:
                        self.encoders[col] = LabelEncoder()
                    df_encoded[col] = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.encoders:
                        # Handle unseen categories
                        try:
                            df_encoded[col] = self.encoders[col].transform(df[col].astype(str))
                        except ValueError:
                            # Assign default value for unseen categories
                            df_encoded[col] = 0
        
        return df_encoded
    
    def save_preprocessors(self, filepath: str) -> None:
        """Save fitted preprocessors."""
        preprocessors = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'tfidf_vectorizers': self.tfidf_vectorizers
        }
        joblib.dump(preprocessors, filepath)
        logger.info(f"Preprocessors saved to {filepath}")
    
    def load_preprocessors(self, filepath: str) -> None:
        """Load fitted preprocessors."""
        preprocessors = joblib.load(filepath)
        self.scalers = preprocessors.get('scalers', {})
        self.encoders = preprocessors.get('encoders', {})
        self.tfidf_vectorizers = preprocessors.get('tfidf_vectorizers', {})
        logger.info(f"Preprocessors loaded from {filepath}")


class TrendFeatureEngineer:
    """Extract features for trend analysis and prediction."""
    
    def __init__(self):
        self.trend_history_days = 30
        
    def extract_trend_features(self, keyword_data: Dict) -> Dict[str, float]:
        """Extract features from trending keyword data."""
        features = {}
        
        # Trend momentum
        if 'search_volume_history' in keyword_data:
            volumes = keyword_data['search_volume_history']
            if len(volumes) >= 7:
                recent_avg = np.mean(volumes[-7:])
                older_avg = np.mean(volumes[-14:-7]) if len(volumes) >= 14 else np.mean(volumes[:-7])
                features['trend_momentum'] = recent_avg / max(older_avg, 1)
                features['trend_volatility'] = np.std(volumes[-7:]) / max(np.mean(volumes[-7:]), 1)
            
        # Seasonality features
        features['keyword_length'] = len(keyword_data.get('keyword', ''))
        features['is_seasonal_keyword'] = int(self._is_seasonal_keyword(keyword_data.get('keyword', '')))
        
        return features
    
    def _is_seasonal_keyword(self, keyword: str) -> bool:
        """Check if keyword is seasonal."""
        seasonal_terms = ['christmas', 'halloween', 'summer', 'winter', 'spring', 'fall', 'holiday', 'valentine', 'easter']
        return any(term in keyword.lower() for term in seasonal_terms)


# Example usage and testing
if __name__ == "__main__":
    # Example video data
    sample_video = {
        'id': 'video_123',
        'title': 'Amazing AI Tricks That Will SHOCK You!',
        'description': 'Check out these incredible AI tricks! Subscribe for more amazing content. #AI #Technology #Amazing',
        'publish_time': '2025-01-10 18:30:00',
        'channel_data': {
            'subscriber_count': 50000,
            'video_count': 200,
            'total_view_count': 1000000,
            'created_date': '2020-01-01',
            'category': 'Technology'
        },
        'cost_data': {
            'script_cost': 0.75,
            'voice_cost': 1.20,
            'image_cost': 0.45,
            'duration_minutes': 8.5
        }
    }
    
    # Initialize feature engineer
    fe = VideoFeatureEngineer()
    
    # Engineer features for sample video
    features_df = fe.engineer_features([sample_video])
    
    print("Engineered Features:")
    print(features_df.T)  # Transpose for better readability
    print(f"\nTotal features: {len(features_df.columns)}")