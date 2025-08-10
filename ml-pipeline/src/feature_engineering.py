"""
Feature Engineering Pipeline for YTEmpire
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VideoFeatures:
    """Video feature container"""
    title_length: int
    title_caps_ratio: float
    title_emoji_count: int
    title_keyword_score: float
    description_length: int
    tags_count: int
    optimal_upload_time: int
    trend_score: float
    niche_relevance: float
    competition_score: float
    predicted_views: float
    predicted_engagement: float


class FeatureEngineer:
    """Main feature engineering pipeline"""
    
    def __init__(self):
        self.trending_keywords = []
        self.niche_keywords = {}
        self.optimal_times = {}
        
    def extract_title_features(self, title: str) -> Dict[str, Any]:
        """Extract features from video title"""
        features = {
            "length": len(title),
            "word_count": len(title.split()),
            "caps_ratio": sum(1 for c in title if c.isupper()) / max(len(title), 1),
            "has_emoji": any(ord(c) > 127 for c in title),
            "emoji_count": sum(1 for c in title if ord(c) > 127),
            "has_numbers": any(c.isdigit() for c in title),
            "exclamation_count": title.count("!"),
            "question_mark": "?" in title,
            "has_brackets": "(" in title or "[" in title,
        }
        
        # Keyword scoring
        features["keyword_score"] = self._calculate_keyword_score(title)
        
        # Emotional appeal
        features["emotional_score"] = self._calculate_emotional_score(title)
        
        return features
    
    def extract_temporal_features(self, upload_time: datetime) -> Dict[str, Any]:
        """Extract temporal features"""
        features = {
            "hour": upload_time.hour,
            "day_of_week": upload_time.weekday(),
            "day_of_month": upload_time.day,
            "month": upload_time.month,
            "quarter": (upload_time.month - 1) // 3 + 1,
            "is_weekend": upload_time.weekday() >= 5,
            "is_holiday": self._is_holiday(upload_time),
        }
        
        # Peak hours (typically 2-4 PM and 7-9 PM)
        features["is_peak_hour"] = (
            (14 <= upload_time.hour <= 16) or 
            (19 <= upload_time.hour <= 21)
        )
        
        return features
    
    def extract_trend_features(self, topic: str, timestamp: datetime) -> Dict[str, Any]:
        """Extract trend-related features"""
        features = {
            "trend_score": self._calculate_trend_score(topic, timestamp),
            "trend_velocity": self._calculate_trend_velocity(topic),
            "trend_acceleration": self._calculate_trend_acceleration(topic),
            "seasonal_relevance": self._calculate_seasonal_relevance(topic, timestamp),
            "evergreen_score": self._calculate_evergreen_score(topic),
        }
        
        return features
    
    def extract_competition_features(self, niche: str, keywords: List[str]) -> Dict[str, Any]:
        """Extract competition-related features"""
        features = {
            "niche_saturation": self._calculate_niche_saturation(niche),
            "keyword_difficulty": self._calculate_keyword_difficulty(keywords),
            "opportunity_score": self._calculate_opportunity_score(niche, keywords),
            "competitor_count": self._estimate_competitor_count(niche),
        }
        
        return features
    
    def extract_engagement_features(self, 
                                   title: str, 
                                   thumbnail_quality: float,
                                   video_length: int) -> Dict[str, Any]:
        """Extract engagement prediction features"""
        features = {
            "ctr_prediction": self._predict_ctr(title, thumbnail_quality),
            "retention_prediction": self._predict_retention(video_length),
            "like_ratio_prediction": self._predict_like_ratio(title),
            "comment_rate_prediction": self._predict_comment_rate(title),
            "share_probability": self._predict_share_probability(title),
        }
        
        return features
    
    def create_feature_vector(self, video_data: Dict[str, Any]) -> np.ndarray:
        """Create complete feature vector for ML models"""
        all_features = []
        
        # Title features
        title_features = self.extract_title_features(video_data["title"])
        all_features.extend(title_features.values())
        
        # Temporal features
        temporal_features = self.extract_temporal_features(video_data["upload_time"])
        all_features.extend(temporal_features.values())
        
        # Trend features
        trend_features = self.extract_trend_features(
            video_data["topic"], 
            video_data["upload_time"]
        )
        all_features.extend(trend_features.values())
        
        # Competition features
        competition_features = self.extract_competition_features(
            video_data["niche"],
            video_data["keywords"]
        )
        all_features.extend(competition_features.values())
        
        # Engagement features
        engagement_features = self.extract_engagement_features(
            video_data["title"],
            video_data.get("thumbnail_quality", 0.8),
            video_data.get("duration", 600)
        )
        all_features.extend(engagement_features.values())
        
        return np.array(all_features, dtype=np.float32)
    
    def _calculate_keyword_score(self, title: str) -> float:
        """Calculate keyword relevance score"""
        score = 0.0
        title_lower = title.lower()
        
        # Check trending keywords
        for keyword in self.trending_keywords:
            if keyword.lower() in title_lower:
                score += 1.0
        
        # Normalize
        return min(score / max(len(self.trending_keywords), 1), 1.0)
    
    def _calculate_emotional_score(self, title: str) -> float:
        """Calculate emotional appeal score"""
        emotional_words = [
            "amazing", "incredible", "shocking", "unbelievable", "epic",
            "insane", "crazy", "mind-blowing", "ultimate", "best",
            "worst", "fail", "win", "genius", "stupid"
        ]
        
        score = sum(1 for word in emotional_words if word in title.lower())
        return min(score / 3, 1.0)  # Normalize to 0-1
    
    def _calculate_trend_score(self, topic: str, timestamp: datetime) -> float:
        """Calculate current trend score for topic"""
        # This would connect to trend analysis service
        # Placeholder implementation
        base_score = np.random.random()
        
        # Add recency bonus
        days_old = (datetime.now() - timestamp).days
        recency_multiplier = max(0, 1 - (days_old / 30))
        
        return base_score * recency_multiplier
    
    def _calculate_trend_velocity(self, topic: str) -> float:
        """Calculate rate of trend growth"""
        # Placeholder - would analyze trend history
        return np.random.random()
    
    def _calculate_trend_acceleration(self, topic: str) -> float:
        """Calculate acceleration of trend growth"""
        # Placeholder - would analyze trend history
        return np.random.random() - 0.5  # Can be negative
    
    def _calculate_seasonal_relevance(self, topic: str, timestamp: datetime) -> float:
        """Calculate seasonal relevance score"""
        month = timestamp.month
        
        # Seasonal topics mapping (simplified)
        seasonal_topics = {
            12: ["christmas", "holiday", "gift"],
            10: ["halloween", "spooky", "costume"],
            7: ["summer", "vacation", "beach"],
            2: ["valentine", "love", "romance"],
        }
        
        score = 0.0
        if month in seasonal_topics:
            for keyword in seasonal_topics[month]:
                if keyword in topic.lower():
                    score = 1.0
                    break
        
        return score
    
    def _calculate_evergreen_score(self, topic: str) -> float:
        """Calculate evergreen content score"""
        evergreen_keywords = [
            "how to", "tutorial", "guide", "tips", "tricks",
            "explained", "review", "best", "top", "ultimate"
        ]
        
        score = sum(1 for keyword in evergreen_keywords if keyword in topic.lower())
        return min(score / 2, 1.0)
    
    def _calculate_niche_saturation(self, niche: str) -> float:
        """Calculate market saturation for niche"""
        # Placeholder - would analyze competitor data
        saturation_scores = {
            "gaming": 0.9,
            "tech": 0.85,
            "education": 0.6,
            "finance": 0.7,
            "lifestyle": 0.8,
        }
        return saturation_scores.get(niche.lower(), 0.5)
    
    def _calculate_keyword_difficulty(self, keywords: List[str]) -> float:
        """Calculate average keyword difficulty"""
        # Placeholder - would use SEO data
        return np.random.random() * 0.7 + 0.3
    
    def _calculate_opportunity_score(self, niche: str, keywords: List[str]) -> float:
        """Calculate opportunity score (low competition, high demand)"""
        saturation = self._calculate_niche_saturation(niche)
        difficulty = self._calculate_keyword_difficulty(keywords)
        
        # Inverse relationship - low saturation/difficulty = high opportunity
        return 1.0 - (saturation * 0.6 + difficulty * 0.4)
    
    def _estimate_competitor_count(self, niche: str) -> int:
        """Estimate number of competitors in niche"""
        # Placeholder - would analyze channel data
        competitor_estimates = {
            "gaming": 10000,
            "tech": 5000,
            "education": 3000,
            "finance": 2000,
            "lifestyle": 8000,
        }
        return competitor_estimates.get(niche.lower(), 1000)
    
    def _predict_ctr(self, title: str, thumbnail_quality: float) -> float:
        """Predict click-through rate"""
        title_score = self._calculate_keyword_score(title) * 0.3
        emotional_score = self._calculate_emotional_score(title) * 0.2
        thumbnail_score = thumbnail_quality * 0.5
        
        return min(title_score + emotional_score + thumbnail_score, 1.0)
    
    def _predict_retention(self, video_length: int) -> float:
        """Predict audience retention based on video length"""
        # Optimal length is typically 8-12 minutes
        if 480 <= video_length <= 720:
            return 0.8
        elif video_length < 300:
            return 0.6  # Too short
        elif video_length > 1200:
            return 0.5  # Too long
        else:
            return 0.7
    
    def _predict_like_ratio(self, title: str) -> float:
        """Predict like/dislike ratio"""
        emotional_score = self._calculate_emotional_score(title)
        return 0.7 + (emotional_score * 0.2)
    
    def _predict_comment_rate(self, title: str) -> float:
        """Predict comment rate"""
        # Questions and controversial topics get more comments
        has_question = "?" in title
        controversy_score = 0.2 if has_question else 0.1
        
        return controversy_score + np.random.random() * 0.1
    
    def _predict_share_probability(self, title: str) -> float:
        """Predict share probability"""
        emotional_score = self._calculate_emotional_score(title)
        return emotional_score * 0.3 + np.random.random() * 0.1
    
    def _is_holiday(self, date: datetime) -> bool:
        """Check if date is a holiday"""
        # Simplified holiday check
        holidays = [
            (1, 1),   # New Year
            (7, 4),   # July 4th
            (12, 25), # Christmas
            (10, 31), # Halloween
            (11, -1), # Thanksgiving (simplified)
        ]
        
        for month, day in holidays:
            if date.month == month and (day == -1 or date.day == day):
                return True
        
        return False


class FeatureStore:
    """Store and retrieve features for ML pipeline"""
    
    def __init__(self):
        self.features = {}
        self.feature_history = []
        
    def store_features(self, video_id: str, features: np.ndarray, metadata: Dict[str, Any]):
        """Store features for a video"""
        self.features[video_id] = {
            "features": features,
            "metadata": metadata,
            "timestamp": datetime.now()
        }
        
        # Keep history for training
        self.feature_history.append({
            "video_id": video_id,
            "features": features,
            "timestamp": datetime.now()
        })
        
    def get_features(self, video_id: str) -> Optional[np.ndarray]:
        """Retrieve features for a video"""
        if video_id in self.features:
            return self.features[video_id]["features"]
        return None
    
    def get_training_data(self, limit: int = 1000) -> pd.DataFrame:
        """Get recent features for training"""
        if not self.feature_history:
            return pd.DataFrame()
        
        recent_features = self.feature_history[-limit:]
        return pd.DataFrame(recent_features)