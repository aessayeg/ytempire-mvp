"""
Personalization Model for Channel Preferences
Implements content personalization based on channel history and performance
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum

# ML Libraries
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Deep Learning (optional)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class PersonalizationType(Enum):
    """Types of personalization strategies"""
    CONTENT_BASED = "content_based"
    COLLABORATIVE_FILTERING = "collaborative_filtering"
    HYBRID = "hybrid"
    DEEP_LEARNING = "deep_learning"
    RULE_BASED = "rule_based"


class ContentStyle(Enum):
    """Content style categories for channels"""
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"
    NEWS = "news"
    TUTORIAL = "tutorial"
    REVIEW = "review"
    VLOG = "vlog"
    GAMING = "gaming"
    TECH = "tech"
    LIFESTYLE = "lifestyle"
    BUSINESS = "business"


@dataclass
class ChannelProfile:
    """Profile for a YouTube channel"""
    channel_id: str
    channel_name: str
    niche: str
    content_style: ContentStyle
    target_audience: Dict[str, Any]
    performance_metrics: Dict[str, float]
    content_preferences: Dict[str, Any]
    historical_videos: List[Dict[str, Any]]
    voice_preferences: Dict[str, Any]
    visual_preferences: Dict[str, Any]
    posting_schedule: Dict[str, Any]
    engagement_patterns: Dict[str, float]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class VideoRecommendation:
    """Personalized video recommendation"""
    video_id: str
    title: str
    script_template: str
    keywords: List[str]
    tone: str
    style: str
    estimated_engagement: float
    confidence_score: float
    reasoning: str
    personalization_factors: Dict[str, float]


@dataclass
class PersonalizationConfig:
    """Configuration for personalization model"""
    model_type: PersonalizationType = PersonalizationType.HYBRID
    embedding_dim: int = 128
    n_clusters: int = 10
    similarity_threshold: float = 0.7
    min_videos_for_training: int = 5
    update_frequency_days: int = 3
    use_deep_learning: bool = False
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 10
    save_path: Path = Path("models/personalization")


class PersonalizationEngine:
    """
    Main personalization engine for channel-specific content optimization
    """
    
    def __init__(self, config: Optional[PersonalizationConfig] = None):
        self.config = config or PersonalizationConfig()
        self.channel_profiles: Dict[str, ChannelProfile] = {}
        self.content_embeddings: Optional[np.ndarray] = None
        self.user_embeddings: Optional[np.ndarray] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.scaler = StandardScaler()
        self.kmeans_model: Optional[KMeans] = None
        self.neural_model = None
        
        self.models_path = self.config.save_path
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        if self.config.use_deep_learning and TORCH_AVAILABLE:
            self._init_neural_model()
    
    def _init_neural_model(self):
        """Initialize neural network for deep personalization"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Skipping neural model initialization.")
            return
        
        self.neural_model = PersonalizationNeuralNet(
            input_dim=self.config.embedding_dim * 2,
            hidden_dim=256,
            output_dim=self.config.embedding_dim
        )
    
    def create_channel_profile(
        self,
        channel_id: str,
        channel_data: Dict[str, Any],
        historical_videos: List[Dict[str, Any]]
    ) -> ChannelProfile:
        """
        Create a comprehensive profile for a channel
        """
        # Analyze historical performance
        performance_metrics = self._analyze_performance(historical_videos)
        
        # Extract content preferences
        content_preferences = self._extract_content_preferences(historical_videos)
        
        # Determine content style
        content_style = self._determine_content_style(
            channel_data.get('description', ''),
            historical_videos
        )
        
        # Analyze engagement patterns
        engagement_patterns = self._analyze_engagement_patterns(historical_videos)
        
        # Extract voice and visual preferences
        voice_prefs = self._extract_voice_preferences(historical_videos)
        visual_prefs = self._extract_visual_preferences(historical_videos)
        
        # Determine posting schedule
        posting_schedule = self._analyze_posting_schedule(historical_videos)
        
        profile = ChannelProfile(
            channel_id=channel_id,
            channel_name=channel_data.get('name', ''),
            niche=channel_data.get('niche', 'general'),
            content_style=content_style,
            target_audience=channel_data.get('target_audience', {}),
            performance_metrics=performance_metrics,
            content_preferences=content_preferences,
            historical_videos=historical_videos[-50:],  # Keep last 50 videos
            voice_preferences=voice_prefs,
            visual_preferences=visual_prefs,
            posting_schedule=posting_schedule,
            engagement_patterns=engagement_patterns
        )
        
        self.channel_profiles[channel_id] = profile
        logger.info(f"Created profile for channel {channel_id}")
        
        return profile
    
    def _analyze_performance(self, videos: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze historical video performance"""
        if not videos:
            return {}
        
        views = [v.get('views', 0) for v in videos]
        likes = [v.get('likes', 0) for v in videos]
        comments = [v.get('comments', 0) for v in videos]
        
        return {
            'avg_views': np.mean(views) if views else 0,
            'avg_likes': np.mean(likes) if likes else 0,
            'avg_comments': np.mean(comments) if comments else 0,
            'engagement_rate': np.mean([
                (l + c) / max(v, 1) for v, l, c in zip(views, likes, comments)
            ]) if views else 0,
            'view_velocity': np.mean([
                v / max((datetime.now() - datetime.fromisoformat(video.get('published_at', datetime.now().isoformat()))).days, 1)
                for v, video in zip(views, videos)
            ]) if views else 0,
            'viral_coefficient': len([v for v in views if v > np.mean(views) * 2]) / max(len(views), 1)
        }
    
    def _extract_content_preferences(self, videos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract content preferences from historical videos"""
        if not videos:
            return {}
        
        # Analyze titles and descriptions
        titles = [v.get('title', '') for v in videos]
        descriptions = [v.get('description', '') for v in videos]
        
        # Extract common themes
        all_text = ' '.join(titles + descriptions)
        
        # Simple keyword extraction (can be enhanced with NLP)
        words = all_text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 4:  # Filter short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Analyze video lengths
        durations = [v.get('duration', 0) for v in videos if v.get('duration')]
        
        return {
            'top_keywords': [k for k, _ in top_keywords],
            'avg_title_length': np.mean([len(t.split()) for t in titles]) if titles else 0,
            'avg_description_length': np.mean([len(d.split()) for d in descriptions]) if descriptions else 0,
            'preferred_duration': np.median(durations) if durations else 600,  # 10 minutes default
            'duration_variance': np.std(durations) if durations else 0,
            'title_patterns': self._extract_title_patterns(titles),
            'content_themes': self._extract_themes(titles + descriptions)
        }
    
    def _determine_content_style(
        self,
        channel_description: str,
        videos: List[Dict[str, Any]]
    ) -> ContentStyle:
        """Determine the content style of a channel"""
        # Simple rule-based classification (can be enhanced with ML)
        text = channel_description.lower()
        
        for video in videos[:10]:  # Check recent videos
            text += ' ' + video.get('title', '').lower()
            text += ' ' + video.get('description', '').lower()
        
        style_keywords = {
            ContentStyle.EDUCATIONAL: ['learn', 'tutorial', 'how to', 'guide', 'course'],
            ContentStyle.ENTERTAINMENT: ['funny', 'comedy', 'laugh', 'fun', 'entertainment'],
            ContentStyle.NEWS: ['news', 'breaking', 'update', 'report', 'coverage'],
            ContentStyle.TUTORIAL: ['tutorial', 'how to', 'diy', 'step by step', 'guide'],
            ContentStyle.REVIEW: ['review', 'unboxing', 'first look', 'comparison', 'vs'],
            ContentStyle.VLOG: ['vlog', 'day in', 'life', 'daily', 'routine'],
            ContentStyle.GAMING: ['gaming', 'gameplay', 'game', 'play', 'stream'],
            ContentStyle.TECH: ['tech', 'technology', 'gadget', 'software', 'hardware'],
            ContentStyle.LIFESTYLE: ['lifestyle', 'fashion', 'beauty', 'wellness', 'health'],
            ContentStyle.BUSINESS: ['business', 'entrepreneur', 'startup', 'marketing', 'finance']
        }
        
        style_scores = {}
        for style, keywords in style_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            style_scores[style] = score
        
        # Return style with highest score
        if style_scores:
            return max(style_scores, key=style_scores.get)
        
        return ContentStyle.ENTERTAINMENT  # Default
    
    def _analyze_engagement_patterns(self, videos: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze engagement patterns from historical videos"""
        if not videos:
            return {}
        
        # Time-based engagement analysis
        engagement_by_hour = {}
        engagement_by_day = {}
        
        for video in videos:
            published_at = video.get('published_at')
            if published_at:
                dt = datetime.fromisoformat(published_at)
                hour = dt.hour
                day = dt.weekday()
                
                views = video.get('views', 0)
                engagement = video.get('likes', 0) + video.get('comments', 0)
                
                if hour not in engagement_by_hour:
                    engagement_by_hour[hour] = []
                engagement_by_hour[hour].append(engagement / max(views, 1))
                
                if day not in engagement_by_day:
                    engagement_by_day[day] = []
                engagement_by_day[day].append(engagement / max(views, 1))
        
        # Calculate average engagement by time
        avg_by_hour = {h: np.mean(e) for h, e in engagement_by_hour.items()}
        avg_by_day = {d: np.mean(e) for d, e in engagement_by_day.items()}
        
        # Find best times
        best_hour = max(avg_by_hour, key=avg_by_hour.get) if avg_by_hour else 12
        best_day = max(avg_by_day, key=avg_by_day.get) if avg_by_day else 4  # Friday
        
        return {
            'best_posting_hour': float(best_hour),
            'best_posting_day': float(best_day),
            'engagement_consistency': np.std(list(avg_by_hour.values())) if avg_by_hour else 0,
            'weekend_vs_weekday': self._calculate_weekend_weekday_ratio(avg_by_day)
        }
    
    def _extract_voice_preferences(self, videos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract voice preferences from video metadata"""
        voice_data = {
            'preferred_gender': 'neutral',
            'preferred_speed': 1.0,
            'preferred_pitch': 'medium',
            'emotion_level': 'moderate',
            'accent': 'neutral'
        }
        
        # Analyze voice metadata if available
        for video in videos:
            voice_meta = video.get('voice_metadata', {})
            if voice_meta:
                # Aggregate voice characteristics
                # This would require actual voice analysis data
                pass
        
        return voice_data
    
    def _extract_visual_preferences(self, videos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract visual style preferences"""
        visual_data = {
            'thumbnail_style': 'vibrant',
            'color_scheme': 'bright',
            'text_overlay': True,
            'face_in_thumbnail': False,
            'complexity': 'medium'
        }
        
        # Analyze thumbnail and visual metadata if available
        for video in videos:
            visual_meta = video.get('visual_metadata', {})
            if visual_meta:
                # Aggregate visual characteristics
                pass
        
        return visual_data
    
    def _analyze_posting_schedule(self, videos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze posting schedule patterns"""
        if not videos:
            return {'frequency': 'weekly', 'videos_per_week': 2}
        
        # Extract posting dates
        dates = []
        for video in videos:
            published_at = video.get('published_at')
            if published_at:
                dates.append(datetime.fromisoformat(published_at))
        
        if len(dates) < 2:
            return {'frequency': 'weekly', 'videos_per_week': 2}
        
        dates.sort()
        
        # Calculate posting frequency
        time_diffs = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
        avg_days_between = np.mean(time_diffs) if time_diffs else 7
        
        # Determine frequency category
        if avg_days_between <= 1:
            frequency = 'daily'
            videos_per_week = 7
        elif avg_days_between <= 3:
            frequency = 'bi-weekly'
            videos_per_week = 3
        elif avg_days_between <= 7:
            frequency = 'weekly'
            videos_per_week = 1
        else:
            frequency = 'monthly'
            videos_per_week = 0.25
        
        return {
            'frequency': frequency,
            'videos_per_week': videos_per_week,
            'avg_days_between_posts': avg_days_between,
            'posting_consistency': np.std(time_diffs) if time_diffs else 0
        }
    
    def _extract_title_patterns(self, titles: List[str]) -> List[str]:
        """Extract common patterns from titles"""
        patterns = []
        
        # Common title formats
        if any('|' in t for t in titles):
            patterns.append('pipe_separator')
        if any(':' in t for t in titles):
            patterns.append('colon_separator')
        if any('?' in t for t in titles):
            patterns.append('question_format')
        if any(t.isupper() for t in titles):
            patterns.append('all_caps')
        if any(any(char.isdigit() for char in t) for t in titles):
            patterns.append('contains_numbers')
        
        return patterns
    
    def _extract_themes(self, texts: List[str]) -> List[str]:
        """Extract main themes from text content"""
        if not texts:
            return []
        
        # Simple theme extraction (can be enhanced with topic modeling)
        all_text = ' '.join(texts).lower()
        
        themes = []
        theme_keywords = {
            'technology': ['tech', 'software', 'hardware', 'ai', 'machine learning'],
            'business': ['business', 'startup', 'entrepreneur', 'marketing', 'sales'],
            'education': ['learn', 'course', 'tutorial', 'education', 'study'],
            'entertainment': ['fun', 'comedy', 'entertainment', 'funny', 'laugh'],
            'lifestyle': ['lifestyle', 'life', 'daily', 'routine', 'wellness'],
            'gaming': ['game', 'gaming', 'play', 'stream', 'esports']
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in all_text for keyword in keywords):
                themes.append(theme)
        
        return themes[:3]  # Return top 3 themes
    
    def _calculate_weekend_weekday_ratio(self, avg_by_day: Dict[int, float]) -> float:
        """Calculate engagement ratio between weekend and weekday"""
        if not avg_by_day:
            return 1.0
        
        weekend_days = [5, 6]  # Saturday, Sunday
        weekday_days = [0, 1, 2, 3, 4]  # Monday to Friday
        
        weekend_avg = np.mean([avg_by_day.get(d, 0) for d in weekend_days])
        weekday_avg = np.mean([avg_by_day.get(d, 0) for d in weekday_days])
        
        if weekday_avg == 0:
            return 1.0
        
        return weekend_avg / weekday_avg
    
    def generate_personalized_content(
        self,
        channel_id: str,
        content_type: str = 'video',
        trending_topics: Optional[List[str]] = None
    ) -> VideoRecommendation:
        """
        Generate personalized content recommendations for a channel
        """
        if channel_id not in self.channel_profiles:
            raise ValueError(f"Channel profile not found for {channel_id}")
        
        profile = self.channel_profiles[channel_id]
        
        # Generate title based on channel preferences
        title = self._generate_personalized_title(profile, trending_topics)
        
        # Generate script template
        script_template = self._generate_script_template(profile)
        
        # Extract relevant keywords
        keywords = self._extract_personalized_keywords(profile, trending_topics)
        
        # Determine tone and style
        tone = self._determine_tone(profile)
        style = profile.content_style.value
        
        # Estimate engagement
        estimated_engagement = self._estimate_engagement(profile, title, keywords)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(profile)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(profile, trending_topics)
        
        # Calculate personalization factors
        personalization_factors = self._calculate_personalization_factors(profile)
        
        return VideoRecommendation(
            video_id=f"{channel_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            title=title,
            script_template=script_template,
            keywords=keywords,
            tone=tone,
            style=style,
            estimated_engagement=estimated_engagement,
            confidence_score=confidence_score,
            reasoning=reasoning,
            personalization_factors=personalization_factors
        )
    
    def _generate_personalized_title(
        self,
        profile: ChannelProfile,
        trending_topics: Optional[List[str]] = None
    ) -> str:
        """Generate a personalized title based on channel preferences"""
        title_templates = {
            ContentStyle.EDUCATIONAL: [
                "How to {topic}: Complete Guide",
                "{topic} Explained in {duration} Minutes",
                "Master {topic} with These {number} Tips"
            ],
            ContentStyle.ENTERTAINMENT: [
                "{topic} Gone Wrong!",
                "You Won't Believe This {topic}",
                "{topic} Challenge - Epic Fail or Win?"
            ],
            ContentStyle.TUTORIAL: [
                "{topic} Tutorial for Beginners",
                "Step-by-Step: {topic}",
                "{topic} in {duration} Minutes"
            ],
            ContentStyle.REVIEW: [
                "{topic} Review: Is It Worth It?",
                "Honest {topic} Review After {duration} Days",
                "{topic} vs {alternative}: Which Is Better?"
            ]
        }
        
        templates = title_templates.get(
            profile.content_style,
            ["{topic}: What You Need to Know"]
        )
        
        template = np.random.choice(templates)
        
        # Fill in template
        topic = trending_topics[0] if trending_topics else profile.niche
        duration = np.random.choice([5, 10, 15, 20])
        number = np.random.choice([3, 5, 7, 10])
        alternative = "Alternative" if "{alternative}" in template else ""
        
        title = template.format(
            topic=topic,
            duration=duration,
            number=number,
            alternative=alternative
        )
        
        # Apply title patterns
        if 'pipe_separator' in profile.content_preferences.get('title_patterns', []):
            title = f"{profile.channel_name} | {title}"
        
        return title
    
    def _generate_script_template(self, profile: ChannelProfile) -> str:
        """Generate a script template based on channel style"""
        templates = {
            ContentStyle.EDUCATIONAL: """
            Introduction: Hook the audience with a question or surprising fact
            Overview: What will be covered in this video
            Main Content:
            - Point 1: Detailed explanation
            - Point 2: Examples and demonstrations
            - Point 3: Common mistakes to avoid
            Summary: Recap key points
            Call to Action: Subscribe and comment
            """,
            ContentStyle.ENTERTAINMENT: """
            Cold Open: Start with the most exciting moment
            Introduction: Quick personal greeting
            Setup: Build anticipation
            Main Event: The core entertainment content
            Reaction: Personal commentary
            Outro: Thank viewers and tease next video
            """,
            ContentStyle.TUTORIAL: """
            Introduction: What you'll learn
            Prerequisites: What you need
            Step 1: First action with clear instructions
            Step 2: Continue with visual demonstrations
            Step 3: Final steps and verification
            Troubleshooting: Common issues
            Conclusion: Next steps and resources
            """
        }
        
        return templates.get(profile.content_style, "Introduction, Main Content, Conclusion")
    
    def _extract_personalized_keywords(
        self,
        profile: ChannelProfile,
        trending_topics: Optional[List[str]] = None
    ) -> List[str]:
        """Extract keywords personalized for the channel"""
        keywords = profile.content_preferences.get('top_keywords', [])[:5]
        
        if trending_topics:
            # Blend channel keywords with trending topics
            keywords.extend(trending_topics[:3])
        
        # Add niche-specific keywords
        keywords.append(profile.niche)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords[:8]
    
    def _determine_tone(self, profile: ChannelProfile) -> str:
        """Determine the appropriate tone for content"""
        tone_map = {
            ContentStyle.EDUCATIONAL: "informative",
            ContentStyle.ENTERTAINMENT: "energetic",
            ContentStyle.NEWS: "professional",
            ContentStyle.TUTORIAL: "friendly",
            ContentStyle.REVIEW: "honest",
            ContentStyle.VLOG: "casual",
            ContentStyle.GAMING: "enthusiastic",
            ContentStyle.TECH: "analytical",
            ContentStyle.LIFESTYLE: "inspiring",
            ContentStyle.BUSINESS: "authoritative"
        }
        
        return tone_map.get(profile.content_style, "conversational")
    
    def _estimate_engagement(
        self,
        profile: ChannelProfile,
        title: str,
        keywords: List[str]
    ) -> float:
        """Estimate expected engagement for content"""
        base_engagement = profile.performance_metrics.get('engagement_rate', 0.05)
        
        # Factors that affect engagement
        factors = []
        
        # Title length factor
        title_length = len(title.split())
        optimal_length = profile.content_preferences.get('avg_title_length', 10)
        length_factor = 1 - abs(title_length - optimal_length) / max(optimal_length, 1) * 0.1
        factors.append(length_factor)
        
        # Keyword relevance factor
        channel_keywords = set(profile.content_preferences.get('top_keywords', []))
        keyword_overlap = len(set(keywords) & channel_keywords) / max(len(keywords), 1)
        factors.append(1 + keyword_overlap * 0.2)
        
        # Historical performance factor
        if profile.historical_videos:
            recent_engagement = np.mean([
                v.get('engagement_rate', 0) 
                for v in profile.historical_videos[-5:]
            ])
            trend_factor = 1 + (recent_engagement - base_engagement) / max(base_engagement, 0.01)
            factors.append(trend_factor)
        
        # Calculate final engagement estimate
        estimated_engagement = base_engagement * np.prod(factors)
        
        # Bound between 0 and 1
        return min(max(estimated_engagement, 0), 1)
    
    def _calculate_confidence(self, profile: ChannelProfile) -> float:
        """Calculate confidence score for personalization"""
        confidence_factors = []
        
        # Data availability
        n_videos = len(profile.historical_videos)
        data_confidence = min(n_videos / 20, 1.0)  # Max confidence at 20+ videos
        confidence_factors.append(data_confidence)
        
        # Performance consistency
        if profile.historical_videos:
            engagements = [v.get('engagement_rate', 0) for v in profile.historical_videos]
            if engagements:
                consistency = 1 - np.std(engagements) / (np.mean(engagements) + 0.001)
                confidence_factors.append(max(consistency, 0))
        
        # Profile completeness
        profile_fields = [
            profile.niche,
            profile.content_style,
            profile.content_preferences,
            profile.performance_metrics
        ]
        completeness = sum(1 for f in profile_fields if f) / len(profile_fields)
        confidence_factors.append(completeness)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _generate_reasoning(
        self,
        profile: ChannelProfile,
        trending_topics: Optional[List[str]] = None
    ) -> str:
        """Generate reasoning for the recommendation"""
        reasons = []
        
        reasons.append(f"Based on {len(profile.historical_videos)} historical videos")
        reasons.append(f"Channel style: {profile.content_style.value}")
        
        if profile.performance_metrics:
            avg_engagement = profile.performance_metrics.get('engagement_rate', 0)
            reasons.append(f"Average engagement rate: {avg_engagement:.2%}")
        
        if trending_topics:
            reasons.append(f"Incorporating trending topic: {trending_topics[0]}")
        
        if profile.engagement_patterns:
            best_hour = profile.engagement_patterns.get('best_posting_hour', 12)
            reasons.append(f"Optimal posting time: {int(best_hour)}:00")
        
        return ". ".join(reasons)
    
    def _calculate_personalization_factors(self, profile: ChannelProfile) -> Dict[str, float]:
        """Calculate individual personalization factor contributions"""
        return {
            'channel_history': 0.3,
            'content_style': 0.25,
            'performance_metrics': 0.2,
            'engagement_patterns': 0.15,
            'trending_alignment': 0.1
        }
    
    def update_profile_with_feedback(
        self,
        channel_id: str,
        video_id: str,
        performance_data: Dict[str, Any]
    ) -> None:
        """
        Update channel profile based on video performance feedback
        """
        if channel_id not in self.channel_profiles:
            logger.warning(f"Channel {channel_id} not found")
            return
        
        profile = self.channel_profiles[channel_id]
        
        # Add video to historical data
        video_data = {
            'video_id': video_id,
            'published_at': datetime.now().isoformat(),
            **performance_data
        }
        profile.historical_videos.append(video_data)
        
        # Keep only recent videos
        profile.historical_videos = profile.historical_videos[-100:]
        
        # Update performance metrics
        profile.performance_metrics = self._analyze_performance(profile.historical_videos)
        
        # Update content preferences
        profile.content_preferences = self._extract_content_preferences(profile.historical_videos)
        
        # Update engagement patterns
        profile.engagement_patterns = self._analyze_engagement_patterns(profile.historical_videos)
        
        # Update timestamp
        profile.updated_at = datetime.now()
        
        logger.info(f"Updated profile for channel {channel_id}")
    
    def get_channel_insights(self, channel_id: str) -> Dict[str, Any]:
        """
        Get detailed insights for a channel
        """
        if channel_id not in self.channel_profiles:
            return {"error": "Channel not found"}
        
        profile = self.channel_profiles[channel_id]
        
        return {
            "channel_id": channel_id,
            "channel_name": profile.channel_name,
            "content_style": profile.content_style.value,
            "niche": profile.niche,
            "performance": {
                "avg_views": profile.performance_metrics.get('avg_views', 0),
                "avg_engagement": profile.performance_metrics.get('engagement_rate', 0),
                "viral_coefficient": profile.performance_metrics.get('viral_coefficient', 0)
            },
            "content_preferences": {
                "top_keywords": profile.content_preferences.get('top_keywords', [])[:10],
                "preferred_duration": profile.content_preferences.get('preferred_duration', 600),
                "title_patterns": profile.content_preferences.get('title_patterns', [])
            },
            "optimal_schedule": {
                "best_hour": profile.engagement_patterns.get('best_posting_hour', 12),
                "best_day": profile.engagement_patterns.get('best_posting_day', 4),
                "posting_frequency": profile.posting_schedule.get('frequency', 'weekly')
            },
            "recent_videos": len(profile.historical_videos),
            "profile_confidence": self._calculate_confidence(profile)
        }
    
    def save_profiles(self, path: Optional[Path] = None) -> None:
        """Save all channel profiles"""
        save_path = path or self.models_path / "channel_profiles.pkl"
        
        with open(save_path, 'wb') as f:
            joblib.dump(self.channel_profiles, f)
        
        logger.info(f"Saved {len(self.channel_profiles)} profiles to {save_path}")
    
    def load_profiles(self, path: Path) -> None:
        """Load channel profiles"""
        with open(path, 'rb') as f:
            self.channel_profiles = joblib.load(f)
        
        logger.info(f"Loaded {len(self.channel_profiles)} profiles from {path}")


class PersonalizationNeuralNet(nn.Module):
    """
    Neural network for deep personalization (requires PyTorch)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(PersonalizationNeuralNet, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
        self.dropout = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim // 2)
    
    def forward(self, x):
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize personalization engine
    config = PersonalizationConfig(
        model_type=PersonalizationType.HYBRID,
        use_deep_learning=TORCH_AVAILABLE
    )
    
    engine = PersonalizationEngine(config)
    
    # Create sample channel data
    channel_data = {
        'name': 'TechExplained',
        'description': 'Technology tutorials and reviews',
        'niche': 'technology',
        'target_audience': {
            'age_range': '18-35',
            'interests': ['tech', 'gadgets', 'programming']
        }
    }
    
    # Sample historical videos
    historical_videos = [
        {
            'title': 'Python Tutorial for Beginners',
            'description': 'Learn Python programming from scratch',
            'views': 50000,
            'likes': 2500,
            'comments': 300,
            'duration': 900,
            'published_at': (datetime.now() - timedelta(days=7)).isoformat(),
            'engagement_rate': 0.056
        },
        {
            'title': 'Top 10 VS Code Extensions',
            'description': 'Must-have extensions for developers',
            'views': 35000,
            'likes': 1800,
            'comments': 250,
            'duration': 600,
            'published_at': (datetime.now() - timedelta(days=14)).isoformat(),
            'engagement_rate': 0.059
        }
    ]
    
    # Create channel profile
    profile = engine.create_channel_profile(
        'channel_123',
        channel_data,
        historical_videos
    )
    
    print(f"Created profile for: {profile.channel_name}")
    print(f"Content style: {profile.content_style.value}")
    print(f"Performance metrics: {profile.performance_metrics}")
    
    # Generate personalized content
    trending_topics = ['AI', 'Machine Learning', 'ChatGPT']
    recommendation = engine.generate_personalized_content(
        'channel_123',
        trending_topics=trending_topics
    )
    
    print(f"\nPersonalized recommendation:")
    print(f"Title: {recommendation.title}")
    print(f"Keywords: {recommendation.keywords}")
    print(f"Tone: {recommendation.tone}")
    print(f"Estimated engagement: {recommendation.estimated_engagement:.2%}")
    print(f"Confidence: {recommendation.confidence_score:.2%}")
    print(f"Reasoning: {recommendation.reasoning}")
    
    # Get channel insights
    insights = engine.get_channel_insights('channel_123')
    print(f"\nChannel insights: {json.dumps(insights, indent=2, default=str)}")
    
    # Save profiles
    engine.save_profiles()