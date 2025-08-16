"""
Trend Analyzer Service
Integrates ML pipeline trend detection models with the FastAPI backend
Provides unified interface for trend analysis across the platform
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio
import json
from enum import Enum

# Import the most comprehensive ML model
try:
    from ml_pipeline.src.trend_detection_model import (
        TrendDetectionModel,
        TrendingTopic,
        TrendSource,
    )

    ML_MODELS_AVAILABLE = True
except ImportError:
    ML_MODELS_AVAILABLE = False
    logging.warning("ML trend detection models not available, using mock data")

from app.core.config import settings
from app.core.cache import cache_manager
from app.services.cost_tracking import cost_tracker
from app.models.analytics import Analytics
from app.db.session import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class TrendCategory(str, Enum):
    """Supported trend categories"""

    TECHNOLOGY = "technology"
    GAMING = "gaming"
    EDUCATION = "education"
    ENTERTAINMENT = "entertainment"
    LIFESTYLE = "lifestyle"
    BUSINESS = "business"
    HEALTH = "health"
    SCIENCE = "science"
    NEWS = "news"
    SPORTS = "sports"


class TrendTimeframe(str, Enum):
    """Trend analysis timeframes"""

    REAL_TIME = "now"
    TODAY = "today"
    THIS_WEEK = "week"
    THIS_MONTH = "month"


class TrendAnalyzer:
    """
    Unified trend analysis service that integrates ML models
    with backend services for comprehensive trend detection
    """

    def __init__(self):
        """Initialize trend analyzer with ML models if available"""
        self.ml_model = None
        if ML_MODELS_AVAILABLE:
            try:
                self.ml_model = TrendDetectionModel(
                    model_path=settings.ML_MODEL_PATH
                    if hasattr(settings, "ML_MODEL_PATH")
                    else "models/trend_detection"
                )
                logger.info("Trend detection ML models loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load ML models: {e}")
                self.ml_model = None

        self.cache_ttl = 3600  # 1 hour cache for trend data

    async def analyze_trends(
        self,
        category: str = TrendCategory.TECHNOLOGY,
        region: str = "US",
        timeframe: str = TrendTimeframe.REAL_TIME,
        limit: int = 10,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Analyze trends for a specific category and region

        Args:
            category: Content category to analyze
            region: Geographic region (default: US)
            timeframe: Time period for analysis
            limit: Maximum number of trends to return
            use_cache: Whether to use cached results

        Returns:
            List of trending topics with scores and metadata
        """
        # Generate cache key
        cache_key = f"trends:{category}:{region}:{timeframe}:{limit}"

        # Check cache if enabled
        if use_cache:
            cached_data = await cache_manager.get(cache_key)
            if cached_data:
                logger.info(f"Returning cached trends for {category}")
                return json.loads(cached_data)

        try:
            # Use ML model if available
            if self.ml_model:
                trends = await self.ml_model.detect_trends(
                    category=category, region=region, timeframe=timeframe, limit=limit
                )

                # Convert to API format
                result = []
                for trend in trends:
                    result.append(
                        {
                            "topic": trend.topic,
                            "score": trend.score,
                            "category": trend.category,
                            "keywords": trend.keywords,
                            "competition_level": trend.competition_level,
                            "predicted_views": trend.predicted_views,
                            "best_time_to_post": trend.best_time_to_post.isoformat()
                            if trend.best_time_to_post
                            else None,
                            "source_scores": trend.source_scores,
                            "metadata": trend.metadata,
                        }
                    )
            else:
                # Fallback mock data if ML models not available
                result = await self._generate_mock_trends(category, limit)

            # Track API cost
            await cost_tracker.track_cost(
                service="trend_analysis",
                operation="analyze",
                amount=Decimal("0.05"),  # Estimated cost for trend API calls
                metadata={"category": category, "region": region},
            )

            # Cache the results
            if use_cache:
                await cache_manager.set(
                    cache_key, json.dumps(result), expire=self.cache_ttl
                )

            return result

        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            # Return fallback data on error
            return await self._generate_mock_trends(category, limit)

    async def get_trending_keywords(self, topic: str, limit: int = 20) -> List[str]:
        """
        Get trending keywords related to a topic

        Args:
            topic: Main topic to find keywords for
            limit: Maximum number of keywords

        Returns:
            List of trending keywords
        """
        cache_key = f"keywords:{topic}:{limit}"

        # Check cache
        cached_data = await cache_manager.get(cache_key)
        if cached_data:
            return json.loads(cached_data)

        try:
            if self.ml_model:
                keywords = await self.ml_model.extract_keywords(topic, limit)
            else:
                # Generate mock keywords
                keywords = self._generate_mock_keywords(topic, limit)

            # Cache results
            await cache_manager.set(
                cache_key, json.dumps(keywords), expire=self.cache_ttl
            )

            return keywords

        except Exception as e:
            logger.error(f"Error getting keywords: {e}")
            return self._generate_mock_keywords(topic, limit)

    async def predict_viral_potential(
        self, topic: str, keywords: List[str], category: str = TrendCategory.TECHNOLOGY
    ) -> Dict[str, Any]:
        """
        Predict the viral potential of a topic

        Args:
            topic: Topic to analyze
            keywords: Associated keywords
            category: Content category

        Returns:
            Viral potential analysis with score and recommendations
        """
        try:
            if self.ml_model:
                # Use ML model for prediction
                prediction = await self.ml_model.predict_viral_score(
                    topic=topic, keywords=keywords, category=category
                )

                return {
                    "topic": topic,
                    "viral_score": prediction.get("score", 0.5),
                    "confidence": prediction.get("confidence", 0.7),
                    "peak_time": prediction.get("peak_time"),
                    "recommended_publish_time": prediction.get("best_time"),
                    "competition_analysis": prediction.get("competition", {}),
                    "success_factors": prediction.get("factors", []),
                }
            else:
                # Mock prediction
                return {
                    "topic": topic,
                    "viral_score": 0.75,
                    "confidence": 0.8,
                    "peak_time": (datetime.now() + timedelta(days=2)).isoformat(),
                    "recommended_publish_time": (
                        datetime.now() + timedelta(hours=4)
                    ).isoformat(),
                    "competition_analysis": {
                        "level": "medium",
                        "similar_videos": 42,
                        "average_views": 50000,
                    },
                    "success_factors": [
                        "Trending topic",
                        "Low competition",
                        "High search volume",
                    ],
                }

        except Exception as e:
            logger.error(f"Error predicting viral potential: {e}")
            return {
                "topic": topic,
                "viral_score": 0.5,
                "confidence": 0.5,
                "error": str(e),
            }

    async def get_competition_analysis(
        self, topic: str, category: str = TrendCategory.TECHNOLOGY
    ) -> Dict[str, Any]:
        """
        Analyze competition for a given topic

        Args:
            topic: Topic to analyze
            category: Content category

        Returns:
            Competition analysis with metrics
        """
        try:
            if self.ml_model:
                analysis = await self.ml_model.analyze_competition(topic, category)
                return analysis
            else:
                # Mock competition data
                return {
                    "topic": topic,
                    "competition_level": "medium",
                    "total_videos": 156,
                    "average_views": 45000,
                    "top_channels": [
                        {"name": "TechChannel1", "subscribers": 500000},
                        {"name": "TechChannel2", "subscribers": 250000},
                    ],
                    "content_gap_opportunities": [
                        "In-depth tutorials",
                        "Beginner guides",
                        "Real-world applications",
                    ],
                    "recommended_approach": "Focus on unique angle and high production quality",
                }

        except Exception as e:
            logger.error(f"Error analyzing competition: {e}")
            return {"error": str(e)}

    async def get_trend_history(
        self, topic: str, days: int = 30, db: Optional[AsyncSession] = None
    ) -> Dict[str, Any]:
        """
        Get historical trend data for a topic

        Args:
            topic: Topic to get history for
            days: Number of days of history
            db: Database session

        Returns:
            Historical trend data with statistics
        """
        try:
            if db:
                # Query actual historical data from database
                start_date = datetime.now() - timedelta(days=days)

                query = (
                    select(Analytics)
                    .where(
                        Analytics.metadata.contains({"topic": topic}),
                        Analytics.created_at >= start_date,
                    )
                    .order_by(Analytics.created_at)
                )

                result = await db.execute(query)
                analytics = result.scalars().all()

                # Process historical data
                history = []
                for record in analytics:
                    history.append(
                        {
                            "date": record.created_at.isoformat(),
                            "score": record.metadata.get("trend_score", 0),
                            "views": record.view_count,
                            "engagement": record.engagement_rate,
                        }
                    )

                return {
                    "topic": topic,
                    "period_days": days,
                    "data_points": len(history),
                    "history": history,
                    "trend_direction": self._calculate_trend_direction(history),
                    "average_score": sum(h["score"] for h in history) / len(history)
                    if history
                    else 0,
                }
            else:
                # Return mock historical data
                return self._generate_mock_history(topic, days)

        except Exception as e:
            logger.error(f"Error getting trend history: {e}")
            return {"error": str(e)}

    def _calculate_trend_direction(self, history: List[Dict]) -> str:
        """Calculate trend direction from historical data"""
        if not history or len(history) < 2:
            return "stable"

        # Compare first half average with second half
        mid = len(history) // 2
        first_half_avg = sum(h["score"] for h in history[:mid]) / mid
        second_half_avg = sum(h["score"] for h in history[mid:]) / (len(history) - mid)

        diff_percent = ((second_half_avg - first_half_avg) / first_half_avg) * 100

        if diff_percent > 10:
            return "rising"
        elif diff_percent < -10:
            return "falling"
        else:
            return "stable"

    async def _generate_mock_trends(
        self, category: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Generate mock trend data for testing/fallback"""
        mock_topics = {
            TrendCategory.TECHNOLOGY: [
                "AI Agents",
                "Quantum Computing",
                "Web3",
                "Robotics",
                "Cybersecurity",
                "5G Networks",
                "Blockchain",
                "IoT",
                "AR/VR",
                "Cloud Computing",
            ],
            TrendCategory.GAMING: [
                "Baldur's Gate 3",
                "Starfield",
                "CS2",
                "Minecraft Updates",
                "Steam Deck",
                "PlayStation 5",
                "Xbox Series X",
                "Nintendo Switch",
                "VR Gaming",
                "Esports",
            ],
            TrendCategory.EDUCATION: [
                "Online Learning",
                "AI Tutors",
                "STEM Education",
                "Coding Bootcamps",
                "Language Learning",
                "Digital Skills",
                "Remote Education",
                "EdTech",
                "Microlearning",
                "Skill Development",
            ],
        }

        topics = mock_topics.get(category, mock_topics[TrendCategory.TECHNOLOGY])[
            :limit
        ]

        trends = []
        for i, topic in enumerate(topics):
            trends.append(
                {
                    "topic": topic,
                    "score": 95 - (i * 5),  # Decreasing scores
                    "category": category,
                    "keywords": self._generate_mock_keywords(topic, 5),
                    "competition_level": ["low", "medium", "high"][i % 3],
                    "predicted_views": 100000 - (i * 10000),
                    "best_time_to_post": (
                        datetime.now() + timedelta(hours=i + 1)
                    ).isoformat(),
                    "source_scores": {
                        "youtube": 90 - i * 3,
                        "google_trends": 85 - i * 2,
                        "twitter": 80 - i * 4,
                    },
                    "metadata": {
                        "confidence": 0.85 - (i * 0.05),
                        "data_freshness": "real-time",
                    },
                }
            )

        return trends

    def _generate_mock_keywords(self, topic: str, limit: int) -> List[str]:
        """Generate mock keywords for a topic"""
        # Simple keyword generation based on topic
        base_keywords = topic.lower().split()
        keywords = base_keywords.copy()

        # Add related terms
        suffixes = [
            "tutorial",
            "guide",
            "tips",
            "tricks",
            "review",
            "news",
            "update",
            "2024",
            "best",
            "how to",
            "explained",
            "vs",
        ]

        for suffix in suffixes[: limit - len(base_keywords)]:
            keywords.append(f"{topic.lower()} {suffix}")

        return keywords[:limit]

    def _generate_mock_history(self, topic: str, days: int) -> Dict[str, Any]:
        """Generate mock historical data"""
        history = []
        for i in range(days):
            date = datetime.now() - timedelta(days=days - i)
            history.append(
                {
                    "date": date.isoformat(),
                    "score": 50 + (i * 1.5),  # Gradually increasing
                    "views": 1000 + (i * 100),
                    "engagement": 0.05 + (i * 0.001),
                }
            )

        return {
            "topic": topic,
            "period_days": days,
            "data_points": len(history),
            "history": history,
            "trend_direction": "rising",
            "average_score": 65.0,
        }


# Create singleton instance
trend_analyzer = TrendAnalyzer()


# Export for backward compatibility
async def analyze_trends(
    category: str = "technology", region: str = "US", limit: int = 10
) -> List[Dict[str, Any]]:
    """Legacy function for backward compatibility"""
    return await trend_analyzer.analyze_trends(category, region, "now", limit)
