"""
ML Integration Service
Integrates AutoML and Personalization models with the existing YTEmpire backend
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

# Import our new ML models
import sys
import os

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "ml-pipeline", "src")
)

try:
    from automl_pipeline import (
        AutoMLPipeline,
        AutoMLConfig,
        OptimizationMetric,
        ModelType,
    )
    from personalization_model import (
        PersonalizationEngine,
        PersonalizationConfig,
        PersonalizationType,
        ContentStyle,
    )

    ML_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ML models not available: {e}")
    ML_AVAILABLE = False

try:
    from app.models.video import Video, VideoStatus
    from app.models.channel import Channel
    from app.models.cost import Cost
    from app.services.cost_tracking import cost_tracker
    from app.core.config import settings

    BACKEND_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Backend modules not available: {e}")
    BACKEND_AVAILABLE = False

    # Mock classes for testing
    class Video:
        pass

    class VideoStatus:
        PUBLISHED = "published"

    class Channel:
        pass

    class Cost:
        pass

    class MockCostTracker:
        async def track_cost(self, *args, **kwargs):
            pass

    cost_tracker = MockCostTracker()

    class MockSettings:
        ML_MODELS_PATH = "models"
        ML_ENABLED = True
        AUTOML_RETRAIN_DAYS = 7
        PERSONALIZATION_UPDATE_DAYS = 3

    settings = MockSettings()

logger = logging.getLogger(__name__)


class MLIntegrationService:
    """
    Service that integrates AutoML and Personalization with the YTEmpire backend
    """

    def __init__(self):
        self.automl_pipeline = None
        self.personalization_engine = None
        self.models_path = Path("models")
        self.models_path.mkdir(exist_ok=True)

        if ML_AVAILABLE:
            self._initialize_ml_models()
        else:
            logger.warning("ML models not available, using fallback methods")

    def _initialize_ml_models(self):
        """Initialize ML models with configurations"""
        try:
            # Initialize AutoML for performance prediction
            automl_config = AutoMLConfig(
                task_type="regression",
                optimization_metric=OptimizationMetric.R2,
                test_size=0.2,
                cv_folds=5,
                enable_feature_engineering=True,
                enable_ensemble=True,
                auto_retrain_days=7,
                min_performance_threshold=0.75,
                save_path=self.models_path / "automl",
            )
            self.automl_pipeline = AutoMLPipeline(automl_config)

            # Load existing model if available
            automl_model_path = self.models_path / "automl" / "production_model.pkl"
            if automl_model_path.exists():
                self.automl_pipeline.load_model(automl_model_path)
                logger.info("Loaded existing AutoML model")

            # Initialize Personalization Engine
            personalization_config = PersonalizationConfig(
                model_type=PersonalizationType.HYBRID,
                embedding_dim=128,
                n_clusters=10,
                min_videos_for_training=5,
                update_frequency_days=3,
                save_path=self.models_path / "personalization",
            )
            self.personalization_engine = PersonalizationEngine(personalization_config)

            # Load existing profiles if available
            profiles_path = (
                self.models_path / "personalization" / "channel_profiles.pkl"
            )
            if profiles_path.exists():
                self.personalization_engine.load_profiles(profiles_path)
                logger.info("Loaded existing channel profiles")

            logger.info("ML models initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
            self.automl_pipeline = None
            self.personalization_engine = None

    async def get_personalized_content_recommendation(
        self,
        channel_id: str,
        channel_data: Dict[str, Any],
        historical_videos: List[Dict[str, Any]],
        trending_topics: Optional[List[str]] = None,
        db: Optional[AsyncSession] = None,
    ) -> Dict[str, Any]:
        """
        Get personalized content recommendations for a channel
        """
        if not ML_AVAILABLE or not self.personalization_engine:
            return self._get_fallback_recommendation(channel_data, trending_topics)

        try:
            # Create or update channel profile
            if channel_id not in self.personalization_engine.channel_profiles:
                profile = self.personalization_engine.create_channel_profile(
                    channel_id, channel_data, historical_videos
                )
                logger.info(f"Created new profile for channel {channel_id}")
            else:
                # Update with latest videos
                for video in historical_videos[-5:]:
                    self.personalization_engine.update_profile_with_feedback(
                        channel_id, video.get("id", "unknown"), video
                    )

            # Generate personalized recommendation
            recommendation = self.personalization_engine.generate_personalized_content(
                channel_id, trending_topics=trending_topics
            )

            # Track cost
            await cost_tracker.track_cost(
                service="personalization",
                operation="content_recommendation",
                cost=Decimal("0.01"),  # Minimal cost for inference
                metadata={"channel_id": channel_id},
            )

            return {
                "title": recommendation.title,
                "script_template": recommendation.script_template,
                "keywords": recommendation.keywords,
                "tone": recommendation.tone,
                "style": recommendation.style,
                "estimated_engagement": recommendation.estimated_engagement,
                "confidence_score": recommendation.confidence_score,
                "reasoning": recommendation.reasoning,
                "personalization_factors": recommendation.personalization_factors,
            }

        except Exception as e:
            logger.error(f"Error getting personalized recommendation: {e}")
            return self._get_fallback_recommendation(channel_data, trending_topics)

    async def predict_video_performance(
        self, video_features: Dict[str, Any], db: Optional[AsyncSession] = None
    ) -> Dict[str, Any]:
        """
        Predict video performance using AutoML
        """
        if (
            not ML_AVAILABLE
            or not self.automl_pipeline
            or not self.automl_pipeline.best_model
        ):
            return self._get_fallback_performance_prediction(video_features)

        try:
            # Prepare features for prediction
            feature_df = pd.DataFrame([video_features])

            # Ensure all required features are present
            required_features = [
                "title_length",
                "description_length",
                "keyword_count",
                "trending_score",
                "channel_subscriber_count",
                "channel_video_count",
                "posting_hour",
                "posting_day",
                "video_duration",
            ]

            for feature in required_features:
                if feature not in feature_df.columns:
                    feature_df[feature] = 0

            # Make prediction
            predictions = self.automl_pipeline.predict(feature_df)
            predicted_engagement = predictions[0]

            # Get model confidence
            model_summary = self.automl_pipeline.get_model_summary()

            # Track cost
            await cost_tracker.track_cost(
                service="automl",
                operation="performance_prediction",
                cost=Decimal("0.005"),  # Minimal cost for inference
                metadata={"features": len(video_features)},
            )

            return {
                "predicted_views": int(predicted_engagement * 100000),  # Scale to views
                "predicted_engagement_rate": float(predicted_engagement),
                "confidence_score": model_summary["current_model"].get(
                    "validation_score", 0.5
                ),
                "model_type": model_summary["current_model"].get("type", "unknown"),
                "prediction_factors": {
                    "trending_impact": video_features.get("trending_score", 0) * 0.3,
                    "channel_strength": min(
                        video_features.get("channel_subscriber_count", 0) / 100000, 1.0
                    )
                    * 0.2,
                    "content_quality": 0.5,  # Placeholder
                },
            }

        except Exception as e:
            logger.error(f"Error predicting performance: {e}")
            return self._get_fallback_performance_prediction(video_features)

    async def train_performance_model(
        self, db: AsyncSession, min_videos: int = 100
    ) -> Dict[str, Any]:
        """
        Train or retrain the AutoML performance prediction model
        """
        if not ML_AVAILABLE or not self.automl_pipeline:
            return {"status": "ML not available"}

        try:
            # Fetch historical video data
            query = (
                select(Video).where(Video.status == VideoStatus.PUBLISHED).limit(1000)
            )

            result = await db.execute(query)
            videos = result.scalars().all()

            if len(videos) < min_videos:
                return {
                    "status": "insufficient_data",
                    "videos_found": len(videos),
                    "min_required": min_videos,
                }

            # Prepare training data
            features = []
            targets = []

            for video in videos:
                # Extract features
                video_features = {
                    "title_length": len(video.title.split()) if video.title else 0,
                    "description_length": len(video.description.split())
                    if video.description
                    else 0,
                    "keyword_count": len(video.keywords) if video.keywords else 0,
                    "trending_score": video.metadata.get("trending_score", 0)
                    if video.metadata
                    else 0,
                    "channel_subscriber_count": 10000,  # Default, should fetch from channel
                    "channel_video_count": 50,  # Default
                    "posting_hour": video.created_at.hour if video.created_at else 12,
                    "posting_day": video.created_at.weekday()
                    if video.created_at
                    else 0,
                    "video_duration": video.duration if video.duration else 600,
                }
                features.append(video_features)

                # Calculate engagement rate as target
                views = video.views or 1
                engagement = ((video.likes or 0) + (video.comments or 0)) / views
                targets.append(engagement)

            # Convert to DataFrame
            X = pd.DataFrame(features)
            y = pd.Series(targets)

            # Train model
            logger.info(f"Training AutoML model with {len(X)} samples")
            results = await self.automl_pipeline.train_async(X, y)

            # Save model
            model_path = self.models_path / "automl" / "production_model.pkl"
            self.automl_pipeline.save_model(model_path)

            # Track training cost
            await cost_tracker.track_cost(
                service="automl",
                operation="model_training",
                cost=Decimal("0.10"),  # Estimated compute cost
                metadata={"samples": len(X), "features": len(X.columns)},
            )

            return {
                "status": "success",
                "best_model": results["best_model_type"],
                "score": results["best_score"],
                "models_evaluated": len(results["all_performances"]),
                "training_samples": len(X),
                "feature_importance": results.get("feature_importance", {}),
            }

        except Exception as e:
            logger.error(f"Error training model: {e}")
            return {"status": "error", "message": str(e)}

    async def update_channel_profile(
        self,
        channel_id: str,
        video_id: str,
        performance_data: Dict[str, Any],
        db: Optional[AsyncSession] = None,
    ) -> bool:
        """
        Update channel profile with new video performance data
        """
        if not ML_AVAILABLE or not self.personalization_engine:
            return False

        try:
            self.personalization_engine.update_profile_with_feedback(
                channel_id, video_id, performance_data
            )

            # Save updated profiles
            profiles_path = (
                self.models_path / "personalization" / "channel_profiles.pkl"
            )
            self.personalization_engine.save_profiles(profiles_path)

            return True

        except Exception as e:
            logger.error(f"Error updating channel profile: {e}")
            return False

    async def get_channel_insights(
        self, channel_id: str, db: Optional[AsyncSession] = None
    ) -> Dict[str, Any]:
        """
        Get ML-powered insights for a channel
        """
        if not ML_AVAILABLE or not self.personalization_engine:
            return self._get_fallback_channel_insights(channel_id)

        try:
            insights = self.personalization_engine.get_channel_insights(channel_id)

            # Add AutoML predictions if available
            if self.automl_pipeline and self.automl_pipeline.best_model:
                # Predict next video performance
                next_video_features = {
                    "title_length": 10,
                    "description_length": 100,
                    "keyword_count": 5,
                    "trending_score": 0.7,
                    "channel_subscriber_count": insights.get("performance", {}).get(
                        "avg_views", 10000
                    )
                    / 10,
                    "channel_video_count": insights.get("recent_videos", 10),
                    "posting_hour": insights.get("optimal_schedule", {}).get(
                        "best_hour", 12
                    ),
                    "posting_day": insights.get("optimal_schedule", {}).get(
                        "best_day", 4
                    ),
                    "video_duration": insights.get("content_preferences", {}).get(
                        "preferred_duration", 600
                    ),
                }

                performance_prediction = await self.predict_video_performance(
                    next_video_features
                )
                insights["next_video_prediction"] = performance_prediction

            return insights

        except Exception as e:
            logger.error(f"Error getting channel insights: {e}")
            return self._get_fallback_channel_insights(channel_id)

    async def check_retraining_needed(self) -> Dict[str, bool]:
        """
        Check if any models need retraining
        """
        results = {
            "automl_needs_retraining": False,
            "personalization_needs_update": False,
        }

        if ML_AVAILABLE:
            if self.automl_pipeline:
                results[
                    "automl_needs_retraining"
                ] = self.automl_pipeline.should_retrain()

            # Check if personalization profiles are stale
            if self.personalization_engine:
                for profile in self.personalization_engine.channel_profiles.values():
                    if (datetime.now() - profile.updated_at).days > 7:
                        results["personalization_needs_update"] = True
                        break

        return results

    def _get_fallback_recommendation(
        self, channel_data: Dict[str, Any], trending_topics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Fallback recommendation when ML is not available"""
        topic = (
            trending_topics[0]
            if trending_topics
            else channel_data.get("niche", "general")
        )

        return {
            "title": f"Amazing {topic} Content You Need to See",
            "script_template": "Introduction, Main Content, Conclusion",
            "keywords": [topic, "tutorial", "guide", "tips", "2024"],
            "tone": "informative",
            "style": "educational",
            "estimated_engagement": 0.05,
            "confidence_score": 0.3,
            "reasoning": "Using fallback recommendation (ML not available)",
            "personalization_factors": {"default": 1.0},
        }

    def _get_fallback_performance_prediction(
        self, video_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback performance prediction when ML is not available"""
        base_views = 10000
        trending_boost = video_features.get("trending_score", 0.5) * 5000

        return {
            "predicted_views": int(base_views + trending_boost),
            "predicted_engagement_rate": 0.05,
            "confidence_score": 0.3,
            "model_type": "fallback",
            "prediction_factors": {
                "trending_impact": 0.3,
                "channel_strength": 0.3,
                "content_quality": 0.4,
            },
        }

    def _get_fallback_channel_insights(self, channel_id: str) -> Dict[str, Any]:
        """Fallback insights when ML is not available"""
        return {
            "channel_id": channel_id,
            "status": "ML not available",
            "performance": {"avg_views": 10000, "avg_engagement": 0.05},
            "optimal_schedule": {
                "best_hour": 14,
                "best_day": 4,
                "posting_frequency": "weekly",
            },
            "content_preferences": {
                "top_keywords": ["tutorial", "guide", "tips"],
                "preferred_duration": 600,
            },
        }


# Singleton instance
ml_service = MLIntegrationService()


# Async helper functions for use in other services
async def get_personalized_video_content(
    channel_id: str,
    channel_data: Dict[str, Any],
    historical_videos: List[Dict[str, Any]],
    trending_topics: Optional[List[str]] = None,
    db: Optional[AsyncSession] = None,
) -> Dict[str, Any]:
    """Helper function to get personalized content"""
    return await ml_service.get_personalized_content_recommendation(
        channel_id, channel_data, historical_videos, trending_topics, db
    )


async def predict_video_performance(
    video_features: Dict[str, Any], db: Optional[AsyncSession] = None
) -> Dict[str, Any]:
    """Helper function to predict video performance"""
    return await ml_service.predict_video_performance(video_features, db)


async def update_channel_ml_profile(
    channel_id: str,
    video_id: str,
    performance_data: Dict[str, Any],
    db: Optional[AsyncSession] = None,
) -> bool:
    """Helper function to update channel ML profile"""
    return await ml_service.update_channel_profile(
        channel_id, video_id, performance_data, db
    )
