"""
Training Pipeline Service
Integrates ML Training Pipeline with backend services
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add ML pipeline path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data'))

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.config import settings
from app.models.video import Video
from app.models.channel import Channel
# from app.models.analytics import VideoAnalytics  # Use the actual analytics model if available

# Import ML Training Pipeline
try:
    from ml.ml_training_pipeline import (
        MLTrainingPipeline,
        TrainingConfig,
        TrainingResult,
        ModelType
    )
    ML_TRAINING_AVAILABLE = True
except ImportError:
    ML_TRAINING_AVAILABLE = False
    logging.warning("ML Training Pipeline not available")

logger = logging.getLogger(__name__)


class TrainingPipelineService:
    """Service for managing ML model training pipelines"""
    
    def __init__(self):
        self.pipeline = None
        self.is_initialized = False
        
        if ML_TRAINING_AVAILABLE:
            self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the ML training pipeline"""
        try:
            self.pipeline = MLTrainingPipeline(
                database_url=settings.DATABASE_URL,
                redis_url=getattr(settings, 'REDIS_URL', None),
                mlflow_tracking_uri=getattr(settings, 'MLFLOW_URI', None),
                storage_path=getattr(settings, 'ML_MODELS_PATH', 'models')
            )
            logger.info("ML Training Pipeline initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ML Training Pipeline: {e}")
            self.pipeline = None
    
    async def initialize(self):
        """Initialize async components"""
        if self.pipeline and not self.is_initialized:
            try:
                await self.pipeline.initialize()
                self.is_initialized = True
                logger.info("ML Training Pipeline async components initialized")
            except Exception as e:
                logger.error(f"Failed to initialize async components: {e}")
    
    async def train_video_performance_model(
        self,
        db: AsyncSession,
        min_videos: int = 100,
        force_retrain: bool = False
    ) -> Dict[str, Any]:
        """
        Train video performance prediction model
        """
        if not self.pipeline:
            return {
                "status": "error",
                "message": "ML Training Pipeline not available"
            }
        
        await self.initialize()
        
        try:
            # Check if we have enough data
            video_count = await db.scalar(select(func.count(Video.id)))
            
            if video_count < min_videos and not force_retrain:
                return {
                    "status": "insufficient_data",
                    "message": f"Need at least {min_videos} videos, have {video_count}"
                }
            
            # Prepare training data from database
            query = select(
                Video.id.label("video_id"),
                Video.title,
                Video.description,
                Video.duration,
                Video.published_at,
                Channel.subscriber_count,
                Channel.video_count
                # VideoAnalytics.views,
                # VideoAnalytics.likes,
                # VideoAnalytics.comments,
                # VideoAnalytics.engagement_rate
            ).join(
                Channel, Video.channel_id == Channel.id
            # ).join(
            #     VideoAnalytics, Video.id == VideoAnalytics.video_id
            # ).where(
            #     VideoAnalytics.views > 0
            )
            
            result = await db.execute(query)
            data = result.fetchall()
            
            if not data:
                return {
                    "status": "no_data",
                    "message": "No training data available"
                }
            
            # Create training configuration
            config = TrainingConfig(
                model_name="video_performance_predictor",
                model_type=ModelType.REGRESSION,
                algorithm="gradient_boosting_regressor",
                data_source={
                    "type": "database",
                    "query": str(query)
                },
                feature_columns=[
                    "title_length",
                    "description_length",
                    "duration",
                    "publish_hour",
                    "publish_day",
                    "channel_subscribers",
                    "channel_videos"
                ],
                target_column="views",
                test_size=0.2,
                validation_size=0.1,
                hyperparameter_tuning=True,
                param_grid={
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.05, 0.1, 0.2],
                    "max_depth": [3, 5, 7]
                },
                auto_deploy=True,
                deployment_target="production" if settings.ENVIRONMENT == "production" else "staging",
                min_performance_threshold=0.75,
                experiment_name="video_performance",
                tags={
                    "version": "v1",
                    "trained_by": "training_pipeline_service"
                }
            )
            
            # Train model
            result: TrainingResult = await self.pipeline.train_model(config)
            
            return {
                "status": "success" if result.success else "failed",
                "model_id": result.model_id,
                "model_version": result.model_version,
                "metrics": result.test_metrics,
                "deployed": result.deployed,
                "deployment_endpoint": result.deployment_endpoint,
                "training_duration": result.training_duration
            }
            
        except Exception as e:
            logger.error(f"Failed to train video performance model: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def train_content_quality_model(
        self,
        db: AsyncSession,
        min_videos: int = 100
    ) -> Dict[str, Any]:
        """
        Train content quality prediction model
        """
        if not self.pipeline:
            return {
                "status": "error",
                "message": "ML Training Pipeline not available"
            }
        
        await self.initialize()
        
        try:
            # Create training configuration for content quality
            config = TrainingConfig(
                model_name="content_quality_predictor",
                model_type=ModelType.CLASSIFICATION,
                algorithm="random_forest_classifier",
                data_source={
                    "type": "database",
                    "table": "video_quality_metrics"
                },
                feature_columns=[
                    "script_coherence",
                    "keyword_density",
                    "sentiment_score",
                    "readability_score",
                    "topic_relevance"
                ],
                target_column="quality_category",
                test_size=0.2,
                validation_size=0.1,
                hyperparameter_tuning=True,
                param_grid={
                    "n_estimators": [100, 200],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5]
                },
                auto_deploy=True,
                min_performance_threshold=0.85,
                experiment_name="content_quality",
                tags={
                    "version": "v1",
                    "model_type": "quality"
                }
            )
            
            # Train model
            result = await self.pipeline.train_model(config)
            
            return {
                "status": "success" if result.success else "failed",
                "model_id": result.model_id,
                "metrics": result.test_metrics,
                "deployed": result.deployed
            }
            
        except Exception as e:
            logger.error(f"Failed to train content quality model: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def schedule_periodic_training(
        self,
        model_name: str,
        cron_expression: str,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Schedule periodic model training
        """
        if not self.pipeline:
            return {
                "status": "error",
                "message": "ML Training Pipeline not available"
            }
        
        await self.initialize()
        
        try:
            # Get existing model configuration
            if model_name == "video_performance_predictor":
                config = self._get_performance_model_config()
            elif model_name == "content_quality_predictor":
                config = self._get_quality_model_config()
            else:
                return {
                    "status": "error",
                    "message": f"Unknown model: {model_name}"
                }
            
            # Schedule training
            await self.pipeline.schedule_training(config, cron_expression)
            
            return {
                "status": "scheduled",
                "model_name": model_name,
                "schedule": cron_expression
            }
            
        except Exception as e:
            logger.error(f"Failed to schedule training: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def trigger_model_retraining(
        self,
        model_name: str,
        reason: str,
        priority: str = "normal"
    ) -> Dict[str, Any]:
        """
        Trigger model retraining
        """
        if not self.pipeline:
            return {
                "status": "error",
                "message": "ML Training Pipeline not available"
            }
        
        await self.initialize()
        
        try:
            await self.pipeline.trigger_retraining(
                model_name=model_name,
                reason=reason,
                priority=priority
            )
            
            return {
                "status": "triggered",
                "model_name": model_name,
                "reason": reason,
                "priority": priority
            }
            
        except Exception as e:
            logger.error(f"Failed to trigger retraining: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def monitor_model_performance(
        self,
        model_name: str,
        threshold: float = 0.75
    ) -> Dict[str, Any]:
        """
        Monitor model performance and trigger retraining if needed
        """
        if not self.pipeline:
            return {
                "status": "error",
                "message": "ML Training Pipeline not available"
            }
        
        await self.initialize()
        
        try:
            needs_retraining = await self.pipeline.monitor_model_performance(
                model_name=model_name,
                threshold=threshold
            )
            
            return {
                "status": "monitored",
                "model_name": model_name,
                "needs_retraining": needs_retraining,
                "threshold": threshold
            }
            
        except Exception as e:
            logger.error(f"Failed to monitor model: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def get_training_history(
        self,
        model_name: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get training history
        """
        if not self.pipeline:
            return []
        
        await self.initialize()
        
        try:
            history = await self.pipeline.get_training_history(
                model_name=model_name,
                limit=limit
            )
            return history
            
        except Exception as e:
            logger.error(f"Failed to get training history: {e}")
            return []
    
    async def get_scheduled_trainings(self) -> List[Dict[str, Any]]:
        """
        Get all scheduled training jobs
        """
        if not self.pipeline:
            return []
        
        await self.initialize()
        
        try:
            schedules = await self.pipeline.get_scheduled_trainings()
            return schedules
            
        except Exception as e:
            logger.error(f"Failed to get scheduled trainings: {e}")
            return []
    
    async def load_model(
        self,
        model_name: str,
        version: Optional[str] = None
    ) -> Optional[Any]:
        """
        Load a trained model
        """
        if not self.pipeline:
            return None
        
        await self.initialize()
        
        try:
            model, artifacts = await self.pipeline.load_model(
                model_name=model_name,
                version=version
            )
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    
    def _get_performance_model_config(self) -> TrainingConfig:
        """Get configuration for performance model"""
        return TrainingConfig(
            model_name="video_performance_predictor",
            model_type=ModelType.REGRESSION,
            algorithm="gradient_boosting_regressor",
            data_source={
                "type": "database",
                "table": "video_performance_features"
            },
            feature_columns=[
                "title_length", "description_length", "duration",
                "publish_hour", "publish_day", "channel_subscribers"
            ],
            target_column="views",
            hyperparameter_tuning=True,
            auto_deploy=True,
            min_performance_threshold=0.75
        )
    
    def _get_quality_model_config(self) -> TrainingConfig:
        """Get configuration for quality model"""
        return TrainingConfig(
            model_name="content_quality_predictor",
            model_type=ModelType.CLASSIFICATION,
            algorithm="random_forest_classifier",
            data_source={
                "type": "database",
                "table": "video_quality_metrics"
            },
            feature_columns=[
                "script_coherence", "keyword_density",
                "sentiment_score", "readability_score"
            ],
            target_column="quality_category",
            hyperparameter_tuning=True,
            auto_deploy=True,
            min_performance_threshold=0.85
        )


# Singleton instance
training_service = TrainingPipelineService()