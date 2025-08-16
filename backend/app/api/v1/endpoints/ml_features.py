"""
ML Features API Endpoints
Exposes AutoML and Personalization capabilities via REST API
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from datetime import datetime

from app.db.session import get_db
from app.api.v1.endpoints.auth import get_current_verified_user as get_current_user
from app.models.user import User
from app.services.ml_integration_service import ml_service
from app.services.video_generation_pipeline import (
    generate_personalized_video,
    generate_batch_videos_with_ml,
)

router = APIRouter(prefix="/ml", tags=["ML Features"])


# Request/Response Models
class PersonalizationRequest(BaseModel):
    """Request model for content personalization"""

    channel_id: str = Field(..., description="Channel ID for personalization")
    trending_topics: Optional[List[str]] = Field(
        None, description="Optional trending topics to consider"
    )
    use_history: bool = Field(True, description="Whether to use historical data")


class PersonalizationResponse(BaseModel):
    """Response model for personalized content"""

    title: str
    script_template: str
    keywords: List[str]
    tone: str
    style: str
    estimated_engagement: float
    confidence_score: float
    reasoning: str
    personalization_factors: Dict[str, float]


class PerformancePredictionRequest(BaseModel):
    """Request model for performance prediction"""

    title_length: int = Field(..., description="Number of words in title")
    description_length: int = Field(..., description="Number of words in description")
    keyword_count: int = Field(..., description="Number of keywords")
    trending_score: float = Field(
        0.5, ge=0, le=1, description="Trending relevance score"
    )
    channel_subscriber_count: int = Field(
        1000, ge=0, description="Channel subscriber count"
    )
    channel_video_count: int = Field(10, ge=0, description="Total videos on channel")
    posting_hour: int = Field(14, ge=0, le=23, description="Hour of posting (0-23)")
    posting_day: int = Field(
        4, ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)"
    )
    video_duration: int = Field(600, ge=60, description="Video duration in seconds")


class PerformancePredictionResponse(BaseModel):
    """Response model for performance prediction"""

    predicted_views: int
    predicted_engagement_rate: float
    confidence_score: float
    model_type: str
    prediction_factors: Dict[str, float]


class ChannelInsightsResponse(BaseModel):
    """Response model for channel insights"""

    channel_id: str
    content_style: Optional[str]
    performance: Dict[str, Any]
    optimal_schedule: Dict[str, Any]
    content_preferences: Dict[str, Any]
    profile_confidence: float
    next_video_prediction: Optional[Dict[str, Any]]


class TrainingRequest(BaseModel):
    """Request model for model training"""

    min_videos: int = Field(
        100, ge=10, description="Minimum videos required for training"
    )
    force_retrain: bool = Field(
        False, description="Force retraining even if not needed"
    )


class TrainingResponse(BaseModel):
    """Response model for training status"""

    status: str
    best_model: Optional[str]
    score: Optional[float]
    training_samples: Optional[int]
    feature_importance: Optional[Dict[str, float]]
    message: Optional[str]


class MLVideoGenerationRequest(BaseModel):
    """Request for ML-enhanced video generation"""

    channel_id: str
    topic: Optional[str] = None
    use_personalization: bool = True
    use_performance_prediction: bool = True


class BatchMLVideoRequest(BaseModel):
    """Request for batch ML video generation"""

    channel_ids: List[str]
    topics: Optional[List[str]] = None
    optimize_order: bool = True


# Endpoints
@router.post("/personalize", response_model=PersonalizationResponse)
async def get_personalized_content(
    request: PersonalizationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> PersonalizationResponse:
    """
    Get personalized content recommendations for a channel
    """
    try:
        # Get channel data (simplified for now)
        channel_data = {
            "name": f"Channel_{request.channel_id}",
            "niche": "technology",  # Should fetch from database
            "target_audience": {"age_range": "18-35"},
        }

        # Get historical videos if requested
        historical_videos = []
        if request.use_history:
            # Should fetch from database
            historical_videos = [
                {
                    "title": "Sample Video",
                    "views": 10000,
                    "likes": 500,
                    "comments": 50,
                    "duration": 600,
                    "published_at": datetime.now().isoformat(),
                }
            ]

        # Get personalized content
        result = await ml_service.get_personalized_content_recommendation(
            channel_id=request.channel_id,
            channel_data=channel_data,
            historical_videos=historical_videos,
            trending_topics=request.trending_topics,
            db=db,
        )

        return PersonalizationResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-performance", response_model=PerformancePredictionResponse)
async def predict_video_performance(
    request: PerformancePredictionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> PerformancePredictionResponse:
    """
    Predict video performance based on features
    """
    try:
        # Convert request to feature dict
        features = request.dict()

        # Get prediction
        result = await ml_service.predict_video_performance(features, db)

        return PerformancePredictionResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/channel-insights/{channel_id}", response_model=ChannelInsightsResponse)
async def get_channel_insights(
    channel_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ChannelInsightsResponse:
    """
    Get ML-powered insights for a channel
    """
    try:
        insights = await ml_service.get_channel_insights(channel_id, db)

        if "error" in insights:
            raise HTTPException(status_code=404, detail=insights["error"])

        return ChannelInsightsResponse(**insights)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train-model", response_model=TrainingResponse)
async def train_performance_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TrainingResponse:
    """
    Train or retrain the AutoML performance prediction model
    """
    try:
        # Check if retraining is needed
        if not request.force_retrain:
            retrain_status = await ml_service.check_retraining_needed()
            if not retrain_status["automl_needs_retraining"]:
                return TrainingResponse(
                    status="not_needed",
                    message="Model is up to date, no retraining needed",
                )

        # Train model (could be done in background for large datasets)
        result = await ml_service.train_performance_model(db, request.min_videos)

        return TrainingResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-status")
async def get_model_status(
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get current status of ML models
    """
    try:
        retrain_status = await ml_service.check_retraining_needed()

        # Get AutoML status
        automl_status = {
            "available": ml_service.automl_pipeline is not None,
            "has_trained_model": (
                ml_service.automl_pipeline.best_model is not None
                if ml_service.automl_pipeline
                else False
            ),
            "needs_retraining": retrain_status["automl_needs_retraining"],
        }

        if ml_service.automl_pipeline and ml_service.automl_pipeline.best_model:
            automl_status.update(ml_service.automl_pipeline.get_model_summary())

        # Get Personalization status
        personalization_status = {
            "available": ml_service.personalization_engine is not None,
            "profiles_count": (
                len(ml_service.personalization_engine.channel_profiles)
                if ml_service.personalization_engine
                else 0
            ),
            "needs_update": retrain_status["personalization_needs_update"],
        }

        return {
            "automl": automl_status,
            "personalization": personalization_status,
            "ml_available": ml_service.automl_pipeline is not None,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-video", response_model=Dict[str, Any])
async def generate_ml_video(
    request: MLVideoGenerationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Generate a video with ML enhancements
    """
    try:
        result = await generate_personalized_video(
            channel_id=request.channel_id, topic=request.topic, db=db
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-generate", response_model=List[Dict[str, Any]])
async def batch_generate_ml_videos(
    request: BatchMLVideoRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """
    Generate multiple videos with ML optimization
    """
    try:
        # This could be done in background for large batches
        results = await generate_batch_videos_with_ml(
            channel_ids=request.channel_ids, topics=request.topics, db=db
        )

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update-profile/{channel_id}")
async def update_channel_profile(
    channel_id: str,
    video_id: str,
    performance_data: Dict[str, Any],
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, str]:
    """
    Update channel ML profile with new performance data
    """
    try:
        success = await ml_service.update_channel_profile(
            channel_id=channel_id,
            video_id=video_id,
            performance_data=performance_data,
            db=db,
        )

        if success:
            return {"status": "success", "message": "Profile updated"}
        else:
            raise HTTPException(status_code=500, detail="Failed to update profile")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations/{channel_id}")
async def get_content_recommendations(
    channel_id: str,
    limit: int = 5,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """
    Get multiple content recommendations for a channel
    """
    try:
        recommendations = []

        # Generate multiple recommendations with different trending topics
        trending_topics_sets = [
            ["AI", "Machine Learning"],
            ["Technology", "Innovation"],
            ["Tutorial", "Education"],
            ["Review", "Comparison"],
            ["News", "Updates"],
        ]

        for topics in trending_topics_sets[:limit]:
            channel_data = {"name": f"Channel_{channel_id}", "niche": "technology"}

            rec = await ml_service.get_personalized_content_recommendation(
                channel_id=channel_id,
                channel_data=channel_data,
                historical_videos=[],
                trending_topics=topics,
                db=db,
            )

            rec["trending_topics"] = topics
            recommendations.append(rec)

        # Sort by estimated engagement
        recommendations.sort(key=lambda x: x["estimated_engagement"], reverse=True)

        return recommendations[:limit]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
