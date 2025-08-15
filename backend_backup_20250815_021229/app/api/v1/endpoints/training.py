"""
Training Pipeline API Endpoints
Exposes ML model training and management capabilities
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.db.session import get_db
from app.api.v1.endpoints.auth import get_current_verified_user as get_current_user
from app.models.user import User
from app.services.training_pipeline_service import training_service

router = APIRouter(prefix="/training", tags=["Training Pipeline"])


# Request/Response Models
class TrainModelRequest(BaseModel):
    """Request model for model training"""
    model_name: str = Field(..., description="Model name to train")
    min_videos: int = Field(100, description="Minimum videos required")
    force_retrain: bool = Field(False, description="Force retraining even if not needed")


class ScheduleTrainingRequest(BaseModel):
    """Request model for scheduling training"""
    model_name: str = Field(..., description="Model name to schedule")
    cron_expression: str = Field(..., description="Cron expression for scheduling")


class TriggerRetrainingRequest(BaseModel):
    """Request model for triggering retraining"""
    model_name: str = Field(..., description="Model name to retrain")
    reason: str = Field(..., description="Reason for retraining")
    priority: str = Field("normal", description="Priority level (low, normal, high)")


class MonitorModelRequest(BaseModel):
    """Request model for model monitoring"""
    model_name: str = Field(..., description="Model name to monitor")
    threshold: float = Field(0.75, ge=0, le=1, description="Performance threshold")


# Endpoints
@router.post("/train-performance-model")
async def train_performance_model(
    request: TrainModelRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Train video performance prediction model
    """
    try:
        # Run training in background for long-running tasks
        result = await training_service.train_video_performance_model(
            db=db,
            min_videos=request.min_videos,
            force_retrain=request.force_retrain
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train-quality-model")
async def train_quality_model(
    request: TrainModelRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Train content quality prediction model
    """
    try:
        result = await training_service.train_content_quality_model(
            db=db,
            min_videos=request.min_videos
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/schedule")
async def schedule_training(
    request: ScheduleTrainingRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Schedule periodic model training
    
    Example cron expressions:
    - "0 0 * * *" - Daily at midnight
    - "0 0 * * 0" - Weekly on Sunday
    - "0 0 1 * *" - Monthly on the 1st
    """
    try:
        result = await training_service.schedule_periodic_training(
            model_name=request.model_name,
            cron_expression=request.cron_expression,
            db=db
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trigger-retraining")
async def trigger_retraining(
    request: TriggerRetrainingRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Trigger model retraining
    """
    try:
        result = await training_service.trigger_model_retraining(
            model_name=request.model_name,
            reason=request.reason,
            priority=request.priority
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monitor")
async def monitor_model(
    request: MonitorModelRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Monitor model performance and trigger retraining if needed
    """
    try:
        result = await training_service.monitor_model_performance(
            model_name=request.model_name,
            threshold=request.threshold
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_training_history(
    model_name: Optional[str] = None,
    limit: int = 10,
    current_user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """
    Get training history for models
    """
    try:
        history = await training_service.get_training_history(
            model_name=model_name,
            limit=limit
        )
        
        return history
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scheduled")
async def get_scheduled_trainings(
    current_user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """
    Get all scheduled training jobs
    """
    try:
        schedules = await training_service.get_scheduled_trainings()
        return schedules
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_available_models(
    current_user: User = Depends(get_current_user)
) -> List[Dict[str, str]]:
    """
    List available models for training
    """
    return [
        {
            "name": "video_performance_predictor",
            "type": "regression",
            "description": "Predicts video views and engagement"
        },
        {
            "name": "content_quality_predictor",
            "type": "classification",
            "description": "Classifies content quality level"
        },
        {
            "name": "channel_growth_predictor",
            "type": "regression",
            "description": "Predicts channel subscriber growth"
        },
        {
            "name": "trending_topic_classifier",
            "type": "classification",
            "description": "Classifies trending topic categories"
        }
    ]


@router.get("/status")
async def get_training_status(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get overall training pipeline status
    """
    try:
        # Get recent training history
        recent_trainings = await training_service.get_training_history(limit=5)
        
        # Get scheduled trainings
        scheduled = await training_service.get_scheduled_trainings()
        
        return {
            "pipeline_available": training_service.pipeline is not None,
            "recent_trainings": len(recent_trainings),
            "scheduled_trainings": len(scheduled),
            "last_training": recent_trainings[0] if recent_trainings else None,
            "status": "operational" if training_service.pipeline else "unavailable"
        }
        
    except Exception as e:
        return {
            "pipeline_available": False,
            "status": "error",
            "message": str(e)
        }