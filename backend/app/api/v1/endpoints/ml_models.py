"""
ML Model Management Endpoints
Handles model deployment, versioning, and performance monitoring
"""
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import logging

from app.db.session import get_db
from app.core.auth import get_current_user
from app.services.ml_integration_service import ml_service
from app.services.trend_analyzer import trend_analyzer
from app.core.config import settings
from app.models.user import User
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models for request/response
class ModelConfig(BaseModel):
    """ML Model configuration"""
    model_name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Model type (trend, script, voice, thumbnail)")
    version: str = Field("1.0.0", description="Model version")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Model parameters")
    active: bool = Field(True, description="Whether model is active")


class ModelDeployRequest(BaseModel):
    """Model deployment request"""
    config: ModelConfig
    environment: str = Field("production", description="Deployment environment")
    auto_scale: bool = Field(True, description="Enable auto-scaling")


class ModelMetrics(BaseModel):
    """Model performance metrics"""
    model_id: str
    accuracy: float
    latency_ms: float
    throughput: int
    error_rate: float
    cost_per_inference: float


class TrendAnalysisRequest(BaseModel):
    """Trend analysis request"""
    category: str = Field("technology", description="Content category")
    region: str = Field("US", description="Geographic region")
    timeframe: str = Field("now", description="Analysis timeframe")
    limit: int = Field(10, description="Number of trends to return")


class PredictionRequest(BaseModel):
    """General prediction request"""
    model_type: str = Field(..., description="Type of prediction model")
    input_data: Dict[str, Any] = Field(..., description="Input data for prediction")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional parameters")


@router.get("/models", response_model=List[Dict[str, Any]])
async def list_models(
    model_type: Optional[str] = None,
    active_only: bool = True,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List available ML models
    
    Args:
        model_type: Filter by model type (trend, script, voice, thumbnail)
        active_only: Only show active models
        
    Returns:
        List of available models with metadata
    """
    try:
        models = await ml_service.list_models(
            model_type=model_type,
            active_only=active_only
        )
        
        return models
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )


@router.get("/models/{model_id}", response_model=Dict[str, Any])
async def get_model_details(
    model_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed information about a specific model
    
    Args:
        model_id: Model identifier
        
    Returns:
        Model details including configuration and metrics
    """
    try:
        model_info = await ml_service.get_model_info(model_id)
        
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_id} not found"
            )
        
        return model_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model details: {str(e)}"
        )


@router.post("/models/deploy", response_model=Dict[str, Any])
async def deploy_model(
    request: ModelDeployRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Deploy a new ML model
    
    Args:
        request: Model deployment configuration
        
    Returns:
        Deployment status and model ID
    """
    try:
        # Check user permissions (admin only for production deployment)
        if request.environment == "production" and not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required for production deployment"
            )
        
        # Deploy model
        deployment_result = await ml_service.deploy_model(
            model_config=request.config.dict(),
            environment=request.environment,
            auto_scale=request.auto_scale
        )
        
        # Schedule background monitoring
        background_tasks.add_task(
            ml_service.start_model_monitoring,
            deployment_result["model_id"]
        )
        
        return deployment_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deploying model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to deploy model: {str(e)}"
        )


@router.put("/models/{model_id}/update", response_model=Dict[str, Any])
async def update_model(
    model_id: str,
    config: ModelConfig,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update model configuration
    
    Args:
        model_id: Model to update
        config: New configuration
        
    Returns:
        Update status
    """
    try:
        result = await ml_service.update_model(
            model_id=model_id,
            config=config.dict()
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error updating model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update model: {str(e)}"
        )


@router.delete("/models/{model_id}")
async def delete_model(
    model_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete/deactivate a model
    
    Args:
        model_id: Model to delete
        
    Returns:
        Deletion status
    """
    try:
        # Check admin permissions
        if not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required to delete models"
            )
        
        result = await ml_service.delete_model(model_id)
        
        return {"status": "success", "message": f"Model {model_id} deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete model: {str(e)}"
        )


@router.get("/models/{model_id}/metrics", response_model=ModelMetrics)
async def get_model_metrics(
    model_id: str,
    timeframe: str = "24h",
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get model performance metrics
    
    Args:
        model_id: Model identifier
        timeframe: Metrics timeframe (1h, 24h, 7d, 30d)
        
    Returns:
        Model performance metrics
    """
    try:
        metrics = await ml_service.get_model_metrics(
            model_id=model_id,
            timeframe=timeframe
        )
        
        return ModelMetrics(**metrics)
        
    except Exception as e:
        logger.error(f"Error getting model metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model metrics: {str(e)}"
        )


@router.post("/predict", response_model=Dict[str, Any])
async def make_prediction(
    request: PredictionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Make a prediction using specified model
    
    Args:
        request: Prediction request with model type and input data
        
    Returns:
        Prediction results
    """
    try:
        # Route to appropriate model
        if request.model_type == "trend":
            # Use trend analyzer
            results = await trend_analyzer.analyze_trends(
                category=request.input_data.get("category", "technology"),
                region=request.input_data.get("region", "US"),
                limit=request.input_data.get("limit", 10)
            )
            
        elif request.model_type == "viral_potential":
            # Predict viral potential
            results = await trend_analyzer.predict_viral_potential(
                topic=request.input_data.get("topic"),
                keywords=request.input_data.get("keywords", []),
                category=request.input_data.get("category", "technology")
            )
            
        else:
            # Generic model prediction
            results = await ml_service.predict(
                model_type=request.model_type,
                input_data=request.input_data,
                parameters=request.parameters
            )
        
        return {
            "model_type": request.model_type,
            "prediction": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/models/{model_id}/train")
async def train_model(
    model_id: str,
    training_data: UploadFile = File(...),
    epochs: int = 10,
    batch_size: int = 32,
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Train or fine-tune a model
    
    Args:
        model_id: Model to train
        training_data: Training data file
        epochs: Number of training epochs
        batch_size: Training batch size
        
    Returns:
        Training job ID and status
    """
    try:
        # Check admin permissions
        if not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required for model training"
            )
        
        # Save training data
        content = await training_data.read()
        
        # Start training job
        job_id = await ml_service.start_training(
            model_id=model_id,
            training_data=content,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Schedule background monitoring
        if background_tasks:
            background_tasks.add_task(
                ml_service.monitor_training,
                job_id
            )
        
        return {
            "job_id": job_id,
            "status": "training_started",
            "model_id": model_id,
            "epochs": epochs,
            "batch_size": batch_size
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}"
        )


@router.get("/models/{model_id}/versions", response_model=List[Dict[str, Any]])
async def get_model_versions(
    model_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all versions of a model
    
    Args:
        model_id: Model identifier
        
    Returns:
        List of model versions with metadata
    """
    try:
        versions = await ml_service.get_model_versions(model_id)
        
        return versions
        
    except Exception as e:
        logger.error(f"Error getting model versions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model versions: {str(e)}"
        )


@router.post("/models/{model_id}/rollback")
async def rollback_model(
    model_id: str,
    version: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Rollback model to a previous version
    
    Args:
        model_id: Model identifier
        version: Version to rollback to
        
    Returns:
        Rollback status
    """
    try:
        # Check admin permissions
        if not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required for model rollback"
            )
        
        result = await ml_service.rollback_model(
            model_id=model_id,
            version=version
        )
        
        return {
            "status": "success",
            "message": f"Model {model_id} rolled back to version {version}",
            "result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rolling back model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rollback failed: {str(e)}"
        )


@router.post("/trends/analyze", response_model=List[Dict[str, Any]])
async def analyze_trends(
    request: TrendAnalysisRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze trends using ML models
    
    Args:
        request: Trend analysis parameters
        
    Returns:
        List of trending topics with scores
    """
    try:
        trends = await trend_analyzer.analyze_trends(
            category=request.category,
            region=request.region,
            timeframe=request.timeframe,
            limit=request.limit
        )
        
        return trends
        
    except Exception as e:
        logger.error(f"Error analyzing trends: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Trend analysis failed: {str(e)}"
        )