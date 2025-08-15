"""
ETL Pipeline API Endpoints
Exposes ETL pipeline execution and management capabilities
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.db.session import get_db
from app.api.v1.endpoints.auth import get_current_verified_user as get_current_user
from app.models.user import User
from app.services.etl_pipeline_service import etl_service

router = APIRouter(prefix="/etl", tags=["ETL Pipeline"])


# Request/Response Models
class RunETLRequest(BaseModel):
    """Request model for running ETL pipeline"""
    pipeline_name: str = Field(..., description="Pipeline name to run")
    incremental: bool = Field(True, description="Run incremental load")
    channel_id: Optional[str] = Field(None, description="Optional channel ID filter")


class ScheduleETLRequest(BaseModel):
    """Request model for scheduling ETL"""
    pipeline_name: str = Field(..., description="Pipeline name to schedule")
    cron_expression: str = Field(..., description="Cron expression for scheduling")


class JobStatusResponse(BaseModel):
    """Response model for job status"""
    job_id: str
    pipeline_name: str
    status: str
    start_time: Optional[str]
    end_time: Optional[str]
    records_processed: int
    quality_score: float
    errors: List[str]


# Endpoints
@router.post("/run/video-performance")
async def run_video_performance_etl(
    request: RunETLRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Run ETL pipeline for video performance data
    """
    try:
        result = await etl_service.run_video_performance_etl(
            db=db,
            incremental=request.incremental
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run/generation-metrics")
async def run_generation_metrics_etl(
    request: RunETLRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Run ETL pipeline for generation metrics
    """
    try:
        result = await etl_service.run_generation_metrics_etl(
            db=db,
            incremental=request.incremental
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run/channel-analytics")
async def run_channel_analytics_etl(
    request: RunETLRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Run ETL pipeline for channel analytics
    """
    try:
        result = await etl_service.run_channel_analytics_etl(
            db=db,
            channel_id=request.channel_id
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/schedule")
async def schedule_etl_pipeline(
    request: ScheduleETLRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Schedule periodic ETL pipeline execution
    
    Example cron expressions:
    - "0 */6 * * *" - Every 6 hours
    - "0 0 * * *" - Daily at midnight
    - "*/15 * * * *" - Every 15 minutes
    """
    try:
        result = await etl_service.schedule_etl_pipeline(
            pipeline_name=request.pipeline_name,
            cron_expression=request.cron_expression
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/job/{job_id}")
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user)
) -> Optional[JobStatusResponse]:
    """
    Get ETL job status by job ID
    """
    try:
        status = await etl_service.get_job_status(job_id)
        
        if status:
            return JobStatusResponse(**status)
        
        raise HTTPException(status_code=404, detail="Job not found")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quality-report")
async def get_data_quality_report(
    dimension: Optional[str] = None,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get data quality report for dimensions
    """
    try:
        report = await etl_service.get_data_quality_report(dimension=dimension)
        return report
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipelines")
async def list_available_pipelines(
    current_user: User = Depends(get_current_user)
) -> List[Dict[str, str]]:
    """
    List available ETL pipelines
    """
    return [
        {
            "name": "video_performance_etl",
            "description": "ETL for video performance metrics",
            "schedule": "Every 6 hours",
            "incremental": True
        },
        {
            "name": "generation_metrics_etl",
            "description": "ETL for video generation metrics",
            "schedule": "Every 12 hours",
            "incremental": True
        },
        {
            "name": "channel_analytics_etl",
            "description": "ETL for channel analytics",
            "schedule": "Daily",
            "incremental": False
        },
        {
            "name": "cost_analytics_etl",
            "description": "ETL for cost analytics",
            "schedule": "Hourly",
            "incremental": True
        },
        {
            "name": "user_behavior_etl",
            "description": "ETL for user behavior analytics",
            "schedule": "Every 4 hours",
            "incremental": True
        }
    ]


@router.get("/dimensions")
async def list_dimensions(
    current_user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """
    List available dimension tables
    """
    return [
        {
            "name": "dim_channel",
            "description": "Channel dimension with SCD Type 2",
            "key": "channel_key",
            "attributes": ["channel_id", "channel_name", "subscriber_count", "video_count"]
        },
        {
            "name": "dim_video",
            "description": "Video dimension with SCD Type 1",
            "key": "video_key",
            "attributes": ["video_id", "title", "description", "duration", "category"]
        },
        {
            "name": "dim_date",
            "description": "Date dimension for time-based analysis",
            "key": "date_key",
            "attributes": ["full_date", "day_of_week", "month", "quarter", "year"]
        },
        {
            "name": "dim_time",
            "description": "Time dimension for hourly analysis",
            "key": "time_key",
            "attributes": ["hour", "minute", "time_of_day", "am_pm"]
        },
        {
            "name": "dim_user",
            "description": "User dimension with subscription tiers",
            "key": "user_key",
            "attributes": ["user_id", "username", "user_type", "subscription_tier"]
        }
    ]


@router.get("/facts")
async def list_fact_tables(
    current_user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """
    List available fact tables
    """
    return [
        {
            "name": "fact_video_performance",
            "description": "Video performance metrics",
            "dimensions": ["channel_key", "video_key", "date_key", "time_key"],
            "measures": ["views", "likes", "comments", "engagement_rate", "revenue_usd"]
        },
        {
            "name": "fact_generation_metrics",
            "description": "Video generation metrics and costs",
            "dimensions": ["video_key", "user_key", "date_key"],
            "measures": ["generation_time_seconds", "total_cost", "quality_score"]
        }
    ]


@router.get("/status")
async def get_etl_status(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get overall ETL pipeline status
    """
    try:
        # Get data quality report
        quality_report = await etl_service.get_data_quality_report()
        
        return {
            "pipeline_available": etl_service.pipeline is not None,
            "dimension_tables_created": etl_service.is_initialized,
            "overall_data_quality": quality_report.get("overall_quality", 0),
            "status": "operational" if etl_service.pipeline else "unavailable"
        }
        
    except Exception as e:
        return {
            "pipeline_available": False,
            "status": "error",
            "message": str(e)
        }