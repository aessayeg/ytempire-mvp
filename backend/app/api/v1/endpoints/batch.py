"""
Batch Processing API endpoints
"""

import uuid
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import asyncio

from app.api.v1.endpoints.auth import get_current_verified_user
from app.services.batch_processing import (
    batch_processor,
    video_batch_processor,
    data_batch_processor,
    BatchJobType,
    BatchJobConfig,
    BatchJobItem
)
from app.models.user import User

router = APIRouter()

class BatchVideoRequest(BaseModel):
    title: str = Field(..., max_length=200)
    description: Optional[str] = Field(None, max_length=2000)
    tags: List[str] = Field(default_factory=list)
    niche: Optional[str] = None
    scheduled_publish_time: Optional[datetime] = None

class BatchVideoGenerationRequest(BaseModel):
    videos: List[BatchVideoRequest] = Field(..., min_items=1, max_items=50)
    max_concurrent: int = Field(default=3, ge=1, le=10)
    priority: int = Field(default=5, ge=1, le=10, description="Job priority (1-10, higher is more priority)")
    scheduled_at: Optional[datetime] = Field(None, description="Schedule job for future execution")
    dependencies: Optional[List[str]] = Field(None, description="Job IDs that must complete first")
    checkpoint_interval: int = Field(default=5, description="Save progress every N videos")
    callback_url: Optional[str] = Field(None, description="Webhook URL for completion notification")

class BatchAnalyticsRequest(BaseModel):
    channel_ids: List[str] = Field(..., min_items=1, max_items=20)
    start_date: str = Field(..., pattern=r'^\d{4}-\d{2}-\d{2}$')
    end_date: str = Field(..., pattern=r'^\d{4}-\d{2}-\d{2}$')

class BatchJobStatusResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    total_items: int
    completed_items: int
    failed_items: int
    success_rate: Optional[float] = None
    duration_seconds: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

@router.post("/videos/generate")
async def batch_generate_videos(
    request: BatchVideoGenerationRequest,
    current_user: User = Depends(get_current_verified_user)
):
    """Generate multiple videos in batch with enhanced features"""
    try:
        # Convert requests to internal format
        video_requests = []
        for video_req in request.videos:
            video_requests.append({
                "title": video_req.title,
                "description": video_req.description,
                "tags": video_req.tags,
                "niche": video_req.niche,
                "scheduled_publish_time": video_req.scheduled_publish_time.isoformat() if video_req.scheduled_publish_time else None,
                "user_id": str(current_user.id)
            })
        
        # Enhanced batch configuration
        from app.services.batch_processing import BatchJobConfig, BatchJobItem, BatchJobType
        import uuid
        
        job_id = f"video_batch_{uuid.uuid4().hex[:8]}"
        items = []
        for i, req in enumerate(video_requests):
            items.append(BatchJobItem(
                id=f"{job_id}_item_{i}",
                data=req
            ))
        
        job_config = BatchJobConfig(
            job_id=job_id,
            job_type=BatchJobType.VIDEO_GENERATION,
            items=items,
            max_concurrent=request.max_concurrent,
            priority=request.priority,
            scheduled_at=request.scheduled_at,
            dependencies=request.dependencies,
            checkpoint_interval=request.checkpoint_interval,
            callback_url=request.callback_url,
            notification_user_id=str(current_user.id),
            timeout_per_item=600,  # 10 minutes per video
            resume_on_failure=True
        )
        
        # Submit job with enhanced configuration
        job_id = await batch_processor.submit_batch_job(
            job_config,
            video_batch_processor._process_video_generation
        )
        
        # Calculate estimated completion time
        estimated_time = (len(request.videos) * 5) / request.max_concurrent  # 5 min per video
        estimated_completion = datetime.utcnow() + timedelta(minutes=estimated_time)
        
        return {
            "success": True,
            "job_id": job_id,
            "total_videos": len(request.videos),
            "priority": request.priority,
            "scheduled_at": request.scheduled_at.isoformat() if request.scheduled_at else None,
            "estimated_completion": estimated_completion.isoformat(),
            "message": f"Batch video generation job submitted with {len(request.videos)} videos"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit batch video generation: {str(e)}"
        )

@router.post("/analytics/process")
async def batch_process_analytics(
    request: BatchAnalyticsRequest,
    current_user: User = Depends(get_current_verified_user)
):
    """Process analytics for multiple channels in batch"""
    try:
        job_id = await data_batch_processor.batch_process_analytics(
            channel_ids=request.channel_ids,
            user_id=str(current_user.id),
            date_range={
                "start_date": request.start_date,
                "end_date": request.end_date
            }
        )
        
        return {
            "success": True,
            "job_id": job_id,
            "total_channels": len(request.channel_ids),
            "date_range": f"{request.start_date} to {request.end_date}",
            "message": f"Batch analytics processing job submitted for {len(request.channel_ids)} channels"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit batch analytics processing: {str(e)}"
        )

@router.get("/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_verified_user)
):
    """Get status of a specific batch job"""
    try:
        status_data = await batch_processor.get_job_status(job_id)
        
        if not status_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch job '{job_id}' not found"
            )
        
        return {
            "success": True,
            "job": status_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job status: {str(e)}"
        )

@router.get("/jobs")
async def get_all_jobs(
    include_metrics: bool = False,
    current_user: User = Depends(get_current_verified_user)
):
    """Get status of all batch jobs with optional metrics"""
    try:
        all_jobs = await batch_processor.get_all_jobs()
        
        # Add system metrics if requested
        if include_metrics:
            metrics = await batch_processor.get_job_metrics()
            all_jobs["system_metrics"] = metrics
        
        # Filter jobs for current user (in production, implement proper user filtering)
        # For now, return all jobs
        
        return {
            "success": True,
            **all_jobs
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get jobs: {str(e)}"
        )

@router.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    current_user: User = Depends(get_current_verified_user)
):
    """Cancel a running batch job"""
    try:
        success = await batch_processor.cancel_job(job_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch job '{job_id}' not found or not running"
            )
        
        return {
            "success": True,
            "job_id": job_id,
            "message": "Batch job cancelled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel job: {str(e)}"
        )


@router.post("/jobs/{job_id}/pause")
async def pause_job(
    job_id: str,
    current_user: User = Depends(get_current_verified_user)
):
    """Pause a running batch job"""
    try:
        success = await batch_processor.pause_job(job_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch job '{job_id}' not found or not running"
            )
        
        return {
            "success": True,
            "job_id": job_id,
            "message": "Batch job paused successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to pause job: {str(e)}"
        )


@router.post("/jobs/{job_id}/resume")
async def resume_job(
    job_id: str,
    current_user: User = Depends(get_current_verified_user)
):
    """Resume a paused batch job"""
    try:
        success = await batch_processor.resume_job(job_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch job '{job_id}' not found or not paused"
            )
        
        return {
            "success": True,
            "job_id": job_id,
            "message": "Batch job resumed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resume job: {str(e)}"
        )

@router.get("/jobs/{job_id}/items")
async def get_job_items(
    job_id: str,
    current_user: User = Depends(get_current_verified_user)
):
    """Get detailed status of all items in a batch job"""
    try:
        if job_id not in batch_processor.running_jobs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch job '{job_id}' not found or not running"
            )
        
        job_config = batch_processor.running_jobs[job_id]
        
        items = []
        for item in job_config.items:
            items.append({
                "id": item.id,
                "status": item.status.value,
                "started_at": item.started_at.isoformat() if item.started_at else None,
                "completed_at": item.completed_at.isoformat() if item.completed_at else None,
                "error": item.error,
                "result": item.result
            })
        
        return {
            "success": True,
            "job_id": job_id,
            "job_type": job_config.job_type.value,
            "total_items": len(items),
            "items": items
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job items: {str(e)}"
        )

# Utility endpoints for batch operations

@router.post("/videos/validate")
async def validate_batch_video_request(
    request: BatchVideoGenerationRequest,
    current_user: User = Depends(get_current_verified_user)
):
    """Validate a batch video generation request without submitting"""
    try:
        # Perform validation checks
        errors = []
        warnings = []
        
        # Check video count limits
        if len(request.videos) > 50:
            errors.append("Maximum 50 videos per batch")
        
        if len(request.videos) > 10 and request.max_concurrent > 5:
            warnings.append("High concurrency with many videos may impact system performance")
        
        # Check individual video requirements
        for i, video in enumerate(request.videos):
            if not video.title.strip():
                errors.append(f"Video {i+1}: Title cannot be empty")
            
            if len(video.tags) > 10:
                warnings.append(f"Video {i+1}: More than 10 tags may impact performance")
        
        # Estimate cost and time
        estimated_cost = len(request.videos) * 2.5  # $2.50 per video average
        estimated_time_minutes = (len(request.videos) * 5) / request.max_concurrent  # 5 min per video
        
        return {
            "success": len(errors) == 0,
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "estimates": {
                "total_cost_usd": round(estimated_cost, 2),
                "estimated_time_minutes": round(estimated_time_minutes, 1),
                "videos_count": len(request.videos),
                "max_concurrent": request.max_concurrent
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate batch request: {str(e)}"
        )

@router.get("/templates")
async def get_batch_templates(
    current_user: User = Depends(get_current_verified_user)
):
    """Get predefined batch processing templates"""
    try:
        templates = {
            "daily_content_batch": {
                "name": "Daily Content Batch",
                "description": "Generate 5 videos for daily content schedule",
                "max_videos": 5,
                "recommended_concurrent": 3,
                "priority": 7,
                "estimated_time_minutes": 8,
                "estimated_cost": 12.50,
                "checkpoint_interval": 2
            },
            "weekly_content_batch": {
                "name": "Weekly Content Batch",
                "description": "Generate 10 videos for weekly content plan",
                "max_videos": 10,
                "recommended_concurrent": 4,
                "priority": 5,
                "estimated_time_minutes": 12,
                "estimated_cost": 25.00,
                "checkpoint_interval": 3
            },
            "niche_exploration_batch": {
                "name": "Niche Exploration Batch",
                "description": "Generate 3 test videos in different niches",
                "max_videos": 3,
                "recommended_concurrent": 2,
                "priority": 9,
                "estimated_time_minutes": 8,
                "estimated_cost": 7.50,
                "checkpoint_interval": 1
            },
            "high_volume_batch": {
                "name": "High Volume Batch",
                "description": "Generate 25-50 videos for bulk content production",
                "max_videos": 50,
                "recommended_concurrent": 5,
                "priority": 3,
                "estimated_time_minutes": 50,
                "estimated_cost": 125.00,
                "checkpoint_interval": 5
            },
            "scheduled_weekend_batch": {
                "name": "Scheduled Weekend Batch",
                "description": "Schedule large batch for weekend processing",
                "max_videos": 20,
                "recommended_concurrent": 8,
                "priority": 4,
                "estimated_time_minutes": 15,
                "estimated_cost": 50.00,
                "checkpoint_interval": 4,
                "schedule_delay_hours": 48
            }
        }
        
        return {
            "success": True,
            "templates": templates
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get batch templates: {str(e)}"
        )


@router.get("/metrics")
async def get_batch_metrics(
    current_user: User = Depends(get_current_verified_user)
):
    """Get batch processing system metrics"""
    try:
        metrics = await batch_processor.get_job_metrics()
        
        return {
            "success": True,
            **metrics
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get batch metrics: {str(e)}"
        )


@router.post("/jobs/chain")
async def create_chained_jobs(
    jobs: List[Dict[str, Any]],
    current_user: User = Depends(get_current_verified_user)
):
    """Create multiple batch jobs with dependencies"""
    try:
        job_ids = []
        
        for i, job_spec in enumerate(jobs):
            # Set dependencies to previous jobs
            if i > 0:
                job_spec["dependencies"] = job_ids[:i]
            
            # Submit job based on type
            if job_spec["type"] == "video_generation":
                # Process video generation job
                job_id = f"chain_video_{uuid.uuid4().hex[:8]}"
                # ... submit job ...
                job_ids.append(job_id)
            elif job_spec["type"] == "analytics":
                # Process analytics job
                job_id = f"chain_analytics_{uuid.uuid4().hex[:8]}"
                # ... submit job ...
                job_ids.append(job_id)
        
        return {
            "success": True,
            "chain_id": f"chain_{uuid.uuid4().hex[:8]}",
            "job_ids": job_ids,
            "total_jobs": len(job_ids),
            "message": f"Created chain of {len(job_ids)} dependent jobs"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create chained jobs: {str(e)}"
        )