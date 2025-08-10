"""
Batch Processing API endpoints
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from datetime import datetime

from ....core.security import get_current_user
from ....services.batch_processing import (
    batch_processor,
    video_batch_processor,
    data_batch_processor,
    BatchJobType,
    BatchJobConfig,
    BatchJobItem
)
from ....models.user import User

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

class BatchAnalyticsRequest(BaseModel):
    channel_ids: List[str] = Field(..., min_items=1, max_items=20)
    start_date: str = Field(..., regex=r'^\d{4}-\d{2}-\d{2}$')
    end_date: str = Field(..., regex=r'^\d{4}-\d{2}-\d{2}$')

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
    current_user: User = Depends(get_current_user)
):
    """Generate multiple videos in batch"""
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
        
        job_id = await video_batch_processor.batch_generate_videos(
            video_requests=video_requests,
            user_id=str(current_user.id),
            max_concurrent=request.max_concurrent
        )
        
        return {
            "success": True,
            "job_id": job_id,
            "total_videos": len(request.videos),
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
    current_user: User = Depends(get_current_user)
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
    current_user: User = Depends(get_current_user)
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
    current_user: User = Depends(get_current_user)
):
    """Get status of all batch jobs for the current user"""
    try:
        all_jobs = await batch_processor.get_all_jobs()
        
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
    current_user: User = Depends(get_current_user)
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

@router.get("/jobs/{job_id}/items")
async def get_job_items(
    job_id: str,
    current_user: User = Depends(get_current_user)
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
    current_user: User = Depends(get_current_user)
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
    current_user: User = Depends(get_current_user)
):
    """Get predefined batch processing templates"""
    try:
        templates = {
            "daily_content_batch": {
                "name": "Daily Content Batch",
                "description": "Generate 5 videos for daily content schedule",
                "max_videos": 5,
                "recommended_concurrent": 3,
                "estimated_time_minutes": 8,
                "estimated_cost": 12.50
            },
            "weekly_content_batch": {
                "name": "Weekly Content Batch",
                "description": "Generate 10 videos for weekly content plan",
                "max_videos": 10,
                "recommended_concurrent": 4,
                "estimated_time_minutes": 12,
                "estimated_cost": 25.00
            },
            "niche_exploration_batch": {
                "name": "Niche Exploration Batch",
                "description": "Generate 3 test videos in different niches",
                "max_videos": 3,
                "recommended_concurrent": 2,
                "estimated_time_minutes": 8,
                "estimated_cost": 7.50
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