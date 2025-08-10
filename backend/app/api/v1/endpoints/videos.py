"""
Video Management Endpoints
Owner: API Developer
"""

from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.api.v1.endpoints.auth import get_current_user
from app.schemas.video import (
    VideoCreate, VideoResponse, VideoGenerateRequest, VideoUpdate,
    VideoListResponse, VideoStats, VideoCostBreakdown
)
from app.models.user import User
from app.models.video import VideoStatus
from app.repositories.video_repository import VideoRepository
from app.repositories.channel_repository import ChannelRepository
from app.services.n8n_service import N8NService
from app.tasks.video_pipeline import create_video_pipeline, get_pipeline_status, cancel_pipeline
import logging

logger = logging.getLogger(__name__)
from app.core.config import settings

router = APIRouter()


def get_video_repo(db: AsyncSession = Depends(get_db)) -> VideoRepository:
    """Get video repository instance."""
    return VideoRepository(db)


def get_channel_repo(db: AsyncSession = Depends(get_db)) -> ChannelRepository:
    """Get channel repository instance."""
    return ChannelRepository(db)


def get_n8n_service(db: AsyncSession = Depends(get_db)) -> N8NService:
    """Get N8N service instance."""
    return N8NService(db)


@router.post("/generate", response_model=VideoResponse)
async def generate_video_endpoint(
    request: VideoGenerateRequest,
    current_user: User = Depends(get_current_user),
    video_repo: VideoRepository = Depends(get_video_repo),
    channel_repo: ChannelRepository = Depends(get_channel_repo),
    n8n_service: N8NService = Depends(get_n8n_service)
):
    """
    Generate a new video
    - Check user limits
    - Check channel ownership
    - Queue video generation task
    - Return video with queued status
    """
    # Verify channel ownership
    if not await channel_repo.check_ownership(request.channel_id, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Channel not found or access denied"
        )
    
    # Check daily limit
    daily_count = await video_repo.get_user_daily_video_count(current_user.id)
    if daily_count >= settings.MAX_VIDEOS_PER_DAY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Daily video limit reached. Maximum {settings.MAX_VIDEOS_PER_DAY} videos per day."
        )
    
    # Create video record
    video_create = VideoCreate(
        channel_id=request.channel_id,
        title=f"Video about {request.topic}",
        description=f"AI-generated video about {request.topic}",
        priority=request.priority or 1,
        scheduled_publish_at=request.schedule_time,
        content_settings={
            "topic": request.topic,
            "style": request.style,
            "target_duration": request.target_duration,
            "keywords": request.keywords or [],
            "custom_prompt": request.custom_prompt,
            "auto_publish": request.auto_publish
        },
        generation_settings={
            "model_version": "gpt-4",
            "voice_style": "professional",
            "video_style": request.style
        },
        metadata={
            "generated_from_request": True,
            "request_timestamp": str(request.schedule_time) if request.schedule_time else None
        }
    )
    
    video = await video_repo.create_video(video_create, current_user.id)
    
    # Trigger N8N video generation workflow
    video_config = {
        "title": video.title,
        "description": video.description,
        "keywords": request.keywords or [],
        "duration": request.target_duration or 60,
        "style": request.style,
        "voice_id": None,  # Will be selected by N8N
        "background_music": True,
        "thumbnail_style": "auto"
    }
    
    workflow_success = await n8n_service.trigger_video_generation_workflow(
        video.id,
        current_user.id,
        request.channel_id,
        video_config
    )
    
    if not workflow_success:
        # If workflow failed to trigger, update video status to failed
        await video_repo.update_status(video.id, VideoStatus.FAILED)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to trigger video generation workflow"
        )
    
    return VideoResponse.from_orm(video)


@router.post("/generate-celery", response_model=VideoResponse)
async def generate_video_with_celery_pipeline(
    request: VideoGenerateRequest,
    current_user: User = Depends(get_current_user),
    video_repo: VideoRepository = Depends(get_video_repo),
    channel_repo: ChannelRepository = Depends(get_channel_repo)
):
    """
    Generate a new video using Celery pipeline (alternative to N8N).
    - Check user limits
    - Check channel ownership
    - Queue video generation task with Celery
    - Return video with queued status
    """
    # Verify channel ownership
    if not await channel_repo.check_ownership(request.channel_id, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Channel not found or access denied"
        )
    
    # Check daily limit
    daily_count = await video_repo.get_user_daily_video_count(current_user.id)
    if daily_count >= settings.MAX_VIDEOS_PER_DAY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Daily video limit reached. Maximum {settings.MAX_VIDEOS_PER_DAY} videos per day."
        )
    
    # Create video record
    video_create = VideoCreate(
        channel_id=request.channel_id,
        title=f"Video about {request.topic}",
        description=f"AI-generated video about {request.topic}",
        priority=request.priority or 1,
        scheduled_publish_at=request.schedule_time,
        content_settings={
            "topic": request.topic,
            "style": request.style,
            "target_duration": request.target_duration,
            "keywords": request.keywords or [],
            "custom_prompt": request.custom_prompt,
            "auto_publish": request.auto_publish
        },
        generation_settings={
            "model_version": "gpt-4",
            "voice_style": "professional",
            "video_style": request.style,
            "pipeline_type": "celery"
        },
        metadata={
            "generated_from_request": True,
            "request_timestamp": str(request.schedule_time) if request.schedule_time else None,
            "pipeline_type": "celery"
        }
    )
    
    video = await video_repo.create_video(video_create, current_user.id)
    
    # Create Celery pipeline request
    pipeline_request = {
        "id": video.id,
        "channel_id": request.channel_id,
        "user_id": current_user.id,
        "topic": request.topic,
        "style": request.style,
        "target_duration": request.target_duration or 480,
        "keywords": request.keywords or [],
        "auto_publish": request.auto_publish or False,
        "created_at": datetime.utcnow().isoformat()
    }
    
    # Start Celery pipeline
    pipeline_task_id = create_video_pipeline(pipeline_request)
    
    # Update video with pipeline task ID
    await video_repo.update(video.id, {
        "metadata": {
            **video.metadata,
            "pipeline_task_id": pipeline_task_id,
            "pipeline_started": datetime.utcnow().isoformat()
        }
    })
    
    return VideoResponse.from_orm(video)


@router.post("/", response_model=VideoResponse)
async def create_video(
    video_data: VideoCreate,
    current_user: User = Depends(get_current_user),
    video_repo: VideoRepository = Depends(get_video_repo),
    channel_repo: ChannelRepository = Depends(get_channel_repo)
):
    """
    Create a new video manually
    """
    # Verify channel ownership
    if not await channel_repo.check_ownership(video_data.channel_id, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Channel not found or access denied"
        )
    
    video = await video_repo.create_video(video_data, current_user.id)
    return VideoResponse.from_orm(video)


@router.get("/", response_model=VideoListResponse)
async def list_videos(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    channel_id: Optional[str] = Query(None, description="Filter by channel"),
    current_user: User = Depends(get_current_user),
    video_repo: VideoRepository = Depends(get_video_repo),
    channel_repo: ChannelRepository = Depends(get_channel_repo)
):
    """
    List videos for the current user with pagination
    """
    offset = (page - 1) * per_page
    
    # If filtering by channel, verify ownership
    if channel_id:
        if not await channel_repo.check_ownership(channel_id, current_user.id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Channel not found or access denied"
            )
        videos = await video_repo.get_by_channel_id(channel_id, per_page, offset)
    else:
        videos = await video_repo.get_by_user_id(current_user.id, per_page, offset)
    
    # Filter by status if provided
    if status_filter:
        try:
            status_enum = VideoStatus(status_filter.upper())
            videos = [v for v in videos if v.status == status_enum]
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status filter: {status_filter}"
            )
    
    total_count = len(videos)  # Simplified - in production, get actual count
    has_next = len(videos) == per_page
    has_prev = page > 1
    
    return VideoListResponse(
        videos=[VideoResponse.from_orm(video) for video in videos],
        total_count=total_count,
        page=page,
        per_page=per_page,
        has_next=has_next,
        has_prev=has_prev
    )


@router.get("/{video_id}", response_model=VideoResponse)
async def get_video(
    video_id: str,
    current_user: User = Depends(get_current_user),
    video_repo: VideoRepository = Depends(get_video_repo)
):
    """
    Get specific video details
    """
    # Verify ownership
    if not await video_repo.check_ownership(video_id, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found or access denied"
        )
    
    video = await video_repo.get_by_id(video_id)
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found"
        )
    
    return VideoResponse.from_orm(video)


@router.patch("/{video_id}", response_model=VideoResponse)
async def update_video(
    video_id: str,
    update_data: VideoUpdate,
    current_user: User = Depends(get_current_user),
    video_repo: VideoRepository = Depends(get_video_repo)
):
    """
    Update video metadata
    """
    # Verify ownership
    if not await video_repo.check_ownership(video_id, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found or access denied"
        )
    
    # Get current video
    video = await video_repo.get_by_id(video_id)
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found"
        )
    
    # Update video content
    updated_video = await video_repo.update_video_content(
        video_id=video_id,
        title=update_data.title,
        description=update_data.description
    )
    
    if not updated_video:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to update video"
        )
    
    return VideoResponse.from_orm(updated_video)


@router.delete("/{video_id}")
async def delete_video(
    video_id: str,
    current_user: User = Depends(get_current_user),
    video_repo: VideoRepository = Depends(get_video_repo)
):
    """
    Delete a video
    """
    # Verify ownership
    if not await video_repo.check_ownership(video_id, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found or access denied"
        )
    
    success = await video_repo.delete_video(video_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found"
        )
    
    return {"message": "Video deleted successfully"}


@router.post("/{video_id}/retry")
async def retry_video_generation(
    video_id: str,
    current_user: User = Depends(get_current_user),
    video_repo: VideoRepository = Depends(get_video_repo)
):
    """
    Retry failed video generation
    """
    # Verify ownership
    if not await video_repo.check_ownership(video_id, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found or access denied"
        )
    
    video = await video_repo.get_by_id(video_id)
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found"
        )
    
    if video.status != VideoStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only failed videos can be retried"
        )
    
    if video.retry_count >= 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum retry attempts reached"
        )
    
    # Reset status and increment retry count
    await video_repo.update_video_status(video_id, VideoStatus.PENDING)
    await video_repo.increment_retry_count(video_id)
    
    # Here we would queue the generation task again
    # from app.tasks.video_generation import generate_video_task
    # task = generate_video_task.delay(video_id)
    
    return {"message": "Video generation retry initiated"}


@router.get("/{video_id}/stats", response_model=VideoStats)
async def get_video_stats(
    video_id: str,
    current_user: User = Depends(get_current_user),
    video_repo: VideoRepository = Depends(get_video_repo)
):
    """
    Get video statistics (from YouTube API if available)
    """
    # Verify ownership
    if not await video_repo.check_ownership(video_id, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found or access denied"
        )
    
    video = await video_repo.get_by_id(video_id)
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found"
        )
    
    # If video has YouTube ID, fetch stats from YouTube API
    stats = VideoStats(video_id=video_id)
    
    if video.youtube_video_id:
        try:
            # In a real implementation, we'd fetch from YouTube API
            # from app.services.youtube_service import get_youtube_service
            # youtube = await get_youtube_service(current_user.id)
            # yt_stats = await youtube.get_video_stats(video.youtube_video_id)
            # stats = VideoStats(video_id=video_id, **yt_stats)
            pass
        except Exception as e:
            # Fallback to basic stats
            pass
    
    return stats


@router.get("/{video_id}/cost-breakdown", response_model=VideoCostBreakdown)
async def get_video_cost_breakdown(
    video_id: str,
    current_user: User = Depends(get_current_user),
    video_repo: VideoRepository = Depends(get_video_repo)
):
    """
    Get detailed cost breakdown for video generation
    """
    # Verify ownership
    if not await video_repo.check_ownership(video_id, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found or access denied"
        )
    
    video = await video_repo.get_by_id(video_id)
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found"
        )
    
    # In a real implementation, we'd fetch detailed costs from cost tracking
    total_cost = video.total_cost or 0.0
    
    return VideoCostBreakdown(
        video_id=video_id,
        content_generation_cost=total_cost * 0.4,
        audio_synthesis_cost=total_cost * 0.3,
        visual_cost=total_cost * 0.2,
        compilation_cost=total_cost * 0.1,
        total_cost=total_cost,
        within_budget=total_cost <= settings.MAX_COST_PER_VIDEO
    )


@router.get("/queue/pending")
async def get_pending_videos(
    limit: int = Query(10, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    video_repo: VideoRepository = Depends(get_video_repo)
):
    """
    Get pending videos in the generation queue (admin/monitoring endpoint)
    """
    # In production, this would have admin access control
    videos = await video_repo.get_pending_videos(limit)
    return {
        "pending_videos": [VideoResponse.from_orm(video) for video in videos],
        "count": len(videos)
    }


@router.get("/queue/processing")
async def get_processing_videos(
    current_user: User = Depends(get_current_user),
    video_repo: VideoRepository = Depends(get_video_repo)
):
    """
    Get videos currently being processed (admin/monitoring endpoint)
    """
    # In production, this would have admin access control
    videos = await video_repo.get_processing_videos()
    return {
        "processing_videos": [VideoResponse.from_orm(video) for video in videos],
        "count": len(videos)
    }


@router.get("/user/stats")
async def get_user_video_stats(
    current_user: User = Depends(get_current_user),
    video_repo: VideoRepository = Depends(get_video_repo)
):
    """
    Get user's video statistics summary
    """
    stats = await video_repo.get_user_video_stats(current_user.id)
    
    return {
        "user_id": current_user.id,
        "video_statistics": stats,
        "daily_limit": settings.MAX_VIDEOS_PER_DAY,
        "daily_remaining": max(0, settings.MAX_VIDEOS_PER_DAY - stats.get('status_counts', {}).get('pending', 0)),
        "cost_limit": settings.MAX_COST_PER_VIDEO
    }


@router.get("/{video_id}/pipeline-status")
async def get_video_pipeline_status(
    video_id: str,
    current_user: User = Depends(get_current_user),
    video_repo: VideoRepository = Depends(get_video_repo)
):
    """
    Get detailed pipeline status for a video (Celery or N8N).
    """
    # Verify ownership
    if not await video_repo.check_ownership(video_id, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found or access denied"
        )
    
    video = await video_repo.get_by_id(video_id)
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found"
        )
    
    pipeline_info = {
        "video_id": video_id,
        "current_status": video.status.value,
        "pipeline_type": video.metadata.get("pipeline_type", "unknown"),
        "created_at": video.created_at.isoformat() if video.created_at else None,
        "updated_at": video.updated_at.isoformat() if video.updated_at else None
    }
    
    # Check Celery pipeline status if applicable
    pipeline_task_id = video.metadata.get("pipeline_task_id")
    if pipeline_task_id:
        try:
            celery_status = get_pipeline_status(pipeline_task_id)
            pipeline_info["celery_pipeline"] = celery_status
        except Exception as e:
            pipeline_info["celery_pipeline"] = {"error": str(e)}
    
    # Add cost information if available
    if video.total_cost:
        pipeline_info["costs"] = {
            "total_cost": float(video.total_cost),
            "content_cost": float(video.content_cost or 0),
            "within_budget": float(video.total_cost) <= settings.MAX_COST_PER_VIDEO
        }
    
    # Add file paths if available
    if video.video_file_path:
        pipeline_info["output_files"] = {
            "video_file": video.video_file_path,
            "thumbnail_file": video.thumbnail_file_path,
            "youtube_video_id": video.youtube_video_id,
            "youtube_url": f"https://youtube.com/watch?v={video.youtube_video_id}" if video.youtube_video_id else None
        }
    
    return pipeline_info


@router.post("/{video_id}/cancel-pipeline")
async def cancel_video_pipeline(
    video_id: str,
    current_user: User = Depends(get_current_user),
    video_repo: VideoRepository = Depends(get_video_repo)
):
    """
    Cancel a running video generation pipeline.
    """
    # Verify ownership
    if not await video_repo.check_ownership(video_id, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found or access denied"
        )
    
    video = await video_repo.get_by_id(video_id)
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found"
        )
    
    # Check if video can be cancelled
    if video.status not in [VideoStatus.PENDING, VideoStatus.PROCESSING]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel video in {video.status.value} status"
        )
    
    # Cancel Celery pipeline if applicable
    pipeline_task_id = video.metadata.get("pipeline_task_id")
    celery_cancelled = False
    if pipeline_task_id:
        try:
            celery_cancelled = cancel_pipeline(pipeline_task_id)
        except Exception as e:
            logger.warning(f"Failed to cancel Celery pipeline: {str(e)}")
    
    # Update video status
    await video_repo.update_status(video_id, VideoStatus.CANCELLED)
    
    return {
        "video_id": video_id,
        "status": "cancelled",
        "celery_pipeline_cancelled": celery_cancelled,
        "cancelled_at": datetime.utcnow().isoformat()
    }