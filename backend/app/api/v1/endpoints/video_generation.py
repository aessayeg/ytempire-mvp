"""
Video Generation API Endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from app.db.session import get_db
from app.models.user import User
from app.models.channel import Channel
from app.models.video import Video, VideoStatus
from app.api.v1.endpoints.auth import get_current_verified_user
from app.services.video_generation_pipeline import (
    VideoGenerationPipeline,
    VideoGenerationConfig,
)
from app.services.video_generation_pipeline import (
    enhanced_orchestrator,
    generate_personalized_video,
)
from app.services.websocket_manager import ConnectionManager
from app.core.config import settings
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()
ws_manager = ConnectionManager()


# Request/Response Models
class VideoGenerationRequest(BaseModel):
    """Request model for video generation"""

    channel_id: str
    topic: str
    style: str = Field(
        default="educational",
        description="Video style: educational, entertainment, tech, etc.",
    )
    length: str = Field(
        default="medium",
        description="Video length: short (1-3min), medium (3-7min), long (7-15min)",
    )
    target_audience: str = Field(
        default="general", description="Target audience demographics"
    )
    keywords: Optional[List[str]] = Field(
        default=None, description="SEO keywords to include"
    )
    voice_provider: str = Field(
        default="elevenlabs", description="Voice provider: elevenlabs or google_tts"
    )
    voice_id: Optional[str] = Field(
        default=None, description="Specific voice ID to use"
    )
    thumbnail_style: str = Field(default="modern", description="Thumbnail style")
    auto_publish: bool = Field(
        default=False, description="Automatically publish to YouTube"
    )
    optimize_for_cost: bool = Field(
        default=True, description="Optimize for cost vs quality"
    )
    max_cost: float = Field(default=3.00, description="Maximum cost per video")
    quality_threshold: float = Field(
        default=0.7, description="Minimum quality score (0-1)"
    )
    use_ml_personalization: bool = Field(
        default=True, description="Use ML-powered personalization"
    )
    use_performance_prediction: bool = Field(
        default=True, description="Use ML to predict performance"
    )


class VideoGenerationResponse(BaseModel):
    """Response model for video generation"""

    video_id: str
    status: str
    message: str
    estimated_completion_time: Optional[int] = None  # seconds
    estimated_cost: Optional[float] = None


class VideoGenerationStatus(BaseModel):
    """Status model for video generation"""

    video_id: str
    status: str
    progress: float  # 0-100
    current_stage: str
    stages_completed: List[str]
    total_cost: float
    errors: List[str]
    created_at: datetime
    updated_at: datetime


@router.post("/generate", response_model=VideoGenerationResponse)
async def generate_video(
    request: VideoGenerationRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
) -> VideoGenerationResponse:
    """
    Generate a new video with AI

    This endpoint triggers the video generation pipeline which includes:
    1. Script generation using GPT-4
    2. Voice synthesis using ElevenLabs/Google TTS
    3. Thumbnail generation using DALL-E
    4. Video assembly
    5. Optional YouTube upload
    """

    # Verify user owns the channel
    result = await db.execute(
        select(Channel).where(
            Channel.id == request.channel_id, Channel.owner_id == current_user.id
        )
    )
    channel = result.scalar_one_or_none()

    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    # Check user's video generation limit
    if current_user.total_videos_generated >= current_user.videos_per_day_limit:
        raise HTTPException(
            status_code=429,
            detail=f"Daily video limit reached ({current_user.videos_per_day_limit})",
        )

    # Check user's budget limit
    if current_user.total_cost_incurred >= current_user.monthly_budget_limit:
        raise HTTPException(
            status_code=429,
            detail=f"Monthly budget limit reached (${current_user.monthly_budget_limit})",
        )

    # Create video generation config with ML parameters
    config_dict = {
        "channel_id": request.channel_id,
        "user_id": current_user.id,
        "topic": request.topic,
        "style": request.style,
        "length": request.length,
        "target_audience": request.target_audience,
        "keywords": request.keywords,
        "voice_provider": request.voice_provider,
        "voice_id": request.voice_id,
        "thumbnail_style": request.thumbnail_style,
        "auto_publish": request.auto_publish,
        "optimize_for_cost": request.optimize_for_cost,
        "max_cost": request.max_cost,
        "quality_threshold": request.quality_threshold,
    }

    # Add ML parameters if available
    if settings.ML_ENABLED:
        config_dict["use_ml_personalization"] = request.use_ml_personalization
        config_dict["use_performance_prediction"] = request.use_performance_prediction

    config = VideoGenerationConfig(**config_dict)

    # Create initial video record
    video = Video(
        channel_id=request.channel_id,
        title=f"Generating: {request.topic}",
        description="Video generation in progress...",
        generation_status=VideoStatus.PENDING,
        tags=[request.topic, request.style, request.length],  # Store as tags instead
    )
    db.add(video)
    await db.commit()
    await db.refresh(video)

    # Start generation in background
    background_tasks.add_task(run_video_generation, video.id, config, current_user.id)

    # Estimate completion time based on video length
    estimated_time = {
        "short": 120,  # 2 minutes
        "medium": 180,  # 3 minutes
        "long": 300,  # 5 minutes
    }.get(request.length, 180)

    # Estimate cost
    estimated_cost = {"short": 1.50, "medium": 2.00, "long": 2.50}.get(
        request.length, 2.00
    )

    if request.optimize_for_cost:
        estimated_cost *= 0.7  # 30% reduction with optimization

    return VideoGenerationResponse(
        video_id=video.id,
        status="processing",
        message="Video generation started successfully",
        estimated_completion_time=estimated_time,
        estimated_cost=estimated_cost,
    )


@router.get("/status/{video_id}", response_model=VideoGenerationStatus)
async def get_generation_status(
    video_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
) -> VideoGenerationStatus:
    """Get the status of a video generation job"""

    # Get video with ownership check
    result = await db.execute(
        select(Video)
        .join(Channel)
        .where(Video.id == video_id, Channel.owner_id == current_user.id)
    )
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Calculate progress based on status
    progress_map = {
        VideoStatus.PENDING: 0,
        VideoStatus.PROCESSING: 50,
        VideoStatus.COMPLETED: 100,
        VideoStatus.FAILED: 0,
        VideoStatus.PUBLISHED: 100,
    }

    # Get stages based on status
    stages_completed = []
    if video.generation_status in [VideoStatus.COMPLETED, VideoStatus.PUBLISHED]:
        stages_completed = [
            "script_generation",
            "voice_synthesis",
            "thumbnail_generation",
            "video_assembly",
        ]
        if video.generation_status == VideoStatus.PUBLISHED:
            stages_completed.append("youtube_upload")

    return VideoGenerationStatus(
        video_id=video.id,
        status=video.generation_status,
        progress=progress_map.get(video.generation_status, 0),
        current_stage=video.generation_status,
        stages_completed=stages_completed,
        total_cost=video.total_cost or 0.0,
        errors=[],  # Video model doesn't have error_message field
        created_at=video.created_at,
        updated_at=video.updated_at or video.created_at,
    )


@router.post("/cancel/{video_id}")
async def cancel_generation(
    video_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
) -> Dict[str, Any]:
    """Cancel a video generation job"""

    # Get video with ownership check
    result = await db.execute(
        select(Video)
        .join(Channel)
        .where(Video.id == video_id, Channel.owner_id == current_user.id)
    )
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    if video.generation_status not in [VideoStatus.PENDING, VideoStatus.PROCESSING]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel video in {video.generation_status} status",
        )

    # Update video status
    video.generation_status = VideoStatus.FAILED
    None  # Video model doesn't have error_message = "Cancelled by user"
    video.updated_at = datetime.utcnow()

    await db.commit()

    return {
        "video_id": video_id,
        "status": "cancelled",
        "message": "Video generation cancelled successfully",
    }


@router.get("/history", response_model=List[VideoGenerationStatus])
async def get_generation_history(
    channel_id: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
) -> List[VideoGenerationStatus]:
    """Get video generation history for user's channels"""

    query = select(Video).join(Channel).where(Channel.owner_id == current_user.id)

    if channel_id:
        query = query.where(Video.channel_id == channel_id)

    query = query.order_by(Video.created_at.desc()).limit(limit).offset(offset)

    result = await db.execute(query)
    videos = result.scalars().all()

    return [
        VideoGenerationStatus(
            video_id=video.id,
            status=video.generation_status,
            progress=100 if video.generation_status == VideoStatus.COMPLETED else 50,
            current_stage=video.generation_status,
            stages_completed=[],
            total_cost=video.total_cost or 0.0,
            errors=[],  # Video model doesn't have error_message field
            created_at=video.created_at,
            updated_at=video.updated_at or video.created_at,
        )
        for video in videos
    ]


async def run_video_generation(
    video_id: str, config: VideoGenerationConfig, user_id: str
):
    """Background task to run video generation"""
    try:
        # Send WebSocket update
        await ws_manager.send_personal_message(
            f"Video generation started for: {config.topic}", user_id
        )

        # Check if ML features are enabled and requested
        use_ml = (
            settings.ML_ENABLED
            and hasattr(config, "use_ml_personalization")
            and config.use_ml_personalization
        )

        if use_ml:
            # Use ML-enhanced generation
            from app.db.session import async_session

            async with async_session() as session:
                result = await generate_personalized_video(
                    channel_id=config.channel_id, topic=config.topic, db=session
                )
        else:
            # Use standard pipeline
            pipeline = VideoGenerationPipeline()
            result = await pipeline.generate_video(config)

        # Send completion notification
        total_cost = (
            result.get("metrics", {}).get("total_cost", 0)
            if use_ml
            else result.get("total_cost", 0)
        )
        await ws_manager.send_personal_message(
            f"Video generation completed! Cost: ${total_cost:.2f}", user_id
        )

        logger.info(f"Video generation completed: {video_id} (ML: {use_ml})")

    except Exception as e:
        logger.error(f"Video generation failed for {video_id}: {e}")

        # Send error notification
        await ws_manager.send_personal_message(
            f"Video generation failed: {str(e)}", user_id
        )

        # Update video status to failed
        from app.db.session import async_session

        async with async_session() as session:
            result = await session.execute(select(Video).where(Video.id == video_id))
            video = result.scalar_one_or_none()
            if video:
                video.generation_status = VideoStatus.FAILED
                None  # Video model doesn't have error_message = str(e)
                video.updated_at = datetime.utcnow()
                await session.commit()
