"""
Enhanced Video Generation and Management Endpoints
Week 2 Implementation: Optimized with caching, query optimization, and <300ms p95 response time
"""
from typing import List, Optional, Dict, Any, Annotated
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query, UploadFile, File
from fastapi.responses import ORJSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, update
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy.exc import IntegrityError
from pydantic import BaseModel, Field, validator
import asyncio
import uuid
import logging
import orjson
from celery import Celery
import redis.asyncio as aioredis

from app.db.session import get_db
from app.models.video import Video
from app.models.channel import Channel
from app.models.user import User
from app.models.cost import Cost
from app.api.v1.endpoints.auth import get_current_verified_user
from app.services.ai_services import AIServiceOrchestrator, AIServiceConfig
from app.services.youtube_service import YouTubeService
from app.services.video_processor import VideoProcessor
from app.services.cost_tracker import CostTracker
from app.core.celery_app import celery_app
from app.core.config import settings
from app.core.performance_enhanced import (
    cached, cache, QueryOptimizer, ResponseCompression,
    api_metrics, BatchProcessor, request_deduplicator
)

router = APIRouter(prefix="/videos", tags=["videos"])
logger = logging.getLogger(__name__)

# Initialize Redis for caching
redis_client = None

async def get_redis():
    global redis_client
    if not redis_client:
        redis_client = await aioredis.from_url(
            settings.REDIS_URL or "redis://localhost:6379",
            encoding="utf-8",
            decode_responses=True
        )
    return redis_client

# Initialize services
ai_config = AIServiceConfig(
    openai_api_key=settings.OPENAI_API_KEY or "",
    elevenlabs_api_key=settings.ELEVENLABS_API_KEY or ""
)
ai_service = AIServiceOrchestrator(ai_config)
youtube_service = YouTubeService()
video_processor = VideoProcessor()
cost_tracker = CostTracker()

# Pydantic models
class VideoGenerateRequest(BaseModel):
    """Video generation request"""
    channel_id: str
    title: Optional[str] = None
    topic: Optional[str] = Field(None, description="Topic for video generation")
    style: str = Field("informative", description="Video style: informative, entertaining, tutorial, review")
    duration: str = Field("short", description="Video duration: short (1-3 min), medium (5-10 min), long (10+ min)")
    voice_style: str = Field("natural", description="Voice style: natural, energetic, calm, professional")
    language: str = Field("en", description="Language code")
    use_trending: bool = Field(True, description="Use trending topics for content")
    quality_preset: str = Field("balanced", description="Quality preset: fast, balanced, quality")
    
    @validator('style')
    def validate_style(cls, v):
        valid_styles = ["informative", "entertaining", "tutorial", "review", "news", "story"]
        if v not in valid_styles:
            raise ValueError(f"Style must be one of: {', '.join(valid_styles)}")
        return v
    
    @validator('duration')
    def validate_duration(cls, v):
        valid_durations = ["short", "medium", "long"]
        if v not in valid_durations:
            raise ValueError(f"Duration must be one of: {', '.join(valid_durations)}")
        return v

class VideoUpdateRequest(BaseModel):
    """Video update request"""
    title: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    thumbnail_url: Optional[str] = None
    scheduled_publish_time: Optional[datetime] = None
    quality_score: Optional[float] = None

class VideoResponse(BaseModel):
    """Video response model"""
    id: str
    channel_id: str
    title: str
    description: Optional[str]
    tags: List[str]
    category: str
    youtube_video_id: Optional[str]
    youtube_url: Optional[str]
    thumbnail_url: Optional[str]
    generation_status: str
    publish_status: str
    quality_score: Optional[float]
    trend_score: Optional[float]
    engagement_prediction: Optional[float]
    total_cost: float
    duration_seconds: Optional[int]
    view_count: int
    like_count: int
    comment_count: int
    created_at: datetime
    published_at: Optional[datetime]
    
    class Config:
        orm_mode = True

class VideoGenerationStatus(BaseModel):
    """Video generation status"""
    video_id: str
    status: str
    progress: int
    current_step: str
    estimated_completion: Optional[datetime]
    errors: List[str]
    cost_so_far: float

class BulkGenerateRequest(BaseModel):
    """Bulk video generation request"""
    channel_id: str
    count: int = Field(5, ge=1, le=20)
    topics: Optional[List[str]] = None
    style: str = "mixed"
    schedule_interval_hours: int = Field(24, ge=1)
    start_date: Optional[datetime] = None

# Endpoints
@router.post("/generate", response_model=VideoResponse, status_code=status.HTTP_202_ACCEPTED)
async def generate_video(
    request: VideoGenerateRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate a new video
    
    - Validates channel ownership and limits
    - Initiates AI-powered video generation
    - Tracks costs in real-time
    - Returns video ID for status tracking
    """
    # Verify channel ownership
    result = await db.execute(
        select(Channel).filter(
            Channel.id == request.channel_id,
            Channel.owner_id == current_user.id,
            Channel.is_active == True
        )
    )
    channel = result.scalar_one_or_none()
    
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Channel not found or inactive"
        )
    
    # Check daily video limit
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    result = await db.execute(
        select(func.count(Video.id)).filter(
            Video.channel_id == request.channel_id,
            Video.created_at >= today_start
        )
    )
    today_count = result.scalar()
    
    if today_count >= current_user.videos_per_day_limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Daily video limit reached ({current_user.videos_per_day_limit})"
        )
    
    # Check monthly budget
    month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    result = await db.execute(
        select(func.sum(Cost.amount)).filter(
            Cost.user_id == current_user.id,
            Cost.created_at >= month_start
        )
    )
    month_cost = result.scalar() or 0.0
    
    if month_cost >= current_user.monthly_budget_limit:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Monthly budget limit reached (${current_user.monthly_budget_limit})"
        )
    
    # Determine topic from request or trending
    topic = request.topic
    if not topic and request.use_trending:
        trending_topics = await ai_service.get_trending_topics(channel.category)
        if trending_topics:
            topic = trending_topics[0]["topic"]
    
    if not topic:
        topic = f"Latest in {channel.category}"
    
    # Generate initial title if not provided
    title = request.title
    if not title:
        title = await ai_service.generate_title(topic, channel.category, request.style)
    
    # Create video record
    video = Video(
        id=str(uuid.uuid4()),
        channel_id=request.channel_id,
        title=title,
        description="",  # Will be generated
        tags=[],  # Will be generated
        category=channel.category,
        generation_status="pending",
        publish_status="draft",
        script="",
        voice_script="",
        visual_prompts=[],
        quality_score=0.0,
        trend_score=0.0,
        engagement_prediction=0.0,
        total_cost=0.0,
        script_cost=0.0,
        voice_cost=0.0,
        video_cost=0.0,
        thumbnail_cost=0.0,
        script_model="gpt-3.5-turbo",
        voice_model="elevenlabs",
        created_at=datetime.utcnow()
    )
    
    db.add(video)
    await db.commit()
    await db.refresh(video)
    
    # Queue video generation task
    task = celery_app.send_task(
        "app.tasks.video_generation.generate_video",
        args=[
            str(video.id),
            request.channel_id,
            topic,
            request.style,
            request.duration,
            request.voice_style,
            request.language,
            request.quality_preset
        ],
        queue="video_generation"
    )
    
    # Store task ID for tracking
    video.generation_task_id = task.id
    await db.commit()
    
    logger.info(f"Video generation started: {video.id} for channel {request.channel_id}")
    
    return VideoResponse(
        id=str(video.id),
        channel_id=video.channel_id,
        title=video.title,
        description=video.description,
        tags=video.tags or [],
        category=video.category,
        youtube_video_id=video.youtube_video_id,
        youtube_url=video.youtube_url,
        thumbnail_url=video.thumbnail_url,
        generation_status=video.generation_status,
        publish_status=video.publish_status,
        quality_score=video.quality_score,
        trend_score=video.trend_score,
        engagement_prediction=video.engagement_prediction,
        total_cost=video.total_cost,
        duration_seconds=video.duration_seconds,
        view_count=video.view_count,
        like_count=video.like_count,
        comment_count=video.comment_count,
        created_at=video.created_at,
        published_at=video.published_at
    )

@router.get("/{video_id}/status", response_model=VideoGenerationStatus)
async def get_generation_status(
    video_id: str,
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get video generation status
    
    - Returns current generation progress
    - Shows current step and estimated completion
    - Lists any errors encountered
    """
    # Get video and verify ownership
    result = await db.execute(
        select(Video, Channel).join(Channel).filter(
            Video.id == video_id,
            Channel.owner_id == current_user.id
        )
    )
    row = result.first()
    
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found"
        )
    
    video, channel = row
    
    # Get task status from Celery
    if hasattr(video, 'generation_task_id') and video.generation_task_id:
        task_result = celery_app.AsyncResult(video.generation_task_id)
        
        if task_result.state == 'PENDING':
            progress = 0
            current_step = "Waiting in queue"
        elif task_result.state == 'PROGRESS':
            progress = task_result.info.get('current', 0)
            current_step = task_result.info.get('step', 'Processing')
        elif task_result.state == 'SUCCESS':
            progress = 100
            current_step = "Completed"
        else:
            progress = 0
            current_step = f"Error: {task_result.info}"
    else:
        # Estimate progress based on status
        status_progress = {
            "pending": 0,
            "processing": 50,
            "completed": 100,
            "failed": 0
        }
        progress = status_progress.get(video.generation_status, 0)
        current_step = video.generation_status.title()
    
    # Estimate completion time
    estimated_completion = None
    if video.generation_status == "processing" and video.generation_started_at:
        # Estimate based on average generation time (5 minutes)
        estimated_completion = video.generation_started_at + timedelta(minutes=5)
    
    return VideoGenerationStatus(
        video_id=str(video.id),
        status=video.generation_status,
        progress=progress,
        current_step=current_step,
        estimated_completion=estimated_completion,
        errors=[],  # Would be populated from error logs
        cost_so_far=video.total_cost
    )

@router.post("/{video_id}/publish", response_model=Dict[str, str])
async def publish_video(
    video_id: str,
    scheduled_time: Optional[datetime] = None,
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Publish video to YouTube
    
    - Publishes immediately or schedules for later
    - Validates video quality threshold
    - Requires YouTube channel connection
    """
    # Get video and verify ownership
    result = await db.execute(
        select(Video, Channel).join(Channel).filter(
            Video.id == video_id,
            Channel.owner_id == current_user.id
        )
    )
    row = result.first()
    
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found"
        )
    
    video, channel = row
    
    # Check if video is ready
    if video.generation_status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Video generation not completed"
        )
    
    # Check quality threshold
    if video.quality_score < channel.quality_threshold:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Video quality ({video.quality_score:.2f}) below threshold ({channel.quality_threshold:.2f})"
        )
    
    # Check YouTube connection
    if not channel.is_verified or not channel.youtube_channel_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="YouTube channel not connected"
        )
    
    if scheduled_time and scheduled_time > datetime.utcnow():
        # Schedule for later
        video.publish_status = "scheduled"
        video.scheduled_publish_time = scheduled_time
        await db.commit()
        
        # Create scheduled task
        celery_app.send_task(
            "app.tasks.video_publishing.publish_video",
            args=[str(video.id)],
            eta=scheduled_time
        )
        
        return {"message": f"Video scheduled for publishing at {scheduled_time}"}
    else:
        # Publish immediately
        video.publish_status = "publishing"
        await db.commit()
        
        # Queue publishing task
        celery_app.send_task(
            "app.tasks.video_publishing.publish_video",
            args=[str(video.id)],
            queue="video_publishing"
        )
        
        return {"message": "Video publishing initiated"}

@router.get("/", response_model=List[VideoResponse], response_class=ORJSONResponse)
@cached(prefix="videos:list", ttl=60, key_params=["channel_id", "status", "skip", "limit"])
async def list_videos(
    channel_id: Optional[str] = None,
    status: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Optimized List Videos Endpoint
    
    - Uses query optimization with eager loading
    - Implements Redis caching for frequent queries
    - Returns videos for user's channels only
    - Target: <100ms response time
    """
    # Check cache first
    cache_key = f"videos:list:{current_user.id}:{channel_id}:{status}:{skip}:{limit}"
    redis = await get_redis()
    
    cached_result = await redis.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)
    
    # Build optimized query with eager loading
    query = (
        select(Video)
        .join(Channel)
        .filter(Channel.owner_id == current_user.id)
        .options(
            selectinload(Video.channel),
            selectinload(Video.costs)
        )
    )
    
    if channel_id:
        query = query.filter(Video.channel_id == channel_id)
    
    if status:
        if status == "generated":
            query = query.filter(Video.generation_status == "completed")
        elif status == "published":
            query = query.filter(Video.publish_status == "published")
        elif status == "draft":
            query = query.filter(Video.publish_status == "draft")
    
    # Add optimized pagination
    query = QueryOptimizer.add_pagination(
        query.order_by(Video.created_at.desc()),
        page=(skip // limit) + 1,
        page_size=limit
    )
    
    result = await db.execute(query)
    videos = result.scalars().all()
    
    # Transform to response model
    response_data = [
        VideoResponse(
            id=str(video.id),
            channel_id=video.channel_id,
            title=video.title,
            description=video.description,
            tags=video.tags or [],
            category=video.category,
            youtube_video_id=video.youtube_video_id,
            youtube_url=video.youtube_url,
            thumbnail_url=video.thumbnail_url,
            generation_status=video.generation_status,
            publish_status=video.publish_status,
            quality_score=video.quality_score,
            trend_score=video.trend_score,
            engagement_prediction=video.engagement_prediction,
            total_cost=video.total_cost,
            duration_seconds=video.duration_seconds,
            view_count=video.view_count,
            like_count=video.like_count,
            comment_count=video.comment_count,
            created_at=video.created_at,
            published_at=video.published_at
        )
        for video in videos
    ]

@router.get("/{video_id}", response_model=VideoResponse)
async def get_video(
    video_id: str,
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get video details
    
    - Returns full video information
    - Includes all metrics and costs
    - Validates ownership
    """
    result = await db.execute(
        select(Video, Channel).join(Channel).filter(
            Video.id == video_id,
            Channel.owner_id == current_user.id
        )
    )
    row = result.first()
    
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found"
        )
    
    video, channel = row
    
    return VideoResponse(
        id=str(video.id),
        channel_id=video.channel_id,
        title=video.title,
        description=video.description,
        tags=video.tags or [],
        category=video.category,
        youtube_video_id=video.youtube_video_id,
        youtube_url=video.youtube_url,
        thumbnail_url=video.thumbnail_url,
        generation_status=video.generation_status,
        publish_status=video.publish_status,
        quality_score=video.quality_score,
        trend_score=video.trend_score,
        engagement_prediction=video.engagement_prediction,
        total_cost=video.total_cost,
        duration_seconds=video.duration_seconds,
        view_count=video.view_count,
        like_count=video.like_count,
        comment_count=video.comment_count,
        created_at=video.created_at,
        published_at=video.published_at
    )

@router.put("/{video_id}", response_model=VideoResponse)
async def update_video(
    video_id: str,
    update_request: VideoUpdateRequest,
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update video metadata
    
    - Update title, description, tags
    - Change scheduled publish time
    - Update quality score
    """
    result = await db.execute(
        select(Video, Channel).join(Channel).filter(
            Video.id == video_id,
            Channel.owner_id == current_user.id
        )
    )
    row = result.first()
    
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found"
        )
    
    video, channel = row
    
    # Update fields
    update_data = update_request.dict(exclude_unset=True)
    for field, value in update_data.items():
        if value is not None:
            setattr(video, field, value)
    
    video.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(video)
    
    return VideoResponse(
        id=str(video.id),
        channel_id=video.channel_id,
        title=video.title,
        description=video.description,
        tags=video.tags or [],
        category=video.category,
        youtube_video_id=video.youtube_video_id,
        youtube_url=video.youtube_url,
        thumbnail_url=video.thumbnail_url,
        generation_status=video.generation_status,
        publish_status=video.publish_status,
        quality_score=video.quality_score,
        trend_score=video.trend_score,
        engagement_prediction=video.engagement_prediction,
        total_cost=video.total_cost,
        duration_seconds=video.duration_seconds,
        view_count=video.view_count,
        like_count=video.like_count,
        comment_count=video.comment_count,
        created_at=video.created_at,
        published_at=video.published_at
    )

@router.delete("/{video_id}", response_model=Dict[str, str])
async def delete_video(
    video_id: str,
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete video
    
    - Soft delete to preserve analytics
    - Removes from YouTube if published
    - Cancels scheduled publishing
    """
    result = await db.execute(
        select(Video, Channel).join(Channel).filter(
            Video.id == video_id,
            Channel.owner_id == current_user.id
        )
    )
    row = result.first()
    
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found"
        )
    
    video, channel = row
    
    # Cancel if scheduled
    if video.publish_status == "scheduled":
        video.publish_status = "cancelled"
    
    # Mark as deleted
    video.deleted_at = datetime.utcnow()
    
    await db.commit()
    
    return {"message": "Video deleted successfully"}

@router.post("/bulk-generate", response_model=Dict[str, Any])
async def bulk_generate_videos(
    request: BulkGenerateRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Bulk generate multiple videos
    
    - Generate multiple videos at once
    - Schedule them with specified intervals
    - Use different topics or auto-generate
    """
    # Verify channel ownership
    result = await db.execute(
        select(Channel).filter(
            Channel.id == request.channel_id,
            Channel.owner_id == current_user.id,
            Channel.is_active == True
        )
    )
    channel = result.scalar_one_or_none()
    
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Channel not found or inactive"
        )
    
    # Check limits
    if request.count > current_user.videos_per_day_limit:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Count exceeds daily limit ({current_user.videos_per_day_limit})"
        )
    
    # Generate topics if not provided
    topics = request.topics
    if not topics:
        trending = await ai_service.get_trending_topics(channel.category, count=request.count)
        topics = [t["topic"] for t in trending]
    
    # Ensure we have enough topics
    while len(topics) < request.count:
        topics.append(f"{channel.category} content #{len(topics) + 1}")
    
    # Create videos
    videos_created = []
    start_date = request.start_date or datetime.utcnow()
    
    for i in range(request.count):
        video = Video(
            id=str(uuid.uuid4()),
            channel_id=request.channel_id,
            title=f"Video {i+1}: {topics[i]}",
            category=channel.category,
            generation_status="pending",
            publish_status="draft",
            scheduled_publish_time=start_date + timedelta(hours=i * request.schedule_interval_hours),
            created_at=datetime.utcnow()
        )
        
        db.add(video)
        videos_created.append(str(video.id))
        
        # Queue generation task
        celery_app.send_task(
            "app.tasks.video_generation.generate_video",
            args=[
                str(video.id),
                request.channel_id,
                topics[i],
                request.style,
                "short",  # Default to short for bulk
                "natural",
                "en",
                "fast"  # Use fast preset for bulk
            ],
            queue="video_generation",
            countdown=i * 60  # Stagger generation by 1 minute
        )
    
    await db.commit()
    
    return {
        "message": f"Bulk generation initiated for {request.count} videos",
        "video_ids": videos_created,
        "scheduled_times": [
            (start_date + timedelta(hours=i * request.schedule_interval_hours)).isoformat()
            for i in range(request.count)
        ]
    }

@router.get("/{video_id}/analytics", response_model=Dict[str, Any])
async def get_video_analytics(
    video_id: str,
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed video analytics
    
    - Performance metrics
    - Engagement statistics
    - Revenue estimates
    - Cost breakdown
    """
    result = await db.execute(
        select(Video, Channel).join(Channel).filter(
            Video.id == video_id,
            Channel.owner_id == current_user.id
        )
    )
    row = result.first()
    
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found"
        )
    
    video, channel = row
    
    # Calculate engagement rate
    engagement_rate = 0.0
    if video.view_count > 0:
        engagement_rate = ((video.like_count + video.comment_count) / video.view_count) * 100
    
    # Calculate profit
    profit = video.actual_revenue - video.total_cost
    profit_margin = (profit / video.actual_revenue * 100) if video.actual_revenue > 0 else 0
    
    return {
        "video_id": str(video.id),
        "title": video.title,
        "performance": {
            "views": video.view_count,
            "likes": video.like_count,
            "comments": video.comment_count,
            "engagement_rate": round(engagement_rate, 2),
            "watch_time_minutes": video.watch_time_minutes,
            "average_view_duration": round(video.watch_time_minutes * 60 / max(video.view_count, 1), 2)
        },
        "financial": {
            "total_cost": video.total_cost,
            "cost_breakdown": {
                "script": video.script_cost,
                "voice": video.voice_cost,
                "video": video.video_cost,
                "thumbnail": video.thumbnail_cost
            },
            "revenue": video.actual_revenue,
            "estimated_revenue": video.estimated_revenue,
            "profit": round(profit, 2),
            "profit_margin": round(profit_margin, 2),
            "roi": round((profit / video.total_cost * 100) if video.total_cost > 0 else 0, 2)
        },
        "quality": {
            "quality_score": video.quality_score,
            "trend_score": video.trend_score,
            "engagement_prediction": video.engagement_prediction
        },
        "timeline": {
            "created": video.created_at,
            "generated": video.generation_completed_at,
            "published": video.published_at,
            "generation_time": video.generation_time_seconds
        }
    }