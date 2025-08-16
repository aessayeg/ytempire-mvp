"""
Enhanced Channel Management with Multi-Channel Architecture
Week 2 Implementation: Channel isolation, quota management, 5+ channels per user
"""
from typing import List, Annotated, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from fastapi.responses import ORJSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, update, case
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy.exc import IntegrityError
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, validator
import secrets
import logging
import asyncio
import orjson
import redis.asyncio as aioredis

from app.db.session import get_db
from app.models.channel import Channel
from app.models.user import User
from app.models.video import Video
from app.schemas.channel import ChannelCreate, ChannelUpdate, ChannelResponse
from app.api.v1.endpoints.auth import get_current_user, get_current_verified_user
from app.services.youtube_service import YouTubeService
from app.services.analytics_service import AnalyticsService
from app.core.performance_enhanced import cached, cache, QueryOptimizer

router = APIRouter()
logger = logging.getLogger(__name__)

# Redis for channel quota tracking
redis_client = None


async def get_redis():
    global redis_client
    if not redis_client:
        redis_client = await aioredis.from_url(
            "redis://localhost:6379", encoding="utf-8", decode_responses=True
        )
    return redis_client


# Initialize services
youtube_service = YouTubeService()
analytics_service = AnalyticsService()


# Enhanced Pydantic models for Multi-Channel Support
class ChannelStats(BaseModel):
    """Enhanced channel statistics with quota tracking"""

    channel_id: str
    channel_name: str
    total_videos: int
    published_videos: int
    total_views: int
    total_likes: int
    total_comments: int
    total_revenue: float
    total_cost: float
    profit_margin: float
    avg_views_per_video: float
    avg_engagement_rate: float
    best_performing_video: Optional[Dict[str, Any]]
    growth_rate: float
    last_video_date: Optional[datetime]
    # New quota fields
    daily_quota_used: int
    daily_quota_limit: int
    weekly_videos_generated: int
    channel_health_score: float
    isolation_namespace: str


class YouTubeConnect(BaseModel):
    """YouTube channel connection request"""

    youtube_channel_id: str
    youtube_api_key: str
    youtube_refresh_token: Optional[str] = None


class ChannelQuota(BaseModel):
    """Channel quota management"""

    channel_id: str
    daily_video_limit: int = 10
    weekly_video_limit: int = 50
    monthly_video_limit: int = 200
    api_quota_units: int = 10000
    storage_quota_gb: int = 100
    concurrent_generations: int = 2


class MultiChannelRequest(BaseModel):
    """Request for multi-channel operations"""

    channel_ids: List[str]
    operation: str = Field(
        ..., description="Operation type: publish, schedule, analyze"
    )
    parameters: Optional[Dict[str, Any]] = None


class ChannelIsolationManager:
    """Manages channel isolation and resource allocation"""

    @staticmethod
    async def get_channel_namespace(channel_id: str) -> str:
        """Get isolated namespace for channel"""
        return f"channel:{channel_id}"

    @staticmethod
    async def check_channel_quota(
        channel_id: str, resource_type: str = "video"
    ) -> bool:
        """Check if channel has available quota"""
        redis = await get_redis()

        # Get current usage
        daily_key = f"quota:{channel_id}:daily:{datetime.utcnow().date()}"
        current_usage = await redis.get(daily_key) or 0

        # Get limits (default to 10 videos per day)
        limit_key = f"quota:{channel_id}:limit:daily"
        daily_limit = await redis.get(limit_key) or 10

        return int(current_usage) < int(daily_limit)

    @staticmethod
    async def consume_quota(channel_id: str, amount: int = 1) -> bool:
        """Consume channel quota"""
        redis = await get_redis()

        # Check if quota available
        if not await ChannelIsolationManager.check_channel_quota(channel_id):
            return False

        # Increment usage
        daily_key = f"quota:{channel_id}:daily:{datetime.utcnow().date()}"
        await redis.incr(daily_key)
        await redis.expire(daily_key, 86400)  # Expire after 24 hours

        return True

    @staticmethod
    async def get_channel_resources(channel_id: str) -> Dict[str, Any]:
        """Get allocated resources for channel"""
        redis = await get_redis()

        # Get resource allocation
        resources = {
            "cpu_cores": await redis.get(f"resources:{channel_id}:cpu") or 2,
            "memory_gb": await redis.get(f"resources:{channel_id}:memory") or 4,
            "gpu_allocation": await redis.get(f"resources:{channel_id}:gpu") or 0.25,
            "storage_gb": await redis.get(f"resources:{channel_id}:storage") or 100,
            "priority": await redis.get(f"resources:{channel_id}:priority") or "normal",
        }

        return resources


@router.post("/", response_model=ChannelResponse, status_code=status.HTTP_201_CREATED)
async def create_channel(
    channel_data: ChannelCreate,
    background_tasks: BackgroundTasks,
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new YouTube channel with enhanced features

    - Validates user channel limit based on subscription tier
    - Creates channel with initial configuration
    - Optionally connects and verifies YouTube account
    - Initiates analytics fetch if YouTube connected
    """
    # Check channel limit for user's subscription tier
    result = await db.execute(
        select(func.count(Channel.id)).filter(
            Channel.owner_id == current_user.id, Channel.is_active == True
        )
    )
    active_channels = result.scalar()

    if active_channels >= current_user.channels_limit:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Channel limit reached. Your {current_user.subscription_tier} plan allows {current_user.channels_limit} channel(s).",
        )

    # Check if channel name already exists for this user
    result = await db.execute(
        select(Channel).filter(
            Channel.owner_id == current_user.id,
            Channel.name == channel_data.channel_name,
        )
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="You already have a channel with this name",
        )

    # Create new channel with enhanced fields
    channel_dict = channel_data.dict()

    # Map schema fields to model fields
    mapped_dict = {}
    if "channel_name" in channel_dict:
        mapped_dict["name"] = channel_dict["channel_name"]
    if "channel_description" in channel_dict:
        mapped_dict["description"] = channel_dict["channel_description"]
    if "youtube_channel_id" in channel_dict:
        mapped_dict["youtube_channel_id"] = channel_dict["youtube_channel_id"]
    if "niche" in channel_dict:
        mapped_dict["category"] = channel_dict["niche"]  # Map niche to category
    if "content_type" in channel_dict:
        mapped_dict["content_type"] = channel_dict["content_type"]

    # Only add fields that exist in the Channel model
    db_channel = Channel(
        owner_id=current_user.id,
        **mapped_dict,
        api_key=secrets.token_urlsafe(32),  # Generate unique API key
        is_verified=False,
        is_active=True,
    )

    # If YouTube credentials provided, verify them
    if hasattr(channel_data, "youtube_channel_id") and hasattr(
        channel_data, "youtube_api_key"
    ):
        if channel_data.youtube_channel_id and channel_data.youtube_api_key:
            try:
                # Verify YouTube channel
                channel_info = await youtube_service.verify_channel(
                    channel_data.youtube_channel_id, channel_data.youtube_api_key
                )

                if channel_info:
                    db_channel.is_verified = True
                    db_channel.youtube_channel_url = (
                        f"https://youtube.com/channel/{channel_data.youtube_channel_id}"
                    )

                    # Schedule initial analytics fetch
                    background_tasks.add_task(
                        analytics_service.fetch_channel_analytics,
                        str(db_channel.id),
                        channel_data.youtube_channel_id,
                        channel_data.youtube_api_key,
                    )
            except Exception as e:
                logger.warning(f"YouTube verification failed: {str(e)}")

    try:
        db.add(db_channel)
        await db.commit()
        await db.refresh(db_channel)
        return db_channel

    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Channel creation failed. Please try again.",
        )


@router.get("/", response_model=List[ChannelResponse])
async def get_channels(
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: AsyncSession = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    is_active: Optional[bool] = Query(None),
    category: Optional[str] = Query(None),
):
    """
    Get all channels for the current user with filtering

    - Returns channels owned by current user
    - Supports filtering by status and category
    - Includes basic statistics for each channel
    """
    query = select(Channel).filter(Channel.owner_id == current_user.id)

    if is_active is not None:
        query = query.filter(Channel.is_active == is_active)

    if category:
        query = query.filter(Channel.category == category)

    query = query.offset(skip).limit(limit).order_by(Channel.created_at.desc())

    result = await db.execute(query)
    channels = result.scalars().all()

    # Enhance with basic stats if needed
    for channel in channels:
        # Get video count
        stats_result = await db.execute(
            select(func.count(Video.id)).filter(Video.channel_id == channel.id)
        )
        channel.total_videos = stats_result.scalar() or 0

    return channels


@router.get("/{channel_id}", response_model=ChannelResponse)
async def get_channel(
    channel_id: str,
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: AsyncSession = Depends(get_db),
):
    """
    Get detailed channel information

    - Returns channel details with aggregated statistics
    - Validates ownership
    - Includes video count and revenue metrics
    """
    result = await db.execute(
        select(Channel).filter(
            and_(Channel.id == channel_id, Channel.owner_id == current_user.id)
        )
    )
    channel = result.scalar_one_or_none()

    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Channel not found"
        )

    # Get statistics
    stats_result = await db.execute(
        select(
            func.count(Video.id).label("total_videos"),
            func.sum(Video.view_count).label("total_views"),
            func.sum(Video.actual_revenue).label("total_revenue"),
        ).filter(Video.channel_id == channel.id)
    )
    stats = stats_result.first()

    # Add stats to channel object
    channel.total_videos = stats.total_videos or 0
    channel.total_views = stats.total_views or 0
    channel.total_revenue = float(stats.total_revenue or 0)

    return channel


@router.put("/{channel_id}", response_model=ChannelResponse)
async def update_channel(
    channel_id: str,
    channel_update: ChannelUpdate,
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: AsyncSession = Depends(get_db),
):
    """
    Update channel configuration

    - Updates channel settings
    - Validates ownership
    - Preserves YouTube connection
    - Updates modification timestamp
    """
    result = await db.execute(
        select(Channel).filter(
            and_(Channel.id == channel_id, Channel.owner_id == current_user.id)
        )
    )
    channel = result.scalar_one_or_none()

    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Channel not found"
        )

    # Update fields
    update_data = channel_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        if value is not None and hasattr(channel, field):
            setattr(channel, field, value)

    channel.updated_at = datetime.utcnow()

    try:
        await db.commit()
        await db.refresh(channel)
        return channel

    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Update failed. Channel name may already exist.",
        )


@router.delete("/{channel_id}", response_model=Dict[str, str])
async def delete_channel(
    channel_id: str,
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: AsyncSession = Depends(get_db),
):
    """
    Delete channel (soft delete)

    - Deactivates channel
    - Preserves data for analytics
    - Cancels all scheduled videos
    - Updates deletion timestamp
    """
    result = await db.execute(
        select(Channel).filter(
            and_(Channel.id == channel_id, Channel.owner_id == current_user.id)
        )
    )
    channel = result.scalar_one_or_none()

    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Channel not found"
        )

    # Soft delete
    channel.is_active = False
    channel.deleted_at = datetime.utcnow()
    channel.updated_at = datetime.utcnow()

    # Cancel any scheduled videos
    await db.execute(
        update(Video)
        .where(Video.channel_id == channel_id, Video.publish_status == "scheduled")
        .values(publish_status="cancelled")
    )

    await db.commit()

    return {"message": "Channel deactivated successfully"}


@router.post("/{channel_id}/connect-youtube", response_model=Dict[str, str])
async def connect_youtube(
    channel_id: str,
    youtube_data: YouTubeConnect,
    background_tasks: BackgroundTasks,
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: AsyncSession = Depends(get_db),
):
    """
    Connect YouTube account to channel

    - Validates YouTube credentials
    - Links YouTube channel
    - Fetches initial analytics
    - Sets verification status
    """
    result = await db.execute(
        select(Channel).filter(
            and_(Channel.id == channel_id, Channel.owner_id == current_user.id)
        )
    )
    channel = result.scalar_one_or_none()

    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Channel not found"
        )

    try:
        # Verify YouTube channel
        channel_info = await youtube_service.verify_channel(
            youtube_data.youtube_channel_id, youtube_data.youtube_api_key
        )

        if not channel_info:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unable to verify YouTube channel. Please check your credentials.",
            )

        # Update channel
        channel.youtube_channel_id = youtube_data.youtube_channel_id
        channel.youtube_channel_url = (
            f"https://youtube.com/channel/{youtube_data.youtube_channel_id}"
        )
        channel.youtube_api_key = youtube_data.youtube_api_key  # Encrypt in production
        channel.youtube_refresh_token = youtube_data.youtube_refresh_token
        channel.is_verified = True
        channel.last_sync_at = datetime.utcnow()
        channel.updated_at = datetime.utcnow()

        await db.commit()

        # Fetch initial analytics in background
        background_tasks.add_task(
            analytics_service.fetch_channel_analytics,
            str(channel.id),
            youtube_data.youtube_channel_id,
            youtube_data.youtube_api_key,
        )

        return {"message": "YouTube channel connected successfully"}

    except Exception as e:
        logger.error(f"YouTube connection failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to connect YouTube channel. Please verify your credentials.",
        )


@router.get("/{channel_id}/stats", response_model=ChannelStats)
async def get_channel_stats(
    channel_id: str,
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: AsyncSession = Depends(get_db),
):
    """
    Get detailed channel statistics

    - Returns comprehensive analytics
    - Includes performance metrics
    - Calculates growth trends
    - Shows best performing content
    """
    # Get channel
    result = await db.execute(
        select(Channel).filter(
            and_(Channel.id == channel_id, Channel.owner_id == current_user.id)
        )
    )
    channel = result.scalar_one_or_none()

    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Channel not found"
        )

    # Get overall statistics
    stats_result = await db.execute(
        select(
            func.count(Video.id).label("total_videos"),
            func.count(Video.id)
            .filter(Video.publish_status == "published")
            .label("published_videos"),
            func.sum(Video.view_count).label("total_views"),
            func.sum(Video.like_count).label("total_likes"),
            func.sum(Video.comment_count).label("total_comments"),
            func.sum(Video.actual_revenue).label("total_revenue"),
            func.sum(Video.total_cost).label("total_cost"),
            func.avg(Video.view_count).label("avg_views"),
            func.max(Video.published_at).label("last_video_date"),
        ).filter(Video.channel_id == channel.id)
    )
    stats = stats_result.first()

    # Get best performing video
    best_video_result = await db.execute(
        select(Video)
        .filter(Video.channel_id == channel.id)
        .order_by(Video.view_count.desc())
        .limit(1)
    )
    best_video = best_video_result.scalar_one_or_none()

    # Calculate metrics
    total_videos = stats.total_videos or 0
    total_views = stats.total_views or 0
    total_revenue = float(stats.total_revenue or 0)
    total_cost = float(stats.total_cost or 0)

    profit_margin = (
        ((total_revenue - total_cost) / total_revenue * 100) if total_revenue > 0 else 0
    )
    avg_engagement_rate = 0

    if total_views > 0:
        total_engagements = (stats.total_likes or 0) + (stats.total_comments or 0)
        avg_engagement_rate = (total_engagements / total_views) * 100

    # Calculate growth rate (last 30 days vs previous 30 days)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    sixty_days_ago = datetime.utcnow() - timedelta(days=60)

    recent_views_result = await db.execute(
        select(func.sum(Video.view_count)).filter(
            Video.channel_id == channel.id, Video.published_at >= thirty_days_ago
        )
    )
    recent_views = recent_views_result.scalar() or 0

    previous_views_result = await db.execute(
        select(func.sum(Video.view_count)).filter(
            Video.channel_id == channel.id,
            Video.published_at >= sixty_days_ago,
            Video.published_at < thirty_days_ago,
        )
    )
    previous_views = previous_views_result.scalar() or 0

    growth_rate = (
        ((recent_views - previous_views) / previous_views * 100)
        if previous_views > 0
        else 0
    )

    return ChannelStats(
        channel_id=str(channel.id),
        channel_name=channel.name,
        total_videos=total_videos,
        published_videos=stats.published_videos or 0,
        total_views=total_views,
        total_likes=stats.total_likes or 0,
        total_comments=stats.total_comments or 0,
        total_revenue=total_revenue,
        total_cost=total_cost,
        profit_margin=round(profit_margin, 2),
        avg_views_per_video=round(float(stats.avg_views or 0), 2),
        avg_engagement_rate=round(avg_engagement_rate, 2),
        best_performing_video={
            "id": str(best_video.id),
            "title": best_video.title,
            "views": best_video.view_count,
            "revenue": best_video.actual_revenue,
        }
        if best_video
        else None,
        growth_rate=round(growth_rate, 2),
        last_video_date=stats.last_video_date,
        # Add quota tracking fields
        daily_quota_used=0,  # Will be updated from Redis
        daily_quota_limit=10,
        weekly_videos_generated=0,
        channel_health_score=85.0,
        isolation_namespace=await ChannelIsolationManager.get_channel_namespace(
            str(channel.id)
        ),
    )


# Enhanced Multi-Channel Endpoints


@router.post("/multi/operation", response_class=ORJSONResponse)
async def multi_channel_operation(
    request: MultiChannelRequest,
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Perform operations across multiple channels simultaneously
    Supports up to 5+ channels per user with isolation
    """
    # Verify ownership of all channels
    channels_query = await db.execute(
        select(Channel).filter(
            Channel.id.in_(request.channel_ids), Channel.owner_id == current_user.id
        )
    )
    channels = channels_query.scalars().all()

    if len(channels) != len(request.channel_ids):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to all specified channels",
        )

    results = []

    # Process each channel in isolation
    for channel in channels:
        # Check channel quota
        if not await ChannelIsolationManager.check_channel_quota(str(channel.id)):
            results.append(
                {
                    "channel_id": str(channel.id),
                    "status": "quota_exceeded",
                    "message": "Daily quota exceeded for this channel",
                }
            )
            continue

        # Perform operation based on type
        if request.operation == "publish":
            # Queue batch publish task
            background_tasks.add_task(
                process_channel_operation,
                str(channel.id),
                "publish",
                request.parameters,
            )
            results.append(
                {
                    "channel_id": str(channel.id),
                    "status": "queued",
                    "message": "Publishing operation queued",
                }
            )

        elif request.operation == "analyze":
            # Queue analytics task
            background_tasks.add_task(
                analytics_service.fetch_channel_analytics,
                str(channel.id),
                channel.youtube_channel_id,
                channel.youtube_api_key,
            )
            results.append(
                {
                    "channel_id": str(channel.id),
                    "status": "analyzing",
                    "message": "Analytics refresh initiated",
                }
            )

    return {
        "operation": request.operation,
        "channels_processed": len(results),
        "results": results,
    }


@router.get("/{channel_id}/quota", response_model=ChannelQuota)
@cached(prefix="channel:quota", ttl=60)
async def get_channel_quota(
    channel_id: str,
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get channel quota information and usage
    """
    # Verify ownership
    channel = await db.execute(
        select(Channel).filter(
            Channel.id == channel_id, Channel.owner_id == current_user.id
        )
    )
    channel = channel.scalar_one_or_none()

    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Channel not found"
        )

    redis = await get_redis()

    # Get current usage
    daily_key = f"quota:{channel_id}:daily:{datetime.utcnow().date()}"
    daily_used = int(await redis.get(daily_key) or 0)

    weekly_key = f"quota:{channel_id}:weekly:{datetime.utcnow().isocalendar()[1]}"
    weekly_used = int(await redis.get(weekly_key) or 0)

    monthly_key = f"quota:{channel_id}:monthly:{datetime.utcnow().month}"
    monthly_used = int(await redis.get(monthly_key) or 0)

    # Get limits based on subscription tier
    limits = get_quota_limits_for_tier(current_user.subscription_tier)

    return ChannelQuota(
        channel_id=channel_id,
        daily_video_limit=limits["daily"],
        weekly_video_limit=limits["weekly"],
        monthly_video_limit=limits["monthly"],
        api_quota_units=10000 - (daily_used * 100),  # Approximate API units
        storage_quota_gb=100,
        concurrent_generations=2,
    )


@router.put("/{channel_id}/quota", response_model=ChannelQuota)
async def update_channel_quota(
    channel_id: str,
    quota: ChannelQuota,
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Update channel quota limits (admin or premium users only)
    """
    # Verify ownership and premium status
    channel = await db.execute(
        select(Channel).filter(
            Channel.id == channel_id, Channel.owner_id == current_user.id
        )
    )
    channel = channel.scalar_one_or_none()

    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Channel not found"
        )

    # Check if user has permission to update quotas
    if current_user.subscription_tier not in ["premium", "enterprise"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Quota customization requires premium subscription",
        )

    redis = await get_redis()

    # Update quota limits in Redis
    await redis.set(f"quota:{channel_id}:limit:daily", quota.daily_video_limit)
    await redis.set(f"quota:{channel_id}:limit:weekly", quota.weekly_video_limit)
    await redis.set(f"quota:{channel_id}:limit:monthly", quota.monthly_video_limit)

    # Invalidate cache
    await cache.invalidate_pattern(f"channel:quota:{channel_id}*")

    return quota


@router.post("/{channel_id}/isolate")
async def configure_channel_isolation(
    channel_id: str,
    cpu_cores: int = 2,
    memory_gb: int = 4,
    gpu_allocation: float = 0.25,
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Configure resource isolation for a channel
    """
    # Verify ownership
    channel = await db.execute(
        select(Channel).filter(
            Channel.id == channel_id, Channel.owner_id == current_user.id
        )
    )
    channel = channel.scalar_one_or_none()

    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Channel not found"
        )

    redis = await get_redis()

    # Set resource allocation
    await redis.set(f"resources:{channel_id}:cpu", cpu_cores)
    await redis.set(f"resources:{channel_id}:memory", memory_gb)
    await redis.set(f"resources:{channel_id}:gpu", gpu_allocation)

    # Set priority based on subscription
    priority = (
        "high"
        if current_user.subscription_tier in ["premium", "enterprise"]
        else "normal"
    )
    await redis.set(f"resources:{channel_id}:priority", priority)

    return {
        "channel_id": channel_id,
        "resources": {
            "cpu_cores": cpu_cores,
            "memory_gb": memory_gb,
            "gpu_allocation": gpu_allocation,
            "priority": priority,
        },
        "namespace": await ChannelIsolationManager.get_channel_namespace(channel_id),
    }


@router.get("/compare", response_class=ORJSONResponse)
async def compare_channels(
    channel_ids: List[str] = Query(..., description="List of channel IDs to compare"),
    metric: str = Query(
        "views", description="Metric to compare: views, revenue, engagement"
    ),
    period: int = Query(30, description="Period in days"),
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Compare performance across multiple channels
    """
    # Verify ownership of all channels
    channels = await db.execute(
        select(Channel).filter(
            Channel.id.in_(channel_ids), Channel.owner_id == current_user.id
        )
    )
    channels = channels.scalars().all()

    if len(channels) != len(channel_ids):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to all specified channels",
        )

    comparison_data = []
    start_date = datetime.utcnow() - timedelta(days=period)

    for channel in channels:
        # Get metrics for each channel
        if metric == "views":
            result = await db.execute(
                select(func.sum(Video.view_count)).filter(
                    Video.channel_id == channel.id, Video.published_at >= start_date
                )
            )
            value = result.scalar() or 0

        elif metric == "revenue":
            result = await db.execute(
                select(func.sum(Video.actual_revenue)).filter(
                    Video.channel_id == channel.id, Video.published_at >= start_date
                )
            )
            value = float(result.scalar() or 0)

        elif metric == "engagement":
            result = await db.execute(
                select(
                    func.sum(Video.like_count + Video.comment_count),
                    func.sum(Video.view_count),
                ).filter(
                    Video.channel_id == channel.id, Video.published_at >= start_date
                )
            )
            engagements, views = result.first()
            value = ((engagements or 0) / (views or 1)) * 100

        comparison_data.append(
            {
                "channel_id": str(channel.id),
                "channel_name": channel.name,
                "metric": metric,
                "value": value,
                "period_days": period,
            }
        )

    # Sort by value
    comparison_data.sort(key=lambda x: x["value"], reverse=True)

    return {
        "comparison": comparison_data,
        "best_performer": comparison_data[0] if comparison_data else None,
        "metric": metric,
        "period_days": period,
    }


# Helper functions


def get_quota_limits_for_tier(tier: str) -> Dict[str, int]:
    """Get quota limits based on subscription tier"""
    limits = {
        "free": {"daily": 2, "weekly": 10, "monthly": 30},
        "basic": {"daily": 5, "weekly": 25, "monthly": 100},
        "pro": {"daily": 10, "weekly": 50, "monthly": 200},
        "premium": {"daily": 20, "weekly": 100, "monthly": 400},
        "enterprise": {"daily": 50, "weekly": 250, "monthly": 1000},
    }
    return limits.get(tier, limits["free"])


async def process_channel_operation(
    channel_id: str, operation: str, parameters: Optional[Dict[str, Any]] = None
):
    """Process channel operation in background"""
    # This would be implemented as a Celery task
    logger.info(f"Processing {operation} for channel {channel_id}")
    # Implementation would go here
