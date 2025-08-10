"""
Channel Management CRUD endpoints with enhanced features
"""
from typing import List, Annotated, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, update
from sqlalchemy.exc import IntegrityError
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, validator
import secrets
import logging

from app.db.session import get_db
from app.models.channel import Channel
from app.models.user import User
from app.models.video import Video
from app.schemas.channel import ChannelCreate, ChannelUpdate, ChannelResponse
from app.api.v1.endpoints.auth import get_current_user, get_current_verified_user
from app.services.youtube_service import YouTubeService
from app.services.analytics_service import AnalyticsService

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
youtube_service = YouTubeService()
analytics_service = AnalyticsService()

# Enhanced Pydantic models
class ChannelStats(BaseModel):
    """Channel statistics"""
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

class YouTubeConnect(BaseModel):
    """YouTube channel connection request"""
    youtube_channel_id: str
    youtube_api_key: str
    youtube_refresh_token: Optional[str] = None


@router.post("/", response_model=ChannelResponse, status_code=status.HTTP_201_CREATED)
async def create_channel(
    channel_data: ChannelCreate,
    background_tasks: BackgroundTasks,
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: AsyncSession = Depends(get_db)
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
            Channel.owner_id == current_user.id,
            Channel.is_active == True
        )
    )
    active_channels = result.scalar()
    
    if active_channels >= current_user.channels_limit:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Channel limit reached. Your {current_user.subscription_tier} plan allows {current_user.channels_limit} channel(s)."
        )
    
    # Check if channel name already exists for this user
    result = await db.execute(
        select(Channel).filter(
            Channel.owner_id == current_user.id,
            Channel.name == channel_data.name
        )
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="You already have a channel with this name"
        )
    
    # Create new channel with enhanced fields
    db_channel = Channel(
        owner_id=current_user.id,
        **channel_data.dict(),
        api_key=secrets.token_urlsafe(32),  # Generate unique API key
        is_verified=False
    )
    
    # If YouTube credentials provided, verify them
    if hasattr(channel_data, 'youtube_channel_id') and hasattr(channel_data, 'youtube_api_key'):
        if channel_data.youtube_channel_id and channel_data.youtube_api_key:
            try:
                # Verify YouTube channel
                channel_info = await youtube_service.verify_channel(
                    channel_data.youtube_channel_id,
                    channel_data.youtube_api_key
                )
                
                if channel_info:
                    db_channel.is_verified = True
                    db_channel.youtube_channel_url = f"https://youtube.com/channel/{channel_data.youtube_channel_id}"
                    
                    # Schedule initial analytics fetch
                    background_tasks.add_task(
                        analytics_service.fetch_channel_analytics,
                        str(db_channel.id),
                        channel_data.youtube_channel_id,
                        channel_data.youtube_api_key
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
            detail="Channel creation failed. Please try again."
        )


@router.get("/", response_model=List[ChannelResponse])
async def get_channels(
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: AsyncSession = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    is_active: Optional[bool] = Query(None),
    category: Optional[str] = Query(None)
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
            select(func.count(Video.id))
            .filter(Video.channel_id == channel.id)
        )
        channel.total_videos = stats_result.scalar() or 0
    
    return channels


@router.get("/{channel_id}", response_model=ChannelResponse)
async def get_channel(
    channel_id: str,
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed channel information
    
    - Returns channel details with aggregated statistics
    - Validates ownership
    - Includes video count and revenue metrics
    """
    result = await db.execute(
        select(Channel).filter(
            and_(
                Channel.id == channel_id,
                Channel.owner_id == current_user.id
            )
        )
    )
    channel = result.scalar_one_or_none()
    
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Channel not found"
        )
    
    # Get statistics
    stats_result = await db.execute(
        select(
            func.count(Video.id).label("total_videos"),
            func.sum(Video.view_count).label("total_views"),
            func.sum(Video.actual_revenue).label("total_revenue")
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
    db: AsyncSession = Depends(get_db)
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
            and_(
                Channel.id == channel_id,
                Channel.owner_id == current_user.id
            )
        )
    )
    channel = result.scalar_one_or_none()
    
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Channel not found"
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
            detail="Update failed. Channel name may already exist."
        )


@router.delete("/{channel_id}", response_model=Dict[str, str])
async def delete_channel(
    channel_id: str,
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: AsyncSession = Depends(get_db)
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
            and_(
                Channel.id == channel_id,
                Channel.owner_id == current_user.id
            )
        )
    )
    channel = result.scalar_one_or_none()
    
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Channel not found"
        )
    
    # Soft delete
    channel.is_active = False
    channel.deleted_at = datetime.utcnow()
    channel.updated_at = datetime.utcnow()
    
    # Cancel any scheduled videos
    await db.execute(
        update(Video)
        .where(
            Video.channel_id == channel_id,
            Video.publish_status == "scheduled"
        )
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
    db: AsyncSession = Depends(get_db)
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
            and_(
                Channel.id == channel_id,
                Channel.owner_id == current_user.id
            )
        )
    )
    channel = result.scalar_one_or_none()
    
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Channel not found"
        )
    
    try:
        # Verify YouTube channel
        channel_info = await youtube_service.verify_channel(
            youtube_data.youtube_channel_id,
            youtube_data.youtube_api_key
        )
        
        if not channel_info:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unable to verify YouTube channel. Please check your credentials."
            )
        
        # Update channel
        channel.youtube_channel_id = youtube_data.youtube_channel_id
        channel.youtube_channel_url = f"https://youtube.com/channel/{youtube_data.youtube_channel_id}"
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
            youtube_data.youtube_api_key
        )
        
        return {"message": "YouTube channel connected successfully"}
        
    except Exception as e:
        logger.error(f"YouTube connection failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to connect YouTube channel. Please verify your credentials."
        )

@router.get("/{channel_id}/stats", response_model=ChannelStats)
async def get_channel_stats(
    channel_id: str,
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: AsyncSession = Depends(get_db)
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
            and_(
                Channel.id == channel_id,
                Channel.owner_id == current_user.id
            )
        )
    )
    channel = result.scalar_one_or_none()
    
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Channel not found"
        )
    
    # Get overall statistics
    stats_result = await db.execute(
        select(
            func.count(Video.id).label("total_videos"),
            func.count(Video.id).filter(Video.publish_status == "published").label("published_videos"),
            func.sum(Video.view_count).label("total_views"),
            func.sum(Video.like_count).label("total_likes"),
            func.sum(Video.comment_count).label("total_comments"),
            func.sum(Video.actual_revenue).label("total_revenue"),
            func.sum(Video.total_cost).label("total_cost"),
            func.avg(Video.view_count).label("avg_views"),
            func.max(Video.published_at).label("last_video_date")
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
    
    profit_margin = ((total_revenue - total_cost) / total_revenue * 100) if total_revenue > 0 else 0
    avg_engagement_rate = 0
    
    if total_views > 0:
        total_engagements = (stats.total_likes or 0) + (stats.total_comments or 0)
        avg_engagement_rate = (total_engagements / total_views) * 100
    
    # Calculate growth rate (last 30 days vs previous 30 days)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    sixty_days_ago = datetime.utcnow() - timedelta(days=60)
    
    recent_views_result = await db.execute(
        select(func.sum(Video.view_count))
        .filter(
            Video.channel_id == channel.id,
            Video.published_at >= thirty_days_ago
        )
    )
    recent_views = recent_views_result.scalar() or 0
    
    previous_views_result = await db.execute(
        select(func.sum(Video.view_count))
        .filter(
            Video.channel_id == channel.id,
            Video.published_at >= sixty_days_ago,
            Video.published_at < thirty_days_ago
        )
    )
    previous_views = previous_views_result.scalar() or 0
    
    growth_rate = ((recent_views - previous_views) / previous_views * 100) if previous_views > 0 else 0
    
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
            "revenue": best_video.actual_revenue
        } if best_video else None,
        growth_rate=round(growth_rate, 2),
        last_video_date=stats.last_video_date
    )