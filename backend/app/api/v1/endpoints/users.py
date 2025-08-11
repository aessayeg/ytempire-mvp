"""
User profile and settings management endpoints
"""
from typing import Optional, Dict, Any, Annotated
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from pydantic import BaseModel, EmailStr, Field, validator
import aiofiles
import os
import uuid
import logging

from app.db.session import get_db
from app.models.user import User
from app.api.v1.endpoints.auth import get_current_user, get_current_verified_user
from app.services.storage_service import StorageService
from app.core.config import settings

router = APIRouter(prefix="/users", tags=["users"])
logger = logging.getLogger(__name__)

# Initialize services
storage_service = StorageService()

# Pydantic models
class UserProfileUpdate(BaseModel):
    """User profile update request"""
    full_name: Optional[str] = Field(None, min_length=1, max_length=100)
    company_name: Optional[str] = Field(None, max_length=100)
    phone: Optional[str] = Field(None, pattern=r'^\+?1?\d{9,15}$')
    bio: Optional[str] = Field(None, max_length=500)
    website: Optional[str] = Field(None, pattern=r'^https?://')
    social_links: Optional[Dict[str, str]] = None

class UserPreferences(BaseModel):
    """User preferences"""
    email_notifications: bool = True
    push_notifications: bool = False
    weekly_report: bool = True
    auto_publish: bool = False
    default_quality: str = "balanced"
    default_language: str = "en"
    timezone: str = "UTC"
    
    @validator('default_quality')
    def validate_quality(cls, v):
        valid_qualities = ["fast", "balanced", "quality"]
        if v not in valid_qualities:
            raise ValueError(f"Quality must be one of: {', '.join(valid_qualities)}")
        return v

class SubscriptionUpdate(BaseModel):
    """Subscription tier update"""
    tier: str = Field(..., description="Subscription tier: free, starter, pro, enterprise")
    payment_method_id: Optional[str] = None
    
    @validator('tier')
    def validate_tier(cls, v):
        valid_tiers = ["free", "starter", "pro", "enterprise"]
        if v not in valid_tiers:
            raise ValueError(f"Tier must be one of: {', '.join(valid_tiers)}")
        return v

class UserStats(BaseModel):
    """User statistics response"""
    user_id: str
    total_channels: int
    active_channels: int
    total_videos: int
    published_videos: int
    total_views: int
    total_revenue: float
    total_cost: float
    profit: float
    avg_video_cost: float
    avg_video_revenue: float
    subscription_tier: str
    api_quota_remaining: int
    storage_used_gb: float
    account_created: datetime

# Endpoints
@router.get("/profile", response_model=Dict[str, Any])
async def get_profile(
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get current user profile
    
    - Returns complete user profile
    - Includes subscription details
    - Shows usage statistics
    """
    # Get user statistics
    from app.models.channel import Channel
    from app.models.video import Video
    from sqlalchemy import func
    
    # Count channels
    channels_result = await db.execute(
        select(
            func.count(Channel.id).label("total_channels"),
            func.count(Channel.id).filter(Channel.is_active == True).label("active_channels")
        ).filter(Channel.owner_id == current_user.id)
    )
    channels_stats = channels_result.first()
    
    # Count videos and stats
    videos_result = await db.execute(
        select(
            func.count(Video.id).label("total_videos"),
            func.count(Video.id).filter(Video.publish_status == "published").label("published_videos"),
            func.sum(Video.view_count).label("total_views"),
            func.sum(Video.actual_revenue).label("total_revenue"),
            func.sum(Video.total_cost).label("total_cost")
        ).join(Channel).filter(Channel.owner_id == current_user.id)
    )
    video_stats = videos_result.first()
    
    return {
        "profile": {
            "id": str(current_user.id),
            "email": current_user.email,
            "username": current_user.username,
            "full_name": current_user.full_name,
            "company_name": current_user.company_name,
            "phone": current_user.phone,
            "is_verified": current_user.is_verified,
            "created_at": current_user.created_at,
            "last_login": current_user.last_login
        },
        "subscription": {
            "tier": current_user.subscription_tier,
            "status": current_user.subscription_status,
            "end_date": current_user.subscription_end_date,
            "channels_limit": current_user.channels_limit,
            "videos_per_day_limit": current_user.videos_per_day_limit,
            "monthly_budget_limit": current_user.monthly_budget_limit,
            "api_quota_remaining": current_user.api_quota_remaining,
            "api_quota_reset_at": current_user.api_quota_reset_at
        },
        "statistics": {
            "total_channels": channels_stats.total_channels or 0,
            "active_channels": channels_stats.active_channels or 0,
            "total_videos": video_stats.total_videos or 0,
            "published_videos": video_stats.published_videos or 0,
            "total_views": video_stats.total_views or 0,
            "total_revenue": float(video_stats.total_revenue or 0),
            "total_cost": float(video_stats.total_cost or 0),
            "profit": float((video_stats.total_revenue or 0) - (video_stats.total_cost or 0))
        }
    }

@router.put("/profile", response_model=Dict[str, str])
async def update_profile(
    profile_update: UserProfileUpdate,
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update user profile
    
    - Update personal information
    - Change company details
    - Update contact information
    """
    # Update user fields
    update_data = profile_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        if value is not None and hasattr(current_user, field):
            setattr(current_user, field, value)
    
    current_user.updated_at = datetime.utcnow()
    
    await db.commit()
    
    return {"message": "Profile updated successfully"}

@router.post("/avatar", response_model=Dict[str, str])
async def upload_avatar(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload user avatar
    
    - Accepts image files (jpg, png, webp)
    - Max size 5MB
    - Automatically resizes to optimal dimensions
    """
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Allowed: jpg, png, webp"
        )
    
    # Check file size (5MB max)
    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File too large. Max size: 5MB"
        )
    
    # Generate unique filename
    file_extension = file.filename.split(".")[-1]
    filename = f"avatars/{current_user.id}/{uuid.uuid4()}.{file_extension}"
    
    # Upload to storage
    avatar_url = await storage_service.upload_file(contents, filename, file.content_type)
    
    # Update user avatar URL
    current_user.avatar_url = avatar_url
    current_user.updated_at = datetime.utcnow()
    
    await db.commit()
    
    return {"avatar_url": avatar_url}

@router.get("/preferences", response_model=UserPreferences)
async def get_preferences(
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get user preferences
    
    - Notification settings
    - Default generation settings
    - Timezone and language preferences
    """
    # Get preferences from user model or return defaults
    preferences = UserPreferences(
        email_notifications=getattr(current_user, 'email_notifications', True),
        push_notifications=getattr(current_user, 'push_notifications', False),
        weekly_report=getattr(current_user, 'weekly_report', True),
        auto_publish=getattr(current_user, 'auto_publish', False),
        default_quality=getattr(current_user, 'default_quality', 'balanced'),
        default_language=getattr(current_user, 'default_language', 'en'),
        timezone=getattr(current_user, 'timezone', 'UTC')
    )
    
    return preferences

@router.put("/preferences", response_model=Dict[str, str])
async def update_preferences(
    preferences: UserPreferences,
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update user preferences
    
    - Configure notification settings
    - Set default generation options
    - Update timezone and language
    """
    # Update preferences
    for field, value in preferences.dict().items():
        if hasattr(current_user, field):
            setattr(current_user, field, value)
    
    current_user.updated_at = datetime.utcnow()
    
    await db.commit()
    
    return {"message": "Preferences updated successfully"}

@router.post("/subscription/upgrade", response_model=Dict[str, Any])
async def upgrade_subscription(
    subscription: SubscriptionUpdate,
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Upgrade subscription tier
    
    - Change subscription plan
    - Update limits and quotas
    - Process payment if required
    """
    # Define tier limits
    tier_limits = {
        "free": {
            "channels_limit": 1,
            "videos_per_day_limit": 5,
            "monthly_budget_limit": 100.0,
            "api_quota": 100
        },
        "starter": {
            "channels_limit": 3,
            "videos_per_day_limit": 20,
            "monthly_budget_limit": 500.0,
            "api_quota": 1000
        },
        "pro": {
            "channels_limit": 10,
            "videos_per_day_limit": 50,
            "monthly_budget_limit": 2000.0,
            "api_quota": 5000
        },
        "enterprise": {
            "channels_limit": 100,
            "videos_per_day_limit": 200,
            "monthly_budget_limit": 10000.0,
            "api_quota": 50000
        }
    }
    
    # Check if downgrade
    tier_order = ["free", "starter", "pro", "enterprise"]
    current_tier_index = tier_order.index(current_user.subscription_tier)
    new_tier_index = tier_order.index(subscription.tier)
    
    if new_tier_index < current_tier_index:
        # Downgrade - check if user exceeds new limits
        from app.models.channel import Channel
        
        active_channels = await db.execute(
            select(func.count(Channel.id)).filter(
                Channel.owner_id == current_user.id,
                Channel.is_active == True
            )
        )
        channel_count = active_channels.scalar()
        
        if channel_count > tier_limits[subscription.tier]["channels_limit"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"You have {channel_count} active channels. Please deactivate some before downgrading."
            )
    
    # Process payment for paid tiers
    if subscription.tier != "free" and subscription.payment_method_id:
        # TODO: Integrate with payment processor (Stripe)
        pass
    
    # Update user subscription
    limits = tier_limits[subscription.tier]
    current_user.subscription_tier = subscription.tier
    current_user.subscription_status = "active"
    current_user.channels_limit = limits["channels_limit"]
    current_user.videos_per_day_limit = limits["videos_per_day_limit"]
    current_user.monthly_budget_limit = limits["monthly_budget_limit"]
    current_user.api_quota_remaining = limits["api_quota"]
    current_user.updated_at = datetime.utcnow()
    
    if subscription.tier != "free":
        from datetime import timedelta
        current_user.subscription_end_date = datetime.utcnow() + timedelta(days=30)
    
    await db.commit()
    
    return {
        "message": f"Subscription upgraded to {subscription.tier}",
        "new_limits": limits,
        "subscription_end_date": current_user.subscription_end_date
    }

@router.get("/stats", response_model=UserStats)
async def get_user_stats(
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get comprehensive user statistics
    
    - Channel and video counts
    - Financial summary
    - Usage metrics
    - Account information
    """
    from app.models.channel import Channel
    from app.models.video import Video
    from sqlalchemy import func
    
    # Get channel stats
    channels_result = await db.execute(
        select(
            func.count(Channel.id).label("total_channels"),
            func.count(Channel.id).filter(Channel.is_active == True).label("active_channels")
        ).filter(Channel.owner_id == current_user.id)
    )
    channels_stats = channels_result.first()
    
    # Get video stats
    videos_result = await db.execute(
        select(
            func.count(Video.id).label("total_videos"),
            func.count(Video.id).filter(Video.publish_status == "published").label("published_videos"),
            func.sum(Video.view_count).label("total_views"),
            func.sum(Video.actual_revenue).label("total_revenue"),
            func.sum(Video.total_cost).label("total_cost"),
            func.avg(Video.total_cost).label("avg_cost"),
            func.avg(Video.actual_revenue).label("avg_revenue")
        ).join(Channel).filter(Channel.owner_id == current_user.id)
    )
    video_stats = videos_result.first()
    
    # Calculate storage (estimate based on video count)
    storage_gb = (video_stats.total_videos or 0) * 0.1  # Estimate 100MB per video
    
    return UserStats(
        user_id=str(current_user.id),
        total_channels=channels_stats.total_channels or 0,
        active_channels=channels_stats.active_channels or 0,
        total_videos=video_stats.total_videos or 0,
        published_videos=video_stats.published_videos or 0,
        total_views=video_stats.total_views or 0,
        total_revenue=float(video_stats.total_revenue or 0),
        total_cost=float(video_stats.total_cost or 0),
        profit=float((video_stats.total_revenue or 0) - (video_stats.total_cost or 0)),
        avg_video_cost=float(video_stats.avg_cost or 0),
        avg_video_revenue=float(video_stats.avg_revenue or 0),
        subscription_tier=current_user.subscription_tier,
        api_quota_remaining=current_user.api_quota_remaining,
        storage_used_gb=round(storage_gb, 2),
        account_created=current_user.created_at
    )

@router.delete("/account", response_model=Dict[str, str])
async def delete_account(
    password: str,
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete user account
    
    - Requires password confirmation
    - Soft delete for 30-day recovery period
    - Cancels all subscriptions
    """
    from app.core.security import verify_password
    
    # Verify password
    if not verify_password(password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password"
        )
    
    # Soft delete user
    current_user.is_active = False
    current_user.deleted_at = datetime.utcnow()
    current_user.subscription_status = "cancelled"
    
    # Deactivate all channels
    from app.models.channel import Channel
    await db.execute(
        update(Channel)
        .where(Channel.owner_id == current_user.id)
        .values(is_active=False, deleted_at=datetime.utcnow())
    )
    
    await db.commit()
    
    return {
        "message": "Account scheduled for deletion. You have 30 days to reactivate."
    }