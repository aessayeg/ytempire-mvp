"""
User Management Endpoints
Owner: API Developer
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.api.deps import get_current_user, get_current_superuser
from app.schemas.user import UserResponse, UserUpdate
from app.models.user import User

router = APIRouter()


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """Get current user information"""
    return current_user


@router.patch("/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update current user information"""
    # Update user in database
    return current_user


@router.get("/me/usage")
async def get_usage_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user usage statistics and costs"""
    return {
        "user_id": current_user.id,
        "total_spent": current_user.total_spent,
        "monthly_budget": current_user.monthly_budget,
        "budget_remaining": current_user.monthly_budget - current_user.total_spent,
        "channels_used": 2,
        "channels_limit": current_user.channels_limit,
        "videos_today": 3,
        "daily_video_limit": current_user.daily_video_limit,
        "total_videos": 25,
        "subscription_tier": current_user.subscription_tier
    }


@router.get("/", response_model=list[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_superuser),
    db: AsyncSession = Depends(get_db)
):
    """List all users (admin only)"""
    # Query all users
    return []