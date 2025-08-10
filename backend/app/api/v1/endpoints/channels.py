"""
Channel Management Endpoints
Owner: Backend Team Lead
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.api.v1.endpoints.auth import get_current_user
from app.schemas.channel import ChannelCreate, ChannelUpdate, ChannelResponse, ChannelStats, ChannelWithStats
from app.models.user import User
from app.repositories.channel_repository import ChannelRepository
from app.core.config import settings

router = APIRouter()


def get_channel_repo(db: AsyncSession = Depends(get_db)) -> ChannelRepository:
    """Get channel repository instance."""
    return ChannelRepository(db)


@router.post("/", response_model=ChannelResponse)
async def create_channel(
    channel_data: ChannelCreate,
    current_user: User = Depends(get_current_user),
    channel_repo: ChannelRepository = Depends(get_channel_repo)
):
    """
    Create a new channel for the current user
    - Validate channel limit (5 per user)
    - Initialize channel settings
    """
    # Check channel limit
    channel_count = await channel_repo.get_user_channel_count(current_user.id)
    if channel_count >= settings.MAX_CHANNELS_PER_USER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Channel limit reached. Maximum {settings.MAX_CHANNELS_PER_USER} channels per user."
        )
    
    # Create channel
    channel = await channel_repo.create_channel(current_user.id, channel_data)
    
    return ChannelResponse.from_orm(channel)


@router.get("/", response_model=List[ChannelResponse])
async def list_channels(
    current_user: User = Depends(get_current_user),
    channel_repo: ChannelRepository = Depends(get_channel_repo)
):
    """
    List all channels for the current user
    """
    channels = await channel_repo.get_by_user_id(current_user.id)
    return [ChannelResponse.from_orm(channel) for channel in channels]


@router.get("/{channel_id}", response_model=ChannelResponse)
async def get_channel(
    channel_id: str,
    current_user: User = Depends(get_current_user),
    channel_repo: ChannelRepository = Depends(get_channel_repo)
):
    """
    Get specific channel details
    """
    # Verify ownership
    if not await channel_repo.check_ownership(channel_id, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Channel not found or access denied"
        )
    
    channel = await channel_repo.get_by_id(channel_id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Channel not found"
        )
    
    return ChannelResponse.from_orm(channel)


@router.patch("/{channel_id}", response_model=ChannelResponse)
async def update_channel(
    channel_id: str,
    channel_update: ChannelUpdate,
    current_user: User = Depends(get_current_user),
    channel_repo: ChannelRepository = Depends(get_channel_repo)
):
    """
    Update channel settings
    """
    # Verify ownership
    if not await channel_repo.check_ownership(channel_id, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Channel not found or access denied"
        )
    
    # Update channel
    updated_channel = await channel_repo.update_channel(channel_id, channel_update)
    if not updated_channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Channel not found"
        )
    
    return ChannelResponse.from_orm(updated_channel)


@router.delete("/{channel_id}")
async def delete_channel(
    channel_id: str,
    current_user: User = Depends(get_current_user),
    channel_repo: ChannelRepository = Depends(get_channel_repo)
):
    """
    Delete a channel (soft delete)
    """
    # Verify ownership
    if not await channel_repo.check_ownership(channel_id, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Channel not found or access denied"
        )
    
    # Soft delete channel
    success = await channel_repo.soft_delete_channel(channel_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Channel not found"
        )
    
    return {"message": "Channel deleted successfully"}


@router.post("/{channel_id}/connect-youtube")
async def connect_youtube(
    channel_id: str,
    youtube_channel_id: str,
    current_user: User = Depends(get_current_user),
    channel_repo: ChannelRepository = Depends(get_channel_repo)
):
    """
    Connect channel to YouTube account
    """
    # Verify ownership
    if not await channel_repo.check_ownership(channel_id, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Channel not found or access denied"
        )
    
    # Update YouTube connection
    success = await channel_repo.update_youtube_connection(channel_id, youtube_channel_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to connect YouTube account"
        )
    
    return {"message": "YouTube account connected successfully"}


@router.get("/{channel_id}/stats", response_model=ChannelStats)
async def get_channel_stats(
    channel_id: str,
    current_user: User = Depends(get_current_user),
    channel_repo: ChannelRepository = Depends(get_channel_repo)
):
    """
    Get channel statistics and analytics
    """
    # Verify ownership
    if not await channel_repo.check_ownership(channel_id, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Channel not found or access denied"
        )
    
    # Get stats
    channel_data = await channel_repo.get_channel_with_stats(channel_id)
    if not channel_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Channel not found"
        )
    
    return ChannelStats(
        channel_id=channel_id,
        total_videos=channel_data['total_videos'],
        total_cost=channel_data['total_cost'],
        average_cost_per_video=channel_data['average_cost_per_video'],
        total_views=0,  # Would need YouTube API integration
        total_subscribers=0,  # Would need YouTube API integration
        average_views_per_video=0.0  # Would need YouTube API integration
    )