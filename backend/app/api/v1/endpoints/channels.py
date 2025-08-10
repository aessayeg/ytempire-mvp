"""
Channel Management CRUD endpoints
"""
from typing import List, Annotated
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from datetime import datetime

from app.db.session import get_db
from app.models.channel import Channel
from app.models.user import User
from app.schemas.channel import ChannelCreate, ChannelUpdate, ChannelResponse
from app.api.v1.endpoints.auth import get_current_user

router = APIRouter()


@router.post("/", response_model=ChannelResponse)
async def create_channel(
    channel_data: ChannelCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new YouTube channel for the current user
    """
    # Check channel limit
    result = await db.execute(
        select(Channel).filter(Channel.user_id == current_user.id)
    )
    existing_channels = len(result.scalars().all())
    
    if existing_channels >= current_user.channels_limit:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Channel limit reached ({current_user.channels_limit})"
        )
    
    # Check if channel already exists
    result = await db.execute(
        select(Channel).filter(Channel.youtube_channel_id == channel_data.youtube_channel_id)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Channel already registered"
        )
    
    # Create new channel
    db_channel = Channel(
        user_id=current_user.id,
        **channel_data.dict()
    )
    
    db.add(db_channel)
    await db.commit()
    await db.refresh(db_channel)
    
    return db_channel


@router.get("/", response_model=List[ChannelResponse])
async def get_channels(
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    is_active: bool = Query(None)
):
    """
    Get all channels for the current user
    """
    query = select(Channel).filter(Channel.user_id == current_user.id)
    
    if is_active is not None:
        query = query.filter(Channel.is_active == is_active)
    
    query = query.offset(skip).limit(limit)
    
    result = await db.execute(query)
    channels = result.scalars().all()
    
    return channels


@router.get("/{channel_id}", response_model=ChannelResponse)
async def get_channel(
    channel_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific channel by ID
    """
    result = await db.execute(
        select(Channel).filter(
            and_(
                Channel.id == channel_id,
                Channel.user_id == current_user.id
            )
        )
    )
    channel = result.scalar_one_or_none()
    
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Channel not found"
        )
    
    return channel


@router.put("/{channel_id}", response_model=ChannelResponse)
async def update_channel(
    channel_id: str,
    channel_update: ChannelUpdate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db)
):
    """
    Update a channel's configuration
    """
    result = await db.execute(
        select(Channel).filter(
            and_(
                Channel.id == channel_id,
                Channel.user_id == current_user.id
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
        setattr(channel, field, value)
    
    channel.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(channel)
    
    return channel


@router.delete("/{channel_id}")
async def delete_channel(
    channel_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a channel (soft delete by setting is_active=False)
    """
    result = await db.execute(
        select(Channel).filter(
            and_(
                Channel.id == channel_id,
                Channel.user_id == current_user.id
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
    channel.updated_at = datetime.utcnow()
    
    await db.commit()
    
    return {"message": "Channel deactivated successfully"}


@router.post("/{channel_id}/sync")
async def sync_channel(
    channel_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db)
):
    """
    Sync channel data with YouTube
    """
    result = await db.execute(
        select(Channel).filter(
            and_(
                Channel.id == channel_id,
                Channel.user_id == current_user.id
            )
        )
    )
    channel = result.scalar_one_or_none()
    
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Channel not found"
        )
    
    # TODO: Implement actual YouTube sync
    # This would trigger a Celery task to fetch latest channel data
    
    channel.last_sync_at = datetime.utcnow()
    await db.commit()
    
    return {"message": "Channel sync initiated", "channel_id": channel_id}