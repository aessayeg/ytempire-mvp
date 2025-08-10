"""
Channel Repository
Owner: Backend Team Lead
"""

from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from sqlalchemy.orm import selectinload
import uuid
from datetime import datetime

from app.models.channel import Channel
from app.models.video import Video
from app.schemas.channel import ChannelCreate, ChannelUpdate


class ChannelRepository:
    """Repository for channel data operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_by_id(self, channel_id: str) -> Optional[Channel]:
        """Get channel by ID."""
        result = await self.db.execute(
            select(Channel).where(Channel.id == channel_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_user_id(self, user_id: str) -> List[Channel]:
        """Get all channels for a user."""
        result = await self.db.execute(
            select(Channel)
            .where(and_(Channel.user_id == user_id, Channel.is_active == True))
            .order_by(Channel.created_at.desc())
        )
        return result.scalars().all()
    
    async def get_user_channel_count(self, user_id: str) -> int:
        """Get count of active channels for a user."""
        result = await self.db.execute(
            select(func.count(Channel.id))
            .where(and_(Channel.user_id == user_id, Channel.is_active == True))
        )
        return result.scalar_one()
    
    async def create_channel(self, user_id: str, channel_data: ChannelCreate) -> Channel:
        """Create a new channel."""
        channel = Channel(
            id=str(uuid.uuid4()),
            user_id=user_id,
            name=channel_data.name,
            description=channel_data.description or "",
            category=channel_data.category,
            target_audience=channel_data.target_audience,
            content_style=channel_data.content_style,
            upload_schedule=channel_data.upload_schedule,
            branding={
                "primary_color": channel_data.primary_color,
                "secondary_color": channel_data.secondary_color,
                "logo_url": channel_data.logo_url
            } if any([channel_data.primary_color, channel_data.secondary_color, channel_data.logo_url]) else {},
            automation_settings={
                "auto_publish": channel_data.auto_publish or False,
                "seo_optimization": channel_data.seo_optimization or True,
                "thumbnail_generation": channel_data.thumbnail_generation or True,
                "content_scheduling": channel_data.content_scheduling or False
            },
            is_active=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        self.db.add(channel)
        await self.db.commit()
        await self.db.refresh(channel)
        return channel
    
    async def update_channel(self, channel_id: str, update_data: ChannelUpdate) -> Optional[Channel]:
        """Update channel information."""
        result = await self.db.execute(
            select(Channel).where(Channel.id == channel_id)
        )
        channel = result.scalar_one_or_none()
        
        if not channel:
            return None
        
        # Update fields
        if update_data.name is not None:
            channel.name = update_data.name
        if update_data.description is not None:
            channel.description = update_data.description
        if update_data.category is not None:
            channel.category = update_data.category
        if update_data.target_audience is not None:
            channel.target_audience = update_data.target_audience
        if update_data.content_style is not None:
            channel.content_style = update_data.content_style
        if update_data.upload_schedule is not None:
            channel.upload_schedule = update_data.upload_schedule
        
        # Update branding if provided
        if any([update_data.primary_color, update_data.secondary_color, update_data.logo_url]):
            if not channel.branding:
                channel.branding = {}
            if update_data.primary_color is not None:
                channel.branding["primary_color"] = update_data.primary_color
            if update_data.secondary_color is not None:
                channel.branding["secondary_color"] = update_data.secondary_color
            if update_data.logo_url is not None:
                channel.branding["logo_url"] = update_data.logo_url
        
        # Update automation settings
        if any([
            update_data.auto_publish is not None,
            update_data.seo_optimization is not None,
            update_data.thumbnail_generation is not None,
            update_data.content_scheduling is not None
        ]):
            if not channel.automation_settings:
                channel.automation_settings = {}
            if update_data.auto_publish is not None:
                channel.automation_settings["auto_publish"] = update_data.auto_publish
            if update_data.seo_optimization is not None:
                channel.automation_settings["seo_optimization"] = update_data.seo_optimization
            if update_data.thumbnail_generation is not None:
                channel.automation_settings["thumbnail_generation"] = update_data.thumbnail_generation
            if update_data.content_scheduling is not None:
                channel.automation_settings["content_scheduling"] = update_data.content_scheduling
        
        channel.updated_at = datetime.utcnow()
        
        await self.db.commit()
        await self.db.refresh(channel)
        return channel
    
    async def soft_delete_channel(self, channel_id: str) -> bool:
        """Soft delete a channel."""
        result = await self.db.execute(
            select(Channel).where(Channel.id == channel_id)
        )
        channel = result.scalar_one_or_none()
        
        if channel:
            channel.is_active = False
            channel.updated_at = datetime.utcnow()
            await self.db.commit()
            return True
        return False
    
    async def get_channel_with_stats(self, channel_id: str) -> Optional[dict]:
        """Get channel with aggregated statistics."""
        # Get channel
        result = await self.db.execute(
            select(Channel).where(Channel.id == channel_id)
        )
        channel = result.scalar_one_or_none()
        
        if not channel:
            return None
        
        # Get video statistics
        video_stats = await self.db.execute(
            select(
                func.count(Video.id).label('total_videos'),
                func.coalesce(func.sum(Video.total_cost), 0).label('total_cost')
            )
            .where(and_(Video.channel_id == channel_id, Video.status != 'FAILED'))
        )
        stats = video_stats.first()
        
        return {
            'channel': channel,
            'total_videos': stats.total_videos or 0,
            'total_cost': float(stats.total_cost or 0),
            'average_cost_per_video': float(stats.total_cost or 0) / max(stats.total_videos or 1, 1)
        }
    
    async def update_youtube_connection(self, channel_id: str, youtube_channel_id: str) -> bool:
        """Update YouTube channel connection."""
        result = await self.db.execute(
            select(Channel).where(Channel.id == channel_id)
        )
        channel = result.scalar_one_or_none()
        
        if channel:
            channel.youtube_channel_id = youtube_channel_id
            channel.updated_at = datetime.utcnow()
            await self.db.commit()
            return True
        return False
    
    async def check_ownership(self, channel_id: str, user_id: str) -> bool:
        """Check if user owns the channel."""
        result = await self.db.execute(
            select(Channel.id)
            .where(and_(
                Channel.id == channel_id,
                Channel.user_id == user_id,
                Channel.is_active == True
            ))
        )
        return result.scalar_one_or_none() is not None