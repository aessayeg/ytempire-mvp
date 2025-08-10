"""
Video Repository
Owner: Backend Team Lead
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, desc, asc
from sqlalchemy.orm import selectinload
import uuid
from datetime import datetime, timedelta

from app.models.video import Video, VideoStatus
from app.models.channel import Channel
from app.schemas.video import VideoCreate


class VideoRepository:
    """Repository for video data operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_by_id(self, video_id: str) -> Optional[Video]:
        """Get video by ID."""
        result = await self.db.execute(
            select(Video).where(Video.id == video_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_channel_id(self, channel_id: str, limit: int = 50, offset: int = 0) -> List[Video]:
        """Get videos by channel ID."""
        result = await self.db.execute(
            select(Video)
            .where(Video.channel_id == channel_id)
            .order_by(desc(Video.created_at))
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()
    
    async def get_by_user_id(self, user_id: str, limit: int = 50, offset: int = 0) -> List[Video]:
        """Get videos by user ID."""
        result = await self.db.execute(
            select(Video)
            .where(Video.user_id == user_id)
            .order_by(desc(Video.created_at))
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()
    
    async def get_by_status(self, status: VideoStatus, limit: int = 50) -> List[Video]:
        """Get videos by status."""
        result = await self.db.execute(
            select(Video)
            .where(Video.status == status)
            .order_by(asc(Video.created_at))
            .limit(limit)
        )
        return result.scalars().all()
    
    async def create_video(self, video_data: VideoCreate, user_id: str) -> Video:
        """Create a new video."""
        video = Video(
            id=str(uuid.uuid4()),
            channel_id=video_data.channel_id,
            user_id=user_id,
            title=video_data.title,
            description=video_data.description,
            status=VideoStatus.PENDING,
            priority=video_data.priority or 1,
            scheduled_publish_at=video_data.scheduled_publish_at,
            content_settings=video_data.content_settings or {},
            generation_settings=video_data.generation_settings or {},
            metadata=video_data.metadata or {},
            tags=video_data.tags or [],
            category=video_data.category,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        self.db.add(video)
        await self.db.commit()
        await self.db.refresh(video)
        return video
    
    async def update_video_status(
        self,
        video_id: str,
        status: VideoStatus,
        current_stage: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> Optional[Video]:
        """Update video status."""
        result = await self.db.execute(
            select(Video).where(Video.id == video_id)
        )
        video = result.scalar_one_or_none()
        
        if not video:
            return None
        
        video.status = status
        if current_stage:
            video.current_stage = current_stage
        if error_message:
            video.error_message = error_message
        
        # Set completion time for completed videos
        if status == VideoStatus.COMPLETED and not video.pipeline_completed_at:
            video.pipeline_completed_at = datetime.utcnow()
        
        video.updated_at = datetime.utcnow()
        
        await self.db.commit()
        await self.db.refresh(video)
        return video
    
    async def update_video_content(
        self,
        video_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        script_content: Optional[str] = None,
        thumbnail_url: Optional[str] = None,
        video_url: Optional[str] = None,
        youtube_video_id: Optional[str] = None
    ) -> Optional[Video]:
        """Update video content."""
        result = await self.db.execute(
            select(Video).where(Video.id == video_id)
        )
        video = result.scalar_one_or_none()
        
        if not video:
            return None
        
        if title is not None:
            video.title = title
        if description is not None:
            video.description = description
        if script_content is not None:
            video.script_content = script_content
        if thumbnail_url is not None:
            video.thumbnail_url = thumbnail_url
        if video_url is not None:
            video.video_url = video_url
        if youtube_video_id is not None:
            video.youtube_video_id = youtube_video_id
        
        video.updated_at = datetime.utcnow()
        
        await self.db.commit()
        await self.db.refresh(video)
        return video
    
    async def update_video_cost(self, video_id: str, total_cost: float) -> Optional[Video]:
        """Update video total cost."""
        result = await self.db.execute(
            select(Video).where(Video.id == video_id)
        )
        video = result.scalar_one_or_none()
        
        if not video:
            return None
        
        video.total_cost = total_cost
        video.updated_at = datetime.utcnow()
        
        await self.db.commit()
        await self.db.refresh(video)
        return video
    
    async def set_pipeline_info(
        self,
        video_id: str,
        pipeline_id: str,
        pipeline_started_at: Optional[datetime] = None
    ) -> Optional[Video]:
        """Set video pipeline information."""
        result = await self.db.execute(
            select(Video).where(Video.id == video_id)
        )
        video = result.scalar_one_or_none()
        
        if not video:
            return None
        
        video.pipeline_id = pipeline_id
        video.pipeline_started_at = pipeline_started_at or datetime.utcnow()
        video.updated_at = datetime.utcnow()
        
        await self.db.commit()
        await self.db.refresh(video)
        return video
    
    async def get_user_daily_video_count(self, user_id: str, date: Optional[datetime] = None) -> int:
        """Get count of videos created by user today."""
        target_date = date or datetime.utcnow()
        start_of_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)
        
        result = await self.db.execute(
            select(func.count(Video.id))
            .where(and_(
                Video.user_id == user_id,
                Video.created_at >= start_of_day,
                Video.created_at < end_of_day
            ))
        )
        return result.scalar_one()
    
    async def check_ownership(self, video_id: str, user_id: str) -> bool:
        """Check if user owns the video."""
        result = await self.db.execute(
            select(Video.id)
            .where(and_(
                Video.id == video_id,
                Video.user_id == user_id
            ))
        )
        return result.scalar_one_or_none() is not None
    
    async def check_channel_ownership(self, video_id: str, channel_id: str, user_id: str) -> bool:
        """Check if video belongs to user's channel."""
        result = await self.db.execute(
            select(Video.id)
            .where(and_(
                Video.id == video_id,
                Video.channel_id == channel_id,
                Video.user_id == user_id
            ))
        )
        return result.scalar_one_or_none() is not None
    
    async def get_pending_videos(self, limit: int = 10) -> List[Video]:
        """Get pending videos for processing."""
        result = await self.db.execute(
            select(Video)
            .where(Video.status == VideoStatus.PENDING)
            .order_by(asc(Video.priority), asc(Video.created_at))
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_processing_videos(self) -> List[Video]:
        """Get videos currently being processed."""
        result = await self.db.execute(
            select(Video)
            .where(Video.status == VideoStatus.PROCESSING)
            .order_by(asc(Video.created_at))
        )
        return result.scalars().all()
    
    async def get_scheduled_videos(self, up_to: Optional[datetime] = None) -> List[Video]:
        """Get videos scheduled for publishing."""
        target_time = up_to or datetime.utcnow()
        
        result = await self.db.execute(
            select(Video)
            .where(and_(
                Video.status == VideoStatus.SCHEDULED,
                Video.scheduled_publish_at <= target_time
            ))
            .order_by(asc(Video.scheduled_publish_at))
        )
        return result.scalars().all()
    
    async def increment_retry_count(self, video_id: str) -> Optional[Video]:
        """Increment retry count for failed video."""
        result = await self.db.execute(
            select(Video).where(Video.id == video_id)
        )
        video = result.scalar_one_or_none()
        
        if not video:
            return None
        
        video.retry_count += 1
        video.updated_at = datetime.utcnow()
        
        await self.db.commit()
        await self.db.refresh(video)
        return video
    
    async def get_user_video_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user's video statistics."""
        # Get counts by status
        status_counts = await self.db.execute(
            select(
                Video.status,
                func.count(Video.id).label('count')
            )
            .where(Video.user_id == user_id)
            .group_by(Video.status)
        )
        
        stats = {status.value: 0 for status in VideoStatus}
        for row in status_counts:
            stats[row.status.value] = row.count
        
        # Get cost statistics
        cost_stats = await self.db.execute(
            select(
                func.count(Video.id).label('total_videos'),
                func.coalesce(func.sum(Video.total_cost), 0).label('total_cost'),
                func.coalesce(func.avg(Video.total_cost), 0).label('avg_cost')
            )
            .where(and_(
                Video.user_id == user_id,
                Video.total_cost.isnot(None)
            ))
        )
        
        cost_data = cost_stats.first()
        
        return {
            'status_counts': stats,
            'total_videos': cost_data.total_videos or 0,
            'total_cost': float(cost_data.total_cost or 0),
            'average_cost': float(cost_data.avg_cost or 0)
        }
    
    async def delete_video(self, video_id: str) -> bool:
        """Delete video from database."""
        result = await self.db.execute(
            select(Video).where(Video.id == video_id)
        )
        video = result.scalar_one_or_none()
        
        if video:
            await self.db.delete(video)
            await self.db.commit()
            return True
        return False