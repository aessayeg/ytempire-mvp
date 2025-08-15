"""
Optimized Database Queries Service
Fixes N+1 query problems with eager loading and query optimization
"""
from typing import List, Optional, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from sqlalchemy.orm import selectinload, joinedload, subqueryload, contains_eager
from app.models.user import User
from app.models.channel import Channel
from app.models.video import Video
from app.models.analytics import Analytics
from app.models.cost import Cost
import logging

logger = logging.getLogger(__name__)

class OptimizedQueryService:
    """Service for optimized database queries without N+1 problems"""
    
    @staticmethod
    async def get_user_with_channels(db: AsyncSession, user_id: int) -> Optional[User]:
        """Get user with all channels eagerly loaded"""
        result = await db.execute(
            select(User)
            .options(
                selectinload(User.channels).selectinload(Channel.videos),
                selectinload(User.payments)
            )
            .where(User.id == user_id)
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_channels_with_videos(db: AsyncSession, user_id: int) -> List[Channel]:
        """Get all channels for a user with videos eagerly loaded"""
        result = await db.execute(
            select(Channel)
            .options(
                selectinload(Channel.videos).selectinload(Video.analytics),
                selectinload(Channel.videos).selectinload(Video.costs),
                selectinload(Channel.owner)
            )
            .where(Channel.owner_id == user_id)
            .where(Channel.is_active == True)
        )
        return result.scalars().all()
    
    @staticmethod
    async def get_channel_with_all_relations(db: AsyncSession, channel_id: int) -> Optional[Channel]:
        """Get channel with all related data eagerly loaded"""
        result = await db.execute(
            select(Channel)
            .options(
                selectinload(Channel.owner),
                selectinload(Channel.videos).options(
                    selectinload(Video.analytics),
                    selectinload(Video.costs)
                )
            )
            .where(Channel.id == channel_id)
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_videos_with_analytics(db: AsyncSession, channel_id: int, limit: int = 50) -> List[Video]:
        """Get videos with analytics data eagerly loaded"""
        result = await db.execute(
            select(Video)
            .options(
                selectinload(Video.analytics),
                selectinload(Video.costs),
                selectinload(Video.channel).selectinload(Channel.owner)
            )
            .where(Video.channel_id == channel_id)
            .order_by(Video.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()
    
    @staticmethod
    async def get_dashboard_data(db: AsyncSession, user_id: int) -> dict:
        """Get optimized dashboard data with single query"""
        # Use subqueries to get aggregated data in one go
        channels_subq = select(
            Channel.owner_id,
            func.count(Channel.id).label('channel_count')
        ).where(
            Channel.owner_id == user_id,
            Channel.is_active == True
        ).group_by(Channel.owner_id).subquery()
        
        videos_subq = select(
            Channel.owner_id,
            func.count(Video.id).label('video_count'),
            func.sum(Video.views).label('total_views'),
            func.sum(Video.likes).label('total_likes')
        ).select_from(Channel).join(Video).where(
            Channel.owner_id == user_id
        ).group_by(Channel.owner_id).subquery()
        
        costs_subq = select(
            Channel.owner_id,
            func.sum(Cost.amount).label('total_cost')
        ).select_from(Channel).join(Video).join(Cost).where(
            Channel.owner_id == user_id
        ).group_by(Channel.owner_id).subquery()
        
        # Main query joining all subqueries
        result = await db.execute(
            select(
                User,
                channels_subq.c.channel_count,
                videos_subq.c.video_count,
                videos_subq.c.total_views,
                videos_subq.c.total_likes,
                costs_subq.c.total_cost
            ).outerjoin(
                channels_subq, User.id == channels_subq.c.owner_id
            ).outerjoin(
                videos_subq, User.id == videos_subq.c.owner_id
            ).outerjoin(
                costs_subq, User.id == costs_subq.c.owner_id
            ).where(User.id == user_id)
        )
        
        row = result.first()
        if row:
            return {
                'user': row[0],
                'channel_count': row[1] or 0,
                'video_count': row[2] or 0,
                'total_views': row[3] or 0,
                'total_likes': row[4] or 0,
                'total_cost': float(row[5] or 0)
            }
        return None
    
    @staticmethod
    async def get_videos_batch(db: AsyncSession, video_ids: List[int]) -> List[Video]:
        """Get multiple videos in single query with all relations"""
        result = await db.execute(
            select(Video)
            .options(
                selectinload(Video.channel).selectinload(Channel.owner),
                selectinload(Video.analytics),
                selectinload(Video.costs)
            )
            .where(Video.id.in_(video_ids))
        )
        return result.scalars().all()
    
    @staticmethod
    async def get_recent_videos_optimized(db: AsyncSession, limit: int = 10) -> List[Video]:
        """Get recent videos with optimized loading"""
        result = await db.execute(
            select(Video)
            .options(
                joinedload(Video.channel).joinedload(Channel.owner),
                selectinload(Video.analytics),
                selectinload(Video.costs)
            )
            .order_by(Video.created_at.desc())
            .limit(limit)
        )
        return result.unique().scalars().all()
    
    @staticmethod
    async def get_channel_performance_metrics(db: AsyncSession, channel_id: int) -> dict:
        """Get channel performance metrics in optimized query"""
        result = await db.execute(
            select(
                func.count(Video.id).label('total_videos'),
                func.sum(Video.views).label('total_views'),
                func.sum(Video.likes).label('total_likes'),
                func.sum(Video.comments).label('total_comments'),
                func.avg(Video.views).label('avg_views'),
                func.avg(Video.duration).label('avg_duration'),
                func.max(Video.created_at).label('last_video_date'),
                func.sum(Cost.amount).label('total_cost')
            ).select_from(Channel)
            .join(Video, Channel.id == Video.channel_id)
            .outerjoin(Cost, Video.id == Cost.video_id)
            .where(Channel.id == channel_id)
            .group_by(Channel.id)
        )
        
        row = result.first()
        if row:
            return {
                'total_videos': row.total_videos or 0,
                'total_views': row.total_views or 0,
                'total_likes': row.total_likes or 0,
                'total_comments': row.total_comments or 0,
                'avg_views': float(row.avg_views or 0),
                'avg_duration': float(row.avg_duration or 0),
                'last_video_date': row.last_video_date,
                'total_cost': float(row.total_cost or 0)
            }
        return {}

# Global instance
optimized_queries = OptimizedQueryService()