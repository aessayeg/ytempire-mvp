"""
Analytics Repository
Owner: Analytics Engineer
"""

from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, desc, asc, text
from sqlalchemy.orm import selectinload
from datetime import datetime, timedelta, date

from app.models.analytics import ChannelAnalytics, VideoAnalytics, CostTracking, ServiceType
from app.models.channel import Channel
from app.models.video import Video, VideoStatus
from app.models.user import User


class AnalyticsRepository:
    """Repository for analytics data operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_user_dashboard_metrics(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive dashboard metrics for user."""
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=days)
        
        # Get user's channels
        user_channels = await self.db.execute(
            select(Channel.id)
            .where(and_(Channel.user_id == user_id, Channel.is_active == True))
        )
        channel_ids = [row[0] for row in user_channels]
        
        if not channel_ids:
            return self._empty_dashboard_metrics()
        
        # Video statistics
        video_stats = await self.db.execute(
            select(
                func.count(Video.id).label('total_videos'),
                func.count(Video.id).filter(Video.status == VideoStatus.COMPLETED).label('completed_videos'),
                func.count(Video.id).filter(Video.status == VideoStatus.FAILED).label('failed_videos'),
                func.coalesce(func.sum(Video.total_cost), 0).label('total_cost'),
                func.coalesce(func.avg(Video.total_cost), 0).label('avg_cost_per_video')
            )
            .where(and_(
                Video.user_id == user_id,
                Video.created_at >= start_date
            ))
        )
        video_data = video_stats.first()
        
        # Channel analytics aggregation
        channel_analytics = await self.db.execute(
            select(
                func.sum(ChannelAnalytics.views).label('total_views'),
                func.sum(ChannelAnalytics.subscribers).label('total_subscribers'),
                func.sum(ChannelAnalytics.estimated_revenue).label('total_revenue'),
                func.avg(ChannelAnalytics.engagement_rate).label('avg_engagement'),
                func.avg(ChannelAnalytics.click_through_rate).label('avg_ctr')
            )
            .where(and_(
                ChannelAnalytics.channel_id.in_(channel_ids),
                ChannelAnalytics.date >= start_date,
                ChannelAnalytics.date <= end_date
            ))
        )
        analytics_data = channel_analytics.first()
        
        # Best performing video
        best_video = await self.db.execute(
            select(Video, func.coalesce(func.sum(VideoAnalytics.views), 0).label('total_views'))
            .outerjoin(VideoAnalytics, Video.id == VideoAnalytics.video_id)
            .where(Video.user_id == user_id)
            .group_by(Video.id)
            .order_by(desc('total_views'))
            .limit(1)
        )
        best_video_data = best_video.first()
        
        # Channel performance
        channel_performance = await self.db.execute(
            select(
                Channel,
                func.count(Video.id).label('video_count'),
                func.coalesce(func.sum(ChannelAnalytics.views), 0).label('total_views'),
                func.coalesce(func.max(ChannelAnalytics.subscribers), 0).label('subscribers')
            )
            .outerjoin(Video, Channel.id == Video.channel_id)
            .outerjoin(ChannelAnalytics, Channel.id == ChannelAnalytics.channel_id)
            .where(and_(
                Channel.user_id == user_id,
                Channel.is_active == True
            ))
            .group_by(Channel.id)
            .order_by(desc('total_views'))
        )
        
        return {
            'total_videos': video_data.total_videos or 0,
            'completed_videos': video_data.completed_videos or 0,
            'failed_videos': video_data.failed_videos or 0,
            'total_views': int(analytics_data.total_views or 0),
            'total_subscribers': int(analytics_data.total_subscribers or 0),
            'total_revenue': float(analytics_data.total_revenue or 0),
            'total_cost': float(video_data.total_cost or 0),
            'profit': float(analytics_data.total_revenue or 0) - float(video_data.total_cost or 0),
            'average_cost_per_video': float(video_data.avg_cost_per_video or 0),
            'average_engagement_rate': float(analytics_data.avg_engagement or 0),
            'average_ctr': float(analytics_data.avg_ctr or 0),
            'best_performing_video': {
                'id': best_video_data[0].id if best_video_data else None,
                'title': best_video_data[0].title if best_video_data else None,
                'views': int(best_video_data[1]) if best_video_data else 0,
            } if best_video_data else None,
            'channels_performance': [
                {
                    'id': row[0].id,
                    'name': row[0].name,
                    'videos': row.video_count,
                    'views': int(row.total_views),
                    'subscribers': int(row.subscribers)
                }
                for row in channel_performance
            ],
            'period_days': days
        }
    
    async def get_channel_analytics(
        self,
        channel_id: str,
        start_date: date,
        end_date: date
    ) -> List[Dict[str, Any]]:
        """Get channel analytics for date range."""
        result = await self.db.execute(
            select(ChannelAnalytics)
            .where(and_(
                ChannelAnalytics.channel_id == channel_id,
                ChannelAnalytics.date >= start_date,
                ChannelAnalytics.date <= end_date
            ))
            .order_by(ChannelAnalytics.date)
        )
        
        analytics = result.scalars().all()
        
        return [
            {
                'date': analytic.date.isoformat(),
                'views': analytic.views,
                'subscribers': analytic.subscribers,
                'videos_published': analytic.videos_published,
                'watch_time_minutes': analytic.watch_time_minutes,
                'estimated_revenue': float(analytic.estimated_revenue),
                'engagement_rate': analytic.engagement_rate,
                'click_through_rate': analytic.click_through_rate,
                'average_view_duration': analytic.average_view_duration
            }
            for analytic in analytics
        ]
    
    async def get_video_analytics(
        self,
        video_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """Get video analytics for date range."""
        query = select(VideoAnalytics).where(VideoAnalytics.video_id == video_id)
        
        if start_date:
            query = query.where(VideoAnalytics.date >= start_date)
        if end_date:
            query = query.where(VideoAnalytics.date <= end_date)
        
        query = query.order_by(VideoAnalytics.date)
        
        result = await self.db.execute(query)
        analytics = result.scalars().all()
        
        return [
            {
                'date': analytic.date.isoformat(),
                'views': analytic.views,
                'likes': analytic.likes,
                'dislikes': analytic.dislikes,
                'comments': analytic.comments,
                'shares': analytic.shares,
                'watch_time_minutes': analytic.watch_time_minutes,
                'impressions': analytic.impressions,
                'click_through_rate': analytic.click_through_rate,
                'average_view_duration': analytic.average_view_duration,
                'estimated_revenue': float(analytic.estimated_revenue),
                'engagement_score': analytic.engagement_score,
                'trending_score': analytic.trending_score
            }
            for analytic in analytics
        ]
    
    async def get_cost_breakdown(
        self,
        user_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        video_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get cost breakdown by service type."""
        query = select(
            CostTracking.service_type,
            func.sum(CostTracking.cost_amount).label('total_cost'),
            func.count(CostTracking.id).label('operation_count'),
            func.avg(CostTracking.cost_amount).label('avg_cost')
        ).where(CostTracking.user_id == user_id)
        
        if start_date:
            query = query.where(CostTracking.created_at >= start_date)
        if end_date:
            query = query.where(CostTracking.created_at <= end_date + timedelta(days=1))
        if video_id:
            query = query.where(CostTracking.video_id == video_id)
        
        query = query.group_by(CostTracking.service_type)
        
        result = await self.db.execute(query)
        cost_data = result.all()
        
        breakdown = {}
        total_cost = 0
        
        for row in cost_data:
            service_cost = float(row.total_cost)
            breakdown[row.service_type.value] = {
                'total_cost': service_cost,
                'operation_count': row.operation_count,
                'average_cost': float(row.avg_cost)
            }
            total_cost += service_cost
        
        return {
            'breakdown': breakdown,
            'total_cost': total_cost,
            'period_start': start_date.isoformat() if start_date else None,
            'period_end': end_date.isoformat() if end_date else None
        }
    
    async def get_trending_content(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get user's trending content based on engagement metrics."""
        result = await self.db.execute(
            select(
                Video,
                func.coalesce(func.avg(VideoAnalytics.trending_score), 0).label('avg_trending_score'),
                func.coalesce(func.sum(VideoAnalytics.views), 0).label('total_views'),
                func.coalesce(func.avg(VideoAnalytics.engagement_score), 0).label('avg_engagement')
            )
            .outerjoin(VideoAnalytics, Video.id == VideoAnalytics.video_id)
            .where(and_(
                Video.user_id == user_id,
                Video.status == VideoStatus.COMPLETED
            ))
            .group_by(Video.id)
            .order_by(desc('avg_trending_score'), desc('total_views'))
            .limit(limit)
        )
        
        trending = result.all()
        
        return [
            {
                'video_id': row[0].id,
                'title': row[0].title,
                'channel_id': row[0].channel_id,
                'youtube_video_id': row[0].youtube_video_id,
                'trending_score': float(row.avg_trending_score),
                'total_views': int(row.total_views),
                'engagement_score': float(row.avg_engagement),
                'published_at': row[0].published_at.isoformat() if row[0].published_at else None
            }
            for row in trending
        ]
    
    async def get_performance_comparison(
        self,
        user_id: str,
        current_period_days: int = 30
    ) -> Dict[str, Any]:
        """Compare current period performance with previous period."""
        end_date = datetime.utcnow().date()
        current_start = end_date - timedelta(days=current_period_days)
        previous_start = current_start - timedelta(days=current_period_days)
        previous_end = current_start - timedelta(days=1)
        
        # Current period metrics
        current_metrics = await self.get_user_dashboard_metrics(user_id, current_period_days)
        
        # Previous period metrics
        previous_videos = await self.db.execute(
            select(
                func.count(Video.id).label('total_videos'),
                func.coalesce(func.sum(Video.total_cost), 0).label('total_cost')
            )
            .where(and_(
                Video.user_id == user_id,
                Video.created_at >= previous_start,
                Video.created_at < current_start
            ))
        )
        previous_video_data = previous_videos.first()
        
        # Calculate percentage changes
        def calculate_change(current: float, previous: float) -> Dict[str, Any]:
            if previous == 0:
                return {'value': current, 'change_percent': 0, 'direction': 'neutral'}
            
            change_percent = ((current - previous) / previous) * 100
            direction = 'up' if change_percent > 0 else 'down' if change_percent < 0 else 'neutral'
            
            return {
                'value': current,
                'change_percent': round(change_percent, 2),
                'direction': direction
            }
        
        return {
            'current_period': {
                'days': current_period_days,
                'start_date': current_start.isoformat(),
                'end_date': end_date.isoformat()
            },
            'previous_period': {
                'days': current_period_days,
                'start_date': previous_start.isoformat(),
                'end_date': previous_end.isoformat()
            },
            'metrics': {
                'total_videos': calculate_change(
                    current_metrics['total_videos'],
                    previous_video_data.total_videos or 0
                ),
                'total_views': calculate_change(
                    current_metrics['total_views'],
                    0  # Would need previous analytics data
                ),
                'total_cost': calculate_change(
                    current_metrics['total_cost'],
                    float(previous_video_data.total_cost or 0)
                )
            }
        }
    
    def _empty_dashboard_metrics(self) -> Dict[str, Any]:
        """Return empty dashboard metrics structure."""
        return {
            'total_videos': 0,
            'completed_videos': 0,
            'failed_videos': 0,
            'total_views': 0,
            'total_subscribers': 0,
            'total_revenue': 0.0,
            'total_cost': 0.0,
            'profit': 0.0,
            'average_cost_per_video': 0.0,
            'average_engagement_rate': 0.0,
            'average_ctr': 0.0,
            'best_performing_video': None,
            'channels_performance': [],
            'period_days': 30
        }