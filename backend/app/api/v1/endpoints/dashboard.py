"""
User Dashboard API Endpoints
Provides comprehensive dashboard data and analytics
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from pydantic import BaseModel
from datetime import datetime, timedelta
import logging

from app.db.session import get_db
from app.api.v1.endpoints.auth import get_current_verified_user
from app.models.user import User
from app.models.channel import Channel
from app.models.video import Video
from app.models.cost import Cost
from app.models.analytics import Analytics

logger = logging.getLogger(__name__)

router = APIRouter()


class DashboardStats(BaseModel):
    """Dashboard statistics response"""
    total_channels: int
    total_videos: int
    total_views: int
    total_revenue: float
    total_cost: float
    profit_margin: float
    videos_today: int
    videos_this_week: int
    videos_this_month: int
    avg_video_performance: float
    top_performing_channel: Optional[Dict[str, Any]]
    recent_activity: List[Dict[str, Any]]


class PerformanceMetrics(BaseModel):
    """Performance metrics response"""
    period: str
    views: int
    engagement_rate: float
    watch_time_hours: float
    subscriber_growth: int
    revenue: float
    cost: float
    profit: float
    roi: float


class ChannelSummary(BaseModel):
    """Channel summary for dashboard"""
    channel_id: str
    channel_name: str
    subscriber_count: int
    video_count: int
    total_views: int
    last_video_date: Optional[datetime]
    status: str
    performance_score: float


class VideoQueueItem(BaseModel):
    """Video queue item for dashboard"""
    queue_id: str
    channel_id: str
    channel_name: str
    video_title: str
    scheduled_time: datetime
    status: str
    priority: int
    estimated_cost: float


class RecentActivity(BaseModel):
    """Recent activity item"""
    timestamp: datetime
    type: str  # video_generated, channel_created, milestone_reached
    title: str
    description: str
    metadata: Dict[str, Any]


@router.get("/overview", response_model=DashboardStats)
async def get_dashboard_overview(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> DashboardStats:
    """
    Get comprehensive dashboard overview statistics
    """
    try:
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = now - timedelta(days=now.weekday())
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Get channel count
        channel_count_result = await db.execute(
            select(func.count(Channel.id)).where(Channel.user_id == current_user.id)
        )
        total_channels = channel_count_result.scalar() or 0
        
        # Get video statistics
        video_stats = await db.execute(
            select(
                func.count(Video.id).label('total'),
                func.sum(Video.view_count).label('views'),
                func.sum(
                    func.case(
                        (Video.created_at >= today_start, 1),
                        else_=0
                    )
                ).label('today'),
                func.sum(
                    func.case(
                        (Video.created_at >= week_start, 1),
                        else_=0
                    )
                ).label('week'),
                func.sum(
                    func.case(
                        (Video.created_at >= month_start, 1),
                        else_=0
                    )
                ).label('month')
            ).where(
                Video.channel_id.in_(
                    select(Channel.id).where(Channel.user_id == current_user.id)
                )
            )
        )
        video_data = video_stats.one()
        
        # Get cost and revenue data
        cost_stats = await db.execute(
            select(
                func.sum(Cost.amount).label('total_cost'),
                func.sum(Cost.estimated_revenue).label('total_revenue')
            ).where(
                Cost.user_id == current_user.id
            )
        )
        cost_data = cost_stats.one()
        
        total_cost = cost_data.total_cost or 0
        total_revenue = cost_data.total_revenue or 0
        profit = total_revenue - total_cost
        profit_margin = (profit / total_revenue * 100) if total_revenue > 0 else 0
        
        # Get top performing channel
        top_channel_result = await db.execute(
            select(
                Channel.id,
                Channel.name,
                func.count(Video.id).label('video_count'),
                func.sum(Video.view_count).label('total_views')
            ).join(
                Video, Channel.id == Video.channel_id
            ).where(
                Channel.user_id == current_user.id
            ).group_by(
                Channel.id, Channel.name
            ).order_by(
                func.sum(Video.view_count).desc()
            ).limit(1)
        )
        top_channel_data = top_channel_result.first()
        
        top_channel = None
        if top_channel_data:
            top_channel = {
                "channel_id": str(top_channel_data.id),
                "channel_name": top_channel_data.name,
                "video_count": top_channel_data.video_count,
                "total_views": top_channel_data.total_views or 0
            }
        
        # Get recent activity
        recent_activity = await get_recent_activity(db, current_user.id, limit=10)
        
        # Calculate average video performance
        avg_performance = (video_data.views / video_data.total) if video_data.total > 0 else 0
        
        return DashboardStats(
            total_channels=total_channels,
            total_videos=video_data.total or 0,
            total_views=video_data.views or 0,
            total_revenue=total_revenue,
            total_cost=total_cost,
            profit_margin=profit_margin,
            videos_today=video_data.today or 0,
            videos_this_week=video_data.week or 0,
            videos_this_month=video_data.month or 0,
            avg_video_performance=avg_performance,
            top_performing_channel=top_channel,
            recent_activity=recent_activity
        )
        
    except Exception as e:
        logger.error(f"Dashboard overview error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch dashboard data"
        )


@router.get("/performance", response_model=List[PerformanceMetrics])
async def get_performance_metrics(
    period: str = Query("7d", description="Time period: 24h, 7d, 30d, 90d"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> List[PerformanceMetrics]:
    """
    Get performance metrics over time
    """
    try:
        # Determine date range
        now = datetime.utcnow()
        if period == "24h":
            start_date = now - timedelta(days=1)
            interval = "hour"
        elif period == "7d":
            start_date = now - timedelta(days=7)
            interval = "day"
        elif period == "30d":
            start_date = now - timedelta(days=30)
            interval = "day"
        elif period == "90d":
            start_date = now - timedelta(days=90)
            interval = "week"
        else:
            start_date = now - timedelta(days=7)
            interval = "day"
        
        # Get analytics data
        analytics_data = await db.execute(
            select(
                Analytics.date,
                func.sum(Analytics.views).label('views'),
                func.avg(Analytics.engagement_rate).label('engagement_rate'),
                func.sum(Analytics.watch_time_minutes).label('watch_time'),
                func.sum(Analytics.subscriber_change).label('subscriber_growth'),
                func.sum(Analytics.estimated_revenue).label('revenue')
            ).where(
                and_(
                    Analytics.user_id == current_user.id,
                    Analytics.date >= start_date
                )
            ).group_by(Analytics.date)
            .order_by(Analytics.date)
        )
        
        metrics = []
        for row in analytics_data:
            # Get costs for this date
            cost_result = await db.execute(
                select(func.sum(Cost.amount)).where(
                    and_(
                        Cost.user_id == current_user.id,
                        func.date(Cost.created_at) == row.date
                    )
                )
            )
            daily_cost = cost_result.scalar() or 0
            
            revenue = row.revenue or 0
            profit = revenue - daily_cost
            roi = (profit / daily_cost * 100) if daily_cost > 0 else 0
            
            metrics.append(PerformanceMetrics(
                period=row.date.isoformat(),
                views=row.views or 0,
                engagement_rate=row.engagement_rate or 0,
                watch_time_hours=(row.watch_time or 0) / 60,
                subscriber_growth=row.subscriber_growth or 0,
                revenue=revenue,
                cost=daily_cost,
                profit=profit,
                roi=roi
            ))
        
        return metrics
        
    except Exception as e:
        logger.error(f"Performance metrics error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch performance metrics"
        )


@router.get("/channels", response_model=List[ChannelSummary])
async def get_channel_summaries(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> List[ChannelSummary]:
    """
    Get summary of all user channels for dashboard
    """
    try:
        channels_data = await db.execute(
            select(
                Channel,
                func.count(Video.id).label('video_count'),
                func.sum(Video.view_count).label('total_views'),
                func.max(Video.created_at).label('last_video_date')
            ).outerjoin(
                Video, Channel.id == Video.channel_id
            ).where(
                Channel.user_id == current_user.id
            ).group_by(Channel.id)
        )
        
        summaries = []
        for row in channels_data:
            channel = row[0]
            
            # Calculate performance score (0-100)
            performance_score = calculate_channel_performance(
                video_count=row.video_count or 0,
                total_views=row.total_views or 0,
                subscriber_count=channel.subscriber_count or 0
            )
            
            summaries.append(ChannelSummary(
                channel_id=str(channel.id),
                channel_name=channel.name,
                subscriber_count=channel.subscriber_count or 0,
                video_count=row.video_count or 0,
                total_views=row.total_views or 0,
                last_video_date=row.last_video_date,
                status=channel.status,
                performance_score=performance_score
            ))
        
        return summaries
        
    except Exception as e:
        logger.error(f"Channel summaries error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch channel summaries"
        )


@router.get("/video-queue", response_model=List[VideoQueueItem])
async def get_video_queue(
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> List[VideoQueueItem]:
    """
    Get upcoming video queue
    """
    try:
        # This would query a VideoQueue table
        # For now, return mock data
        queue_items = []
        
        channels = await db.execute(
            select(Channel).where(Channel.user_id == current_user.id).limit(3)
        )
        
        for i, channel_row in enumerate(channels):
            channel = channel_row[0]
            queue_items.append(VideoQueueItem(
                queue_id=f"queue_{i+1}",
                channel_id=str(channel.id),
                channel_name=channel.name,
                video_title=f"Upcoming Video {i+1}",
                scheduled_time=datetime.utcnow() + timedelta(hours=i*2),
                status="scheduled",
                priority=i+1,
                estimated_cost=0.45
            ))
        
        return queue_items
        
    except Exception as e:
        logger.error(f"Video queue error: {e}")
        return []


@router.get("/analytics-summary")
async def get_analytics_summary(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Get analytics summary for dashboard widgets
    """
    try:
        now = datetime.utcnow()
        last_30_days = now - timedelta(days=30)
        
        # Get key metrics
        result = await db.execute(
            select(
                func.sum(Analytics.views).label('total_views'),
                func.avg(Analytics.engagement_rate).label('avg_engagement'),
                func.sum(Analytics.watch_time_minutes).label('total_watch_time'),
                func.sum(Analytics.subscriber_change).label('subscriber_growth'),
                func.sum(Analytics.estimated_revenue).label('total_revenue')
            ).where(
                and_(
                    Analytics.user_id == current_user.id,
                    Analytics.date >= last_30_days
                )
            )
        )
        
        data = result.one()
        
        return {
            "period": "last_30_days",
            "metrics": {
                "total_views": data.total_views or 0,
                "avg_engagement_rate": round(data.avg_engagement or 0, 2),
                "total_watch_hours": round((data.total_watch_time or 0) / 60, 1),
                "subscriber_growth": data.subscriber_growth or 0,
                "estimated_revenue": round(data.total_revenue or 0, 2)
            },
            "trends": {
                "views_trend": "up",  # Would calculate actual trend
                "engagement_trend": "stable",
                "revenue_trend": "up"
            }
        }
        
    except Exception as e:
        logger.error(f"Analytics summary error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch analytics summary"
        )


# Helper functions
async def get_recent_activity(
    db: AsyncSession,
    user_id: str,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Get recent activity for user"""
    activities = []
    
    # Get recent videos
    recent_videos = await db.execute(
        select(Video, Channel.name).join(
            Channel, Video.channel_id == Channel.id
        ).where(
            Channel.user_id == user_id
        ).order_by(
            Video.created_at.desc()
        ).limit(5)
    )
    
    for video, channel_name in recent_videos:
        activities.append({
            "timestamp": video.created_at.isoformat(),
            "type": "video_generated",
            "title": "New Video Generated",
            "description": f"'{video.title}' for {channel_name}",
            "metadata": {
                "video_id": str(video.id),
                "channel_name": channel_name,
                "views": video.view_count
            }
        })
    
    # Sort by timestamp
    activities.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return activities[:limit]


def calculate_channel_performance(
    video_count: int,
    total_views: int,
    subscriber_count: int
) -> float:
    """Calculate channel performance score (0-100)"""
    # Simple scoring algorithm
    score = 0
    
    # Video frequency score (max 30 points)
    if video_count > 0:
        score += min(30, video_count * 2)
    
    # Views score (max 40 points)
    if total_views > 0:
        score += min(40, total_views / 1000)
    
    # Subscriber score (max 30 points)
    if subscriber_count > 0:
        score += min(30, subscriber_count / 100)
    
    return min(100, score)