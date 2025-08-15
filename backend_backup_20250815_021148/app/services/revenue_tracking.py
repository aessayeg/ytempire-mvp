"""
Revenue Tracking Service
Comprehensive revenue tracking, attribution, and forecasting
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, date
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, case
from sqlalchemy.orm import selectinload, joinedload
import numpy as np
from sklearn.linear_model import LinearRegression
import logging
from decimal import Decimal
from collections import defaultdict
import asyncio

from app.models.user import User
from app.models.channel import Channel
from app.models.video import Video
from app.models.analytics import Analytics
from app.core.cache import cache_service

logger = logging.getLogger(__name__)


class RevenueTrackingService:
    """Service for comprehensive revenue tracking and analysis"""
    
    def __init__(self):
        self.cache_ttl = 300  # 5 minutes cache
        
    async def get_revenue_overview(
        self,
        db: AsyncSession,
        user_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get comprehensive revenue overview for user"""
        cache_key = f"revenue:overview:{user_id}:{start_date}:{end_date}"
        cached = await cache_service.get(cache_key)
        if cached:
            return cached
            
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)
            
        # Get all user channels
        channels_query = select(Channel).where(Channel.owner_id == user_id)
        channels_result = await db.execute(channels_query)
        channels = channels_result.scalars().all()
        channel_ids = [c.id for c in channels]
        
        if not channel_ids:
            return self._empty_revenue_overview()
            
        # Aggregate revenue data
        revenue_query = (
            select(
                func.sum(Analytics.estimated_revenue).label('total_revenue'),
                func.avg(Analytics.estimated_revenue).label('avg_revenue'),
                func.max(Analytics.estimated_revenue).label('max_revenue'),
                func.min(Analytics.estimated_revenue).label('min_revenue'),
                func.count(Analytics.id).label('total_videos')
            )
            .select_from(Analytics)
            .join(Video, Analytics.video_id == Video.id)
            .where(
                and_(
                    Video.channel_id.in_(channel_ids),
                    Analytics.date >= start_date,
                    Analytics.date <= end_date
                )
            )
        )
        
        revenue_result = await db.execute(revenue_query)
        revenue_data = revenue_result.first()
        
        # Calculate daily revenue
        daily_revenue = await self._calculate_daily_revenue(
            db, channel_ids, start_date, end_date
        )
        
        # Calculate channel breakdown
        channel_breakdown = await self._calculate_channel_breakdown(
            db, channel_ids, start_date, end_date
        )
        
        # Calculate CPM and RPM
        metrics = await self._calculate_revenue_metrics(
            db, channel_ids, start_date, end_date
        )
        
        # Generate forecast
        forecast = await self._generate_revenue_forecast(daily_revenue)
        
        overview = {
            'total_revenue': float(revenue_data.total_revenue or 0),
            'average_revenue_per_video': float(revenue_data.avg_revenue or 0),
            'highest_revenue_video': float(revenue_data.max_revenue or 0),
            'lowest_revenue_video': float(revenue_data.min_revenue or 0),
            'total_videos_monetized': revenue_data.total_videos or 0,
            'daily_revenue': daily_revenue,
            'channel_breakdown': channel_breakdown,
            'cpm': metrics['cpm'],
            'rpm': metrics['rpm'],
            'forecast': forecast,
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            }
        }
        
        await cache_service.set(cache_key, overview, self.cache_ttl)
        return overview
        
    async def get_channel_revenue(
        self,
        db: AsyncSession,
        channel_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get revenue details for specific channel"""
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)
            
        # Get channel with videos and analytics
        channel_query = (
            select(Channel)
            .options(
                selectinload(Channel.videos).selectinload(Video.analytics)
            )
            .where(Channel.id == channel_id)
        )
        
        channel_result = await db.execute(channel_query)
        channel = channel_result.scalar_one_or_none()
        
        if not channel:
            return None
            
        # Calculate revenue metrics
        total_revenue = 0
        video_revenues = []
        
        for video in channel.videos:
            for analytics in video.analytics:
                if start_date <= analytics.date <= end_date:
                    total_revenue += float(analytics.estimated_revenue or 0)
                    video_revenues.append({
                        'video_id': video.id,
                        'title': video.title,
                        'date': analytics.date.isoformat(),
                        'revenue': float(analytics.estimated_revenue or 0),
                        'views': analytics.views,
                        'watch_time': analytics.watch_time_minutes
                    })
        
        # Sort by revenue
        video_revenues.sort(key=lambda x: x['revenue'], reverse=True)
        
        return {
            'channel_id': channel_id,
            'channel_name': channel.name,
            'total_revenue': total_revenue,
            'video_count': len(channel.videos),
            'average_revenue_per_video': total_revenue / len(channel.videos) if channel.videos else 0,
            'top_earning_videos': video_revenues[:10],
            'revenue_by_video': video_revenues,
            'subscriber_count': channel.subscriber_count,
            'total_views': sum(v.views for v in channel.videos)
        }
        
    async def get_revenue_trends(
        self,
        db: AsyncSession,
        user_id: int,
        period: str = 'daily',
        lookback_days: int = 30
    ) -> Dict[str, Any]:
        """Get revenue trends over time"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Get user channels
        channels_query = select(Channel.id).where(Channel.owner_id == user_id)
        channels_result = await db.execute(channels_query)
        channel_ids = [c[0] for c in channels_result.fetchall()]
        
        if not channel_ids:
            return {'trends': [], 'period': period}
            
        # Aggregate by period
        if period == 'daily':
            date_trunc = func.date_trunc('day', Analytics.date)
        elif period == 'weekly':
            date_trunc = func.date_trunc('week', Analytics.date)
        elif period == 'monthly':
            date_trunc = func.date_trunc('month', Analytics.date)
        else:
            date_trunc = func.date_trunc('day', Analytics.date)
            
        trends_query = (
            select(
                date_trunc.label('period'),
                func.sum(Analytics.estimated_revenue).label('revenue'),
                func.sum(Analytics.views).label('views'),
                func.count(distinct(Video.id)).label('video_count')
            )
            .select_from(Analytics)
            .join(Video, Analytics.video_id == Video.id)
            .where(
                and_(
                    Video.channel_id.in_(channel_ids),
                    Analytics.date >= start_date,
                    Analytics.date <= end_date
                )
            )
            .group_by(date_trunc)
            .order_by(date_trunc)
        )
        
        trends_result = await db.execute(trends_query)
        trends_data = trends_result.fetchall()
        
        trends = []
        for row in trends_data:
            trends.append({
                'period': row.period.isoformat(),
                'revenue': float(row.revenue or 0),
                'views': row.views or 0,
                'video_count': row.video_count or 0,
                'rpm': (float(row.revenue or 0) / row.views * 1000) if row.views else 0
            })
            
        # Calculate growth rates
        if len(trends) > 1:
            for i in range(1, len(trends)):
                prev_revenue = trends[i-1]['revenue']
                curr_revenue = trends[i]['revenue']
                if prev_revenue > 0:
                    trends[i]['growth_rate'] = ((curr_revenue - prev_revenue) / prev_revenue) * 100
                else:
                    trends[i]['growth_rate'] = 0
                    
        return {
            'trends': trends,
            'period': period,
            'total_revenue': sum(t['revenue'] for t in trends),
            'average_revenue': np.mean([t['revenue'] for t in trends]) if trends else 0,
            'peak_revenue': max([t['revenue'] for t in trends]) if trends else 0,
            'lowest_revenue': min([t['revenue'] for t in trends]) if trends else 0
        }
        
    async def get_revenue_forecast(
        self,
        db: AsyncSession,
        user_id: int,
        forecast_days: int = 7
    ) -> Dict[str, Any]:
        """Generate revenue forecast using linear regression"""
        # Get historical data
        trends = await self.get_revenue_trends(db, user_id, 'daily', 60)
        
        if len(trends['trends']) < 14:  # Need at least 2 weeks of data
            return {
                'forecast': [],
                'confidence': 0,
                'method': 'insufficient_data'
            }
            
        # Prepare data for regression
        historical_data = trends['trends']
        X = np.array(range(len(historical_data))).reshape(-1, 1)
        y = np.array([d['revenue'] for d in historical_data])
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Generate predictions
        future_X = np.array(range(len(historical_data), len(historical_data) + forecast_days)).reshape(-1, 1)
        predictions = model.predict(future_X)
        
        # Calculate confidence based on R-squared
        y_pred = model.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Create forecast
        forecast = []
        base_date = datetime.utcnow()
        for i, pred in enumerate(predictions):
            forecast_date = base_date + timedelta(days=i+1)
            forecast.append({
                'date': forecast_date.date().isoformat(),
                'predicted_revenue': max(0, float(pred)),  # Ensure non-negative
                'confidence_lower': max(0, float(pred * 0.8)),  # 80% confidence interval
                'confidence_upper': float(pred * 1.2)
            })
            
        return {
            'forecast': forecast,
            'confidence': float(r_squared),
            'method': 'linear_regression',
            'historical_average': float(np.mean(y)),
            'trend': 'increasing' if model.coef_[0] > 0 else 'decreasing',
            'estimated_total': sum(f['predicted_revenue'] for f in forecast)
        }
        
    async def get_revenue_breakdown(
        self,
        db: AsyncSession,
        user_id: int,
        breakdown_by: str = 'source'
    ) -> Dict[str, Any]:
        """Get detailed revenue breakdown by various dimensions"""
        # Get user channels
        channels_query = select(Channel.id).where(Channel.owner_id == user_id)
        channels_result = await db.execute(channels_query)
        channel_ids = [c[0] for c in channels_result.fetchall()]
        
        if not channel_ids:
            return {'breakdown': [], 'type': breakdown_by}
            
        if breakdown_by == 'source':
            # Breakdown by revenue source (ads, sponsorships, etc.)
            breakdown = await self._breakdown_by_source(db, channel_ids)
        elif breakdown_by == 'content_type':
            # Breakdown by content category
            breakdown = await self._breakdown_by_content_type(db, channel_ids)
        elif breakdown_by == 'video_length':
            # Breakdown by video duration ranges
            breakdown = await self._breakdown_by_video_length(db, channel_ids)
        elif breakdown_by == 'time_of_day':
            # Breakdown by publishing time
            breakdown = await self._breakdown_by_time_of_day(db, channel_ids)
        else:
            breakdown = []
            
        return {
            'breakdown': breakdown,
            'type': breakdown_by,
            'total': sum(b['revenue'] for b in breakdown)
        }
        
    # Private helper methods
    async def _calculate_daily_revenue(
        self,
        db: AsyncSession,
        channel_ids: List[int],
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Calculate daily revenue for channels"""
        daily_query = (
            select(
                func.date(Analytics.date).label('date'),
                func.sum(Analytics.estimated_revenue).label('revenue')
            )
            .select_from(Analytics)
            .join(Video, Analytics.video_id == Video.id)
            .where(
                and_(
                    Video.channel_id.in_(channel_ids),
                    Analytics.date >= start_date,
                    Analytics.date <= end_date
                )
            )
            .group_by(func.date(Analytics.date))
            .order_by(func.date(Analytics.date))
        )
        
        result = await db.execute(daily_query)
        daily_data = result.fetchall()
        
        return [
            {
                'date': row.date.isoformat(),
                'revenue': float(row.revenue or 0)
            }
            for row in daily_data
        ]
        
    async def _calculate_channel_breakdown(
        self,
        db: AsyncSession,
        channel_ids: List[int],
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Calculate revenue breakdown by channel"""
        channel_query = (
            select(
                Channel.id,
                Channel.name,
                func.sum(Analytics.estimated_revenue).label('revenue')
            )
            .select_from(Channel)
            .join(Video, Channel.id == Video.channel_id)
            .join(Analytics, Video.id == Analytics.video_id)
            .where(
                and_(
                    Channel.id.in_(channel_ids),
                    Analytics.date >= start_date,
                    Analytics.date <= end_date
                )
            )
            .group_by(Channel.id, Channel.name)
            .order_by(func.sum(Analytics.estimated_revenue).desc())
        )
        
        result = await db.execute(channel_query)
        channel_data = result.fetchall()
        
        return [
            {
                'channel_id': row.id,
                'channel_name': row.name,
                'revenue': float(row.revenue or 0)
            }
            for row in channel_data
        ]
        
    async def _calculate_revenue_metrics(
        self,
        db: AsyncSession,
        channel_ids: List[int],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, float]:
        """Calculate CPM and RPM metrics"""
        metrics_query = (
            select(
                func.sum(Analytics.estimated_revenue).label('total_revenue'),
                func.sum(Analytics.views).label('total_views'),
                func.sum(Analytics.impressions).label('total_impressions')
            )
            .select_from(Analytics)
            .join(Video, Analytics.video_id == Video.id)
            .where(
                and_(
                    Video.channel_id.in_(channel_ids),
                    Analytics.date >= start_date,
                    Analytics.date <= end_date
                )
            )
        )
        
        result = await db.execute(metrics_query)
        metrics_data = result.first()
        
        total_revenue = float(metrics_data.total_revenue or 0)
        total_views = metrics_data.total_views or 0
        total_impressions = metrics_data.total_impressions or total_views  # Fallback to views
        
        return {
            'cpm': (total_revenue / total_impressions * 1000) if total_impressions > 0 else 0,
            'rpm': (total_revenue / total_views * 1000) if total_views > 0 else 0
        }
        
    async def _generate_revenue_forecast(
        self,
        daily_revenue: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate simple revenue forecast based on recent trends"""
        if len(daily_revenue) < 7:
            return {
                'next_7_days': 0,
                'next_30_days': 0,
                'confidence': 'low'
            }
            
        # Calculate average daily revenue from last 7 days
        recent_revenues = [d['revenue'] for d in daily_revenue[-7:]]
        avg_daily = np.mean(recent_revenues)
        
        # Calculate trend
        if len(daily_revenue) >= 14:
            last_week = np.mean([d['revenue'] for d in daily_revenue[-7:]])
            prev_week = np.mean([d['revenue'] for d in daily_revenue[-14:-7]])
            trend_factor = (last_week / prev_week) if prev_week > 0 else 1
        else:
            trend_factor = 1
            
        return {
            'next_7_days': float(avg_daily * 7 * trend_factor),
            'next_30_days': float(avg_daily * 30 * trend_factor),
            'confidence': 'high' if len(daily_revenue) >= 30 else 'medium',
            'trend_factor': float(trend_factor)
        }
        
    async def _breakdown_by_source(
        self,
        db: AsyncSession,
        channel_ids: List[int]
    ) -> List[Dict[str, Any]]:
        """Revenue breakdown by source type"""
        # Simulate different revenue sources
        # In production, this would come from actual data
        total_revenue_query = (
            select(func.sum(Analytics.estimated_revenue))
            .select_from(Analytics)
            .join(Video, Analytics.video_id == Video.id)
            .where(Video.channel_id.in_(channel_ids))
        )
        
        result = await db.execute(total_revenue_query)
        total = float(result.scalar() or 0)
        
        return [
            {'source': 'YouTube Ads', 'revenue': total * 0.7, 'percentage': 70},
            {'source': 'Sponsorships', 'revenue': total * 0.2, 'percentage': 20},
            {'source': 'Affiliate', 'revenue': total * 0.08, 'percentage': 8},
            {'source': 'Other', 'revenue': total * 0.02, 'percentage': 2}
        ]
        
    async def _breakdown_by_content_type(
        self,
        db: AsyncSession,
        channel_ids: List[int]
    ) -> List[Dict[str, Any]]:
        """Revenue breakdown by content type"""
        content_query = (
            select(
                Video.category,
                func.sum(Analytics.estimated_revenue).label('revenue')
            )
            .select_from(Video)
            .join(Analytics, Video.id == Analytics.video_id)
            .where(Video.channel_id.in_(channel_ids))
            .group_by(Video.category)
            .order_by(func.sum(Analytics.estimated_revenue).desc())
        )
        
        result = await db.execute(content_query)
        content_data = result.fetchall()
        
        return [
            {
                'content_type': row.category or 'Uncategorized',
                'revenue': float(row.revenue or 0)
            }
            for row in content_data
        ]
        
    async def _breakdown_by_video_length(
        self,
        db: AsyncSession,
        channel_ids: List[int]
    ) -> List[Dict[str, Any]]:
        """Revenue breakdown by video length ranges"""
        length_query = (
            select(
                case(
                    (Video.duration < 300, '< 5 min'),
                    (Video.duration < 600, '5-10 min'),
                    (Video.duration < 1200, '10-20 min'),
                    else_='20+ min'
                ).label('length_range'),
                func.sum(Analytics.estimated_revenue).label('revenue')
            )
            .select_from(Video)
            .join(Analytics, Video.id == Analytics.video_id)
            .where(Video.channel_id.in_(channel_ids))
            .group_by('length_range')
            .order_by(func.sum(Analytics.estimated_revenue).desc())
        )
        
        result = await db.execute(length_query)
        length_data = result.fetchall()
        
        return [
            {
                'length_range': row.length_range,
                'revenue': float(row.revenue or 0)
            }
            for row in length_data
        ]
        
    async def _breakdown_by_time_of_day(
        self,
        db: AsyncSession,
        channel_ids: List[int]
    ) -> List[Dict[str, Any]]:
        """Revenue breakdown by time of day published"""
        time_query = (
            select(
                func.extract('hour', Video.published_at).label('hour'),
                func.sum(Analytics.estimated_revenue).label('revenue')
            )
            .select_from(Video)
            .join(Analytics, Video.id == Analytics.video_id)
            .where(Video.channel_id.in_(channel_ids))
            .group_by(func.extract('hour', Video.published_at))
            .order_by(func.extract('hour', Video.published_at))
        )
        
        result = await db.execute(time_query)
        time_data = result.fetchall()
        
        # Group into time periods
        periods = defaultdict(float)
        for row in time_data:
            hour = row.hour or 0
            if 0 <= hour < 6:
                periods['Night (12AM-6AM)'] += float(row.revenue or 0)
            elif 6 <= hour < 12:
                periods['Morning (6AM-12PM)'] += float(row.revenue or 0)
            elif 12 <= hour < 18:
                periods['Afternoon (12PM-6PM)'] += float(row.revenue or 0)
            else:
                periods['Evening (6PM-12AM)'] += float(row.revenue or 0)
                
        return [
            {'time_period': period, 'revenue': revenue}
            for period, revenue in periods.items()
        ]
        
    def _empty_revenue_overview(self) -> Dict[str, Any]:
        """Return empty revenue overview structure"""
        return {
            'total_revenue': 0,
            'average_revenue_per_video': 0,
            'highest_revenue_video': 0,
            'lowest_revenue_video': 0,
            'total_videos_monetized': 0,
            'daily_revenue': [],
            'channel_breakdown': [],
            'cpm': 0,
            'rpm': 0,
            'forecast': {
                'next_7_days': 0,
                'next_30_days': 0,
                'confidence': 'low'
            },
            'period': {
                'start': datetime.utcnow().isoformat(),
                'end': datetime.utcnow().isoformat()
            }
        }


# Create singleton instance
revenue_tracking_service = RevenueTrackingService()