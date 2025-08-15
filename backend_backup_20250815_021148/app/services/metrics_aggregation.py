"""
Metrics Aggregation Procedures for YTEmpire
"""
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np

from app.models.video import Video
from app.models.channel import Channel
from app.models.cost import Cost
from app.models.analytics import Analytics
from app.models.user import User


class MetricsAggregator:
    """Aggregate and compute business metrics"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        
    async def aggregate_channel_performance(self, 
                                           channel_id: str,
                                           period_days: int = 30) -> Dict[str, Any]:
        """Aggregate channel performance metrics"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)
        
        # Get videos in period
        videos_query = select(Video).filter(
            and_(
                Video.channel_id == channel_id,
                Video.created_at >= start_date,
                Video.created_at <= end_date
            )
        )
        result = await self.db.execute(videos_query)
        videos = result.scalars().all()
        
        if not videos:
            return self._empty_channel_metrics()
            
        # Calculate metrics
        total_views = sum(v.view_count for v in videos)
        total_revenue = sum(v.estimated_revenue for v in videos)
        total_cost = sum(v.total_cost for v in videos)
        
        metrics = {
            'channel_id': channel_id,
            'period_days': period_days,
            'total_videos': len(videos),
            'total_views': total_views,
            'total_revenue': total_revenue,
            'total_cost': total_cost,
            'profit': total_revenue - total_cost,
            'roi': ((total_revenue - total_cost) / total_cost * 100) if total_cost > 0 else 0,
            'average_views_per_video': total_views / len(videos) if videos else 0,
            'average_revenue_per_video': total_revenue / len(videos) if videos else 0,
            'average_cost_per_video': total_cost / len(videos) if videos else 0,
            'best_performing_video': self._get_best_video(videos),
            'worst_performing_video': self._get_worst_video(videos),
            'growth_rate': await self._calculate_growth_rate(channel_id, period_days),
            'engagement_metrics': self._calculate_engagement_metrics(videos),
            'content_performance': self._analyze_content_performance(videos),
        }
        
        return metrics
        
    async def aggregate_user_metrics(self, 
                                    user_id: str,
                                    period_days: int = 30) -> Dict[str, Any]:
        """Aggregate user-level metrics across all channels"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)
        
        # Get user's channels
        channels_query = select(Channel).filter(Channel.user_id == user_id)
        result = await self.db.execute(channels_query)
        channels = result.scalars().all()
        
        if not channels:
            return self._empty_user_metrics()
            
        # Aggregate metrics across channels
        total_metrics = {
            'user_id': user_id,
            'period_days': period_days,
            'total_channels': len(channels),
            'active_channels': sum(1 for c in channels if c.is_active),
            'total_videos': 0,
            'total_views': 0,
            'total_revenue': 0.0,
            'total_cost': 0.0,
            'channels_performance': []
        }
        
        for channel in channels:
            channel_metrics = await self.aggregate_channel_performance(
                channel.id, period_days
            )
            total_metrics['total_videos'] += channel_metrics['total_videos']
            total_metrics['total_views'] += channel_metrics['total_views']
            total_metrics['total_revenue'] += channel_metrics['total_revenue']
            total_metrics['total_cost'] += channel_metrics['total_cost']
            
            total_metrics['channels_performance'].append({
                'channel_id': channel.id,
                'channel_name': channel.channel_name,
                'videos': channel_metrics['total_videos'],
                'views': channel_metrics['total_views'],
                'revenue': channel_metrics['total_revenue'],
                'roi': channel_metrics['roi']
            })
            
        # Calculate aggregated metrics
        total_metrics['profit'] = total_metrics['total_revenue'] - total_metrics['total_cost']
        total_metrics['roi'] = (
            (total_metrics['profit'] / total_metrics['total_cost'] * 100)
            if total_metrics['total_cost'] > 0 else 0
        )
        total_metrics['average_revenue_per_channel'] = (
            total_metrics['total_revenue'] / len(channels)
        )
        
        return total_metrics
        
    async def aggregate_cost_metrics(self,
                                    user_id: Optional[str] = None,
                                    period_days: int = 30) -> Dict[str, Any]:
        """Aggregate cost metrics by service and operation"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)
        
        # Build query
        query = select(Cost).filter(
            and_(
                Cost.created_at >= start_date,
                Cost.created_at <= end_date
            )
        )
        
        if user_id:
            query = query.filter(Cost.user_id == user_id)
            
        result = await self.db.execute(query)
        costs = result.scalars().all()
        
        if not costs:
            return self._empty_cost_metrics()
            
        # Aggregate by service
        service_costs = {}
        operation_costs = {}
        daily_costs = {}
        
        for cost in costs:
            # By service
            if cost.service_type not in service_costs:
                service_costs[cost.service_type] = 0.0
            service_costs[cost.service_type] += cost.amount
            
            # By operation
            if cost.operation not in operation_costs:
                operation_costs[cost.operation] = 0.0
            operation_costs[cost.operation] += cost.amount
            
            # By day
            day_key = cost.created_at.date().isoformat()
            if day_key not in daily_costs:
                daily_costs[day_key] = 0.0
            daily_costs[day_key] += cost.amount
            
        total_cost = sum(service_costs.values())
        
        return {
            'period_days': period_days,
            'total_cost': total_cost,
            'average_daily_cost': total_cost / period_days,
            'service_breakdown': service_costs,
            'operation_breakdown': operation_costs,
            'daily_trend': daily_costs,
            'most_expensive_service': max(service_costs, key=service_costs.get),
            'most_expensive_operation': max(operation_costs, key=operation_costs.get),
            'cost_optimization_potential': self._calculate_optimization_potential(costs),
        }
        
    async def aggregate_revenue_metrics(self,
                                       user_id: Optional[str] = None,
                                       period_days: int = 30) -> Dict[str, Any]:
        """Aggregate revenue metrics and projections"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)
        
        # Build query for videos
        query = select(Video).filter(
            and_(
                Video.published_at >= start_date,
                Video.published_at <= end_date,
                Video.is_monetized == True
            )
        )
        
        if user_id:
            # Join with channels to filter by user
            query = query.join(Channel).filter(Channel.user_id == user_id)
            
        result = await self.db.execute(query)
        videos = result.scalars().all()
        
        if not videos:
            return self._empty_revenue_metrics()
            
        # Calculate revenue metrics
        total_revenue = sum(v.estimated_revenue for v in videos)
        total_views = sum(v.view_count for v in videos)
        
        # Group by date for trend analysis
        daily_revenue = {}
        for video in videos:
            day_key = video.published_at.date().isoformat()
            if day_key not in daily_revenue:
                daily_revenue[day_key] = 0.0
            daily_revenue[day_key] += video.estimated_revenue
            
        # Calculate RPM (Revenue Per Mille)
        rpm = (total_revenue / total_views * 1000) if total_views > 0 else 0
        
        # Project future revenue
        average_daily = total_revenue / period_days
        projected_monthly = average_daily * 30
        projected_yearly = average_daily * 365
        
        return {
            'period_days': period_days,
            'total_revenue': total_revenue,
            'total_views': total_views,
            'rpm': rpm,
            'average_daily_revenue': average_daily,
            'daily_trend': daily_revenue,
            'best_day': max(daily_revenue, key=daily_revenue.get) if daily_revenue else None,
            'worst_day': min(daily_revenue, key=daily_revenue.get) if daily_revenue else None,
            'projections': {
                'next_30_days': projected_monthly,
                'next_90_days': projected_monthly * 3,
                'next_year': projected_yearly,
            },
            'revenue_sources': await self._analyze_revenue_sources(videos),
        }
        
    async def aggregate_performance_trends(self,
                                          channel_id: str,
                                          period_days: int = 90) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        # Divide period into weeks
        weeks = period_days // 7
        weekly_metrics = []
        
        for week in range(weeks):
            week_end = datetime.utcnow() - timedelta(days=week * 7)
            week_start = week_end - timedelta(days=7)
            
            # Get videos for this week
            query = select(Video).filter(
                and_(
                    Video.channel_id == channel_id,
                    Video.created_at >= week_start,
                    Video.created_at <= week_end
                )
            )
            result = await self.db.execute(query)
            videos = result.scalars().all()
            
            if videos:
                weekly_metrics.append({
                    'week': week + 1,
                    'videos_count': len(videos),
                    'total_views': sum(v.view_count for v in videos),
                    'average_views': sum(v.view_count for v in videos) / len(videos),
                    'total_revenue': sum(v.estimated_revenue for v in videos),
                })
                
        if not weekly_metrics:
            return {'trend': 'insufficient_data'}
            
        # Calculate trends
        views_trend = self._calculate_trend([w['total_views'] for w in weekly_metrics])
        revenue_trend = self._calculate_trend([w['total_revenue'] for w in weekly_metrics])
        
        return {
            'period_days': period_days,
            'weekly_metrics': weekly_metrics,
            'views_trend': views_trend,
            'revenue_trend': revenue_trend,
            'growth_rate': self._calculate_period_growth(weekly_metrics),
            'volatility': self._calculate_volatility(weekly_metrics),
            'predictions': self._predict_next_period(weekly_metrics),
        }
        
    def _calculate_engagement_metrics(self, videos: List[Video]) -> Dict[str, float]:
        """Calculate engagement metrics from videos"""
        if not videos:
            return {'like_rate': 0, 'comment_rate': 0, 'share_rate': 0}
            
        total_views = sum(v.view_count for v in videos)
        if total_views == 0:
            return {'like_rate': 0, 'comment_rate': 0, 'share_rate': 0}
            
        return {
            'like_rate': sum(v.like_count for v in videos) / total_views * 100,
            'comment_rate': sum(v.comment_count for v in videos) / total_views * 100,
            'share_rate': sum(v.share_count for v in videos) / total_views * 100,
            'average_watch_time': np.mean([v.watch_time_minutes for v in videos]),
        }
        
    def _analyze_content_performance(self, videos: List[Video]) -> Dict[str, Any]:
        """Analyze performance by content characteristics"""
        if not videos:
            return {}
            
        # Group by duration
        short_videos = [v for v in videos if v.duration_seconds < 60]
        medium_videos = [v for v in videos if 60 <= v.duration_seconds < 600]
        long_videos = [v for v in videos if v.duration_seconds >= 600]
        
        return {
            'by_duration': {
                'shorts': {
                    'count': len(short_videos),
                    'avg_views': np.mean([v.view_count for v in short_videos]) if short_videos else 0,
                },
                'medium': {
                    'count': len(medium_videos),
                    'avg_views': np.mean([v.view_count for v in medium_videos]) if medium_videos else 0,
                },
                'long': {
                    'count': len(long_videos),
                    'avg_views': np.mean([v.view_count for v in long_videos]) if long_videos else 0,
                },
            },
            'optimal_duration': self._find_optimal_duration(videos),
            'best_upload_time': self._find_best_upload_time(videos),
        }
        
    def _get_best_video(self, videos: List[Video]) -> Dict[str, Any]:
        """Get best performing video"""
        if not videos:
            return {}
            
        best = max(videos, key=lambda v: v.view_count)
        return {
            'id': best.id,
            'title': best.title,
            'views': best.view_count,
            'revenue': best.estimated_revenue,
        }
        
    def _get_worst_video(self, videos: List[Video]) -> Dict[str, Any]:
        """Get worst performing video"""
        if not videos:
            return {}
            
        worst = min(videos, key=lambda v: v.view_count)
        return {
            'id': worst.id,
            'title': worst.title,
            'views': worst.view_count,
            'revenue': worst.estimated_revenue,
        }
        
    async def _calculate_growth_rate(self, channel_id: str, period_days: int) -> float:
        """Calculate channel growth rate"""
        # Compare with previous period
        current_end = datetime.utcnow()
        current_start = current_end - timedelta(days=period_days)
        previous_start = current_start - timedelta(days=period_days)
        
        # Get current period metrics
        current_query = select(func.sum(Video.view_count)).filter(
            and_(
                Video.channel_id == channel_id,
                Video.created_at >= current_start,
                Video.created_at <= current_end
            )
        )
        current_result = await self.db.execute(current_query)
        current_views = current_result.scalar() or 0
        
        # Get previous period metrics
        previous_query = select(func.sum(Video.view_count)).filter(
            and_(
                Video.channel_id == channel_id,
                Video.created_at >= previous_start,
                Video.created_at < current_start
            )
        )
        previous_result = await self.db.execute(previous_query)
        previous_views = previous_result.scalar() or 0
        
        if previous_views == 0:
            return 100.0 if current_views > 0 else 0.0
            
        return ((current_views - previous_views) / previous_views) * 100
        
    def _calculate_optimization_potential(self, costs: List[Cost]) -> Dict[str, Any]:
        """Calculate cost optimization potential"""
        if not costs:
            return {'potential_savings': 0, 'recommendations': []}
            
        # Analyze cost patterns
        service_costs = {}
        for cost in costs:
            if cost.service_type not in service_costs:
                service_costs[cost.service_type] = []
            service_costs[cost.service_type].append(cost.amount)
            
        recommendations = []
        potential_savings = 0.0
        
        # Check for expensive services
        for service, amounts in service_costs.items():
            avg_cost = np.mean(amounts)
            if service == 'openai' and avg_cost > 0.5:
                recommendations.append(f"Consider using GPT-3.5-turbo instead of GPT-4 for {service}")
                potential_savings += sum(amounts) * 0.3  # 30% potential savings
                
        return {
            'potential_savings': potential_savings,
            'recommendations': recommendations,
        }
        
    async def _analyze_revenue_sources(self, videos: List[Video]) -> Dict[str, float]:
        """Analyze revenue by source"""
        # Simplified - in reality would track different revenue sources
        total_revenue = sum(v.estimated_revenue for v in videos)
        
        return {
            'youtube_ads': total_revenue * 0.7,  # 70% from ads
            'sponsorships': total_revenue * 0.2,  # 20% from sponsorships
            'affiliates': total_revenue * 0.1,    # 10% from affiliates
        }
        
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return 'insufficient_data'
            
        # Simple linear regression
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
            
    def _calculate_period_growth(self, weekly_metrics: List[Dict]) -> float:
        """Calculate growth over period"""
        if len(weekly_metrics) < 2:
            return 0.0
            
        first_week = weekly_metrics[-1]  # Oldest
        last_week = weekly_metrics[0]    # Most recent
        
        if first_week['total_views'] == 0:
            return 100.0 if last_week['total_views'] > 0 else 0.0
            
        return ((last_week['total_views'] - first_week['total_views']) / 
                first_week['total_views'] * 100)
                
    def _calculate_volatility(self, weekly_metrics: List[Dict]) -> float:
        """Calculate volatility of metrics"""
        if len(weekly_metrics) < 2:
            return 0.0
            
        views = [w['total_views'] for w in weekly_metrics]
        return np.std(views) / np.mean(views) if np.mean(views) > 0 else 0
        
    def _predict_next_period(self, weekly_metrics: List[Dict]) -> Dict[str, float]:
        """Simple prediction for next period"""
        if len(weekly_metrics) < 3:
            return {'views': 0, 'revenue': 0}
            
        # Simple moving average
        recent_views = [w['total_views'] for w in weekly_metrics[:3]]
        recent_revenue = [w['total_revenue'] for w in weekly_metrics[:3]]
        
        return {
            'predicted_views': np.mean(recent_views),
            'predicted_revenue': np.mean(recent_revenue),
        }
        
    def _find_optimal_duration(self, videos: List[Video]) -> int:
        """Find optimal video duration based on performance"""
        if not videos:
            return 600  # Default 10 minutes
            
        # Group by duration buckets
        duration_performance = {}
        for v in videos:
            bucket = (v.duration_seconds // 60) * 60  # Round to nearest minute
            if bucket not in duration_performance:
                duration_performance[bucket] = []
            duration_performance[bucket].append(v.view_count)
            
        # Find bucket with highest average views
        best_duration = 600
        best_avg = 0
        
        for duration, views in duration_performance.items():
            avg_views = np.mean(views)
            if avg_views > best_avg:
                best_avg = avg_views
                best_duration = duration
                
        return best_duration
        
    def _find_best_upload_time(self, videos: List[Video]) -> int:
        """Find best upload hour based on performance"""
        if not videos:
            return 14  # Default 2 PM
            
        # Group by hour
        hour_performance = {}
        for v in videos:
            if v.published_at:
                hour = v.published_at.hour
                if hour not in hour_performance:
                    hour_performance[hour] = []
                hour_performance[hour].append(v.view_count)
                
        # Find hour with highest average views
        best_hour = 14
        best_avg = 0
        
        for hour, views in hour_performance.items():
            avg_views = np.mean(views)
            if avg_views > best_avg:
                best_avg = avg_views
                best_hour = hour
                
        return best_hour
        
    def _empty_channel_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure"""
        return {
            'total_videos': 0,
            'total_views': 0,
            'total_revenue': 0.0,
            'total_cost': 0.0,
            'profit': 0.0,
            'roi': 0.0,
        }
        
    def _empty_user_metrics(self) -> Dict[str, Any]:
        """Return empty user metrics"""
        return {
            'total_channels': 0,
            'total_videos': 0,
            'total_views': 0,
            'total_revenue': 0.0,
            'total_cost': 0.0,
        }
        
    def _empty_cost_metrics(self) -> Dict[str, Any]:
        """Return empty cost metrics"""
        return {
            'total_cost': 0.0,
            'service_breakdown': {},
            'operation_breakdown': {},
        }
        
    def _empty_revenue_metrics(self) -> Dict[str, Any]:
        """Return empty revenue metrics"""
        return {
            'total_revenue': 0.0,
            'total_views': 0,
            'rpm': 0.0,
        }