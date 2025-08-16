"""
ROI Calculator Service
Advanced Return on Investment calculations for YTEmpire
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from app.models.video import Video
from app.models.channel import Channel
from app.models.cost import Cost
from app.core.cache import cache_service

logger = logging.getLogger(__name__)


class ROIMetric(str, Enum):
    """ROI metric types"""

    BASIC_ROI = "basic_roi"  # (Revenue - Cost) / Cost
    ANNUALIZED_ROI = "annualized_roi"  # ROI adjusted for time period
    MARGINAL_ROI = "marginal_roi"  # Incremental ROI
    CUMULATIVE_ROI = "cumulative_roi"  # Total ROI over time
    PREDICTED_ROI = "predicted_roi"  # ML-based future ROI
    RISK_ADJUSTED_ROI = "risk_adjusted_roi"  # ROI with risk factor


@dataclass
class ROICalculation:
    """ROI calculation result"""

    metric_type: ROIMetric
    value: float
    percentage: float
    revenue: float
    cost: float
    profit: float
    period_days: int
    confidence: float = 0.95
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChannelROI:
    """Channel-specific ROI metrics"""

    channel_id: str
    channel_name: str
    total_revenue: float
    total_cost: float
    total_profit: float
    roi_percentage: float
    avg_video_roi: float
    best_video_roi: float
    worst_video_roi: float
    videos_count: int
    profitable_videos: int
    break_even_days: Optional[int]
    payback_period: Optional[float]
    lifetime_value: float


@dataclass
class VideoROI:
    """Video-specific ROI metrics"""

    video_id: str
    title: str
    revenue: float
    cost: float
    profit: float
    roi_percentage: float
    views: int
    revenue_per_view: float
    cost_per_view: float
    break_even_views: int
    days_to_profit: Optional[int]
    performance_score: float


class ROICalculatorService:
    """Service for calculating comprehensive ROI metrics"""

    def __init__(self):
        self.cache_ttl = 3600  # 1 hour cache
        self.min_data_points = 7  # Minimum days for trend analysis

    async def calculate_overall_roi(
        self,
        db: AsyncSession,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate overall platform ROI"""
        cache_key = f"roi:overall:{user_id}:{start_date}:{end_date}"
        cached = await cache_service.get(cache_key)
        if cached:
            return cached

        # Default to last 30 days
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)

        # Get revenue and cost data
        revenue_data = await self._get_revenue_data(db, start_date, end_date, user_id)
        cost_data = await self._get_cost_data(db, start_date, end_date, user_id)

        # Calculate basic metrics
        total_revenue = sum(r["amount"] for r in revenue_data)
        total_cost = sum(c["amount"] for c in cost_data)
        total_profit = total_revenue - total_cost

        # Calculate different ROI metrics
        roi_metrics = {
            "basic_roi": self._calculate_basic_roi(total_revenue, total_cost),
            "annualized_roi": self._calculate_annualized_roi(
                total_revenue, total_cost, (end_date - start_date).days
            ),
            "marginal_roi": await self._calculate_marginal_roi(
                db, revenue_data, cost_data
            ),
            "cumulative_roi": await self._calculate_cumulative_roi(
                revenue_data, cost_data
            ),
            "predicted_roi": await self._calculate_predicted_roi(
                db, revenue_data, cost_data
            ),
            "risk_adjusted_roi": self._calculate_risk_adjusted_roi(
                total_revenue, total_cost, revenue_data
            ),
        }

        # Calculate additional insights
        result = {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": (end_date - start_date).days,
            },
            "summary": {
                "total_revenue": round(total_revenue, 2),
                "total_cost": round(total_cost, 2),
                "total_profit": round(total_profit, 2),
                "profit_margin": round(
                    (total_profit / total_revenue * 100) if total_revenue > 0 else 0, 2
                ),
                "cost_revenue_ratio": round(
                    (total_cost / total_revenue) if total_revenue > 0 else 0, 3
                ),
            },
            "roi_metrics": roi_metrics,
            "insights": await self._generate_roi_insights(
                roi_metrics, revenue_data, cost_data
            ),
            "recommendations": self._generate_recommendations(roi_metrics),
        }

        await cache_service.set(cache_key, result, self.cache_ttl)
        return result

    async def calculate_channel_roi(
        self, db: AsyncSession, channel_id: str, period_days: int = 30
    ) -> ChannelROI:
        """Calculate ROI for a specific channel"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)

        # Get channel data
        channel_query = select(Channel).where(Channel.id == channel_id)
        channel = (await db.execute(channel_query)).scalar_one_or_none()

        if not channel:
            raise ValueError(f"Channel {channel_id} not found")

        # Get videos for channel
        videos_query = select(Video).where(
            and_(
                Video.channel_id == channel_id,
                Video.created_at >= start_date,
                Video.created_at <= end_date,
            )
        )
        videos = (await db.execute(videos_query)).scalars().all()

        # Calculate video ROIs
        video_rois = []
        for video in videos:
            video_roi = await self.calculate_video_roi(db, video.id)
            video_rois.append(video_roi)

        # Aggregate metrics
        total_revenue = sum(v.revenue for v in video_rois)
        total_cost = sum(v.cost for v in video_rois)
        total_profit = total_revenue - total_cost

        profitable_videos = sum(1 for v in video_rois if v.profit > 0)

        # Calculate channel ROI
        channel_roi = ChannelROI(
            channel_id=channel_id,
            channel_name=channel.name,
            total_revenue=round(total_revenue, 2),
            total_cost=round(total_cost, 2),
            total_profit=round(total_profit, 2),
            roi_percentage=self._calculate_basic_roi(total_revenue, total_cost),
            avg_video_roi=np.mean([v.roi_percentage for v in video_rois])
            if video_rois
            else 0,
            best_video_roi=max([v.roi_percentage for v in video_rois])
            if video_rois
            else 0,
            worst_video_roi=min([v.roi_percentage for v in video_rois])
            if video_rois
            else 0,
            videos_count=len(videos),
            profitable_videos=profitable_videos,
            break_even_days=self._calculate_break_even_days(video_rois),
            payback_period=self._calculate_payback_period(
                total_cost, total_revenue, period_days
            ),
            lifetime_value=await self._estimate_lifetime_value(
                db, channel_id, total_revenue, period_days
            ),
        )

        return channel_roi

    async def calculate_video_roi(self, db: AsyncSession, video_id: str) -> VideoROI:
        """Calculate ROI for a specific video"""
        # Get video data
        video_query = select(Video).where(Video.id == video_id)
        video = (await db.execute(video_query)).scalar_one_or_none()

        if not video:
            raise ValueError(f"Video {video_id} not found")

        # Get cost data
        cost_query = select(func.sum(Cost.amount)).where(Cost.video_id == video_id)
        total_cost = (await db.execute(cost_query)).scalar() or 0

        # Calculate revenue (from views and engagement)
        revenue = self._estimate_video_revenue(
            video.view_count, video.like_count, video.comment_count
        )

        profit = revenue - total_cost
        roi_percentage = self._calculate_basic_roi(revenue, total_cost)

        # Calculate per-view metrics
        views = video.view_count or 1  # Avoid division by zero
        revenue_per_view = revenue / views
        cost_per_view = total_cost / views

        # Calculate break-even point
        if cost_per_view > 0:
            break_even_views = (
                int(total_cost / revenue_per_view) if revenue_per_view > 0 else 0
            )
        else:
            break_even_views = 0

        # Calculate days to profitability
        if video.created_at and profit > 0:
            days_to_profit = (datetime.utcnow() - video.created_at).days
        else:
            days_to_profit = None

        # Calculate performance score (0-100)
        performance_score = self._calculate_performance_score(
            roi_percentage, views, video.engagement_rate
        )

        return VideoROI(
            video_id=video_id,
            title=video.title,
            revenue=round(revenue, 2),
            cost=round(total_cost, 2),
            profit=round(profit, 2),
            roi_percentage=round(roi_percentage, 2),
            views=views,
            revenue_per_view=round(revenue_per_view, 4),
            cost_per_view=round(cost_per_view, 4),
            break_even_views=break_even_views,
            days_to_profit=days_to_profit,
            performance_score=round(performance_score, 2),
        )

    def _calculate_basic_roi(self, revenue: float, cost: float) -> float:
        """Calculate basic ROI percentage"""
        if cost == 0:
            return 0 if revenue == 0 else 100
        return round(((revenue - cost) / cost) * 100, 2)

    def _calculate_annualized_roi(
        self, revenue: float, cost: float, period_days: int
    ) -> float:
        """Calculate annualized ROI"""
        if period_days == 0 or cost == 0:
            return 0

        roi = (revenue - cost) / cost
        annualized = (1 + roi) ** (365 / period_days) - 1
        return round(annualized * 100, 2)

    async def _calculate_marginal_roi(
        self, db: AsyncSession, revenue_data: List[Dict], cost_data: List[Dict]
    ) -> float:
        """Calculate marginal ROI (incremental ROI for last unit)"""
        if len(revenue_data) < 2 or len(cost_data) < 2:
            return 0

        # Get last period's incremental values
        last_revenue = revenue_data[-1]["amount"] if revenue_data else 0
        last_cost = cost_data[-1]["amount"] if cost_data else 0

        return self._calculate_basic_roi(last_revenue, last_cost)

    async def _calculate_cumulative_roi(
        self, revenue_data: List[Dict], cost_data: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Calculate cumulative ROI over time"""
        cumulative_roi = []
        cumulative_revenue = 0
        cumulative_cost = 0

        # Combine and sort by date
        all_data = []
        for r in revenue_data:
            all_data.append({"date": r["date"], "revenue": r["amount"], "cost": 0})
        for c in cost_data:
            existing = next((d for d in all_data if d["date"] == c["date"]), None)
            if existing:
                existing["cost"] = c["amount"]
            else:
                all_data.append({"date": c["date"], "revenue": 0, "cost": c["amount"]})

        all_data.sort(key=lambda x: x["date"])

        for data in all_data:
            cumulative_revenue += data["revenue"]
            cumulative_cost += data["cost"]

            roi = self._calculate_basic_roi(cumulative_revenue, cumulative_cost)
            cumulative_roi.append(
                {
                    "date": data["date"],
                    "cumulative_revenue": cumulative_revenue,
                    "cumulative_cost": cumulative_cost,
                    "cumulative_roi": roi,
                }
            )

        return cumulative_roi

    async def _calculate_predicted_roi(
        self, db: AsyncSession, revenue_data: List[Dict], cost_data: List[Dict]
    ) -> float:
        """Predict future ROI using trend analysis"""
        if len(revenue_data) < self.min_data_points:
            return 0

        # Simple linear trend projection
        revenues = [r["amount"] for r in revenue_data]
        costs = [c["amount"] for c in cost_data]

        # Calculate trend
        revenue_trend = np.polyfit(range(len(revenues)), revenues, 1)[0]
        cost_trend = np.polyfit(range(len(costs)), costs, 1)[0]

        # Project 30 days forward
        future_revenue = revenues[-1] + (revenue_trend * 30)
        future_cost = costs[-1] + (cost_trend * 30)

        return self._calculate_basic_roi(future_revenue, future_cost)

    def _calculate_risk_adjusted_roi(
        self, revenue: float, cost: float, revenue_data: List[Dict]
    ) -> float:
        """Calculate risk-adjusted ROI using revenue volatility"""
        if not revenue_data or cost == 0:
            return 0

        # Calculate revenue volatility (standard deviation)
        revenues = [r["amount"] for r in revenue_data]
        volatility = np.std(revenues) if len(revenues) > 1 else 0

        # Risk adjustment factor (higher volatility = lower adjusted ROI)
        risk_factor = 1 - min(volatility / np.mean(revenues), 0.5) if revenues else 1

        basic_roi = self._calculate_basic_roi(revenue, cost)
        return round(basic_roi * risk_factor, 2)

    def _estimate_video_revenue(self, views: int, likes: int, comments: int) -> float:
        """Estimate video revenue based on engagement metrics"""
        # YouTube CPM estimates ($ per 1000 views)
        base_cpm = 2.0  # Conservative estimate

        # Engagement bonus
        engagement_rate = (likes + comments) / max(views, 1)
        engagement_multiplier = 1 + min(
            engagement_rate * 10, 2
        )  # Up to 3x for high engagement

        # Calculate revenue
        revenue = (views / 1000) * base_cpm * engagement_multiplier

        return revenue

    def _calculate_performance_score(
        self, roi: float, views: int, engagement_rate: float
    ) -> float:
        """Calculate overall performance score (0-100)"""
        # Weighted scoring
        roi_score = min(max(roi / 100, 0), 1) * 40  # 40% weight
        view_score = min(views / 100000, 1) * 30  # 30% weight
        engagement_score = min(engagement_rate / 0.1, 1) * 30  # 30% weight

        return roi_score + view_score + engagement_score

    def _calculate_break_even_days(self, video_rois: List[VideoROI]) -> Optional[int]:
        """Calculate average days to break even"""
        profitable_days = [v.days_to_profit for v in video_rois if v.days_to_profit]
        return int(np.mean(profitable_days)) if profitable_days else None

    def _calculate_payback_period(
        self, cost: float, revenue: float, period_days: int
    ) -> Optional[float]:
        """Calculate payback period in days"""
        if revenue <= 0 or period_days <= 0:
            return None

        daily_revenue = revenue / period_days
        if daily_revenue <= 0:
            return None

        return round(cost / daily_revenue, 1)

    async def _estimate_lifetime_value(
        self,
        db: AsyncSession,
        channel_id: str,
        current_revenue: float,
        period_days: int,
    ) -> float:
        """Estimate channel lifetime value"""
        # Simple projection: assume linear growth for 1 year
        if period_days <= 0:
            return 0

        daily_revenue = current_revenue / period_days
        yearly_projection = daily_revenue * 365

        # Apply decay factor (assume 20% yearly decay)
        lifetime_value = yearly_projection * 2.5  # ~3 year projection with decay

        return round(lifetime_value, 2)

    async def _generate_roi_insights(
        self, roi_metrics: Dict, revenue_data: List[Dict], cost_data: List[Dict]
    ) -> Dict[str, Any]:
        """Generate actionable ROI insights"""
        insights = {"performance": [], "opportunities": [], "risks": []}

        # Performance insights
        if roi_metrics["basic_roi"] > 100:
            insights["performance"].append(
                {
                    "type": "success",
                    "message": f"Excellent ROI of {roi_metrics['basic_roi']}% - well above target",
                    "impact": "high",
                }
            )
        elif roi_metrics["basic_roi"] > 50:
            insights["performance"].append(
                {
                    "type": "good",
                    "message": f"Good ROI of {roi_metrics['basic_roi']}% - meeting expectations",
                    "impact": "medium",
                }
            )
        else:
            insights["performance"].append(
                {
                    "type": "warning",
                    "message": f"Low ROI of {roi_metrics['basic_roi']}% - needs improvement",
                    "impact": "high",
                }
            )

        # Opportunity insights
        if roi_metrics["predicted_roi"] > roi_metrics["basic_roi"] * 1.2:
            insights["opportunities"].append(
                {
                    "type": "growth",
                    "message": "Positive ROI trend detected - consider increasing investment",
                    "potential_gain": roi_metrics["predicted_roi"]
                    - roi_metrics["basic_roi"],
                }
            )

        # Risk insights
        if roi_metrics["risk_adjusted_roi"] < roi_metrics["basic_roi"] * 0.7:
            insights["risks"].append(
                {
                    "type": "volatility",
                    "message": "High revenue volatility detected - consider diversification",
                    "risk_level": "high",
                }
            )

        return insights

    def _generate_recommendations(self, roi_metrics: Dict) -> List[Dict[str, str]]:
        """Generate actionable recommendations based on ROI"""
        recommendations = []

        if roi_metrics["basic_roi"] < 50:
            recommendations.append(
                {
                    "priority": "high",
                    "action": "Reduce operational costs",
                    "expected_impact": "Improve ROI by 20-30%",
                }
            )

        if roi_metrics["marginal_roi"] < roi_metrics["basic_roi"]:
            recommendations.append(
                {
                    "priority": "medium",
                    "action": "Focus on high-performing content types",
                    "expected_impact": "Increase marginal returns",
                }
            )

        if roi_metrics["predicted_roi"] > roi_metrics["basic_roi"]:
            recommendations.append(
                {
                    "priority": "medium",
                    "action": "Scale successful strategies",
                    "expected_impact": f"Achieve {roi_metrics['predicted_roi']}% ROI",
                }
            )

        return recommendations

    async def _get_revenue_data(
        self,
        db: AsyncSession,
        start_date: datetime,
        end_date: datetime,
        user_id: Optional[str],
    ) -> List[Dict]:
        """Get revenue data for period"""
        # Implementation would query actual revenue tables
        # For now, return mock data
        return [
            {"date": start_date + timedelta(days=i), "amount": 100 + i * 10}
            for i in range((end_date - start_date).days)
        ]

    async def _get_cost_data(
        self,
        db: AsyncSession,
        start_date: datetime,
        end_date: datetime,
        user_id: Optional[str],
    ) -> List[Dict]:
        """Get cost data for period"""
        # Implementation would query actual cost tables
        # For now, return mock data
        return [
            {"date": start_date + timedelta(days=i), "amount": 30 + i * 2}
            for i in range((end_date - start_date).days)
        ]


# Singleton instance
roi_calculator = ROICalculatorService()
