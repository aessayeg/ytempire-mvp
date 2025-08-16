"""
Channel Manager Service for YTEmpire
Handles channel operations, health monitoring, and resource allocation
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
from decimal import Decimal

from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import selectinload
from app.db.session import AsyncSessionLocal
from app.models.channel import Channel, ChannelHealth
from app.models.video import Video, VideoStatus
from app.models.analytics import Analytics
from app.services.youtube_multi_account import get_youtube_manager
from app.services.notification_service import notification_service
from app.core.config import settings

logger = logging.getLogger(__name__)


class ChannelManager:
    """
    Manages YouTube channels with health scoring and resource optimization
    """

    def __init__(self):
        self.youtube_manager = get_youtube_manager()
        self.health_thresholds = {
            "excellent": 90,
            "good": 70,
            "warning": 50,
            "critical": 30,
        }
        self.quota_limits = {
            "daily_videos": 10,
            "daily_uploads": 50,
            "api_quota": 10000,
        }

    async def create_channel(
        self,
        user_id: str,
        youtube_channel_id: str,
        channel_name: str,
        api_key: str,
        refresh_token: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create and register a new channel

        Args:
            user_id: Owner user ID
            youtube_channel_id: YouTube channel ID
            channel_name: Channel display name
            api_key: YouTube API key
            refresh_token: OAuth refresh token
            **kwargs: Additional channel properties

        Returns:
            Created channel information
        """
        try:
            async with AsyncSessionLocal() as db:
                # Check channel limit for user
                existing_count = await db.execute(
                    select(func.count(Channel.id)).where(Channel.user_id == user_id)
                )
                count = existing_count.scalar()

                if count >= 5:  # 5+ channels per user
                    return {
                        "success": False,
                        "error": "Channel limit reached (max 5 per user)",
                    }

                # Create channel
                channel = Channel(
                    user_id=user_id,
                    youtube_channel_id=youtube_channel_id,
                    name=channel_name,
                    youtube_api_key=api_key,
                    youtube_refresh_token=refresh_token,
                    is_active=True,
                    health_score=100,
                    **kwargs,
                )

                db.add(channel)
                await db.commit()
                await db.refresh(channel)

                # Register with multi-account manager
                await self.youtube_manager.register_account(
                    account_id=str(channel.id),
                    api_key=api_key,
                    refresh_token=refresh_token,
                    channel_id=youtube_channel_id,
                )

                # Initialize health record
                health = ChannelHealth(
                    channel_id=channel.id, health_score=100, status="healthy"
                )
                db.add(health)
                await db.commit()

                return {
                    "success": True,
                    "channel_id": str(channel.id),
                    "name": channel.name,
                    "health_score": 100,
                }

        except Exception as e:
            logger.error(f"Channel creation failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get_channel_health(self, channel_id: str) -> Dict[str, Any]:
        """
        Get comprehensive channel health metrics

        Args:
            channel_id: Channel ID

        Returns:
            Health metrics and status
        """
        async with AsyncSessionLocal() as db:
            channel = await db.get(Channel, channel_id)
            if not channel:
                return {"error": "Channel not found"}

            # Calculate health metrics
            metrics = await self._calculate_health_metrics(db, channel)

            # Calculate overall health score
            health_score = self._calculate_health_score(metrics)

            # Update channel health
            channel.health_score = health_score
            channel.last_health_check = datetime.utcnow()

            # Determine status
            if health_score >= self.health_thresholds["excellent"]:
                status = "excellent"
            elif health_score >= self.health_thresholds["good"]:
                status = "good"
            elif health_score >= self.health_thresholds["warning"]:
                status = "warning"
            else:
                status = "critical"

            await db.commit()

            return {
                "channel_id": channel_id,
                "health_score": health_score,
                "status": status,
                "metrics": metrics,
                "recommendations": self._get_health_recommendations(
                    metrics, health_score
                ),
            }

    async def _calculate_health_metrics(self, db, channel) -> Dict[str, Any]:
        """Calculate detailed health metrics for a channel"""
        now = datetime.utcnow()
        day_ago = now - timedelta(days=1)
        week_ago = now - timedelta(weeks=1)

        # Video performance metrics
        recent_videos = await db.execute(
            select(Video).where(
                and_(Video.channel_id == channel.id, Video.created_at >= week_ago)
            )
        )
        videos = recent_videos.scalars().all()

        # Analytics metrics
        analytics_result = await db.execute(
            select(
                func.avg(Analytics.views).label("avg_views"),
                func.avg(Analytics.ctr).label("avg_ctr"),
                func.sum(Analytics.revenue).label("total_revenue"),
            ).where(
                and_(
                    Analytics.channel_id == channel.id,
                    Analytics.recorded_at >= week_ago,
                )
            )
        )
        analytics = analytics_result.first()

        # Quota usage
        daily_videos = await db.execute(
            select(func.count(Video.id)).where(
                and_(Video.channel_id == channel.id, Video.created_at >= day_ago)
            )
        )

        # Error rate
        failed_videos = await db.execute(
            select(func.count(Video.id)).where(
                and_(
                    Video.channel_id == channel.id,
                    Video.status.in_([VideoStatus.FAILED, VideoStatus.ERROR]),
                    Video.created_at >= week_ago,
                )
            )
        )

        metrics = {
            "total_videos": len(videos),
            "daily_videos": daily_videos.scalar() or 0,
            "avg_views": float(analytics.avg_views or 0) if analytics else 0,
            "avg_ctr": float(analytics.avg_ctr or 0) if analytics else 0,
            "weekly_revenue": float(analytics.total_revenue or 0) if analytics else 0,
            "error_rate": (failed_videos.scalar() or 0) / max(len(videos), 1) * 100,
            "quota_usage": (daily_videos.scalar() or 0)
            / self.quota_limits["daily_videos"]
            * 100,
            "last_upload": max([v.created_at for v in videos], default=None),
            "subscriber_count": channel.total_subscribers or 0,
            "total_views": channel.total_views or 0,
        }

        return metrics

    def _calculate_health_score(self, metrics: Dict[str, Any]) -> int:
        """Calculate overall health score from metrics"""
        score = 100

        # Penalize high error rate
        if metrics["error_rate"] > 10:
            score -= min(30, metrics["error_rate"])

        # Penalize quota overuse
        if metrics["quota_usage"] > 80:
            score -= 20
        elif metrics["quota_usage"] > 60:
            score -= 10

        # Penalize low engagement
        if metrics["avg_ctr"] < 2:
            score -= 15

        # Penalize inactivity
        if metrics["last_upload"]:
            days_inactive = (datetime.utcnow() - metrics["last_upload"]).days
            if days_inactive > 7:
                score -= min(20, days_inactive * 2)

        # Bonus for good performance
        if metrics["avg_views"] > 10000:
            score += 10
        if metrics["weekly_revenue"] > 100:
            score += 10

        return max(0, min(100, score))

    def _get_health_recommendations(
        self, metrics: Dict[str, Any], health_score: int
    ) -> List[str]:
        """Generate health improvement recommendations"""
        recommendations = []

        if metrics["error_rate"] > 10:
            recommendations.append(
                "High error rate detected. Review video generation settings."
            )

        if metrics["quota_usage"] > 60:
            recommendations.append(
                "Approaching quota limit. Consider spreading uploads throughout the day."
            )

        if metrics["avg_ctr"] < 2:
            recommendations.append(
                "Low CTR. Improve thumbnails and titles for better engagement."
            )

        if (
            metrics["last_upload"]
            and (datetime.utcnow() - metrics["last_upload"]).days > 3
        ):
            recommendations.append(
                "No recent uploads. Schedule new content to maintain engagement."
            )

        if health_score < 50:
            recommendations.append(
                "Channel health is critical. Immediate attention required."
            )

        return recommendations

    async def allocate_channel_for_video(
        self, user_id: str, video_type: str = "general"
    ) -> Optional[str]:
        """
        Allocate best available channel for video upload

        Args:
            user_id: User ID
            video_type: Type of video for channel matching

        Returns:
            Allocated channel ID or None
        """
        async with AsyncSessionLocal() as db:
            # Get user's active channels sorted by health
            channels = await db.execute(
                select(Channel)
                .where(
                    and_(
                        Channel.user_id == user_id,
                        Channel.is_active == True,
                        Channel.health_score > self.health_thresholds["critical"],
                    )
                )
                .order_by(Channel.health_score.desc())
            )

            for channel in channels.scalars():
                # Check quota availability
                daily_videos = await db.execute(
                    select(func.count(Video.id)).where(
                        and_(
                            Video.channel_id == channel.id,
                            Video.created_at >= datetime.utcnow() - timedelta(days=1),
                        )
                    )
                )

                if daily_videos.scalar() < self.quota_limits["daily_videos"]:
                    # Check YouTube account availability
                    account = await self.youtube_manager.get_best_account_for_upload(
                        channel.youtube_channel_id
                    )

                    if account:
                        return str(channel.id)

            return None

    async def rotate_channels(self, user_id: str) -> Dict[str, Any]:
        """
        Implement channel rotation strategy

        Args:
            user_id: User ID

        Returns:
            Rotation results
        """
        async with AsyncSessionLocal() as db:
            channels = await db.execute(
                select(Channel)
                .where(Channel.user_id == user_id)
                .order_by(Channel.last_used)
            )

            rotation_order = []
            for channel in channels.scalars():
                health = await self.get_channel_health(str(channel.id))
                rotation_order.append(
                    {
                        "channel_id": str(channel.id),
                        "name": channel.name,
                        "health_score": health["health_score"],
                        "priority": self._calculate_rotation_priority(channel, health),
                    }
                )

            # Sort by priority
            rotation_order.sort(key=lambda x: x["priority"], reverse=True)

            return {
                "success": True,
                "rotation_order": rotation_order,
                "active_channel": rotation_order[0]["channel_id"]
                if rotation_order
                else None,
            }

    def _calculate_rotation_priority(self, channel, health) -> float:
        """Calculate channel rotation priority"""
        priority = health["health_score"]

        # Boost priority for channels not used recently
        if channel.last_used:
            days_since_use = (datetime.utcnow() - channel.last_used).days
            priority += min(20, days_since_use * 2)

        # Reduce priority for channels near quota
        if health["metrics"]["quota_usage"] > 50:
            priority -= (health["metrics"]["quota_usage"] - 50) / 2

        return priority

    async def pause_unhealthy_channels(self) -> Dict[str, Any]:
        """Automatically pause channels with critical health"""
        async with AsyncSessionLocal() as db:
            unhealthy = await db.execute(
                select(Channel).where(
                    and_(
                        Channel.is_active == True,
                        Channel.health_score < self.health_thresholds["critical"],
                    )
                )
            )

            paused = []
            for channel in unhealthy.scalars():
                channel.is_active = False
                channel.paused_at = datetime.utcnow()
                paused.append(channel.name)

                # Send notification
                await notification_service.send_notification(
                    user_id=channel.user_id,
                    title="Channel Paused",
                    message=f"Channel '{channel.name}' has been paused due to low health score",
                    type="warning",
                )

            await db.commit()

            return {"success": True, "paused_channels": paused, "count": len(paused)}

    async def optimize_channel_settings(self, channel_id: str) -> Dict[str, Any]:
        """
        Optimize channel settings based on performance

        Args:
            channel_id: Channel ID

        Returns:
            Optimization results
        """
        health = await self.get_channel_health(channel_id)
        metrics = health["metrics"]

        optimizations = {}

        # Optimize upload frequency
        if metrics["error_rate"] > 5:
            optimizations["upload_frequency"] = "reduce"
            optimizations["daily_limit"] = max(3, self.quota_limits["daily_videos"] - 2)
        elif metrics["error_rate"] < 2 and metrics["quota_usage"] < 50:
            optimizations["upload_frequency"] = "increase"
            optimizations["daily_limit"] = min(
                15, self.quota_limits["daily_videos"] + 2
            )

        # Optimize video settings
        if metrics["avg_ctr"] < 3:
            optimizations["thumbnail_style"] = "high_contrast"
            optimizations["title_strategy"] = "curiosity_gap"

        # Apply optimizations
        async with AsyncSessionLocal() as db:
            channel = await db.get(Channel, channel_id)
            if channel:
                channel.settings = {**channel.settings, **optimizations}
                await db.commit()

        return {
            "success": True,
            "channel_id": channel_id,
            "optimizations": optimizations,
            "expected_improvement": self._estimate_improvement(optimizations),
        }

    def _estimate_improvement(self, optimizations: Dict) -> str:
        """Estimate expected improvement from optimizations"""
        if "upload_frequency" in optimizations:
            if optimizations["upload_frequency"] == "reduce":
                return "5-10% reduction in errors expected"
            else:
                return "10-20% increase in revenue potential"
        return "Marginal improvements expected"


# Singleton instance
channel_manager = ChannelManager()
