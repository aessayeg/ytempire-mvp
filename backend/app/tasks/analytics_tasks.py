"""
Analytics Tasks for Celery
Handles analytics aggregation, YouTube sync, cost tracking, and reporting
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from celery import Task
from decimal import Decimal

from app.core.celery_app import celery_app
from app.services.analytics_service import AnalyticsService
from app.services.youtube_service import YouTubeService
from app.services.cost_tracking import cost_tracker
from app.services.realtime_analytics_service import RealtimeAnalyticsService
from app.db.session import AsyncSessionLocal
from app.models.video import Video
from app.models.channel import Channel
from app.models.analytics import Analytics
from app.models.cost import Cost
from sqlalchemy import select, func, and_

logger = logging.getLogger(__name__)


class AnalyticsTask(Task):
    """Base class for analytics tasks"""

    _analytics_service = None
    _youtube_service = None
    _realtime_service = None

    @property
    def analytics_service(self):
        if self._analytics_service is None:
            self._analytics_service = AnalyticsService()
        return self._analytics_service

    @property
    def youtube_service(self):
        if self._youtube_service is None:
            self._youtube_service = YouTubeService()
        return self._youtube_service

    @property
    def realtime_service(self):
        if self._realtime_service is None:
            self._realtime_service = RealtimeAnalyticsService()
        return self._realtime_service


@celery_app.task(
    bind=True,
    base=AnalyticsTask,
    name="analytics.sync_youtube",
    queue="analytics",
    max_retries=3,
)
def sync_youtube_analytics(self, channel_id: Optional[str] = None):
    """
    Sync YouTube analytics for channels
    Runs every 30 minutes as configured in beat schedule

    Args:
        channel_id: Specific channel to sync, or None for all channels
    """
    try:
        logger.info(f"Syncing YouTube analytics for channel: {channel_id or 'all'}")

        async def sync():
            async with AsyncSessionLocal() as db:
                # Get channels to sync
                if channel_id:
                    channels = [await db.get(Channel, channel_id)]
                else:
                    # Get all active channels
                    result = await db.execute(
                        select(Channel).where(Channel.is_active == True)
                    )
                    channels = result.scalars().all()

                synced_count = 0
                errors = []

                for channel in channels:
                    if not channel:
                        continue

                    try:
                        # Fetch YouTube analytics
                        analytics_data = (
                            await self.youtube_service.get_channel_analytics(
                                channel.youtube_channel_id, channel.youtube_api_key
                            )
                        )

                        # Update channel statistics
                        channel.total_views = analytics_data.get("total_views", 0)
                        channel.total_subscribers = analytics_data.get("subscribers", 0)
                        channel.total_videos = analytics_data.get("video_count", 0)
                        channel.last_sync = datetime.utcnow()

                        # Fetch video analytics
                        videos = await db.execute(
                            select(Video).where(Video.channel_id == channel.id)
                        )

                        for video in videos.scalars():
                            video_analytics = (
                                await self.youtube_service.get_video_analytics(
                                    video.youtube_video_id, channel.youtube_api_key
                                )
                            )

                            # Create or update analytics record
                            analytics = Analytics(
                                video_id=video.id,
                                channel_id=channel.id,
                                views=video_analytics.get("views", 0),
                                likes=video_analytics.get("likes", 0),
                                comments=video_analytics.get("comments", 0),
                                watch_time_minutes=video_analytics.get("watch_time", 0),
                                ctr=video_analytics.get("ctr", 0.0),
                                revenue=Decimal(
                                    str(video_analytics.get("revenue", 0.0))
                                ),
                                recorded_at=datetime.utcnow(),
                            )
                            db.add(analytics)

                        await db.commit()
                        synced_count += 1

                    except Exception as e:
                        logger.error(f"Failed to sync channel {channel.id}: {str(e)}")
                        errors.append({"channel_id": channel.id, "error": str(e)})

                return synced_count, errors

        synced, errors = asyncio.run(sync())

        return {
            "success": True,
            "channels_synced": synced,
            "errors": errors,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"YouTube analytics sync failed: {str(e)}")
        raise self.retry(exc=e)


@celery_app.task(
    bind=True, base=AnalyticsTask, name="analytics.aggregate_costs", queue="analytics"
)
def aggregate_costs(self, period: str = "hourly"):
    """
    Aggregate costs for reporting
    Runs every 10 minutes as configured in beat schedule

    Args:
        period: Aggregation period (hourly, daily, weekly)
    """
    try:
        logger.info(f"Aggregating costs for period: {period}")

        async def aggregate():
            async with AsyncSessionLocal() as db:
                now = datetime.utcnow()

                if period == "hourly":
                    start_time = now - timedelta(hours=1)
                elif period == "daily":
                    start_time = now - timedelta(days=1)
                else:
                    start_time = now - timedelta(weeks=1)

                # Aggregate costs by service
                result = await db.execute(
                    select(
                        Cost.service,
                        func.sum(Cost.amount).label("total_cost"),
                        func.count(Cost.id).label("operation_count"),
                    )
                    .where(Cost.created_at >= start_time)
                    .group_by(Cost.service)
                )

                aggregated = {}
                for row in result:
                    aggregated[row.service] = {
                        "total_cost": float(row.total_cost or 0),
                        "operations": row.operation_count,
                    }

                # Calculate total
                total_cost = sum(s["total_cost"] for s in aggregated.values())

                # Store aggregated data for reporting
                await self.realtime_service.update_cost_metrics(
                    {
                        "period": period,
                        "start_time": start_time.isoformat(),
                        "end_time": now.isoformat(),
                        "services": aggregated,
                        "total_cost": total_cost,
                    }
                )

                return aggregated, total_cost

        aggregated_data, total = asyncio.run(aggregate())

        return {
            "success": True,
            "period": period,
            "aggregated": aggregated_data,
            "total_cost": total,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Cost aggregation failed: {str(e)}")
        return {"success": False, "error": str(e)}


@celery_app.task(name="analytics.calculate_roi", queue="analytics")
def calculate_channel_roi(channel_id: str, period_days: int = 30) -> Dict[str, Any]:
    """
    Calculate ROI for a channel

    Args:
        channel_id: Channel ID
        period_days: Period to calculate ROI for
    """
    try:
        logger.info(f"Calculating ROI for channel {channel_id}")

        async def calculate():
            async with AsyncSessionLocal() as db:
                start_date = datetime.utcnow() - timedelta(days=period_days)

                # Get revenue
                revenue_result = await db.execute(
                    select(func.sum(Analytics.revenue)).where(
                        and_(
                            Analytics.channel_id == channel_id,
                            Analytics.recorded_at >= start_date,
                        )
                    )
                )
                total_revenue = float(revenue_result.scalar() or 0)

                # Get costs
                cost_result = await db.execute(
                    select(func.sum(Cost.amount)).where(
                        and_(
                            Cost.channel_id == channel_id, Cost.created_at >= start_date
                        )
                    )
                )
                total_cost = float(cost_result.scalar() or 0)

                # Calculate ROI
                if total_cost > 0:
                    roi = ((total_revenue - total_cost) / total_cost) * 100
                else:
                    roi = 0

                return {
                    "revenue": total_revenue,
                    "cost": total_cost,
                    "profit": total_revenue - total_cost,
                    "roi_percentage": roi,
                }

        roi_data = asyncio.run(calculate())

        return {
            "success": True,
            "channel_id": channel_id,
            "period_days": period_days,
            "metrics": roi_data,
            "calculated_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"ROI calculation failed: {str(e)}")
        return {"success": False, "error": str(e)}


@celery_app.task(name="analytics.generate_report", queue="analytics")
def generate_analytics_report(
    user_id: str, report_type: str = "weekly", email: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate analytics report for user

    Args:
        user_id: User ID
        report_type: Type of report (daily, weekly, monthly)
        email: Email to send report to
    """
    try:
        logger.info(f"Generating {report_type} report for user {user_id}")

        # Generate report data
        report_data = asyncio.run(generate_report_data(user_id, report_type))

        # Send email if provided
        if email:
            # TODO: Implement email sending
            logger.info(f"Would send report to {email}")

        return {
            "success": True,
            "report_type": report_type,
            "user_id": user_id,
            "data": report_data,
            "generated_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        return {"success": False, "error": str(e)}


async def generate_report_data(user_id: str, report_type: str) -> Dict[str, Any]:
    """Generate report data for user"""
    async with AsyncSessionLocal() as db:
        # This would fetch and aggregate all relevant data
        # For now, returning sample structure
        return {
            "total_videos": 0,
            "total_views": 0,
            "total_revenue": 0.0,
            "total_cost": 0.0,
            "top_videos": [],
            "channel_performance": [],
        }


# Cleanup task
@celery_app.task(name="analytics.cleanup_old_data", queue="analytics")
def cleanup_old_analytics_data(days_to_keep: int = 90):
    """
    Clean up old analytics data

    Args:
        days_to_keep: Number of days of data to keep
    """
    try:
        logger.info(f"Cleaning up analytics data older than {days_to_keep} days")

        async def cleanup():
            async with AsyncSessionLocal() as db:
                cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

                # Delete old analytics records
                result = await db.execute(
                    select(Analytics).where(Analytics.recorded_at < cutoff_date)
                )
                old_records = result.scalars().all()

                for record in old_records:
                    await db.delete(record)

                await db.commit()
                return len(old_records)

        deleted_count = asyncio.run(cleanup())

        return {
            "success": True,
            "deleted_records": deleted_count,
            "cleanup_date": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Analytics cleanup failed: {str(e)}")
        return {"success": False, "error": str(e)}
