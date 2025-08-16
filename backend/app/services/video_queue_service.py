"""
Video Queue Service
Manages video generation queue with database persistence and advanced features
"""
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
import uuid
import logging
from collections import defaultdict

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_, func, desc, asc
from sqlalchemy.orm import selectinload

from app.models.video_queue import VideoQueue, QueueWorkerStatus, QueueMetrics
from app.models.user import User
from app.models.channel import Channel
from app.core.celery_app import celery_app
from app.services.websocket_manager import ConnectionManager

logger = logging.getLogger(__name__)


class QueueStatus:
    """Queue status constants"""

    PENDING = "pending"
    SCHEDULED = "scheduled"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class Priority:
    """Priority levels"""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


class VideoQueueService:
    """
    Advanced video queue management service
    """

    def __init__(self, websocket_manager: Optional[ConnectionManager] = None):
        self.websocket_manager = websocket_manager
        self.active_workers: Dict[str, Dict] = {}
        self._metrics_cache = {}

    async def add_to_queue(
        self,
        db: AsyncSession,
        user_id: str,
        channel_id: str,
        title: str,
        topic: str,
        duration_minutes: int = 5,
        style: str = "informative",
        description: Optional[str] = None,
        scheduled_time: Optional[datetime] = None,
        priority: int = Priority.NORMAL,
        tags: List[str] = None,
        queue_metadata: Dict[str, Any] = None,
    ) -> VideoQueue:
        """
        Add video to generation queue
        """
        try:
            # Calculate estimates
            estimated_cost = self._calculate_cost(
                duration_minutes=duration_minutes,
                style=style,
                queue_metadata=queue_metadata or {},
            )

            processing_time = self._estimate_processing_time(
                duration_minutes=duration_minutes, priority=priority
            )

            # Create queue item
            queue_item = VideoQueue(
                user_id=user_id,
                channel_id=channel_id,
                title=title,
                description=description,
                topic=topic,
                style=style,
                duration_minutes=duration_minutes,
                tags=tags or [],
                status=QueueStatus.SCHEDULED if scheduled_time else QueueStatus.PENDING,
                priority=priority,
                scheduled_time=scheduled_time,
                estimated_cost=estimated_cost,
                processing_time_estimate=processing_time,
                queue_metadata=queue_metadata or {},
            )

            db.add(queue_item)
            await db.commit()
            await db.refresh(queue_item)

            # Schedule Celery task
            await self._schedule_celery_task(queue_item)

            # Send WebSocket notification
            if self.websocket_manager:
                await self.websocket_manager.send_to_user(
                    user_id,
                    {
                        "type": "queue_added",
                        "data": {
                            "queue_id": str(queue_item.id),
                            "title": title,
                            "estimated_cost": estimated_cost,
                            "position": await self._get_queue_position(
                                db, queue_item.id
                            ),
                        },
                    },
                )

            logger.info(f"Added video '{title}' to queue for user {user_id}")
            return queue_item

        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to add to queue: {e}")
            raise

    async def get_queue(
        self,
        db: AsyncSession,
        user_id: str,
        status: Optional[str] = None,
        channel_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[VideoQueue]:
        """
        Get user's queue items with filtering
        """
        query = select(VideoQueue).where(VideoQueue.user_id == user_id)

        if status:
            query = query.where(VideoQueue.status == status)
        if channel_id:
            query = query.where(VideoQueue.channel_id == channel_id)

        # Order by priority (desc) and scheduled time (asc)
        query = query.order_by(
            desc(VideoQueue.priority),
            VideoQueue.scheduled_time.nulls_last(),
            VideoQueue.created_at,
        )

        query = query.offset(offset).limit(limit)

        result = await db.execute(query)
        return result.scalars().all()

    async def update_queue_item(
        self, db: AsyncSession, queue_id: uuid.UUID, user_id: str, **updates
    ) -> VideoQueue:
        """
        Update queue item
        """
        # Get existing item
        result = await db.execute(
            select(VideoQueue).where(
                and_(VideoQueue.id == queue_id, VideoQueue.user_id == user_id)
            )
        )
        queue_item = result.scalar_one_or_none()

        if not queue_item:
            raise ValueError("Queue item not found")

        # Update fields
        for field, value in updates.items():
            if hasattr(queue_item, field):
                setattr(queue_item, field, value)

        queue_item.updated_at = datetime.utcnow()

        await db.commit()
        await db.refresh(queue_item)

        # Reschedule if needed
        if "scheduled_time" in updates or "priority" in updates:
            await self._schedule_celery_task(queue_item)

        return queue_item

    async def cancel_queue_item(
        self, db: AsyncSession, queue_id: uuid.UUID, user_id: str
    ) -> bool:
        """
        Cancel queue item
        """
        try:
            result = await db.execute(
                select(VideoQueue).where(
                    and_(VideoQueue.id == queue_id, VideoQueue.user_id == user_id)
                )
            )
            queue_item = result.scalar_one_or_none()

            if not queue_item:
                return False

            # Cancel Celery task if exists
            if queue_item.queue_metadata and "task_id" in queue_item.queue_metadata:
                celery_app.control.revoke(
                    queue_item.queue_metadata["task_id"], terminate=True
                )

            # Update status
            queue_item.status = QueueStatus.CANCELLED
            queue_item.updated_at = datetime.utcnow()

            await db.commit()

            return True

        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to cancel queue item: {e}")
            return False

    async def retry_failed_item(
        self, db: AsyncSession, queue_id: uuid.UUID, user_id: str
    ) -> VideoQueue:
        """
        Retry failed queue item
        """
        result = await db.execute(
            select(VideoQueue).where(
                and_(
                    VideoQueue.id == queue_id,
                    VideoQueue.user_id == user_id,
                    VideoQueue.status == QueueStatus.FAILED,
                )
            )
        )
        queue_item = result.scalar_one_or_none()

        if not queue_item:
            raise ValueError("Failed queue item not found")

        # Update for retry
        queue_item.status = QueueStatus.PENDING
        queue_item.retry_count += 1
        queue_item.error_message = None
        queue_item.updated_at = datetime.utcnow()

        await db.commit()
        await db.refresh(queue_item)

        # Schedule retry task
        await self._schedule_celery_task(queue_item, countdown=60)

        return queue_item

    async def get_queue_statistics(
        self, db: AsyncSession, user_id: str
    ) -> Dict[str, Any]:
        """
        Get comprehensive queue statistics
        """
        # Count by status
        status_counts = await db.execute(
            select(
                VideoQueue.status,
                func.count(VideoQueue.id).label("count"),
                func.sum(VideoQueue.estimated_cost).label("cost"),
            )
            .where(VideoQueue.user_id == user_id)
            .group_by(VideoQueue.status)
        )

        stats = {
            "total_items": 0,
            "by_status": {},
            "estimated_total_cost": 0.0,
            "estimated_completion_time": None,
        }

        total_processing_time = 0
        for row in status_counts:
            stats["by_status"][row.status] = {
                "count": row.count,
                "cost": row.cost or 0.0,
            }
            stats["total_items"] += row.count
            stats["estimated_total_cost"] += row.cost or 0.0

            if row.status in [QueueStatus.PENDING, QueueStatus.SCHEDULED]:
                # Get processing time estimates
                time_result = await db.execute(
                    select(func.sum(VideoQueue.processing_time_estimate)).where(
                        and_(
                            VideoQueue.user_id == user_id,
                            VideoQueue.status == row.status,
                        )
                    )
                )
                processing_time = time_result.scalar() or 0
                total_processing_time += processing_time

        # Calculate completion time
        if total_processing_time > 0:
            stats["estimated_completion_time"] = (
                datetime.utcnow() + timedelta(minutes=total_processing_time)
            ).isoformat()

        # Get processing rate
        stats["processing_rate"] = await self._get_processing_rate(db, user_id)

        return stats

    async def batch_add_to_queue(
        self,
        db: AsyncSession,
        user_id: str,
        channel_id: str,
        video_requests: List[Dict[str, Any]],
        stagger_minutes: int = 60,
    ) -> List[VideoQueue]:
        """
        Add multiple videos to queue with staggered scheduling
        """
        queue_items = []
        base_time = datetime.utcnow()

        try:
            for i, request in enumerate(video_requests):
                # Set staggered schedule time
                scheduled_time = base_time + timedelta(minutes=i * stagger_minutes)
                request["scheduled_time"] = scheduled_time
                request["channel_id"] = channel_id
                request["user_id"] = user_id

                queue_item = await self.add_to_queue(db, **request)
                queue_items.append(queue_item)

            return queue_items

        except Exception as e:
            await db.rollback()
            logger.error(f"Batch queue operation failed: {e}")
            raise

    async def pause_user_queue(self, db: AsyncSession, user_id: str) -> int:
        """
        Pause all pending/scheduled items for user
        """
        result = await db.execute(
            update(VideoQueue)
            .where(
                and_(
                    VideoQueue.user_id == user_id,
                    VideoQueue.status.in_([QueueStatus.PENDING, QueueStatus.SCHEDULED]),
                )
            )
            .values(status=QueueStatus.PAUSED, updated_at=datetime.utcnow())
            .returning(VideoQueue.id)
        )

        paused_ids = result.fetchall()
        await db.commit()

        # Cancel Celery tasks
        for (queue_id,) in paused_ids:
            await self._cancel_celery_task(queue_id)

        return len(paused_ids)

    async def resume_user_queue(self, db: AsyncSession, user_id: str) -> int:
        """
        Resume all paused items for user
        """
        result = await db.execute(
            select(VideoQueue).where(
                and_(
                    VideoQueue.user_id == user_id,
                    VideoQueue.status == QueueStatus.PAUSED,
                )
            )
        )

        paused_items = result.scalars().all()

        resumed_count = 0
        for item in paused_items:
            item.status = QueueStatus.PENDING
            item.updated_at = datetime.utcnow()
            await self._schedule_celery_task(item)
            resumed_count += 1

        await db.commit()
        return resumed_count

    async def update_processing_status(
        self,
        db: AsyncSession,
        queue_id: uuid.UUID,
        status: str,
        progress: Optional[float] = None,
        error_message: Optional[str] = None,
        video_id: Optional[str] = None,
        actual_cost: Optional[float] = None,
    ):
        """
        Update processing status (called by workers)
        """
        update_data = {"status": status, "updated_at": datetime.utcnow()}

        if status == QueueStatus.PROCESSING:
            update_data["started_at"] = datetime.utcnow()
        elif status == QueueStatus.COMPLETED:
            update_data["completed_at"] = datetime.utcnow()
            if video_id:
                update_data["video_id"] = video_id
            if actual_cost:
                update_data["actual_cost"] = actual_cost
        elif status == QueueStatus.FAILED:
            update_data["error_message"] = error_message

        await db.execute(
            update(VideoQueue).where(VideoQueue.id == queue_id).values(**update_data)
        )

        await db.commit()

        # Send WebSocket update
        if self.websocket_manager:
            queue_item = await db.get(VideoQueue, queue_id)
            if queue_item:
                await self.websocket_manager.send_to_user(
                    queue_item.user_id,
                    {
                        "type": "queue_status_update",
                        "data": {
                            "queue_id": str(queue_id),
                            "status": status,
                            "progress": progress,
                            "video_id": video_id,
                        },
                    },
                )

    async def get_next_items(
        self, db: AsyncSession, limit: int = 10, worker_type: str = "video_generator"
    ) -> List[VideoQueue]:
        """
        Get next items for processing (used by workers)
        """
        now = datetime.utcnow()

        query = (
            select(VideoQueue)
            .where(
                or_(
                    and_(
                        VideoQueue.status == QueueStatus.PENDING,
                        VideoQueue.scheduled_time.is_(None),
                    ),
                    and_(
                        VideoQueue.status == QueueStatus.SCHEDULED,
                        VideoQueue.scheduled_time <= now,
                    ),
                )
            )
            .order_by(
                desc(VideoQueue.priority),
                VideoQueue.scheduled_time.nulls_first(),
                VideoQueue.created_at,
            )
            .limit(limit)
        )

        result = await db.execute(query)
        return result.scalars().all()

    # Private helper methods
    async def _schedule_celery_task(
        self, queue_item: VideoQueue, countdown: Optional[int] = None
    ):
        """Schedule Celery task for queue item"""
        try:
            task_kwargs = {
                "args": [str(queue_item.id)],
                "priority": queue_item.priority,
            }

            if countdown:
                task_kwargs["countdown"] = countdown
            elif queue_item.scheduled_time:
                task_kwargs["eta"] = queue_item.scheduled_time
            else:
                # Immediate processing based on priority
                delay = 1 if queue_item.priority >= Priority.HIGH else 30
                task_kwargs["countdown"] = delay

            task = celery_app.send_task(
                "app.tasks.video_generation.process_video_queue", **task_kwargs
            )

            # Store task ID in metadata
            if not queue_item.queue_metadata:
                queue_item.queue_metadata = {}
            queue_item.queue_metadata["task_id"] = task.id

        except Exception as e:
            logger.error(f"Failed to schedule Celery task: {e}")

    async def _cancel_celery_task(self, queue_id: uuid.UUID):
        """Cancel Celery task for queue item"""
        # This would require getting the task ID from the queue item
        # and calling celery_app.control.revoke()
        pass

    async def _get_queue_position(self, db: AsyncSession, queue_id: uuid.UUID) -> int:
        """Get position of item in queue"""
        # Implementation to calculate queue position
        result = await db.execute(
            select(func.count(VideoQueue.id)).where(
                and_(
                    VideoQueue.status.in_([QueueStatus.PENDING, QueueStatus.SCHEDULED]),
                    or_(
                        VideoQueue.priority
                        > select(VideoQueue.priority).where(VideoQueue.id == queue_id),
                        and_(
                            VideoQueue.priority
                            == select(VideoQueue.priority).where(
                                VideoQueue.id == queue_id
                            ),
                            VideoQueue.created_at
                            < select(VideoQueue.created_at).where(
                                VideoQueue.id == queue_id
                            ),
                        ),
                    ),
                )
            )
        )
        return result.scalar() or 0

    def _calculate_cost(
        self, duration_minutes: int, style: str, queue_metadata: Dict[str, Any]
    ) -> float:
        """Calculate estimated cost"""
        base_cost = 0.10  # Base cost

        # Duration cost
        base_cost += duration_minutes * 0.05

        # Style multiplier
        style_multipliers = {
            "informative": 1.0,
            "entertaining": 1.2,
            "tutorial": 1.1,
            "review": 1.15,
        }
        base_cost *= style_multipliers.get(style, 1.0)

        # Add-on costs
        if queue_metadata.get("voice_style") == "elevenlabs":
            base_cost += 0.20
        if queue_metadata.get("thumbnail_style"):
            base_cost += 0.05
        if queue_metadata.get("auto_publish"):
            base_cost += 0.02

        return round(base_cost, 2)

    def _estimate_processing_time(self, duration_minutes: int, priority: int) -> int:
        """Estimate processing time in minutes"""
        # Base time: 2x video duration
        base_time = duration_minutes * 2

        # Priority adjustment
        if priority >= Priority.HIGH:
            base_time = int(base_time * 0.8)
        elif priority == Priority.LOW:
            base_time = int(base_time * 1.2)

        return max(5, base_time)  # Minimum 5 minutes

    async def _get_processing_rate(self, db: AsyncSession, user_id: str) -> float:
        """Get videos processed per hour"""
        # Get completed videos in last 24 hours
        yesterday = datetime.utcnow() - timedelta(days=1)

        result = await db.execute(
            select(func.count(VideoQueue.id)).where(
                and_(
                    VideoQueue.user_id == user_id,
                    VideoQueue.status == QueueStatus.COMPLETED,
                    VideoQueue.completed_at >= yesterday,
                )
            )
        )

        completed_24h = result.scalar() or 0
        return completed_24h / 24.0


# Global instance
queue_service = VideoQueueService()
