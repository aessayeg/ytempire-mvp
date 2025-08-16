"""
WebSocket Event Handlers and Messages
Handles real-time communication events for YTEmpire
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import asyncio
import logging
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EventType(Enum):
    """WebSocket event types"""

    # Video events
    VIDEO_GENERATION_STARTED = "video.generation.started"
    VIDEO_GENERATION_PROGRESS = "video.generation.progress"
    VIDEO_GENERATION_COMPLETED = "video.generation.completed"
    VIDEO_GENERATION_FAILED = "video.generation.failed"
    VIDEO_PUBLISHED = "video.published"
    VIDEO_ANALYTICS_UPDATE = "video.analytics.update"

    # Channel events
    CHANNEL_STATUS_CHANGED = "channel.status.changed"
    CHANNEL_METRICS_UPDATE = "channel.metrics.update"
    CHANNEL_QUOTA_WARNING = "channel.quota.warning"
    CHANNEL_HEALTH_UPDATE = "channel.health.update"

    # System events
    SYSTEM_ALERT = "system.alert"
    SYSTEM_METRICS = "system.metrics"
    COST_ALERT = "cost.alert"
    PERFORMANCE_WARNING = "performance.warning"

    # User events
    USER_NOTIFICATION = "user.notification"
    USER_ACTION_REQUIRED = "user.action.required"

    # AI/ML events
    MODEL_UPDATE = "model.update"
    TREND_DETECTED = "trend.detected"
    QUALITY_SCORE_UPDATE = "quality.score.update"


class WebSocketMessage(BaseModel):
    """Base WebSocket message structure"""

    event: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class VideoGenerationEvent(BaseModel):
    """Video generation event data"""

    video_id: str
    channel_id: str
    status: str
    progress: float = 0.0
    current_step: Optional[str] = None
    estimated_completion: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ChannelMetricsEvent(BaseModel):
    """Channel metrics update event"""

    channel_id: str
    subscribers: int
    views_today: int
    revenue_today: float
    videos_published: int
    health_score: float
    quota_used: int
    quota_limit: int


class SystemMetricsEvent(BaseModel):
    """System metrics event"""

    active_generations: int
    queue_depth: int
    avg_generation_time: float
    success_rate: float
    cost_today: float
    api_health: Dict[str, str]  # Service name -> status
    performance_metrics: Dict[str, float]


class EventHandler:
    """Handles WebSocket events and broadcasts"""

    def __init__(self, manager):
        self.manager = manager

    async def emit_video_generation_started(
        self,
        user_id: str,
        video_id: str,
        channel_id: str,
        title: str,
        estimated_duration: int,
    ):
        """Emit video generation started event"""
        event = VideoGenerationEvent(
            video_id=video_id,
            channel_id=channel_id,
            status="started",
            progress=0.0,
            current_step="Initializing",
            estimated_completion=datetime.utcnow().replace(
                second=datetime.utcnow().second + estimated_duration
            ),
            metadata={"title": title},
        )

        message = WebSocketMessage(
            event=EventType.VIDEO_GENERATION_STARTED.value, data=event.dict()
        )

        # Send to user and channel room
        await self.manager.send_personal_message(user_id, message.dict())
        await self.manager.send_to_room(f"channel:{channel_id}", message.dict())

        logger.info(f"Emitted video generation started: {video_id}")

    async def emit_video_generation_progress(
        self,
        user_id: str,
        video_id: str,
        channel_id: str,
        progress: float,
        current_step: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Emit video generation progress update"""
        event = VideoGenerationEvent(
            video_id=video_id,
            channel_id=channel_id,
            status="processing",
            progress=progress,
            current_step=current_step,
            metadata=metadata,
        )

        message = WebSocketMessage(
            event=EventType.VIDEO_GENERATION_PROGRESS.value, data=event.dict()
        )

        await self.manager.send_personal_message(user_id, message.dict())
        await self.manager.send_to_room(f"channel:{channel_id}", message.dict())

    async def emit_video_generation_completed(
        self,
        user_id: str,
        video_id: str,
        channel_id: str,
        video_url: str,
        thumbnail_url: str,
        duration: int,
        cost: float,
    ):
        """Emit video generation completed event"""
        event = VideoGenerationEvent(
            video_id=video_id,
            channel_id=channel_id,
            status="completed",
            progress=100.0,
            current_step="Completed",
            metadata={
                "video_url": video_url,
                "thumbnail_url": thumbnail_url,
                "duration": duration,
                "cost": cost,
            },
        )

        message = WebSocketMessage(
            event=EventType.VIDEO_GENERATION_COMPLETED.value, data=event.dict()
        )

        await self.manager.send_personal_message(user_id, message.dict())
        await self.manager.send_to_room(f"channel:{channel_id}", message.dict())

        logger.info(f"Emitted video generation completed: {video_id}")

    async def emit_video_generation_failed(
        self, user_id: str, video_id: str, channel_id: str, error: str, step: str
    ):
        """Emit video generation failed event"""
        event = VideoGenerationEvent(
            video_id=video_id,
            channel_id=channel_id,
            status="failed",
            current_step=step,
            error=error,
        )

        message = WebSocketMessage(
            event=EventType.VIDEO_GENERATION_FAILED.value, data=event.dict()
        )

        await self.manager.send_personal_message(user_id, message.dict())
        await self.manager.send_to_room(f"channel:{channel_id}", message.dict())

        logger.error(f"Emitted video generation failed: {video_id} - {error}")

    async def emit_channel_metrics_update(
        self, channel_id: str, metrics: ChannelMetricsEvent
    ):
        """Emit channel metrics update"""
        message = WebSocketMessage(
            event=EventType.CHANNEL_METRICS_UPDATE.value, data=metrics.dict()
        )

        await self.manager.send_to_room(f"channel:{channel_id}", message.dict())

    async def emit_system_metrics(self, metrics: SystemMetricsEvent):
        """Broadcast system metrics to all connected users"""
        message = WebSocketMessage(
            event=EventType.SYSTEM_METRICS.value, data=metrics.dict()
        )

        await self.manager.broadcast(message.dict())

    async def emit_cost_alert(
        self,
        user_id: str,
        service: str,
        current_cost: float,
        limit: float,
        percentage: float,
    ):
        """Emit cost alert to user"""
        message = WebSocketMessage(
            event=EventType.COST_ALERT.value,
            data={
                "service": service,
                "current_cost": current_cost,
                "limit": limit,
                "percentage": percentage,
                "severity": "warning" if percentage < 90 else "critical",
            },
        )

        await self.manager.send_personal_message(user_id, message.dict())

        logger.warning(f"Cost alert for {user_id}: {service} at {percentage}% of limit")

    async def emit_trend_detected(
        self, trend_data: Dict[str, Any], channels: List[str]
    ):
        """Emit trend detection to relevant channels"""
        message = WebSocketMessage(
            event=EventType.TREND_DETECTED.value, data=trend_data
        )

        for channel_id in channels:
            await self.manager.send_to_room(f"channel:{channel_id}", message.dict())

    async def emit_quality_score_update(
        self, user_id: str, video_id: str, scores: Dict[str, float]
    ):
        """Emit quality score update"""
        message = WebSocketMessage(
            event=EventType.QUALITY_SCORE_UPDATE.value,
            data={
                "video_id": video_id,
                "scores": scores,
                "overall_score": sum(scores.values()) / len(scores),
            },
        )

        await self.manager.send_personal_message(user_id, message.dict())

    async def emit_user_notification(
        self,
        user_id: str,
        title: str,
        message: str,
        severity: str = "info",
        action: Optional[Dict[str, str]] = None,
    ):
        """Send notification to user"""
        msg = WebSocketMessage(
            event=EventType.USER_NOTIFICATION.value,
            data={
                "title": title,
                "message": message,
                "severity": severity,
                "action": action,
            },
        )

        await self.manager.send_personal_message(user_id, msg.dict())


class WebSocketMetricsCollector:
    """Collects and broadcasts system metrics periodically"""

    def __init__(self, manager, event_handler):
        self.manager = manager
        self.event_handler = event_handler
        self.running = False

    async def start(self):
        """Start metrics collection"""
        self.running = True
        asyncio.create_task(self._collect_metrics())
        logger.info("WebSocket metrics collector started")

    async def stop(self):
        """Stop metrics collection"""
        self.running = False
        logger.info("WebSocket metrics collector stopped")

    async def _collect_metrics(self):
        """Collect and broadcast metrics periodically"""
        while self.running:
            try:
                # Collect system metrics
                metrics = SystemMetricsEvent(
                    active_generations=0,  # TODO: Get from queue manager
                    queue_depth=0,  # TODO: Get from Redis
                    avg_generation_time=0.0,  # TODO: Calculate from database
                    success_rate=0.0,  # TODO: Calculate from database
                    cost_today=0.0,  # TODO: Get from cost tracker
                    api_health={
                        "openai": "healthy",
                        "youtube": "healthy",
                        "elevenlabs": "healthy",
                    },
                    performance_metrics={
                        "api_latency_ms": 150,
                        "db_latency_ms": 5,
                        "cache_hit_rate": 0.85,
                    },
                )

                # Broadcast metrics
                await self.event_handler.emit_system_metrics(metrics)

            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")

            # Wait 30 seconds before next collection
            await asyncio.sleep(30)


class WebSocketRateLimiter:
    """Rate limiter for WebSocket messages"""

    def __init__(self, max_messages_per_second: int = 10):
        self.max_messages_per_second = max_messages_per_second
        self.message_counts: Dict[str, List[float]] = {}

    async def check_rate_limit(self, user_id: str) -> bool:
        """Check if user has exceeded rate limit"""
        now = datetime.utcnow().timestamp()

        if user_id not in self.message_counts:
            self.message_counts[user_id] = []

        # Remove messages older than 1 second
        self.message_counts[user_id] = [
            ts for ts in self.message_counts[user_id] if now - ts < 1.0
        ]

        # Check rate limit
        if len(self.message_counts[user_id]) >= self.max_messages_per_second:
            return False

        # Add current message
        self.message_counts[user_id].append(now)
        return True

    def cleanup(self):
        """Clean up old entries"""
        now = datetime.utcnow().timestamp()

        for user_id in list(self.message_counts.keys()):
            self.message_counts[user_id] = [
                ts
                for ts in self.message_counts[user_id]
                if now - ts < 60.0  # Keep last minute
            ]

            if not self.message_counts[user_id]:
                del self.message_counts[user_id]
