"""
Celery configuration and app initialization
"""
from celery import Celery
from app.core.config import settings

# Create Celery instance
celery_app = Celery(
    "ytempire",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.tasks.video_generation",
        "app.tasks.youtube_upload",
        "app.tasks.analytics",
        "app.tasks.cost_tracking"
    ]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    result_expires=3600,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

# Beat schedule for periodic tasks
celery_app.conf.beat_schedule = {
    "sync-youtube-analytics": {
        "task": "app.tasks.analytics.sync_youtube_analytics",
        "schedule": 3600.0,  # Every hour
    },
    "check-video-status": {
        "task": "app.tasks.video_generation.check_pending_videos",
        "schedule": 300.0,  # Every 5 minutes
    },
    "update-cost-metrics": {
        "task": "app.tasks.cost_tracking.update_daily_costs",
        "schedule": 1800.0,  # Every 30 minutes
    },
    "trend-analysis": {
        "task": "app.tasks.analytics.analyze_trends",
        "schedule": 3600.0,  # Every hour
    },
}