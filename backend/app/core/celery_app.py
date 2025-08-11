"""
Celery Configuration for YTEmpire
Handles asynchronous video processing tasks
"""
import os
from celery import Celery
from kombu import Exchange, Queue
from dotenv import load_dotenv

load_dotenv()

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_DB = os.getenv("REDIS_DB", "0")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

# Build Redis URL
if REDIS_PASSWORD:
    REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
else:
    REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

# Initialize Celery
celery_app = Celery(
    "ytempire",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        "app.tasks.video_tasks",
        "app.tasks.youtube_tasks",
        "app.tasks.ai_tasks",
        "app.tasks.analytics_tasks"
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task execution settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task routing
    task_routes={
        "app.tasks.video_tasks.*": {"queue": "video_processing"},
        "app.tasks.youtube_tasks.*": {"queue": "youtube_upload"},
        "app.tasks.ai_tasks.*": {"queue": "ai_generation"},
        "app.tasks.analytics_tasks.*": {"queue": "analytics"}
    },
    
    # Queue configuration
    task_queues=(
        Queue("default", Exchange("default"), routing_key="default"),
        Queue("video_processing", Exchange("video"), routing_key="video.process", 
              queue_arguments={"x-max-priority": 10}),
        Queue("youtube_upload", Exchange("youtube"), routing_key="youtube.upload",
              queue_arguments={"x-max-priority": 5}),
        Queue("ai_generation", Exchange("ai"), routing_key="ai.generate",
              queue_arguments={"x-max-priority": 8}),
        Queue("analytics", Exchange("analytics"), routing_key="analytics.process",
              queue_arguments={"x-max-priority": 3})
    ),
    
    # Worker settings
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=1000,
    worker_disable_rate_limits=False,
    
    # Task time limits
    task_soft_time_limit=600,  # 10 minutes soft limit
    task_time_limit=900,  # 15 minutes hard limit
    
    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_persistent=True,
    
    # Beat schedule for periodic tasks
    beat_schedule={
        "cleanup-old-videos": {
            "task": "app.tasks.video_tasks.cleanup_old_videos",
            "schedule": 3600.0,  # Every hour
        },
        "sync-youtube-analytics": {
            "task": "app.tasks.analytics_tasks.sync_youtube_analytics",
            "schedule": 1800.0,  # Every 30 minutes
        },
        "check-video-processing": {
            "task": "app.tasks.video_tasks.check_stuck_videos",
            "schedule": 300.0,  # Every 5 minutes
        },
        "cost-aggregation": {
            "task": "app.tasks.analytics_tasks.aggregate_costs",
            "schedule": 600.0,  # Every 10 minutes
        }
    },
    
    # Error handling
    task_reject_on_worker_lost=True,
    task_acks_late=True,
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True
)

# Task priorities
class TaskPriority:
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10
