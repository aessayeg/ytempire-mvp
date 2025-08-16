"""
Celery Configuration for YTEmpire
Handles asynchronous video processing tasks with scaling support for 100+ videos/day
Enhanced for Week 2: Distributed processing, auto-scaling, and performance optimization
"""
import os
from celery import Celery
from kombu import Exchange, Queue
from dotenv import load_dotenv
from datetime import timedelta

load_dotenv()

# Redis configuration with Sentinel support for HA
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_DB = os.getenv("REDIS_DB", "0")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_SENTINEL_HOSTS = (
    os.getenv("REDIS_SENTINEL_HOSTS", "").split(",")
    if os.getenv("REDIS_SENTINEL_HOSTS")
    else []
)

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
        "app.tasks.analytics_tasks",
    ],
)

# Celery configuration optimized for 100+ videos/day
celery_app.conf.update(
    # Task execution settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Enhanced task routing with batch processing support
    task_routes={
        "app.tasks.video_tasks.*": {"queue": "video_processing"},
        "app.tasks.youtube_tasks.*": {"queue": "youtube_upload"},
        "app.tasks.ai_tasks.*": {"queue": "ai_generation"},
        "app.tasks.analytics_tasks.*": {"queue": "analytics"},
        "app.tasks.pipeline_tasks.*": {"queue": "pipeline_orchestration"},
        "app.tasks.batch_tasks.*": {"queue": "batch_processing"},
    },
    # Enhanced queue configuration for high throughput
    task_queues=(
        Queue("default", Exchange("default"), routing_key="default"),
        Queue(
            "video_processing",
            Exchange("video"),
            routing_key="video.process",
            queue_arguments={"x-max-priority": 10, "x-max-length": 1000},
        ),
        Queue(
            "youtube_upload",
            Exchange("youtube"),
            routing_key="youtube.upload",
            queue_arguments={"x-max-priority": 5, "x-max-length": 500},
        ),
        Queue(
            "ai_generation",
            Exchange("ai"),
            routing_key="ai.generate",
            queue_arguments={"x-max-priority": 8, "x-max-length": 500},
        ),
        Queue(
            "analytics",
            Exchange("analytics"),
            routing_key="analytics.process",
            queue_arguments={"x-max-priority": 3, "x-max-length": 200},
        ),
        Queue(
            "pipeline_orchestration",
            Exchange("pipeline"),
            routing_key="pipeline.orchestrate",
            queue_arguments={"x-max-priority": 10, "x-max-length": 200},
        ),
        Queue(
            "batch_processing",
            Exchange("batch"),
            routing_key="batch.process",
            queue_arguments={"x-max-priority": 7, "x-max-length": 100},
        ),
    ),
    # Optimized worker settings for concurrent processing
    worker_prefetch_multiplier=8,  # Increased for better throughput
    worker_max_tasks_per_child=500,  # Reduced for memory management
    worker_disable_rate_limits=False,
    worker_pool="prefork",  # Process pool for CPU-bound tasks
    worker_concurrency=8,  # Number of concurrent worker processes
    worker_autoscale=[16, 4],  # Auto-scale between 4-16 workers [max, min]
    # Adjusted task time limits for video processing
    task_soft_time_limit=900,  # 15 minutes soft limit
    task_time_limit=1200,  # 20 minutes hard limit
    # Batch processing settings
    task_annotations={
        "app.tasks.video_tasks.batch_generate_videos": {
            "rate_limit": "50/m",  # Limit batch processing rate
            "max_retries": 3,
        },
        "app.tasks.pipeline_tasks.process_video_pipeline": {
            "rate_limit": "100/m",
            "max_retries": 5,
        },
    },
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
        },
    },
    # Error handling
    task_reject_on_worker_lost=True,
    task_acks_late=True,
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)


# Task priorities
class TaskPriority:
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


# Auto-scaling configuration
class AutoScalingConfig:
    """Configuration for dynamic worker scaling based on queue depth"""

    MIN_WORKERS = 4
    MAX_WORKERS = 16
    SCALE_UP_THRESHOLD = 20  # Queue depth to trigger scale up
    SCALE_DOWN_THRESHOLD = 5  # Queue depth to trigger scale down
    COOLDOWN_PERIOD = 60  # Seconds between scaling operations

    # Queue-specific scaling rules
    QUEUE_SCALING = {
        "video_processing": {"min": 4, "max": 10, "threshold": 15},
        "ai_generation": {"min": 2, "max": 8, "threshold": 10},
        "youtube_upload": {"min": 2, "max": 6, "threshold": 8},
        "batch_processing": {"min": 2, "max": 8, "threshold": 5},
    }


# Performance monitoring hooks
@celery_app.task
def monitor_queue_depth():
    """Monitor queue depths for auto-scaling decisions"""
    from celery import current_app
    from redis import Redis
    import json

    redis_client = Redis.from_url(REDIS_URL)
    metrics = {}

    for queue_name in [
        "video_processing",
        "ai_generation",
        "youtube_upload",
        "batch_processing",
    ]:
        queue_key = f"celery:queue:{queue_name}"
        depth = redis_client.llen(queue_key)
        metrics[queue_name] = depth

        # Store metrics for monitoring
        redis_client.setex(
            f"metrics:queue_depth:{queue_name}",
            300,  # 5 minute expiry
            json.dumps({"depth": depth, "timestamp": os.time()}),
        )

    return metrics


@celery_app.task
def auto_scale_workers():
    """Auto-scale workers based on queue depth"""
    from celery import current_app
    import subprocess

    metrics = monitor_queue_depth()

    for queue_name, depth in metrics.items():
        if queue_name in AutoScalingConfig.QUEUE_SCALING:
            config = AutoScalingConfig.QUEUE_SCALING[queue_name]

            if depth > config["threshold"]:
                # Scale up
                subprocess.run(
                    [
                        "celery",
                        "-A",
                        "app.core.celery_app",
                        "worker",
                        "--queue",
                        queue_name,
                        "--concurrency",
                        str(config["max"]),
                        "--detach",
                    ]
                )
            elif depth < config["threshold"] // 2:
                # Scale down (handled by worker lifecycle)
                pass

    return metrics
