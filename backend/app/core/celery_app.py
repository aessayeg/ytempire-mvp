"""
Celery Configuration
Owner: Data Pipeline Engineer #1
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
        "app.tasks.video_pipeline",
        "app.tasks.content_generation",
        "app.tasks.audio_synthesis",
        "app.tasks.video_compilation", 
        "app.tasks.youtube_upload", 
        "app.tasks.analytics",
        "app.tasks.vector_indexing",
        "app.tasks.secret_rotation"
    ]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task execution settings
    task_soft_time_limit=3600,  # 1 hour soft limit
    task_time_limit=7200,  # 2 hour hard limit
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    
    # Worker settings
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
    
    # Result backend settings
    result_expires=86400,  # 24 hours
    
    # Beat schedule for periodic tasks
    beat_schedule={
        "update-channel-analytics": {
            "task": "app.tasks.analytics.update_channel_analytics",
            "schedule": 3600.0,  # Every hour
        },
        "check-video-queue": {
            "task": "app.tasks.video_generation.check_video_queue",
            "schedule": 300.0,  # Every 5 minutes
        },
        "cost-report": {
            "task": "app.tasks.analytics.generate_cost_report",
            "schedule": 86400.0,  # Daily
        },
        "cleanup-old-tasks": {
            "task": "app.tasks.maintenance.cleanup_old_tasks",
            "schedule": 86400.0,  # Daily
        },
        "check-expired-secrets": {
            "task": "app.tasks.secret_rotation.check_expired_secrets",
            "schedule": 86400.0,  # Daily
        },
        "audit-secrets-access": {
            "task": "app.tasks.secret_rotation.audit_secrets_access", 
            "schedule": 43200.0,  # Every 12 hours
        },
        "validate-secret-integrity": {
            "task": "app.tasks.secret_rotation.validate_secret_integrity",
            "schedule": 604800.0,  # Weekly
        },
        "cleanup-old-secret-versions": {
            "task": "app.tasks.secret_rotation.cleanup_old_secret_versions",
            "schedule": 604800.0,  # Weekly
        },
        "generate-security-report": {
            "task": "app.tasks.secret_rotation.generate_security_report",
            "schedule": 86400.0,  # Daily
        },
    },
    
    # Queue routing
    task_routes={
        "app.tasks.video_generation.*": {"queue": "video_generation"},
        "app.tasks.youtube_upload.*": {"queue": "youtube_upload"},
        "app.tasks.analytics.*": {"queue": "analytics"},
        "app.tasks.maintenance.*": {"queue": "maintenance"},
    },
    
    # Rate limits
    task_annotations={
        "app.tasks.video_generation.generate_video": {
            "rate_limit": "10/h",  # 10 videos per hour
        },
        "app.tasks.youtube_upload.upload_video": {
            "rate_limit": "50/h",  # YouTube API limit consideration
        },
    },
)

# Queue configuration
CELERY_QUEUES = {
    "video_generation": {
        "exchange": "video_generation",
        "routing_key": "video_generation",
        "priority": 10,
    },
    "youtube_upload": {
        "exchange": "youtube_upload",
        "routing_key": "youtube_upload",
        "priority": 8,
    },
    "analytics": {
        "exchange": "analytics",
        "routing_key": "analytics",
        "priority": 5,
    },
    "maintenance": {
        "exchange": "maintenance",
        "routing_key": "maintenance",
        "priority": 1,
    },
}