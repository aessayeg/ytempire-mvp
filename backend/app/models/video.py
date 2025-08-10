"""
Video Model
Owner: Backend Team Lead
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, JSON, Float, Text, Enum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import enum
from app.core.database import Base


class VideoStatus(enum.Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    READY_FOR_UPLOAD = "ready_for_upload"
    UPLOADING = "uploading"
    PUBLISHED = "published"
    FAILED = "failed"
    SCHEDULED = "scheduled"


class VideoPrivacy(enum.Enum):
    PUBLIC = "public"
    UNLISTED = "unlisted"
    PRIVATE = "private"


class Video(Base):
    __tablename__ = "videos"
    
    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(Integer, ForeignKey("channels.id"), nullable=False)
    
    # Video Information
    title = Column(String(100), nullable=False)
    description = Column(Text)
    tags = Column(JSON)  # List of tags
    category_id = Column(String, default="28")  # Science & Technology
    
    # Content Details
    topic = Column(String)
    script = Column(Text)
    hook = Column(Text)
    call_to_action = Column(String)
    
    # Media Files
    video_file_path = Column(String)
    thumbnail_path = Column(String)
    audio_file_path = Column(String)
    
    # Video Properties
    duration = Column(Integer)  # seconds
    resolution = Column(String, default="1920x1080")
    fps = Column(Integer, default=30)
    file_size_mb = Column(Float)
    
    # YouTube Data
    youtube_video_id = Column(String, unique=True, index=True)
    youtube_url = Column(String)
    privacy = Column(Enum(VideoPrivacy), default=VideoPrivacy.PUBLIC)
    scheduled_publish_time = Column(DateTime(timezone=True))
    
    # Performance Metrics
    views = Column(Integer, default=0)
    likes = Column(Integer, default=0)
    dislikes = Column(Integer, default=0)
    comments = Column(Integer, default=0)
    watch_time_hours = Column(Float, default=0.0)
    ctr = Column(Float, default=0.0)  # Click-through rate
    retention_rate = Column(Float, default=0.0)
    
    # Cost Tracking (VP of AI requirement)
    script_cost = Column(Float, default=0.0)
    voice_cost = Column(Float, default=0.0)
    image_cost = Column(Float, default=0.0)
    video_cost = Column(Float, default=0.0)
    music_cost = Column(Float, default=0.0)
    total_cost = Column(Float, default=0.0)
    
    # Quality Metrics (ML Engineer)
    quality_score = Column(Float, default=0.0)
    audio_quality = Column(Float, default=0.0)
    video_quality = Column(Float, default=0.0)
    content_score = Column(Float, default=0.0)
    
    # Status
    status = Column(Enum(VideoStatus), default=VideoStatus.QUEUED, index=True)
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    
    # Celery Task IDs (Data Pipeline Engineer)
    generation_task_id = Column(String)
    upload_task_id = Column(String)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    processing_started_at = Column(DateTime(timezone=True))
    processing_completed_at = Column(DateTime(timezone=True))
    published_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())