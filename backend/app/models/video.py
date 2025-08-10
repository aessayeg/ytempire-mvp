"""
Video Model
"""
from sqlalchemy import Column, String, Boolean, DateTime, Integer, Float, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.db.base import Base


class Video(Base):
    __tablename__ = "videos"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    channel_id = Column(String, ForeignKey("channels.id"), nullable=False)
    
    # Video Info
    title = Column(String, nullable=False)
    description = Column(Text)
    tags = Column(JSON)
    category = Column(String)
    
    # YouTube Info
    youtube_video_id = Column(String, unique=True, index=True)
    youtube_url = Column(String)
    thumbnail_url = Column(String)
    
    # Content
    script = Column(Text)
    voice_script = Column(Text)
    visual_prompts = Column(JSON)
    
    # Files
    video_file_path = Column(String)
    audio_file_path = Column(String)
    thumbnail_file_path = Column(String)
    
    # Generation Details
    generation_status = Column(String, default="pending")  # pending, processing, completed, failed, published
    generation_started_at = Column(DateTime(timezone=True))
    generation_completed_at = Column(DateTime(timezone=True))
    generation_time_seconds = Column(Integer)
    
    # Publishing
    publish_status = Column(String, default="draft")  # draft, scheduled, published, failed
    scheduled_publish_time = Column(DateTime(timezone=True))
    published_at = Column(DateTime(timezone=True))
    
    # Quality Metrics
    quality_score = Column(Float)  # 0-1 AI quality assessment
    trend_score = Column(Float)  # 0-1 trend relevance
    engagement_prediction = Column(Float)  # predicted engagement rate
    
    # Performance Metrics
    view_count = Column(Integer, default=0)
    like_count = Column(Integer, default=0)
    dislike_count = Column(Integer, default=0)
    comment_count = Column(Integer, default=0)
    share_count = Column(Integer, default=0)
    watch_time_minutes = Column(Float, default=0.0)
    
    # Monetization
    is_monetized = Column(Boolean, default=False)
    estimated_revenue = Column(Float, default=0.0)
    actual_revenue = Column(Float, default=0.0)
    
    # Cost Tracking
    total_cost = Column(Float, default=0.0)
    script_cost = Column(Float, default=0.0)
    voice_cost = Column(Float, default=0.0)
    video_cost = Column(Float, default=0.0)
    thumbnail_cost = Column(Float, default=0.0)
    
    # AI Models Used
    script_model = Column(String)  # gpt-4, claude-3, etc.
    voice_model = Column(String)  # elevenlabs, google-tts, etc.
    video_model = Column(String)  # runway, stable-video, etc.
    
    # Metadata
    duration_seconds = Column(Integer)
    file_size_mb = Column(Float)
    resolution = Column(String)  # 1080p, 720p, etc.
    aspect_ratio = Column(String)  # 16:9, 9:16, 1:1
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    channel = relationship("Channel", back_populates="videos")
    costs = relationship("Cost", back_populates="video", cascade="all, delete-orphan")