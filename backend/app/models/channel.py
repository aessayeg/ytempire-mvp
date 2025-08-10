"""
YouTube Channel Model
"""
from sqlalchemy import Column, String, Boolean, DateTime, Integer, Float, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.db.base import Base


class Channel(Base):
    __tablename__ = "channels"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    
    # YouTube Info
    youtube_channel_id = Column(String, unique=True, index=True)
    channel_name = Column(String, nullable=False)
    channel_handle = Column(String)
    channel_description = Column(String)
    channel_url = Column(String)
    
    # OAuth
    access_token = Column(String)
    refresh_token = Column(String)
    token_expires_at = Column(DateTime(timezone=True))
    
    # Configuration
    niche = Column(String)  # gaming, tech, education, etc.
    content_type = Column(String)  # shorts, long-form, mixed
    target_audience = Column(JSON)  # demographics
    posting_schedule = Column(JSON)  # schedule configuration
    
    # Automation Settings
    is_active = Column(Boolean, default=True)
    auto_generate = Column(Boolean, default=True)
    auto_publish = Column(Boolean, default=False)
    quality_threshold = Column(Float, default=0.8)  # 0-1 quality score
    
    # Performance Metrics
    subscriber_count = Column(Integer, default=0)
    total_views = Column(Integer, default=0)
    total_videos = Column(Integer, default=0)
    average_views = Column(Float, default=0.0)
    
    # Monetization
    is_monetized = Column(Boolean, default=False)
    estimated_revenue = Column(Float, default=0.0)
    rpm = Column(Float, default=0.0)  # Revenue per mille
    
    # AI Settings
    voice_id = Column(String)  # ElevenLabs voice ID
    style_preferences = Column(JSON)  # style configuration
    content_guidelines = Column(JSON)  # content rules
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_video_at = Column(DateTime(timezone=True))
    last_sync_at = Column(DateTime(timezone=True))
    
    # Relationships
    owner = relationship("User", back_populates="channels")
    videos = relationship("Video", back_populates="channel", cascade="all, delete-orphan")
    analytics = relationship("Analytics", back_populates="channel", cascade="all, delete-orphan")