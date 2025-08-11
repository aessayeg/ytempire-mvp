"""
YouTube Channel Model
"""
from sqlalchemy import Column, String, Boolean, DateTime, Integer, Float, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.db.base_class import Base


class Channel(Base):
    __tablename__ = "channels"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id = Column(String, ForeignKey("users.id"), nullable=False)  # Renamed from user_id for clarity
    
    # Channel Basics
    name = Column(String, nullable=False)  # Internal name for the channel
    description = Column(String)
    category = Column(String)  # gaming, tech, education, etc.
    language = Column(String, default="en")
    
    # YouTube Info
    youtube_channel_id = Column(String, unique=True, index=True)
    youtube_channel_url = Column(String)
    youtube_api_key = Column(String)  # Encrypted in production
    youtube_refresh_token = Column(String)
    
    # API Keys
    api_key = Column(String, unique=True)  # Unique API key for this channel
    api_secret = Column(String)  # For additional security
    
    # Configuration
    target_audience = Column(String)  # Target audience description
    upload_schedule = Column(String, default="daily")  # daily, weekly, custom
    content_type = Column(String, default="mixed")  # shorts, long-form, mixed
    posting_schedule = Column(JSON)  # Detailed schedule configuration
    
    # Automation Settings
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)  # YouTube connection verified
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
    deleted_at = Column(DateTime(timezone=True))  # For soft delete
    last_video_at = Column(DateTime(timezone=True))
    last_sync_at = Column(DateTime(timezone=True))
    
    # Relationships
    owner = relationship("User", back_populates="channels")
    videos = relationship("Video", back_populates="channel", cascade="all, delete-orphan")
    analytics = relationship("Analytics", back_populates="channel", cascade="all, delete-orphan")
