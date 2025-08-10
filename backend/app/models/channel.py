"""
Channel Model
Owner: Backend Team Lead
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, JSON, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base


class Channel(Base):
    __tablename__ = "channels"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Channel Info
    name = Column(String, nullable=False)
    youtube_channel_id = Column(String, unique=True, index=True)
    niche = Column(String, nullable=False)  # AI/ML Team Lead requirement
    description = Column(String)
    
    # Configuration
    is_active = Column(Boolean, default=True)
    auto_publish = Column(Boolean, default=False)
    publish_schedule = Column(JSON)  # {"days": ["mon", "wed", "fri"], "time": "10:00"}
    
    # Content Settings (ML Engineer requirement)
    content_style = Column(String, default="educational")  # educational, entertainment, news
    target_duration = Column(Integer, default=600)  # seconds
    voice_type = Column(String, default="neural")  # neural, standard
    language = Column(String, default="en")
    
    # Performance Metrics (Analytics Engineer)
    total_videos = Column(Integer, default=0)
    total_views = Column(Integer, default=0)
    total_subscribers = Column(Integer, default=0)
    average_ctr = Column(Float, default=0.0)
    
    # Cost Tracking (VP of AI requirement)
    total_cost = Column(Float, default=0.0)
    average_cost_per_video = Column(Float, default=0.0)
    
    # YouTube API Credentials (Integration Specialist)
    youtube_refresh_token = Column(String)
    youtube_access_token = Column(String)
    token_expires_at = Column(DateTime(timezone=True))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_video_at = Column(DateTime(timezone=True))