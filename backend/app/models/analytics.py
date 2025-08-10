"""
Analytics Models
Owner: Analytics Engineer
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Date
from sqlalchemy.sql import func
from app.core.database import Base


class ChannelAnalytics(Base):
    __tablename__ = "channel_analytics"
    
    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(Integer, ForeignKey("channels.id"), nullable=False)
    date = Column(Date, nullable=False, index=True)
    
    # Subscriber Metrics
    subscribers = Column(Integer, default=0)
    subscribers_gained = Column(Integer, default=0)
    subscribers_lost = Column(Integer, default=0)
    
    # View Metrics
    views = Column(Integer, default=0)
    unique_viewers = Column(Integer, default=0)
    returning_viewers = Column(Integer, default=0)
    impressions = Column(Integer, default=0)
    ctr = Column(Float, default=0.0)
    
    # Engagement Metrics
    watch_time_hours = Column(Float, default=0.0)
    average_view_duration = Column(Float, default=0.0)
    likes = Column(Integer, default=0)
    dislikes = Column(Integer, default=0)
    comments = Column(Integer, default=0)
    shares = Column(Integer, default=0)
    
    # Revenue Metrics (if monetized)
    estimated_revenue = Column(Float, default=0.0)
    cpm = Column(Float, default=0.0)  # Cost per mille
    
    # Demographics
    demographics = Column(JSON)  # Age, gender, location breakdown
    traffic_sources = Column(JSON)  # Search, suggested, external, etc.
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class VideoAnalytics(Base):
    __tablename__ = "video_analytics"
    
    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    date = Column(Date, nullable=False, index=True)
    
    # View Metrics
    views = Column(Integer, default=0)
    unique_viewers = Column(Integer, default=0)
    impressions = Column(Integer, default=0)
    ctr = Column(Float, default=0.0)
    
    # Watch Time
    watch_time_hours = Column(Float, default=0.0)
    average_view_duration = Column(Float, default=0.0)
    average_percentage_viewed = Column(Float, default=0.0)
    
    # Engagement
    likes = Column(Integer, default=0)
    dislikes = Column(Integer, default=0)
    comments = Column(Integer, default=0)
    shares = Column(Integer, default=0)
    saves = Column(Integer, default=0)
    
    # Retention
    retention_data = Column(JSON)  # {0: 100%, 15: 85%, 30: 70%, ...}
    
    # Traffic Sources
    traffic_sources = Column(JSON)
    search_terms = Column(JSON)
    
    # Device & Demographics
    device_types = Column(JSON)
    demographics = Column(JSON)
    
    # Revenue
    estimated_revenue = Column(Float, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class CostTracking(Base):
    __tablename__ = "cost_tracking"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Reference
    user_id = Column(Integer, ForeignKey("users.id"))
    channel_id = Column(Integer, ForeignKey("channels.id"))
    video_id = Column(Integer, ForeignKey("videos.id"))
    
    # Cost Details
    service = Column(String, nullable=False)  # openai, elevenlabs, google_tts, etc.
    operation = Column(String, nullable=False)  # script_generation, voice_synthesis, etc.
    
    # Usage
    tokens_used = Column(Integer)
    characters_used = Column(Integer)
    api_calls = Column(Integer, default=1)
    
    # Cost
    unit_cost = Column(Float, nullable=False)
    total_cost = Column(Float, nullable=False)
    
    # Metadata
    metadata = Column(JSON)  # Additional service-specific data
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())