"""
Analytics Model
"""
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, JSON
from sqlalchemy.sql import func
import uuid
from app.db.base import Base


class Analytics(Base):
    __tablename__ = "analytics"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    channel_id = Column(String, ForeignKey("channels.id"), nullable=False)
    
    # Time Period
    period_type = Column(String)  # daily, weekly, monthly
    period_start = Column(DateTime(timezone=True))
    period_end = Column(DateTime(timezone=True))
    
    # Channel Metrics
    subscriber_count = Column(Integer, default=0)
    subscriber_growth = Column(Integer, default=0)
    total_views = Column(Integer, default=0)
    view_growth = Column(Integer, default=0)
    
    # Video Metrics
    videos_published = Column(Integer, default=0)
    average_views_per_video = Column(Float, default=0.0)
    average_watch_time = Column(Float, default=0.0)
    average_engagement_rate = Column(Float, default=0.0)
    
    # Revenue Metrics
    estimated_revenue = Column(Float, default=0.0)
    actual_revenue = Column(Float, default=0.0)
    rpm = Column(Float, default=0.0)
    cpm = Column(Float, default=0.0)
    
    # Cost Metrics
    total_cost = Column(Float, default=0.0)
    cost_per_video = Column(Float, default=0.0)
    roi = Column(Float, default=0.0)  # Return on Investment
    profit_margin = Column(Float, default=0.0)
    
    # Performance Metrics
    best_performing_video = Column(JSON)
    worst_performing_video = Column(JSON)
    trending_topics = Column(JSON)
    audience_demographics = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    channel = relationship("Channel", back_populates="analytics")