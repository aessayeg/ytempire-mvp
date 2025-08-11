"""
Cost Tracking Models
"""
from sqlalchemy import Column, String, Float, DateTime, ForeignKey, JSON, Boolean, Numeric, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.db.base_class import Base


class Cost(Base):
    """Cost tracking record"""
    __tablename__ = "costs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    video_id = Column(String, ForeignKey("videos.id"), nullable=True)
    
    # Cost Details
    service_type = Column(String, nullable=False, index=True)  # openai, elevenlabs, google-tts, etc.
    service_name = Column(String)  # gpt-4, claude-3, voice-synthesis, etc.
    operation = Column(String, index=True)  # script-generation, voice-synthesis, thumbnail, etc.
    
    # Metrics
    amount = Column(Numeric(10, 6), nullable=False)  # High precision cost
    units = Column(Float, nullable=False)  # tokens, characters, images, etc.
    unit_cost = Column(Float, nullable=False)  # cost per unit
    tokens_used = Column(JSON)  # {"input": 1000, "output": 500}
    characters_used = Column(Float)
    api_calls = Column(Float, default=1)
    
    # Tracking
    request_id = Column(String)
    response_time_ms = Column(Float)
    extra_data = Column(JSON, default={})  # Additional tracking data
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Relationships
    user = relationship("User", back_populates="costs")
    video = relationship("Video", back_populates="costs")


class CostThreshold(Base):
    """Cost threshold configuration"""
    __tablename__ = "cost_thresholds"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    threshold_type = Column(String, nullable=False)  # daily, monthly, per_video, service
    service = Column(String, nullable=True)  # specific service if applicable
    value = Column(Numeric(10, 2), nullable=False)  # threshold value in USD
    alert_email = Column(String, nullable=True)
    alert_webhook = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class CostAggregation(Base):
    """Pre-aggregated cost data for reporting"""
    __tablename__ = "cost_aggregations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    period_type = Column(String, nullable=False)  # hour, day, week, month
    period_start = Column(DateTime(timezone=True), nullable=False, index=True)
    period_end = Column(DateTime(timezone=True), nullable=False)
    service = Column(String, index=True)
    total_cost = Column(Numeric(10, 2), nullable=False)
    operation_count = Column(Integer, default=0)
    video_count = Column(Integer, default=0)
    user_count = Column(Integer, default=0)
    extra_data = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class CostBudget(Base):
    """Cost budget allocation"""
    __tablename__ = "cost_budgets"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    budget_type = Column(String, nullable=False)  # monthly, project, user
    amount = Column(Numeric(10, 2), nullable=False)
    spent = Column(Numeric(10, 2), default=0)
    start_date = Column(DateTime(timezone=True), nullable=False)
    end_date = Column(DateTime(timezone=True), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User", back_populates="cost_budgets")
