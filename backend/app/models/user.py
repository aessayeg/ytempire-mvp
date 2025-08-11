"""
User Model
"""
from sqlalchemy import Column, String, Boolean, DateTime, Integer, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.db.base_class import Base


class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String)
    hashed_password = Column(String, nullable=False)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    is_verified = Column(Boolean, default=False)
    
    # Verification & Security
    verification_token = Column(String, unique=True, index=True)
    verified_at = Column(DateTime(timezone=True))
    password_reset_token = Column(String, unique=True, index=True)
    password_reset_expires = Column(DateTime(timezone=True))
    
    # Subscription
    subscription_tier = Column(String, default="free")  # free, starter, pro, enterprise
    subscription_status = Column(String, default="active")  # active, cancelled, expired
    subscription_end_date = Column(DateTime(timezone=True))
    
    # Limits
    channels_limit = Column(Integer, default=1)
    videos_per_day_limit = Column(Integer, default=5)
    monthly_budget_limit = Column(Float, default=100.0)
    api_quota_remaining = Column(Integer, default=100)
    api_quota_reset_at = Column(DateTime(timezone=True))
    
    # Tracking
    total_videos_generated = Column(Integer, default=0)
    total_revenue_generated = Column(Float, default=0.0)
    total_cost_incurred = Column(Float, default=0.0)
    
    # Profile
    company_name = Column(String)
    phone = Column(String)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True))
    deleted_at = Column(DateTime(timezone=True))
    
    # Relationships
    channels = relationship("Channel", back_populates="owner", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    costs = relationship("Cost", back_populates="user", cascade="all, delete-orphan")
    subscription = relationship("Subscription", back_populates="user", uselist=False)
    cost_budgets = relationship("CostBudget", back_populates="user", cascade="all, delete-orphan")