"""
User Model
Owner: Backend Team Lead
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float
from sqlalchemy.sql import func
from app.core.database import Base


class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    
    # Profile
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    
    # Subscription & Limits
    subscription_tier = Column(String, default="free")  # free, pro, enterprise
    channels_limit = Column(Integer, default=5)
    daily_video_limit = Column(Integer, default=10)
    
    # Cost Tracking (Analytics Engineer requirement)
    total_spent = Column(Float, default=0.0)
    monthly_budget = Column(Float, default=100.0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True))
    
    # Beta User Flag (Product Owner requirement)
    is_beta_user = Column(Boolean, default=False)