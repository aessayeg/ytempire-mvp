"""
User Schemas
Owner: Backend Team Lead
"""

from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime


class UserBase(BaseModel):
    email: EmailStr
    username: str
    full_name: Optional[str] = None


class UserCreate(UserBase):
    password: str


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    monthly_budget: Optional[float] = None


class UserResponse(UserBase):
    id: int
    is_active: bool
    subscription_tier: str
    channels_limit: int
    daily_video_limit: int
    total_spent: float = 0.0
    monthly_budget: float = 100.0
    is_beta_user: bool
    created_at: datetime
    
    class Config:
        from_attributes = True