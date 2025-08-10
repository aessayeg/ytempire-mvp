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
    id: str
    is_active: bool
    is_verified: bool
    subscription_tier: str
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        from_attributes = True