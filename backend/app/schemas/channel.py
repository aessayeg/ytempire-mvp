"""
Channel Schemas
Owner: Backend Team Lead
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime


class ChannelBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None


class ChannelCreate(ChannelBase):
    category: Optional[str] = Field(default="technology", max_length=100)
    target_audience: Optional[str] = Field(default="general", max_length=255)
    content_style: Optional[str] = Field(default="educational", max_length=100)
    upload_schedule: Optional[str] = Field(default="weekly", max_length=100)
    
    # Branding options
    primary_color: Optional[str] = Field(default=None, max_length=7)  # Hex color
    secondary_color: Optional[str] = Field(default=None, max_length=7)
    logo_url: Optional[str] = Field(default=None, max_length=500)
    
    # Automation settings
    auto_publish: Optional[bool] = Field(default=False)
    seo_optimization: Optional[bool] = Field(default=True)
    thumbnail_generation: Optional[bool] = Field(default=True)
    content_scheduling: Optional[bool] = Field(default=False)


class ChannelUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    category: Optional[str] = Field(None, max_length=100)
    target_audience: Optional[str] = Field(None, max_length=255)
    content_style: Optional[str] = Field(None, max_length=100)
    upload_schedule: Optional[str] = Field(None, max_length=100)
    
    # Branding options
    primary_color: Optional[str] = Field(None, max_length=7)
    secondary_color: Optional[str] = Field(None, max_length=7)
    logo_url: Optional[str] = Field(None, max_length=500)
    
    # Automation settings
    auto_publish: Optional[bool] = None
    seo_optimization: Optional[bool] = None
    thumbnail_generation: Optional[bool] = None
    content_scheduling: Optional[bool] = None


class ChannelResponse(ChannelBase):
    id: str
    user_id: str
    category: Optional[str] = None
    target_audience: Optional[str] = None
    content_style: Optional[str] = None
    upload_schedule: Optional[str] = None
    youtube_channel_id: Optional[str] = None
    is_active: bool
    branding: Optional[Dict] = None
    automation_settings: Optional[Dict] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ChannelStats(BaseModel):
    channel_id: str
    total_videos: int
    total_cost: float
    average_cost_per_video: float
    total_views: int = 0
    total_subscribers: int = 0
    average_views_per_video: float = 0.0


class ChannelWithStats(ChannelResponse):
    stats: ChannelStats