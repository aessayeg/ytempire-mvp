"""
Channel schemas
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, validator


class ChannelBase(BaseModel):
    channel_name: str = Field(..., min_length=1, max_length=100)
    channel_handle: Optional[str] = None
    channel_description: Optional[str] = None
    niche: Optional[str] = None
    content_type: Optional[str] = Field(None, regex="^(shorts|long-form|mixed)$")
    

class ChannelCreate(ChannelBase):
    youtube_channel_id: str
    channel_url: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    

class ChannelUpdate(BaseModel):
    channel_name: Optional[str] = None
    channel_description: Optional[str] = None
    niche: Optional[str] = None
    content_type: Optional[str] = None
    target_audience: Optional[Dict[str, Any]] = None
    posting_schedule: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None
    auto_generate: Optional[bool] = None
    auto_publish: Optional[bool] = None
    quality_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    voice_id: Optional[str] = None
    style_preferences: Optional[Dict[str, Any]] = None
    content_guidelines: Optional[Dict[str, Any]] = None
    

class ChannelResponse(ChannelBase):
    id: str
    user_id: str
    youtube_channel_id: str
    channel_url: Optional[str]
    is_active: bool
    auto_generate: bool
    auto_publish: bool
    quality_threshold: float
    subscriber_count: int
    total_views: int
    total_videos: int
    average_views: float
    is_monetized: bool
    estimated_revenue: float
    created_at: datetime
    updated_at: Optional[datetime]
    last_video_at: Optional[datetime]
    last_sync_at: Optional[datetime]
    
    class Config:
        from_attributes = True