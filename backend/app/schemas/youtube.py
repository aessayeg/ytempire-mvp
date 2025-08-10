"""
YouTube API Schemas
Owner: Integration Specialist
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Dict, Any
from datetime import datetime


class VideoUploadRequest(BaseModel):
    title: str = Field(..., max_length=100, description="Video title")
    description: str = Field(default="", max_length=5000, description="Video description")
    tags: Optional[List[str]] = Field(default=None, max_items=30, description="Video tags")
    category_id: str = Field(default="22", description="YouTube category ID")
    privacy_status: str = Field(default="private", regex="^(private|unlisted|public)$")


class VideoUploadResponse(BaseModel):
    video_id: str = Field(..., description="YouTube video ID")
    url: str = Field(..., description="YouTube video URL")
    status: str = Field(..., description="Upload status")
    title: str = Field(..., description="Video title")
    privacy_status: str = Field(..., description="Privacy status")


class VideoStatsResponse(BaseModel):
    video_id: str
    title: Optional[str] = None
    published_at: Optional[str] = None
    views: int = 0
    likes: int = 0
    comments: int = 0
    privacy_status: Optional[str] = None
    upload_status: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = []


class ChannelInfoResponse(BaseModel):
    channel_id: str
    title: str
    description: Optional[str] = None
    custom_url: Optional[str] = None
    subscriber_count: int = 0
    video_count: int = 0
    view_count: int = 0
    created_at: Optional[str] = None


class VideoUpdateRequest(BaseModel):
    title: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = Field(None, max_length=5000)
    tags: Optional[List[str]] = Field(None, max_items=30)
    privacy_status: Optional[str] = Field(None, regex="^(private|unlisted|public)$")


class YouTubeOAuthResponse(BaseModel):
    authorization_url: str = Field(..., description="OAuth authorization URL")
    state: str = Field(..., description="OAuth state parameter")


class VideoSearchResult(BaseModel):
    video_id: str
    title: str
    description: str
    channel_title: str
    published_at: str
    thumbnail_url: Optional[str] = None


class QuotaUsageResponse(BaseModel):
    quota_used: int
    daily_limit: int
    remaining: int
    last_reset: str
    percentage_used: float


class WebhookData(BaseModel):
    video_id: str
    status: str
    user_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None