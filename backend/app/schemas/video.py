"""
Video Schemas
Owner: API Developer
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class VideoGenerateRequest(BaseModel):
    channel_id: str
    topic: str = Field(..., min_length=1, max_length=200)
    style: str = "educational"
    target_duration: int = Field(600, ge=60, le=3600)  # 1-60 minutes
    keywords: Optional[List[str]] = None
    custom_prompt: Optional[str] = None
    auto_publish: bool = False
    schedule_time: Optional[datetime] = None
    priority: Optional[int] = Field(default=1, ge=1, le=5)


class VideoCreate(BaseModel):
    channel_id: str
    title: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    category: Optional[str] = None
    priority: Optional[int] = Field(default=1, ge=1, le=5)
    scheduled_publish_at: Optional[datetime] = None
    content_settings: Optional[Dict[str, Any]] = None
    generation_settings: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class VideoUpdate(BaseModel):
    title: Optional[str] = Field(None, max_length=500)
    description: Optional[str] = Field(None, max_length=5000)
    tags: Optional[List[str]] = Field(None, max_items=50)
    category: Optional[str] = Field(None, max_length=100)
    priority: Optional[int] = Field(None, ge=1, le=5)
    scheduled_publish_at: Optional[datetime] = None


class VideoResponse(BaseModel):
    id: str
    channel_id: str
    user_id: str
    title: Optional[str] = None
    description: Optional[str] = None
    script_content: Optional[str] = None
    thumbnail_url: Optional[str] = None
    video_url: Optional[str] = None
    youtube_video_id: Optional[str] = None
    status: str
    current_stage: Optional[str] = None
    priority: int
    scheduled_publish_at: Optional[datetime] = None
    published_at: Optional[datetime] = None
    content_settings: Optional[Dict[str, Any]] = None
    generation_settings: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    category: Optional[str] = None
    duration: Optional[int] = None
    total_cost: Optional[float] = None
    pipeline_id: Optional[str] = None
    pipeline_started_at: Optional[datetime] = None
    pipeline_completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class VideoStats(BaseModel):
    video_id: str
    views: int = 0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    watch_time_minutes: int = 0
    click_through_rate: float = 0.0
    engagement_score: float = 0.0


class VideoCostBreakdown(BaseModel):
    video_id: str
    content_generation_cost: float = 0.0
    audio_synthesis_cost: float = 0.0
    visual_cost: float = 0.0
    compilation_cost: float = 0.0
    other_costs: float = 0.0
    total_cost: float = 0.0
    within_budget: bool = True


class VideoListResponse(BaseModel):
    videos: List[VideoResponse]
    total_count: int
    page: int
    per_page: int
    has_next: bool
    has_prev: bool