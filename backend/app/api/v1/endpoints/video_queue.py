"""
Video Queue Management API Endpoints
Handles video generation queue and scheduling
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, or_
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging

from app.db.session import get_db
from app.api.v1.endpoints.auth import get_current_verified_user
from app.models.user import User
from app.core.celery_app import celery_app

logger = logging.getLogger(__name__)

router = APIRouter()


class QueueStatus(str, Enum):
    """Video queue status"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class Priority(int, Enum):
    """Queue priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


class VideoQueueRequest(BaseModel):
    """Request to add video to queue"""
    channel_id: str
    title: str
    description: Optional[str] = None
    topic: str
    style: str = "informative"
    duration_minutes: int = Field(default=5, ge=1, le=30)
    scheduled_time: Optional[datetime] = None
    priority: Priority = Priority.NORMAL
    tags: List[str] = []
    thumbnail_style: Optional[str] = None
    voice_style: Optional[str] = None
    target_audience: str = "general"
    keywords: List[str] = []
    auto_publish: bool = False


class VideoQueueUpdate(BaseModel):
    """Update queue item request"""
    scheduled_time: Optional[datetime] = None
    priority: Optional[Priority] = None
    status: Optional[QueueStatus] = None
    title: Optional[str] = None
    description: Optional[str] = None


class VideoQueueResponse(BaseModel):
    """Video queue item response"""
    queue_id: str
    channel_id: str
    user_id: str
    title: str
    description: Optional[str]
    topic: str
    style: str
    duration_minutes: int
    scheduled_time: Optional[datetime]
    priority: int
    status: str
    tags: List[str]
    estimated_cost: float
    processing_time_estimate: int  # minutes
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]
    error_message: Optional[str]
    retry_count: int
    metadata: Dict[str, Any]


class QueueBatchRequest(BaseModel):
    """Batch queue request"""
    channel_id: str
    videos: List[VideoQueueRequest]
    stagger_minutes: int = Field(default=60, description="Minutes between each video")


class QueueStatsResponse(BaseModel):
    """Queue statistics response"""
    total_items: int
    pending: int
    scheduled: int
    processing: int
    completed: int
    failed: int
    estimated_total_cost: float
    estimated_completion_time: Optional[datetime]
    processing_rate: float  # videos per hour


# In-memory storage (should be replaced with database)
video_queue_storage: Dict[str, Dict] = {}


@router.post("/add", response_model=VideoQueueResponse)
async def add_to_queue(
    request: VideoQueueRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> VideoQueueResponse:
    """
    Add a video to the generation queue
    """
    try:
        # Generate queue ID
        queue_id = str(uuid.uuid4())
        
        # Estimate cost based on duration and features
        estimated_cost = calculate_estimated_cost(
            duration_minutes=request.duration_minutes,
            style=request.style,
            voice_style=request.voice_style,
            thumbnail_style=request.thumbnail_style
        )
        
        # Estimate processing time
        processing_time = estimate_processing_time(
            duration_minutes=request.duration_minutes,
            priority=request.priority
        )
        
        # Create queue item
        queue_item = {
            "queue_id": queue_id,
            "channel_id": request.channel_id,
            "user_id": str(current_user.id),
            "title": request.title,
            "description": request.description,
            "topic": request.topic,
            "style": request.style,
            "duration_minutes": request.duration_minutes,
            "scheduled_time": request.scheduled_time or datetime.utcnow(),
            "priority": request.priority.value,
            "status": QueueStatus.SCHEDULED if request.scheduled_time else QueueStatus.PENDING,
            "tags": request.tags,
            "estimated_cost": estimated_cost,
            "processing_time_estimate": processing_time,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "completed_at": None,
            "error_message": None,
            "retry_count": 0,
            "metadata": {
                "target_audience": request.target_audience,
                "keywords": request.keywords,
                "voice_style": request.voice_style,
                "thumbnail_style": request.thumbnail_style,
                "auto_publish": request.auto_publish
            }
        }
        
        # Store in memory (replace with database)
        video_queue_storage[queue_id] = queue_item
        
        # Schedule processing task
        if request.scheduled_time:
            # Schedule for later
            eta = request.scheduled_time
        else:
            # Process based on priority
            eta = datetime.utcnow() + timedelta(minutes=1 if request.priority == Priority.URGENT else 5)
        
        # Queue Celery task
        task = celery_app.send_task(
            'app.tasks.video_generation.process_video',
            args=[queue_id],
            eta=eta,
            priority=request.priority.value
        )
        
        queue_item['metadata']['task_id'] = task.id
        
        return VideoQueueResponse(**queue_item)
        
    except Exception as e:
        logger.error(f"Failed to add to queue: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue video: {str(e)}"
        )


@router.post("/batch", response_model=List[VideoQueueResponse])
async def add_batch_to_queue(
    request: QueueBatchRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> List[VideoQueueResponse]:
    """
    Add multiple videos to queue with staggered scheduling
    """
    try:
        queued_items = []
        base_time = datetime.utcnow()
        
        for i, video_request in enumerate(request.videos):
            # Set scheduled time with stagger
            video_request.scheduled_time = base_time + timedelta(minutes=i * request.stagger_minutes)
            video_request.channel_id = request.channel_id
            
            # Add to queue
            queue_item = await add_to_queue(
                video_request,
                background_tasks,
                db,
                current_user
            )
            queued_items.append(queue_item)
        
        return queued_items
        
    except Exception as e:
        logger.error(f"Batch queue failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue batch: {str(e)}"
        )


@router.get("/list", response_model=List[VideoQueueResponse])
async def get_queue(
    status: Optional[QueueStatus] = None,
    channel_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> List[VideoQueueResponse]:
    """
    Get video queue items
    """
    try:
        # Filter queue items
        items = []
        for queue_id, item in video_queue_storage.items():
            if item['user_id'] != str(current_user.id):
                continue
            if status and item['status'] != status:
                continue
            if channel_id and item['channel_id'] != channel_id:
                continue
            items.append(VideoQueueResponse(**item))
        
        # Sort by priority and scheduled time
        items.sort(key=lambda x: (-x.priority, x.scheduled_time or datetime.max))
        
        # Apply pagination
        return items[offset:offset + limit]
        
    except Exception as e:
        logger.error(f"Failed to get queue: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch queue"
        )


@router.get("/{queue_id}", response_model=VideoQueueResponse)
async def get_queue_item(
    queue_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> VideoQueueResponse:
    """
    Get specific queue item details
    """
    if queue_id not in video_queue_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Queue item not found"
        )
    
    item = video_queue_storage[queue_id]
    
    if item['user_id'] != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    return VideoQueueResponse(**item)


@router.patch("/{queue_id}", response_model=VideoQueueResponse)
async def update_queue_item(
    queue_id: str,
    update_request: VideoQueueUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> VideoQueueResponse:
    """
    Update queue item (reschedule, change priority, etc.)
    """
    if queue_id not in video_queue_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Queue item not found"
        )
    
    item = video_queue_storage[queue_id]
    
    if item['user_id'] != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Update fields
    if update_request.scheduled_time is not None:
        item['scheduled_time'] = update_request.scheduled_time
        item['status'] = QueueStatus.SCHEDULED
    
    if update_request.priority is not None:
        item['priority'] = update_request.priority.value
    
    if update_request.status is not None:
        item['status'] = update_request.status
    
    if update_request.title is not None:
        item['title'] = update_request.title
    
    if update_request.description is not None:
        item['description'] = update_request.description
    
    item['updated_at'] = datetime.utcnow()
    
    # Reschedule Celery task if needed
    if update_request.scheduled_time or update_request.priority:
        # Cancel existing task
        if 'task_id' in item['metadata']:
            celery_app.control.revoke(item['metadata']['task_id'])
        
        # Create new task
        eta = item['scheduled_time'] or datetime.utcnow()
        task = celery_app.send_task(
            'app.tasks.video_generation.process_video',
            args=[queue_id],
            eta=eta,
            priority=item['priority']
        )
        item['metadata']['task_id'] = task.id
    
    return VideoQueueResponse(**item)


@router.delete("/{queue_id}")
async def cancel_queue_item(
    queue_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, str]:
    """
    Cancel/remove item from queue
    """
    if queue_id not in video_queue_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Queue item not found"
        )
    
    item = video_queue_storage[queue_id]
    
    if item['user_id'] != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Cancel Celery task
    if 'task_id' in item['metadata']:
        celery_app.control.revoke(item['metadata']['task_id'], terminate=True)
    
    # Update status
    item['status'] = QueueStatus.CANCELLED
    item['updated_at'] = datetime.utcnow()
    
    return {"status": "cancelled", "queue_id": queue_id}


@router.post("/{queue_id}/retry")
async def retry_failed_item(
    queue_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> VideoQueueResponse:
    """
    Retry a failed queue item
    """
    if queue_id not in video_queue_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Queue item not found"
        )
    
    item = video_queue_storage[queue_id]
    
    if item['user_id'] != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    if item['status'] != QueueStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only failed items can be retried"
        )
    
    # Update retry count and status
    item['retry_count'] += 1
    item['status'] = QueueStatus.PENDING
    item['error_message'] = None
    item['updated_at'] = datetime.utcnow()
    
    # Queue new task
    task = celery_app.send_task(
        'app.tasks.video_generation.process_video',
        args=[queue_id],
        countdown=60  # Retry after 1 minute
    )
    item['metadata']['task_id'] = task.id
    
    return VideoQueueResponse(**item)


@router.get("/stats/summary", response_model=QueueStatsResponse)
async def get_queue_statistics(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> QueueStatsResponse:
    """
    Get queue statistics and estimates
    """
    try:
        stats = {
            "total_items": 0,
            "pending": 0,
            "scheduled": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0,
            "estimated_total_cost": 0.0
        }
        
        processing_times = []
        
        for item in video_queue_storage.values():
            if item['user_id'] != str(current_user.id):
                continue
            
            stats['total_items'] += 1
            stats[item['status']] = stats.get(item['status'], 0) + 1
            stats['estimated_total_cost'] += item['estimated_cost']
            
            if item['status'] in [QueueStatus.PENDING, QueueStatus.SCHEDULED]:
                processing_times.append(item['processing_time_estimate'])
        
        # Calculate completion time
        total_processing_minutes = sum(processing_times)
        estimated_completion = None
        if total_processing_minutes > 0:
            estimated_completion = datetime.utcnow() + timedelta(minutes=total_processing_minutes)
        
        # Calculate processing rate
        completed_today = 0
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        for item in video_queue_storage.values():
            if (item['user_id'] == str(current_user.id) and 
                item['status'] == QueueStatus.COMPLETED and
                item.get('completed_at') and 
                item['completed_at'] >= today_start):
                completed_today += 1
        
        hours_elapsed = (datetime.utcnow() - today_start).total_seconds() / 3600
        processing_rate = completed_today / max(hours_elapsed, 1)
        
        return QueueStatsResponse(
            **stats,
            estimated_completion_time=estimated_completion,
            processing_rate=processing_rate
        )
        
    except Exception as e:
        logger.error(f"Failed to get queue stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch queue statistics"
        )


@router.post("/pause-all")
async def pause_all_queue_items(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Pause all pending/scheduled items in queue
    """
    paused_count = 0
    
    for queue_id, item in video_queue_storage.items():
        if (item['user_id'] == str(current_user.id) and 
            item['status'] in [QueueStatus.PENDING, QueueStatus.SCHEDULED]):
            
            item['status'] = QueueStatus.PAUSED
            item['updated_at'] = datetime.utcnow()
            
            # Cancel Celery task
            if 'task_id' in item['metadata']:
                celery_app.control.revoke(item['metadata']['task_id'])
            
            paused_count += 1
    
    return {
        "status": "success",
        "paused_count": paused_count
    }


@router.post("/resume-all")
async def resume_all_queue_items(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Resume all paused items in queue
    """
    resumed_count = 0
    
    for queue_id, item in video_queue_storage.items():
        if (item['user_id'] == str(current_user.id) and 
            item['status'] == QueueStatus.PAUSED):
            
            item['status'] = QueueStatus.PENDING
            item['updated_at'] = datetime.utcnow()
            
            # Queue new task
            task = celery_app.send_task(
                'app.tasks.video_generation.process_video',
                args=[queue_id],
                priority=item['priority']
            )
            item['metadata']['task_id'] = task.id
            
            resumed_count += 1
    
    return {
        "status": "success",
        "resumed_count": resumed_count
    }


# Helper functions
def calculate_estimated_cost(
    duration_minutes: int,
    style: str,
    voice_style: Optional[str],
    thumbnail_style: Optional[str]
) -> float:
    """Calculate estimated cost for video generation"""
    base_cost = 0.1  # Base cost
    
    # Duration cost
    base_cost += duration_minutes * 0.05
    
    # Style cost
    style_costs = {
        "informative": 0.1,
        "entertaining": 0.15,
        "tutorial": 0.12,
        "review": 0.13
    }
    base_cost += style_costs.get(style, 0.1)
    
    # Voice cost
    if voice_style:
        voice_costs = {
            "elevenlabs": 0.2,
            "google": 0.05,
            "azure": 0.08
        }
        base_cost += voice_costs.get(voice_style, 0.05)
    
    # Thumbnail cost
    if thumbnail_style:
        base_cost += 0.05
    
    return round(base_cost, 2)


def estimate_processing_time(duration_minutes: int, priority: Priority) -> int:
    """Estimate processing time in minutes"""
    base_time = duration_minutes * 2  # 2x video duration
    
    # Priority adjustment
    if priority == Priority.URGENT:
        base_time *= 0.8
    elif priority == Priority.LOW:
        base_time *= 1.2
    
    return int(base_time)