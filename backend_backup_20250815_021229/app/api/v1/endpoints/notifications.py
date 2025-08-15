"""
Notifications API endpoints
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from datetime import datetime

from app.api.v1.endpoints.auth import get_current_verified_user
from app.db.session import get_db
from app.services.notification_service import (
    notification_service, 
    NotificationPayload, 
    NotificationType, 
    NotificationPriority
)
from app.models.user import User

router = APIRouter()

class SendNotificationRequest(BaseModel):
    user_id: Optional[str] = None
    type: NotificationType = NotificationType.IN_APP
    title: str = Field(..., max_length=200)
    message: str = Field(..., max_length=1000)
    data: Dict[str, Any] = Field(default_factory=dict)
    priority: NotificationPriority = NotificationPriority.MEDIUM

class SendTemplateNotificationRequest(BaseModel):
    user_id: Optional[str] = None
    template_id: str
    variables: Dict[str, Any] = Field(default_factory=dict)
    notification_types: List[NotificationType] = Field(default_factory=list)

class BulkNotificationRequest(BaseModel):
    user_ids: List[str] = Field(..., min_items=1, max_items=1000)
    template_id: str
    variables: Dict[str, Any] = Field(default_factory=dict)
    notification_types: List[NotificationType] = Field(default_factory=list)
    batch_size: int = Field(default=50, ge=1, le=100)

class ScheduleNotificationRequest(BaseModel):
    user_id: Optional[str] = None
    type: NotificationType = NotificationType.IN_APP
    title: str = Field(..., max_length=200)
    message: str = Field(..., max_length=1000)
    data: Dict[str, Any] = Field(default_factory=dict)
    priority: NotificationPriority = NotificationPriority.MEDIUM
    scheduled_time: datetime
    expires_at: Optional[datetime] = None

@router.post("/send")
async def send_notification(
    request: SendNotificationRequest,
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db)
):
    """Send a single notification"""
    try:
        # Use current user ID if not specified (admin can send to others)
        target_user_id = request.user_id or str(current_user.id)
        
        # Create payload
        payload = NotificationPayload(
            user_id=target_user_id,
            template_id="custom",
            type=request.type,
            title=request.title,
            message=request.message,
            data=request.data,
            priority=request.priority
        )
        
        result = await notification_service.send_notification(payload, db)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "Failed to send notification")
            )
        
        return {
            "success": True,
            "notification_id": result["notification_id"],
            "message": "Notification sent successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/send-template")
async def send_template_notification(
    request: SendTemplateNotificationRequest,
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db)
):
    """Send notification using a template"""
    try:
        # Use current user ID if not specified
        target_user_id = request.user_id or str(current_user.id)
        
        result = await notification_service.send_template_notification(
            user_id=target_user_id,
            template_id=request.template_id,
            variables=request.variables,
            notification_types=request.notification_types or None,
            db=db
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to send template notification"
            )
        
        return {
            "success": True,
            "template_id": result["template_id"],
            "results": result["results"],
            "message": "Template notification sent successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/send-bulk")
async def send_bulk_notification(
    request: BulkNotificationRequest,
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db)
):
    """Send bulk notifications (admin only)"""
    try:
        # TODO: Add admin permission check
        # For now, allow all authenticated users
        
        result = await notification_service.send_bulk_notification(
            user_ids=request.user_ids,
            template_id=request.template_id,
            variables=request.variables,
            notification_types=request.notification_types or None,
            batch_size=request.batch_size
        )
        
        return {
            "success": True,
            "total_sent": result["total_sent"],
            "successful": result["successful"],
            "failed": result["failed"],
            "message": f"Bulk notification completed: {result['successful']}/{result['total_sent']} successful"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/schedule")
async def schedule_notification(
    request: ScheduleNotificationRequest,
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db)
):
    """Schedule a notification for future delivery"""
    try:
        # Use current user ID if not specified
        target_user_id = request.user_id or str(current_user.id)
        
        # Validate scheduled time is in future
        if request.scheduled_time <= datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Scheduled time must be in the future"
            )
        
        payload = NotificationPayload(
            user_id=target_user_id,
            template_id="custom_scheduled",
            type=request.type,
            title=request.title,
            message=request.message,
            data=request.data,
            priority=request.priority,
            scheduled_at=request.scheduled_time,
            expires_at=request.expires_at
        )
        
        notification_id = await notification_service.schedule_notification(
            payload=payload,
            scheduled_time=request.scheduled_time
        )
        
        return {
            "success": True,
            "notification_id": notification_id,
            "scheduled_time": request.scheduled_time,
            "message": "Notification scheduled successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/templates")
async def get_notification_templates(
    current_user: User = Depends(get_current_verified_user)
):
    """Get all available notification templates"""
    try:
        templates = notification_service.get_templates()
        
        return {
            "success": True,
            "templates": templates,
            "count": len(templates)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/test/{notification_type}")
async def test_notification(
    notification_type: NotificationType,
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db)
):
    """Send a test notification (for testing purposes)"""
    try:
        test_payload = NotificationPayload(
            user_id=str(current_user.id),
            template_id="test",
            type=notification_type,
            title="🧪 Test Notification",
            message=f"This is a test {notification_type.value} notification sent at {datetime.utcnow().isoformat()}",
            data={"test": True, "user_email": current_user.email},
            priority=NotificationPriority.LOW
        )
        
        result = await notification_service.send_notification(test_payload, db)
        
        return {
            "success": result["success"],
            "notification_id": result.get("notification_id"),
            "type": notification_type.value,
            "message": "Test notification sent" if result["success"] else "Test notification failed",
            "error": result.get("error")
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# Video generation notification helpers
@router.post("/video-generation-complete")
async def notify_video_generation_complete(
    video_id: str,
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db)
):
    """Send video generation complete notification"""
    try:
        # TODO: Fetch video details from database
        result = await notification_service.send_template_notification(
            user_id=str(current_user.id),
            template_id="video_generation_complete",
            variables={
                "title": f"Video {video_id}",
                "cost": "2.45",
                "duration": "180",
                "quality_score": "87"
            },
            notification_types=[NotificationType.EMAIL, NotificationType.IN_APP],
            db=db
        )
        
        return {"success": result["success"], "results": result["results"]}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/cost-alert")
async def send_cost_alert(
    threshold: float,
    current_cost: float,
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db)
):
    """Send cost threshold exceeded alert"""
    try:
        if current_cost <= threshold:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current cost has not exceeded threshold"
            )
        
        result = await notification_service.send_template_notification(
            user_id=str(current_user.id),
            template_id="cost_threshold_exceeded",
            variables={
                "threshold": f"{threshold:.2f}",
                "current_cost": f"{current_cost:.2f}"
            },
            notification_types=[NotificationType.EMAIL, NotificationType.IN_APP],
            db=db
        )
        
        return {"success": result["success"], "results": result["results"]}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )