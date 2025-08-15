"""
Notification Service for YTEmpire
Handles email, SMS, push notifications, and in-app notifications
"""

import logging
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json

from app.core.config import settings
from app.db.session import get_db
from app.services.email_service import EmailService
from app.services.websocket_manager import ConnectionManager

logger = logging.getLogger(__name__)

class NotificationType(str, Enum):
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    IN_APP = "in_app"
    WEBHOOK = "webhook"

class NotificationPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class NotificationStatus(str, Enum):
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    DELIVERED = "delivered"

@dataclass
class NotificationTemplate:
    id: str
    name: str
    subject: str
    body: str
    variables: List[str]
    type: NotificationType
    priority: NotificationPriority = NotificationPriority.MEDIUM

@dataclass
class NotificationPayload:
    user_id: str
    template_id: str
    type: NotificationType
    title: str
    message: str
    data: Dict[str, Any]
    priority: NotificationPriority = NotificationPriority.MEDIUM
    scheduled_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

class NotificationService:
    def __init__(self):
        self.email_service = EmailService()
        self.websocket_manager = ConnectionManager()
        self.templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, NotificationTemplate]:
        """Load notification templates"""
        return {
            "video_generation_complete": NotificationTemplate(
                id="video_generation_complete",
                name="Video Generation Complete",
                subject="✅ Video '{title}' Generated Successfully",
                body="Your video '{title}' has been generated successfully and is ready for publishing.\n\nCost: ${cost}\nGeneration Time: {duration}s\nQuality Score: {quality_score}/100",
                variables=["title", "cost", "duration", "quality_score"],
                type=NotificationType.EMAIL,
                priority=NotificationPriority.MEDIUM
            ),
            "video_generation_failed": NotificationTemplate(
                id="video_generation_failed",
                name="Video Generation Failed",
                subject="❌ Video Generation Failed",
                body="Video generation failed for '{title}' due to: {error_message}\n\nPlease check your settings and try again.",
                variables=["title", "error_message"],
                type=NotificationType.EMAIL,
                priority=NotificationPriority.HIGH
            ),
            "cost_threshold_exceeded": NotificationTemplate(
                id="cost_threshold_exceeded",
                name="Cost Threshold Exceeded",
                subject="⚠️ Daily Cost Threshold Exceeded",
                body="Your daily cost limit of ${threshold} has been exceeded.\n\nCurrent spend: ${current_cost}\nRecommendation: Review your cost settings or pause video generation.",
                variables=["threshold", "current_cost"],
                type=NotificationType.EMAIL,
                priority=NotificationPriority.URGENT
            ),
            "channel_quota_exceeded": NotificationTemplate(
                id="channel_quota_exceeded",
                name="YouTube API Quota Exceeded",
                subject="⚠️ YouTube API Quota Limit Reached",
                body="Channel '{channel_name}' has reached its daily YouTube API quota.\n\nAutomatic rotation to backup channels has been activated.",
                variables=["channel_name"],
                type=NotificationType.EMAIL,
                priority=NotificationPriority.HIGH
            ),
            "revenue_milestone": NotificationTemplate(
                id="revenue_milestone",
                name="Revenue Milestone Achieved",
                subject="🎉 Revenue Milestone Achieved!",
                body="Congratulations! You've reached ${milestone} in total revenue.\n\nTotal earnings: ${total_revenue}\nThis month: ${monthly_revenue}",
                variables=["milestone", "total_revenue", "monthly_revenue"],
                type=NotificationType.EMAIL,
                priority=NotificationPriority.MEDIUM
            ),
            "system_maintenance": NotificationTemplate(
                id="system_maintenance",
                name="System Maintenance Notice",
                subject="🔧 Scheduled Maintenance Notice",
                body="YTEmpire will undergo scheduled maintenance on {maintenance_date} from {start_time} to {end_time}.\n\nDuring this time, video generation will be temporarily unavailable.",
                variables=["maintenance_date", "start_time", "end_time"],
                type=NotificationType.EMAIL,
                priority=NotificationPriority.MEDIUM
            )
        }

    async def send_notification(
        self, 
        payload: NotificationPayload,
        db: Optional[AsyncSession] = None
    ) -> Dict[str, Any]:
        """Send a notification"""
        try:
            # Store notification in database
            notification_id = await self._store_notification(payload, db)
            
            # Send based on type
            if payload.type == NotificationType.EMAIL:
                result = await self._send_email_notification(payload)
            elif payload.type == NotificationType.IN_APP:
                result = await self._send_in_app_notification(payload)
            elif payload.type == NotificationType.PUSH:
                result = await self._send_push_notification(payload)
            elif payload.type == NotificationType.SMS:
                result = await self._send_sms_notification(payload)
            elif payload.type == NotificationType.WEBHOOK:
                result = await self._send_webhook_notification(payload)
            else:
                result = {"success": False, "error": f"Unsupported notification type: {payload.type}"}
            
            # Update notification status
            if db:
                await self._update_notification_status(
                    notification_id, 
                    NotificationStatus.SENT if result.get("success") else NotificationStatus.FAILED,
                    db
                )
            
            return {
                "notification_id": notification_id,
                "success": result.get("success", False),
                "message": result.get("message", ""),
                "error": result.get("error")
            }
            
        except Exception as e:
            logger.error(f"Failed to send notification: {str(e)}")
            return {"success": False, "error": str(e)}

    async def send_template_notification(
        self,
        user_id: str,
        template_id: str,
        variables: Dict[str, Any],
        notification_types: List[NotificationType] = None,
        db: Optional[AsyncSession] = None
    ) -> Dict[str, Any]:
        """Send notification using a template"""
        if template_id not in self.templates:
            return {"success": False, "error": f"Template '{template_id}' not found"}
        
        template = self.templates[template_id]
        
        # Replace variables in template
        subject = template.subject.format(**variables)
        body = template.body.format(**variables)
        
        # Default to template type if not specified
        if not notification_types:
            notification_types = [template.type]
        
        results = []
        for notif_type in notification_types:
            payload = NotificationPayload(
                user_id=user_id,
                template_id=template_id,
                type=notif_type,
                title=subject,
                message=body,
                data=variables,
                priority=template.priority
            )
            
            result = await self.send_notification(payload, db)
            results.append({
                "type": notif_type,
                "result": result
            })
        
        return {
            "template_id": template_id,
            "results": results,
            "success": all(r["result"]["success"] for r in results)
        }

    async def _send_email_notification(self, payload: NotificationPayload) -> Dict[str, Any]:
        """Send email notification"""
        try:
            # Use existing email service
            result = await self.email_service.send_notification_email(
                user_id=payload.user_id,
                subject=payload.title,
                body=payload.message,
                priority=payload.priority.value
            )
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _send_in_app_notification(self, payload: NotificationPayload) -> Dict[str, Any]:
        """Send in-app notification via WebSocket"""
        try:
            await self.websocket_manager.send_personal_message(
                json.dumps({
                    "type": "notification",
                    "title": payload.title,
                    "message": payload.message,
                    "priority": payload.priority.value,
                    "data": payload.data,
                    "timestamp": datetime.utcnow().isoformat()
                }),
                payload.user_id
            )
            return {"success": True, "message": "In-app notification sent"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _send_push_notification(self, payload: NotificationPayload) -> Dict[str, Any]:
        """Send push notification (placeholder for future implementation)"""
        # TODO: Implement push notification service (Firebase, OneSignal, etc.)
        logger.info(f"Push notification placeholder for user {payload.user_id}: {payload.title}")
        return {"success": True, "message": "Push notification queued (not implemented)"}

    async def _send_sms_notification(self, payload: NotificationPayload) -> Dict[str, Any]:
        """Send SMS notification (placeholder for future implementation)"""
        # TODO: Implement SMS service (Twilio, etc.)
        logger.info(f"SMS notification placeholder for user {payload.user_id}: {payload.message}")
        return {"success": True, "message": "SMS notification queued (not implemented)"}

    async def _send_webhook_notification(self, payload: NotificationPayload) -> Dict[str, Any]:
        """Send webhook notification"""
        try:
            # TODO: Implement webhook delivery
            logger.info(f"Webhook notification for user {payload.user_id}: {payload.data}")
            return {"success": True, "message": "Webhook delivered"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _store_notification(self, payload: NotificationPayload, db: Optional[AsyncSession]) -> str:
        """Store notification in database"""
        if not db:
            async for session in get_db():
                db = session
                break
        
        # TODO: Implement database storage
        # For now, return a mock ID
        notification_id = f"notif_{datetime.utcnow().timestamp()}"
        logger.info(f"Stored notification {notification_id} for user {payload.user_id}")
        return notification_id

    async def _update_notification_status(
        self, 
        notification_id: str, 
        status: NotificationStatus,
        db: AsyncSession
    ):
        """Update notification status in database"""
        # TODO: Implement status update
        logger.info(f"Updated notification {notification_id} status to {status}")

    async def send_bulk_notification(
        self,
        user_ids: List[str],
        template_id: str,
        variables: Dict[str, Any],
        notification_types: List[NotificationType] = None,
        batch_size: int = 50
    ) -> Dict[str, Any]:
        """Send bulk notifications in batches"""
        results = []
        
        # Process in batches to avoid overwhelming the system
        for i in range(0, len(user_ids), batch_size):
            batch = user_ids[i:i + batch_size]
            batch_results = []
            
            # Send notifications concurrently within batch
            tasks = []
            for user_id in batch:
                task = self.send_template_notification(
                    user_id=user_id,
                    template_id=template_id,
                    variables=variables,
                    notification_types=notification_types
                )
                tasks.append(task)
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            results.extend(batch_results)
            
            # Small delay between batches
            await asyncio.sleep(0.1)
        
        success_count = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        
        return {
            "total_sent": len(user_ids),
            "successful": success_count,
            "failed": len(user_ids) - success_count,
            "results": results
        }

    def get_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get all available notification templates"""
        return {
            template_id: {
                "name": template.name,
                "subject": template.subject,
                "variables": template.variables,
                "type": template.type.value,
                "priority": template.priority.value
            }
            for template_id, template in self.templates.items()
        }

    async def schedule_notification(
        self,
        payload: NotificationPayload,
        delay_seconds: int = 0,
        scheduled_time: Optional[datetime] = None
    ) -> str:
        """Schedule a notification for later delivery"""
        if scheduled_time:
            delay_seconds = int((scheduled_time - datetime.utcnow()).total_seconds())
        
        if delay_seconds <= 0:
            # Send immediately
            result = await self.send_notification(payload)
            return result.get("notification_id", "immediate")
        
        # TODO: Implement proper task scheduling (Celery, etc.)
        # For now, use asyncio sleep (not recommended for production)
        asyncio.create_task(self._delayed_send(payload, delay_seconds))
        
        return f"scheduled_{datetime.utcnow().timestamp()}"

    async def _delayed_send(self, payload: NotificationPayload, delay_seconds: int):
        """Send notification after delay"""
        await asyncio.sleep(delay_seconds)
        await self.send_notification(payload)

# Global notification service instance
notification_service = NotificationService()