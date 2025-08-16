"""
Webhook Management Service
Advanced webhook system with event delivery, retries, and monitoring
"""
import asyncio
import json
import hmac
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Set
import uuid
import logging
from urllib.parse import urlparse

import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_, func, desc
from sqlalchemy.orm import selectinload
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.models.webhook import Webhook, WebhookDelivery, WebhookEvent, WebhookRateLimit
from app.models.user import User
from app.core.celery_app import celery_app
from app.services.websocket_manager import ConnectionManager

logger = logging.getLogger(__name__)


class WebhookEvents:
    """Webhook event type constants"""

    VIDEO_GENERATED = "video.generated"
    VIDEO_PUBLISHED = "video.published"
    VIDEO_FAILED = "video.failed"
    CHANNEL_CREATED = "channel.created"
    CHANNEL_UPDATED = "channel.updated"
    CHANNEL_DELETED = "channel.deleted"
    QUOTA_WARNING = "quota.warning"
    COST_THRESHOLD = "cost.threshold"
    SUBSCRIPTION_UPDATED = "subscription.updated"
    ANALYTICS_REPORT = "analytics.report"
    QUEUE_COMPLETED = "queue.completed"
    PAYMENT_SUCCESS = "payment.success"
    PAYMENT_FAILED = "payment.failed"


class WebhookStatus:
    """Webhook status constants"""

    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"
    SUSPENDED = "suspended"
    RATE_LIMITED = "rate_limited"


class WebhookService:
    """
    Advanced webhook management service
    """

    def __init__(self, websocket_manager: Optional[ConnectionManager] = None):
        self.websocket_manager = websocket_manager
        self.client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )
        self._event_processors = {}
        self._register_event_processors()

    def _register_event_processors(self):
        """Register event processors for different event types"""
        self._event_processors = {
            WebhookEvents.VIDEO_GENERATED: self._process_video_event,
            WebhookEvents.VIDEO_PUBLISHED: self._process_video_event,
            WebhookEvents.VIDEO_FAILED: self._process_video_event,
            WebhookEvents.CHANNEL_CREATED: self._process_channel_event,
            WebhookEvents.CHANNEL_UPDATED: self._process_channel_event,
            WebhookEvents.CHANNEL_DELETED: self._process_channel_event,
            WebhookEvents.QUOTA_WARNING: self._process_system_event,
            WebhookEvents.COST_THRESHOLD: self._process_system_event,
            WebhookEvents.ANALYTICS_REPORT: self._process_analytics_event,
        }

    async def create_webhook(
        self,
        db: AsyncSession,
        user_id: str,
        name: str,
        url: str,
        events: List[str],
        secret: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        active: bool = True,
        retry_policy: Optional[Dict[str, Any]] = None,
    ) -> Webhook:
        """
        Create a new webhook endpoint
        """
        try:
            # Generate secret if not provided
            if not secret:
                secret = self._generate_secret()

            # Validate URL
            self._validate_webhook_url(url)

            # Create webhook
            webhook = Webhook(
                user_id=user_id,
                name=name,
                url=url,
                secret=secret,
                events=events,
                headers=headers or {},
                active=active,
                status=WebhookStatus.ACTIVE if active else WebhookStatus.INACTIVE,
                retry_policy=retry_policy or {"max_attempts": 3, "backoff_seconds": 60},
            )

            db.add(webhook)
            await db.commit()
            await db.refresh(webhook)

            # Send verification webhook
            if active:
                await self._send_verification_webhook(webhook)

            logger.info(f"Created webhook {webhook.id} for user {user_id}")
            return webhook

        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to create webhook: {e}")
            raise

    async def update_webhook(
        self, db: AsyncSession, webhook_id: uuid.UUID, user_id: str, **updates
    ) -> Webhook:
        """
        Update webhook configuration
        """
        result = await db.execute(
            select(Webhook).where(
                and_(Webhook.id == webhook_id, Webhook.user_id == user_id)
            )
        )
        webhook = result.scalar_one_or_none()

        if not webhook:
            raise ValueError("Webhook not found")

        # Update fields
        for field, value in updates.items():
            if hasattr(webhook, field):
                setattr(webhook, field, value)

        webhook.updated_at = datetime.utcnow()

        # Validate URL if updated
        if "url" in updates:
            self._validate_webhook_url(updates["url"])

        await db.commit()
        await db.refresh(webhook)

        return webhook

    async def delete_webhook(
        self, db: AsyncSession, webhook_id: uuid.UUID, user_id: str
    ) -> bool:
        """
        Delete webhook
        """
        try:
            result = await db.execute(
                delete(Webhook)
                .where(and_(Webhook.id == webhook_id, Webhook.user_id == user_id))
                .returning(Webhook.id)
            )

            deleted = result.fetchone()
            await db.commit()

            if deleted:
                # Clean up related deliveries (optional - might want to keep for audit)
                await db.execute(
                    delete(WebhookDelivery).where(
                        WebhookDelivery.webhook_id == webhook_id
                    )
                )
                await db.commit()

            return bool(deleted)

        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to delete webhook: {e}")
            return False

    async def get_webhooks(
        self,
        db: AsyncSession,
        user_id: str,
        active_only: bool = False,
        event_type: Optional[str] = None,
    ) -> List[Webhook]:
        """
        Get user's webhooks with filtering
        """
        query = select(Webhook).where(Webhook.user_id == user_id)

        if active_only:
            query = query.where(Webhook.active == True)
        if event_type:
            query = query.where(Webhook.events.contains([event_type]))

        query = query.order_by(desc(Webhook.created_at))

        result = await db.execute(query)
        return result.scalars().all()

    async def trigger_event(
        self,
        db: AsyncSession,
        event_type: str,
        source_type: str,
        source_id: str,
        user_id: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Trigger webhook event for all subscribed endpoints
        """
        try:
            # Create event ID for deduplication
            event_id = f"{event_type}:{source_type}:{source_id}:{int(datetime.utcnow().timestamp())}"

            # Check if event already processed
            existing = await db.execute(
                select(WebhookEvent).where(WebhookEvent.event_id == event_id)
            )
            if existing.scalar_one_or_none():
                logger.info(f"Event {event_id} already processed")
                return event_id

            # Get webhooks subscribed to this event
            webhooks_result = await db.execute(
                select(Webhook).where(
                    and_(
                        Webhook.user_id == user_id,
                        Webhook.active == True,
                        Webhook.events.contains([event_type]),
                    )
                )
            )
            webhooks = webhooks_result.scalars().all()

            # Create event record
            webhook_event = WebhookEvent(
                event_id=event_id,
                event_type=event_type,
                source_type=source_type,
                source_id=source_id,
                user_id=user_id,
                data=data,
                metadata=metadata or {},
                webhook_count=len(webhooks),
            )

            db.add(webhook_event)
            await db.commit()
            await db.refresh(webhook_event)

            # Process event data
            processed_data = await self._process_event_data(event_type, data)

            # Queue webhook deliveries
            for webhook in webhooks:
                await self._queue_webhook_delivery(
                    db=db,
                    webhook=webhook,
                    event_id=event_id,
                    event_type=event_type,
                    payload=processed_data,
                )

            logger.info(f"Triggered event {event_id} for {len(webhooks)} webhooks")
            return event_id

        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to trigger event: {e}")
            raise

    async def deliver_webhook(self, db: AsyncSession, delivery_id: uuid.UUID) -> bool:
        """
        Deliver webhook (called by background task)
        """
        try:
            # Get delivery record
            result = await db.execute(
                select(WebhookDelivery, Webhook)
                .join(Webhook, WebhookDelivery.webhook_id == Webhook.id)
                .where(WebhookDelivery.id == delivery_id)
            )
            row = result.first()

            if not row:
                logger.error(f"Delivery {delivery_id} not found")
                return False

            delivery, webhook = row

            # Check rate limiting
            if not await self._check_rate_limit(db, webhook.id):
                # Reschedule delivery
                await self._reschedule_delivery(db, delivery, "Rate limited")
                return False

            # Prepare payload
            payload = {
                "event": {
                    "id": delivery.event_id,
                    "type": delivery.event_type,
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": delivery.payload,
                },
                "webhook": {"id": str(webhook.id), "name": webhook.name},
            }

            # Generate signature
            signature = self._generate_signature(json.dumps(payload), webhook.secret)

            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "YTEmpire-Webhook/1.0",
                "X-Webhook-Event": delivery.event_type,
                "X-Webhook-ID": str(webhook.id),
                "X-Webhook-Delivery": str(delivery.id),
                "X-Webhook-Signature": signature,
                **webhook.headers,
            }

            # Make request
            start_time = datetime.utcnow()

            try:
                response = await self.client.post(
                    webhook.url, json=payload, headers=headers, timeout=30.0
                )

                duration_ms = int(
                    (datetime.utcnow() - start_time).total_seconds() * 1000
                )

                # Update delivery record
                success = response.status_code < 400

                delivery.status_code = response.status_code
                delivery.response_headers = dict(response.headers)
                delivery.response_body = response.text[:10000]  # Limit response size
                delivery.duration_ms = duration_ms
                delivery.success = success
                delivery.delivered_at = datetime.utcnow()
                delivery.request_headers = headers
                delivery.signature = signature

                # Update webhook stats
                if success:
                    webhook.success_count += 1
                    webhook.last_status_code = response.status_code
                    webhook.last_triggered_at = datetime.utcnow()
                else:
                    webhook.failure_count += 1
                    webhook.last_status_code = response.status_code
                    webhook.last_error_message = (
                        f"HTTP {response.status_code}: {response.text[:500]}"
                    )

                    # Schedule retry if needed
                    if delivery.attempt_number < delivery.max_attempts:
                        await self._reschedule_delivery(
                            db, delivery, f"HTTP {response.status_code}"
                        )

                await db.commit()

                logger.info(
                    f"Delivered webhook {delivery.id} with status {response.status_code}"
                )
                return success

            except httpx.RequestError as e:
                # Network/request error
                duration_ms = int(
                    (datetime.utcnow() - start_time).total_seconds() * 1000
                )

                delivery.error_message = str(e)
                delivery.duration_ms = duration_ms
                delivery.delivered_at = datetime.utcnow()
                delivery.request_headers = headers
                delivery.signature = signature

                webhook.failure_count += 1
                webhook.last_error_message = str(e)

                # Schedule retry
                if delivery.attempt_number < delivery.max_attempts:
                    await self._reschedule_delivery(db, delivery, str(e))

                await db.commit()

                logger.error(f"Failed to deliver webhook {delivery.id}: {e}")
                return False

        except Exception as e:
            logger.error(f"Error delivering webhook {delivery_id}: {e}")
            return False

    async def get_delivery_history(
        self,
        db: AsyncSession,
        webhook_id: uuid.UUID,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
        event_type: Optional[str] = None,
        success_only: bool = False,
    ) -> List[WebhookDelivery]:
        """
        Get webhook delivery history
        """
        # Verify webhook ownership
        webhook_result = await db.execute(
            select(Webhook).where(
                and_(Webhook.id == webhook_id, Webhook.user_id == user_id)
            )
        )
        webhook = webhook_result.scalar_one_or_none()

        if not webhook:
            raise ValueError("Webhook not found")

        # Get deliveries
        query = select(WebhookDelivery).where(WebhookDelivery.webhook_id == webhook_id)

        if event_type:
            query = query.where(WebhookDelivery.event_type == event_type)
        if success_only:
            query = query.where(WebhookDelivery.success == True)

        query = query.order_by(desc(WebhookDelivery.created_at))
        query = query.offset(offset).limit(limit)

        result = await db.execute(query)
        return result.scalars().all()

    async def get_webhook_stats(
        self, db: AsyncSession, user_id: str, days: int = 30
    ) -> Dict[str, Any]:
        """
        Get webhook statistics
        """
        since = datetime.utcnow() - timedelta(days=days)

        # Get webhook counts
        webhook_count = await db.execute(
            select(func.count(Webhook.id)).where(Webhook.user_id == user_id)
        )

        # Get delivery stats
        delivery_stats = await db.execute(
            select(
                func.count(WebhookDelivery.id).label("total"),
                func.count(WebhookDelivery.id)
                .filter(WebhookDelivery.success == True)
                .label("successful"),
                func.count(WebhookDelivery.id)
                .filter(WebhookDelivery.success == False)
                .label("failed"),
                func.avg(WebhookDelivery.duration_ms).label("avg_duration"),
            )
            .join(Webhook, WebhookDelivery.webhook_id == Webhook.id)
            .where(
                and_(Webhook.user_id == user_id, WebhookDelivery.created_at >= since)
            )
        )

        stats_row = delivery_stats.first()

        return {
            "webhook_count": webhook_count.scalar() or 0,
            "delivery_stats": {
                "total_deliveries": stats_row.total or 0,
                "successful_deliveries": stats_row.successful or 0,
                "failed_deliveries": stats_row.failed or 0,
                "success_rate": (stats_row.successful / stats_row.total * 100)
                if stats_row.total > 0
                else 0,
                "avg_duration_ms": stats_row.avg_duration or 0,
            },
            "period_days": days,
        }

    # Private helper methods
    def _generate_secret(self) -> str:
        """Generate webhook secret"""
        return secrets.token_urlsafe(32)

    def _validate_webhook_url(self, url: str):
        """Validate webhook URL"""
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Invalid webhook URL")
        if parsed.scheme not in ["http", "https"]:
            raise ValueError("Webhook URL must use HTTP or HTTPS")
        # Add more validation as needed

    def _generate_signature(self, payload: str, secret: str) -> str:
        """Generate HMAC signature for webhook"""
        if not secret:
            return ""
        return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

    async def _send_verification_webhook(self, webhook: Webhook):
        """Send verification webhook"""
        payload = {
            "event": {
                "type": "webhook.verification",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "webhook_id": str(webhook.id),
                    "challenge": secrets.token_urlsafe(16),
                },
            }
        }

        # Queue background delivery
        celery_app.send_task(
            "app.tasks.webhook.send_verification",
            args=[str(webhook.id), payload],
            countdown=5,
        )

    async def _process_event_data(
        self, event_type: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process event data using registered processors"""
        processor = self._event_processors.get(event_type)
        if processor:
            return await processor(data)
        return data

    async def _process_video_event(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process video-related events"""
        # Add additional video context if needed
        return {**data, "timestamp": datetime.utcnow().isoformat()}

    async def _process_channel_event(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process channel-related events"""
        return data

    async def _process_system_event(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process system events"""
        return data

    async def _process_analytics_event(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process analytics events"""
        return data

    async def _queue_webhook_delivery(
        self,
        db: AsyncSession,
        webhook: Webhook,
        event_id: str,
        event_type: str,
        payload: Dict[str, Any],
    ):
        """Queue webhook delivery"""
        delivery = WebhookDelivery(
            webhook_id=webhook.id,
            event_id=event_id,
            event_type=event_type,
            payload=payload,
            max_attempts=webhook.retry_policy.get("max_attempts", 3),
            retry_backoff_seconds=webhook.retry_policy.get("backoff_seconds", 60),
        )

        db.add(delivery)
        await db.commit()
        await db.refresh(delivery)

        # Queue background task
        celery_app.send_task(
            "app.tasks.webhook.deliver_webhook",
            args=[str(delivery.id)],
            countdown=1,  # Small delay for immediate processing
        )

    async def _reschedule_delivery(
        self, db: AsyncSession, delivery: WebhookDelivery, error_message: str
    ):
        """Reschedule failed delivery"""
        delivery.attempt_number += 1
        delivery.error_message = error_message

        if delivery.attempt_number <= delivery.max_attempts:
            # Calculate backoff delay
            backoff = delivery.retry_backoff_seconds * (
                2 ** (delivery.attempt_number - 1)
            )
            delivery.next_retry_at = datetime.utcnow() + timedelta(seconds=backoff)

            # Schedule retry
            celery_app.send_task(
                "app.tasks.webhook.deliver_webhook",
                args=[str(delivery.id)],
                countdown=backoff,
            )

        await db.commit()

    async def _check_rate_limit(self, db: AsyncSession, webhook_id: uuid.UUID) -> bool:
        """Check if webhook is rate limited"""
        # Implementation for rate limiting check
        # For now, return True (no rate limiting)
        return True


# Global instance
webhook_service = WebhookService()
