"""
Webhook Management System API Endpoints
Handles webhook registration, management, and delivery
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime, timedelta
from enum import Enum
import uuid
import hmac
import hashlib
import json
import httpx
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

from app.db.session import get_db
from app.api.v1.endpoints.auth import get_current_verified_user
from app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter()


class WebhookEvent(str, Enum):
    """Webhook event types"""

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


class WebhookStatus(str, Enum):
    """Webhook status"""

    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"
    SUSPENDED = "suspended"


class WebhookRequest(BaseModel):
    """Request to create/update webhook"""

    name: str = Field(..., description="Webhook name")
    url: HttpUrl = Field(..., description="Webhook endpoint URL")
    events: List[WebhookEvent] = Field(..., description="Events to subscribe to")
    secret: Optional[str] = Field(None, description="Secret for HMAC signature")
    headers: Optional[Dict[str, str]] = Field(default={}, description="Custom headers")
    active: bool = Field(default=True, description="Whether webhook is active")
    retry_policy: Optional[Dict[str, Any]] = Field(
        default={"max_attempts": 3, "backoff_seconds": 60},
        description="Retry policy configuration",
    )


class WebhookResponse(BaseModel):
    """Webhook response model"""

    webhook_id: str
    user_id: str
    name: str
    url: str
    events: List[str]
    secret: Optional[str]
    headers: Dict[str, str]
    status: str
    active: bool
    retry_policy: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    last_triggered_at: Optional[datetime]
    last_status_code: Optional[int]
    failure_count: int
    success_count: int


class WebhookDelivery(BaseModel):
    """Webhook delivery record"""

    delivery_id: str
    webhook_id: str
    event: str
    payload: Dict[str, Any]
    status_code: Optional[int]
    response_body: Optional[str]
    delivered_at: datetime
    duration_ms: int
    success: bool
    error_message: Optional[str]
    attempt_number: int


class WebhookTestRequest(BaseModel):
    """Request to test webhook"""

    event: WebhookEvent
    sample_data: Optional[Dict[str, Any]] = None


# Import webhook service
from app.services.webhook_service import webhook_service, WebhookEvents, WebhookStatus
from app.models.webhook import Webhook


@router.post("/", response_model=WebhookResponse)
async def create_webhook(
    request: WebhookRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
) -> WebhookResponse:
    """
    Register a new webhook endpoint
    """
    try:
        webhook = await webhook_service.create_webhook(
            db=db,
            user_id=str(current_user.id),
            name=request.name,
            url=str(request.url),
            events=[e.value for e in request.events],
            secret=request.secret,
            headers=request.headers or {},
            active=request.active,
            retry_policy=request.retry_policy,
        )

        return WebhookResponse(
            webhook_id=str(webhook.id),
            user_id=webhook.user_id,
            name=webhook.name,
            url=webhook.url,
            events=webhook.events,
            secret=webhook.secret,
            headers=webhook.headers,
            status=webhook.status,
            active=webhook.active,
            retry_policy=webhook.retry_policy,
            created_at=webhook.created_at,
            updated_at=webhook.updated_at,
            last_triggered_at=webhook.last_triggered_at,
            last_status_code=webhook.last_status_code,
            failure_count=webhook.failure_count,
            success_count=webhook.success_count,
        )

    except Exception as e:
        logger.error(f"Failed to create webhook: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create webhook: {str(e)}",
        )


@router.get("/", response_model=List[WebhookResponse])
async def list_webhooks(
    active_only: bool = False,
    event: Optional[WebhookEvent] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
) -> List[WebhookResponse]:
    """
    List all webhooks for the current user
    """
    webhooks = []

    for webhook in webhooks_storage.values():
        if webhook["user_id"] != str(current_user.id):
            continue
        if active_only and not webhook["active"]:
            continue
        if event and event.value not in webhook["events"]:
            continue

        webhooks.append(WebhookResponse(**webhook))

    return webhooks


@router.get("/{webhook_id}", response_model=WebhookResponse)
async def get_webhook(
    webhook_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
) -> WebhookResponse:
    """
    Get webhook details
    """
    if webhook_id not in webhooks_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Webhook not found"
        )

    webhook = webhooks_storage[webhook_id]

    if webhook["user_id"] != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    return WebhookResponse(**webhook)


@router.put("/{webhook_id}", response_model=WebhookResponse)
async def update_webhook(
    webhook_id: str,
    request: WebhookRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
) -> WebhookResponse:
    """
    Update webhook configuration
    """
    if webhook_id not in webhooks_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Webhook not found"
        )

    webhook = webhooks_storage[webhook_id]

    if webhook["user_id"] != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    # Update webhook
    webhook.update(
        {
            "name": request.name,
            "url": str(request.url),
            "events": [e.value for e in request.events],
            "secret": request.secret or webhook["secret"],
            "headers": request.headers or {},
            "active": request.active,
            "status": WebhookStatus.ACTIVE
            if request.active
            else WebhookStatus.INACTIVE,
            "retry_policy": request.retry_policy or webhook["retry_policy"],
            "updated_at": datetime.utcnow(),
        }
    )

    return WebhookResponse(**webhook)


@router.delete("/{webhook_id}")
async def delete_webhook(
    webhook_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
) -> Dict[str, str]:
    """
    Delete a webhook
    """
    if webhook_id not in webhooks_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Webhook not found"
        )

    webhook = webhooks_storage[webhook_id]

    if webhook["user_id"] != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    del webhooks_storage[webhook_id]

    return {"status": "deleted", "webhook_id": webhook_id}


@router.post("/{webhook_id}/test")
async def test_webhook(
    webhook_id: str,
    test_request: WebhookTestRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
) -> Dict[str, Any]:
    """
    Send a test payload to webhook
    """
    if webhook_id not in webhooks_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Webhook not found"
        )

    webhook = webhooks_storage[webhook_id]

    if webhook["user_id"] != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    # Prepare test payload
    test_payload = test_request.sample_data or generate_sample_payload(
        test_request.event
    )

    # Send webhook
    background_tasks.add_task(
        deliver_webhook, webhook, test_request.event.value, test_payload, is_test=True
    )

    return {
        "status": "test_sent",
        "webhook_id": webhook_id,
        "event": test_request.event.value,
        "url": webhook["url"],
    }


@router.get("/{webhook_id}/deliveries", response_model=List[WebhookDelivery])
async def get_webhook_deliveries(
    webhook_id: str,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
) -> List[WebhookDelivery]:
    """
    Get delivery history for a webhook
    """
    if webhook_id not in webhooks_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Webhook not found"
        )

    webhook = webhooks_storage[webhook_id]

    if webhook["user_id"] != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    # Get deliveries for this webhook
    webhook_deliveries = [
        WebhookDelivery(**d)
        for d in deliveries_storage
        if d["webhook_id"] == webhook_id
    ]

    # Sort by delivered_at descending
    webhook_deliveries.sort(key=lambda x: x.delivered_at, reverse=True)

    return webhook_deliveries[:limit]


@router.post("/{webhook_id}/toggle")
async def toggle_webhook(
    webhook_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
) -> WebhookResponse:
    """
    Toggle webhook active status
    """
    if webhook_id not in webhooks_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Webhook not found"
        )

    webhook = webhooks_storage[webhook_id]

    if webhook["user_id"] != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    # Toggle active status
    webhook["active"] = not webhook["active"]
    webhook["status"] = (
        WebhookStatus.ACTIVE if webhook["active"] else WebhookStatus.INACTIVE
    )
    webhook["updated_at"] = datetime.utcnow()

    return WebhookResponse(**webhook)


@router.post("/trigger/{event}")
async def trigger_event(
    event: WebhookEvent,
    payload: Dict[str, Any],
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
) -> Dict[str, Any]:
    """
    Manually trigger a webhook event (for testing)
    """
    triggered_count = 0

    for webhook in webhooks_storage.values():
        if (
            webhook["user_id"] == str(current_user.id)
            and webhook["active"]
            and event.value in webhook["events"]
        ):
            background_tasks.add_task(deliver_webhook, webhook, event.value, payload)
            triggered_count += 1

    return {
        "status": "triggered",
        "event": event.value,
        "webhooks_triggered": triggered_count,
    }


# Helper functions
def generate_webhook_secret() -> str:
    """Generate a secure webhook secret"""
    import secrets

    return secrets.token_urlsafe(32)


def calculate_signature(secret: str, payload: bytes) -> str:
    """Calculate HMAC signature for webhook payload"""
    return hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


async def deliver_webhook(
    webhook: Dict, event: str, payload: Dict[str, Any], is_test: bool = False
):
    """Deliver webhook with retry logic"""
    delivery_id = str(uuid.uuid4())
    start_time = datetime.utcnow()

    # Prepare payload
    webhook_payload = {
        "webhook_id": webhook["webhook_id"],
        "event": event,
        "timestamp": datetime.utcnow().isoformat(),
        "data": payload,
        "test": is_test,
    }

    payload_bytes = json.dumps(webhook_payload).encode()

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "X-Webhook-Event": event,
        "X-Webhook-ID": webhook["webhook_id"],
        "X-Webhook-Timestamp": str(int(datetime.utcnow().timestamp())),
        **webhook.get("headers", {}),
    }

    # Add signature if secret is configured
    if webhook.get("secret"):
        signature = calculate_signature(webhook["secret"], payload_bytes)
        headers["X-Webhook-Signature"] = f"sha256={signature}"

    # Delivery record
    delivery = {
        "delivery_id": delivery_id,
        "webhook_id": webhook["webhook_id"],
        "event": event,
        "payload": webhook_payload,
        "status_code": None,
        "response_body": None,
        "delivered_at": datetime.utcnow(),
        "duration_ms": 0,
        "success": False,
        "error_message": None,
        "attempt_number": 1,
    }

    # Attempt delivery with retries
    max_attempts = webhook.get("retry_policy", {}).get("max_attempts", 3)
    backoff_seconds = webhook.get("retry_policy", {}).get("backoff_seconds", 60)

    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    webhook["url"], content=payload_bytes, headers=headers
                )

                duration_ms = int(
                    (datetime.utcnow() - start_time).total_seconds() * 1000
                )

                delivery.update(
                    {
                        "status_code": response.status_code,
                        "response_body": response.text[:1000],  # Limit response size
                        "duration_ms": duration_ms,
                        "success": 200 <= response.status_code < 300,
                        "attempt_number": attempt,
                    }
                )

                # Update webhook stats
                webhook["last_triggered_at"] = datetime.utcnow()
                webhook["last_status_code"] = response.status_code

                if delivery["success"]:
                    webhook["success_count"] += 1
                    break
                else:
                    webhook["failure_count"] += 1

                    if attempt < max_attempts:
                        await asyncio.sleep(backoff_seconds * attempt)

        except Exception as e:
            delivery["error_message"] = str(e)
            webhook["failure_count"] += 1

            if attempt < max_attempts:
                await asyncio.sleep(backoff_seconds * attempt)
            else:
                # Mark webhook as failed after max attempts
                if webhook["failure_count"] > 10:
                    webhook["status"] = WebhookStatus.FAILED

    # Store delivery record
    deliveries_storage.append(delivery)

    # Keep only last 1000 deliveries
    if len(deliveries_storage) > 1000:
        deliveries_storage.pop(0)


async def send_test_webhook(webhook_id: str, url: str, secret: str):
    """Send test webhook to verify endpoint"""
    test_payload = {
        "webhook_id": webhook_id,
        "event": "webhook.test",
        "timestamp": datetime.utcnow().isoformat(),
        "data": {
            "message": "This is a test webhook from YTEmpire",
            "webhook_id": webhook_id,
        },
        "test": True,
    }

    payload_bytes = json.dumps(test_payload).encode()

    headers = {
        "Content-Type": "application/json",
        "X-Webhook-Event": "webhook.test",
        "X-Webhook-ID": webhook_id,
    }

    if secret:
        signature = calculate_signature(secret, payload_bytes)
        headers["X-Webhook-Signature"] = f"sha256={signature}"

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(url, content=payload_bytes, headers=headers)
    except Exception as e:
        logger.error(f"Test webhook failed: {e}")


def generate_sample_payload(event: WebhookEvent) -> Dict[str, Any]:
    """Generate sample payload for webhook testing"""
    sample_payloads = {
        WebhookEvent.VIDEO_GENERATED: {
            "video_id": "sample_video_123",
            "title": "Sample Video Title",
            "channel_id": "channel_456",
            "duration": 300,
            "cost": 0.45,
        },
        WebhookEvent.VIDEO_PUBLISHED: {
            "video_id": "sample_video_123",
            "youtube_id": "dQw4w9WgXcQ",
            "url": "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "published_at": datetime.utcnow().isoformat(),
        },
        WebhookEvent.VIDEO_FAILED: {
            "video_id": "sample_video_123",
            "error": "Generation failed",
            "error_code": "GEN_001",
            "timestamp": datetime.utcnow().isoformat(),
        },
        WebhookEvent.CHANNEL_CREATED: {
            "channel_id": "channel_789",
            "name": "Sample Channel",
            "created_at": datetime.utcnow().isoformat(),
        },
        WebhookEvent.QUOTA_WARNING: {
            "service": "youtube",
            "usage_percentage": 85,
            "quota_remaining": 1500,
            "reset_time": (datetime.utcnow() + timedelta(hours=6)).isoformat(),
        },
        WebhookEvent.COST_THRESHOLD: {
            "current_cost": 45.67,
            "threshold": 50.00,
            "period": "daily",
            "timestamp": datetime.utcnow().isoformat(),
        },
    }

    return sample_payloads.get(
        event, {"event": event.value, "timestamp": datetime.utcnow().isoformat()}
    )
