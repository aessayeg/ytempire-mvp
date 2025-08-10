# YTEMPIRE Webhook Implementation & Error Handling Standards
**Version 1.0 | January 2025**  
**Owner: API Development Engineer**  
**Status: Implementation Standard**

---

## 1. Webhook Implementation Guide

### 1.1 Webhook Architecture

```python
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import httpx
import hmac
import hashlib
import json
import asyncio
from sqlalchemy import Column, String, JSON, DateTime, Boolean, Integer
from pydantic import BaseModel, HttpUrl, validator

class WebhookEvent(BaseModel):
    """Webhook event structure"""
    
    id: str
    type: str  # e.g., "video.published", "channel.suspended"
    created_at: datetime
    data: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "id": "evt_1234567890abcdef",
                "type": "video.published",
                "created_at": "2025-01-15T10:30:00Z",
                "data": {
                    "video": {
                        "id": "vid_0987654321fedcba",
                        "title": "iPhone 15 Review",
                        "channel_id": "ch_1234567890abcdef",
                        "youtube_video_id": "dQw4w9WgXcQ"
                    }
                }
            }
        }

class WebhookSubscription(BaseModel):
    """Webhook subscription configuration"""
    
    url: HttpUrl
    events: List[str]
    secret: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    is_active: bool = True
    
    @validator('events')
    def validate_events(cls, v):
        valid_events = {
            "video.created",
            "video.published",
            "video.failed",
            "video.deleted",
            "channel.created",
            "channel.updated",
            "channel.suspended",
            "channel.deleted",
            "analytics.daily_report",
            "subscription.updated",
            "subscription.cancelled"
        }
        
        for event in v:
            if event not in valid_events:
                raise ValueError(f"Invalid event type: {event}")
        
        return v

class WebhookDelivery(Base):
    """Track webhook delivery attempts"""
    
    __tablename__ = "webhook_deliveries"
    
    id = Column(String, primary_key=True)
    webhook_id = Column(String, ForeignKey("webhooks.id"))
    event_id = Column(String)
    event_type = Column(String)
    url = Column(String)
    request_headers = Column(JSON)
    request_body = Column(JSON)
    response_status = Column(Integer)
    response_headers = Column(JSON)
    response_body = Column(String)
    attempted_at = Column(DateTime)
    succeeded_at = Column(DateTime, nullable=True)
    failed_at = Column(DateTime, nullable=True)
    retry_count = Column(Integer, default=0)
    next_retry_at = Column(DateTime, nullable=True)
    error_message = Column(String, nullable=True)
```

### 1.2 Webhook Service Implementation

```python
class WebhookService:
    """Core webhook delivery service"""
    
    def __init__(self, db: Session, http_client: httpx.AsyncClient):
        self.db = db
        self.http_client = http_client
        self.max_retries = 5
        self.timeout = 30  # seconds
        
    async def trigger_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        user_id: Optional[str] = None
    ):
        """Trigger webhook event for all subscribers"""
        
        # Create event
        event = WebhookEvent(
            id=f"evt_{generate_unique_id()}",
            type=event_type,
            created_at=datetime.utcnow(),
            data=data
        )
        
        # Find active subscriptions for this event
        query = self.db.query(WebhookSubscription).filter(
            WebhookSubscription.is_active == True,
            WebhookSubscription.events.contains([event_type])
        )
        
        if user_id:
            query = query.filter(WebhookSubscription.user_id == user_id)
        
        subscriptions = query.all()
        
        # Queue deliveries
        for subscription in subscriptions:
            await self.queue_delivery(subscription, event)
    
    async def queue_delivery(
        self,
        subscription: WebhookSubscription,
        event: WebhookEvent
    ):
        """Queue webhook delivery for processing"""
        
        delivery = WebhookDelivery(
            id=f"del_{generate_unique_id()}",
            webhook_id=subscription.id,
            event_id=event.id,
            event_type=event.type,
            url=subscription.url,
            request_headers=self._build_headers(subscription, event),
            request_body=self._build_payload(subscription, event),
            attempted_at=datetime.utcnow(),
            retry_count=0
        )
        
        self.db.add(delivery)
        self.db.commit()
        
        # Process delivery asynchronously
        asyncio.create_task(self.deliver_webhook(delivery.id))
    
    async def deliver_webhook(self, delivery_id: str):
        """Deliver webhook with retry logic"""
        
        delivery = self.db.query(WebhookDelivery).filter(
            WebhookDelivery.id == delivery_id
        ).first()
        
        if not delivery:
            return
        
        try:
            # Make HTTP request
            response = await self.http_client.post(
                delivery.url,
                json=delivery.request_body,
                headers=delivery.request_headers,
                timeout=self.timeout
            )
            
            # Update delivery record
            delivery.response_status = response.status_code
            delivery.response_headers = dict(response.headers)
            delivery.response_body = response.text[:1000]  # Store first 1000 chars
            
            if 200 <= response.status_code < 300:
                # Success
                delivery.succeeded_at = datetime.utcnow()
                self.db.commit()
                
                # Log success
                await self._log_delivery(delivery, "success")
            else:
                # HTTP error - schedule retry
                await self._handle_delivery_failure(delivery, f"HTTP {response.status_code}")
                
        except httpx.TimeoutException:
            await self._handle_delivery_failure(delivery, "Request timeout")
        except httpx.RequestError as e:
            await self._handle_delivery_failure(delivery, str(e))
        except Exception as e:
            await self._handle_delivery_failure(delivery, f"Unexpected error: {str(e)}")
    
    async def _handle_delivery_failure(
        self,
        delivery: WebhookDelivery,
        error_message: str
    ):
        """Handle failed webhook delivery"""
        
        delivery.error_message = error_message
        delivery.retry_count += 1
        
        if delivery.retry_count >= self.max_retries:
            # Max retries reached
            delivery.failed_at = datetime.utcnow()
            self.db.commit()
            
            # Notify webhook owner
            await self._notify_webhook_failure(delivery)
            
            # Log final failure
            await self._log_delivery(delivery, "failed")
        else:
            # Schedule retry with exponential backoff
            backoff_seconds = min(300, 2 ** delivery.retry_count)  # Max 5 minutes
            delivery.next_retry_at = datetime.utcnow() + timedelta(seconds=backoff_seconds)
            self.db.commit()
            
            # Schedule retry
            asyncio.create_task(self._retry_delivery(delivery.id, backoff_seconds))
    
    async def _retry_delivery(self, delivery_id: str, delay: int):
        """Retry webhook delivery after delay"""
        
        await asyncio.sleep(delay)
        await self.deliver_webhook(delivery_id)
    
    def _build_headers(
        self,
        subscription: WebhookSubscription,
        event: WebhookEvent
    ) -> Dict[str, str]:
        """Build webhook request headers"""
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "YTEMPIRE-Webhook/1.0",
            "X-YTEMPIRE-Event": event.type,
            "X-YTEMPIRE-Event-ID": event.id,
            "X-YTEMPIRE-Delivery-ID": generate_unique_id(),
            "X-YTEMPIRE-Timestamp": str(int(event.created_at.timestamp()))
        }
        
        # Add custom headers
        if subscription.headers:
            headers.update(subscription.headers)
        
        # Add signature if secret configured
        if subscription.secret:
            signature = self._generate_signature(
                subscription.secret,
                event.json()
            )
            headers["X-YTEMPIRE-Signature"] = signature
        
        return headers
    
    def _build_payload(
        self,
        subscription: WebhookSubscription,
        event: WebhookEvent
    ) -> Dict[str, Any]:
        """Build webhook payload"""
        
        return {
            "id": event.id,
            "type": event.type,
            "created_at": event.created_at.isoformat(),
            "data": event.data
        }
    
    def _generate_signature(self, secret: str, payload: str) -> str:
        """Generate HMAC signature for webhook"""
        
        signature = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return f"sha256={signature}"
    
    async def _log_delivery(
        self,
        delivery: WebhookDelivery,
        status: str
    ):
        """Log webhook delivery for monitoring"""
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "delivery_id": delivery.id,
            "webhook_id": delivery.webhook_id,
            "event_type": delivery.event_type,
            "url": delivery.url,
            "status": status,
            "response_status": delivery.response_status,
            "retry_count": delivery.retry_count,
            "error_message": delivery.error_message
        }
        
        # Send to monitoring system
        await send_to_monitoring("webhook_delivery", log_entry)
```

### 1.3 Webhook Endpoints

```python
@router.post("/webhooks", response_model=WebhookResponse)
async def create_webhook(
    request: WebhookCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create webhook subscription"""
    
    # Validate URL is reachable
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(request.url, timeout=10)
            if response.status_code >= 500:
                raise HTTPException(
                    status_code=400,
                    detail="Webhook URL is not reachable"
                )
    except:
        raise HTTPException(
            status_code=400,
            detail="Webhook URL is not reachable"
        )
    
    # Generate webhook secret
    secret = secrets.token_urlsafe(32)
    
    # Create webhook
    webhook = Webhook(
        id=f"wh_{generate_unique_id()}",
        user_id=current_user.id,
        url=request.url,
        events=request.events,
        secret=secret,
        headers=request.headers,
        is_active=True,
        created_at=datetime.utcnow()
    )
    
    db.add(webhook)
    db.commit()
    
    return {
        "data": {
            "type": "webhook",
            "id": webhook.id,
            "attributes": {
                "url": webhook.url,
                "events": webhook.events,
                "secret": webhook.secret,
                "is_active": webhook.is_active,
                "created_at": webhook.created_at
            }
        }
    }

@router.post("/webhooks/test/{webhook_id}")
async def test_webhook(
    webhook_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    webhook_service: WebhookService = Depends(get_webhook_service)
):
    """Send test event to webhook"""
    
    webhook = db.query(Webhook).filter(
        Webhook.id == webhook_id,
        Webhook.user_id == current_user.id
    ).first()
    
    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    # Create test event
    test_event = WebhookEvent(
        id=f"evt_test_{generate_unique_id()}",
        type="test.webhook",
        created_at=datetime.utcnow(),
        data={
            "message": "This is a test webhook event",
            "webhook_id": webhook_id
        }
    )
    
    # Deliver test webhook
    await webhook_service.queue_delivery(webhook, test_event)
    
    return {
        "message": "Test webhook sent",
        "event_id": test_event.id
    }

@router.get("/webhooks/{webhook_id}/deliveries")
async def get_webhook_deliveries(
    webhook_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100)
):
    """Get webhook delivery history"""
    
    # Verify ownership
    webhook = db.query(Webhook).filter(
        Webhook.id == webhook_id,
        Webhook.user_id == current_user.id
    ).first()
    
    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    # Get deliveries
    deliveries = db.query(WebhookDelivery).filter(
        WebhookDelivery.webhook_id == webhook_id
    ).order_by(
        WebhookDelivery.attempted_at.desc()
    ).offset(
        (page - 1) * per_page
    ).limit(per_page).all()
    
    return {
        "data": [
            {
                "id": d.id,
                "event_type": d.event_type,
                "attempted_at": d.attempted_at,
                "succeeded_at": d.succeeded_at,
                "failed_at": d.failed_at,
                "response_status": d.response_status,
                "retry_count": d.retry_count,
                "error_message": d.error_message
            }
            for d in deliveries
        ]
    }
```

### 1.4 Webhook Security

```python
class WebhookSecurityService:
    """Webhook security and validation"""
    
    @staticmethod
    def verify_webhook_signature(
        payload: bytes,
        signature_header: str,
        secret: str
    ) -> bool:
        """Verify webhook signature from YTEMPIRE"""
        
        if not signature_header or not signature_header.startswith("sha256="):
            return False
        
        expected_signature = hmac.new(
            secret.encode('utf-8'),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        provided_signature = signature_header[7:]  # Remove "sha256=" prefix
        
        # Use constant-time comparison to prevent timing attacks
        return hmac.compare_digest(expected_signature, provided_signature)
    
    @staticmethod
    def validate_webhook_payload(payload: Dict) -> bool:
        """Validate webhook payload structure"""
        
        required_fields = ["id", "type", "created_at", "data"]
        
        for field in required_fields:
            if field not in payload:
                return False
        
        # Validate timestamp is recent (within 5 minutes)
        try:
            created_at = datetime.fromisoformat(payload["created_at"])
            age = datetime.utcnow() - created_at
            if age.total_seconds() > 300:  # 5 minutes
                return False
        except:
            return False
        
        return True

# Example webhook receiver (customer's endpoint)
@app.post("/receive-webhook")
async def receive_webhook(
    request: Request,
    x_ytempire_signature: str = Header(None)
):
    """Example webhook receiver implementation"""
    
    # Get raw payload
    payload = await request.body()
    
    # Verify signature
    webhook_secret = "your_webhook_secret_here"
    if not WebhookSecurityService.verify_webhook_signature(
        payload, x_ytempire_signature, webhook_secret
    ):
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    # Parse and validate payload
    data = json.loads(payload)
    if not WebhookSecurityService.validate_webhook_payload(data):
        raise HTTPException(status_code=400, detail="Invalid payload")
    
    # Process webhook based on event type
    event_type = data["type"]
    event_data = data["data"]
    
    if event_type == "video.published":
        # Handle video published event
        await handle_video_published(event_data)
    elif event_type == "channel.suspended":
        # Handle channel suspended event
        await handle_channel_suspended(event_data)
    
    # Return success
    return {"status": "success"}
```

---

## 2. Error Handling Standards

### 2.1 Error Response Structure

```python
from typing import Optional, List, Dict, Any
from enum import Enum

class ErrorCode(Enum):
    """Standardized error codes"""
    
    # Authentication errors (1xxx)
    UNAUTHORIZED = "1001"
    INVALID_TOKEN = "1002"
    TOKEN_EXPIRED = "1003"
    INVALID_CREDENTIALS = "1004"
    
    # Authorization errors (2xxx)
    FORBIDDEN = "2001"
    INSUFFICIENT_PERMISSIONS = "2002"
    RESOURCE_ACCESS_DENIED = "2003"
    
    # Validation errors (3xxx)
    VALIDATION_ERROR = "3001"
    INVALID_INPUT = "3002"
    MISSING_REQUIRED_FIELD = "3003"
    INVALID_FORMAT = "3004"
    
    # Resource errors (4xxx)
    NOT_FOUND = "4001"
    ALREADY_EXISTS = "4002"
    CONFLICT = "4003"
    GONE = "4004"
    
    # Rate limiting errors (5xxx)
    RATE_LIMIT_EXCEEDED = "5001"
    QUOTA_EXCEEDED = "5002"
    
    # Server errors (6xxx)
    INTERNAL_ERROR = "6001"
    SERVICE_UNAVAILABLE = "6002"
    EXTERNAL_SERVICE_ERROR = "6003"
    DATABASE_ERROR = "6004"
    
    # Business logic errors (7xxx)
    CHANNEL_LIMIT_REACHED = "7001"
    VIDEO_GENERATION_FAILED = "7002"
    INSUFFICIENT_CREDITS = "7003"
    SUBSCRIPTION_REQUIRED = "7004"

class ErrorDetail(BaseModel):
    """Detailed error information"""
    
    field: Optional[str] = None
    code: str
    message: str
    context: Optional[Dict[str, Any]] = None

class ErrorResponse(BaseModel):
    """Standardized error response"""
    
    error: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "error": {
                    "id": "err_1234567890abcdef",
                    "status": 422,
                    "code": "3001",
                    "title": "Validation Error",
                    "detail": "The request contains invalid data",
                    "source": {
                        "pointer": "/data/attributes/name",
                        "parameter": "name"
                    },
                    "meta": {
                        "timestamp": "2025-01-15T10:30:00Z",
                        "request_id": "req_1234567890abcdef",
                        "validation_errors": [
                            {
                                "field": "name",
                                "code": "too_short",
                                "message": "Channel name must be at least 3 characters"
                            }
                        ]
                    }
                }
            }
        }
```

### 2.2 Exception Handlers

```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import traceback
import logging

logger = logging.getLogger(__name__)

class APIException(Exception):
    """Base API exception"""
    
    def __init__(
        self,
        status_code: int,
        error_code: ErrorCode,
        title: str,
        detail: str,
        source: Optional[Dict] = None,
        meta: Optional[Dict] = None
    ):
        self.status_code = status_code
        self.error_code = error_code
        self.title = title
        self.detail = detail
        self.source = source
        self.meta = meta or {}

class ValidationException(APIException):
    """Validation error exception"""
    
    def __init__(self, errors: List[ErrorDetail]):
        super().__init__(
            status_code=422,
            error_code=ErrorCode.VALIDATION_ERROR,
            title="Validation Error",
            detail="The request contains invalid data",
            meta={"validation_errors": [e.dict() for e in errors]}
        )

class NotFoundException(APIException):
    """Resource not found exception"""
    
    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            status_code=404,
            error_code=ErrorCode.NOT_FOUND,
            title="Resource Not Found",
            detail=f"{resource_type} with id '{resource_id}' not found",
            source={"parameter": "id"}
        )

class RateLimitException(APIException):
    """Rate limit exceeded exception"""
    
    def __init__(self, limit: int, window: int, retry_after: int):
        super().__init__(
            status_code=429,
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            title="Rate Limit Exceeded",
            detail="Too many requests. Please retry after some time.",
            meta={
                "limit": limit,
                "window": window,
                "retry_after": retry_after
            }
        )

def create_error_response(
    error_id: str,
    status_code: int,
    error_code: str,
    title: str,
    detail: str,
    source: Optional[Dict] = None,
    meta: Optional[Dict] = None
) -> Dict:
    """Create standardized error response"""
    
    error_response = {
        "error": {
            "id": error_id,
            "status": status_code,
            "code": error_code,
            "title": title,
            "detail": detail
        }
    }
    
    if source:
        error_response["error"]["source"] = source
    
    if meta:
        error_response["error"]["meta"] = meta
    
    return error_response

# Exception handlers
async def api_exception_handler(request: Request, exc: APIException):
    """Handle API exceptions"""
    
    error_id = f"err_{generate_unique_id()}"
    
    # Log error
    logger.error(
        f"API Exception: {exc.title}",
        extra={
            "error_id": error_id,
            "status_code": exc.status_code,
            "error_code": exc.error_code.value,
            "detail": exc.detail,
            "path": request.url.path,
            "method": request.method
        }
    )
    
    # Add request context to meta
    exc.meta["timestamp"] = datetime.utcnow().isoformat()
    exc.meta["request_id"] = request.headers.get("X-Request-ID", "unknown")
    
    response = create_error_response(
        error_id=error_id,
        status_code=exc.status_code,
        error_code=exc.error_code.value,
        title=exc.title,
        detail=exc.detail,
        source=exc.source,
        meta=exc.meta
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation exceptions"""
    
    error_id = f"err_{generate_unique_id()}"
    
    # Parse validation errors
    validation_errors = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"])
        validation_errors.append({
            "field": field,
            "code": error["type"],
            "message": error["msg"]
        })
    
    response = create_error_response(
        error_id=error_id,
        status_code=422,
        error_code=ErrorCode.VALIDATION_ERROR.value,
        title="Validation Error",
        detail="The request contains invalid data",
        meta={
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request.headers.get("X-Request-ID", "unknown"),
            "validation_errors": validation_errors
        }
    )
    
    return JSONResponse(
        status_code=422,
        content=response
    )

async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    
    error_id = f"err_{generate_unique_id()}"
    
    # Map status codes to error codes
    error_code_map = {
        401: ErrorCode.UNAUTHORIZED,
        403: ErrorCode.FORBIDDEN,
        404: ErrorCode.NOT_FOUND,
        409: ErrorCode.CONFLICT,
        429: ErrorCode.RATE_LIMIT_EXCEEDED
    }
    
    error_code = error_code_map.get(exc.status_code, ErrorCode.INTERNAL_ERROR)
    
    response = create_error_response(
        error_id=error_id,
        status_code=exc.status_code,
        error_code=error_code.value,
        title=exc.detail,
        detail=exc.detail,
        meta={
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request.headers.get("X-Request-ID", "unknown")
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response,
        headers=exc.headers
    )

async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    
    error_id = f"err_{generate_unique_id()}"
    
    # Log full exception
    logger.exception(
        "Unhandled exception",
        extra={
            "error_id": error_id,
            "path": request.url.path,
            "method": request.method,
            "exception_type": type(exc).__name__
        }
    )
    
    # Don't expose internal details in production
    if settings.ENVIRONMENT == "production":
        detail = "An unexpected error occurred"
    else:
        detail = str(exc)
    
    response = create_error_response(
        error_id=error_id,
        status_code=500,
        error_code=ErrorCode.INTERNAL_ERROR.value,
        title="Internal Server Error",
        detail=detail,
        meta={
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request.headers.get("X-Request-ID", "unknown"),
            "support_reference": error_id
        }
    )
    
    return JSONResponse(
        status_code=500,
        content=response
    )

# Register exception handlers
app = FastAPI()
app.add_exception_handler(APIException, api_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, global_exception_handler)
```

### 2.3 Error Recovery Patterns

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import circuit_breaker

class ErrorRecoveryService:
    """Implement error recovery patterns"""
    
    def __init__(self):
        self.circuit_breaker = circuit_breaker.CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=ExternalServiceError
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ExternalServiceError)
    )
    async def call_external_service(self, *args, **kwargs):
        """Call external service with retry logic"""
        
        try:
            # Check circuit breaker
            if self.circuit_breaker.is_open:
                raise ServiceUnavailableException("Service temporarily unavailable")
            
            # Make the call
            result = await self._make_external_call(*args, **kwargs)
            
            # Reset circuit breaker on success
            self.circuit_breaker.record_success()
            
            return result
            
        except ExternalServiceError as e:
            # Record failure
            self.circuit_breaker.record_failure()
            
            # Log error with context
            logger.error(
                "External service error",
                extra={
                    "service": "youtube_api",
                    "error": str(e),
                    "retry_count": self.call_external_service.retry.statistics["attempt_number"]
                }
            )
            
            raise
    
    async def graceful_degradation(self, primary_func, fallback_func, *args, **kwargs):
        """Implement graceful degradation pattern"""
        
        try:
            # Try primary function
            return await primary_func(*args, **kwargs)
        except Exception as e:
            # Log degradation
            logger.warning(
                "Degrading to fallback function",
                extra={
                    "primary": primary_func.__name__,
                    "fallback": fallback_func.__name__,
                    "error": str(e)
                }
            )
            
            # Use fallback
            return await fallback_func(*args, **kwargs)

# Usage example
@router.get("/videos/{video_id}/analytics")
async def get_video_analytics(
    video_id: str,
    recovery_service: ErrorRecoveryService = Depends()
):
    """Get video analytics with error recovery"""
    
    async def get_realtime_analytics():
        # Get real-time analytics from YouTube
        return await youtube_api.get_analytics(video_id)
    
    async def get_cached_analytics():
        # Fallback to cached data
        return await cache.get(f"analytics:{video_id}")
    
    # Try real-time, fallback to cache
    analytics = await recovery_service.graceful_degradation(
        get_realtime_analytics,
        get_cached_analytics
    )
    
    if not analytics:
        raise NotFoundException("video", video_id)
    
    return {"data": analytics}
```

### 2.4 Error Monitoring and Alerting

```python
class ErrorMonitoringService:
    """Monitor and alert on errors"""
    
    def __init__(self):
        self.error_thresholds = {
            500: 10,  # Alert if >10 500 errors in 5 minutes
            429: 100,  # Alert if >100 rate limit errors in 5 minutes
            404: 50    # Alert if >50 404 errors in 5 minutes
        }
        
        self.error_counts = defaultdict(int)
        self.window_start = datetime.utcnow()
    
    async def track_error(
        self,
        status_code: int,
        error_code: str,
        path: str,
        user_id: Optional[str] = None
    ):
        """Track error occurrence"""
        
        # Increment counter
        self.error_counts[status_code] += 1
        
        # Check if window expired
        if datetime.utcnow() - self.window_start > timedelta(minutes=5):
            await self._check_thresholds()
            self._reset_window()
        
        # Log to monitoring system
        await send_to_monitoring("api_error", {
            "timestamp": datetime.utcnow().isoformat(),
            "status_code": status_code,
            "error_code": error_code,
            "path": path,
            "user_id": user_id
        })
    
    async def _check_thresholds(self):
        """Check if error thresholds exceeded"""
        
        for status_code, threshold in self.error_thresholds.items():
            if self.error_counts[status_code] > threshold:
                await self._send_alert(
                    f"High error rate detected: {self.error_counts[status_code]} "
                    f"{status_code} errors in 5 minutes (threshold: {threshold})"
                )
    
    def _reset_window(self):
        """Reset error counting window"""
        
        self.error_counts.clear()
        self.window_start = datetime.utcnow()
```

---

## Document Control

- **Version**: 1.0
- **Last Updated**: January 2025
- **Owner**: API Development Engineer
- **Review Cycle**: Quarterly
- **Dependencies**: REST API Contract, OpenAPI Specification

**Implementation Checklist:**
- [ ] Webhook event types defined
- [ ] Webhook delivery service implemented
- [ ] Retry logic with exponential backoff
- [ ] Webhook signature verification
- [ ] Error response standardization
- [ ] Exception handlers registered
- [ ] Error recovery patterns
- [ ] Error monitoring and alerting