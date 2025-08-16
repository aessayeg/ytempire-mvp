"""
Audit logging system for YTEmpire platform.
Tracks all security-relevant events and user actions.
"""

import json
import hashlib
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from enum import Enum
from contextlib import asynccontextmanager
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from pydantic import BaseModel, Field
import redis.asyncio as redis
from fastapi import Request, Response
from fastapi.security import HTTPBearer

logger = logging.getLogger(__name__)

# ============================================================================
# Audit Event Types
# ============================================================================


class AuditEventType(str, Enum):
    """Types of audit events"""

    # Authentication events
    LOGIN_SUCCESS = "auth.login.success"
    LOGIN_FAILURE = "auth.login.failure"
    LOGOUT = "auth.logout"
    PASSWORD_CHANGE = "auth.password.change"
    PASSWORD_RESET = "auth.password.reset"
    MFA_ENABLE = "auth.mfa.enable"
    MFA_DISABLE = "auth.mfa.disable"
    MFA_VERIFY = "auth.mfa.verify"
    TOKEN_REFRESH = "auth.token.refresh"
    SESSION_EXPIRE = "auth.session.expire"

    # Authorization events
    ACCESS_GRANTED = "authz.access.granted"
    ACCESS_DENIED = "authz.access.denied"
    PERMISSION_CHANGE = "authz.permission.change"
    ROLE_ASSIGNMENT = "authz.role.assign"
    ROLE_REMOVAL = "authz.role.remove"

    # Data access events
    DATA_READ = "data.read"
    DATA_CREATE = "data.create"
    DATA_UPDATE = "data.update"
    DATA_DELETE = "data.delete"
    DATA_EXPORT = "data.export"
    DATA_IMPORT = "data.import"

    # API events
    API_KEY_CREATE = "api.key.create"
    API_KEY_REVOKE = "api.key.revoke"
    API_RATE_LIMIT = "api.rate.limit"
    API_QUOTA_EXCEED = "api.quota.exceed"

    # System events
    CONFIG_CHANGE = "system.config.change"
    SERVICE_START = "system.service.start"
    SERVICE_STOP = "system.service.stop"
    BACKUP_CREATE = "system.backup.create"
    BACKUP_RESTORE = "system.backup.restore"
    SECURITY_SCAN = "system.security.scan"

    # Security events
    SECURITY_ALERT = "security.alert"
    INTRUSION_ATTEMPT = "security.intrusion"
    MALWARE_DETECTED = "security.malware"
    SUSPICIOUS_ACTIVITY = "security.suspicious"
    BRUTE_FORCE_ATTEMPT = "security.brute_force"
    SQL_INJECTION_ATTEMPT = "security.sql_injection"
    XSS_ATTEMPT = "security.xss"

    # Compliance events
    GDPR_DATA_REQUEST = "compliance.gdpr.request"
    GDPR_DATA_DELETE = "compliance.gdpr.delete"
    CONSENT_GIVEN = "compliance.consent.given"
    CONSENT_WITHDRAWN = "compliance.consent.withdrawn"

    # Business events
    VIDEO_GENERATED = "business.video.generated"
    VIDEO_PUBLISHED = "business.video.published"
    VIDEO_DELETED = "business.video.deleted"
    CHANNEL_CREATED = "business.channel.created"
    CHANNEL_DELETED = "business.channel.deleted"
    PAYMENT_PROCESSED = "business.payment.processed"
    SUBSCRIPTION_CREATED = "business.subscription.created"
    SUBSCRIPTION_CANCELLED = "business.subscription.cancelled"


class AuditSeverity(str, Enum):
    """Severity levels for audit events"""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# Audit Event Model
# ============================================================================


class AuditEvent(BaseModel):
    """Model for audit events"""

    id: str = Field(description="Unique event ID")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: AuditEventType
    severity: AuditSeverity = AuditSeverity.INFO

    # Actor information
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    user_role: Optional[str] = None
    service_account: Optional[str] = None

    # Request context
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None

    # Resource information
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    resource_name: Optional[str] = None

    # Event details
    action: str
    result: str  # success, failure, error
    reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Security context
    risk_score: Optional[float] = None
    threat_indicators: List[str] = Field(default_factory=list)

    # Compliance
    compliance_relevant: bool = False
    data_classification: Optional[str] = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ============================================================================
# Audit Logger
# ============================================================================


class AuditLogger:
    """Centralized audit logging system"""

    def __init__(
        self,
        db_session: Optional[AsyncSession] = None,
        redis_client: Optional[redis.Redis] = None,
        elasticsearch_client: Optional[Any] = None,
    ):
        self.db_session = db_session
        self.redis_client = redis_client
        self.elasticsearch_client = elasticsearch_client
        self.buffer: List[AuditEvent] = []
        self.buffer_size = 100
        self.flush_interval = 10  # seconds
        self._running = False
        self._flush_task = None

    async def start(self):
        """Start the audit logger background tasks"""
        self._running = True
        self._flush_task = asyncio.create_task(self._periodic_flush())
        logger.info("Audit logger started")

    async def stop(self):
        """Stop the audit logger and flush remaining events"""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
        await self.flush()
        logger.info("Audit logger stopped")

    async def log_event(
        self,
        event_type: AuditEventType,
        action: str,
        result: str,
        severity: AuditSeverity = AuditSeverity.INFO,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        request: Optional[Request] = None,
        **kwargs,
    ) -> str:
        """Log an audit event"""

        # Generate unique event ID
        event_id = self._generate_event_id()

        # Extract request context if available
        ip_address = None
        user_agent = None
        request_id = None

        if request:
            ip_address = request.client.host if request.client else None
            user_agent = request.headers.get("user-agent")
            request_id = request.headers.get("x-request-id")

        # Create audit event
        event = AuditEvent(
            id=event_id,
            event_type=event_type,
            severity=severity,
            action=action,
            result=result,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            request_id=request_id,
            resource_type=resource_type,
            resource_id=resource_id,
            metadata=metadata or {},
            **kwargs,
        )

        # Calculate risk score for security events
        if event_type.value.startswith("security."):
            event.risk_score = self._calculate_risk_score(event)

        # Mark compliance-relevant events
        if event_type in [
            AuditEventType.GDPR_DATA_REQUEST,
            AuditEventType.GDPR_DATA_DELETE,
            AuditEventType.CONSENT_GIVEN,
            AuditEventType.CONSENT_WITHDRAWN,
            AuditEventType.DATA_DELETE,
            AuditEventType.DATA_EXPORT,
        ]:
            event.compliance_relevant = True

        # Add to buffer
        self.buffer.append(event)

        # Log high severity events immediately
        if severity in [AuditSeverity.HIGH, AuditSeverity.CRITICAL]:
            await self._log_immediate(event)

        # Flush buffer if full
        if len(self.buffer) >= self.buffer_size:
            await self.flush()

        return event_id

    async def log_login_attempt(
        self,
        email: str,
        success: bool,
        ip_address: str,
        user_agent: Optional[str] = None,
        reason: Optional[str] = None,
    ):
        """Log login attempt"""
        event_type = (
            AuditEventType.LOGIN_SUCCESS if success else AuditEventType.LOGIN_FAILURE
        )
        severity = AuditSeverity.INFO if success else AuditSeverity.MEDIUM

        await self.log_event(
            event_type=event_type,
            action="User login attempt",
            result="success" if success else "failure",
            severity=severity,
            user_email=email,
            ip_address=ip_address,
            user_agent=user_agent,
            reason=reason,
            metadata={
                "authentication_method": "password",
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    async def log_data_access(
        self,
        user_id: str,
        operation: str,
        resource_type: str,
        resource_id: str,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log data access event"""
        event_map = {
            "read": AuditEventType.DATA_READ,
            "create": AuditEventType.DATA_CREATE,
            "update": AuditEventType.DATA_UPDATE,
            "delete": AuditEventType.DATA_DELETE,
            "export": AuditEventType.DATA_EXPORT,
        }

        event_type = event_map.get(operation.lower(), AuditEventType.DATA_READ)

        await self.log_event(
            event_type=event_type,
            action=f"Data {operation} operation",
            result="success" if success else "failure",
            severity=AuditSeverity.INFO,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            metadata=metadata,
        )

    async def log_security_event(
        self,
        event_type: AuditEventType,
        description: str,
        ip_address: str,
        severity: AuditSeverity = AuditSeverity.HIGH,
        threat_indicators: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log security event"""
        await self.log_event(
            event_type=event_type,
            action=description,
            result="detected",
            severity=severity,
            ip_address=ip_address,
            threat_indicators=threat_indicators or [],
            metadata=metadata,
        )

        # Trigger immediate alert for critical security events
        if severity == AuditSeverity.CRITICAL:
            await self._trigger_security_alert(event_type, description, ip_address)

    async def log_api_access(
        self,
        user_id: str,
        endpoint: str,
        method: str,
        status_code: int,
        response_time: float,
        ip_address: str,
        api_key: Optional[str] = None,
    ):
        """Log API access"""
        severity = AuditSeverity.INFO
        if status_code >= 500:
            severity = AuditSeverity.MEDIUM
        elif status_code == 403:
            severity = AuditSeverity.LOW

        await self.log_event(
            event_type=AuditEventType.ACCESS_GRANTED
            if status_code < 400
            else AuditEventType.ACCESS_DENIED,
            action=f"{method} {endpoint}",
            result="success" if status_code < 400 else "failure",
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            metadata={
                "endpoint": endpoint,
                "method": method,
                "status_code": status_code,
                "response_time_ms": response_time * 1000,
                "api_key": api_key[:8] + "..." if api_key else None,
            },
        )

    async def flush(self):
        """Flush buffered events to storage"""
        if not self.buffer:
            return

        events_to_flush = self.buffer.copy()
        self.buffer.clear()

        # Store in database
        if self.db_session:
            await self._store_in_database(events_to_flush)

        # Store in Elasticsearch for searching
        if self.elasticsearch_client:
            await self._store_in_elasticsearch(events_to_flush)

        # Cache recent events in Redis
        if self.redis_client:
            await self._cache_in_redis(events_to_flush)

        logger.info(f"Flushed {len(events_to_flush)} audit events")

    async def _periodic_flush(self):
        """Periodically flush audit events"""
        while self._running:
            await asyncio.sleep(self.flush_interval)
            await self.flush()

    async def _log_immediate(self, event: AuditEvent):
        """Log high-priority event immediately"""
        # Store immediately in database
        if self.db_session:
            await self._store_in_database([event])

        # Send to monitoring system
        logger.warning(f"High severity audit event: {event.json()}")

        # Trigger alerts if needed
        if event.severity == AuditSeverity.CRITICAL:
            await self._trigger_alert(event)

    async def _store_in_database(self, events: List[AuditEvent]):
        """Store events in database"""
        if not self.db_session:
            return

        try:
            for event in events:
                await self.db_session.execute(
                    text(
                        """
                        INSERT INTO audit_logs (
                            id, timestamp, event_type, severity,
                            user_id, user_email, ip_address,
                            resource_type, resource_id, action,
                            result, metadata
                        ) VALUES (
                            :id, :timestamp, :event_type, :severity,
                            :user_id, :user_email, :ip_address,
                            :resource_type, :resource_id, :action,
                            :result, :metadata
                        )
                    """
                    ),
                    {
                        "id": event.id,
                        "timestamp": event.timestamp,
                        "event_type": event.event_type.value,
                        "severity": event.severity.value,
                        "user_id": event.user_id,
                        "user_email": event.user_email,
                        "ip_address": event.ip_address,
                        "resource_type": event.resource_type,
                        "resource_id": event.resource_id,
                        "action": event.action,
                        "result": event.result,
                        "metadata": json.dumps(event.metadata),
                    },
                )
            await self.db_session.commit()
        except Exception as e:
            logger.error(f"Failed to store audit events in database: {e}")
            await self.db_session.rollback()

    async def _store_in_elasticsearch(self, events: List[AuditEvent]):
        """Store events in Elasticsearch for searching"""
        if not self.elasticsearch_client:
            return

        try:
            # Bulk index events
            actions = []
            for event in events:
                actions.append(
                    {"_index": "audit-logs", "_id": event.id, "_source": event.dict()}
                )

            # Perform bulk indexing
            # await self.elasticsearch_client.bulk(actions)
            pass  # Placeholder for actual Elasticsearch implementation
        except Exception as e:
            logger.error(f"Failed to store audit events in Elasticsearch: {e}")

    async def _cache_in_redis(self, events: List[AuditEvent]):
        """Cache recent events in Redis"""
        if not self.redis_client:
            return

        try:
            for event in events:
                # Store in sorted set by timestamp
                await self.redis_client.zadd(
                    "audit:events", {event.id: event.timestamp.timestamp()}
                )

                # Store event details
                await self.redis_client.setex(
                    f"audit:event:{event.id}", 86400, event.json()  # 24 hour TTL
                )

                # Index by user
                if event.user_id:
                    await self.redis_client.zadd(
                        f"audit:user:{event.user_id}",
                        {event.id: event.timestamp.timestamp()},
                    )

                # Index by event type
                await self.redis_client.zadd(
                    f"audit:type:{event.event_type.value}",
                    {event.id: event.timestamp.timestamp()},
                )

            # Trim old events (keep last 10000)
            await self.redis_client.zremrangebyrank("audit:events", 0, -10001)

        except Exception as e:
            logger.error(f"Failed to cache audit events in Redis: {e}")

    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        timestamp = datetime.utcnow().isoformat()
        random_bytes = hashlib.sha256(f"{timestamp}{id(self)}".encode()).hexdigest()[
            :16
        ]
        return f"audit_{timestamp.replace(':', '').replace('.', '')}_{random_bytes}"

    def _calculate_risk_score(self, event: AuditEvent) -> float:
        """Calculate risk score for security events"""
        base_score = {
            AuditSeverity.INFO: 0.0,
            AuditSeverity.LOW: 0.2,
            AuditSeverity.MEDIUM: 0.5,
            AuditSeverity.HIGH: 0.8,
            AuditSeverity.CRITICAL: 1.0,
        }.get(event.severity, 0.0)

        # Adjust based on threat indicators
        if event.threat_indicators:
            base_score += len(event.threat_indicators) * 0.1

        # Adjust based on event type
        high_risk_events = [
            AuditEventType.INTRUSION_ATTEMPT,
            AuditEventType.MALWARE_DETECTED,
            AuditEventType.SQL_INJECTION_ATTEMPT,
            AuditEventType.BRUTE_FORCE_ATTEMPT,
        ]

        if event.event_type in high_risk_events:
            base_score += 0.3

        return min(base_score, 1.0)

    async def _trigger_alert(self, event: AuditEvent):
        """Trigger alert for critical events"""
        # Send to monitoring system
        logger.critical(
            f"CRITICAL AUDIT EVENT: {event.event_type.value} - {event.action}"
        )

        # Send notification (email, Slack, etc.)
        # Placeholder for actual alert implementation
        pass

    async def _trigger_security_alert(
        self, event_type: AuditEventType, description: str, ip_address: str
    ):
        """Trigger security alert"""
        logger.critical(
            f"SECURITY ALERT: {event_type.value} from {ip_address}: {description}"
        )

        # Block IP if necessary
        # Send to SIEM
        # Notify security team
        pass

    async def query_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
        user_id: Optional[str] = None,
        severity: Optional[AuditSeverity] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Query audit events"""
        if self.redis_client:
            # Query from Redis cache first
            events = []

            # Get event IDs from sorted set
            if user_id:
                event_ids = await self.redis_client.zrevrange(
                    f"audit:user:{user_id}", 0, limit - 1
                )
            else:
                event_ids = await self.redis_client.zrevrange(
                    "audit:events", 0, limit - 1
                )

            # Get event details
            for event_id in event_ids:
                event_json = await self.redis_client.get(f"audit:event:{event_id}")
                if event_json:
                    event = AuditEvent.parse_raw(event_json)

                    # Apply filters
                    if start_time and event.timestamp < start_time:
                        continue
                    if end_time and event.timestamp > end_time:
                        continue
                    if event_types and event.event_type not in event_types:
                        continue
                    if severity and event.severity != severity:
                        continue

                    events.append(event)

            return events[:limit]

        return []

    async def get_user_activity(
        self,
        user_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Get user activity log"""
        return await self.query_events(
            user_id=user_id, start_time=start_time, end_time=end_time, limit=limit
        )

    async def get_security_events(
        self, severity_threshold: AuditSeverity = AuditSeverity.MEDIUM, limit: int = 100
    ) -> List[AuditEvent]:
        """Get recent security events"""
        events = await self.query_events(limit=limit * 2)

        security_events = [
            e
            for e in events
            if e.event_type.value.startswith("security.")
            and e.severity.value >= severity_threshold.value
        ]

        return security_events[:limit]


# ============================================================================
# Audit Context Manager
# ============================================================================


@asynccontextmanager
async def audit_context(
    audit_logger: AuditLogger,
    event_type: AuditEventType,
    action: str,
    user_id: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Context manager for audit logging"""
    start_time = datetime.utcnow()

    try:
        yield
        # Log success
        await audit_logger.log_event(
            event_type=event_type,
            action=action,
            result="success",
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            metadata={
                **(metadata or {}),
                "duration_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
            },
        )
    except Exception as e:
        # Log failure
        await audit_logger.log_event(
            event_type=event_type,
            action=action,
            result="failure",
            severity=AuditSeverity.MEDIUM,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            reason=str(e),
            metadata={
                **(metadata or {}),
                "error": str(e),
                "duration_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
            },
        )
        raise


# ============================================================================
# Global Audit Logger Instance
# ============================================================================

audit_logger = AuditLogger()
