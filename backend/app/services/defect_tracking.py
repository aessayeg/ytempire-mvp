"""
Defect Tracking System for YTEmpire
Comprehensive bug and issue management system based on QA documentation requirements
"""

import uuid
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass, asdict
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, and_, or_
import redis.asyncio as redis
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ============================================================================
# Defect Classification (Based on QA Documentation)
# ============================================================================


class DefectSeverity(str, Enum):
    """Defect severity levels from QA documentation"""

    CRITICAL = "critical"  # P0 - System down, data loss, security breach
    HIGH = "high"  # P1 - Major feature broken, significant impact
    MEDIUM = "medium"  # P2 - Minor feature issues, moderate impact
    LOW = "low"  # P3 - Cosmetic issues, minimal impact


class DefectPriority(str, Enum):
    """Defect priority levels"""

    P0 = "p0"  # Fix immediately (within hours)
    P1 = "p1"  # Fix within 24 hours
    P2 = "p2"  # Fix within 72 hours
    P3 = "p3"  # Fix when convenient


class DefectStatus(str, Enum):
    """Defect status from QA lifecycle"""

    NEW = "new"
    TRIAGED = "triaged"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    READY_FOR_TEST = "ready_for_test"
    VERIFIED = "verified"
    CLOSED = "closed"
    REOPENED = "reopened"


class DefectType(str, Enum):
    """Types of defects"""

    BUG = "bug"
    REGRESSION = "regression"
    PERFORMANCE = "performance"
    SECURITY = "security"
    USABILITY = "usability"
    COMPATIBILITY = "compatibility"
    DATA_CORRUPTION = "data_corruption"
    INTEGRATION = "integration"


class DefectSource(str, Enum):
    """Source of defect discovery"""

    USER_REPORT = "user_report"
    AUTOMATED_TEST = "automated_test"
    MANUAL_TEST = "manual_test"
    MONITORING = "monitoring"
    CODE_REVIEW = "code_review"
    BETA_TESTING = "beta_testing"
    PRODUCTION = "production"


@dataclass
class DefectMetadata:
    """Additional defect metadata"""

    browser: Optional[str] = None
    os: Optional[str] = None
    device: Optional[str] = None
    version: Optional[str] = None
    user_agent: Optional[str] = None
    api_endpoint: Optional[str] = None
    error_code: Optional[str] = None
    stack_trace: Optional[str] = None
    reproduction_rate: Optional[float] = None
    affected_users: Optional[int] = None


@dataclass
class Defect:
    """Defect entity based on QA bug report template"""

    id: str
    title: str
    description: str
    severity: DefectSeverity
    priority: DefectPriority
    status: DefectStatus
    type: DefectType
    source: DefectSource

    # Assignment and tracking
    reporter_id: str
    assignee_id: Optional[str] = None
    component: Optional[str] = None
    tags: List[str] = None

    # Timestamps
    created_at: datetime = None
    updated_at: datetime = None
    resolved_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None

    # Resolution details
    resolution: Optional[str] = None
    resolution_notes: Optional[str] = None
    fix_version: Optional[str] = None

    # Metrics for QA reporting
    time_to_resolution: Optional[int] = None  # minutes
    reopened_count: int = 0
    duplicate_of: Optional[str] = None

    # Additional data
    metadata: DefectMetadata = None
    attachments: List[str] = None
    comments: List[Dict] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = self.created_at
        if self.tags is None:
            self.tags = []
        if self.attachments is None:
            self.attachments = []
        if self.comments is None:
            self.comments = []
        if self.metadata is None:
            self.metadata = DefectMetadata()


# ============================================================================
# Defect Tracking Manager
# ============================================================================


class DefectTracker:
    """Main defect tracking system implementing QA requirements"""

    def __init__(self):
        self.defects: Dict[str, Defect] = {}
        self._redis_client: Optional[redis.Redis] = None

    async def initialize(self):
        """Initialize the defect tracker"""
        try:
            # Initialize Redis client for caching and real-time updates
            self._redis_client = redis.from_url("redis://localhost:6379/0")
            await self._redis_client.ping()
            logger.info("Defect tracker initialized successfully")
        except Exception as e:
            logger.warning(f"Redis not available for defect tracker: {e}")
            # Continue without Redis - use in-memory storage

    async def create_defect(
        self,
        title: str,
        description: str,
        severity: DefectSeverity,
        reporter_id: str,
        defect_type: DefectType = DefectType.BUG,
        source: DefectSource = DefectSource.USER_REPORT,
        component: Optional[str] = None,
        metadata: Optional[DefectMetadata] = None,
    ) -> Defect:
        """Create a new defect following QA procedures"""

        defect_id = f"BUG-{uuid.uuid4().hex[:8].upper()}"

        # Auto-assign priority based on severity (QA guidelines)
        priority_mapping = {
            DefectSeverity.CRITICAL: DefectPriority.P0,
            DefectSeverity.HIGH: DefectPriority.P1,
            DefectSeverity.MEDIUM: DefectPriority.P2,
            DefectSeverity.LOW: DefectPriority.P3,
        }

        defect = Defect(
            id=defect_id,
            title=title,
            description=description,
            severity=severity,
            priority=priority_mapping[severity],
            status=DefectStatus.NEW,
            type=defect_type,
            source=source,
            reporter_id=reporter_id,
            component=component,
            metadata=metadata or DefectMetadata(),
        )

        self.defects[defect_id] = defect

        # Cache in Redis if available
        if self._redis_client:
            try:
                await self._redis_client.setex(
                    f"defect:{defect_id}",
                    3600,  # 1 hour TTL
                    json.dumps(asdict(defect), default=str),
                )
            except Exception as e:
                logger.warning(f"Failed to cache defect in Redis: {e}")

        logger.info(f"Created defect {defect_id}: {title} [{severity.value}]")
        return defect

    async def get_defect(self, defect_id: str) -> Optional[Defect]:
        """Get a specific defect by ID"""
        # Try in-memory first
        if defect_id in self.defects:
            return self.defects[defect_id]

        # Try Redis cache
        if self._redis_client:
            try:
                cached = await self._redis_client.get(f"defect:{defect_id}")
                if cached:
                    data = json.loads(cached)
                    # Reconstruct defect from cached data
                    # In production, this would be more sophisticated
                    return self._deserialize_defect(data)
            except Exception as e:
                logger.warning(f"Failed to get defect from Redis: {e}")

        return None

    async def get_defects(
        self, filters: Optional[Dict] = None, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get defects with optional filtering (QA dashboard requirements)"""

        defects = list(self.defects.values())

        # Apply filters
        if filters:
            if filters.get("severity"):
                defects = [d for d in defects if d.severity == filters["severity"]]
            if filters.get("status"):
                defects = [d for d in defects if d.status == filters["status"]]
            if filters.get("assignee_id"):
                defects = [
                    d for d in defects if d.assignee_id == filters["assignee_id"]
                ]
            if filters.get("component"):
                defects = [d for d in defects if d.component == filters["component"]]
            if filters.get("created_after"):
                created_after = datetime.fromisoformat(filters["created_after"])
                defects = [d for d in defects if d.created_at >= created_after]
            if filters.get("created_before"):
                created_before = datetime.fromisoformat(filters["created_before"])
                defects = [d for d in defects if d.created_at <= created_before]

        # Sort by created_at descending
        defects.sort(key=lambda x: x.created_at, reverse=True)

        # Apply pagination
        paginated = defects[offset : offset + limit]

        # Convert to dict format for API response
        return [self._defect_to_dict(defect) for defect in paginated]

    async def update_defect_status(
        self,
        defect_id: str,
        new_status: DefectStatus,
        assignee_id: Optional[str] = None,
        resolution_notes: Optional[str] = None,
    ) -> Optional[Defect]:
        """Update defect status following QA lifecycle"""

        defect = await self.get_defect(defect_id)
        if not defect:
            return None

        old_status = defect.status
        defect.status = new_status
        defect.updated_at = datetime.now(timezone.utc)

        if assignee_id:
            defect.assignee_id = assignee_id

        if resolution_notes:
            defect.resolution_notes = resolution_notes

        # Handle status-specific logic
        if new_status == DefectStatus.VERIFIED:
            defect.resolved_at = datetime.now(timezone.utc)
            if defect.created_at:
                defect.time_to_resolution = int(
                    (defect.resolved_at - defect.created_at).total_seconds() / 60
                )

        elif new_status == DefectStatus.CLOSED:
            defect.closed_at = datetime.now(timezone.utc)

        elif new_status == DefectStatus.REOPENED:
            defect.reopened_count += 1
            defect.resolved_at = None
            defect.closed_at = None

        # Update in storage
        self.defects[defect_id] = defect

        logger.info(
            f"Updated defect {defect_id}: {old_status.value} → {new_status.value}"
        )
        return defect

    async def get_defect_statistics(self) -> Dict[str, Any]:
        """Get defect statistics for QA dashboard"""

        defects = list(self.defects.values())
        total_count = len(defects)

        if total_count == 0:
            return {
                "total": 0,
                "by_status": {},
                "by_severity": {},
                "by_component": {},
                "resolution_metrics": {},
                "trends": {},
            }

        # Count by status
        status_counts = {}
        for status in DefectStatus:
            status_counts[status.value] = len(
                [d for d in defects if d.status == status]
            )

        # Count by severity
        severity_counts = {}
        for severity in DefectSeverity:
            severity_counts[severity.value] = len(
                [d for d in defects if d.severity == severity]
            )

        # Count by component
        component_counts = {}
        for defect in defects:
            component = defect.component or "unassigned"
            component_counts[component] = component_counts.get(component, 0) + 1

        # Resolution metrics
        resolved_defects = [d for d in defects if d.time_to_resolution is not None]
        avg_resolution_time = 0
        if resolved_defects:
            avg_resolution_time = sum(
                d.time_to_resolution for d in resolved_defects
            ) / len(resolved_defects)

        # Defect escape rate (simplified calculation)
        production_defects = len(
            [d for d in defects if d.source == DefectSource.PRODUCTION]
        )
        escape_rate = (production_defects / total_count * 100) if total_count > 0 else 0

        return {
            "total": total_count,
            "by_status": status_counts,
            "by_severity": severity_counts,
            "by_component": component_counts,
            "resolution_metrics": {
                "average_resolution_time_minutes": int(avg_resolution_time),
                "defect_escape_rate_percent": round(escape_rate, 2),
                "resolved_count": len(resolved_defects),
                "reopened_count": len([d for d in defects if d.reopened_count > 0]),
            },
            "trends": self._calculate_trends(defects),
        }

    def _calculate_trends(self, defects: List[Defect]) -> Dict[str, Any]:
        """Calculate defect trends for reporting"""
        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(days=7)

        recent_defects = [d for d in defects if d.created_at >= week_ago]

        return {
            "new_this_week": len(recent_defects),
            "resolved_this_week": len(
                [d for d in defects if d.resolved_at and d.resolved_at >= week_ago]
            ),
            "critical_open": len(
                [
                    d
                    for d in defects
                    if d.severity == DefectSeverity.CRITICAL
                    and d.status not in [DefectStatus.VERIFIED, DefectStatus.CLOSED]
                ]
            ),
        }

    def _defect_to_dict(self, defect: Defect) -> Dict[str, Any]:
        """Convert defect to dictionary for API response"""
        return {
            "id": defect.id,
            "title": defect.title,
            "description": defect.description,
            "severity": defect.severity.value,
            "priority": defect.priority.value,
            "status": defect.status.value,
            "type": defect.type.value,
            "source": defect.source.value,
            "reporter_id": defect.reporter_id,
            "assignee_id": defect.assignee_id,
            "component": defect.component,
            "tags": defect.tags,
            "created_at": defect.created_at.isoformat() if defect.created_at else None,
            "updated_at": defect.updated_at.isoformat() if defect.updated_at else None,
            "resolved_at": defect.resolved_at.isoformat()
            if defect.resolved_at
            else None,
            "closed_at": defect.closed_at.isoformat() if defect.closed_at else None,
            "resolution": defect.resolution,
            "resolution_notes": defect.resolution_notes,
            "fix_version": defect.fix_version,
            "time_to_resolution": defect.time_to_resolution,
            "reopened_count": defect.reopened_count,
            "duplicate_of": defect.duplicate_of,
            "metadata": asdict(defect.metadata) if defect.metadata else None,
            "attachments": defect.attachments,
            "comments": defect.comments,
        }

    def _deserialize_defect(self, data: Dict) -> Defect:
        """Deserialize defect from dictionary (simplified for MVP)"""
        # In production, this would be more robust with proper validation
        metadata = (
            DefectMetadata(**data.get("metadata", {}))
            if data.get("metadata")
            else DefectMetadata()
        )

        return Defect(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            severity=DefectSeverity(data["severity"]),
            priority=DefectPriority(data["priority"]),
            status=DefectStatus(data["status"]),
            type=DefectType(data["type"]),
            source=DefectSource(data["source"]),
            reporter_id=data["reporter_id"],
            assignee_id=data.get("assignee_id"),
            component=data.get("component"),
            tags=data.get("tags", []),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else None,
            updated_at=datetime.fromisoformat(data["updated_at"])
            if data.get("updated_at")
            else None,
            resolved_at=datetime.fromisoformat(data["resolved_at"])
            if data.get("resolved_at")
            else None,
            closed_at=datetime.fromisoformat(data["closed_at"])
            if data.get("closed_at")
            else None,
            resolution=data.get("resolution"),
            resolution_notes=data.get("resolution_notes"),
            fix_version=data.get("fix_version"),
            time_to_resolution=data.get("time_to_resolution"),
            reopened_count=data.get("reopened_count", 0),
            duplicate_of=data.get("duplicate_of"),
            metadata=metadata,
            attachments=data.get("attachments", []),
            comments=data.get("comments", []),
        )


# Create global instance
defect_tracker = DefectTracker()
