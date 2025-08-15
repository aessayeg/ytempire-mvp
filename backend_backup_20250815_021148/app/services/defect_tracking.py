"""
Defect Tracking System for YTEmpire
Comprehensive bug and issue management system
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

from app.core.audit_logging import audit_logger, AuditEventType, AuditSeverity

logger = logging.getLogger(__name__)

# ============================================================================
# Defect Classification
# ============================================================================

class DefectSeverity(str, Enum):
    """Defect severity levels"""
    CRITICAL = "critical"      # System down, data loss, security breach
    HIGH = "high"              # Major feature broken, significant impact
    MEDIUM = "medium"          # Minor feature issues, moderate impact
    LOW = "low"               # Cosmetic issues, minimal impact

class DefectPriority(str, Enum):
    """Defect priority levels"""
    P0 = "p0"  # Fix immediately (within hours)
    P1 = "p1"  # Fix within 24 hours
    P2 = "p2"  # Fix within 1 week
    P3 = "p3"  # Fix when convenient

class DefectStatus(str, Enum):
    """Defect status"""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    TESTING = "testing"
    RESOLVED = "resolved"
    CLOSED = "closed"
    REOPENED = "reopened"
    DEFERRED = "deferred"
    DUPLICATE = "duplicate"
    WONT_FIX = "wont_fix"

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
    """Defect entity"""
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
    
    # Metrics
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
    """Main defect tracking system"""
    
    def __init__(
        self,
        db_session: Optional[AsyncSession] = None,
        redis_client: Optional[redis.Redis] = None
    ):
        self.db_session = db_session
        self.redis_client = redis_client
        self.sla_config = self._load_sla_config()
    
    def _load_sla_config(self) -> Dict[DefectPriority, Dict[str, int]]:
        """Load SLA configuration for different priority levels"""
        return {
            DefectPriority.P0: {
                "response_time_minutes": 60,     # 1 hour
                "resolution_time_hours": 4       # 4 hours
            },
            DefectPriority.P1: {
                "response_time_minutes": 240,    # 4 hours
                "resolution_time_hours": 24      # 24 hours
            },
            DefectPriority.P2: {
                "response_time_minutes": 1440,   # 24 hours
                "resolution_time_hours": 168     # 1 week
            },
            DefectPriority.P3: {
                "response_time_minutes": 4320,   # 3 days
                "resolution_time_hours": 720     # 30 days
            }
        }
    
    async def create_defect(
        self,
        title: str,
        description: str,
        severity: DefectSeverity,
        type: DefectType,
        source: DefectSource,
        reporter_id: str,
        component: Optional[str] = None,
        metadata: Optional[DefectMetadata] = None,
        tags: Optional[List[str]] = None,
        attachments: Optional[List[str]] = None
    ) -> Defect:
        """Create a new defect"""
        
        defect_id = str(uuid.uuid4())
        
        # Auto-assign priority based on severity
        priority = self._determine_priority(severity, type)
        
        defect = Defect(
            id=defect_id,
            title=title,
            description=description,
            severity=severity,
            priority=priority,
            status=DefectStatus.OPEN,
            type=type,
            source=source,
            reporter_id=reporter_id,
            component=component,
            metadata=metadata or DefectMetadata(),
            tags=tags or [],
            attachments=attachments or []
        )
        
        # Store in database
        if self.db_session:
            await self._store_defect(defect)
        
        # Cache in Redis
        if self.redis_client:
            await self._cache_defect(defect)
        
        # Log audit event
        await audit_logger.log_event(
            event_type=AuditEventType.DATA_CREATE,
            action=f"Defect created: {defect_id}",
            result="success",
            severity=AuditSeverity.MEDIUM if severity in [DefectSeverity.CRITICAL, DefectSeverity.HIGH] else AuditSeverity.LOW,
            user_id=reporter_id,
            metadata={
                "defect_id": defect_id,
                "severity": severity.value,
                "type": type.value,
                "component": component
            }
        )
        
        # Trigger notifications for high severity defects
        if severity in [DefectSeverity.CRITICAL, DefectSeverity.HIGH]:
            await self._send_alert(defect)
        
        logger.info(f"Created defect {defect_id}: {title} ({severity.value})")
        return defect
    
    def _determine_priority(self, severity: DefectSeverity, type: DefectType) -> DefectPriority:
        """Automatically determine priority based on severity and type"""
        if severity == DefectSeverity.CRITICAL:
            return DefectPriority.P0
        elif severity == DefectSeverity.HIGH:
            if type in [DefectType.SECURITY, DefectType.DATA_CORRUPTION]:
                return DefectPriority.P0
            else:
                return DefectPriority.P1
        elif severity == DefectSeverity.MEDIUM:
            return DefectPriority.P2
        else:
            return DefectPriority.P3
    
    async def _store_defect(self, defect: Defect):
        """Store defect in database"""
        try:
            await self.db_session.execute(
                text("""
                    INSERT INTO defects (
                        id, title, description, severity, priority, status, type, source,
                        reporter_id, assignee_id, component, tags, created_at, updated_at,
                        metadata, attachments, comments
                    ) VALUES (
                        :id, :title, :description, :severity, :priority, :status, :type, :source,
                        :reporter_id, :assignee_id, :component, :tags, :created_at, :updated_at,
                        :metadata, :attachments, :comments
                    )
                """),
                {
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
                    "tags": json.dumps(defect.tags),
                    "created_at": defect.created_at,
                    "updated_at": defect.updated_at,
                    "metadata": json.dumps(asdict(defect.metadata)),
                    "attachments": json.dumps(defect.attachments),
                    "comments": json.dumps(defect.comments)
                }
            )
            await self.db_session.commit()
        except Exception as e:
            logger.error(f"Failed to store defect {defect.id}: {e}")
            raise
    
    async def _cache_defect(self, defect: Defect):
        """Cache defect in Redis"""
        try:
            key = f"defect:{defect.id}"
            await self.redis_client.setex(
                key,
                3600,  # 1 hour TTL
                json.dumps(asdict(defect), default=str)
            )
            
            # Add to priority queues
            priority_key = f"defects:priority:{defect.priority.value}"
            await self.redis_client.zadd(
                priority_key,
                {defect.id: int(defect.created_at.timestamp())}
            )
            
        except Exception as e:
            logger.error(f"Failed to cache defect {defect.id}: {e}")
    
    async def _send_alert(self, defect: Defect):
        """Send alert for high priority defects"""
        # This would integrate with your notification system
        logger.warning(f"HIGH PRIORITY DEFECT: {defect.title} (ID: {defect.id})")
        
        # Send to monitoring systems, Slack, email, etc.
        pass
    
    async def update_defect(
        self,
        defect_id: str,
        user_id: str,
        **updates
    ) -> Optional[Defect]:
        """Update an existing defect"""
        
        defect = await self.get_defect(defect_id)
        if not defect:
            return None
        
        # Track changes for audit
        changes = {}
        for key, new_value in updates.items():
            if hasattr(defect, key):
                old_value = getattr(defect, key)
                if old_value != new_value:
                    changes[key] = {"old": old_value, "new": new_value}
                    setattr(defect, key, new_value)
        
        # Special handling for status changes
        if "status" in changes:
            await self._handle_status_change(defect, changes["status"]["old"], changes["status"]["new"])
        
        defect.updated_at = datetime.now(timezone.utc)
        
        # Update in database
        if self.db_session and changes:
            await self._update_defect_in_db(defect)
        
        # Update cache
        if self.redis_client:
            await self._cache_defect(defect)
        
        # Log audit event
        if changes:
            await audit_logger.log_event(
                event_type=AuditEventType.DATA_UPDATE,
                action=f"Defect updated: {defect_id}",
                result="success",
                severity=AuditSeverity.LOW,
                user_id=user_id,
                metadata={
                    "defect_id": defect_id,
                    "changes": changes
                }
            )
        
        return defect
    
    async def _handle_status_change(self, defect: Defect, old_status: str, new_status: str):
        """Handle special logic for status changes"""
        now = datetime.now(timezone.utc)
        
        if new_status == DefectStatus.RESOLVED.value:
            defect.resolved_at = now
            if defect.created_at:
                defect.time_to_resolution = int((now - defect.created_at).total_seconds() / 60)
        
        elif new_status == DefectStatus.CLOSED.value:
            defect.closed_at = now
        
        elif new_status == DefectStatus.REOPENED.value:
            defect.reopened_count += 1
            defect.resolved_at = None
            defect.closed_at = None
    
    async def _update_defect_in_db(self, defect: Defect):
        """Update defect in database"""
        try:
            await self.db_session.execute(
                text("""
                    UPDATE defects SET
                        title = :title,
                        description = :description,
                        severity = :severity,
                        priority = :priority,
                        status = :status,
                        assignee_id = :assignee_id,
                        component = :component,
                        tags = :tags,
                        updated_at = :updated_at,
                        resolved_at = :resolved_at,
                        closed_at = :closed_at,
                        resolution = :resolution,
                        resolution_notes = :resolution_notes,
                        fix_version = :fix_version,
                        time_to_resolution = :time_to_resolution,
                        reopened_count = :reopened_count,
                        duplicate_of = :duplicate_of,
                        metadata = :metadata,
                        attachments = :attachments,
                        comments = :comments
                    WHERE id = :id
                """),
                {
                    "id": defect.id,
                    "title": defect.title,
                    "description": defect.description,
                    "severity": defect.severity.value,
                    "priority": defect.priority.value,
                    "status": defect.status.value,
                    "assignee_id": defect.assignee_id,
                    "component": defect.component,
                    "tags": json.dumps(defect.tags),
                    "updated_at": defect.updated_at,
                    "resolved_at": defect.resolved_at,
                    "closed_at": defect.closed_at,
                    "resolution": defect.resolution,
                    "resolution_notes": defect.resolution_notes,
                    "fix_version": defect.fix_version,
                    "time_to_resolution": defect.time_to_resolution,
                    "reopened_count": defect.reopened_count,
                    "duplicate_of": defect.duplicate_of,
                    "metadata": json.dumps(asdict(defect.metadata)),
                    "attachments": json.dumps(defect.attachments),
                    "comments": json.dumps(defect.comments)
                }
            )
            await self.db_session.commit()
        except Exception as e:
            logger.error(f"Failed to update defect {defect.id}: {e}")
            raise
    
    async def get_defect(self, defect_id: str) -> Optional[Defect]:
        """Get defect by ID"""
        
        # Try cache first
        if self.redis_client:
            try:
                cached = await self.redis_client.get(f"defect:{defect_id}")
                if cached:
                    data = json.loads(cached)
                    return self._dict_to_defect(data)
            except Exception as e:
                logger.error(f"Failed to get defect from cache: {e}")
        
        # Fallback to database
        if self.db_session:
            try:
                result = await self.db_session.execute(
                    text("SELECT * FROM defects WHERE id = :id"),
                    {"id": defect_id}
                )
                row = result.first()
                if row:
                    return self._row_to_defect(row)
            except Exception as e:
                logger.error(f"Failed to get defect from database: {e}")
        
        return None
    
    def _dict_to_defect(self, data: Dict) -> Defect:
        """Convert dictionary to Defect object"""
        # Parse datetime strings
        for field in ['created_at', 'updated_at', 'resolved_at', 'closed_at']:
            if data.get(field):
                data[field] = datetime.fromisoformat(data[field])
        
        # Parse enums
        data['severity'] = DefectSeverity(data['severity'])
        data['priority'] = DefectPriority(data['priority'])
        data['status'] = DefectStatus(data['status'])
        data['type'] = DefectType(data['type'])
        data['source'] = DefectSource(data['source'])
        
        # Parse metadata
        if data.get('metadata'):
            if isinstance(data['metadata'], dict):
                data['metadata'] = DefectMetadata(**data['metadata'])
            else:
                data['metadata'] = DefectMetadata()
        
        return Defect(**data)
    
    def _row_to_defect(self, row) -> Defect:
        """Convert database row to Defect object"""
        return Defect(
            id=row.id,
            title=row.title,
            description=row.description,
            severity=DefectSeverity(row.severity),
            priority=DefectPriority(row.priority),
            status=DefectStatus(row.status),
            type=DefectType(row.type),
            source=DefectSource(row.source),
            reporter_id=row.reporter_id,
            assignee_id=row.assignee_id,
            component=row.component,
            tags=json.loads(row.tags or "[]"),
            created_at=row.created_at,
            updated_at=row.updated_at,
            resolved_at=row.resolved_at,
            closed_at=row.closed_at,
            resolution=row.resolution,
            resolution_notes=row.resolution_notes,
            fix_version=row.fix_version,
            time_to_resolution=row.time_to_resolution,
            reopened_count=row.reopened_count or 0,
            duplicate_of=row.duplicate_of,
            metadata=DefectMetadata(**json.loads(row.metadata or "{}")),
            attachments=json.loads(row.attachments or "[]"),
            comments=json.loads(row.comments or "[]")
        )
    
    async def add_comment(self, defect_id: str, user_id: str, comment: str) -> bool:
        """Add comment to defect"""
        defect = await self.get_defect(defect_id)
        if not defect:
            return False
        
        comment_data = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "comment": comment,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        defect.comments.append(comment_data)
        defect.updated_at = datetime.now(timezone.utc)
        
        # Update in database
        if self.db_session:
            await self._update_defect_in_db(defect)
        
        # Update cache
        if self.redis_client:
            await self._cache_defect(defect)
        
        return True
    
    async def get_defects(
        self,
        filters: Optional[Dict] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        limit: int = 100,
        offset: int = 0
    ) -> List[Defect]:
        """Get defects with filtering and pagination"""
        
        if not self.db_session:
            return []
        
        # Build query
        query = "SELECT * FROM defects"
        params = {}
        where_clauses = []
        
        if filters:
            for key, value in filters.items():
                if key in ['severity', 'priority', 'status', 'type', 'source']:
                    where_clauses.append(f"{key} = :{key}")
                    params[key] = value
                elif key == 'assignee_id':
                    where_clauses.append("assignee_id = :assignee_id")
                    params['assignee_id'] = value
                elif key == 'component':
                    where_clauses.append("component = :component")
                    params['component'] = value
                elif key == 'created_after':
                    where_clauses.append("created_at >= :created_after")
                    params['created_after'] = value
                elif key == 'created_before':
                    where_clauses.append("created_at <= :created_before")
                    params['created_before'] = value
        
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        
        query += f" ORDER BY {sort_by} {sort_order}"
        query += f" LIMIT {limit} OFFSET {offset}"
        
        try:
            result = await self.db_session.execute(text(query), params)
            return [self._row_to_defect(row) for row in result]
        except Exception as e:
            logger.error(f"Failed to get defects: {e}")
            return []
    
    async def get_defect_statistics(self) -> Dict[str, Any]:
        """Get defect statistics"""
        if not self.db_session:
            return {}
        
        stats = {
            "total_defects": 0,
            "open_defects": 0,
            "resolved_defects": 0,
            "by_severity": {},
            "by_priority": {},
            "by_type": {},
            "by_component": {},
            "avg_resolution_time_hours": 0,
            "sla_compliance": {}
        }
        
        try:
            # Total counts
            result = await self.db_session.execute(
                text("SELECT COUNT(*) as total FROM defects")
            )
            stats["total_defects"] = result.scalar()
            
            # Status breakdown
            result = await self.db_session.execute(
                text("""
                    SELECT status, COUNT(*) as count 
                    FROM defects 
                    GROUP BY status
                """)
            )
            for row in result:
                if row.status == "open":
                    stats["open_defects"] = row.count
                elif row.status in ["resolved", "closed"]:
                    stats["resolved_defects"] += row.count
            
            # Severity breakdown
            result = await self.db_session.execute(
                text("""
                    SELECT severity, COUNT(*) as count 
                    FROM defects 
                    GROUP BY severity
                """)
            )
            stats["by_severity"] = {row.severity: row.count for row in result}
            
            # Priority breakdown
            result = await self.db_session.execute(
                text("""
                    SELECT priority, COUNT(*) as count 
                    FROM defects 
                    GROUP BY priority
                """)
            )
            stats["by_priority"] = {row.priority: row.count for row in result}
            
            # Type breakdown
            result = await self.db_session.execute(
                text("""
                    SELECT type, COUNT(*) as count 
                    FROM defects 
                    GROUP BY type
                """)
            )
            stats["by_type"] = {row.type: row.count for row in result}
            
            # Average resolution time
            result = await self.db_session.execute(
                text("""
                    SELECT AVG(time_to_resolution) as avg_resolution 
                    FROM defects 
                    WHERE time_to_resolution IS NOT NULL
                """)
            )
            avg_minutes = result.scalar()
            if avg_minutes:
                stats["avg_resolution_time_hours"] = round(avg_minutes / 60, 2)
            
            # SLA compliance
            for priority in DefectPriority:
                sla = self.sla_config[priority]
                result = await self.db_session.execute(
                    text("""
                        SELECT 
                            COUNT(*) as total,
                            COUNT(CASE WHEN time_to_resolution <= :sla_minutes THEN 1 END) as within_sla
                        FROM defects 
                        WHERE priority = :priority 
                        AND time_to_resolution IS NOT NULL
                    """),
                    {
                        "priority": priority.value,
                        "sla_minutes": sla["resolution_time_hours"] * 60
                    }
                )
                row = result.first()
                if row and row.total > 0:
                    compliance = (row.within_sla / row.total) * 100
                    stats["sla_compliance"][priority.value] = round(compliance, 2)
            
        except Exception as e:
            logger.error(f"Failed to get defect statistics: {e}")
        
        return stats


# ============================================================================
# Global Instance
# ============================================================================

defect_tracker = DefectTracker()