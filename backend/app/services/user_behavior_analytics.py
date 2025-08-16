"""
User Behavior Analytics Service
Comprehensive user behavior tracking and analysis
"""
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, case, distinct
from sqlalchemy.orm import selectinload
import numpy as np
import pandas as pd
from collections import defaultdict
import logging
import json
import hashlib
from enum import Enum

from app.models.user import User
from app.core.cache import cache_service

logger = logging.getLogger(__name__)


class EventType(Enum):
    """User event types"""

    PAGE_VIEW = "page_view"
    CLICK = "click"
    FORM_SUBMIT = "form_submit"
    VIDEO_GENERATE = "video_generate"
    CHANNEL_CREATE = "channel_create"
    FEATURE_USE = "feature_use"
    ERROR = "error"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    SIGNUP = "signup"
    LOGIN = "login"
    LOGOUT = "logout"
    UPGRADE = "upgrade"
    DOWNGRADE = "downgrade"


class UserBehaviorAnalyticsService:
    """Service for tracking and analyzing user behavior"""

    def __init__(self):
        self.cache_ttl = 300  # 5 minutes cache
        self.session_timeout = 30 * 60  # 30 minutes session timeout

    async def track_event(
        self,
        db: AsyncSession,
        user_id: int,
        event_type: str,
        event_data: Dict[str, Any],
        session_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Track a user behavior event"""
        if not timestamp:
            timestamp = datetime.utcnow()

        # Generate session ID if not provided
        if not session_id:
            session_id = await self._get_or_create_session(db, user_id)

        # Create event record
        from app.models.user_event import UserEvent

        event = UserEvent(
            user_id=user_id,
            session_id=session_id,
            event_type=event_type,
            event_data=json.dumps(event_data),
            timestamp=timestamp,
            page_url=event_data.get("page_url"),
            referrer=event_data.get("referrer"),
            user_agent=event_data.get("user_agent"),
            ip_address=event_data.get("ip_address"),
        )

        db.add(event)
        await db.commit()

        # Update session activity
        await self._update_session_activity(db, session_id, timestamp)

        # Process event for real-time analytics
        await self._process_real_time_event(user_id, event_type, event_data)

        return {"event_id": event.id, "session_id": session_id, "tracked": True}

    async def get_behavior_overview(
        self,
        db: AsyncSession,
        user_id: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get comprehensive behavior analytics overview"""
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)

        # Build base query
        from app.models.user_event import UserEvent

        base_query = select(UserEvent).where(
            and_(UserEvent.timestamp >= start_date, UserEvent.timestamp <= end_date)
        )

        if user_id:
            base_query = base_query.where(UserEvent.user_id == user_id)

        # Get total events
        total_events_query = select(func.count(UserEvent.id)).where(
            and_(UserEvent.timestamp >= start_date, UserEvent.timestamp <= end_date)
        )
        if user_id:
            total_events_query = total_events_query.where(UserEvent.user_id == user_id)

        total_events_result = await db.execute(total_events_query)
        total_events = total_events_result.scalar() or 0

        # Get unique users
        unique_users_query = select(func.count(distinct(UserEvent.user_id))).where(
            and_(UserEvent.timestamp >= start_date, UserEvent.timestamp <= end_date)
        )
        if user_id:
            unique_users = 1
        else:
            unique_users_result = await db.execute(unique_users_query)
            unique_users = unique_users_result.scalar() or 0

        # Get event breakdown
        event_breakdown = await self._get_event_breakdown(
            db, user_id, start_date, end_date
        )

        # Get user journey stats
        journey_stats = await self._analyze_user_journeys(
            db, user_id, start_date, end_date
        )

        # Get feature usage
        feature_usage = await self._analyze_feature_usage(
            db, user_id, start_date, end_date
        )

        # Get session stats
        session_stats = await self._analyze_sessions(db, user_id, start_date, end_date)

        return {
            "total_events": total_events,
            "unique_users": unique_users,
            "event_breakdown": event_breakdown,
            "journey_stats": journey_stats,
            "feature_usage": feature_usage,
            "session_stats": session_stats,
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        }

    async def get_conversion_funnels(
        self,
        db: AsyncSession,
        funnel_steps: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Analyze conversion funnels"""
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)

        from app.models.user_event import UserEvent

        funnel_data = []
        total_users = set()

        for i, step in enumerate(funnel_steps):
            # Get users who completed this step
            step_query = select(distinct(UserEvent.user_id)).where(
                and_(
                    UserEvent.event_type == step,
                    UserEvent.timestamp >= start_date,
                    UserEvent.timestamp <= end_date,
                )
            )

            step_result = await db.execute(step_query)
            step_users = set(row[0] for row in step_result.fetchall())

            if i == 0:
                total_users = step_users
                conversion_rate = 100.0
            else:
                conversion_rate = (
                    (len(step_users & total_users) / len(total_users) * 100)
                    if total_users
                    else 0
                )

            funnel_data.append(
                {
                    "step": step,
                    "step_number": i + 1,
                    "users": len(step_users),
                    "conversion_rate": conversion_rate,
                    "drop_off_rate": 100 - conversion_rate if i > 0 else 0,
                }
            )

            # Update total users to only include those who completed this step
            total_users = step_users & total_users

        # Calculate overall conversion
        overall_conversion = (
            (len(total_users) / len(funnel_data[0]["users"]) * 100)
            if funnel_data and funnel_data[0]["users"]
            else 0
        )

        return {
            "funnel_name": " → ".join(funnel_steps),
            "steps": funnel_data,
            "overall_conversion": overall_conversion,
            "total_completions": len(total_users),
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        }

    async def get_cohort_analysis(
        self,
        db: AsyncSession,
        cohort_type: str = "signup",
        metric: str = "retention",
        periods: int = 6,
    ) -> Dict[str, Any]:
        """Perform cohort analysis"""
        from app.models.user import User
        from app.models.user_event import UserEvent

        # Get cohorts based on signup date
        end_date = datetime.utcnow()
        cohort_data = []

        for period in range(periods):
            cohort_start = end_date - timedelta(days=(period + 1) * 7)
            cohort_end = end_date - timedelta(days=period * 7)

            # Get users in this cohort
            cohort_users_query = select(User.id).where(
                and_(User.created_at >= cohort_start, User.created_at < cohort_end)
            )
            cohort_users_result = await db.execute(cohort_users_query)
            cohort_user_ids = [row[0] for row in cohort_users_result.fetchall()]

            if not cohort_user_ids:
                continue

            cohort_size = len(cohort_user_ids)
            retention_data = []

            # Calculate retention for each subsequent period
            for retention_period in range(min(period + 1, 6)):
                retention_start = cohort_end + timedelta(days=retention_period * 7)
                retention_end = retention_start + timedelta(days=7)

                # Count active users in retention period
                active_users_query = select(
                    func.count(distinct(UserEvent.user_id))
                ).where(
                    and_(
                        UserEvent.user_id.in_(cohort_user_ids),
                        UserEvent.timestamp >= retention_start,
                        UserEvent.timestamp < retention_end,
                    )
                )
                active_users_result = await db.execute(active_users_query)
                active_users = active_users_result.scalar() or 0

                retention_rate = (
                    (active_users / cohort_size * 100) if cohort_size > 0 else 0
                )
                retention_data.append(
                    {
                        "period": retention_period,
                        "active_users": active_users,
                        "retention_rate": retention_rate,
                    }
                )

            cohort_data.append(
                {
                    "cohort": f'Week of {cohort_start.strftime("%b %d")}',
                    "size": cohort_size,
                    "retention": retention_data,
                }
            )

        return {
            "cohort_type": cohort_type,
            "metric": metric,
            "cohorts": cohort_data,
            "periods": periods,
        }

    async def get_feature_heatmap(
        self,
        db: AsyncSession,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Generate feature usage heatmap data"""
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=7)

        from app.models.user_event import UserEvent

        # Get feature usage by hour and day
        heatmap_query = (
            select(
                func.date(UserEvent.timestamp).label("date"),
                func.extract("hour", UserEvent.timestamp).label("hour"),
                func.count(UserEvent.id).label("event_count"),
            )
            .where(
                and_(
                    UserEvent.event_type == EventType.FEATURE_USE.value,
                    UserEvent.timestamp >= start_date,
                    UserEvent.timestamp <= end_date,
                )
            )
            .group_by(
                func.date(UserEvent.timestamp),
                func.extract("hour", UserEvent.timestamp),
            )
            .order_by(
                func.date(UserEvent.timestamp),
                func.extract("hour", UserEvent.timestamp),
            )
        )

        heatmap_result = await db.execute(heatmap_query)
        heatmap_data = heatmap_result.fetchall()

        # Format heatmap data
        heatmap = defaultdict(lambda: defaultdict(int))
        max_value = 0

        for row in heatmap_data:
            date_str = row.date.isoformat()
            hour = int(row.hour)
            count = row.event_count
            heatmap[date_str][hour] = count
            max_value = max(max_value, count)

        # Convert to list format
        heatmap_list = []
        for date_str, hours in heatmap.items():
            for hour, count in hours.items():
                heatmap_list.append(
                    {
                        "date": date_str,
                        "hour": hour,
                        "value": count,
                        "intensity": (count / max_value) if max_value > 0 else 0,
                    }
                )

        return {
            "heatmap": heatmap_list,
            "max_value": max_value,
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        }

    async def get_user_segments(
        self, db: AsyncSession, segmentation_criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Segment users based on behavior patterns"""
        from app.models.user import User
        from app.models.user_event import UserEvent

        segments = {}

        # Power users (high activity)
        power_users_query = (
            select(UserEvent.user_id, func.count(UserEvent.id).label("event_count"))
            .where(UserEvent.timestamp >= datetime.utcnow() - timedelta(days=30))
            .group_by(UserEvent.user_id)
            .having(func.count(UserEvent.id) > 100)
        )
        power_users_result = await db.execute(power_users_query)
        power_users = [row[0] for row in power_users_result.fetchall()]
        segments["power_users"] = {
            "count": len(power_users),
            "user_ids": power_users[:100],  # Limit for performance
        }

        # At-risk users (declining activity)
        # Compare last 7 days to previous 7 days
        recent_period = datetime.utcnow() - timedelta(days=7)
        previous_period = datetime.utcnow() - timedelta(days=14)

        recent_activity_query = (
            select(UserEvent.user_id, func.count(UserEvent.id).label("recent_count"))
            .where(UserEvent.timestamp >= recent_period)
            .group_by(UserEvent.user_id)
        )
        recent_activity_result = await db.execute(recent_activity_query)
        recent_activity = {row[0]: row[1] for row in recent_activity_result.fetchall()}

        previous_activity_query = (
            select(UserEvent.user_id, func.count(UserEvent.id).label("previous_count"))
            .where(
                and_(
                    UserEvent.timestamp >= previous_period,
                    UserEvent.timestamp < recent_period,
                )
            )
            .group_by(UserEvent.user_id)
        )
        previous_activity_result = await db.execute(previous_activity_query)
        previous_activity = {
            row[0]: row[1] for row in previous_activity_result.fetchall()
        }

        at_risk_users = []
        for user_id, previous_count in previous_activity.items():
            recent_count = recent_activity.get(user_id, 0)
            if recent_count < previous_count * 0.5:  # 50% drop in activity
                at_risk_users.append(user_id)

        segments["at_risk"] = {
            "count": len(at_risk_users),
            "user_ids": at_risk_users[:100],
        }

        # New users (signed up in last 7 days)
        new_users_query = select(User.id).where(
            User.created_at >= datetime.utcnow() - timedelta(days=7)
        )
        new_users_result = await db.execute(new_users_query)
        new_users = [row[0] for row in new_users_result.fetchall()]
        segments["new_users"] = {"count": len(new_users), "user_ids": new_users[:100]}

        # Dormant users (no activity in 30 days)
        dormant_users_query = (
            select(User.id)
            .outerjoin(
                UserEvent,
                and_(
                    User.id == UserEvent.user_id,
                    UserEvent.timestamp >= datetime.utcnow() - timedelta(days=30),
                ),
            )
            .where(UserEvent.id.is_(None))
        )
        dormant_users_result = await db.execute(dormant_users_query)
        dormant_users = [row[0] for row in dormant_users_result.fetchall()]
        segments["dormant"] = {
            "count": len(dormant_users),
            "user_ids": dormant_users[:100],
        }

        return {
            "segments": segments,
            "total_users": sum(s["count"] for s in segments.values()),
            "criteria": segmentation_criteria,
        }

    # Private helper methods
    async def _get_or_create_session(self, db: AsyncSession, user_id: int) -> str:
        """Get existing session or create new one"""
        from app.models.user_session import UserSession

        # Check for active session
        active_session_query = (
            select(UserSession)
            .where(
                and_(
                    UserSession.user_id == user_id,
                    UserSession.is_active == True,
                    UserSession.last_activity
                    >= datetime.utcnow() - timedelta(seconds=self.session_timeout),
                )
            )
            .order_by(UserSession.last_activity.desc())
        )

        active_session_result = await db.execute(active_session_query)
        active_session = active_session_result.scalar_one_or_none()

        if active_session:
            return active_session.session_id

        # Create new session
        session_id = hashlib.sha256(
            f"{user_id}{datetime.utcnow()}".encode()
        ).hexdigest()
        new_session = UserSession(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            is_active=True,
        )

        db.add(new_session)
        await db.commit()

        return session_id

    async def _update_session_activity(
        self, db: AsyncSession, session_id: str, timestamp: datetime
    ):
        """Update session last activity time"""
        from app.models.user_session import UserSession

        session_query = select(UserSession).where(UserSession.session_id == session_id)
        session_result = await db.execute(session_query)
        session = session_result.scalar_one_or_none()

        if session:
            session.last_activity = timestamp
            session.event_count = (session.event_count or 0) + 1

            # Check if session should be ended
            if (timestamp - session.start_time).total_seconds() > self.session_timeout:
                session.is_active = False
                session.end_time = timestamp
                session.duration_seconds = int(
                    (timestamp - session.start_time).total_seconds()
                )

            await db.commit()

    async def _process_real_time_event(
        self, user_id: int, event_type: str, event_data: Dict[str, Any]
    ):
        """Process event for real-time analytics"""
        # Update real-time metrics in cache
        cache_key = f"realtime:events:{datetime.utcnow().strftime('%Y%m%d%H')}"

        current_data = await cache_service.get(cache_key) or {
            "total_events": 0,
            "unique_users": set(),
            "event_types": defaultdict(int),
        }

        current_data["total_events"] += 1
        current_data["unique_users"].add(user_id)
        current_data["event_types"][event_type] += 1

        await cache_service.set(cache_key, current_data, 3600)  # 1 hour cache

    async def _get_event_breakdown(
        self,
        db: AsyncSession,
        user_id: Optional[int],
        start_date: datetime,
        end_date: datetime,
    ) -> List[Dict[str, Any]]:
        """Get breakdown of events by type"""
        from app.models.user_event import UserEvent

        breakdown_query = (
            select(UserEvent.event_type, func.count(UserEvent.id).label("count"))
            .where(
                and_(UserEvent.timestamp >= start_date, UserEvent.timestamp <= end_date)
            )
            .group_by(UserEvent.event_type)
            .order_by(func.count(UserEvent.id).desc())
        )

        if user_id:
            breakdown_query = breakdown_query.where(UserEvent.user_id == user_id)

        breakdown_result = await db.execute(breakdown_query)
        breakdown_data = breakdown_result.fetchall()

        total_events = sum(row.count for row in breakdown_data)

        return [
            {
                "event_type": row.event_type,
                "count": row.count,
                "percentage": (row.count / total_events * 100)
                if total_events > 0
                else 0,
            }
            for row in breakdown_data
        ]

    async def _analyze_user_journeys(
        self,
        db: AsyncSession,
        user_id: Optional[int],
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """Analyze common user journeys"""
        from app.models.user_event import UserEvent

        # Get sequences of events per session
        journey_query = (
            select(
                UserEvent.session_id,
                UserEvent.user_id,
                UserEvent.event_type,
                UserEvent.timestamp,
            )
            .where(
                and_(UserEvent.timestamp >= start_date, UserEvent.timestamp <= end_date)
            )
            .order_by(UserEvent.session_id, UserEvent.timestamp)
        )

        if user_id:
            journey_query = journey_query.where(UserEvent.user_id == user_id)

        journey_result = await db.execute(journey_query)
        journey_data = journey_result.fetchall()

        # Group events by session
        sessions = defaultdict(list)
        for row in journey_data:
            sessions[row.session_id].append(row.event_type)

        # Find common patterns
        journey_patterns = defaultdict(int)
        for session_events in sessions.values():
            if len(session_events) >= 2:
                # Create 2-step patterns
                for i in range(len(session_events) - 1):
                    pattern = f"{session_events[i]} → {session_events[i+1]}"
                    journey_patterns[pattern] += 1

        # Get top patterns
        top_patterns = sorted(
            journey_patterns.items(), key=lambda x: x[1], reverse=True
        )[:10]

        return {
            "total_sessions": len(sessions),
            "avg_events_per_session": np.mean(
                [len(events) for events in sessions.values()]
            )
            if sessions
            else 0,
            "top_patterns": [
                {"pattern": pattern, "count": count} for pattern, count in top_patterns
            ],
        }

    async def _analyze_feature_usage(
        self,
        db: AsyncSession,
        user_id: Optional[int],
        start_date: datetime,
        end_date: datetime,
    ) -> List[Dict[str, Any]]:
        """Analyze feature usage patterns"""
        from app.models.user_event import UserEvent

        feature_query = (
            select(UserEvent.event_data, func.count(UserEvent.id).label("usage_count"))
            .where(
                and_(
                    UserEvent.event_type == EventType.FEATURE_USE.value,
                    UserEvent.timestamp >= start_date,
                    UserEvent.timestamp <= end_date,
                )
            )
            .group_by(UserEvent.event_data)
        )

        if user_id:
            feature_query = feature_query.where(UserEvent.user_id == user_id)

        feature_result = await db.execute(feature_query)
        feature_data = feature_result.fetchall()

        # Parse feature names from event data
        feature_usage = defaultdict(int)
        for row in feature_data:
            try:
                data = json.loads(row.event_data)
                feature_name = data.get("feature_name", "unknown")
                feature_usage[feature_name] += row.usage_count
            except:
                continue

        # Sort by usage
        sorted_features = sorted(
            feature_usage.items(), key=lambda x: x[1], reverse=True
        )

        return [
            {
                "feature": feature,
                "usage_count": count,
                "adoption_rate": 0,  # Would need total users to calculate
            }
            for feature, count in sorted_features[:20]
        ]

    async def _analyze_sessions(
        self,
        db: AsyncSession,
        user_id: Optional[int],
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """Analyze session statistics"""
        from app.models.user_session import UserSession

        session_query = select(UserSession).where(
            and_(
                UserSession.start_time >= start_date, UserSession.start_time <= end_date
            )
        )

        if user_id:
            session_query = session_query.where(UserSession.user_id == user_id)

        session_result = await db.execute(session_query)
        sessions = session_result.scalars().all()

        if not sessions:
            return {"total_sessions": 0, "avg_duration": 0, "bounce_rate": 0}

        durations = [s.duration_seconds for s in sessions if s.duration_seconds]
        bounce_sessions = sum(1 for s in sessions if (s.event_count or 0) <= 1)

        return {
            "total_sessions": len(sessions),
            "avg_duration": np.mean(durations) if durations else 0,
            "median_duration": np.median(durations) if durations else 0,
            "bounce_rate": (bounce_sessions / len(sessions) * 100) if sessions else 0,
            "avg_events_per_session": np.mean([s.event_count or 0 for s in sessions]),
        }


# Create singleton instance
user_behavior_analytics_service = UserBehaviorAnalyticsService()
