"""
Beta User Success Metrics Service
Track and analyze beta user success indicators and KPIs
"""
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as redis
import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class SuccessStage(Enum):
    """Beta user success stages"""

    SIGNUP = "signup"
    ONBOARDING = "onboarding"
    FIRST_USE = "first_use"
    REGULAR_USE = "regular_use"
    ADVANCED_USE = "advanced_use"
    ADVOCATE = "advocate"


class RiskLevel(Enum):
    """Churn risk levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SuccessMetric:
    """Individual success metric definition"""

    name: str
    description: str
    target_value: float
    weight: float  # Importance weight 0-1
    measurement_type: str  # 'rate', 'count', 'time', 'percentage'
    calculation_window: int  # hours


@dataclass
class UserSuccessProfile:
    """Comprehensive user success profile"""

    user_id: str
    email: str
    signup_date: datetime
    current_stage: SuccessStage
    overall_score: float  # 0-100
    risk_level: RiskLevel

    # Core success metrics
    onboarding_completion: float  # 0-100%
    time_to_first_video: Optional[int]  # minutes
    videos_generated_total: int
    videos_generated_7d: int
    videos_generated_24h: int

    # Engagement metrics
    daily_active_days_7d: int
    session_count_7d: int
    avg_session_duration: float  # minutes
    feature_adoption_score: float  # 0-100%

    # Quality metrics
    avg_video_performance: float  # 0-10
    published_video_ratio: float  # 0-1
    cost_efficiency: float  # revenue/cost

    # Behavior indicators
    support_tickets: int
    help_doc_views: int
    community_engagement: int
    referrals_made: int

    # Predictive indicators
    churn_probability: float  # 0-1
    lifetime_value_prediction: float
    next_likely_action: str

    # Timestamps
    last_activity: datetime
    last_video_generated: Optional[datetime]
    updated_at: datetime


@dataclass
class SuccessKPI:
    """Success KPI definition and current value"""

    name: str
    description: str
    current_value: float
    target_value: float
    trend: str  # 'up', 'down', 'stable'
    trend_percentage: float
    status: str  # 'on_track', 'at_risk', 'critical'
    measurement_unit: str
    category: str  # 'onboarding', 'engagement', 'retention', 'growth'


class BetaSuccessMetricsService:
    """Service for tracking and analyzing beta user success"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None

        # Success metrics configuration
        self.success_metrics = self._define_success_metrics()
        self.success_stages = self._define_success_stages()

        # KPI definitions
        self.kpi_definitions = self._define_kpis()

        # Thresholds for risk assessment
        self.risk_thresholds = {
            RiskLevel.LOW: {"score_min": 80, "activity_days": 5, "videos_7d": 10},
            RiskLevel.MEDIUM: {"score_min": 60, "activity_days": 3, "videos_7d": 5},
            RiskLevel.HIGH: {"score_min": 40, "activity_days": 1, "videos_7d": 1},
            RiskLevel.CRITICAL: {"score_min": 0, "activity_days": 0, "videos_7d": 0},
        }

        # Beta users cache
        self.beta_users: Dict[str, UserSuccessProfile] = {}
        self.last_refresh = datetime.utcnow()

    async def initialize(self):
        """Initialize the service"""
        try:
            self.redis_client = await redis.from_url(self.redis_url)

            # Start background tasks
            asyncio.create_task(self._metrics_calculator())
            asyncio.create_task(self._success_monitor())
            asyncio.create_task(self._kpi_updater())
            asyncio.create_task(self._intervention_detector())

            await self._load_beta_users()
            logger.info("Beta success metrics service initialized")

        except Exception as e:
            logger.error(f"Failed to initialize beta success metrics: {e}")
            raise

    def _define_success_metrics(self) -> Dict[str, SuccessMetric]:
        """Define success metrics for beta users"""
        return {
            "onboarding_completion": SuccessMetric(
                name="Onboarding Completion",
                description="Percentage of onboarding steps completed",
                target_value=100.0,
                weight=0.25,
                measurement_type="percentage",
                calculation_window=168,  # 7 days
            ),
            "time_to_first_video": SuccessMetric(
                name="Time to First Video",
                description="Time from signup to first video generation (minutes)",
                target_value=15.0,  # 15 minutes target
                weight=0.20,
                measurement_type="time",
                calculation_window=168,
            ),
            "video_generation_rate": SuccessMetric(
                name="Video Generation Rate",
                description="Videos generated per day",
                target_value=3.0,
                weight=0.20,
                measurement_type="rate",
                calculation_window=168,
            ),
            "feature_adoption": SuccessMetric(
                name="Feature Adoption",
                description="Percentage of core features used",
                target_value=80.0,
                weight=0.15,
                measurement_type="percentage",
                calculation_window=168,
            ),
            "engagement_consistency": SuccessMetric(
                name="Engagement Consistency",
                description="Active days per week",
                target_value=5.0,
                weight=0.10,
                measurement_type="count",
                calculation_window=168,
            ),
            "quality_score": SuccessMetric(
                name="Content Quality",
                description="Average video performance score",
                target_value=7.0,
                weight=0.10,
                measurement_type="rate",
                calculation_window=168,
            ),
        }

    def _define_success_stages(self) -> Dict[SuccessStage, Dict[str, Any]]:
        """Define success stages and their criteria"""
        return {
            SuccessStage.SIGNUP: {
                "description": "User has signed up",
                "criteria": {"signup_completed": True},
                "expected_actions": ["complete_profile", "connect_first_channel"],
            },
            SuccessStage.ONBOARDING: {
                "description": "User is completing onboarding",
                "criteria": {"onboarding_completion": 50.0},
                "expected_actions": ["generate_first_video", "explore_dashboard"],
            },
            SuccessStage.FIRST_USE: {
                "description": "User has generated first video",
                "criteria": {"videos_generated_total": 1, "time_to_first_video": 30},
                "expected_actions": ["publish_first_video", "check_analytics"],
            },
            SuccessStage.REGULAR_USE: {
                "description": "User is regularly generating content",
                "criteria": {"videos_generated_7d": 5, "daily_active_days_7d": 3},
                "expected_actions": ["optimize_costs", "use_advanced_features"],
            },
            SuccessStage.ADVANCED_USE: {
                "description": "User is using advanced features",
                "criteria": {"feature_adoption_score": 70.0, "videos_generated_7d": 10},
                "expected_actions": [
                    "manage_multiple_channels",
                    "automated_publishing",
                ],
            },
            SuccessStage.ADVOCATE: {
                "description": "User is a power user and potential advocate",
                "criteria": {"overall_score": 85.0, "referrals_made": 1},
                "expected_actions": ["provide_feedback", "participate_in_beta_program"],
            },
        }

    def _define_kpis(self) -> Dict[str, SuccessKPI]:
        """Define key performance indicators for beta success"""
        return {
            "activation_rate": SuccessKPI(
                name="Activation Rate",
                description="Percentage of users who generate first video within 24h",
                current_value=0.0,
                target_value=85.0,
                trend="stable",
                trend_percentage=0.0,
                status="on_track",
                measurement_unit="%",
                category="onboarding",
            ),
            "retention_7d": SuccessKPI(
                name="7-Day Retention",
                description="Percentage of users active after 7 days",
                current_value=0.0,
                target_value=70.0,
                trend="stable",
                trend_percentage=0.0,
                status="on_track",
                measurement_unit="%",
                category="retention",
            ),
            "engagement_rate": SuccessKPI(
                name="Daily Engagement Rate",
                description="Percentage of users active daily",
                current_value=0.0,
                target_value=40.0,
                trend="stable",
                trend_percentage=0.0,
                status="on_track",
                measurement_unit="%",
                category="engagement",
            ),
            "feature_adoption_rate": SuccessKPI(
                name="Core Feature Adoption",
                description="Average adoption rate of core features",
                current_value=0.0,
                target_value=75.0,
                trend="stable",
                trend_percentage=0.0,
                status="on_track",
                measurement_unit="%",
                category="engagement",
            ),
            "quality_threshold": SuccessKPI(
                name="Quality Threshold",
                description="Percentage of videos meeting quality standards",
                current_value=0.0,
                target_value=80.0,
                trend="stable",
                trend_percentage=0.0,
                status="on_track",
                measurement_unit="%",
                category="growth",
            ),
            "cost_efficiency": SuccessKPI(
                name="Cost Efficiency",
                description="Average cost per video under target",
                current_value=0.0,
                target_value=2.50,
                trend="stable",
                trend_percentage=0.0,
                status="on_track",
                measurement_unit="$",
                category="growth",
            ),
            "support_resolution": SuccessKPI(
                name="Support Resolution",
                description="Average time to resolve support tickets",
                current_value=0.0,
                target_value=2.0,
                trend="stable",
                trend_percentage=0.0,
                status="on_track",
                measurement_unit="hours",
                category="retention",
            ),
        }

    async def _load_beta_users(self):
        """Load beta users from database/Redis"""
        try:
            # Get beta user IDs
            beta_users_data = await self.redis_client.get("beta_users:list")
            if not beta_users_data:
                return

            beta_user_ids = json.loads(beta_users_data)

            for user_id in beta_user_ids:
                profile = await self._load_user_profile(user_id)
                if profile:
                    self.beta_users[user_id] = profile

            logger.info(f"Loaded {len(self.beta_users)} beta user profiles")

        except Exception as e:
            logger.error(f"Failed to load beta users: {e}")

    async def _load_user_profile(self, user_id: str) -> Optional[UserSuccessProfile]:
        """Load a user's success profile"""
        try:
            # Check for cached profile first
            profile_key = f"success_profile:{user_id}"
            cached_data = await self.redis_client.get(profile_key)

            if cached_data:
                profile_data = json.loads(cached_data)
                # Reconstruct profile with proper datetime objects
                profile_data["signup_date"] = datetime.fromisoformat(
                    profile_data["signup_date"]
                )
                profile_data["last_activity"] = datetime.fromisoformat(
                    profile_data["last_activity"]
                )
                profile_data["updated_at"] = datetime.fromisoformat(
                    profile_data["updated_at"]
                )

                if profile_data.get("last_video_generated"):
                    profile_data["last_video_generated"] = datetime.fromisoformat(
                        profile_data["last_video_generated"]
                    )

                profile_data["current_stage"] = SuccessStage(
                    profile_data["current_stage"]
                )
                profile_data["risk_level"] = RiskLevel(profile_data["risk_level"])

                return UserSuccessProfile(**profile_data)

            # If no cached profile, create new one
            return await self._create_user_profile(user_id)

        except Exception as e:
            logger.error(f"Failed to load profile for user {user_id}: {e}")
            return None

    async def _create_user_profile(self, user_id: str) -> Optional[UserSuccessProfile]:
        """Create a new user success profile"""
        try:
            # Get user data from various sources
            user_data = await self._gather_user_data(user_id)

            if not user_data:
                return None

            # Calculate success metrics
            metrics = await self._calculate_user_metrics(user_id, user_data)

            # Determine current stage
            current_stage = self._determine_success_stage(metrics)

            # Calculate risk level
            risk_level = self._calculate_risk_level(metrics)

            # Create profile
            profile = UserSuccessProfile(
                user_id=user_id,
                email=user_data.get("email", ""),
                signup_date=user_data.get("signup_date", datetime.utcnow()),
                current_stage=current_stage,
                overall_score=metrics.get("overall_score", 0.0),
                risk_level=risk_level,
                onboarding_completion=metrics.get("onboarding_completion", 0.0),
                time_to_first_video=metrics.get("time_to_first_video"),
                videos_generated_total=metrics.get("videos_generated_total", 0),
                videos_generated_7d=metrics.get("videos_generated_7d", 0),
                videos_generated_24h=metrics.get("videos_generated_24h", 0),
                daily_active_days_7d=metrics.get("daily_active_days_7d", 0),
                session_count_7d=metrics.get("session_count_7d", 0),
                avg_session_duration=metrics.get("avg_session_duration", 0.0),
                feature_adoption_score=metrics.get("feature_adoption_score", 0.0),
                avg_video_performance=metrics.get("avg_video_performance", 0.0),
                published_video_ratio=metrics.get("published_video_ratio", 0.0),
                cost_efficiency=metrics.get("cost_efficiency", 0.0),
                support_tickets=metrics.get("support_tickets", 0),
                help_doc_views=metrics.get("help_doc_views", 0),
                community_engagement=metrics.get("community_engagement", 0),
                referrals_made=metrics.get("referrals_made", 0),
                churn_probability=metrics.get("churn_probability", 0.5),
                lifetime_value_prediction=metrics.get("lifetime_value_prediction", 0.0),
                next_likely_action=metrics.get("next_likely_action", "unknown"),
                last_activity=user_data.get("last_activity", datetime.utcnow()),
                last_video_generated=user_data.get("last_video_generated"),
                updated_at=datetime.utcnow(),
            )

            # Cache the profile
            await self._cache_user_profile(profile)

            return profile

        except Exception as e:
            logger.error(f"Failed to create profile for user {user_id}: {e}")
            return None

    async def _gather_user_data(self, user_id: str) -> Dict[str, Any]:
        """Gather user data from various sources"""
        user_data = {}

        try:
            # Get basic user info
            user_key = f"user:{user_id}"
            basic_data = await self.redis_client.hgetall(user_key)
            if basic_data:
                user_data.update(basic_data)

            # Get events data
            today = datetime.utcnow().strftime("%Y%m%d")
            events_key = f"events:user:{user_id}:{today}"
            events_data = await self.redis_client.lrange(events_key, 0, -1)

            events = []
            for event_json in events_data:
                try:
                    event = json.loads(event_json)
                    events.append(event)
                except:
                    continue

            user_data["recent_events"] = events

            # Get video data
            videos_key = f"user_videos:{user_id}"
            videos_data = await self.redis_client.lrange(videos_key, 0, -1)
            user_data["videos"] = [json.loads(v) for v in videos_data if v]

            # Get session data
            sessions_key = f"user_sessions:{user_id}"
            sessions_data = await self.redis_client.lrange(sessions_key, 0, -1)
            user_data["sessions"] = [json.loads(s) for s in sessions_data if s]

            return user_data

        except Exception as e:
            logger.error(f"Failed to gather data for user {user_id}: {e}")
            return {}

    async def _calculate_user_metrics(
        self, user_id: str, user_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate all success metrics for a user"""
        metrics = {}

        try:
            events = user_data.get("recent_events", [])
            videos = user_data.get("videos", [])
            sessions = user_data.get("sessions", [])

            # Time-based calculations
            now = datetime.utcnow()
            seven_days_ago = now - timedelta(days=7)
            one_day_ago = now - timedelta(days=1)

            # Onboarding completion
            onboarding_steps = [
                "profile_setup",
                "channel_connect",
                "first_video",
                "first_publish",
            ]
            completed_steps = set(e.get("event_type") for e in events)
            metrics["onboarding_completion"] = (
                len(completed_steps & set(onboarding_steps)) / len(onboarding_steps)
            ) * 100

            # Time to first video
            signup_time = user_data.get("signup_date")
            first_video_event = next(
                (e for e in events if e.get("event_type") == "video_generate"), None
            )

            if signup_time and first_video_event:
                signup_dt = (
                    datetime.fromisoformat(signup_time)
                    if isinstance(signup_time, str)
                    else signup_time
                )
                first_video_dt = datetime.fromisoformat(first_video_event["timestamp"])
                metrics["time_to_first_video"] = int(
                    (first_video_dt - signup_dt).total_seconds() / 60
                )

            # Video generation metrics
            metrics["videos_generated_total"] = len(videos)

            video_events_7d = [
                e
                for e in events
                if e.get("event_type") == "video_generate"
                and datetime.fromisoformat(e["timestamp"]) > seven_days_ago
            ]
            metrics["videos_generated_7d"] = len(video_events_7d)

            video_events_24h = [
                e
                for e in events
                if e.get("event_type") == "video_generate"
                and datetime.fromisoformat(e["timestamp"]) > one_day_ago
            ]
            metrics["videos_generated_24h"] = len(video_events_24h)

            # Activity metrics
            active_days = set()
            for event in events:
                if datetime.fromisoformat(event["timestamp"]) > seven_days_ago:
                    active_days.add(datetime.fromisoformat(event["timestamp"]).date())
            metrics["daily_active_days_7d"] = len(active_days)

            # Session metrics
            recent_sessions = [
                s
                for s in sessions
                if datetime.fromisoformat(s.get("start_time", "")) > seven_days_ago
            ]
            metrics["session_count_7d"] = len(recent_sessions)

            if recent_sessions:
                durations = [s.get("duration_minutes", 0) for s in recent_sessions]
                metrics["avg_session_duration"] = np.mean(durations)
            else:
                metrics["avg_session_duration"] = 0.0

            # Feature adoption
            features_used = set()
            for event in events:
                if event.get("event_type") == "feature_use":
                    feature_name = event.get("data", {}).get("feature_name")
                    if feature_name:
                        features_used.add(feature_name)

            core_features = {
                "video_generate",
                "analytics_view",
                "channel_manage",
                "cost_tracking",
                "auto_publish",
            }
            metrics["feature_adoption_score"] = (
                len(features_used & core_features) / len(core_features)
            ) * 100

            # Quality metrics
            if videos:
                performance_scores = [
                    v.get("performance_score", 0)
                    for v in videos
                    if v.get("performance_score")
                ]
                metrics["avg_video_performance"] = (
                    np.mean(performance_scores) if performance_scores else 0.0
                )

                published_videos = len(
                    [v for v in videos if v.get("status") == "published"]
                )
                metrics["published_video_ratio"] = (
                    published_videos / len(videos) if videos else 0.0
                )
            else:
                metrics["avg_video_performance"] = 0.0
                metrics["published_video_ratio"] = 0.0

            # Cost efficiency
            cost_events = [e for e in events if e.get("event_type") == "cost_incurred"]
            revenue_events = [
                e for e in events if e.get("event_type") == "revenue_earned"
            ]

            total_cost = sum(e.get("data", {}).get("amount", 0) for e in cost_events)
            total_revenue = sum(
                e.get("data", {}).get("amount", 0) for e in revenue_events
            )

            metrics["cost_efficiency"] = (
                total_revenue / total_cost if total_cost > 0 else 0.0
            )

            # Support and engagement
            support_events = [
                e for e in events if e.get("event_type") == "support_ticket"
            ]
            metrics["support_tickets"] = len(support_events)

            help_events = [e for e in events if e.get("event_type") == "help_view"]
            metrics["help_doc_views"] = len(help_events)

            community_events = [
                e
                for e in events
                if e.get("event_type") in ["community_post", "community_comment"]
            ]
            metrics["community_engagement"] = len(community_events)

            referral_events = [
                e for e in events if e.get("event_type") == "referral_made"
            ]
            metrics["referrals_made"] = len(referral_events)

            # Calculate overall success score
            metrics["overall_score"] = self._calculate_overall_score(metrics)

            # Predictive metrics
            metrics["churn_probability"] = self._calculate_churn_probability(metrics)
            metrics["lifetime_value_prediction"] = self._predict_lifetime_value(metrics)
            metrics["next_likely_action"] = self._predict_next_action(metrics, events)

            return metrics

        except Exception as e:
            logger.error(f"Failed to calculate metrics for user {user_id}: {e}")
            return {}

    def _calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate weighted overall success score"""
        score = 0.0
        total_weight = 0.0

        for metric_name, metric_def in self.success_metrics.items():
            value = metrics.get(metric_name.replace("_", "_"), 0)

            if metric_def.measurement_type == "time":
                # For time metrics, lower is better
                normalized_score = max(
                    0,
                    min(
                        100,
                        (metric_def.target_value - value)
                        / metric_def.target_value
                        * 100,
                    ),
                )
            else:
                # For other metrics, higher is better
                normalized_score = min(100, (value / metric_def.target_value) * 100)

            score += normalized_score * metric_def.weight
            total_weight += metric_def.weight

        return score / total_weight if total_weight > 0 else 0.0

    def _determine_success_stage(self, metrics: Dict[str, Any]) -> SuccessStage:
        """Determine current success stage based on metrics"""
        # Check stages in reverse order (highest to lowest)
        for stage in [
            SuccessStage.ADVOCATE,
            SuccessStage.ADVANCED_USE,
            SuccessStage.REGULAR_USE,
            SuccessStage.FIRST_USE,
            SuccessStage.ONBOARDING,
            SuccessStage.SIGNUP,
        ]:
            criteria = self.success_stages[stage]["criteria"]
            meets_criteria = True

            for criterion, target in criteria.items():
                metric_value = metrics.get(criterion, 0)
                if isinstance(target, bool):
                    if not metric_value:
                        meets_criteria = False
                        break
                else:
                    if metric_value < target:
                        meets_criteria = False
                        break

            if meets_criteria:
                return stage

        return SuccessStage.SIGNUP

    def _calculate_risk_level(self, metrics: Dict[str, Any]) -> RiskLevel:
        """Calculate churn risk level"""
        overall_score = metrics.get("overall_score", 0)
        active_days = metrics.get("daily_active_days_7d", 0)
        videos_7d = metrics.get("videos_generated_7d", 0)

        # Check risk levels in order
        for risk_level in [
            RiskLevel.LOW,
            RiskLevel.MEDIUM,
            RiskLevel.HIGH,
            RiskLevel.CRITICAL,
        ]:
            thresholds = self.risk_thresholds[risk_level]

            if (
                overall_score >= thresholds["score_min"]
                and active_days >= thresholds["activity_days"]
                and videos_7d >= thresholds["videos_7d"]
            ):
                return risk_level

        return RiskLevel.CRITICAL

    def _calculate_churn_probability(self, metrics: Dict[str, Any]) -> float:
        """Calculate probability of churn based on metrics"""
        # Simple linear model for churn prediction
        factors = [
            metrics.get("overall_score", 0) / 100,  # Overall score (inverted)
            metrics.get("daily_active_days_7d", 0) / 7,  # Activity consistency
            min(
                1.0, metrics.get("videos_generated_7d", 0) / 7
            ),  # Video generation rate
            metrics.get("feature_adoption_score", 0) / 100,  # Feature adoption
            min(1.0, metrics.get("avg_session_duration", 0) / 30),  # Session duration
        ]

        # Weights for each factor
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]

        # Calculate weighted success probability
        success_prob = sum(factor * weight for factor, weight in zip(factors, weights))

        # Churn probability is inverse of success probability
        return max(0.0, min(1.0, 1.0 - success_prob))

    def _predict_lifetime_value(self, metrics: Dict[str, Any]) -> float:
        """Predict user lifetime value"""
        # Simple model based on current behavior
        monthly_videos = metrics.get("videos_generated_7d", 0) * 4.3  # Scale to monthly
        quality_factor = metrics.get("avg_video_performance", 0) / 10
        engagement_factor = metrics.get("overall_score", 0) / 100

        # Assume $10 value per video at full quality and engagement
        ltv = (
            monthly_videos * 10 * quality_factor * engagement_factor * 12
        )  # Annual LTV

        return max(0.0, ltv)

    def _predict_next_action(self, metrics: Dict[str, Any], events: List[Dict]) -> str:
        """Predict most likely next action for user"""
        # Analyze recent behavior patterns
        recent_events = [
            e
            for e in events
            if datetime.fromisoformat(e["timestamp"])
            > datetime.utcnow() - timedelta(days=2)
        ]

        event_counts = Counter(e.get("event_type") for e in recent_events)

        # Decision logic based on current state
        if metrics.get("videos_generated_24h", 0) == 0:
            return "generate_video"
        elif metrics.get("onboarding_completion", 0) < 100:
            return "complete_onboarding"
        elif metrics.get("feature_adoption_score", 0) < 50:
            return "explore_features"
        elif event_counts.get("analytics_view", 0) == 0:
            return "check_analytics"
        else:
            return "continue_video_generation"

    async def _cache_user_profile(self, profile: UserSuccessProfile):
        """Cache user profile in Redis"""
        profile_key = f"success_profile:{profile.user_id}"
        profile_data = asdict(profile)

        # Convert datetime objects to ISO strings for JSON serialization
        profile_data["signup_date"] = profile.signup_date.isoformat()
        profile_data["last_activity"] = profile.last_activity.isoformat()
        profile_data["updated_at"] = profile.updated_at.isoformat()

        if profile.last_video_generated:
            profile_data[
                "last_video_generated"
            ] = profile.last_video_generated.isoformat()

        profile_data["current_stage"] = profile.current_stage.value
        profile_data["risk_level"] = profile.risk_level.value

        await self.redis_client.setex(
            profile_key, 3600, json.dumps(profile_data)  # Cache for 1 hour
        )

    async def _metrics_calculator(self):
        """Background task to calculate metrics for all beta users"""
        while True:
            try:
                await asyncio.sleep(300)  # Update every 5 minutes

                for user_id in list(self.beta_users.keys()):
                    try:
                        await self._update_user_metrics(user_id)
                    except Exception as e:
                        logger.error(
                            f"Failed to update metrics for user {user_id}: {e}"
                        )

            except Exception as e:
                logger.error(f"Metrics calculator error: {e}")
                await asyncio.sleep(600)

    async def _update_user_metrics(self, user_id: str):
        """Update metrics for a specific user"""
        user_data = await self._gather_user_data(user_id)
        if not user_data:
            return

        metrics = await self._calculate_user_metrics(user_id, user_data)

        # Update cached profile
        if user_id in self.beta_users:
            profile = self.beta_users[user_id]

            # Update with new metrics
            profile.overall_score = metrics.get("overall_score", 0.0)
            profile.current_stage = self._determine_success_stage(metrics)
            profile.risk_level = self._calculate_risk_level(metrics)
            profile.onboarding_completion = metrics.get("onboarding_completion", 0.0)
            profile.videos_generated_7d = metrics.get("videos_generated_7d", 0)
            profile.videos_generated_24h = metrics.get("videos_generated_24h", 0)
            profile.daily_active_days_7d = metrics.get("daily_active_days_7d", 0)
            profile.feature_adoption_score = metrics.get("feature_adoption_score", 0.0)
            profile.churn_probability = metrics.get("churn_probability", 0.5)
            profile.updated_at = datetime.utcnow()

            await self._cache_user_profile(profile)

    async def _success_monitor(self):
        """Monitor success metrics and generate alerts"""
        while True:
            try:
                await asyncio.sleep(600)  # Check every 10 minutes

                # Check for users at risk
                high_risk_users = [
                    profile
                    for profile in self.beta_users.values()
                    if profile.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
                ]

                if high_risk_users:
                    await self._generate_risk_alerts(high_risk_users)

                # Check for success milestones
                successful_users = [
                    profile
                    for profile in self.beta_users.values()
                    if profile.overall_score > 85
                    and profile.current_stage == SuccessStage.ADVOCATE
                ]

                if successful_users:
                    await self._generate_success_alerts(successful_users)

            except Exception as e:
                logger.error(f"Success monitoring error: {e}")
                await asyncio.sleep(1200)

    async def _generate_risk_alerts(self, high_risk_users: List[UserSuccessProfile]):
        """Generate alerts for high-risk users"""
        for user in high_risk_users:
            alert = {
                "type": "churn_risk",
                "user_id": user.user_id,
                "risk_level": user.risk_level.value,
                "overall_score": user.overall_score,
                "recommended_actions": self._get_recommended_interventions(user),
                "timestamp": datetime.utcnow().isoformat(),
            }

            alert_key = f"alerts:success:risk:{datetime.utcnow().strftime('%Y%m%d%H')}"
            await self.redis_client.lpush(alert_key, json.dumps(alert))
            await self.redis_client.expire(alert_key, 86400)

    async def _generate_success_alerts(
        self, successful_users: List[UserSuccessProfile]
    ):
        """Generate alerts for successful users"""
        for user in successful_users:
            alert = {
                "type": "success_milestone",
                "user_id": user.user_id,
                "stage": user.current_stage.value,
                "overall_score": user.overall_score,
                "potential_actions": [
                    "request_testimonial",
                    "invite_to_case_study",
                    "referral_program",
                ],
                "timestamp": datetime.utcnow().isoformat(),
            }

            alert_key = (
                f"alerts:success:milestone:{datetime.utcnow().strftime('%Y%m%d%H')}"
            )
            await self.redis_client.lpush(alert_key, json.dumps(alert))
            await self.redis_client.expire(alert_key, 86400)

    def _get_recommended_interventions(self, profile: UserSuccessProfile) -> List[str]:
        """Get recommended interventions for at-risk user"""
        interventions = []

        if profile.onboarding_completion < 50:
            interventions.append("send_onboarding_reminder")

        if profile.videos_generated_7d == 0:
            interventions.append("schedule_video_generation_tutorial")

        if profile.feature_adoption_score < 30:
            interventions.append("provide_feature_walkthrough")

        if profile.daily_active_days_7d < 2:
            interventions.append("send_engagement_email")

        if profile.support_tickets > 2:
            interventions.append("schedule_support_call")

        return interventions

    async def _kpi_updater(self):
        """Update KPI values regularly"""
        while True:
            try:
                await asyncio.sleep(900)  # Update every 15 minutes
                await self._calculate_kpis()

            except Exception as e:
                logger.error(f"KPI updater error: {e}")
                await asyncio.sleep(1800)

    async def _calculate_kpis(self):
        """Calculate current KPI values"""
        if not self.beta_users:
            return

        profiles = list(self.beta_users.values())
        now = datetime.utcnow()

        # Activation Rate (users with first video in 24h)
        activated_users = len(
            [
                p
                for p in profiles
                if p.time_to_first_video and p.time_to_first_video <= 24 * 60
            ]
        )
        self.kpi_definitions["activation_rate"].current_value = (
            activated_users / len(profiles)
        ) * 100

        # 7-Day Retention
        week_ago = now - timedelta(days=7)
        retained_users = len([p for p in profiles if p.last_activity > week_ago])
        self.kpi_definitions["retention_7d"].current_value = (
            retained_users / len(profiles)
        ) * 100

        # Daily Engagement Rate
        daily_active = len([p for p in profiles if p.daily_active_days_7d >= 1])
        self.kpi_definitions["engagement_rate"].current_value = (
            daily_active / len(profiles)
        ) * 100

        # Feature Adoption Rate
        avg_feature_adoption = np.mean([p.feature_adoption_score for p in profiles])
        self.kpi_definitions[
            "feature_adoption_rate"
        ].current_value = avg_feature_adoption

        # Quality Threshold
        quality_users = len([p for p in profiles if p.avg_video_performance >= 7.0])
        self.kpi_definitions["quality_threshold"].current_value = (
            quality_users / len(profiles)
        ) * 100

        # Cost Efficiency
        avg_cost_efficiency = np.mean(
            [p.cost_efficiency for p in profiles if p.cost_efficiency > 0]
        )
        self.kpi_definitions["cost_efficiency"].current_value = avg_cost_efficiency

        # Store KPIs
        kpi_data = {name: asdict(kpi) for name, kpi in self.kpi_definitions.items()}
        kpi_key = f"kpis:success:{now.strftime('%Y%m%d%H%M')}"
        await self.redis_client.setex(kpi_key, 3600, json.dumps(kpi_data))

    async def _intervention_detector(self):
        """Detect when interventions are needed"""
        while True:
            try:
                await asyncio.sleep(1800)  # Check every 30 minutes

                for profile in self.beta_users.values():
                    # Check for intervention triggers
                    interventions_needed = []

                    # Onboarding stalled
                    if (
                        profile.current_stage == SuccessStage.ONBOARDING
                        and (datetime.utcnow() - profile.signup_date).days > 1
                    ):
                        interventions_needed.append("onboarding_stalled")

                    # No recent activity
                    if (datetime.utcnow() - profile.last_activity).hours > 48:
                        interventions_needed.append("inactive_user")

                    # High churn risk
                    if profile.churn_probability > 0.7:
                        interventions_needed.append("high_churn_risk")

                    # Support issues
                    if profile.support_tickets > 3:
                        interventions_needed.append("support_escalation")

                    if interventions_needed:
                        await self._trigger_interventions(profile, interventions_needed)

            except Exception as e:
                logger.error(f"Intervention detector error: {e}")
                await asyncio.sleep(3600)

    async def _trigger_interventions(
        self, profile: UserSuccessProfile, interventions: List[str]
    ):
        """Trigger specific interventions for a user"""
        intervention_data = {
            "user_id": profile.user_id,
            "interventions": interventions,
            "risk_level": profile.risk_level.value,
            "overall_score": profile.overall_score,
            "recommended_actions": self._get_recommended_interventions(profile),
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Store intervention trigger
        intervention_key = f"interventions:{profile.user_id}:{datetime.utcnow().strftime('%Y%m%d%H%M')}"
        await self.redis_client.setex(
            intervention_key, 86400 * 7, json.dumps(intervention_data)
        )

        # Add to intervention queue
        queue_key = "intervention_queue"
        await self.redis_client.lpush(queue_key, json.dumps(intervention_data))

    # Public API methods

    async def get_user_success_profile(
        self, user_id: str
    ) -> Optional[UserSuccessProfile]:
        """Get success profile for a specific user"""
        if user_id in self.beta_users:
            return self.beta_users[user_id]

        # Try to load from cache/database
        return await self._load_user_profile(user_id)

    async def get_success_summary(self) -> Dict[str, Any]:
        """Get overall success summary for all beta users"""
        if not self.beta_users:
            return {}

        profiles = list(self.beta_users.values())

        # Stage distribution
        stage_counts = Counter(p.current_stage for p in profiles)

        # Risk distribution
        risk_counts = Counter(p.risk_level for p in profiles)

        # Average metrics
        avg_metrics = {
            "overall_score": np.mean([p.overall_score for p in profiles]),
            "onboarding_completion": np.mean(
                [p.onboarding_completion for p in profiles]
            ),
            "feature_adoption": np.mean([p.feature_adoption_score for p in profiles]),
            "churn_probability": np.mean([p.churn_probability for p in profiles]),
            "videos_per_week": np.mean([p.videos_generated_7d for p in profiles]),
        }

        return {
            "total_users": len(profiles),
            "stage_distribution": {
                stage.value: count for stage, count in stage_counts.items()
            },
            "risk_distribution": {
                risk.value: count for risk, count in risk_counts.items()
            },
            "average_metrics": avg_metrics,
            "updated_at": datetime.utcnow().isoformat(),
        }

    async def get_kpis(self) -> Dict[str, SuccessKPI]:
        """Get current KPI values"""
        return self.kpi_definitions

    async def add_beta_user(self, user_id: str, email: str = "") -> bool:
        """Add a new beta user to tracking"""
        try:
            # Add to Redis beta users list
            current_users = await self.redis_client.get("beta_users:list")
            if current_users:
                beta_users = set(json.loads(current_users))
            else:
                beta_users = set()

            beta_users.add(user_id)
            await self.redis_client.set("beta_users:list", json.dumps(list(beta_users)))

            # Create initial profile
            profile = await self._create_user_profile(user_id)
            if profile:
                self.beta_users[user_id] = profile
                logger.info(f"Added beta user {user_id} to success tracking")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to add beta user {user_id}: {e}")
            return False

    async def remove_beta_user(self, user_id: str) -> bool:
        """Remove a user from beta tracking"""
        try:
            # Remove from Redis list
            current_users = await self.redis_client.get("beta_users:list")
            if current_users:
                beta_users = set(json.loads(current_users))
                beta_users.discard(user_id)
                await self.redis_client.set(
                    "beta_users:list", json.dumps(list(beta_users))
                )

            # Remove from memory
            if user_id in self.beta_users:
                del self.beta_users[user_id]

            # Clean up cached data
            profile_key = f"success_profile:{user_id}"
            await self.redis_client.delete(profile_key)

            logger.info(f"Removed beta user {user_id} from success tracking")
            return True

        except Exception as e:
            logger.error(f"Failed to remove beta user {user_id}: {e}")
            return False


# Global instance
beta_success_metrics_service = BetaSuccessMetricsService()
