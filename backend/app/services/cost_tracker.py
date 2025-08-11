"""
Cost Tracking Service
Monitors and manages costs for AI services and video generation
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
import json

from app.models.cost import Cost
from app.models.user import User
from app.core.config import settings

logger = logging.getLogger(__name__)


class ServiceType(Enum):
    """Service types for cost tracking"""
    OPENAI = "openai"
    ELEVENLABS = "elevenlabs"
    GOOGLE_TTS = "google_tts"
    DALLE = "dalle"
    YOUTUBE = "youtube"
    STORAGE = "storage"
    COMPUTE = "compute"


@dataclass
class CostLimit:
    """Cost limit configuration"""
    daily: float
    monthly: float
    per_video: float
    alert_threshold: float = 0.8  # Alert at 80% of limit


@dataclass
class CostMetrics:
    """Cost metrics for tracking"""
    service: str
    amount: float
    units: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class CostTracker:
    """Service for tracking and managing costs"""
    
    def __init__(self):
        self.limits = self._load_cost_limits()
        self.cache = {}  # Simple in-memory cache for quick lookups
        self.alert_callbacks = []
        
    def _load_cost_limits(self) -> Dict[str, CostLimit]:
        """Load cost limits from configuration"""
        return {
            ServiceType.OPENAI.value: CostLimit(
                daily=50.0,
                monthly=1500.0,
                per_video=1.0
            ),
            ServiceType.ELEVENLABS.value: CostLimit(
                daily=20.0,
                monthly=600.0,
                per_video=0.5
            ),
            ServiceType.GOOGLE_TTS.value: CostLimit(
                daily=10.0,
                monthly=300.0,
                per_video=0.3
            ),
            ServiceType.DALLE.value: CostLimit(
                daily=15.0,
                monthly=450.0,
                per_video=0.2
            ),
            ServiceType.YOUTUBE.value: CostLimit(
                daily=5.0,
                monthly=150.0,
                per_video=0.1
            )
        }
        
    async def track_cost(
        self,
        db: AsyncSession,
        user_id: str,
        service: str,
        amount: float,
        video_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Cost:
        """Track a cost entry"""
        try:
            # Check limits before tracking
            await self._check_limits(db, user_id, service, amount)
            
            # Create cost entry
            cost = Cost(
                user_id=user_id,
                video_id=video_id,
                service=service,
                amount=amount,
                currency="USD",
                extra_data=metadata or {},
                created_at=datetime.utcnow()
            )
            
            db.add(cost)
            await db.commit()
            await db.refresh(cost)
            
            # Update cache
            self._update_cache(user_id, service, amount)
            
            # Check for alerts
            await self._check_alerts(db, user_id, service)
            
            logger.info(f"Tracked cost: {service} - ${amount:.4f} for user {user_id}")
            
            return cost
            
        except Exception as e:
            logger.error(f"Cost tracking error: {e}")
            raise
            
    async def _check_limits(
        self,
        db: AsyncSession,
        user_id: str,
        service: str,
        amount: float
    ):
        """Check if cost would exceed limits"""
        if service not in self.limits:
            return  # No limits defined for this service
            
        limits = self.limits[service]
        
        # Check daily limit
        daily_total = await self.get_daily_cost(db, user_id, service)
        if daily_total + amount > limits.daily:
            raise ValueError(f"Daily limit exceeded for {service}: ${limits.daily}")
            
        # Check monthly limit
        monthly_total = await self.get_monthly_cost(db, user_id, service)
        if monthly_total + amount > limits.monthly:
            raise ValueError(f"Monthly limit exceeded for {service}: ${limits.monthly}")
            
    async def get_daily_cost(
        self,
        db: AsyncSession,
        user_id: str,
        service: Optional[str] = None
    ) -> float:
        """Get daily cost for user"""
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        query = select(func.sum(Cost.amount)).filter(
            Cost.user_id == user_id,
            Cost.created_at >= today
        )
        
        if service:
            query = query.filter(Cost.service == service)
            
        result = await db.execute(query)
        return result.scalar() or 0.0
        
    async def get_monthly_cost(
        self,
        db: AsyncSession,
        user_id: str,
        service: Optional[str] = None
    ) -> float:
        """Get monthly cost for user"""
        month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        query = select(func.sum(Cost.amount)).filter(
            Cost.user_id == user_id,
            Cost.created_at >= month_start
        )
        
        if service:
            query = query.filter(Cost.service == service)
            
        result = await db.execute(query)
        return result.scalar() or 0.0
        
    async def get_video_cost(
        self,
        db: AsyncSession,
        video_id: str
    ) -> Dict[str, Any]:
        """Get total cost for a video"""
        result = await db.execute(
            select(Cost.service, func.sum(Cost.amount)).filter(
                Cost.video_id == video_id
            ).group_by(Cost.service)
        )
        
        costs = {row[0]: row[1] for row in result}
        total = sum(costs.values())
        
        return {
            "total": total,
            "breakdown": costs,
            "video_id": video_id
        }
        
    async def get_cost_analytics(
        self,
        db: AsyncSession,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get detailed cost analytics"""
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()
            
        # Get costs by service
        result = await db.execute(
            select(
                Cost.service,
                func.sum(Cost.amount),
                func.count(Cost.id)
            ).filter(
                Cost.user_id == user_id,
                Cost.created_at >= start_date,
                Cost.created_at <= end_date
            ).group_by(Cost.service)
        )
        
        service_costs = {}
        for row in result:
            service_costs[row[0]] = {
                "total": float(row[1]),
                "count": row[2],
                "average": float(row[1]) / row[2] if row[2] > 0 else 0
            }
            
        # Get daily costs
        result = await db.execute(
            select(
                func.date(Cost.created_at),
                func.sum(Cost.amount)
            ).filter(
                Cost.user_id == user_id,
                Cost.created_at >= start_date,
                Cost.created_at <= end_date
            ).group_by(func.date(Cost.created_at))
        )
        
        daily_costs = [
            {
                "date": row[0].isoformat(),
                "amount": float(row[1])
            }
            for row in result
        ]
        
        # Get video costs
        result = await db.execute(
            select(
                Cost.video_id,
                func.sum(Cost.amount)
            ).filter(
                Cost.user_id == user_id,
                Cost.video_id.isnot(None),
                Cost.created_at >= start_date,
                Cost.created_at <= end_date
            ).group_by(Cost.video_id)
        )
        
        video_costs = [
            {
                "video_id": row[0],
                "total": float(row[1])
            }
            for row in result
        ]
        
        # Calculate totals and averages
        total_cost = sum(s["total"] for s in service_costs.values())
        avg_daily = total_cost / max((end_date - start_date).days, 1)
        avg_per_video = total_cost / len(video_costs) if video_costs else 0
        
        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "totals": {
                "all_services": total_cost,
                "average_daily": avg_daily,
                "average_per_video": avg_per_video
            },
            "by_service": service_costs,
            "daily_breakdown": daily_costs,
            "video_costs": video_costs,
            "limits": {
                service: {
                    "daily": limit.daily,
                    "monthly": limit.monthly,
                    "per_video": limit.per_video
                }
                for service, limit in self.limits.items()
            }
        }
        
    async def optimize_costs(
        self,
        db: AsyncSession,
        user_id: str
    ) -> Dict[str, Any]:
        """Get cost optimization recommendations"""
        analytics = await self.get_cost_analytics(db, user_id)
        recommendations = []
        
        # Check service usage patterns
        for service, data in analytics["by_service"].items():
            if service in self.limits:
                limit = self.limits[service]
                
                # Check if approaching limits
                daily_usage = data["total"] / 30  # Approximate
                if daily_usage > limit.daily * 0.8:
                    recommendations.append({
                        "service": service,
                        "type": "limit_warning",
                        "message": f"{service} approaching daily limit",
                        "suggestion": f"Consider optimizing {service} usage or increasing limits"
                    })
                    
                # Check cost per video
                if data["average"] > limit.per_video:
                    recommendations.append({
                        "service": service,
                        "type": "efficiency",
                        "message": f"{service} cost per video exceeds target",
                        "current": data["average"],
                        "target": limit.per_video,
                        "suggestion": "Optimize prompts or use lower-tier models"
                    })
                    
        # Service-specific recommendations
        if ServiceType.OPENAI.value in analytics["by_service"]:
            openai_data = analytics["by_service"][ServiceType.OPENAI.value]
            if openai_data["average"] > 0.5:
                recommendations.append({
                    "service": ServiceType.OPENAI.value,
                    "type": "optimization",
                    "message": "High OpenAI costs detected",
                    "suggestion": "Use GPT-3.5-turbo for non-critical tasks"
                })
                
        if ServiceType.ELEVENLABS.value in analytics["by_service"]:
            elevenlabs_data = analytics["by_service"][ServiceType.ELEVENLABS.value]
            if elevenlabs_data["average"] > 0.3:
                recommendations.append({
                    "service": ServiceType.ELEVENLABS.value,
                    "type": "optimization",
                    "message": "High voice synthesis costs",
                    "suggestion": "Consider Google TTS for some videos"
                })
                
        # Overall recommendations
        if analytics["totals"]["average_per_video"] > 3.0:
            recommendations.append({
                "type": "overall",
                "message": "Video cost exceeds $3 target",
                "current": analytics["totals"]["average_per_video"],
                "target": 3.0,
                "suggestion": "Review all service usage and optimize pipeline"
            })
            
        return {
            "current_efficiency": {
                "cost_per_video": analytics["totals"]["average_per_video"],
                "daily_average": analytics["totals"]["average_daily"],
                "target_per_video": 3.0,
                "efficiency_score": min(3.0 / analytics["totals"]["average_per_video"], 1.0) * 100
                if analytics["totals"]["average_per_video"] > 0 else 100
            },
            "recommendations": recommendations,
            "potential_savings": self._calculate_potential_savings(analytics)
        }
        
    def _calculate_potential_savings(self, analytics: Dict) -> Dict[str, float]:
        """Calculate potential cost savings"""
        savings = {}
        
        # Model optimization savings
        if ServiceType.OPENAI.value in analytics["by_service"]:
            openai_total = analytics["by_service"][ServiceType.OPENAI.value]["total"]
            # Assume 40% savings by using GPT-3.5 more
            savings["model_optimization"] = openai_total * 0.4
            
        # Voice optimization savings
        if ServiceType.ELEVENLABS.value in analytics["by_service"]:
            elevenlabs_total = analytics["by_service"][ServiceType.ELEVENLABS.value]["total"]
            # Assume 30% savings by using Google TTS for some videos
            savings["voice_optimization"] = elevenlabs_total * 0.3
            
        # Caching savings (reduce redundant API calls)
        total = analytics["totals"]["all_services"]
        savings["caching_improvement"] = total * 0.15
        
        savings["total_potential"] = sum(savings.values())
        
        return savings
        
    def _update_cache(self, user_id: str, service: str, amount: float):
        """Update in-memory cache"""
        cache_key = f"{user_id}:{service}:{datetime.utcnow().date()}"
        if cache_key in self.cache:
            self.cache[cache_key] += amount
        else:
            self.cache[cache_key] = amount
            
    async def _check_alerts(self, db: AsyncSession, user_id: str, service: str):
        """Check if alerts should be triggered"""
        if service not in self.limits:
            return
            
        limits = self.limits[service]
        daily_total = await self.get_daily_cost(db, user_id, service)
        
        # Check if approaching limit
        if daily_total > limits.daily * limits.alert_threshold:
            alert_data = {
                "user_id": user_id,
                "service": service,
                "current": daily_total,
                "limit": limits.daily,
                "percentage": (daily_total / limits.daily) * 100,
                "type": "daily_limit_warning"
            }
            
            # Trigger alert callbacks
            for callback in self.alert_callbacks:
                await callback(alert_data)
                
            logger.warning(f"Cost alert: {service} at {alert_data['percentage']:.1f}% of daily limit for user {user_id}")
            
    def register_alert_callback(self, callback):
        """Register a callback for cost alerts"""
        self.alert_callbacks.append(callback)
        
    async def reset_daily_cache(self):
        """Reset daily cache (should be called by scheduler)"""
        current_date = datetime.utcnow().date()
        keys_to_remove = [
            key for key in self.cache.keys()
            if not key.endswith(str(current_date))
        ]
        for key in keys_to_remove:
            del self.cache[key]
            
        logger.info(f"Reset daily cache, removed {len(keys_to_remove)} entries")

# Global instance
cost_tracker = CostTracker()
