"""
AI/ML Cost Optimization Service
Implements strategies to achieve <$3 per video target
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import asyncio
from functools import lru_cache

import redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
import openai
from anthropic import Anthropic

from app.db.session import get_db
from app.models.cost import Cost
from app.models.video import Video
from app.core.config import settings

logger = logging.getLogger(__name__)

# Redis for caching
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    db=int(os.getenv("REDIS_DB", "2")),  # DB 2 for cost optimization
    decode_responses=True
)


class ModelTier(Enum):
    """AI Model tiers for cost optimization"""
    PREMIUM = "premium"  # GPT-4, Claude-3-opus
    STANDARD = "standard"  # GPT-3.5-turbo, Claude-3-sonnet
    ECONOMY = "economy"  # GPT-3.5, Claude-instant
    CACHED = "cached"  # Use cached responses


class ServiceType(Enum):
    """AI Service types"""
    SCRIPT_GENERATION = "script_generation"
    VOICE_SYNTHESIS = "voice_synthesis"
    IMAGE_GENERATION = "image_generation"
    VIDEO_PROCESSING = "video_processing"
    QUALITY_CHECK = "quality_check"
    TREND_ANALYSIS = "trend_analysis"


@dataclass
class CostProfile:
    """Cost profile for a service"""
    service: ServiceType
    tier: ModelTier
    cost_per_call: float
    avg_latency_ms: int
    quality_score: float  # 0-100
    cache_ttl_seconds: int


@dataclass
class OptimizationStrategy:
    """Cost optimization strategy"""
    name: str
    description: str
    savings_percentage: float
    quality_impact: float  # -100 to +100
    implementation_complexity: str  # low, medium, high


class CostOptimizer:
    """
    AI/ML Cost Optimization Service
    Target: <$3 per video with quality thresholds
    """
    
    # Cost limits per service (in USD)
    COST_LIMITS = {
        ServiceType.SCRIPT_GENERATION: 0.50,
        ServiceType.VOICE_SYNTHESIS: 1.00,
        ServiceType.IMAGE_GENERATION: 0.80,
        ServiceType.VIDEO_PROCESSING: 0.50,
        ServiceType.QUALITY_CHECK: 0.10,
        ServiceType.TREND_ANALYSIS: 0.10
    }
    
    # Daily budget limits
    DAILY_BUDGET = {
        "openai": 50.00,
        "elevenlabs": 20.00,
        "anthropic": 30.00,
        "google": 10.00,
        "total": 100.00
    }
    
    # Model costs per 1K tokens (approximate)
    MODEL_COSTS = {
        "gpt-4-turbo": 0.03,
        "gpt-3.5-turbo": 0.002,
        "claude-3-opus": 0.03,
        "claude-3-sonnet": 0.012,
        "claude-instant": 0.004,
    }
    
    # Cache TTL settings (in seconds)
    CACHE_TTL = {
        ServiceType.TREND_ANALYSIS: 3600,  # 1 hour
        ServiceType.SCRIPT_GENERATION: 900,  # 15 minutes for similar topics
        ServiceType.QUALITY_CHECK: 1800,  # 30 minutes
        ServiceType.IMAGE_GENERATION: 86400,  # 24 hours for same prompts
    }
    
    def __init__(self):
        """Initialize cost optimizer"""
        self.redis = redis_client
        self.current_costs = {}
        self.optimization_strategies = self._load_strategies()
        
    def _load_strategies(self) -> List[OptimizationStrategy]:
        """Load optimization strategies"""
        return [
            OptimizationStrategy(
                name="Progressive Model Fallback",
                description="Start with GPT-3.5, fallback to GPT-4 only when needed",
                savings_percentage=70,
                quality_impact=-5,
                implementation_complexity="low"
            ),
            OptimizationStrategy(
                name="Aggressive Caching",
                description="Cache all responses with smart key generation",
                savings_percentage=40,
                quality_impact=0,
                implementation_complexity="medium"
            ),
            OptimizationStrategy(
                name="Batch Processing",
                description="Batch multiple requests to reduce API calls",
                savings_percentage=25,
                quality_impact=0,
                implementation_complexity="medium"
            ),
            OptimizationStrategy(
                name="Template Reuse",
                description="Reuse successful templates with variations",
                savings_percentage=35,
                quality_impact=-10,
                implementation_complexity="low"
            ),
            OptimizationStrategy(
                name="Smart Prompt Engineering",
                description="Optimize prompts to reduce token usage",
                savings_percentage=20,
                quality_impact=5,
                implementation_complexity="high"
            ),
            OptimizationStrategy(
                name="Local Model Hybrid",
                description="Use local models for simple tasks",
                savings_percentage=50,
                quality_impact=-15,
                implementation_complexity="high"
            ),
            OptimizationStrategy(
                name="Time-based Pricing",
                description="Schedule non-urgent tasks during off-peak hours",
                savings_percentage=15,
                quality_impact=0,
                implementation_complexity="low"
            ),
            OptimizationStrategy(
                name="Quality Threshold Adjustment",
                description="Accept 85% quality for 50% cost reduction",
                savings_percentage=50,
                quality_impact=-15,
                implementation_complexity="low"
            )
        ]
        
    async def get_optimal_model(
        self, 
        service: ServiceType, 
        quality_required: float = 85.0,
        budget_remaining: float = None
    ) -> Tuple[str, ModelTier]:
        """Get optimal model based on budget and quality requirements"""
        
        # Check cache first
        cache_key = f"optimal_model:{service.value}:{quality_required}"
        cached = self.redis.get(cache_key)
        if cached:
            data = json.loads(cached)
            return data["model"], ModelTier(data["tier"])
            
        # Check daily budget
        daily_spent = await self.get_daily_spending()
        if daily_spent >= self.DAILY_BUDGET["total"]:
            logger.warning("Daily budget exceeded, using economy tier")
            return "gpt-3.5-turbo", ModelTier.ECONOMY
            
        # Determine model based on quality requirements
        if quality_required >= 90:
            model = "gpt-4-turbo"
            tier = ModelTier.PREMIUM
        elif quality_required >= 75:
            model = "gpt-3.5-turbo"
            tier = ModelTier.STANDARD
        else:
            model = "gpt-3.5-turbo"
            tier = ModelTier.ECONOMY
            
        # Check service-specific budget
        if budget_remaining:
            service_limit = self.COST_LIMITS.get(service, 1.0)
            if budget_remaining < service_limit * 0.3:
                # Low budget, downgrade model
                model = "gpt-3.5-turbo"
                tier = ModelTier.ECONOMY
                
        # Cache decision
        self.redis.setex(
            cache_key,
            300,  # 5 minutes
            json.dumps({"model": model, "tier": tier.value})
        )
        
        return model, tier
        
    async def estimate_cost(
        self,
        service: ServiceType,
        model: str,
        input_tokens: int,
        output_tokens: int = None
    ) -> float:
        """Estimate cost for an API call"""
        
        if output_tokens is None:
            output_tokens = input_tokens * 2  # Rough estimate
            
        model_cost = self.MODEL_COSTS.get(model, 0.002)
        total_tokens = (input_tokens + output_tokens) / 1000
        
        estimated_cost = total_tokens * model_cost
        
        # Add service-specific multipliers
        if service == ServiceType.VOICE_SYNTHESIS:
            estimated_cost *= 5  # Voice is more expensive
        elif service == ServiceType.IMAGE_GENERATION:
            estimated_cost = 0.02  # Fixed cost per image
            
        return round(estimated_cost, 4)
        
    async def check_budget_available(
        self,
        service: ServiceType,
        estimated_cost: float
    ) -> bool:
        """Check if budget is available for the operation"""
        
        # Check daily total budget
        daily_spent = await self.get_daily_spending()
        if daily_spent + estimated_cost > self.DAILY_BUDGET["total"]:
            logger.warning(f"Would exceed daily budget: {daily_spent + estimated_cost}")
            return False
            
        # Check service-specific limit
        service_limit = self.COST_LIMITS.get(service, 1.0)
        if estimated_cost > service_limit:
            logger.warning(f"Would exceed service limit for {service.value}: {estimated_cost}")
            return False
            
        return True
        
    async def get_daily_spending(self) -> float:
        """Get total spending for today"""
        
        cache_key = f"daily_spending:{datetime.utcnow().date()}"
        cached = self.redis.get(cache_key)
        
        if cached:
            return float(cached)
            
        # Query database for today's costs
        async for db in get_db():
            today = datetime.utcnow().date()
            result = await db.execute(
                select(func.sum(Cost.amount)).where(
                    func.date(Cost.created_at) == today
                )
            )
            total = result.scalar() or 0.0
            
        # Cache for 5 minutes
        self.redis.setex(cache_key, 300, str(total))
        
        return total
        
    def get_cache_key(
        self,
        service: ServiceType,
        prompt: str,
        params: Dict[str, Any] = None
    ) -> str:
        """Generate cache key for a request"""
        
        import hashlib
        
        # Create a unique key based on service, prompt, and parameters
        key_parts = [service.value, prompt]
        
        if params:
            # Sort params for consistent hashing
            sorted_params = json.dumps(params, sort_keys=True)
            key_parts.append(sorted_params)
            
        key_string = "|".join(key_parts)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        
        return f"ai_cache:{service.value}:{key_hash}"
        
    async def get_cached_response(
        self,
        service: ServiceType,
        prompt: str,
        params: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """Get cached response if available"""
        
        cache_key = self.get_cache_key(service, prompt, params)
        cached = self.redis.get(cache_key)
        
        if cached:
            logger.info(f"Cache hit for {service.value}")
            return json.loads(cached)
            
        return None
        
    async def cache_response(
        self,
        service: ServiceType,
        prompt: str,
        response: Dict[str, Any],
        params: Dict[str, Any] = None
    ):
        """Cache a response"""
        
        cache_key = self.get_cache_key(service, prompt, params)
        ttl = self.CACHE_TTL.get(service, 900)  # Default 15 minutes
        
        self.redis.setex(
            cache_key,
            ttl,
            json.dumps(response)
        )
        
        logger.info(f"Cached response for {service.value} (TTL: {ttl}s)")
        
    async def optimize_prompt(self, prompt: str, max_tokens: int = None) -> str:
        """Optimize prompt to reduce token usage"""
        
        # Remove unnecessary whitespace
        optimized = " ".join(prompt.split())
        
        # Use abbreviations where appropriate
        replacements = {
            "approximately": "~",
            "for example": "e.g.",
            "that is": "i.e.",
            "et cetera": "etc",
            "versus": "vs",
        }
        
        for long_form, short_form in replacements.items():
            optimized = optimized.replace(long_form, short_form)
            
        # Truncate if max_tokens specified
        if max_tokens:
            # Rough estimate: 1 token ≈ 4 characters
            max_chars = max_tokens * 4
            if len(optimized) > max_chars:
                optimized = optimized[:max_chars] + "..."
                
        return optimized
        
    async def batch_requests(
        self,
        requests: List[Dict[str, Any]],
        service: ServiceType
    ) -> List[Dict[str, Any]]:
        """Batch multiple requests to reduce API calls"""
        
        # Group similar requests
        batches = {}
        for req in requests:
            key = f"{req.get('model', 'default')}:{req.get('temperature', 0.7)}"
            if key not in batches:
                batches[key] = []
            batches[key].append(req)
            
        results = []
        
        for batch_key, batch_requests in batches.items():
            # Process batch
            if len(batch_requests) > 1:
                # Combine prompts with separators
                combined_prompt = "\n---SEPARATOR---\n".join(
                    [r["prompt"] for r in batch_requests]
                )
                
                # Make single API call
                response = await self._make_batch_api_call(
                    combined_prompt,
                    batch_requests[0].get("model", "gpt-3.5-turbo"),
                    service
                )
                
                # Split responses
                split_responses = response.split("---SEPARATOR---")
                for i, resp in enumerate(split_responses):
                    if i < len(batch_requests):
                        results.append({"request": batch_requests[i], "response": resp})
            else:
                # Single request, process normally
                results.append(batch_requests[0])
                
        return results
        
    async def _make_batch_api_call(
        self,
        prompt: str,
        model: str,
        service: ServiceType
    ) -> str:
        """Make batched API call (placeholder for actual implementation)"""
        
        # This would be replaced with actual API call
        logger.info(f"Batch API call to {model} for {service.value}")
        
        # For now, return placeholder
        return "Batch response 1---SEPARATOR---Batch response 2"
        
    async def apply_fallback_strategy(
        self,
        service: ServiceType,
        primary_model: str,
        fallback_models: List[str],
        prompt: str,
        quality_threshold: float = 75.0
    ) -> Tuple[str, str]:
        """Apply progressive fallback strategy"""
        
        for model in [primary_model] + fallback_models:
            try:
                # Try with current model
                logger.info(f"Trying {model} for {service.value}")
                
                # Check if quality would be acceptable
                estimated_quality = self._estimate_model_quality(model, service)
                
                if estimated_quality >= quality_threshold:
                    # Model is acceptable, use it
                    return model, f"Response from {model}"
                    
            except Exception as e:
                logger.warning(f"Failed with {model}: {e}")
                continue
                
        # All models failed
        raise Exception("All models failed or below quality threshold")
        
    def _estimate_model_quality(self, model: str, service: ServiceType) -> float:
        """Estimate quality score for a model/service combination"""
        
        base_scores = {
            "gpt-4-turbo": 95,
            "gpt-3.5-turbo": 80,
            "claude-3-opus": 93,
            "claude-3-sonnet": 85,
            "claude-instant": 75,
        }
        
        service_modifiers = {
            ServiceType.SCRIPT_GENERATION: 1.0,
            ServiceType.QUALITY_CHECK: 0.9,
            ServiceType.TREND_ANALYSIS: 0.95,
            ServiceType.IMAGE_GENERATION: 1.0,
        }
        
        base = base_scores.get(model, 70)
        modifier = service_modifiers.get(service, 1.0)
        
        return base * modifier
        
    async def get_cost_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate cost optimization report"""
        
        async for db in get_db():
            # Get costs for the period
            since = datetime.utcnow() - timedelta(days=days)
            
            result = await db.execute(
                select(
                    Cost.service,
                    func.sum(Cost.amount).label("total"),
                    func.count(Cost.id).label("count"),
                    func.avg(Cost.amount).label("average")
                ).where(
                    Cost.created_at >= since
                ).group_by(Cost.service)
            )
            
            costs_by_service = {}
            total_cost = 0
            total_videos = 0
            
            for row in result:
                costs_by_service[row.service] = {
                    "total": float(row.total),
                    "count": row.count,
                    "average": float(row.average)
                }
                total_cost += float(row.total)
                
            # Get video count
            video_result = await db.execute(
                select(func.count(Video.id)).where(
                    Video.created_at >= since
                )
            )
            total_videos = video_result.scalar() or 0
            
        # Calculate metrics
        cost_per_video = total_cost / total_videos if total_videos > 0 else 0
        
        # Identify optimization opportunities
        opportunities = []
        
        if cost_per_video > 3.0:
            opportunities.append({
                "issue": "Cost per video exceeds $3 target",
                "recommendation": "Enable aggressive caching and model fallback",
                "potential_savings": f"${(cost_per_video - 3.0) * total_videos:.2f}"
            })
            
        for service, data in costs_by_service.items():
            limit = self.COST_LIMITS.get(ServiceType(service), 1.0) if service else 1.0
            if data["average"] > limit:
                opportunities.append({
                    "issue": f"{service} exceeds cost limit",
                    "recommendation": f"Optimize {service} prompts or use lower tier model",
                    "potential_savings": f"${(data['average'] - limit) * data['count']:.2f}"
                })
                
        return {
            "period_days": days,
            "total_cost": total_cost,
            "total_videos": total_videos,
            "cost_per_video": cost_per_video,
            "target_cost_per_video": 3.0,
            "within_target": cost_per_video <= 3.0,
            "costs_by_service": costs_by_service,
            "optimization_opportunities": opportunities,
            "active_strategies": [s.name for s in self.optimization_strategies],
            "estimated_savings": self._calculate_potential_savings(costs_by_service)
        }
        
    def _calculate_potential_savings(self, costs_by_service: Dict[str, Any]) -> float:
        """Calculate potential savings from optimization"""
        
        total_current = sum(data["total"] for data in costs_by_service.values())
        
        # Apply strategy savings
        potential_savings = 0
        for strategy in self.optimization_strategies:
            potential_savings += total_current * (strategy.savings_percentage / 100)
            
        # Cap at realistic 60% savings
        return min(potential_savings, total_current * 0.6)
        
    async def validate_video_cost(self, video_id: int) -> Dict[str, Any]:
        """Validate if a video met the <$3 cost target"""
        
        async for db in get_db():
            result = await db.execute(
                select(func.sum(Cost.amount)).where(
                    Cost.video_id == video_id
                )
            )
            total_cost = result.scalar() or 0.0
            
        is_valid = total_cost < 3.0
        
        return {
            "video_id": video_id,
            "total_cost": total_cost,
            "target": 3.0,
            "within_target": is_valid,
            "overage": max(0, total_cost - 3.0),
            "savings_achieved": max(0, 3.0 - total_cost)
        }


    async def predict_future_costs(
        self,
        days_ahead: int = 30,
        include_confidence: bool = True
    ) -> Dict[str, Any]:
        """ML-based cost prediction for future periods"""
        
        try:
            # Get historical cost data
            async for db in get_db():
                # Get last 90 days of cost data
                since = datetime.utcnow() - timedelta(days=90)
                
                result = await db.execute(
                    select(
                        func.date(Cost.created_at).label("date"),
                        func.sum(Cost.amount).label("total_cost"),
                        func.count(Cost.id).label("transaction_count")
                    ).where(
                        Cost.created_at >= since
                    ).group_by(func.date(Cost.created_at))
                    .order_by(func.date(Cost.created_at))
                )
                
                historical_data = []
                for row in result:
                    historical_data.append({
                        "date": row.date.isoformat(),
                        "cost": float(row.total_cost),
                        "transactions": row.transaction_count
                    })
                
                break  # Exit the async generator
            
            if len(historical_data) < 7:
                return {
                    "error": "Insufficient historical data for prediction",
                    "min_required_days": 7,
                    "available_days": len(historical_data)
                }
            
            # Simple trend analysis (in production, use proper ML models)
            costs = [d["cost"] for d in historical_data]
            recent_avg = sum(costs[-7:]) / 7  # Last 7 days average
            older_avg = sum(costs[:7]) / 7 if len(costs) >= 14 else recent_avg
            
            # Calculate trend
            if recent_avg > older_avg * 1.1:
                trend = "increasing"
                trend_factor = 1.05  # 5% increase
            elif recent_avg < older_avg * 0.9:
                trend = "decreasing"
                trend_factor = 0.95  # 5% decrease
            else:
                trend = "stable"
                trend_factor = 1.0
            
            # Generate predictions
            base_daily_cost = recent_avg
            predictions = []
            
            for day in range(1, days_ahead + 1):
                # Apply trend factor with diminishing effect
                trend_effect = 1 + (trend_factor - 1) * (0.95 ** day)
                predicted_cost = base_daily_cost * trend_effect
                
                # Add some realistic variance
                variance = predicted_cost * 0.1  # 10% variance
                
                predictions.append({
                    "day": day,
                    "date": (datetime.utcnow() + timedelta(days=day)).date().isoformat(),
                    "predicted_cost": round(predicted_cost, 2),
                    "confidence_interval": {
                        "lower": round(predicted_cost - variance, 2),
                        "upper": round(predicted_cost + variance, 2)
                    } if include_confidence else None
                })
            
            total_predicted = sum(p["predicted_cost"] for p in predictions)
            
            return {
                "prediction_period_days": days_ahead,
                "historical_data_points": len(historical_data),
                "trend_analysis": {
                    "trend": trend,
                    "recent_daily_avg": round(recent_avg, 2),
                    "trend_factor": trend_factor
                },
                "predictions": predictions,
                "summary": {
                    "total_predicted_cost": round(total_predicted, 2),
                    "avg_daily_predicted": round(total_predicted / days_ahead, 2),
                    "vs_current_avg": round(((total_predicted / days_ahead) / recent_avg - 1) * 100, 1)
                },
                "risk_factors": self._identify_cost_risk_factors(historical_data),
                "recommendations": self._generate_cost_recommendations(trend, recent_avg)
            }
            
        except Exception as e:
            logger.error(f"Cost prediction failed: {e}")
            return {"error": str(e)}
    
    async def generate_optimization_recommendations(
        self,
        current_costs: Dict[str, Any],
        target_reduction: float = 0.3
    ) -> Dict[str, Any]:
        """Generate ML-driven optimization recommendations"""
        
        recommendations = []
        total_potential_savings = 0
        
        # Analyze each service
        for service, cost_data in current_costs.items():
            if isinstance(cost_data, dict) and "total" in cost_data:
                service_cost = cost_data["total"]
                
                # Apply different strategies based on cost patterns
                if service_cost > 20:  # High cost services
                    rec = {
                        "service": service,
                        "strategy": "Progressive Model Fallback",
                        "current_cost": service_cost,
                        "potential_savings": service_cost * 0.4,
                        "implementation": "Use GPT-3.5 instead of GPT-4 for routine tasks",
                        "quality_impact": "Low (-5%)",
                        "priority": "High"
                    }
                    recommendations.append(rec)
                    total_potential_savings += rec["potential_savings"]
                
                elif service_cost > 5:  # Medium cost services
                    rec = {
                        "service": service,
                        "strategy": "Aggressive Caching",
                        "current_cost": service_cost,
                        "potential_savings": service_cost * 0.25,
                        "implementation": "Cache responses for 1-4 hours based on content type",
                        "quality_impact": "None (0%)",
                        "priority": "Medium"
                    }
                    recommendations.append(rec)
                    total_potential_savings += rec["potential_savings"]
        
        # Add general recommendations
        general_recommendations = [
            {
                "strategy": "Batch Processing",
                "description": "Process multiple requests together to reduce API overhead",
                "estimated_savings_percent": 15,
                "implementation_effort": "Medium"
            },
            {
                "strategy": "Smart Prompt Optimization", 
                "description": "Reduce token usage through prompt engineering",
                "estimated_savings_percent": 20,
                "implementation_effort": "High"
            },
            {
                "strategy": "Quality Threshold Adjustment",
                "description": "Accept 85% quality for significant cost reduction",
                "estimated_savings_percent": 30,
                "implementation_effort": "Low"
            }
        ]
        
        return {
            "target_reduction_percent": target_reduction * 100,
            "service_specific_recommendations": recommendations,
            "general_recommendations": general_recommendations,
            "estimated_total_savings": round(total_potential_savings, 2),
            "implementation_timeline": {
                "immediate": "Enable caching and model fallback",
                "short_term": "Implement batch processing",
                "medium_term": "Optimize prompts and quality thresholds",
                "long_term": "Implement local model hybrid approach"
            },
            "monitoring_kpis": [
                "Daily cost per video",
                "Service-specific cost trends",
                "Quality score maintenance",
                "Cache hit rates",
                "Model distribution ratios"
            ]
        }
    
    async def implement_budget_management(
        self,
        monthly_budget: float,
        alert_thresholds: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """Implement intelligent budget management with alerts"""
        
        if alert_thresholds is None:
            alert_thresholds = {
                "warning": 0.75,  # 75% of budget
                "critical": 0.90,  # 90% of budget
                "emergency": 0.95   # 95% of budget
            }
        
        try:
            # Get current month spending
            current_month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            async for db in get_db():
                result = await db.execute(
                    select(func.sum(Cost.amount)).where(
                        Cost.created_at >= current_month_start
                    )
                )
                current_spending = result.scalar() or 0.0
                break
            
            # Calculate budget status
            budget_used_percent = (current_spending / monthly_budget) * 100
            remaining_budget = monthly_budget - current_spending
            
            # Determine alert level
            alert_level = "normal"
            if budget_used_percent >= alert_thresholds["emergency"] * 100:
                alert_level = "emergency"
            elif budget_used_percent >= alert_thresholds["critical"] * 100:
                alert_level = "critical"
            elif budget_used_percent >= alert_thresholds["warning"] * 100:
                alert_level = "warning"
            
            # Calculate daily burn rate and projection
            days_in_month = (datetime.utcnow().replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            days_in_month = days_in_month.day
            current_day = datetime.utcnow().day
            
            if current_day > 0:
                daily_burn_rate = current_spending / current_day
                projected_monthly_spend = daily_burn_rate * days_in_month
            else:
                daily_burn_rate = 0
                projected_monthly_spend = 0
            
            # Generate recommendations based on budget status
            recommendations = []
            
            if alert_level in ["critical", "emergency"]:
                recommendations.extend([
                    "Switch to economy tier models immediately",
                    "Enable aggressive caching for all services",
                    "Defer non-critical video generation",
                    "Implement strict per-video cost limits"
                ])
            elif alert_level == "warning":
                recommendations.extend([
                    "Increase cache TTL settings",
                    "Review and optimize high-cost operations",
                    "Consider model tier adjustments"
                ])
            else:
                recommendations.extend([
                    "Monitor spending trends",
                    "Continue current optimization strategies"
                ])
            
            return {
                "budget_management": {
                    "monthly_budget": monthly_budget,
                    "current_spending": round(current_spending, 2),
                    "remaining_budget": round(remaining_budget, 2),
                    "budget_used_percent": round(budget_used_percent, 1),
                    "alert_level": alert_level
                },
                "projections": {
                    "daily_burn_rate": round(daily_burn_rate, 2),
                    "projected_monthly_spend": round(projected_monthly_spend, 2),
                    "over_budget_risk": projected_monthly_spend > monthly_budget,
                    "days_until_budget_exhausted": int(remaining_budget / daily_burn_rate) if daily_burn_rate > 0 else None
                },
                "recommendations": recommendations,
                "auto_actions": {
                    "enabled": alert_level in ["critical", "emergency"],
                    "actions": [
                        "Switch to economy models",
                        "Increase cache TTL to 4 hours",
                        "Reject requests over $1 per video"
                    ] if alert_level in ["critical", "emergency"] else []
                }
            }
            
        except Exception as e:
            logger.error(f"Budget management failed: {e}")
            return {"error": str(e)}
    
    def _identify_cost_risk_factors(self, historical_data: List[Dict]) -> List[str]:
        """Identify potential cost risk factors"""
        risk_factors = []
        
        if len(historical_data) < 7:
            risk_factors.append("Insufficient historical data for reliable prediction")
            return risk_factors
        
        costs = [d["cost"] for d in historical_data]
        
        # Check for high volatility
        if len(costs) > 1:
            import statistics
            std_dev = statistics.stdev(costs)
            mean_cost = statistics.mean(costs)
            
            if std_dev / mean_cost > 0.5:  # High coefficient of variation
                risk_factors.append("High cost volatility detected")
        
        # Check for recent spikes
        recent_costs = costs[-7:]
        older_costs = costs[:-7] if len(costs) > 7 else costs
        
        if recent_costs and older_costs:
            recent_avg = sum(recent_costs) / len(recent_costs)
            older_avg = sum(older_costs) / len(older_costs)
            
            if recent_avg > older_avg * 1.5:
                risk_factors.append("Recent cost spike detected")
        
        # Check for budget threshold proximity
        recent_daily_avg = sum(costs[-3:]) / min(3, len(costs))
        if recent_daily_avg > 30:  # High daily spend
            risk_factors.append("High daily spending may exceed monthly budget")
        
        return risk_factors or ["No significant risk factors identified"]
    
    def _generate_cost_recommendations(self, trend: str, current_avg: float) -> List[str]:
        """Generate cost management recommendations based on trends"""
        recommendations = []
        
        if trend == "increasing":
            recommendations.extend([
                "Implement cost controls to prevent further increases",
                "Review recent changes in service usage",
                "Consider switching to more cost-effective models",
                "Increase caching to reduce API calls"
            ])
        elif trend == "decreasing":
            recommendations.extend([
                "Current optimizations are working well",
                "Monitor to ensure quality is maintained",
                "Consider applying similar strategies to other services"
            ])
        else:  # stable
            recommendations.extend([
                "Costs are stable - good baseline for optimization",
                "Experiment with new cost reduction strategies",
                "Set up monitoring for early detection of changes"
            ])
        
        # Add general recommendations based on cost level
        if current_avg > 50:
            recommendations.append("Daily costs are high - prioritize aggressive optimization")
        elif current_avg > 20:
            recommendations.append("Daily costs are moderate - implement gradual optimizations")
        else:
            recommendations.append("Daily costs are low - maintain current strategies")
        
        return recommendations


# Singleton instance
_optimizer_instance = None


def get_cost_optimizer() -> CostOptimizer:
    """Get singleton instance of cost optimizer"""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = CostOptimizer()
    return _optimizer_instance