"""
Model Monitoring System
Tracks AI model performance, costs, and health
"""
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import logging
import json
import aioredis
from prometheus_client import Counter, Histogram, Gauge, Summary

logger = logging.getLogger(__name__)

# Prometheus metrics
model_requests = Counter('model_requests_total', 'Total model requests', ['model', 'status'])
model_latency = Histogram('model_latency_seconds', 'Model latency', ['model'])
model_cost = Counter('model_cost_dollars', 'Model costs in dollars', ['model', 'service'])
model_errors = Counter('model_errors_total', 'Model errors', ['model', 'error_type'])
model_quality_score = Gauge('model_quality_score', 'Model output quality score', ['model'])
active_models = Gauge('active_models', 'Number of active models')


class ModelType(Enum):
    """Supported model types"""
    GPT4 = "gpt-4"
    GPT35 = "gpt-3.5-turbo"
    CLAUDE3 = "claude-3-opus"
    CLAUDE2 = "claude-2"
    DALL_E3 = "dall-e-3"
    ELEVENLABS = "elevenlabs"
    CUSTOM_TREND = "custom-trend-model"
    CUSTOM_QUALITY = "custom-quality-model"


class ModelStatus(Enum):
    """Model health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    model_type: ModelType
    timestamp: datetime
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_latency: float = 0.0
    total_cost: float = 0.0
    avg_quality_score: float = 0.0
    error_types: Dict[str, int] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.success_count / self.request_count
    
    @property
    def avg_latency(self) -> float:
        if self.success_count == 0:
            return 0.0
        return self.total_latency / self.success_count
    
    @property
    def error_rate(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count


@dataclass
class ModelHealth:
    """Model health status"""
    model_type: ModelType
    status: ModelStatus
    last_check: datetime
    issues: List[str] = field(default_factory=list)
    metrics: Optional[ModelMetrics] = None
    recommendations: List[str] = field(default_factory=list)


class ModelMonitor:
    """Main model monitoring system"""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url
        self.redis_client = None
        self.metrics_buffer: Dict[ModelType, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.health_status: Dict[ModelType, ModelHealth] = {}
        self.cost_budgets: Dict[ModelType, float] = self._initialize_budgets()
        self.latency_thresholds: Dict[ModelType, float] = self._initialize_latency_thresholds()
        self.quality_thresholds: Dict[ModelType, float] = self._initialize_quality_thresholds()
        
    async def initialize(self):
        """Initialize monitoring system"""
        if self.redis_url:
            self.redis_client = await aioredis.create_redis_pool(self.redis_url)
        logger.info("Model monitoring system initialized")
    
    def _initialize_budgets(self) -> Dict[ModelType, float]:
        """Initialize daily cost budgets per model"""
        return {
            ModelType.GPT4: 50.0,
            ModelType.GPT35: 20.0,
            ModelType.CLAUDE3: 40.0,
            ModelType.CLAUDE2: 30.0,
            ModelType.DALL_E3: 15.0,
            ModelType.ELEVENLABS: 20.0,
            ModelType.CUSTOM_TREND: 5.0,
            ModelType.CUSTOM_QUALITY: 5.0
        }
    
    def _initialize_latency_thresholds(self) -> Dict[ModelType, float]:
        """Initialize latency thresholds (seconds)"""
        return {
            ModelType.GPT4: 30.0,
            ModelType.GPT35: 15.0,
            ModelType.CLAUDE3: 25.0,
            ModelType.CLAUDE2: 20.0,
            ModelType.DALL_E3: 45.0,
            ModelType.ELEVENLABS: 10.0,
            ModelType.CUSTOM_TREND: 5.0,
            ModelType.CUSTOM_QUALITY: 3.0
        }
    
    def _initialize_quality_thresholds(self) -> Dict[ModelType, float]:
        """Initialize quality score thresholds"""
        return {
            ModelType.GPT4: 0.85,
            ModelType.GPT35: 0.75,
            ModelType.CLAUDE3: 0.85,
            ModelType.CLAUDE2: 0.80,
            ModelType.DALL_E3: 0.80,
            ModelType.ELEVENLABS: 0.90,
            ModelType.CUSTOM_TREND: 0.70,
            ModelType.CUSTOM_QUALITY: 0.75
        }
    
    async def record_request(
        self,
        model_type: ModelType,
        success: bool,
        latency: float,
        cost: float,
        quality_score: Optional[float] = None,
        error_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a model request"""
        # Update Prometheus metrics
        status = "success" if success else "error"
        model_requests.labels(model=model_type.value, status=status).inc()
        model_latency.labels(model=model_type.value).observe(latency)
        model_cost.labels(model=model_type.value, service="api").inc(cost)
        
        if not success and error_type:
            model_errors.labels(model=model_type.value, error_type=error_type).inc()
        
        if quality_score is not None:
            model_quality_score.labels(model=model_type.value).set(quality_score)
        
        # Create metrics record
        metrics = ModelMetrics(
            model_type=model_type,
            timestamp=datetime.utcnow(),
            request_count=1,
            success_count=1 if success else 0,
            error_count=0 if success else 1,
            total_latency=latency if success else 0,
            total_cost=cost,
            avg_quality_score=quality_score or 0.0
        )
        
        if error_type:
            metrics.error_types[error_type] = 1
        
        # Add to buffer
        self.metrics_buffer[model_type].append(metrics)
        
        # Store in Redis if available
        if self.redis_client:
            await self._store_metrics_redis(metrics, metadata)
        
        # Check for alerts
        await self._check_alerts(model_type, metrics)
    
    async def _store_metrics_redis(self, metrics: ModelMetrics, metadata: Optional[Dict] = None):
        """Store metrics in Redis"""
        key = f"model:metrics:{metrics.model_type.value}:{metrics.timestamp.isoformat()}"
        data = {
            "model": metrics.model_type.value,
            "timestamp": metrics.timestamp.isoformat(),
            "success_rate": metrics.success_rate,
            "avg_latency": metrics.avg_latency,
            "cost": metrics.total_cost,
            "quality": metrics.avg_quality_score,
            "metadata": json.dumps(metadata) if metadata else "{}"
        }
        
        await self.redis_client.hmset_dict(key, data)
        await self.redis_client.expire(key, 86400 * 7)  # 7 days retention
    
    async def _check_alerts(self, model_type: ModelType, metrics: ModelMetrics):
        """Check for alert conditions"""
        alerts = []
        
        # Latency alert
        if metrics.avg_latency > self.latency_thresholds.get(model_type, 30):
            alerts.append(f"High latency: {metrics.avg_latency:.2f}s")
        
        # Error rate alert
        if metrics.error_rate > 0.1:
            alerts.append(f"High error rate: {metrics.error_rate:.2%}")
        
        # Quality alert
        if metrics.avg_quality_score > 0 and \
           metrics.avg_quality_score < self.quality_thresholds.get(model_type, 0.7):
            alerts.append(f"Low quality score: {metrics.avg_quality_score:.2f}")
        
        # Cost alert (check daily total)
        daily_cost = await self.get_daily_cost(model_type)
        budget = self.cost_budgets.get(model_type, 50)
        if daily_cost > budget * 0.8:
            alerts.append(f"Cost warning: ${daily_cost:.2f} of ${budget:.2f} budget")
        
        if alerts:
            logger.warning(f"Alerts for {model_type.value}: {', '.join(alerts)}")
            await self._send_alerts(model_type, alerts)
    
    async def _send_alerts(self, model_type: ModelType, alerts: List[str]):
        """Send alerts to monitoring system"""
        # TODO: Integrate with alerting system (Slack, email, etc.)
        pass
    
    async def get_model_health(self, model_type: ModelType) -> ModelHealth:
        """Get current health status of a model"""
        # Get recent metrics
        recent_metrics = await self.get_recent_metrics(model_type, minutes=5)
        
        if not recent_metrics:
            return ModelHealth(
                model_type=model_type,
                status=ModelStatus.OFFLINE,
                last_check=datetime.utcnow(),
                issues=["No recent activity"]
            )
        
        # Aggregate metrics
        aggregated = self._aggregate_metrics(recent_metrics)
        
        # Determine health status
        status = ModelStatus.HEALTHY
        issues = []
        recommendations = []
        
        # Check success rate
        if aggregated.success_rate < 0.95:
            status = ModelStatus.DEGRADED
            issues.append(f"Success rate: {aggregated.success_rate:.2%}")
            recommendations.append("Investigate error patterns")
        
        if aggregated.success_rate < 0.8:
            status = ModelStatus.UNHEALTHY
        
        # Check latency
        threshold = self.latency_thresholds.get(model_type, 30)
        if aggregated.avg_latency > threshold:
            if status == ModelStatus.HEALTHY:
                status = ModelStatus.DEGRADED
            issues.append(f"High latency: {aggregated.avg_latency:.2f}s")
            recommendations.append("Consider scaling or optimization")
        
        # Check quality
        quality_threshold = self.quality_thresholds.get(model_type, 0.7)
        if aggregated.avg_quality_score > 0 and aggregated.avg_quality_score < quality_threshold:
            if status == ModelStatus.HEALTHY:
                status = ModelStatus.DEGRADED
            issues.append(f"Low quality: {aggregated.avg_quality_score:.2f}")
            recommendations.append("Review model parameters or prompts")
        
        return ModelHealth(
            model_type=model_type,
            status=status,
            last_check=datetime.utcnow(),
            issues=issues,
            metrics=aggregated,
            recommendations=recommendations
        )
    
    async def get_recent_metrics(
        self,
        model_type: ModelType,
        minutes: int = 60
    ) -> List[ModelMetrics]:
        """Get recent metrics for a model"""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        
        # Get from buffer
        buffer_metrics = [
            m for m in self.metrics_buffer[model_type]
            if m.timestamp > cutoff
        ]
        
        # Get from Redis if available
        if self.redis_client:
            redis_metrics = await self._get_metrics_from_redis(model_type, cutoff)
            buffer_metrics.extend(redis_metrics)
        
        return buffer_metrics
    
    async def _get_metrics_from_redis(
        self,
        model_type: ModelType,
        since: datetime
    ) -> List[ModelMetrics]:
        """Get metrics from Redis"""
        # TODO: Implement Redis query
        return []
    
    def _aggregate_metrics(self, metrics_list: List[ModelMetrics]) -> ModelMetrics:
        """Aggregate multiple metrics"""
        if not metrics_list:
            return ModelMetrics(
                model_type=ModelType.GPT4,
                timestamp=datetime.utcnow()
            )
        
        aggregated = ModelMetrics(
            model_type=metrics_list[0].model_type,
            timestamp=datetime.utcnow(),
            request_count=sum(m.request_count for m in metrics_list),
            success_count=sum(m.success_count for m in metrics_list),
            error_count=sum(m.error_count for m in metrics_list),
            total_latency=sum(m.total_latency for m in metrics_list),
            total_cost=sum(m.total_cost for m in metrics_list)
        )
        
        # Aggregate quality scores
        quality_scores = [m.avg_quality_score for m in metrics_list if m.avg_quality_score > 0]
        if quality_scores:
            aggregated.avg_quality_score = np.mean(quality_scores)
        
        # Aggregate error types
        for metrics in metrics_list:
            for error_type, count in metrics.error_types.items():
                aggregated.error_types[error_type] = \
                    aggregated.error_types.get(error_type, 0) + count
        
        return aggregated
    
    async def get_daily_cost(self, model_type: ModelType) -> float:
        """Get total cost for model today"""
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        metrics = await self.get_recent_metrics(model_type, minutes=1440)  # 24 hours
        today_metrics = [m for m in metrics if m.timestamp >= today_start]
        
        return sum(m.total_cost for m in today_metrics)
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "models": {},
            "summary": {
                "total_requests": 0,
                "total_cost": 0,
                "avg_success_rate": 0,
                "models_healthy": 0,
                "models_degraded": 0,
                "models_unhealthy": 0
            }
        }
        
        for model_type in ModelType:
            # Get health status
            health = await self.get_model_health(model_type)
            
            # Get daily metrics
            daily_metrics = await self.get_recent_metrics(model_type, minutes=1440)
            aggregated = self._aggregate_metrics(daily_metrics)
            
            # Get daily cost
            daily_cost = await self.get_daily_cost(model_type)
            budget = self.cost_budgets.get(model_type, 0)
            
            model_report = {
                "status": health.status.value,
                "issues": health.issues,
                "recommendations": health.recommendations,
                "metrics": {
                    "requests": aggregated.request_count,
                    "success_rate": aggregated.success_rate,
                    "avg_latency": aggregated.avg_latency,
                    "avg_quality": aggregated.avg_quality_score,
                    "error_rate": aggregated.error_rate,
                    "top_errors": sorted(
                        aggregated.error_types.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
                },
                "cost": {
                    "daily": daily_cost,
                    "budget": budget,
                    "utilization": (daily_cost / budget * 100) if budget > 0 else 0
                }
            }
            
            report["models"][model_type.value] = model_report
            
            # Update summary
            report["summary"]["total_requests"] += aggregated.request_count
            report["summary"]["total_cost"] += daily_cost
            
            if health.status == ModelStatus.HEALTHY:
                report["summary"]["models_healthy"] += 1
            elif health.status == ModelStatus.DEGRADED:
                report["summary"]["models_degraded"] += 1
            elif health.status == ModelStatus.UNHEALTHY:
                report["summary"]["models_unhealthy"] += 1
        
        # Calculate average success rate
        total_models = len([m for m in report["models"].values() if m["metrics"]["requests"] > 0])
        if total_models > 0:
            report["summary"]["avg_success_rate"] = \
                sum(m["metrics"]["success_rate"] for m in report["models"].values()) / total_models
        
        return report
    
    async def get_cost_forecast(self, days: int = 7) -> Dict[str, Any]:
        """Forecast costs based on recent usage"""
        forecast = {
            "period_days": days,
            "models": {},
            "total_forecast": 0
        }
        
        for model_type in ModelType:
            # Get last 7 days of data
            recent_metrics = await self.get_recent_metrics(model_type, minutes=10080)  # 7 days
            
            if not recent_metrics:
                continue
            
            # Calculate daily average
            daily_costs = defaultdict(float)
            for metrics in recent_metrics:
                day = metrics.timestamp.date()
                daily_costs[day] += metrics.total_cost
            
            if daily_costs:
                avg_daily_cost = np.mean(list(daily_costs.values()))
                forecast_cost = avg_daily_cost * days
                
                forecast["models"][model_type.value] = {
                    "avg_daily": avg_daily_cost,
                    "forecast": forecast_cost,
                    "budget": self.cost_budgets.get(model_type, 0) * days,
                    "trend": self._calculate_cost_trend(daily_costs)
                }
                
                forecast["total_forecast"] += forecast_cost
        
        return forecast
    
    def _calculate_cost_trend(self, daily_costs: Dict) -> str:
        """Calculate cost trend (increasing/decreasing/stable)"""
        if len(daily_costs) < 3:
            return "insufficient_data"
        
        costs = list(daily_costs.values())
        recent_avg = np.mean(costs[-3:])
        older_avg = np.mean(costs[:-3])
        
        if recent_avg > older_avg * 1.1:
            return "increasing"
        elif recent_avg < older_avg * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    async def optimize_model_selection(
        self,
        task_type: str,
        priority: str = "balanced"
    ) -> ModelType:
        """Recommend optimal model based on current performance and cost"""
        candidates = self._get_candidate_models(task_type)
        
        if not candidates:
            return ModelType.GPT35  # Default fallback
        
        scores = {}
        
        for model_type in candidates:
            health = await self.get_model_health(model_type)
            daily_cost = await self.get_daily_cost(model_type)
            budget = self.cost_budgets.get(model_type, 50)
            
            # Skip unhealthy models
            if health.status == ModelStatus.UNHEALTHY:
                continue
            
            # Calculate score based on priority
            if priority == "quality":
                quality_weight = 0.6
                cost_weight = 0.2
                latency_weight = 0.2
            elif priority == "speed":
                quality_weight = 0.2
                cost_weight = 0.2
                latency_weight = 0.6
            elif priority == "cost":
                quality_weight = 0.2
                cost_weight = 0.6
                latency_weight = 0.2
            else:  # balanced
                quality_weight = 0.4
                cost_weight = 0.3
                latency_weight = 0.3
            
            # Calculate component scores
            quality_score = health.metrics.avg_quality_score if health.metrics else 0.5
            cost_score = 1.0 - (daily_cost / budget) if budget > 0 else 0.5
            latency_score = 1.0 - min(1.0, health.metrics.avg_latency / 30) if health.metrics else 0.5
            
            # Apply health penalty
            health_multiplier = 1.0 if health.status == ModelStatus.HEALTHY else 0.8
            
            # Calculate final score
            final_score = (
                quality_score * quality_weight +
                cost_score * cost_weight +
                latency_score * latency_weight
            ) * health_multiplier
            
            scores[model_type] = final_score
        
        if not scores:
            return ModelType.GPT35  # Default fallback
        
        # Return model with highest score
        return max(scores, key=scores.get)
    
    def _get_candidate_models(self, task_type: str) -> List[ModelType]:
        """Get candidate models for a task type"""
        task_models = {
            "text_generation": [ModelType.GPT4, ModelType.GPT35, ModelType.CLAUDE3, ModelType.CLAUDE2],
            "image_generation": [ModelType.DALL_E3],
            "voice_synthesis": [ModelType.ELEVENLABS],
            "trend_analysis": [ModelType.CUSTOM_TREND, ModelType.GPT4],
            "quality_assessment": [ModelType.CUSTOM_QUALITY, ModelType.GPT4]
        }
        
        return task_models.get(task_type, [ModelType.GPT35])


# Global monitor instance
model_monitor = ModelMonitor()