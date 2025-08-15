"""
Quality Metrics Framework for YTEmpire
Comprehensive quality tracking and KPI management system
"""

import asyncio
import json
import statistics
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
import logging
from collections import defaultdict, deque

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, func, and_, or_
import redis.asyncio as redis
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ============================================================================
# Quality Metrics Definitions
# ============================================================================

class MetricType(str, Enum):
    """Types of quality metrics"""
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    SECURITY = "security"
    USABILITY = "usability"
    BUSINESS = "business"
    TECHNICAL = "technical"
    USER_EXPERIENCE = "user_experience"

class MetricCategory(str, Enum):
    """Metric categories"""
    KPI = "kpi"  # Key Performance Indicator
    SLI = "sli"  # Service Level Indicator  
    SLO = "slo"  # Service Level Objective
    CUSTOM = "custom"

class MetricStatus(str, Enum):
    """Metric status"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class MetricThreshold:
    """Metric threshold configuration"""
    warning_min: Optional[float] = None
    warning_max: Optional[float] = None
    critical_min: Optional[float] = None
    critical_max: Optional[float] = None
    target_value: Optional[float] = None

@dataclass
class QualityMetric:
    """Quality metric definition"""
    id: str
    name: str
    description: str
    type: MetricType
    category: MetricCategory
    unit: str
    threshold: MetricThreshold
    weight: float = 1.0  # Importance weight
    is_higher_better: bool = True
    collection_interval: int = 60  # seconds
    retention_days: int = 90

@dataclass
class MetricValue:
    """Metric value with metadata"""
    metric_id: str
    value: float
    timestamp: datetime
    status: MetricStatus
    metadata: Dict[str, Any] = None
    tags: Dict[str, str] = None

# ============================================================================
# Predefined Quality Metrics
# ============================================================================

QUALITY_METRICS = {
    # Performance KPIs
    "api_response_time_p95": QualityMetric(
        id="api_response_time_p95",
        name="API Response Time P95",
        description="95th percentile API response time",
        type=MetricType.PERFORMANCE,
        category=MetricCategory.KPI,
        unit="ms",
        threshold=MetricThreshold(
            warning_max=500,
            critical_max=1000,
            target_value=200
        ),
        is_higher_better=False
    ),
    
    "video_generation_success_rate": QualityMetric(
        id="video_generation_success_rate",
        name="Video Generation Success Rate",
        description="Percentage of successful video generations",
        type=MetricType.RELIABILITY,
        category=MetricCategory.KPI,
        unit="percent",
        threshold=MetricThreshold(
            warning_min=95,
            critical_min=90,
            target_value=99
        ),
        weight=2.0
    ),
    
    "system_uptime": QualityMetric(
        id="system_uptime",
        name="System Uptime",
        description="System availability percentage",
        type=MetricType.RELIABILITY,
        category=MetricCategory.SLO,
        unit="percent",
        threshold=MetricThreshold(
            warning_min=99,
            critical_min=98,
            target_value=99.9
        ),
        weight=3.0
    ),
    
    # Business KPIs
    "cost_per_video": QualityMetric(
        id="cost_per_video",
        name="Cost Per Video",
        description="Average cost to generate one video",
        type=MetricType.BUSINESS,
        category=MetricCategory.KPI,
        unit="usd",
        threshold=MetricThreshold(
            warning_max=2.50,
            critical_max=3.00,
            target_value=2.00
        ),
        is_higher_better=False,
        weight=2.0
    ),
    
    "user_satisfaction_score": QualityMetric(
        id="user_satisfaction_score",
        name="User Satisfaction Score",
        description="Average user satisfaction rating (1-5)",
        type=MetricType.USER_EXPERIENCE,
        category=MetricCategory.KPI,
        unit="score",
        threshold=MetricThreshold(
            warning_min=4.0,
            critical_min=3.5,
            target_value=4.5
        ),
        weight=2.5
    ),
    
    "video_quality_score": QualityMetric(
        id="video_quality_score",
        name="Video Quality Score",
        description="AI-evaluated video quality score",
        type=MetricType.TECHNICAL,
        category=MetricCategory.KPI,
        unit="score",
        threshold=MetricThreshold(
            warning_min=85,
            critical_min=75,
            target_value=90
        ),
        weight=2.0
    ),
    
    # Security KPIs
    "security_incidents": QualityMetric(
        id="security_incidents",
        name="Security Incidents",
        description="Number of security incidents per day",
        type=MetricType.SECURITY,
        category=MetricCategory.KPI,
        unit="count",
        threshold=MetricThreshold(
            warning_max=1,
            critical_max=3,
            target_value=0
        ),
        is_higher_better=False,
        weight=3.0
    ),
    
    # User Experience KPIs
    "user_onboarding_completion": QualityMetric(
        id="user_onboarding_completion",
        name="User Onboarding Completion Rate",
        description="Percentage of users completing onboarding",
        type=MetricType.USER_EXPERIENCE,
        category=MetricCategory.KPI,
        unit="percent",
        threshold=MetricThreshold(
            warning_min=80,
            critical_min=70,
            target_value=90
        )
    ),
    
    "daily_active_users": QualityMetric(
        id="daily_active_users",
        name="Daily Active Users",
        description="Number of daily active users",
        type=MetricType.BUSINESS,
        category=MetricCategory.KPI,
        unit="count",
        threshold=MetricThreshold(
            warning_min=100,
            critical_min=50,
            target_value=500
        )
    ),
    
    "error_rate": QualityMetric(
        id="error_rate",
        name="Error Rate",
        description="Percentage of requests resulting in errors",
        type=MetricType.RELIABILITY,
        category=MetricCategory.SLI,
        unit="percent",
        threshold=MetricThreshold(
            warning_max=1,
            critical_max=5,
            target_value=0.1
        ),
        is_higher_better=False,
        weight=2.0
    )
}

# ============================================================================
# Quality Metrics Collector
# ============================================================================

class QualityMetricsCollector:
    """Collects and processes quality metrics"""
    
    def __init__(
        self,
        db_session: Optional[AsyncSession] = None,
        redis_client: Optional[redis.Redis] = None
    ):
        self.db_session = db_session
        self.redis_client = redis_client
        self.metrics_cache = deque(maxlen=10000)
        self.collection_tasks = {}
    
    async def collect_metric(
        self,
        metric_id: str,
        value: float,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> MetricValue:
        """Collect a single metric value"""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        metric_def = QUALITY_METRICS.get(metric_id)
        if not metric_def:
            logger.warning(f"Unknown metric ID: {metric_id}")
            return None
        
        # Determine status based on thresholds
        status = self._evaluate_metric_status(metric_def, value)
        
        metric_value = MetricValue(
            metric_id=metric_id,
            value=value,
            timestamp=timestamp,
            status=status,
            metadata=metadata or {},
            tags=tags or {}
        )
        
        # Store in cache
        self.metrics_cache.append(metric_value)
        
        # Store in Redis for real-time access
        if self.redis_client:
            await self._store_in_redis(metric_value)
        
        # Store in database for historical data
        if self.db_session:
            await self._store_in_database(metric_value)
        
        logger.debug(f"Collected metric {metric_id}: {value} ({status})")
        return metric_value
    
    def _evaluate_metric_status(self, metric_def: QualityMetric, value: float) -> MetricStatus:
        """Evaluate metric status based on thresholds"""
        threshold = metric_def.threshold
        
        # Check critical thresholds
        if threshold.critical_min is not None and value < threshold.critical_min:
            return MetricStatus.CRITICAL
        if threshold.critical_max is not None and value > threshold.critical_max:
            return MetricStatus.CRITICAL
        
        # Check warning thresholds
        if threshold.warning_min is not None and value < threshold.warning_min:
            return MetricStatus.WARNING
        if threshold.warning_max is not None and value > threshold.warning_max:
            return MetricStatus.WARNING
        
        return MetricStatus.HEALTHY
    
    async def _store_in_redis(self, metric_value: MetricValue):
        """Store metric in Redis for real-time access"""
        try:
            key = f"metric:{metric_value.metric_id}:latest"
            await self.redis_client.setex(
                key,
                3600,  # 1 hour TTL
                json.dumps({
                    "value": metric_value.value,
                    "timestamp": metric_value.timestamp.isoformat(),
                    "status": metric_value.status.value,
                    "metadata": metric_value.metadata,
                    "tags": metric_value.tags
                })
            )
            
            # Store in time series for trends
            ts_key = f"metric:{metric_value.metric_id}:timeseries"
            await self.redis_client.zadd(
                ts_key,
                {
                    json.dumps({
                        "value": metric_value.value,
                        "status": metric_value.status.value
                    }): int(metric_value.timestamp.timestamp())
                }
            )
            await self.redis_client.expire(ts_key, 86400 * 7)  # 7 days
            
        except Exception as e:
            logger.error(f"Failed to store metric in Redis: {e}")
    
    async def _store_in_database(self, metric_value: MetricValue):
        """Store metric in database for historical data"""
        try:
            await self.db_session.execute(
                text("""
                    INSERT INTO quality_metrics (
                        metric_id, value, timestamp, status, metadata, tags
                    ) VALUES (
                        :metric_id, :value, :timestamp, :status, :metadata, :tags
                    )
                """),
                {
                    "metric_id": metric_value.metric_id,
                    "value": metric_value.value,
                    "timestamp": metric_value.timestamp,
                    "status": metric_value.status.value,
                    "metadata": json.dumps(metric_value.metadata),
                    "tags": json.dumps(metric_value.tags)
                }
            )
            await self.db_session.commit()
        except Exception as e:
            logger.error(f"Failed to store metric in database: {e}")
    
    async def get_latest_metrics(self) -> Dict[str, MetricValue]:
        """Get latest values for all metrics"""
        latest_metrics = {}
        
        if self.redis_client:
            try:
                keys = await self.redis_client.keys("metric:*:latest")
                for key in keys:
                    metric_id = key.decode().split(":")[1]
                    data = await self.redis_client.get(key)
                    if data:
                        parsed = json.loads(data)
                        latest_metrics[metric_id] = MetricValue(
                            metric_id=metric_id,
                            value=parsed["value"],
                            timestamp=datetime.fromisoformat(parsed["timestamp"]),
                            status=MetricStatus(parsed["status"]),
                            metadata=parsed["metadata"],
                            tags=parsed["tags"]
                        )
            except Exception as e:
                logger.error(f"Failed to get latest metrics from Redis: {e}")
        
        return latest_metrics
    
    async def get_metric_history(
        self,
        metric_id: str,
        start_time: datetime,
        end_time: datetime,
        aggregation: str = "raw"
    ) -> List[MetricValue]:
        """Get historical metric data"""
        if self.db_session:
            try:
                query = text("""
                    SELECT metric_id, value, timestamp, status, metadata, tags
                    FROM quality_metrics
                    WHERE metric_id = :metric_id
                    AND timestamp BETWEEN :start_time AND :end_time
                    ORDER BY timestamp
                """)
                
                result = await self.db_session.execute(query, {
                    "metric_id": metric_id,
                    "start_time": start_time,
                    "end_time": end_time
                })
                
                metrics = []
                for row in result:
                    metrics.append(MetricValue(
                        metric_id=row.metric_id,
                        value=row.value,
                        timestamp=row.timestamp,
                        status=MetricStatus(row.status),
                        metadata=json.loads(row.metadata or "{}"),
                        tags=json.loads(row.tags or "{}")
                    ))
                
                return metrics
                
            except Exception as e:
                logger.error(f"Failed to get metric history: {e}")
        
        return []


# ============================================================================
# Quality Dashboard Generator
# ============================================================================

class QualityDashboard:
    """Generates quality dashboards and reports"""
    
    def __init__(self, collector: QualityMetricsCollector):
        self.collector = collector
    
    async def generate_quality_overview(self) -> Dict[str, Any]:
        """Generate overall quality overview"""
        latest_metrics = await self.collector.get_latest_metrics()
        
        overview = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": "healthy",
            "metrics_count": len(latest_metrics),
            "categories": {},
            "critical_issues": [],
            "warnings": [],
            "quality_score": 0.0
        }
        
        # Categorize metrics and calculate scores
        category_scores = defaultdict(list)
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for metric_id, metric_value in latest_metrics.items():
            metric_def = QUALITY_METRICS.get(metric_id)
            if not metric_def:
                continue
            
            category = metric_def.category.value
            if category not in overview["categories"]:
                overview["categories"][category] = {
                    "metrics": [],
                    "status": "healthy",
                    "score": 0.0
                }
            
            # Calculate metric score (0-100)
            metric_score = self._calculate_metric_score(metric_def, metric_value.value)
            
            metric_info = {
                "id": metric_id,
                "name": metric_def.name,
                "value": metric_value.value,
                "unit": metric_def.unit,
                "status": metric_value.status.value,
                "score": metric_score,
                "target": metric_def.threshold.target_value
            }
            
            overview["categories"][category]["metrics"].append(metric_info)
            category_scores[category].append(metric_score * metric_def.weight)
            
            # Track issues
            if metric_value.status == MetricStatus.CRITICAL:
                overview["critical_issues"].append(metric_info)
                overview["overall_status"] = "critical"
            elif metric_value.status == MetricStatus.WARNING:
                overview["warnings"].append(metric_info)
                if overview["overall_status"] == "healthy":
                    overview["overall_status"] = "warning"
            
            # Contribute to overall score
            total_weighted_score += metric_score * metric_def.weight
            total_weight += metric_def.weight
        
        # Calculate category scores
        for category, scores in category_scores.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                overview["categories"][category]["score"] = round(avg_score, 2)
                
                # Determine category status
                if avg_score < 70:
                    overview["categories"][category]["status"] = "critical"
                elif avg_score < 85:
                    overview["categories"][category]["status"] = "warning"
        
        # Calculate overall quality score
        if total_weight > 0:
            overview["quality_score"] = round(total_weighted_score / total_weight, 2)
        
        return overview
    
    def _calculate_metric_score(self, metric_def: QualityMetric, value: float) -> float:
        """Calculate metric score (0-100) based on value and thresholds"""
        threshold = metric_def.threshold
        
        # If we have a target value, calculate distance from target
        if threshold.target_value is not None:
            target = threshold.target_value
            
            if metric_def.is_higher_better:
                if value >= target:
                    return 100.0
                elif threshold.critical_min is not None:
                    # Scale between critical and target
                    if value <= threshold.critical_min:
                        return 0.0
                    else:
                        return ((value - threshold.critical_min) / 
                               (target - threshold.critical_min)) * 100
            else:
                if value <= target:
                    return 100.0
                elif threshold.critical_max is not None:
                    # Scale between target and critical
                    if value >= threshold.critical_max:
                        return 0.0
                    else:
                        return ((threshold.critical_max - value) / 
                               (threshold.critical_max - target)) * 100
        
        # Fallback to threshold-based scoring
        if metric_def.is_higher_better:
            if threshold.critical_min is not None and value < threshold.critical_min:
                return 0.0
            elif threshold.warning_min is not None and value < threshold.warning_min:
                return 50.0
            else:
                return 100.0
        else:
            if threshold.critical_max is not None and value > threshold.critical_max:
                return 0.0
            elif threshold.warning_max is not None and value > threshold.warning_max:
                return 50.0
            else:
                return 100.0
    
    async def generate_trend_analysis(self, days: int = 7) -> Dict[str, Any]:
        """Generate trend analysis for metrics"""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        
        trends = {
            "period": f"Last {days} days",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "metrics": {}
        }
        
        for metric_id in QUALITY_METRICS.keys():
            history = await self.collector.get_metric_history(metric_id, start_time, end_time)
            
            if len(history) >= 2:
                values = [m.value for m in history]
                
                # Calculate trend statistics
                first_value = values[0]
                last_value = values[-1]
                avg_value = statistics.mean(values)
                std_dev = statistics.stdev(values) if len(values) > 1 else 0
                
                # Calculate trend direction
                if len(values) >= 10:
                    # Use linear regression for trend
                    x_values = list(range(len(values)))
                    slope = self._calculate_slope(x_values, values)
                    trend_direction = "improving" if slope > 0 else "declining" if slope < 0 else "stable"
                else:
                    # Simple comparison
                    change_pct = ((last_value - first_value) / first_value) * 100 if first_value != 0 else 0
                    if abs(change_pct) < 5:
                        trend_direction = "stable"
                    elif change_pct > 0:
                        trend_direction = "improving"
                    else:
                        trend_direction = "declining"
                
                trends["metrics"][metric_id] = {
                    "data_points": len(history),
                    "first_value": first_value,
                    "last_value": last_value,
                    "average": round(avg_value, 2),
                    "std_deviation": round(std_dev, 2),
                    "trend_direction": trend_direction,
                    "change_percent": round(((last_value - first_value) / first_value) * 100, 2) if first_value != 0 else 0
                }
        
        return trends
    
    def _calculate_slope(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate slope for linear trend"""
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope
    
    async def generate_quality_report(self, format: str = "json") -> Union[Dict, str]:
        """Generate comprehensive quality report"""
        overview = await self.generate_quality_overview()
        trends = await self.generate_trend_analysis()
        
        report = {
            "report_id": f"quality_report_{int(datetime.now().timestamp())}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "overview": overview,
            "trends": trends,
            "recommendations": self._generate_recommendations(overview, trends),
            "summary": {
                "overall_quality_score": overview["quality_score"],
                "critical_issues_count": len(overview["critical_issues"]),
                "warnings_count": len(overview["warnings"]),
                "metrics_tracked": overview["metrics_count"],
                "data_coverage": self._calculate_data_coverage()
            }
        }
        
        if format == "html":
            return self._generate_html_report(report)
        elif format == "markdown":
            return self._generate_markdown_report(report)
        else:
            return report
    
    def _generate_recommendations(self, overview: Dict, trends: Dict) -> List[Dict]:
        """Generate actionable recommendations based on metrics"""
        recommendations = []
        
        # Check for critical issues
        for issue in overview["critical_issues"]:
            recommendations.append({
                "priority": "high",
                "category": "critical_fix",
                "title": f"Address critical issue: {issue['name']}",
                "description": f"Metric '{issue['name']}' is in critical state with value {issue['value']} {issue['unit']}",
                "action": f"Investigate and fix immediately to reach target of {issue['target']} {issue['unit']}"
            })
        
        # Check for declining trends
        for metric_id, trend in trends["metrics"].items():
            if trend["trend_direction"] == "declining" and abs(trend["change_percent"]) > 10:
                metric_def = QUALITY_METRICS.get(metric_id)
                if metric_def:
                    recommendations.append({
                        "priority": "medium",
                        "category": "trend_improvement",
                        "title": f"Improve declining trend: {metric_def.name}",
                        "description": f"Metric has declined by {abs(trend['change_percent']):.1f}% over the last week",
                        "action": "Analyze root cause and implement corrective measures"
                    })
        
        # Check for low quality scores
        for category, data in overview["categories"].items():
            if data["score"] < 80:
                recommendations.append({
                    "priority": "medium",
                    "category": "category_improvement",
                    "title": f"Improve {category} metrics",
                    "description": f"Category score is {data['score']:.1f}, below target of 80",
                    "action": f"Focus improvement efforts on {category} metrics"
                })
        
        return recommendations
    
    def _calculate_data_coverage(self) -> float:
        """Calculate percentage of metrics with recent data"""
        # This would check how many metrics have recent data
        return 95.0  # Placeholder
    
    def _generate_html_report(self, report: Dict) -> str:
        """Generate HTML quality report"""
        # Placeholder for HTML generation
        return f"<html><body><h1>Quality Report {report['report_id']}</h1></body></html>"
    
    def _generate_markdown_report(self, report: Dict) -> str:
        """Generate Markdown quality report"""
        md = f"""# Quality Report {report['report_id']}

Generated: {report['generated_at']}

## Summary
- **Overall Quality Score**: {report['summary']['overall_quality_score']}/100
- **Critical Issues**: {report['summary']['critical_issues_count']}
- **Warnings**: {report['summary']['warnings_count']}
- **Metrics Tracked**: {report['summary']['metrics_tracked']}

## Status: {report['overview']['overall_status'].upper()}

"""
        
        if report['overview']['critical_issues']:
            md += "## Critical Issues\n"
            for issue in report['overview']['critical_issues']:
                md += f"- **{issue['name']}**: {issue['value']} {issue['unit']} (Target: {issue['target']})\n"
            md += "\n"
        
        if report['recommendations']:
            md += "## Recommendations\n"
            for rec in report['recommendations'][:5]:  # Top 5
                md += f"- **{rec['title']}** ({rec['priority']} priority): {rec['description']}\n"
        
        return md


# ============================================================================
# Automated Quality Monitoring
# ============================================================================

class QualityMonitor:
    """Automated quality monitoring and alerting"""
    
    def __init__(
        self,
        collector: QualityMetricsCollector,
        dashboard: QualityDashboard
    ):
        self.collector = collector
        self.dashboard = dashboard
        self.monitoring_tasks = {}
        self.alert_rules = []
    
    async def start_monitoring(self):
        """Start automated quality monitoring"""
        logger.info("Starting quality monitoring")
        
        # Start metric collection tasks
        for metric_id, metric_def in QUALITY_METRICS.items():
            task = asyncio.create_task(
                self._collect_metric_periodically(metric_id, metric_def.collection_interval)
            )
            self.monitoring_tasks[metric_id] = task
        
        # Start dashboard updates
        dashboard_task = asyncio.create_task(self._update_dashboard_periodically())
        self.monitoring_tasks["dashboard"] = dashboard_task
    
    async def stop_monitoring(self):
        """Stop automated monitoring"""
        logger.info("Stopping quality monitoring")
        
        for task in self.monitoring_tasks.values():
            task.cancel()
        
        self.monitoring_tasks.clear()
    
    async def _collect_metric_periodically(self, metric_id: str, interval: int):
        """Periodically collect a specific metric"""
        while True:
            try:
                # This would call specific collection methods based on metric type
                value = await self._collect_specific_metric(metric_id)
                if value is not None:
                    await self.collector.collect_metric(metric_id, value)
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error collecting metric {metric_id}: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def _collect_specific_metric(self, metric_id: str) -> Optional[float]:
        """Collect specific metric value"""
        # This would implement actual metric collection logic
        # For now, return mock values
        import random
        
        if metric_id == "api_response_time_p95":
            return random.uniform(150, 300)
        elif metric_id == "video_generation_success_rate":
            return random.uniform(95, 99.5)
        elif metric_id == "system_uptime":
            return random.uniform(99.5, 99.99)
        elif metric_id == "cost_per_video":
            return random.uniform(1.80, 2.20)
        elif metric_id == "user_satisfaction_score":
            return random.uniform(4.2, 4.8)
        elif metric_id == "video_quality_score":
            return random.uniform(85, 95)
        elif metric_id == "security_incidents":
            return random.choice([0, 0, 0, 1])  # Mostly 0
        elif metric_id == "user_onboarding_completion":
            return random.uniform(82, 88)
        elif metric_id == "daily_active_users":
            return random.uniform(150, 250)
        elif metric_id == "error_rate":
            return random.uniform(0.1, 0.5)
        
        return None
    
    async def _update_dashboard_periodically(self):
        """Periodically update quality dashboard"""
        while True:
            try:
                # Generate quality report
                report = await self.dashboard.generate_quality_report()
                
                # Store report for access
                if self.collector.redis_client:
                    await self.collector.redis_client.setex(
                        "quality:latest_report",
                        3600,
                        json.dumps(report)
                    )
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error updating dashboard: {e}")
                await asyncio.sleep(60)


# ============================================================================
# Global Instances
# ============================================================================

# Initialize global instances
metrics_collector = QualityMetricsCollector()
quality_dashboard = QualityDashboard(metrics_collector)
quality_monitor = QualityMonitor(metrics_collector, quality_dashboard)