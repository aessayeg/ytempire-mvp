"""
Analytics Data Pipeline Service
Real-time analytics processing and aggregation for YTEmpire
"""
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field, asdict
from enum import Enum
import redis.asyncio as redis
from collections import defaultdict, deque
import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
import httpx

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics to track"""
    VIEWS = "views"
    WATCH_TIME = "watch_time"
    LIKES = "likes"
    COMMENTS = "comments"
    SHARES = "shares"
    SUBSCRIBERS = "subscribers"
    REVENUE = "revenue"
    CTR = "ctr"
    AVG_VIEW_DURATION = "avg_view_duration"
    IMPRESSIONS = "impressions"
    ENGAGEMENT_RATE = "engagement_rate"
    COST = "cost"
    ROI = "roi"
    CONVERSION_RATE = "conversion_rate"


class AggregationLevel(str, Enum):
    """Data aggregation levels"""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class DataSource(str, Enum):
    """Analytics data sources"""
    YOUTUBE = "youtube"
    YOUTUBE_ANALYTICS = "youtube_analytics"
    INTERNAL = "internal"
    STRIPE = "stripe"
    OPENAI = "openai"
    CUSTOM = "custom"


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: datetime
    metric_type: MetricType
    value: float
    source: DataSource
    channel_id: Optional[str] = None
    video_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedMetric:
    """Aggregated metric data"""
    period_start: datetime
    period_end: datetime
    aggregation_level: AggregationLevel
    metric_type: MetricType
    count: int
    sum: float
    avg: float
    min: float
    max: float
    std_dev: float
    percentile_25: float
    percentile_50: float
    percentile_75: float
    percentile_95: float
    channel_id: Optional[str] = None
    video_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyticsReport:
    """Analytics report structure"""
    report_id: str
    report_type: str
    period_start: datetime
    period_end: datetime
    generated_at: datetime
    metrics: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    charts: Dict[str, Any]
    

class AnalyticsPipeline:
    """
    Real-time analytics data pipeline for processing and aggregating metrics
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        batch_size: int = 1000,
        flush_interval: int = 30
    ):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # In-memory buffers
        self.metric_buffer: deque = deque(maxlen=10000)
        self.aggregation_buffer: Dict[str, List[MetricPoint]] = defaultdict(list)
        
        # Processing state
        self.processing = False
        self.last_flush = datetime.utcnow()
        
        # Aggregation windows
        self.aggregation_windows = {
            AggregationLevel.MINUTE: timedelta(minutes=1),
            AggregationLevel.HOUR: timedelta(hours=1),
            AggregationLevel.DAY: timedelta(days=1),
            AggregationLevel.WEEK: timedelta(weeks=1),
            AggregationLevel.MONTH: timedelta(days=30),
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            "low_engagement": 0.02,  # 2% engagement rate
            "high_cost": 100.0,  # $100 daily cost
            "low_roi": 0.5,  # 50% ROI
            "view_drop": 0.3,  # 30% view drop
        }
        
    async def initialize(self):
        """Initialize the analytics pipeline"""
        try:
            self.redis_client = await redis.from_url(self.redis_url)
            
            # Start background tasks
            asyncio.create_task(self.process_metrics())
            asyncio.create_task(self.aggregate_metrics())
            asyncio.create_task(self.generate_reports())
            asyncio.create_task(self.monitor_alerts())
            
            logger.info("Analytics pipeline initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize analytics pipeline: {e}")
            raise
            
    async def ingest_metric(self, metric: MetricPoint):
        """
        Ingest a single metric into the pipeline
        """
        try:
            # Add to buffer
            self.metric_buffer.append(metric)
            
            # Store in Redis for real-time access
            key = f"metric:{metric.metric_type.value}:{metric.timestamp.timestamp()}"
            await self.redis_client.setex(
                key,
                3600,  # 1 hour TTL
                json.dumps(asdict(metric), default=str)
            )
            
            # Check if batch processing needed
            if len(self.metric_buffer) >= self.batch_size:
                await self.flush_metrics()
                
        except Exception as e:
            logger.error(f"Failed to ingest metric: {e}")
            
    async def ingest_batch(self, metrics: List[MetricPoint]):
        """
        Ingest a batch of metrics
        """
        for metric in metrics:
            await self.ingest_metric(metric)
            
    async def process_metrics(self):
        """
        Background task to process buffered metrics
        """
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                
                if self.metric_buffer:
                    await self.flush_metrics()
                    
            except Exception as e:
                logger.error(f"Metric processing error: {e}")
                
    async def flush_metrics(self):
        """
        Flush buffered metrics to storage and aggregation
        """
        if not self.metric_buffer:
            return
            
        try:
            # Copy buffer
            metrics = list(self.metric_buffer)
            self.metric_buffer.clear()
            
            # Group by type and source
            grouped = defaultdict(list)
            for metric in metrics:
                key = f"{metric.metric_type.value}:{metric.source.value}"
                grouped[key].append(metric)
            
            # Store grouped metrics
            for key, group_metrics in grouped.items():
                await self.store_metrics(key, group_metrics)
                
                # Add to aggregation buffer
                self.aggregation_buffer[key].extend(group_metrics)
            
            self.last_flush = datetime.utcnow()
            logger.info(f"Flushed {len(metrics)} metrics")
            
        except Exception as e:
            logger.error(f"Failed to flush metrics: {e}")
            
    async def store_metrics(self, key: str, metrics: List[MetricPoint]):
        """
        Store metrics in time-series format
        """
        # Convert to DataFrame for efficient storage
        df_data = []
        for metric in metrics:
            df_data.append({
                "timestamp": metric.timestamp,
                "value": metric.value,
                "channel_id": metric.channel_id,
                "video_id": metric.video_id,
                "user_id": metric.user_id,
                **metric.metadata
            })
        
        df = pd.DataFrame(df_data)
        
        # Store in Redis as compressed JSON
        redis_key = f"timeseries:{key}:{datetime.utcnow().strftime('%Y%m%d%H')}"
        compressed_data = df.to_json(orient='records', date_format='iso')
        
        await self.redis_client.setex(
            redis_key,
            86400 * 7,  # 7 days retention
            compressed_data
        )
        
    async def aggregate_metrics(self):
        """
        Background task to aggregate metrics at different levels
        """
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                for key, metrics in self.aggregation_buffer.items():
                    if metrics:
                        for level in AggregationLevel:
                            await self.aggregate_at_level(key, metrics, level)
                
                # Clear processed metrics
                self.aggregation_buffer.clear()
                
            except Exception as e:
                logger.error(f"Aggregation error: {e}")
                
    async def aggregate_at_level(
        self,
        key: str,
        metrics: List[MetricPoint],
        level: AggregationLevel
    ):
        """
        Aggregate metrics at a specific level
        """
        if not metrics:
            return
            
        window = self.aggregation_windows.get(level)
        if not window:
            return
        
        # Group metrics by window
        windowed_groups = defaultdict(list)
        for metric in metrics:
            window_start = self.get_window_start(metric.timestamp, level)
            windowed_groups[window_start].append(metric.value)
        
        # Calculate aggregations
        for window_start, values in windowed_groups.items():
            if not values:
                continue
                
            values_array = np.array(values)
            
            aggregated = AggregatedMetric(
                period_start=window_start,
                period_end=window_start + window,
                aggregation_level=level,
                metric_type=metrics[0].metric_type,
                count=len(values),
                sum=float(np.sum(values_array)),
                avg=float(np.mean(values_array)),
                min=float(np.min(values_array)),
                max=float(np.max(values_array)),
                std_dev=float(np.std(values_array)),
                percentile_25=float(np.percentile(values_array, 25)),
                percentile_50=float(np.percentile(values_array, 50)),
                percentile_75=float(np.percentile(values_array, 75)),
                percentile_95=float(np.percentile(values_array, 95)),
                channel_id=metrics[0].channel_id,
                video_id=metrics[0].video_id
            )
            
            # Store aggregated metric
            await self.store_aggregated_metric(aggregated)
            
    async def store_aggregated_metric(self, aggregated: AggregatedMetric):
        """
        Store aggregated metric
        """
        key = f"aggregated:{aggregated.metric_type.value}:{aggregated.aggregation_level.value}:{aggregated.period_start.timestamp()}"
        
        await self.redis_client.setex(
            key,
            86400 * 30,  # 30 days retention
            json.dumps(asdict(aggregated), default=str)
        )
        
    async def query_metrics(
        self,
        metric_type: MetricType,
        start_time: datetime,
        end_time: datetime,
        aggregation_level: Optional[AggregationLevel] = None,
        channel_id: Optional[str] = None,
        video_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query metrics for a time range
        """
        results = []
        
        if aggregation_level:
            # Query aggregated metrics
            pattern = f"aggregated:{metric_type.value}:{aggregation_level.value}:*"
        else:
            # Query raw metrics
            pattern = f"timeseries:{metric_type.value}:*"
        
        # Scan Redis for matching keys
        cursor = 0
        while True:
            cursor, keys = await self.redis_client.scan(
                cursor=cursor,
                match=pattern,
                count=100
            )
            
            for key in keys:
                data = await self.redis_client.get(key)
                if data:
                    metrics = json.loads(data)
                    
                    # Filter by time range and IDs
                    for metric in metrics if isinstance(metrics, list) else [metrics]:
                        metric_time = datetime.fromisoformat(metric.get('timestamp') or metric.get('period_start'))
                        
                        if start_time <= metric_time <= end_time:
                            if channel_id and metric.get('channel_id') != channel_id:
                                continue
                            if video_id and metric.get('video_id') != video_id:
                                continue
                            
                            results.append(metric)
            
            if cursor == 0:
                break
        
        return results
        
    async def get_channel_analytics(
        self,
        channel_id: str,
        start_date: date,
        end_date: date
    ) -> Dict[str, Any]:
        """
        Get comprehensive analytics for a channel
        """
        start_time = datetime.combine(start_date, datetime.min.time())
        end_time = datetime.combine(end_date, datetime.max.time())
        
        analytics = {
            "channel_id": channel_id,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "metrics": {},
            "trends": {},
            "top_videos": [],
            "engagement": {}
        }
        
        # Fetch all metric types
        for metric_type in MetricType:
            metrics = await self.query_metrics(
                metric_type=metric_type,
                start_time=start_time,
                end_time=end_time,
                channel_id=channel_id,
                aggregation_level=AggregationLevel.DAY
            )
            
            if metrics:
                # Calculate summary stats
                values = [m.get('avg', m.get('value', 0)) for m in metrics]
                analytics["metrics"][metric_type.value] = {
                    "total": sum(values),
                    "average": np.mean(values) if values else 0,
                    "trend": self.calculate_trend(values)
                }
        
        # Calculate engagement metrics
        if analytics["metrics"].get("views") and analytics["metrics"].get("likes"):
            views = analytics["metrics"]["views"]["total"]
            likes = analytics["metrics"]["likes"]["total"]
            comments = analytics["metrics"].get("comments", {}).get("total", 0)
            
            analytics["engagement"] = {
                "rate": (likes + comments) / views if views > 0 else 0,
                "likes_per_view": likes / views if views > 0 else 0,
                "comments_per_view": comments / views if views > 0 else 0
            }
        
        return analytics
        
    async def get_video_performance(
        self,
        video_id: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get performance metrics for a specific video
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        performance = {
            "video_id": video_id,
            "period_days": days,
            "metrics": {},
            "hourly_views": [],
            "retention": {},
            "traffic_sources": {}
        }
        
        # Get view metrics
        views = await self.query_metrics(
            metric_type=MetricType.VIEWS,
            start_time=start_time,
            end_time=end_time,
            video_id=video_id,
            aggregation_level=AggregationLevel.HOUR
        )
        
        if views:
            performance["hourly_views"] = [
                {
                    "hour": v.get("period_start"),
                    "views": v.get("sum", 0)
                }
                for v in views
            ]
            
            performance["metrics"]["total_views"] = sum(v.get("sum", 0) for v in views)
            performance["metrics"]["peak_hour_views"] = max(v.get("sum", 0) for v in views)
        
        # Get watch time
        watch_time = await self.query_metrics(
            metric_type=MetricType.WATCH_TIME,
            start_time=start_time,
            end_time=end_time,
            video_id=video_id
        )
        
        if watch_time:
            total_watch_minutes = sum(w.get("value", 0) for w in watch_time)
            performance["metrics"]["total_watch_hours"] = total_watch_minutes / 60
            
            if performance["metrics"].get("total_views"):
                performance["metrics"]["avg_view_duration"] = (
                    total_watch_minutes / performance["metrics"]["total_views"]
                )
        
        return performance
        
    async def generate_reports(self):
        """
        Background task to generate periodic reports
        """
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Generate daily report at midnight
                now = datetime.utcnow()
                if now.hour == 0:
                    await self.generate_daily_report()
                
                # Generate weekly report on Mondays
                if now.weekday() == 0 and now.hour == 0:
                    await self.generate_weekly_report()
                    
            except Exception as e:
                logger.error(f"Report generation error: {e}")
                
    async def generate_daily_report(self) -> AnalyticsReport:
        """
        Generate daily analytics report
        """
        yesterday = date.today() - timedelta(days=1)
        
        report = AnalyticsReport(
            report_id=f"daily_{yesterday.isoformat()}",
            report_type="daily",
            period_start=datetime.combine(yesterday, datetime.min.time()),
            period_end=datetime.combine(yesterday, datetime.max.time()),
            generated_at=datetime.utcnow(),
            metrics={},
            insights=[],
            recommendations=[],
            charts={}
        )
        
        # Collect metrics for all channels
        # This would typically query from database
        
        # Generate insights
        report.insights = await self.generate_insights(report.metrics)
        
        # Generate recommendations
        report.recommendations = await self.generate_recommendations(report.metrics)
        
        # Store report
        await self.store_report(report)
        
        return report
        
    async def generate_weekly_report(self) -> AnalyticsReport:
        """
        Generate weekly analytics report
        """
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=6)
        
        report = AnalyticsReport(
            report_id=f"weekly_{start_date.isoformat()}_{end_date.isoformat()}",
            report_type="weekly",
            period_start=datetime.combine(start_date, datetime.min.time()),
            period_end=datetime.combine(end_date, datetime.max.time()),
            generated_at=datetime.utcnow(),
            metrics={},
            insights=[],
            recommendations=[],
            charts={}
        )
        
        # Generate comprehensive weekly metrics
        # This would aggregate daily reports
        
        return report
        
    async def monitor_alerts(self):
        """
        Monitor metrics for alert conditions
        """
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Check engagement rate
                await self.check_engagement_alerts()
                
                # Check cost alerts
                await self.check_cost_alerts()
                
                # Check performance alerts
                await self.check_performance_alerts()
                
            except Exception as e:
                logger.error(f"Alert monitoring error: {e}")
                
    async def check_engagement_alerts(self):
        """
        Check for low engagement alerts
        """
        # Query recent engagement metrics
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)
        
        engagement_metrics = await self.query_metrics(
            metric_type=MetricType.ENGAGEMENT_RATE,
            start_time=start_time,
            end_time=end_time,
            aggregation_level=AggregationLevel.HOUR
        )
        
        for metric in engagement_metrics:
            if metric.get("avg", 1) < self.alert_thresholds["low_engagement"]:
                await self.send_alert(
                    "low_engagement",
                    f"Low engagement rate detected: {metric.get('avg', 0):.2%}",
                    metric
                )
                
    async def check_cost_alerts(self):
        """
        Check for high cost alerts
        """
        today = date.today()
        start_time = datetime.combine(today, datetime.min.time())
        end_time = datetime.utcnow()
        
        cost_metrics = await self.query_metrics(
            metric_type=MetricType.COST,
            start_time=start_time,
            end_time=end_time
        )
        
        total_cost = sum(m.get("value", 0) for m in cost_metrics)
        
        if total_cost > self.alert_thresholds["high_cost"]:
            await self.send_alert(
                "high_cost",
                f"Daily cost exceeded threshold: ${total_cost:.2f}",
                {"total_cost": total_cost, "threshold": self.alert_thresholds["high_cost"]}
            )
            
    async def check_performance_alerts(self):
        """
        Check for performance degradation alerts
        """
        # Compare current hour to previous hour
        end_time = datetime.utcnow()
        current_hour_start = end_time.replace(minute=0, second=0, microsecond=0)
        previous_hour_start = current_hour_start - timedelta(hours=1)
        
        # Get current hour views
        current_views = await self.query_metrics(
            metric_type=MetricType.VIEWS,
            start_time=current_hour_start,
            end_time=end_time
        )
        
        # Get previous hour views
        previous_views = await self.query_metrics(
            metric_type=MetricType.VIEWS,
            start_time=previous_hour_start,
            end_time=current_hour_start
        )
        
        if current_views and previous_views:
            current_total = sum(v.get("value", 0) for v in current_views)
            previous_total = sum(v.get("value", 0) for v in previous_views)
            
            if previous_total > 0:
                drop_rate = 1 - (current_total / previous_total)
                
                if drop_rate > self.alert_thresholds["view_drop"]:
                    await self.send_alert(
                        "view_drop",
                        f"Significant view drop detected: {drop_rate:.1%}",
                        {
                            "current_views": current_total,
                            "previous_views": previous_total,
                            "drop_rate": drop_rate
                        }
                    )
                    
    async def send_alert(self, alert_type: str, message: str, data: Dict[str, Any]):
        """
        Send alert notification
        """
        alert = {
            "type": alert_type,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
        
        # Store alert in Redis
        await self.redis_client.rpush(
            "alerts:queue",
            json.dumps(alert)
        )
        
        # Log alert
        logger.warning(f"Alert: {alert_type} - {message}")
        
    async def store_report(self, report: AnalyticsReport):
        """
        Store analytics report
        """
        key = f"report:{report.report_type}:{report.report_id}"
        
        await self.redis_client.setex(
            key,
            86400 * 90,  # 90 days retention
            json.dumps(asdict(report), default=str)
        )
        
    async def generate_insights(self, metrics: Dict[str, Any]) -> List[str]:
        """
        Generate insights from metrics
        """
        insights = []
        
        # Example insight generation logic
        if metrics.get("views", {}).get("trend") == "increasing":
            insights.append("View count is trending upward")
            
        if metrics.get("engagement", {}).get("rate", 0) > 0.05:
            insights.append("Engagement rate is above average")
            
        return insights
        
    async def generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on metrics
        """
        recommendations = []
        
        # Example recommendation logic
        if metrics.get("watch_time", {}).get("average", 0) < 2:
            recommendations.append("Consider creating more engaging content to increase watch time")
            
        if metrics.get("cost", {}).get("total", 0) > 50:
            recommendations.append("Review cost optimization strategies")
            
        return recommendations
        
    def calculate_trend(self, values: List[float]) -> str:
        """
        Calculate trend direction from values
        """
        if len(values) < 2:
            return "stable"
            
        # Simple linear regression
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
            
    def get_window_start(self, timestamp: datetime, level: AggregationLevel) -> datetime:
        """
        Get the start of the aggregation window for a timestamp
        """
        if level == AggregationLevel.MINUTE:
            return timestamp.replace(second=0, microsecond=0)
        elif level == AggregationLevel.HOUR:
            return timestamp.replace(minute=0, second=0, microsecond=0)
        elif level == AggregationLevel.DAY:
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        elif level == AggregationLevel.WEEK:
            days_since_monday = timestamp.weekday()
            week_start = timestamp - timedelta(days=days_since_monday)
            return week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        elif level == AggregationLevel.MONTH:
            return timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            return timestamp


# Global instance
analytics_pipeline = AnalyticsPipeline()