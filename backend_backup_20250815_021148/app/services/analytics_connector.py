"""
Analytics Pipeline Connector
Bridges ML pipeline analytics with backend analytics pipeline for seamless data flow
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import redis.asyncio as redis
from dataclasses import asdict

from app.services.analytics_pipeline import (
    AnalyticsPipeline, 
    MetricPoint, 
    MetricType, 
    DataSource,
    AggregationLevel
)
from app.core.config import settings

logger = logging.getLogger(__name__)


class AnalyticsConnector:
    """Connects ML pipeline events to backend analytics pipeline"""
    
    def __init__(self):
        self.backend_pipeline: Optional[AnalyticsPipeline] = None
        self.ml_redis_client: Optional[redis.Redis] = None
        self.backend_redis_client: Optional[redis.Redis] = None
        self.stream_key = "ml:analytics:events"
        self.consumer_group = "backend_consumer"
        self.consumer_name = "analytics_connector"
        self.batch_size = 100
        self.processing_interval = 5  # seconds
        
    async def initialize(self):
        """Initialize pipeline connections"""
        try:
            # Initialize backend analytics pipeline
            self.backend_pipeline = AnalyticsPipeline()
            await self.backend_pipeline.initialize()
            
            # Connect to ML pipeline Redis (could be different instance)
            self.ml_redis_client = await redis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
            
            # Create consumer group for ML events
            try:
                await self.ml_redis_client.xgroup_create(
                    self.stream_key,
                    self.consumer_group,
                    id="0"
                )
            except redis.ResponseError:
                # Group already exists
                pass
                
            logger.info("Analytics connector initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize analytics connector: {e}")
            raise
            
    async def start_processing(self):
        """Start processing ML pipeline events"""
        logger.info("Starting analytics event processing from ML pipeline")
        
        while True:
            try:
                # Read events from ML pipeline stream
                events = await self.ml_redis_client.xreadgroup(
                    self.consumer_group,
                    self.consumer_name,
                    {self.stream_key: ">"},
                    count=self.batch_size,
                    block=self.processing_interval * 1000
                )
                
                if events:
                    await self.process_ml_events(events)
                    
            except Exception as e:
                logger.error(f"Error processing ML events: {e}")
                await asyncio.sleep(self.processing_interval)
                
    async def process_ml_events(self, events: List):
        """Process events from ML pipeline"""
        for stream_name, stream_events in events:
            for event_id, event_data in stream_events:
                try:
                    await self.transform_and_ingest(event_data)
                    
                    # Acknowledge processed event
                    await self.ml_redis_client.xack(
                        self.stream_key,
                        self.consumer_group,
                        event_id
                    )
                except Exception as e:
                    logger.error(f"Failed to process event {event_id}: {e}")
                    
    async def transform_and_ingest(self, event_data: Dict[str, Any]):
        """Transform ML event to backend metric format"""
        event_type = event_data.get("event_type")
        
        # Map ML event types to backend metric types
        metric_mapping = {
            "video_view": MetricType.VIEWS,
            "quality_score": MetricType.ENGAGEMENT_RATE,
            "cost_incurred": MetricType.COST,
            "revenue_earned": MetricType.REVENUE,
            "model_prediction": MetricType.CTR,
            "video_publish": MetricType.IMPRESSIONS
        }
        
        metric_type = metric_mapping.get(event_type)
        if not metric_type:
            logger.warning(f"Unknown event type: {event_type}")
            return
            
        # Create metric point
        metric = MetricPoint(
            timestamp=datetime.fromisoformat(event_data.get("timestamp", datetime.utcnow().isoformat())),
            metric_type=metric_type,
            value=float(event_data.get("value", 0)),
            source=DataSource.INTERNAL,
            channel_id=event_data.get("channel_id"),
            video_id=event_data.get("video_id"),
            user_id=event_data.get("user_id"),
            metadata=event_data.get("metadata", {})
        )
        
        # Ingest into backend pipeline
        await self.backend_pipeline.ingest_metric(metric)
        
        # Check for real-time alerts
        await self.check_alerts(metric)
        
    async def check_alerts(self, metric: MetricPoint):
        """Check if metric triggers any alerts"""
        alert_rules = {
            MetricType.COST: {"threshold": 5.0, "direction": "above"},
            MetricType.ENGAGEMENT_RATE: {"threshold": 0.02, "direction": "below"},
            MetricType.CTR: {"threshold": 0.01, "direction": "below"},
            MetricType.REVENUE: {"threshold": 10.0, "direction": "below"}
        }
        
        rule = alert_rules.get(metric.metric_type)
        if rule:
            threshold = rule["threshold"]
            direction = rule["direction"]
            
            if (direction == "above" and metric.value > threshold) or \
               (direction == "below" and metric.value < threshold):
                await self.trigger_alert(metric, rule)
                
    async def trigger_alert(self, metric: MetricPoint, rule: Dict):
        """Trigger alert for threshold breach"""
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "metric_type": metric.metric_type.value,
            "value": metric.value,
            "threshold": rule["threshold"],
            "direction": rule["direction"],
            "channel_id": metric.channel_id,
            "video_id": metric.video_id,
            "severity": "high" if abs(metric.value - rule["threshold"]) > rule["threshold"] * 0.5 else "medium"
        }
        
        # Publish alert to Redis
        await self.backend_pipeline.redis_client.publish(
            "analytics:alerts",
            json.dumps(alert)
        )
        
        logger.warning(f"Alert triggered: {alert}")
        
    async def sync_aggregations(self):
        """Sync aggregated metrics between pipelines"""
        aggregation_levels = [
            AggregationLevel.HOUR,
            AggregationLevel.DAY,
            AggregationLevel.WEEK
        ]
        
        for level in aggregation_levels:
            # Get ML pipeline aggregations
            ml_aggregations = await self.get_ml_aggregations(level)
            
            # Merge with backend aggregations
            for aggregation in ml_aggregations:
                await self.backend_pipeline.store_aggregated_metric(aggregation)
                
    async def get_ml_aggregations(self, level: AggregationLevel) -> List:
        """Get aggregations from ML pipeline"""
        pattern = f"ml:aggregated:*:{level.value}:*"
        aggregations = []
        
        cursor = 0
        while True:
            cursor, keys = await self.ml_redis_client.scan(
                cursor=cursor,
                match=pattern,
                count=100
            )
            
            for key in keys:
                data = await self.ml_redis_client.get(key)
                if data:
                    aggregations.append(json.loads(data))
                    
            if cursor == 0:
                break
                
        return aggregations
        
    async def get_pipeline_health(self) -> Dict[str, Any]:
        """Get health status of pipeline connections"""
        health = {
            "ml_pipeline_connected": False,
            "backend_pipeline_connected": False,
            "events_processed_today": 0,
            "last_event_time": None,
            "pending_events": 0,
            "consumer_lag": 0
        }
        
        try:
            # Check ML pipeline connection
            await self.ml_redis_client.ping()
            health["ml_pipeline_connected"] = True
            
            # Check backend pipeline connection
            await self.backend_pipeline.redis_client.ping()
            health["backend_pipeline_connected"] = True
            
            # Get stream info
            stream_info = await self.ml_redis_client.xinfo_stream(self.stream_key)
            health["pending_events"] = stream_info.get("length", 0)
            
            # Get consumer group info
            groups = await self.ml_redis_client.xinfo_groups(self.stream_key)
            for group in groups:
                if group["name"] == self.consumer_group:
                    health["consumer_lag"] = group.get("lag", 0)
                    
            # Get today's event count
            today_key = f"events:processed:{datetime.utcnow().date()}"
            count = await self.backend_pipeline.redis_client.get(today_key)
            health["events_processed_today"] = int(count) if count else 0
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            
        return health


# Singleton instance
analytics_connector = AnalyticsConnector()


async def start_connector():
    """Start the analytics connector service"""
    await analytics_connector.initialize()
    await analytics_connector.start_processing()