"""
Real-time Metrics Pipeline for YTEmpire
Handles data collection, processing, and streaming
"""
import asyncio
import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
from enum import Enum
import redis.asyncio as redis
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from prometheus_client import Counter, Histogram, Gauge, Summary
import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.models.video import Video
from app.models.channel import Channel
from app.models.analytics import Analytics
from app.models.cost import Cost

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricEvent:
    """Container for metric events"""
    timestamp: datetime
    metric_name: str
    metric_type: MetricType
    value: float
    labels: Dict[str, str]
    metadata: Dict[str, Any]


@dataclass
class AggregatedMetric:
    """Container for aggregated metrics"""
    metric_name: str
    period: str  # '1m', '5m', '1h', '1d'
    timestamp: datetime
    count: int
    sum: float
    avg: float
    min: float
    max: float
    p50: float
    p95: float
    p99: float
    labels: Dict[str, str]


class MetricsPipeline:
    """
    Real-time metrics collection and processing pipeline
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        kafka_bootstrap_servers: Optional[str] = None,
        db_session: Optional[AsyncSession] = None
    ):
        """
        Initialize metrics pipeline
        
        Args:
            redis_url: Redis connection URL
            kafka_bootstrap_servers: Kafka servers for streaming
            db_session: Database session for persistence
        """
        self.redis_url = redis_url
        self.kafka_servers = kafka_bootstrap_servers
        self.db_session = db_session
        
        # Metric stores
        self.metrics_buffer = []
        self.aggregation_windows = {
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '1h': timedelta(hours=1),
            '1d': timedelta(days=1)
        }
        
        # Prometheus metrics
        self._init_prometheus_metrics()
        
        # Processing tasks
        self.tasks = []
        
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        # Video metrics
        self.video_generated = Counter(
            'ytempire_videos_generated_total',
            'Total number of videos generated',
            ['channel_id', 'status']
        )
        
        self.video_duration = Histogram(
            'ytempire_video_duration_seconds',
            'Video duration in seconds',
            ['channel_id'],
            buckets=[30, 60, 120, 300, 600, 900, 1200, 1800]
        )
        
        self.video_views = Gauge(
            'ytempire_video_views',
            'Current video view count',
            ['video_id', 'channel_id']
        )
        
        # Cost metrics
        self.generation_cost = Summary(
            'ytempire_generation_cost_dollars',
            'Cost of video generation',
            ['service', 'operation']
        )
        
        # Performance metrics
        self.api_latency = Histogram(
            'ytempire_api_latency_seconds',
            'API endpoint latency',
            ['endpoint', 'method', 'status'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        )
        
        self.pipeline_duration = Histogram(
            'ytempire_pipeline_duration_seconds',
            'Video pipeline execution time',
            ['stage'],
            buckets=[1, 5, 10, 30, 60, 120, 300, 600]
        )
        
        # Business metrics
        self.revenue = Gauge(
            'ytempire_revenue_dollars',
            'Revenue generated',
            ['source', 'channel_id']
        )
        
        self.active_users = Gauge(
            'ytempire_active_users',
            'Number of active users',
            ['tier']
        )
    
    async def start(self):
        """Start the metrics pipeline"""
        # Initialize Redis connection
        self.redis_client = await redis.from_url(self.redis_url)
        
        # Initialize Kafka if configured
        if self.kafka_servers:
            await self._init_kafka()
        
        # Start processing tasks
        self.tasks = [
            asyncio.create_task(self._process_metrics()),
            asyncio.create_task(self._aggregate_metrics()),
            asyncio.create_task(self._persist_metrics()),
            asyncio.create_task(self._monitor_health())
        ]
        
        logger.info("Metrics pipeline started")
    
    async def stop(self):
        """Stop the metrics pipeline"""
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Close connections
        await self.redis_client.close()
        
        if hasattr(self, 'kafka_producer'):
            await self.kafka_producer.stop()
        if hasattr(self, 'kafka_consumer'):
            await self.kafka_consumer.stop()
        
        logger.info("Metrics pipeline stopped")
    
    async def record_metric(
        self,
        metric_name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record a metric event
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            metric_type: Type of metric
            labels: Metric labels
            metadata: Additional metadata
        """
        event = MetricEvent(
            timestamp=datetime.utcnow(),
            metric_name=metric_name,
            metric_type=metric_type,
            value=value,
            labels=labels or {},
            metadata=metadata or {}
        )
        
        # Add to buffer
        self.metrics_buffer.append(event)
        
        # Send to Redis for real-time processing
        await self._send_to_redis(event)
        
        # Send to Kafka if configured
        if hasattr(self, 'kafka_producer'):
            await self._send_to_kafka(event)
        
        # Update Prometheus metrics
        self._update_prometheus(event)
    
    async def record_video_generation(
        self,
        video_id: str,
        channel_id: str,
        duration: float,
        cost: float,
        status: str
    ):
        """Record video generation metrics"""
        # Record multiple related metrics
        await self.record_metric(
            'video.generated',
            1,
            MetricType.COUNTER,
            {'channel_id': channel_id, 'status': status},
            {'video_id': video_id}
        )
        
        await self.record_metric(
            'video.duration',
            duration,
            MetricType.HISTOGRAM,
            {'channel_id': channel_id}
        )
        
        await self.record_metric(
            'video.cost',
            cost,
            MetricType.GAUGE,
            {'channel_id': channel_id},
            {'video_id': video_id}
        )
        
        # Update Prometheus
        self.video_generated.labels(channel_id=channel_id, status=status).inc()
        self.video_duration.labels(channel_id=channel_id).observe(duration)
        self.generation_cost.labels(service='total', operation='generate').observe(cost)
    
    async def record_api_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        response_time: float
    ):
        """Record API request metrics"""
        await self.record_metric(
            'api.request',
            1,
            MetricType.COUNTER,
            {
                'endpoint': endpoint,
                'method': method,
                'status': str(status_code)
            }
        )
        
        await self.record_metric(
            'api.latency',
            response_time,
            MetricType.HISTOGRAM,
            {
                'endpoint': endpoint,
                'method': method,
                'status': str(status_code)
            }
        )
        
        # Update Prometheus
        self.api_latency.labels(
            endpoint=endpoint,
            method=method,
            status=str(status_code)
        ).observe(response_time)
    
    async def get_real_time_metrics(
        self,
        metric_names: List[str],
        window: str = '5m'
    ) -> Dict[str, Any]:
        """
        Get real-time metrics
        
        Args:
            metric_names: List of metric names
            window: Time window
        
        Returns:
            Real-time metrics data
        """
        results = {}
        window_delta = self.aggregation_windows.get(window, timedelta(minutes=5))
        cutoff_time = datetime.utcnow() - window_delta
        
        for metric_name in metric_names:
            # Get from Redis
            key = f"metrics:{metric_name}:{window}"
            data = await self.redis_client.get(key)
            
            if data:
                results[metric_name] = json.loads(data)
            else:
                # Calculate from buffer
                relevant_events = [
                    e for e in self.metrics_buffer
                    if e.metric_name == metric_name and e.timestamp >= cutoff_time
                ]
                
                if relevant_events:
                    values = [e.value for e in relevant_events]
                    results[metric_name] = {
                        'count': len(values),
                        'sum': sum(values),
                        'avg': np.mean(values),
                        'min': min(values),
                        'max': max(values),
                        'p50': np.percentile(values, 50),
                        'p95': np.percentile(values, 95),
                        'p99': np.percentile(values, 99)
                    }
                else:
                    results[metric_name] = None
        
        return results
    
    async def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get metrics for dashboard display"""
        metrics = {}
        
        # Get real-time metrics
        real_time = await self.get_real_time_metrics([
            'video.generated',
            'api.request',
            'video.cost',
            'revenue.total'
        ], window='1h')
        
        metrics['real_time'] = real_time
        
        # Get aggregated metrics from database
        if self.db_session:
            # Videos generated today
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0)
            videos_today = await self.db_session.execute(
                select(func.count(Video.id)).where(
                    Video.created_at >= today_start
                )
            )
            metrics['videos_today'] = videos_today.scalar()
            
            # Total revenue today
            revenue_today = await self.db_session.execute(
                select(func.sum(Video.estimated_revenue)).where(
                    Video.created_at >= today_start
                )
            )
            metrics['revenue_today'] = revenue_today.scalar() or 0
            
            # Total cost today
            cost_today = await self.db_session.execute(
                select(func.sum(Cost.amount)).where(
                    Cost.created_at >= today_start
                )
            )
            metrics['cost_today'] = cost_today.scalar() or 0
            
            # Active channels
            active_channels = await self.db_session.execute(
                select(func.count(Channel.id)).where(
                    Channel.is_active == True
                )
            )
            metrics['active_channels'] = active_channels.scalar()
        
        # Calculate derived metrics
        if metrics.get('revenue_today') and metrics.get('cost_today'):
            metrics['profit_today'] = metrics['revenue_today'] - metrics['cost_today']
            metrics['roi_today'] = (
                (metrics['profit_today'] / metrics['cost_today']) * 100
                if metrics['cost_today'] > 0 else 0
            )
        
        return metrics
    
    async def create_metrics_stream(
        self,
        metric_names: List[str],
        callback: Callable[[Dict[str, Any]], None]
    ):
        """
        Create a real-time metrics stream
        
        Args:
            metric_names: Metrics to stream
            callback: Callback for metric updates
        """
        # Subscribe to Redis pub/sub
        pubsub = self.redis_client.pubsub()
        
        for metric_name in metric_names:
            await pubsub.subscribe(f"metrics:stream:{metric_name}")
        
        # Process messages
        async def process_stream():
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    data = json.loads(message['data'])
                    callback(data)
        
        # Start streaming task
        asyncio.create_task(process_stream())
    
    async def _process_metrics(self):
        """Process metrics from buffer"""
        while True:
            try:
                if self.metrics_buffer:
                    # Process batch
                    batch = self.metrics_buffer[:100]
                    self.metrics_buffer = self.metrics_buffer[100:]
                    
                    for event in batch:
                        # Process each event
                        await self._process_event(event)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing metrics: {e}")
                await asyncio.sleep(5)
    
    async def _aggregate_metrics(self):
        """Aggregate metrics at different time windows"""
        while True:
            try:
                current_time = datetime.utcnow()
                
                for window_name, window_delta in self.aggregation_windows.items():
                    # Get metrics for window
                    cutoff_time = current_time - window_delta
                    
                    # Aggregate from Redis
                    pattern = f"metrics:raw:*"
                    cursor = 0
                    
                    while True:
                        cursor, keys = await self.redis_client.scan(
                            cursor,
                            match=pattern,
                            count=100
                        )
                        
                        for key in keys:
                            data = await self.redis_client.get(key)
                            if data:
                                event = json.loads(data)
                                event_time = datetime.fromisoformat(event['timestamp'])
                                
                                if event_time >= cutoff_time:
                                    # Aggregate this metric
                                    await self._aggregate_event(event, window_name)
                        
                        if cursor == 0:
                            break
                
                await asyncio.sleep(60)  # Aggregate every minute
                
            except Exception as e:
                logger.error(f"Error aggregating metrics: {e}")
                await asyncio.sleep(60)
    
    async def _persist_metrics(self):
        """Persist metrics to database"""
        while True:
            try:
                if self.db_session:
                    # Get aggregated metrics from Redis
                    pattern = f"metrics:aggregated:*"
                    cursor = 0
                    
                    while True:
                        cursor, keys = await self.redis_client.scan(
                            cursor,
                            match=pattern,
                            count=100
                        )
                        
                        for key in keys:
                            data = await self.redis_client.get(key)
                            if data:
                                metric = json.loads(data)
                                
                                # Store in database
                                analytics = Analytics(
                                    metric_name=metric['metric_name'],
                                    metric_value=metric['value'],
                                    timestamp=datetime.fromisoformat(metric['timestamp']),
                                    metadata=json.dumps(metric.get('metadata', {}))
                                )
                                self.db_session.add(analytics)
                        
                        if cursor == 0:
                            break
                    
                    await self.db_session.commit()
                
                await asyncio.sleep(300)  # Persist every 5 minutes
                
            except Exception as e:
                logger.error(f"Error persisting metrics: {e}")
                if self.db_session:
                    await self.db_session.rollback()
                await asyncio.sleep(300)
    
    async def _monitor_health(self):
        """Monitor pipeline health"""
        while True:
            try:
                # Check buffer size
                buffer_size = len(self.metrics_buffer)
                if buffer_size > 10000:
                    logger.warning(f"Metrics buffer size high: {buffer_size}")
                
                # Check Redis connection
                await self.redis_client.ping()
                
                # Check Kafka if configured
                if hasattr(self, 'kafka_producer'):
                    # Kafka health check
                    pass
                
                # Record health metrics
                await self.record_metric(
                    'pipeline.health',
                    1,
                    MetricType.GAUGE,
                    {'component': 'metrics_pipeline'},
                    {'buffer_size': buffer_size}
                )
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                await asyncio.sleep(30)
    
    async def _send_to_redis(self, event: MetricEvent):
        """Send metric event to Redis"""
        # Store raw event
        key = f"metrics:raw:{event.metric_name}:{event.timestamp.timestamp()}"
        await self.redis_client.setex(
            key,
            3600,  # Expire after 1 hour
            json.dumps(asdict(event), default=str)
        )
        
        # Publish to stream
        channel = f"metrics:stream:{event.metric_name}"
        await self.redis_client.publish(
            channel,
            json.dumps({'metric': event.metric_name, 'value': event.value}, default=str)
        )
    
    async def _init_kafka(self):
        """Initialize Kafka producer and consumer"""
        self.kafka_producer = AIOKafkaProducer(
            bootstrap_servers=self.kafka_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode()
        )
        await self.kafka_producer.start()
        
        self.kafka_consumer = AIOKafkaConsumer(
            'ytempire.metrics',
            bootstrap_servers=self.kafka_servers,
            value_deserializer=lambda v: json.loads(v.decode())
        )
        await self.kafka_consumer.start()
    
    async def _send_to_kafka(self, event: MetricEvent):
        """Send metric event to Kafka"""
        await self.kafka_producer.send(
            'ytempire.metrics',
            value=asdict(event)
        )
    
    async def _process_event(self, event: MetricEvent):
        """Process individual metric event"""
        # Apply processing rules based on metric type
        if event.metric_type == MetricType.COUNTER:
            # Increment counter
            key = f"counter:{event.metric_name}"
            await self.redis_client.incr(key)
        
        elif event.metric_type == MetricType.GAUGE:
            # Set gauge value
            key = f"gauge:{event.metric_name}"
            await self.redis_client.set(key, event.value)
        
        elif event.metric_type in [MetricType.HISTOGRAM, MetricType.SUMMARY]:
            # Add to distribution
            key = f"dist:{event.metric_name}"
            await self.redis_client.lpush(key, event.value)
            await self.redis_client.ltrim(key, 0, 999)  # Keep last 1000 values
    
    async def _aggregate_event(self, event: Dict[str, Any], window: str):
        """Aggregate event for time window"""
        key = f"metrics:{event['metric_name']}:{window}"
        
        # Get existing aggregation
        existing = await self.redis_client.get(key)
        if existing:
            agg = json.loads(existing)
        else:
            agg = {
                'count': 0,
                'sum': 0,
                'values': []
            }
        
        # Update aggregation
        agg['count'] += 1
        agg['sum'] += event['value']
        agg['values'].append(event['value'])
        
        # Keep only recent values
        if len(agg['values']) > 1000:
            agg['values'] = agg['values'][-1000:]
        
        # Calculate statistics
        values = agg['values']
        agg['avg'] = np.mean(values)
        agg['min'] = min(values)
        agg['max'] = max(values)
        agg['p50'] = np.percentile(values, 50)
        agg['p95'] = np.percentile(values, 95)
        agg['p99'] = np.percentile(values, 99)
        
        # Store aggregation
        await self.redis_client.setex(
            key,
            self.aggregation_windows[window].total_seconds(),
            json.dumps(agg)
        )
    
    def _update_prometheus(self, event: MetricEvent):
        """Update Prometheus metrics"""
        # Map to appropriate Prometheus metric
        if event.metric_name == 'video.generated':
            self.video_generated.labels(**event.labels).inc()
        elif event.metric_name == 'video.views':
            self.video_views.labels(**event.labels).set(event.value)
        elif event.metric_name == 'revenue.total':
            self.revenue.labels(**event.labels).set(event.value)
        # Add more mappings as needed


# Example usage
async def main():
    """Example usage of metrics pipeline"""
    
    # Initialize pipeline
    pipeline = MetricsPipeline()
    await pipeline.start()
    
    # Record some metrics
    await pipeline.record_video_generation(
        video_id="video_123",
        channel_id="channel_456",
        duration=600,
        cost=1.10,
        status="completed"
    )
    
    await pipeline.record_api_request(
        endpoint="/api/v1/videos",
        method="POST",
        status_code=200,
        response_time=0.125
    )
    
    # Get real-time metrics
    metrics = await pipeline.get_real_time_metrics(
        ['video.generated', 'api.request'],
        window='5m'
    )
    print(f"Real-time metrics: {metrics}")
    
    # Get dashboard metrics
    dashboard = await pipeline.get_dashboard_metrics()
    print(f"Dashboard metrics: {dashboard}")
    
    # Create metrics stream
    def on_metric_update(data):
        print(f"Metric update: {data}")
    
    await pipeline.create_metrics_stream(
        ['video.generated'],
        on_metric_update
    )
    
    # Keep running for demo
    await asyncio.sleep(10)
    
    # Stop pipeline
    await pipeline.stop()


if __name__ == "__main__":
    asyncio.run(main())