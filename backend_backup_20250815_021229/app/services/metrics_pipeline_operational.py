"""
Operational Metrics Pipeline
Real-time metrics collection, aggregation, and monitoring
"""
import os
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import json
from enum import Enum
import pandas as pd
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, push_to_gateway
from prometheus_client.exposition import start_http_server
import aioredis
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import statsd

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AggregationType(Enum):
    """Aggregation types for metrics"""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    P50 = "p50"
    P95 = "p95"
    P99 = "p99"

@dataclass
class MetricsPipelineConfig:
    """Configuration for metrics pipeline"""
    # Prometheus config
    prometheus_port: int = 8080
    prometheus_pushgateway: str = "localhost:9091"
    
    # InfluxDB config
    influxdb_url: str = "http://localhost:8086"
    influxdb_token: str = os.getenv("INFLUXDB_TOKEN", "ytempire-token")
    influxdb_org: str = "ytempire"
    influxdb_bucket: str = "metrics"
    
    # Kafka config
    kafka_brokers: List[str] = ["localhost:9092"]
    kafka_topic: str = "ytempire-metrics"
    
    # StatsD config
    statsd_host: str = "localhost"
    statsd_port: int = 8125
    statsd_prefix: str = "ytempire"
    
    # Redis config for buffering
    redis_url: str = "redis://localhost:6379/3"
    
    # Pipeline settings
    batch_size: int = 100
    flush_interval_seconds: int = 10
    retention_days: int = 90
    enable_sampling: bool = True
    sampling_rate: float = 0.1

class MetricsPipeline:
    """Main metrics pipeline for operational monitoring"""
    
    def __init__(self, config: MetricsPipelineConfig = None):
        self.config = config or MetricsPipelineConfig()
        self.registry = CollectorRegistry()
        self._init_collectors()
        self._init_backends()
        self._init_aggregators()
        asyncio.create_task(self._start_pipeline())
        
    def _init_collectors(self):
        """Initialize metric collectors"""
        # Business metrics
        self.video_generated = Counter(
            'videos_generated_total',
            'Total videos generated',
            ['channel_id', 'status'],
            registry=self.registry
        )
        
        self.generation_duration = Histogram(
            'video_generation_duration_seconds',
            'Video generation duration',
            ['channel_id'],
            buckets=[30, 60, 120, 300, 600, 1800, 3600],
            registry=self.registry
        )
        
        self.api_cost = Counter(
            'api_cost_dollars',
            'API costs in dollars',
            ['service', 'operation'],
            registry=self.registry
        )
        
        self.revenue = Counter(
            'revenue_dollars_total',
            'Total revenue in dollars',
            ['channel_id', 'source'],
            registry=self.registry
        )
        
        self.active_channels = Gauge(
            'active_channels',
            'Number of active channels',
            registry=self.registry
        )
        
        self.queue_size = Gauge(
            'video_queue_size',
            'Current video queue size',
            ['status'],
            registry=self.registry
        )
        
        # System metrics
        self.api_requests = Counter(
            'api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.api_latency = Histogram(
            'api_latency_seconds',
            'API request latency',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.db_queries = Counter(
            'database_queries_total',
            'Total database queries',
            ['operation', 'table'],
            registry=self.registry
        )
        
        self.db_latency = Histogram(
            'database_latency_seconds',
            'Database query latency',
            ['operation'],
            registry=self.registry
        )
        
        self.cache_hits = Counter(
            'cache_hits_total',
            'Cache hits',
            ['cache_name'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'cache_misses_total',
            'Cache misses',
            ['cache_name'],
            registry=self.registry
        )
        
        # ML metrics
        self.model_predictions = Counter(
            'model_predictions_total',
            'Total model predictions',
            ['model_name', 'version'],
            registry=self.registry
        )
        
        self.model_latency = Histogram(
            'model_inference_latency_seconds',
            'Model inference latency',
            ['model_name'],
            registry=self.registry
        )
        
        self.model_accuracy = Gauge(
            'model_accuracy',
            'Model accuracy score',
            ['model_name', 'version'],
            registry=self.registry
        )
        
    def _init_backends(self):
        """Initialize metric storage backends"""
        # Start Prometheus HTTP server
        start_http_server(self.config.prometheus_port, registry=self.registry)
        logger.info(f"Prometheus metrics server started on port {self.config.prometheus_port}")
        
        # Initialize InfluxDB client
        self.influxdb_client = InfluxDBClient(
            url=self.config.influxdb_url,
            token=self.config.influxdb_token,
            org=self.config.influxdb_org
        )
        self.influxdb_write_api = self.influxdb_client.write_api(write_options=SYNCHRONOUS)
        
        # Initialize Kafka producer
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=self.config.kafka_brokers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            compression_type='gzip'
        )
        
        # Initialize StatsD client
        self.statsd_client = statsd.StatsClient(
            self.config.statsd_host,
            self.config.statsd_port,
            prefix=self.config.statsd_prefix
        )
        
    async def _init_redis_buffer(self):
        """Initialize Redis for metric buffering"""
        self.redis_client = await aioredis.from_url(self.config.redis_url)
        
    def _init_aggregators(self):
        """Initialize metric aggregators"""
        self.aggregators = {
            "1m": MetricAggregator(window_seconds=60),
            "5m": MetricAggregator(window_seconds=300),
            "1h": MetricAggregator(window_seconds=3600),
            "1d": MetricAggregator(window_seconds=86400)
        }
        
    async def _start_pipeline(self):
        """Start the metrics pipeline"""
        await self._init_redis_buffer()
        
        # Start background tasks
        asyncio.create_task(self._flush_metrics_task())
        asyncio.create_task(self._aggregate_metrics_task())
        asyncio.create_task(self._cleanup_old_metrics_task())
        asyncio.create_task(self._monitor_pipeline_health())
        
        logger.info("Metrics pipeline started")
        
    async def track_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ):
        """Track a metric"""
        timestamp = timestamp or datetime.utcnow()
        tags = tags or {}
        
        # Create metric object
        metric = {
            "name": name,
            "value": value,
            "type": metric_type.value,
            "tags": tags,
            "timestamp": timestamp.isoformat()
        }
        
        # Buffer in Redis
        await self._buffer_metric(metric)
        
        # Send to real-time backends if sampling allows
        if self._should_sample():
            await self._send_to_backends(metric)
            
        # Update Prometheus metrics
        self._update_prometheus(name, value, metric_type, tags)
        
    async def _buffer_metric(self, metric: Dict):
        """Buffer metric in Redis"""
        key = f"metrics:buffer:{datetime.utcnow().strftime('%Y%m%d%H%M')}"
        await self.redis_client.rpush(key, json.dumps(metric))
        await self.redis_client.expire(key, 3600)  # Expire after 1 hour
        
    def _should_sample(self) -> bool:
        """Determine if metric should be sampled"""
        if not self.config.enable_sampling:
            return True
        return np.random.random() < self.config.sampling_rate
        
    async def _send_to_backends(self, metric: Dict):
        """Send metric to various backends"""
        # Send to InfluxDB
        await self._send_to_influxdb(metric)
        
        # Send to Kafka
        self._send_to_kafka(metric)
        
        # Send to StatsD
        self._send_to_statsd(metric)
        
    async def _send_to_influxdb(self, metric: Dict):
        """Send metric to InfluxDB"""
        try:
            point = Point(metric["name"]) \
                .time(datetime.fromisoformat(metric["timestamp"])) \
                .field("value", metric["value"])
                
            for tag_key, tag_value in metric.get("tags", {}).items():
                point.tag(tag_key, tag_value)
                
            self.influxdb_write_api.write(
                bucket=self.config.influxdb_bucket,
                record=point
            )
        except Exception as e:
            logger.error(f"Error sending to InfluxDB: {e}")
            
    def _send_to_kafka(self, metric: Dict):
        """Send metric to Kafka"""
        try:
            self.kafka_producer.send(
                self.config.kafka_topic,
                value=metric
            )
        except KafkaError as e:
            logger.error(f"Error sending to Kafka: {e}")
            
    def _send_to_statsd(self, metric: Dict):
        """Send metric to StatsD"""
        try:
            metric_type = MetricType(metric["type"])
            name = metric["name"]
            value = metric["value"]
            
            if metric_type == MetricType.COUNTER:
                self.statsd_client.incr(name, value)
            elif metric_type == MetricType.GAUGE:
                self.statsd_client.gauge(name, value)
            elif metric_type == MetricType.HISTOGRAM:
                self.statsd_client.timing(name, value * 1000)  # Convert to ms
        except Exception as e:
            logger.error(f"Error sending to StatsD: {e}")
            
    def _update_prometheus(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        tags: Dict[str, str]
    ):
        """Update Prometheus metrics"""
        # Map to appropriate Prometheus collector
        if name == "videos_generated":
            self.video_generated.labels(**tags).inc(value)
        elif name == "generation_duration":
            self.generation_duration.labels(**tags).observe(value)
        elif name == "api_cost":
            self.api_cost.labels(**tags).inc(value)
        elif name == "revenue":
            self.revenue.labels(**tags).inc(value)
        # Add more mappings as needed
        
    async def _flush_metrics_task(self):
        """Periodically flush buffered metrics"""
        while True:
            await asyncio.sleep(self.config.flush_interval_seconds)
            await self._flush_metrics()
            
    async def _flush_metrics(self):
        """Flush buffered metrics to backends"""
        try:
            # Get all buffer keys
            pattern = "metrics:buffer:*"
            keys = await self.redis_client.keys(pattern)
            
            for key in keys:
                # Get all metrics from buffer
                metrics = await self.redis_client.lrange(key, 0, -1)
                
                if metrics:
                    batch = []
                    for metric_json in metrics:
                        metric = json.loads(metric_json)
                        batch.append(metric)
                        
                        if len(batch) >= self.config.batch_size:
                            await self._send_batch_to_backends(batch)
                            batch = []
                            
                    # Send remaining metrics
                    if batch:
                        await self._send_batch_to_backends(batch)
                        
                    # Clear buffer
                    await self.redis_client.delete(key)
                    
        except Exception as e:
            logger.error(f"Error flushing metrics: {e}")
            
    async def _send_batch_to_backends(self, metrics: List[Dict]):
        """Send batch of metrics to backends"""
        # Send to InfluxDB in batch
        points = []
        for metric in metrics:
            point = Point(metric["name"]) \
                .time(datetime.fromisoformat(metric["timestamp"])) \
                .field("value", metric["value"])
                
            for tag_key, tag_value in metric.get("tags", {}).items():
                point.tag(tag_key, tag_value)
                
            points.append(point)
            
        if points:
            self.influxdb_write_api.write(
                bucket=self.config.influxdb_bucket,
                record=points
            )
            
    async def _aggregate_metrics_task(self):
        """Periodically aggregate metrics"""
        while True:
            await asyncio.sleep(60)  # Run every minute
            await self._aggregate_metrics()
            
    async def _aggregate_metrics(self):
        """Aggregate metrics for different time windows"""
        try:
            # Query recent metrics from InfluxDB
            query = f'''
                from(bucket: "{self.config.influxdb_bucket}")
                |> range(start: -5m)
                |> filter(fn: (r) => r._measurement != "aggregated")
            '''
            
            result = self.influxdb_client.query_api().query(query)
            
            for window, aggregator in self.aggregators.items():
                aggregated = aggregator.aggregate(result)
                
                # Store aggregated metrics
                for agg_metric in aggregated:
                    await self.track_metric(
                        name=f"aggregated.{window}.{agg_metric['name']}",
                        value=agg_metric["value"],
                        metric_type=MetricType.GAUGE,
                        tags=agg_metric.get("tags", {})
                    )
                    
        except Exception as e:
            logger.error(f"Error aggregating metrics: {e}")
            
    async def _cleanup_old_metrics_task(self):
        """Periodically clean up old metrics"""
        while True:
            await asyncio.sleep(86400)  # Run daily
            await self._cleanup_old_metrics()
            
    async def _cleanup_old_metrics(self):
        """Clean up metrics older than retention period"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.config.retention_days)
            
            # Delete from InfluxDB
            delete_api = self.influxdb_client.delete_api()
            delete_api.delete(
                start=datetime(2020, 1, 1),
                stop=cutoff_date,
                predicate='',
                bucket=self.config.influxdb_bucket
            )
            
            logger.info(f"Cleaned up metrics older than {cutoff_date}")
            
        except Exception as e:
            logger.error(f"Error cleaning up metrics: {e}")
            
    async def _monitor_pipeline_health(self):
        """Monitor the health of the metrics pipeline"""
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            health_metrics = {
                "buffer_size": await self._get_buffer_size(),
                "backends_healthy": await self._check_backends_health(),
                "pipeline_lag": await self._calculate_pipeline_lag()
            }
            
            # Track pipeline health metrics
            for metric_name, value in health_metrics.items():
                await self.track_metric(
                    name=f"pipeline.health.{metric_name}",
                    value=value,
                    metric_type=MetricType.GAUGE
                )
                
    async def _get_buffer_size(self) -> int:
        """Get current buffer size"""
        keys = await self.redis_client.keys("metrics:buffer:*")
        total_size = 0
        
        for key in keys:
            size = await self.redis_client.llen(key)
            total_size += size
            
        return total_size
        
    async def _check_backends_health(self) -> int:
        """Check health of metric backends"""
        healthy_count = 0
        
        # Check InfluxDB
        try:
            self.influxdb_client.ping()
            healthy_count += 1
        except:
            pass
            
        # Check Kafka
        try:
            metadata = self.kafka_producer.partitions_for(self.config.kafka_topic)
            if metadata:
                healthy_count += 1
        except:
            pass
            
        # Check Redis
        try:
            await self.redis_client.ping()
            healthy_count += 1
        except:
            pass
            
        return healthy_count
        
    async def _calculate_pipeline_lag(self) -> float:
        """Calculate pipeline processing lag"""
        # Get oldest buffered metric
        keys = await self.redis_client.keys("metrics:buffer:*")
        
        if not keys:
            return 0.0
            
        oldest_timestamp = None
        for key in keys:
            first_metric = await self.redis_client.lindex(key, 0)
            if first_metric:
                metric = json.loads(first_metric)
                timestamp = datetime.fromisoformat(metric["timestamp"])
                
                if oldest_timestamp is None or timestamp < oldest_timestamp:
                    oldest_timestamp = timestamp
                    
        if oldest_timestamp:
            lag = (datetime.utcnow() - oldest_timestamp).total_seconds()
            return lag
            
        return 0.0
        
    async def query_metrics(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
        aggregation: Optional[AggregationType] = None
    ) -> pd.DataFrame:
        """Query metrics from storage"""
        end_time = end_time or datetime.utcnow()
        
        # Build InfluxDB query
        query = f'''
            from(bucket: "{self.config.influxdb_bucket}")
            |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
            |> filter(fn: (r) => r._measurement == "{metric_name}")
        '''
        
        # Add tag filters
        if tags:
            for tag_key, tag_value in tags.items():
                query += f'''
                    |> filter(fn: (r) => r.{tag_key} == "{tag_value}")
                '''
                
        # Add aggregation
        if aggregation:
            if aggregation == AggregationType.AVG:
                query += "|> mean()"
            elif aggregation == AggregationType.SUM:
                query += "|> sum()"
            elif aggregation == AggregationType.MAX:
                query += "|> max()"
            elif aggregation == AggregationType.MIN:
                query += "|> min()"
            elif aggregation == AggregationType.COUNT:
                query += "|> count()"
                
        # Execute query
        result = self.influxdb_client.query_api().query_data_frame(query)
        return result
        
    async def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get metrics for dashboard display"""
        now = datetime.utcnow()
        day_ago = now - timedelta(days=1)
        week_ago = now - timedelta(days=7)
        
        return {
            "videos_generated_24h": await self._get_metric_sum("videos_generated", day_ago, now),
            "total_revenue_24h": await self._get_metric_sum("revenue", day_ago, now),
            "total_cost_24h": await self._get_metric_sum("api_cost", day_ago, now),
            "avg_generation_time": await self._get_metric_avg("generation_duration", day_ago, now),
            "active_channels": await self._get_metric_current("active_channels"),
            "queue_size": await self._get_metric_current("queue_size"),
            "api_success_rate": await self._calculate_success_rate(day_ago, now),
            "weekly_trends": await self._get_weekly_trends(week_ago, now)
        }
        
    async def _get_metric_sum(
        self,
        metric_name: str,
        start: datetime,
        end: datetime
    ) -> float:
        """Get sum of metric values"""
        df = await self.query_metrics(metric_name, start, end, aggregation=AggregationType.SUM)
        return df["_value"].sum() if not df.empty else 0.0
        
    async def _get_metric_avg(
        self,
        metric_name: str,
        start: datetime,
        end: datetime
    ) -> float:
        """Get average of metric values"""
        df = await self.query_metrics(metric_name, start, end, aggregation=AggregationType.AVG)
        return df["_value"].mean() if not df.empty else 0.0
        
    async def _get_metric_current(self, metric_name: str) -> float:
        """Get current value of a gauge metric"""
        # Query last value
        start = datetime.utcnow() - timedelta(minutes=5)
        df = await self.query_metrics(metric_name, start)
        return df["_value"].iloc[-1] if not df.empty else 0.0
        
    async def _calculate_success_rate(
        self,
        start: datetime,
        end: datetime
    ) -> float:
        """Calculate API success rate"""
        # Query successful requests
        success_df = await self.query_metrics(
            "api_requests",
            start,
            end,
            tags={"status_code": "200"}
        )
        
        # Query all requests
        total_df = await self.query_metrics("api_requests", start, end)
        
        if not total_df.empty:
            success_count = len(success_df)
            total_count = len(total_df)
            return (success_count / total_count) * 100 if total_count > 0 else 0.0
            
        return 100.0
        
    async def _get_weekly_trends(
        self,
        start: datetime,
        end: datetime
    ) -> Dict[str, List]:
        """Get weekly trend data"""
        # Query daily aggregates
        trends = {}
        
        for metric in ["videos_generated", "revenue", "api_cost"]:
            df = await self.query_metrics(metric, start, end)
            if not df.empty:
                daily = df.resample("D", on="_time")["_value"].sum()
                trends[metric] = daily.tolist()
            else:
                trends[metric] = []
                
        return trends

class MetricAggregator:
    """Aggregates metrics over time windows"""
    
    def __init__(self, window_seconds: int):
        self.window_seconds = window_seconds
        self.buffer = {}
        
    def aggregate(self, metrics: List[Dict]) -> List[Dict]:
        """Aggregate metrics"""
        aggregated = []
        
        # Group by metric name and tags
        grouped = {}
        for metric in metrics:
            key = (metric["name"], json.dumps(metric.get("tags", {})))
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(metric["value"])
            
        # Calculate aggregations
        for (name, tags_json), values in grouped.items():
            tags = json.loads(tags_json)
            
            aggregated.extend([
                {"name": f"{name}.avg", "value": np.mean(values), "tags": tags},
                {"name": f"{name}.min", "value": np.min(values), "tags": tags},
                {"name": f"{name}.max", "value": np.max(values), "tags": tags},
                {"name": f"{name}.p50", "value": np.percentile(values, 50), "tags": tags},
                {"name": f"{name}.p95", "value": np.percentile(values, 95), "tags": tags},
                {"name": f"{name}.p99", "value": np.percentile(values, 99), "tags": tags}
            ])
            
        return aggregated