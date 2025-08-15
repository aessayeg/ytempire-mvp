"""
Custom Metrics Collection for YTEmpire
Tracks business, performance, and system metrics
"""

import time
import asyncio
import psutil
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
from fastapi import Request, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import redis.asyncio as redis

logger = logging.getLogger(__name__)

# Create a custom registry for our metrics
REGISTRY = CollectorRegistry()

# ============================================================================
# System Metrics
# ============================================================================

# HTTP Request metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status'],
    registry=REGISTRY
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=REGISTRY
)

http_request_size_bytes = Summary(
    'http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint'],
    registry=REGISTRY
)

http_response_size_bytes = Summary(
    'http_response_size_bytes',
    'HTTP response size in bytes',
    ['method', 'endpoint'],
    registry=REGISTRY
)

# Database metrics
db_query_duration_seconds = Histogram(
    'db_query_duration_seconds',
    'Database query duration in seconds',
    ['query_type', 'table'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=REGISTRY
)

db_connections_active = Gauge(
    'db_connections_active',
    'Active database connections',
    registry=REGISTRY
)

db_connections_idle = Gauge(
    'db_connections_idle',
    'Idle database connections',
    registry=REGISTRY
)

db_query_errors_total = Counter(
    'db_query_errors_total',
    'Total database query errors',
    ['error_type'],
    registry=REGISTRY
)

# Cache metrics
cache_hits_total = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_type'],
    registry=REGISTRY
)

cache_misses_total = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_type'],
    registry=REGISTRY
)

cache_evictions_total = Counter(
    'cache_evictions_total',
    'Total cache evictions',
    ['cache_type'],
    registry=REGISTRY
)

cache_memory_bytes = Gauge(
    'cache_memory_bytes',
    'Cache memory usage in bytes',
    ['cache_type'],
    registry=REGISTRY
)

# ============================================================================
# Business Metrics
# ============================================================================

# Video generation metrics
videos_generated_total = Counter(
    'videos_generated_total',
    'Total videos generated',
    ['channel_id', 'status'],
    registry=REGISTRY
)

video_generation_duration_seconds = Histogram(
    'video_generation_duration_seconds',
    'Video generation duration in seconds',
    ['stage'],
    buckets=(30, 60, 120, 300, 600, 1200, 1800, 3600),
    registry=REGISTRY
)

video_generation_cost_dollars = Histogram(
    'video_generation_cost_dollars',
    'Video generation cost in dollars',
    ['service'],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0),
    registry=REGISTRY
)

video_quality_score = Histogram(
    'video_quality_score',
    'Video quality score (0-1)',
    ['channel_id'],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=REGISTRY
)

# YouTube metrics
youtube_quota_used = Gauge(
    'youtube_quota_used',
    'YouTube API quota used',
    ['account_id'],
    registry=REGISTRY
)

youtube_quota_limit = Gauge(
    'youtube_quota_limit',
    'YouTube API quota limit',
    ['account_id'],
    registry=REGISTRY
)

youtube_api_calls_total = Counter(
    'youtube_api_calls_total',
    'Total YouTube API calls',
    ['account_id', 'endpoint', 'status'],
    registry=REGISTRY
)

youtube_upload_duration_seconds = Histogram(
    'youtube_upload_duration_seconds',
    'YouTube upload duration in seconds',
    buckets=(10, 30, 60, 120, 300, 600),
    registry=REGISTRY
)

# Revenue metrics
revenue_total_dollars = Counter(
    'revenue_total_dollars',
    'Total revenue in dollars',
    ['source', 'channel_id'],
    registry=REGISTRY
)

subscription_active_count = Gauge(
    'subscription_active_count',
    'Active subscriptions',
    ['plan'],
    registry=REGISTRY
)

payment_transactions_total = Counter(
    'payment_transactions_total',
    'Total payment transactions',
    ['status', 'provider'],
    registry=REGISTRY
)

# Channel metrics
channel_health_score = Gauge(
    'channel_health_score',
    'Channel health score (0-1)',
    ['channel_id'],
    registry=REGISTRY
)

channel_subscriber_count = Gauge(
    'channel_subscriber_count',
    'Channel subscriber count',
    ['channel_id'],
    registry=REGISTRY
)

channel_view_count = Gauge(
    'channel_view_count',
    'Channel total views',
    ['channel_id'],
    registry=REGISTRY
)

# ============================================================================
# AI/ML Metrics
# ============================================================================

# Model inference metrics
ml_inference_duration_seconds = Histogram(
    'ml_inference_duration_seconds',
    'ML model inference duration',
    ['model', 'operation'],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY
)

ml_model_errors_total = Counter(
    'ml_model_errors_total',
    'Total ML model errors',
    ['model', 'error_type'],
    registry=REGISTRY
)

# GPU metrics
gpu_utilization_percent = Gauge(
    'gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id'],
    registry=REGISTRY
)

gpu_memory_used_bytes = Gauge(
    'gpu_memory_used_bytes',
    'GPU memory used in bytes',
    ['gpu_id'],
    registry=REGISTRY
)

gpu_memory_total_bytes = Gauge(
    'gpu_memory_total_bytes',
    'GPU memory total in bytes',
    ['gpu_id'],
    registry=REGISTRY
)

gpu_temperature_celsius = Gauge(
    'gpu_temperature_celsius',
    'GPU temperature in Celsius',
    ['gpu_id'],
    registry=REGISTRY
)

# ============================================================================
# Queue Metrics
# ============================================================================

queue_depth = Gauge(
    'queue_depth',
    'Queue depth',
    ['queue_name'],
    registry=REGISTRY
)

queue_processing_duration_seconds = Histogram(
    'queue_processing_duration_seconds',
    'Queue task processing duration',
    ['queue_name', 'task_type'],
    buckets=(1, 5, 10, 30, 60, 300, 600, 1800, 3600),
    registry=REGISTRY
)

queue_tasks_total = Counter(
    'queue_tasks_total',
    'Total queue tasks',
    ['queue_name', 'task_type', 'status'],
    registry=REGISTRY
)

# ============================================================================
# System Info
# ============================================================================

system_info = Info(
    'system_info',
    'System information',
    registry=REGISTRY
)

# ============================================================================
# Metric Collection Utilities
# ============================================================================

class MetricsCollector:
    """Collects and manages metrics"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.start_time = time.time()
        
    @asynccontextmanager
    async def track_request(self, request: Request):
        """Track HTTP request metrics"""
        start_time = time.time()
        endpoint = f"{request.method} {request.url.path}"
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            status = "success"  # Will be updated by middleware
            
            http_request_duration_seconds.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)
    
    @asynccontextmanager
    async def track_db_query(self, query_type: str, table: str):
        """Track database query metrics"""
        start_time = time.time()
        
        try:
            yield
        except Exception as e:
            db_query_errors_total.labels(
                error_type=type(e).__name__
            ).inc()
            raise
        finally:
            duration = time.time() - start_time
            db_query_duration_seconds.labels(
                query_type=query_type,
                table=table
            ).observe(duration)
    
    def track_cache_hit(self, cache_type: str = "redis"):
        """Track cache hit"""
        cache_hits_total.labels(cache_type=cache_type).inc()
    
    def track_cache_miss(self, cache_type: str = "redis"):
        """Track cache miss"""
        cache_misses_total.labels(cache_type=cache_type).inc()
    
    def track_video_generated(self, channel_id: str, status: str, cost: float, duration: float):
        """Track video generation metrics"""
        videos_generated_total.labels(
            channel_id=channel_id,
            status=status
        ).inc()
        
        if status == "success":
            video_generation_cost_dollars.labels(
                service="total"
            ).observe(cost)
            
            video_generation_duration_seconds.labels(
                stage="total"
            ).observe(duration)
    
    def track_youtube_api_call(self, account_id: str, endpoint: str, status: str):
        """Track YouTube API call"""
        youtube_api_calls_total.labels(
            account_id=account_id,
            endpoint=endpoint,
            status=status
        ).inc()
    
    def track_revenue(self, amount: float, source: str, channel_id: str):
        """Track revenue"""
        revenue_total_dollars.labels(
            source=source,
            channel_id=channel_id
        ).inc(amount)
    
    def track_ml_inference(self, model: str, operation: str, duration: float):
        """Track ML inference"""
        ml_inference_duration_seconds.labels(
            model=model,
            operation=operation
        ).observe(duration)
    
    async def update_system_metrics(self):
        """Update system-level metrics"""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        
        # Process metrics
        process = psutil.Process()
        
        # Update Prometheus gauges
        system_info.info({
            'version': '1.0.0',
            'environment': 'production',
            'uptime_seconds': str(time.time() - self.start_time)
        })
    
    async def update_database_metrics(self, db: AsyncSession):
        """Update database metrics"""
        try:
            # Get connection pool stats
            result = await db.execute(text("""
                SELECT 
                    numbackends as active,
                    (SELECT setting::int FROM pg_settings WHERE name='max_connections') as max_conn
                FROM pg_stat_database 
                WHERE datname = current_database()
            """))
            row = result.first()
            
            if row:
                db_connections_active.set(row.active)
                db_connections_idle.set(row.max_conn - row.active)
            
            # Get slow query count
            result = await db.execute(text("""
                SELECT COUNT(*) as slow_queries
                FROM pg_stat_statements
                WHERE mean_exec_time > 5
            """))
            
        except Exception as e:
            logger.error(f"Failed to update database metrics: {e}")
    
    async def update_cache_metrics(self):
        """Update cache metrics"""
        if self.redis_client:
            try:
                info = await self.redis_client.info("memory")
                cache_memory_bytes.labels(cache_type="redis").set(
                    info.get("used_memory", 0)
                )
                
                # Get cache hit rate
                stats = await self.redis_client.info("stats")
                hits = stats.get("keyspace_hits", 0)
                misses = stats.get("keyspace_misses", 0)
                
                if hits + misses > 0:
                    hit_rate = hits / (hits + misses)
                    # Store as a gauge for monitoring
                    
            except Exception as e:
                logger.error(f"Failed to update cache metrics: {e}")
    
    async def update_gpu_metrics(self):
        """Update GPU metrics"""
        try:
            import pynvml
            pynvml.nvmlInit()
            
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization_percent.labels(gpu_id=str(i)).set(util.gpu)
                
                # Memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_used_bytes.labels(gpu_id=str(i)).set(mem_info.used)
                gpu_memory_total_bytes.labels(gpu_id=str(i)).set(mem_info.total)
                
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                gpu_temperature_celsius.labels(gpu_id=str(i)).set(temp)
                
        except Exception as e:
            logger.debug(f"GPU metrics not available: {e}")
    
    async def update_business_metrics(self, db: AsyncSession):
        """Update business metrics from database"""
        try:
            # Update channel metrics
            result = await db.execute(text("""
                SELECT 
                    channel_id,
                    subscriber_count,
                    total_views,
                    health_score
                FROM channels
                WHERE is_active = true
            """))
            
            for row in result:
                channel_subscriber_count.labels(channel_id=row.channel_id).set(row.subscriber_count)
                channel_view_count.labels(channel_id=row.channel_id).set(row.total_views)
                channel_health_score.labels(channel_id=row.channel_id).set(row.health_score)
            
            # Update subscription metrics
            result = await db.execute(text("""
                SELECT 
                    plan_type,
                    COUNT(*) as count
                FROM subscriptions
                WHERE status = 'active'
                GROUP BY plan_type
            """))
            
            for row in result:
                subscription_active_count.labels(plan=row.plan_type).set(row.count)
            
            # Update YouTube quota metrics
            result = await db.execute(text("""
                SELECT 
                    account_id,
                    quota_used,
                    quota_limit
                FROM youtube_accounts
                WHERE is_active = true
            """))
            
            for row in result:
                youtube_quota_used.labels(account_id=row.account_id).set(row.quota_used)
                youtube_quota_limit.labels(account_id=row.account_id).set(row.quota_limit)
                
        except Exception as e:
            logger.error(f"Failed to update business metrics: {e}")


# ============================================================================
# Metrics Endpoint
# ============================================================================

async def get_metrics() -> Response:
    """Generate Prometheus metrics"""
    metrics = generate_latest(REGISTRY)
    return Response(
        content=metrics,
        media_type=CONTENT_TYPE_LATEST
    )


# ============================================================================
# Background Metrics Updater
# ============================================================================

async def start_metrics_updater(
    collector: MetricsCollector,
    db_session_factory,
    interval: int = 30
):
    """Background task to update metrics periodically"""
    while True:
        try:
            # Update system metrics
            await collector.update_system_metrics()
            
            # Update cache metrics
            await collector.update_cache_metrics()
            
            # Update GPU metrics
            await collector.update_gpu_metrics()
            
            # Update database and business metrics
            async with db_session_factory() as db:
                await collector.update_database_metrics(db)
                await collector.update_business_metrics(db)
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
        
        await asyncio.sleep(interval)