# Data Pipeline Monitoring, Metrics & Alerting

**Document Version**: 3.0 (Consolidated)  
**Date**: January 2025  
**Scope**: Complete Monitoring Implementation  
**Stack**: Prometheus, Grafana, Custom Metrics

---

## Table of Contents
1. [Metrics Collection Setup](#1-metrics-collection-setup)
2. [Prometheus Configuration](#2-prometheus-configuration)
3. [Grafana Dashboards](#3-grafana-dashboards)
4. [Alert Rules](#4-alert-rules)
5. [Custom Monitoring Code](#5-custom-monitoring-code)
6. [Error Recovery Procedures](#6-error-recovery-procedures)

---

## 1. Metrics Collection Setup

### Core Metrics Implementation

```python
"""
metrics.py - Comprehensive metrics collection
"""

from prometheus_client import Counter, Histogram, Gauge, Summary, Info
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from functools import wraps
import time
import logging
from typing import Callable, Any
from fastapi import Response
import asyncio

logger = logging.getLogger(__name__)

class PipelineMetrics:
    """
    Central metrics collection for all pipeline operations
    """
    
    def __init__(self):
        # Pipeline Stage Metrics
        self.stage_duration = Histogram(
            'pipeline_stage_duration_seconds',
            'Time spent in each pipeline stage',
            ['stage', 'status'],
            buckets=[0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600]
        )
        
        self.stage_counter = Counter(
            'pipeline_stage_total',
            'Total number of stage executions',
            ['stage', 'status']
        )
        
        self.stage_errors = Counter(
            'pipeline_stage_errors_total',
            'Total number of stage errors',
            ['stage', 'error_type', 'severity']
        )
        
        # Video Processing Metrics
        self.videos_processed = Counter(
            'pipeline_videos_processed_total',
            'Total videos processed',
            ['status', 'complexity']
        )
        
        self.video_processing_time = Histogram(
            'pipeline_video_processing_seconds',
            'End-to-end video processing time',
            buckets=[60, 120, 180, 300, 450, 600, 900, 1200]
        )
        
        # Cost Metrics
        self.video_cost = Histogram(
            'pipeline_video_cost_dollars',
            'Cost per video in dollars',
            ['optimization_level'],
            buckets=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
        )
        
        self.service_cost = Counter(
            'pipeline_service_cost_total',
            'Cumulative cost per service',
            ['service']
        )
        
        self.cost_threshold_breaches = Counter(
            'pipeline_cost_threshold_breaches_total',
            'Number of times cost thresholds were breached',
            ['threshold_level']
        )
        
        # Queue Metrics
        self.queue_depth = Gauge(
            'pipeline_queue_depth',
            'Current queue depth',
            ['priority', 'complexity']
        )
        
        self.queue_wait_time = Histogram(
            'pipeline_queue_wait_seconds',
            'Time spent waiting in queue',
            buckets=[10, 30, 60, 120, 300, 600, 1800, 3600]
        )
        
        self.dequeue_latency = Histogram(
            'pipeline_dequeue_latency_seconds',
            'Time to dequeue a job',
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
        )
        
        # Resource Metrics
        self.gpu_utilization = Gauge(
            'pipeline_gpu_utilization_percent',
            'GPU utilization percentage'
        )
        
        self.gpu_memory_used = Gauge(
            'pipeline_gpu_memory_used_mb',
            'GPU memory used in MB'
        )
        
        self.cpu_utilization = Gauge(
            'pipeline_cpu_utilization_percent',
            'CPU utilization percentage'
        )
        
        self.memory_used = Gauge(
            'pipeline_memory_used_mb',
            'System memory used in MB'
        )
        
        self.active_workers = Gauge(
            'pipeline_active_workers',
            'Number of active workers',
            ['worker_type']
        )
        
        # API Metrics
        self.api_requests = Counter(
            'pipeline_api_requests_total',
            'Total API requests',
            ['endpoint', 'method', 'status']
        )
        
        self.api_latency = Histogram(
            'pipeline_api_latency_seconds',
            'API request latency',
            ['endpoint', 'method'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        )
        
        # Business Metrics
        self.daily_videos = Gauge(
            'pipeline_daily_videos_total',
            'Videos processed today'
        )
        
        self.daily_cost = Gauge(
            'pipeline_daily_cost_dollars',
            'Total cost today in dollars'
        )
        
        self.success_rate = Gauge(
            'pipeline_success_rate_percent',
            'Current success rate percentage'
        )
        
        # System Info
        self.system_info = Info(
            'pipeline_system',
            'Pipeline system information'
        )
        self.system_info.info({
            'version': '1.0.0',
            'environment': 'production',
            'gpu_model': 'RTX 5090',
            'cpu_model': 'Ryzen 9 9950X3D'
        })
    
    def track_stage(self, stage: str):
        """Decorator to track stage execution"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                start_time = time.time()
                status = 'success'
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                    
                except Exception as e:
                    status = 'failure'
                    self.stage_errors.labels(
                        stage=stage,
                        error_type=type(e).__name__,
                        severity='error'
                    ).inc()
                    raise
                    
                finally:
                    duration = time.time() - start_time
                    self.stage_duration.labels(
                        stage=stage,
                        status=status
                    ).observe(duration)
                    self.stage_counter.labels(
                        stage=stage,
                        status=status
                    ).inc()
                    
                    if duration > 300:  # Log slow stages
                        logger.warning(f"Stage {stage} took {duration:.2f}s")
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                start_time = time.time()
                status = 'success'
                
                try:
                    result = func(*args, **kwargs)
                    return result
                    
                except Exception as e:
                    status = 'failure'
                    self.stage_errors.labels(
                        stage=stage,
                        error_type=type(e).__name__,
                        severity='error'
                    ).inc()
                    raise
                    
                finally:
                    duration = time.time() - start_time
                    self.stage_duration.labels(
                        stage=stage,
                        status=status
                    ).observe(duration)
                    self.stage_counter.labels(
                        stage=stage,
                        status=status
                    ).inc()
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
                
        return decorator
    
    def track_api(self, endpoint: str, method: str):
        """Decorator to track API requests"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> Any:
                start_time = time.time()
                status = 200
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                    
                except Exception as e:
                    status = 500
                    raise
                    
                finally:
                    latency = time.time() - start_time
                    self.api_requests.labels(
                        endpoint=endpoint,
                        method=method,
                        status=str(status)
                    ).inc()
                    self.api_latency.labels(
                        endpoint=endpoint,
                        method=method
                    ).observe(latency)
                    
            return wrapper
        return decorator
    
    async def update_resource_metrics(self):
        """Update resource utilization metrics"""
        import psutil
        import GPUtil
        
        while True:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                self.cpu_utilization.set(cpu_percent)
                
                # Memory metrics
                memory = psutil.virtual_memory()
                self.memory_used.set(memory.used // (1024 * 1024))
                
                # GPU metrics
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        self.gpu_utilization.set(gpu.load * 100)
                        self.gpu_memory_used.set(gpu.memoryUsed)
                except Exception as e:
                    logger.error(f"Failed to get GPU metrics: {e}")
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Failed to update resource metrics: {e}")
                await asyncio.sleep(60)

# Global metrics instance
metrics = PipelineMetrics()

# FastAPI endpoint for metrics
async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
```

---

## 2. Prometheus Configuration

### prometheus.yml

```yaml
# Prometheus configuration for YTEMPIRE pipeline monitoring
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    monitor: 'ytempire-pipeline'
    environment: 'production'

# Alert manager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - localhost:9093

# Load rules files
rule_files:
  - "/etc/prometheus/rules/*.yml"

# Scrape configurations
scrape_configs:
  # Pipeline API metrics
  - job_name: 'pipeline-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
    
  # PostgreSQL exporter
  - job_name: 'postgresql'
    static_configs:
      - targets: ['localhost:9187']
    
  # Redis exporter
  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:9121']
    
  # Node exporter for system metrics
  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']
    
  # NVIDIA GPU exporter
  - job_name: 'nvidia-gpu'
    static_configs:
      - targets: ['localhost:9835']

# Remote write for long-term storage (optional)
remote_write:
  - url: "http://localhost:9009/api/v1/push"
    queue_config:
      capacity: 10000
      max_shards: 5
      max_samples_per_send: 500
```

### Alert Rules (alerts.yml)

```yaml
groups:
  - name: pipeline_critical
    interval: 30s
    rules:
      # Pipeline stalled
      - alert: PipelineStalled
        expr: rate(pipeline_videos_processed_total[5m]) == 0
        for: 5m
        labels:
          severity: critical
          team: backend
        annotations:
          summary: "Pipeline has not processed any videos in 5 minutes"
          description: "No videos completed. Queue depth: {{ $value }}"
          runbook: "https://wiki/runbooks/pipeline-stalled"
      
      # Cost exceeded
      - alert: CostExceeded
        expr: pipeline_video_cost_dollars > 3.0
        for: 1m
        labels:
          severity: critical
          team: backend
        annotations:
          summary: "Video cost exceeded $3.00 limit"
          description: "Video {{ $labels.video_id }} cost: ${{ $value }}"
          action: "Investigate cost optimization immediately"
      
      # High error rate
      - alert: HighErrorRate
        expr: |
          (sum(rate(pipeline_stage_errors_total[5m])) / 
           sum(rate(pipeline_stage_total[5m]))) > 0.1
        for: 3m
        labels:
          severity: warning
          team: backend
        annotations:
          summary: "Pipeline error rate above 10%"
          description: "Current error rate: {{ $value | humanizePercentage }}"
      
      # GPU memory critical
      - alert: GPUMemoryCritical
        expr: pipeline_gpu_memory_used_mb > 30000
        for: 2m
        labels:
          severity: critical
          team: backend
        annotations:
          summary: "GPU memory usage critical"
          description: "GPU memory at {{ $value }}MB of 32768MB"
      
      # Queue backup
      - alert: QueueBackup
        expr: pipeline_queue_depth > 80
        for: 10m
        labels:
          severity: warning
          team: backend
        annotations:
          summary: "Queue depth exceeding threshold"
          description: "Current queue depth: {{ $value }}"
  
  - name: pipeline_performance
    interval: 1m
    rules:
      # Slow processing
      - alert: SlowProcessing
        expr: |
          histogram_quantile(0.95, 
            rate(pipeline_video_processing_seconds_bucket[5m])
          ) > 600
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Video processing P95 > 10 minutes"
          description: "Current P95: {{ $value | humanizeDuration }}"
      
      # Slow dequeue
      - alert: SlowDequeue
        expr: |
          histogram_quantile(0.95,
            rate(pipeline_dequeue_latency_seconds_bucket[5m])
          ) > 0.1
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "Queue dequeue latency > 100ms"
          description: "Current P95: {{ $value }}s"
  
  - name: pipeline_business
    interval: 5m
    rules:
      # Daily target miss
      - alert: DailyTargetMiss
        expr: pipeline_daily_videos_total < 40 and hour() == 20
        for: 5m
        labels:
          severity: warning
          team: business
        annotations:
          summary: "Daily video target at risk"
          description: "Only {{ $value }} videos processed by 8 PM"
      
      # Success rate low
      - alert: LowSuccessRate
        expr: pipeline_success_rate_percent < 90
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Pipeline success rate below 90%"
          description: "Current success rate: {{ $value }}%"
```

---

## 3. Grafana Dashboards

### Main Pipeline Dashboard JSON

```json
{
  "dashboard": {
    "title": "YTEMPIRE Pipeline Monitoring",
    "timezone": "browser",
    "panels": [
      {
        "title": "Videos Processed (24h)",
        "type": "stat",
        "gridPos": {"x": 0, "y": 0, "w": 6, "h": 3},
        "targets": [
          {
            "expr": "sum(increase(pipeline_videos_processed_total[24h]))",
            "legendFormat": "Total Videos"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "yellow", "value": 30},
                {"color": "green", "value": 50}
              ]
            }
          }
        }
      },
      {
        "title": "Current Queue Depth",
        "type": "gauge",
        "gridPos": {"x": 6, "y": 0, "w": 6, "h": 3},
        "targets": [
          {
            "expr": "sum(pipeline_queue_depth)",
            "legendFormat": "Queue Depth"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "max": 100,
            "thresholds": {
              "steps": [
                {"color": "green", "value": 0},
                {"color": "yellow", "value": 50},
                {"color": "red", "value": 80}
              ]
            }
          }
        }
      },
      {
        "title": "Average Cost per Video",
        "type": "stat",
        "gridPos": {"x": 12, "y": 0, "w": 6, "h": 3},
        "targets": [
          {
            "expr": "avg(pipeline_video_cost_dollars)",
            "legendFormat": "Avg Cost"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "currencyUSD",
            "thresholds": {
              "steps": [
                {"color": "green", "value": 0},
                {"color": "yellow", "value": 2.5},
                {"color": "red", "value": 3.0}
              ]
            }
          }
        }
      },
      {
        "title": "Success Rate",
        "type": "gauge",
        "gridPos": {"x": 18, "y": 0, "w": 6, "h": 3},
        "targets": [
          {
            "expr": "pipeline_success_rate_percent",
            "legendFormat": "Success %"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "max": 100,
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "yellow", "value": 90},
                {"color": "green", "value": 95}
              ]
            }
          }
        }
      },
      {
        "title": "Processing Time Distribution",
        "type": "graph",
        "gridPos": {"x": 0, "y": 3, "w": 12, "h": 8},
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(pipeline_video_processing_seconds_bucket[5m]))",
            "legendFormat": "P50"
          },
          {
            "expr": "histogram_quantile(0.95, rate(pipeline_video_processing_seconds_bucket[5m]))",
            "legendFormat": "P95"
          },
          {
            "expr": "histogram_quantile(0.99, rate(pipeline_video_processing_seconds_bucket[5m]))",
            "legendFormat": "P99"
          }
        ],
        "yaxes": [
          {
            "format": "s",
            "label": "Processing Time"
          }
        ]
      },
      {
        "title": "Cost Breakdown by Service",
        "type": "piechart",
        "gridPos": {"x": 12, "y": 3, "w": 12, "h": 8},
        "targets": [
          {
            "expr": "sum by (service) (pipeline_service_cost_total)",
            "legendFormat": "{{service}}"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "type": "graph",
        "gridPos": {"x": 0, "y": 11, "w": 8, "h": 6},
        "targets": [
          {
            "expr": "pipeline_gpu_utilization_percent",
            "legendFormat": "GPU %"
          },
          {
            "expr": "(pipeline_gpu_memory_used_mb / 32768) * 100",
            "legendFormat": "GPU Memory %"
          }
        ],
        "yaxes": [
          {
            "format": "percent",
            "max": 100
          }
        ]
      },
      {
        "title": "Stage Performance Heatmap",
        "type": "heatmap",
        "gridPos": {"x": 8, "y": 11, "w": 16, "h": 6},
        "targets": [
          {
            "expr": "sum by (stage) (rate(pipeline_stage_duration_seconds_sum[5m])) / sum by (stage) (rate(pipeline_stage_duration_seconds_count[5m]))",
            "format": "heatmap"
          }
        ]
      },
      {
        "title": "Error Rate by Stage",
        "type": "table",
        "gridPos": {"x": 0, "y": 17, "w": 24, "h": 6},
        "targets": [
          {
            "expr": "sum by (stage, error_type) (rate(pipeline_stage_errors_total[5m]))",
            "format": "table"
          }
        ]
      }
    ]
  }
}
```

---

## 4. Alert Rules

### Alertmanager Configuration

```yaml
# alertmanager.yml
global:
  resolve_timeout: 5m
  slack_api_url: 'YOUR_SLACK_WEBHOOK_URL'

route:
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 5m
  repeat_interval: 1h
  receiver: 'default'
  
  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'
      continue: true
    
    - match:
        severity: warning
      receiver: 'warning-alerts'

receivers:
  - name: 'default'
    slack_configs:
      - channel: '#pipeline-alerts'
        title: 'Pipeline Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
  
  - name: 'critical-alerts'
    slack_configs:
      - channel: '#critical-alerts'
        title: '🚨 CRITICAL: Pipeline Issue'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_KEY'
  
  - name: 'warning-alerts'
    slack_configs:
      - channel: '#pipeline-warnings'
        title: '⚠️ Pipeline Warning'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname']
```

---

## 5. Custom Monitoring Code

### Health Check Endpoint

```python
"""
health.py - System health checks
"""

from fastapi import FastAPI, Response
from typing import Dict, List
import asyncio
import asyncpg
import redis.asyncio as redis
import psutil
import GPUtil
from datetime import datetime, timedelta

app = FastAPI()

class HealthChecker:
    """
    Comprehensive health checking system
    """
    
    def __init__(self, db_pool, redis_client, queue_manager):
        self.db_pool = db_pool
        self.redis_client = redis_client
        self.queue_manager = queue_manager
        
    async def check_database(self) -> Dict:
        """Check PostgreSQL health"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
                
                # Check queue table
                queue_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM video_queue WHERE status = 'queued'"
                )
                
                return {
                    "status": "healthy",
                    "queued_videos": queue_count,
                    "response_time_ms": 5
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def check_redis(self) -> Dict:
        """Check Redis health"""
        try:
            start = asyncio.get_event_loop().time()
            await self.redis_client.ping()
            latency = (asyncio.get_event_loop().time() - start) * 1000
            
            queue_depth = await self.redis_client.zcard("queue:priority")
            
            return {
                "status": "healthy",
                "queue_depth": queue_depth,
                "latency_ms": round(latency, 2)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def check_gpu(self) -> Dict:
        """Check GPU health"""
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return {"status": "unavailable"}
            
            gpu = gpus[0]
            
            return {
                "status": "healthy" if gpu.load < 0.95 else "overloaded",
                "name": gpu.name,
                "utilization": round(gpu.load * 100, 2),
                "memory_used_mb": round(gpu.memoryUsed),
                "memory_total_mb": round(gpu.memoryTotal),
                "temperature": gpu.temperature
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def check_system(self) -> Dict:
        """Check system resources"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cpu": {
                "utilization": cpu_percent,
                "status": "healthy" if cpu_percent < 80 else "high"
            },
            "memory": {
                "used_gb": round(memory.used / (1024**3), 2),
                "total_gb": round(memory.total / (1024**3), 2),
                "percent": memory.percent,
                "status": "healthy" if memory.percent < 80 else "high"
            },
            "disk": {
                "used_gb": round(disk.used / (1024**3), 2),
                "total_gb": round(disk.total / (1024**3), 2),
                "percent": disk.percent,
                "status": "healthy" if disk.percent < 80 else "high"
            }
        }
    
    async def check_pipeline(self) -> Dict:
        """Check pipeline processing health"""
        try:
            metrics = await self.queue_manager.get_metrics()
            
            # Calculate success rate
            total = metrics['total_processed'] + metrics['total_failed']
            success_rate = (metrics['total_processed'] / total * 100) if total > 0 else 100
            
            # Check if pipeline is stalled
            async with self.db_pool.acquire() as conn:
                last_completed = await conn.fetchval("""
                    SELECT MAX(completed_at) 
                    FROM video_queue 
                    WHERE status = 'completed'
                """)
                
                stalled = False
                if last_completed:
                    time_since = datetime.utcnow() - last_completed
                    stalled = time_since > timedelta(minutes=30)
            
            return {
                "status": "stalled" if stalled else "healthy",
                "queue_depth": metrics['queue_depth'],
                "processing_count": metrics['processing_count'],
                "success_rate": round(success_rate, 2),
                "total_processed": metrics['total_processed'],
                "total_failed": metrics['total_failed']
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def full_health_check(self) -> Dict:
        """Perform full system health check"""
        
        results = await asyncio.gather(
            self.check_database(),
            self.check_redis(),
            self.check_gpu(),
            self.check_system(),
            self.check_pipeline(),
            return_exceptions=True
        )
        
        health = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "components": {
                "database": results[0] if not isinstance(results[0], Exception) else {"status": "error"},
                "redis": results[1] if not isinstance(results[1], Exception) else {"status": "error"},
                "gpu": results[2] if not isinstance(results[2], Exception) else {"status": "error"},
                "system": results[3] if not isinstance(results[3], Exception) else {"status": "error"},
                "pipeline": results[4] if not isinstance(results[4], Exception) else {"status": "error"}
            }
        }
        
        # Determine overall status
        for component in health['components'].values():
            if isinstance(component, dict) and component.get('status') in ['unhealthy', 'error', 'stalled']:
                health['overall_status'] = 'degraded'
                break
        
        return health

# Health check endpoints
health_checker = None  # Initialize with dependencies

@app.get("/health")
async def health_check():
    """Basic health check"""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check"""
    if health_checker:
        return await health_checker.full_health_check()
    return {"status": "error", "message": "Health checker not initialized"}

@app.get("/health/ready")
async def readiness_check():
    """Kubernetes readiness probe"""
    if health_checker:
        health = await health_checker.full_health_check()
        if health['overall_status'] == 'healthy':
            return Response(status_code=200)
    return Response(status_code=503)

@app.get("/health/live")
async def liveness_check():
    """Kubernetes liveness probe"""
    return Response(status_code=200)
```

---

## 6. Error Recovery Procedures

### Automated Recovery System

```python
"""
recovery.py - Automated error recovery procedures
"""

import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

class RecoveryAction(Enum):
    RETRY = "retry"
    RESTART = "restart"
    FAILOVER = "failover"
    ALERT = "alert"
    HALT = "halt"

class ErrorRecoverySystem:
    """
    Automated error recovery for pipeline failures
    """
    
    def __init__(self, queue_manager, resource_scheduler, cost_tracker):
        self.queue_manager = queue_manager
        self.resource_scheduler = resource_scheduler
        self.cost_tracker = cost_tracker
        self.recovery_history = {}
        
    async def handle_error(self, error_context: Dict) -> RecoveryAction:
        """
        Determine and execute recovery action based on error
        """
        
        error_type = error_context.get('error_type')
        video_id = error_context.get('video_id')
        stage = error_context.get('stage')
        retry_count = error_context.get('retry_count', 0)
        
        # Determine recovery action
        action = self._determine_action(error_type, retry_count)
        
        # Execute recovery
        await self._execute_recovery(action, error_context)
        
        # Log recovery attempt
        await self._log_recovery(video_id, action, error_context)
        
        return action
    
    def _determine_action(self, error_type: str, retry_count: int) -> RecoveryAction:
        """Determine appropriate recovery action"""
        
        # Cost limit errors - immediate halt
        if "cost" in error_type.lower():
            return RecoveryAction.HALT
        
        # GPU OOM errors - restart with CPU fallback
        if "gpu" in error_type.lower() and "memory" in error_type.lower():
            return RecoveryAction.FAILOVER
        
        # API errors - retry with backoff
        if "api" in error_type.lower() or "timeout" in error_type.lower():
            if retry_count < 3:
                return RecoveryAction.RETRY
            else:
                return RecoveryAction.ALERT
        
        # Unknown errors - alert after retries
        if retry_count < 2:
            return RecoveryAction.RETRY
        else:
            return RecoveryAction.ALERT
    
    async def _execute_recovery(self, action: RecoveryAction, context: Dict):
        """Execute the recovery action"""
        
        video_id = context.get('video_id')
        
        if action == RecoveryAction.RETRY:
            # Schedule retry with exponential backoff
            retry_count = context.get('retry_count', 0)
            delay = min(10 * (2 ** retry_count), 300)  # Max 5 minutes
            
            logger.info(f"Scheduling retry for {video_id} in {delay}s")
            await asyncio.sleep(delay)
            
            # Re-queue the video
            await self.queue_manager._schedule_retry(video_id, 0)
            
        elif action == RecoveryAction.RESTART:
            # Restart the processing stage
            logger.info(f"Restarting stage {context.get('stage')} for {video_id}")
            # Implementation depends on pipeline architecture
            
        elif action == RecoveryAction.FAILOVER:
            # Switch to alternative processing method
            logger.info(f"Failing over {video_id} to CPU processing")
            
            # Update video complexity to force CPU processing
            async with self.queue_manager.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE video_queue 
                    SET complexity = 'simple',
                        metadata = metadata || '{"failover": true}'::jsonb
                    WHERE id = $1
                """, video_id)
            
            # Re-queue immediately
            await self.queue_manager._schedule_retry(video_id, 0)
            
        elif action == RecoveryAction.ALERT:
            # Send alert to operations team
            await self._send_alert(context)
            
        elif action == RecoveryAction.HALT:
            # Stop processing immediately
            logger.error(f"HALTING processing for {video_id}")
            await self.queue_manager.fail(video_id, "Cost limit exceeded", retry=False)
            await self._send_critical_alert(context)
    
    async def _log_recovery(self, video_id: str, action: RecoveryAction, context: Dict):
        """Log recovery attempt for analysis"""
        
        if video_id not in self.recovery_history:
            self.recovery_history[video_id] = []
        
        self.recovery_history[video_id].append({
            "timestamp": datetime.utcnow().isoformat(),
            "action": action.value,
            "context": context
        })
        
        # Store in database for persistence
        async with self.queue_manager.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE video_queue 
                SET error_log = COALESCE(error_log, '[]'::jsonb) || $1::jsonb
                WHERE id = $2
            """,
                json.dumps([{
                    "recovery_action": action.value,
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": context.get('error_message')
                }]),
                video_id
            )
    
    async def _send_alert(self, context: Dict):
        """Send alert to monitoring system"""
        await self.queue_manager._publish_event("alert", {
            "severity": "warning",
            "video_id": context.get('video_id'),
            "stage": context.get('stage'),
            "error": context.get('error_message'),
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def _send_critical_alert(self, context: Dict):
        """Send critical alert requiring immediate attention"""
        await self.queue_manager._publish_event("alert.critical", {
            "severity": "critical",
            "video_id": context.get('video_id'),
            "reason": context.get('error_message'),
            "action_required": "Manual intervention needed",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Could also send to PagerDuty, Slack, etc.
        logger.critical(f"CRITICAL ALERT: {context}")

# Recovery procedures for specific scenarios

async def recover_from_gpu_oom(video_id: str, resource_scheduler):
    """Recover from GPU out-of-memory error"""
    
    logger.warning(f"GPU OOM recovery for {video_id}")
    
    # 1. Clear GPU memory
    await resource_scheduler._check_gpu_health()
    
    # 2. Release any stuck allocations
    if video_id in resource_scheduler.gpu_allocations:
        await resource_scheduler.release(
            resource_scheduler.gpu_allocations[video_id]
        )
    
    # 3. Force garbage collection
    import gc
    gc.collect()
    
    # 4. Wait for memory to clear
    await asyncio.sleep(5)
    
    return True

async def recover_from_pipeline_stall(queue_manager):
    """Recover from stalled pipeline"""
    
    logger.warning("Attempting pipeline stall recovery")
    
    # 1. Check for stuck jobs
    async with queue_manager.db_pool.acquire() as conn:
        stuck_jobs = await conn.fetch("""
            SELECT id, processing_started_at
            FROM video_queue
            WHERE status = 'processing'
            AND processing_started_at < NOW() - INTERVAL '30 minutes'
        """)
        
        for job in stuck_jobs:
            video_id = str(job['id'])
            logger.warning(f"Found stuck job: {video_id}")
            
            # Reset to queued state
            await conn.execute("""
                UPDATE video_queue
                SET status = 'queued',
                    processing_started_at = NULL,
                    retry_count = retry_count + 1
                WHERE id = $1
            """, video_id)
            
            # Re-add to Redis queue
            await queue_manager._schedule_retry(video_id, 0)
    
    # 2. Clear Redis processing set
    processing_ids = await queue_manager.redis_client.smembers("queue:processing")
    for video_id in processing_ids:
        # Verify if actually processing
        async with queue_manager.db_pool.acquire() as conn:
            status = await conn.fetchval(
                "SELECT status FROM video_queue WHERE id = $1",
                video_id
            )
            
            if status != 'processing':
                await queue_manager.redis_client.srem("queue:processing", video_id)
    
    return True

async def recover_from_cost_overrun(video_id: str, cost_tracker):
    """Recover from cost overrun"""
    
    logger.error(f"Cost overrun recovery for {video_id}")
    
    # 1. Get current cost breakdown
    breakdown = await cost_tracker.get_cost_breakdown(video_id)
    
    # 2. Identify expensive operations
    expensive_ops = [
        service for service, data in breakdown['breakdown'].items()
        if data['total'] > 0.5
    ]
    
    # 3. Mark for optimization
    async with cost_tracker.db_pool.acquire() as conn:
        await conn.execute("""
            UPDATE video_queue
            SET metadata = metadata || $1::jsonb
            WHERE id = $2
        """,
            json.dumps({
                "cost_optimization_required": True,
                "expensive_operations": expensive_ops
            }),
            video_id
        )
    
    return False  # Cannot automatically recover - needs manual review
```

This monitoring and alerting system provides comprehensive observability for your pipeline, enabling quick detection and recovery from failures while maintaining the <5 minute MTTR target.