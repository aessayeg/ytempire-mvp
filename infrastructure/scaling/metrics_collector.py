#!/usr/bin/env python3
"""
Advanced Metrics Collector for Auto-Scaling
Collects, processes, and serves metrics for scaling decisions
"""

import asyncio
import aiohttp
import logging
import json
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import psutil
import redis
import psycopg2
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, start_http_server
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logger = logging.getLogger(__name__)

@dataclass
class MetricValue:
    """Individual metric value with metadata"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = None
    unit: str = ""
    description: str = ""

class PrometheusMetrics:
    """Prometheus metrics registry"""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        
        # System metrics
        self.cpu_usage = Gauge('system_cpu_usage_percent', 'CPU usage percentage', registry=self.registry)
        self.memory_usage = Gauge('system_memory_usage_percent', 'Memory usage percentage', registry=self.registry)
        self.disk_usage = Gauge('system_disk_usage_percent', 'Disk usage percentage', registry=self.registry)
        self.gpu_utilization = Gauge('system_gpu_utilization_percent', 'GPU utilization percentage', registry=self.registry)
        
        # Application metrics
        self.request_rate = Gauge('app_request_rate', 'HTTP requests per second', registry=self.registry)
        self.response_time_p95 = Gauge('app_response_time_p95_seconds', '95th percentile response time', registry=self.registry)
        self.queue_depth = Gauge('app_queue_depth', 'Application queue depth', registry=self.registry)
        self.active_connections = Gauge('app_active_connections', 'Active database connections', registry=self.registry)
        self.error_rate = Gauge('app_error_rate', 'Application error rate', registry=self.registry)
        
        # Video processing metrics
        self.video_queue_length = Gauge('video_queue_length', 'Video processing queue length', registry=self.registry)
        self.processing_rate = Gauge('video_processing_rate', 'Videos processed per minute', registry=self.registry)
        
        # Redis metrics
        self.redis_connected_clients = Gauge('redis_connected_clients', 'Redis connected clients', registry=self.registry)
        self.redis_memory_usage = Gauge('redis_memory_usage_bytes', 'Redis memory usage', registry=self.registry)
        self.redis_operations_per_sec = Gauge('redis_operations_per_sec', 'Redis operations per second', registry=self.registry)
        
        # Scaling events
        self.scaling_events = Counter('scaling_events_total', 'Total scaling events', ['service', 'action'], registry=self.registry)
        self.scaling_duration = Histogram('scaling_duration_seconds', 'Time taken for scaling operations', ['service'], registry=self.registry)

class AdvancedMetricsCollector:
    """Advanced metrics collection with buffering and aggregation"""
    
    def __init__(self):
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', '6379')),
            password=os.getenv('REDIS_PASSWORD') or None,
            decode_responses=True
        )
        
        self.db_config = {
            'host': os.getenv('DATABASE_HOST', 'localhost'),
            'port': int(os.getenv('DATABASE_PORT', '5432')),
            'database': os.getenv('DATABASE_NAME', 'ytempire'),
            'user': os.getenv('DATABASE_USER', 'postgres'),
            'password': os.getenv('DATABASE_PASSWORD', 'password')
        }
        
        self.api_base_url = os.getenv('API_BASE_URL', 'http://localhost:8000')
        self.prometheus_url = os.getenv('PROMETHEUS_URL', 'http://localhost:9090')
        
        self.prometheus_metrics = PrometheusMetrics()
        self.metrics_buffer: List[MetricValue] = []
        self.max_buffer_size = 1000
        
        # Historical data for trend analysis
        self.historical_data: Dict[str, List[float]] = {}
        self.max_history_points = 100
        
    async def collect_system_metrics(self) -> List[MetricValue]:
        """Collect comprehensive system metrics"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
            avg_cpu = sum(cpu_percent) / len(cpu_percent)
            
            metrics.append(MetricValue(
                name='system_cpu_usage_percent',
                value=avg_cpu,
                timestamp=timestamp,
                unit='%',
                description='Average CPU usage across all cores'
            ))
            
            # Individual core usage
            for i, usage in enumerate(cpu_percent):
                metrics.append(MetricValue(
                    name='system_cpu_core_usage_percent',
                    value=usage,
                    timestamp=timestamp,
                    labels={'core': str(i)},
                    unit='%'
                ))
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.extend([
                MetricValue('system_memory_usage_percent', memory.percent, timestamp, unit='%'),
                MetricValue('system_memory_used_bytes', memory.used, timestamp, unit='bytes'),
                MetricValue('system_memory_available_bytes', memory.available, timestamp, unit='bytes'),
                MetricValue('system_memory_total_bytes', memory.total, timestamp, unit='bytes')
            ])
            
            # Swap metrics
            swap = psutil.swap_memory()
            metrics.extend([
                MetricValue('system_swap_usage_percent', swap.percent, timestamp, unit='%'),
                MetricValue('system_swap_used_bytes', swap.used, timestamp, unit='bytes')
            ])
            
            # Disk metrics
            disk_usage = {}
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    usage_percent = (usage.used / usage.total) * 100
                    
                    metrics.append(MetricValue(
                        name='system_disk_usage_percent',
                        value=usage_percent,
                        timestamp=timestamp,
                        labels={'mountpoint': partition.mountpoint},
                        unit='%'
                    ))
                    
                except PermissionError:
                    continue
            
            # Network metrics
            net_io = psutil.net_io_counters()
            metrics.extend([
                MetricValue('system_network_bytes_sent', net_io.bytes_sent, timestamp, unit='bytes'),
                MetricValue('system_network_bytes_recv', net_io.bytes_recv, timestamp, unit='bytes'),
                MetricValue('system_network_packets_sent', net_io.packets_sent, timestamp, unit='packets'),
                MetricValue('system_network_packets_recv', net_io.packets_recv, timestamp, unit='packets')
            ])
            
            # Load average (Unix-like systems)
            if hasattr(os, 'getloadavg'):
                load_avg = os.getloadavg()
                metrics.extend([
                    MetricValue('system_load_average_1m', load_avg[0], timestamp),
                    MetricValue('system_load_average_5m', load_avg[1], timestamp),
                    MetricValue('system_load_average_15m', load_avg[2], timestamp)
                ])
            
            # Process count
            process_count = len(psutil.pids())
            metrics.append(MetricValue('system_process_count', process_count, timestamp, unit='processes'))
            
            # GPU metrics (if available)
            gpu_utilization = await self._get_gpu_metrics()
            metrics.extend(gpu_utilization)
            
            # Update Prometheus metrics
            self.prometheus_metrics.cpu_usage.set(avg_cpu)
            self.prometheus_metrics.memory_usage.set(memory.percent)
            if disk_usage:
                self.prometheus_metrics.disk_usage.set(max(disk_usage.values()))
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
        
        return metrics
    
    async def collect_application_metrics(self) -> List[MetricValue]:
        """Collect application-specific metrics"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # API metrics from application
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(f"{self.api_base_url}/metrics", timeout=5) as response:
                        if response.status == 200:
                            api_data = await response.json()
                            
                            for key, value in api_data.items():
                                if isinstance(value, (int, float)):
                                    metrics.append(MetricValue(
                                        name=f'app_{key}',
                                        value=float(value),
                                        timestamp=timestamp
                                    ))
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.warning(f"Failed to collect API metrics: {e}")
            
            # Database metrics
            db_metrics = await self._collect_database_metrics(timestamp)
            metrics.extend(db_metrics)
            
            # Redis metrics
            redis_metrics = await self._collect_redis_metrics(timestamp)
            metrics.extend(redis_metrics)
            
            # Video processing metrics
            video_metrics = await self._collect_video_processing_metrics(timestamp)
            metrics.extend(video_metrics)
            
        except Exception as e:
            logger.error(f"Failed to collect application metrics: {e}")
        
        return metrics
    
    async def _collect_database_metrics(self, timestamp: datetime) -> List[MetricValue]:
        """Collect PostgreSQL metrics"""
        metrics = []
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Connection statistics
            cursor.execute("""
                SELECT state, count(*) 
                FROM pg_stat_activity 
                WHERE datname = current_database()
                GROUP BY state
            """)
            
            connection_stats = dict(cursor.fetchall())
            active_connections = connection_stats.get('active', 0)
            idle_connections = connection_stats.get('idle', 0)
            
            metrics.extend([
                MetricValue('db_active_connections', active_connections, timestamp),
                MetricValue('db_idle_connections', idle_connections, timestamp),
                MetricValue('db_total_connections', sum(connection_stats.values()), timestamp)
            ])
            
            # Database size
            cursor.execute("SELECT pg_database_size(current_database())")
            db_size = cursor.fetchone()[0]
            metrics.append(MetricValue('db_size_bytes', db_size, timestamp, unit='bytes'))
            
            # Table statistics
            cursor.execute("""
                SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del, n_live_tup, n_dead_tup
                FROM pg_stat_user_tables
                ORDER BY n_live_tup DESC
                LIMIT 10
            """)
            
            for row in cursor.fetchall():
                schema, table, inserts, updates, deletes, live_tuples, dead_tuples = row
                table_label = f"{schema}.{table}"
                
                metrics.extend([
                    MetricValue('db_table_inserts_total', inserts, timestamp, 
                               labels={'table': table_label}),
                    MetricValue('db_table_updates_total', updates, timestamp, 
                               labels={'table': table_label}),
                    MetricValue('db_table_deletes_total', deletes, timestamp, 
                               labels={'table': table_label}),
                    MetricValue('db_table_live_tuples', live_tuples, timestamp, 
                               labels={'table': table_label}),
                    MetricValue('db_table_dead_tuples', dead_tuples, timestamp, 
                               labels={'table': table_label})
                ])
            
            # Query statistics
            cursor.execute("""
                SELECT calls, total_time, mean_time, max_time, query
                FROM pg_stat_statements
                ORDER BY total_time DESC
                LIMIT 5
            """)
            
            query_stats = cursor.fetchall()
            if query_stats:
                for i, (calls, total_time, mean_time, max_time, query) in enumerate(query_stats):
                    query_id = f"top_{i+1}"
                    metrics.extend([
                        MetricValue('db_query_calls_total', calls, timestamp, 
                                   labels={'query_id': query_id}),
                        MetricValue('db_query_total_time_ms', total_time, timestamp,
                                   labels={'query_id': query_id}, unit='ms'),
                        MetricValue('db_query_mean_time_ms', mean_time, timestamp,
                                   labels={'query_id': query_id}, unit='ms')
                    ])
            
            # Video queue specific metrics
            cursor.execute("SELECT status, count(*) FROM video_queue GROUP BY status")
            queue_stats = dict(cursor.fetchall())
            
            for status, count in queue_stats.items():
                metrics.append(MetricValue(
                    'video_queue_by_status',
                    count,
                    timestamp,
                    labels={'status': status}
                ))
            
            queue_depth = queue_stats.get('pending', 0) + queue_stats.get('processing', 0)
            metrics.append(MetricValue('video_queue_depth', queue_depth, timestamp))
            
            cursor.close()
            conn.close()
            
            # Update Prometheus metrics
            self.prometheus_metrics.active_connections.set(active_connections)
            self.prometheus_metrics.queue_depth.set(queue_depth)
            self.prometheus_metrics.video_queue_length.set(queue_stats.get('pending', 0))
            
        except Exception as e:
            logger.warning(f"Failed to collect database metrics: {e}")
        
        return metrics
    
    async def _collect_redis_metrics(self, timestamp: datetime) -> List[MetricValue]:
        """Collect Redis metrics"""
        metrics = []
        
        try:
            info = self.redis_client.info()
            
            # Basic Redis metrics
            metrics.extend([
                MetricValue('redis_connected_clients', info.get('connected_clients', 0), timestamp),
                MetricValue('redis_used_memory_bytes', info.get('used_memory', 0), timestamp, unit='bytes'),
                MetricValue('redis_used_memory_rss_bytes', info.get('used_memory_rss', 0), timestamp, unit='bytes'),
                MetricValue('redis_used_memory_peak_bytes', info.get('used_memory_peak', 0), timestamp, unit='bytes'),
                MetricValue('redis_total_commands_processed', info.get('total_commands_processed', 0), timestamp),
                MetricValue('redis_instantaneous_ops_per_sec', info.get('instantaneous_ops_per_sec', 0), timestamp),
                MetricValue('redis_keyspace_hits', info.get('keyspace_hits', 0), timestamp),
                MetricValue('redis_keyspace_misses', info.get('keyspace_misses', 0), timestamp),
                MetricValue('redis_expired_keys', info.get('expired_keys', 0), timestamp),
                MetricValue('redis_evicted_keys', info.get('evicted_keys', 0), timestamp)
            ])
            
            # Calculate hit ratio
            hits = info.get('keyspace_hits', 0)
            misses = info.get('keyspace_misses', 0)
            if hits + misses > 0:
                hit_ratio = hits / (hits + misses)
                metrics.append(MetricValue('redis_hit_ratio', hit_ratio, timestamp, unit='ratio'))
            
            # Memory fragmentation ratio
            used_memory = info.get('used_memory', 0)
            used_memory_rss = info.get('used_memory_rss', 0)
            if used_memory > 0:
                fragmentation_ratio = used_memory_rss / used_memory
                metrics.append(MetricValue('redis_fragmentation_ratio', fragmentation_ratio, timestamp, unit='ratio'))
            
            # Database keyspace info
            for db_key, db_info in info.items():
                if db_key.startswith('db'):
                    if isinstance(db_info, dict):
                        keys = db_info.get('keys', 0)
                        expires = db_info.get('expires', 0)
                        metrics.extend([
                            MetricValue('redis_db_keys', keys, timestamp, labels={'database': db_key}),
                            MetricValue('redis_db_expires', expires, timestamp, labels={'database': db_key})
                        ])
            
            # Update Prometheus metrics
            self.prometheus_metrics.redis_connected_clients.set(info.get('connected_clients', 0))
            self.prometheus_metrics.redis_memory_usage.set(info.get('used_memory', 0))
            self.prometheus_metrics.redis_operations_per_sec.set(info.get('instantaneous_ops_per_sec', 0))
            
        except Exception as e:
            logger.warning(f"Failed to collect Redis metrics: {e}")
        
        return metrics
    
    async def _collect_video_processing_metrics(self, timestamp: datetime) -> List[MetricValue]:
        """Collect video processing specific metrics"""
        metrics = []
        
        try:
            # Calculate processing rate from completed videos in last hour
            one_hour_ago = datetime.now() - timedelta(hours=1)
            
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT count(*) FROM video_queue 
                WHERE status = 'completed' 
                AND updated_at > %s
            """, (one_hour_ago,))
            
            completed_last_hour = cursor.fetchone()[0]
            processing_rate = completed_last_hour / 60  # videos per minute
            
            metrics.append(MetricValue('video_processing_rate_per_minute', processing_rate, timestamp, unit='videos/min'))
            
            # Average processing time
            cursor.execute("""
                SELECT AVG(EXTRACT(EPOCH FROM (updated_at - created_at))) as avg_time
                FROM video_queue 
                WHERE status = 'completed' 
                AND updated_at > %s
            """, (one_hour_ago,))
            
            result = cursor.fetchone()
            avg_processing_time = result[0] if result[0] is not None else 0
            metrics.append(MetricValue('video_avg_processing_time_seconds', avg_processing_time, timestamp, unit='seconds'))
            
            # Error rate
            cursor.execute("""
                SELECT 
                    count(*) FILTER (WHERE status = 'failed') as failed_count,
                    count(*) as total_count
                FROM video_queue 
                WHERE updated_at > %s
            """, (one_hour_ago,))
            
            failed_count, total_count = cursor.fetchone()
            error_rate = (failed_count / total_count) if total_count > 0 else 0
            metrics.append(MetricValue('video_error_rate', error_rate, timestamp, unit='ratio'))
            
            cursor.close()
            conn.close()
            
            # Update Prometheus metrics
            self.prometheus_metrics.processing_rate.set(processing_rate)
            self.prometheus_metrics.error_rate.set(error_rate)
            
        except Exception as e:
            logger.warning(f"Failed to collect video processing metrics: {e}")
        
        return metrics
    
    async def _get_gpu_metrics(self) -> List[MetricValue]:
        """Collect GPU metrics if available"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            import subprocess
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    values = line.split(', ')
                    if len(values) == 5:
                        gpu_util, mem_used, mem_total, temp, power = [float(v) for v in values]
                        
                        labels = {'gpu': str(i)}
                        metrics.extend([
                            MetricValue('gpu_utilization_percent', gpu_util, timestamp, labels, '%'),
                            MetricValue('gpu_memory_used_mb', mem_used, timestamp, labels, 'MB'),
                            MetricValue('gpu_memory_total_mb', mem_total, timestamp, labels, 'MB'),
                            MetricValue('gpu_memory_usage_percent', (mem_used/mem_total)*100, timestamp, labels, '%'),
                            MetricValue('gpu_temperature_celsius', temp, timestamp, labels, '°C'),
                            MetricValue('gpu_power_draw_watts', power, timestamp, labels, 'W')
                        ])
                
                # Update Prometheus with first GPU
                if metrics:
                    first_gpu_util = next(m.value for m in metrics if m.name == 'gpu_utilization_percent')
                    self.prometheus_metrics.gpu_utilization.set(first_gpu_util)
                        
        except (FileNotFoundError, subprocess.TimeoutExpired, ImportError):
            pass
        except Exception as e:
            logger.debug(f"GPU metrics collection failed: {e}")
        
        return metrics
    
    async def collect_prometheus_metrics(self) -> List[MetricValue]:
        """Collect metrics from Prometheus for trend analysis"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # Define queries for key metrics
            queries = {
                'cpu_usage_trend': 'rate(node_cpu_seconds_total[5m])',
                'memory_usage_trend': 'node_memory_MemAvailable_bytes',
                'request_rate_trend': 'rate(http_requests_total[5m])',
                'response_time_trend': 'histogram_quantile(0.95, http_request_duration_seconds_bucket)'
            }
            
            async with aiohttp.ClientSession() as session:
                for metric_name, query in queries.items():
                    try:
                        url = f"{self.prometheus_url}/api/v1/query"
                        params = {'query': query}
                        
                        async with session.get(url, params=params, timeout=5) as response:
                            if response.status == 200:
                                data = await response.json()
                                if data['status'] == 'success' and data['data']['result']:
                                    for result in data['data']['result']:
                                        value = float(result['value'][1])
                                        metrics.append(MetricValue(
                                            name=metric_name,
                                            value=value,
                                            timestamp=timestamp,
                                            labels=result.get('metric', {})
                                        ))
                    except Exception as e:
                        logger.debug(f"Failed to collect {metric_name}: {e}")
                        
        except Exception as e:
            logger.warning(f"Failed to collect Prometheus metrics: {e}")
        
        return metrics
    
    def add_to_buffer(self, metrics: List[MetricValue]):
        """Add metrics to buffer with automatic cleanup"""
        self.metrics_buffer.extend(metrics)
        
        # Cleanup old metrics from buffer
        if len(self.metrics_buffer) > self.max_buffer_size:
            self.metrics_buffer = self.metrics_buffer[-self.max_buffer_size:]
        
        # Update historical data for trend analysis
        for metric in metrics:
            if metric.name not in self.historical_data:
                self.historical_data[metric.name] = []
            
            self.historical_data[metric.name].append(metric.value)
            
            # Keep only recent history
            if len(self.historical_data[metric.name]) > self.max_history_points:
                self.historical_data[metric.name] = self.historical_data[metric.name][-self.max_history_points:]
    
    def get_trend_analysis(self, metric_name: str, window: int = 10) -> Dict[str, float]:
        """Calculate trend analysis for a metric"""
        if metric_name not in self.historical_data:
            return {}
        
        values = self.historical_data[metric_name][-window:]
        if len(values) < 3:
            return {}
        
        # Simple trend calculation
        trend = (values[-1] - values[0]) / len(values)
        avg = sum(values) / len(values)
        
        # Volatility (standard deviation)
        variance = sum((x - avg) ** 2 for x in values) / len(values)
        volatility = variance ** 0.5
        
        return {
            'trend': trend,
            'average': avg,
            'volatility': volatility,
            'current': values[-1],
            'min': min(values),
            'max': max(values)
        }
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot"""
        system_metrics = await self.collect_system_metrics()
        app_metrics = await self.collect_application_metrics()
        prometheus_metrics = await self.collect_prometheus_metrics()
        
        all_metrics = system_metrics + app_metrics + prometheus_metrics
        self.add_to_buffer(all_metrics)
        
        # Convert to dictionary format
        current_data = {}
        for metric in all_metrics:
            key = metric.name
            if metric.labels:
                label_str = ','.join(f"{k}={v}" for k, v in metric.labels.items())
                key = f"{metric.name}{{{label_str}}}"
            
            current_data[key] = {
                'value': metric.value,
                'timestamp': metric.timestamp.isoformat(),
                'unit': metric.unit,
                'description': metric.description
            }
        
        return current_data
    
    async def run_collection_loop(self):
        """Main collection loop"""
        logger.info("Starting metrics collection loop")
        
        while True:
            try:
                # Collect all metrics
                current_metrics = await self.get_current_metrics()
                
                logger.debug(f"Collected {len(current_metrics)} metrics")
                
                # Store in Redis for other services
                try:
                    self.redis_client.setex(
                        'ytempire:metrics:current',
                        300,  # 5 minutes TTL
                        json.dumps(current_metrics, default=str)
                    )
                except Exception as e:
                    logger.warning(f"Failed to store metrics in Redis: {e}")
                
                # Wait for next collection
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error

# FastAPI application for serving metrics
app = FastAPI(title="YTEmpire Metrics Collector", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global collector instance
collector = None

@app.on_event("startup")
async def startup_event():
    global collector
    collector = AdvancedMetricsCollector()
    
    # Start Prometheus metrics server
    start_http_server(9091, registry=collector.prometheus_metrics.registry)
    
    # Start collection loop in background
    asyncio.create_task(collector.run_collection_loop())

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/metrics")
async def get_metrics():
    """Get current metrics"""
    if not collector:
        raise HTTPException(status_code=503, detail="Collector not initialized")
    
    return await collector.get_current_metrics()

@app.get("/metrics/{metric_name}/trend")
async def get_metric_trend(metric_name: str, window: int = 10):
    """Get trend analysis for a specific metric"""
    if not collector:
        raise HTTPException(status_code=503, detail="Collector not initialized")
    
    trend = collector.get_trend_analysis(metric_name, window)
    if not trend:
        raise HTTPException(status_code=404, detail="Metric not found or insufficient data")
    
    return trend

@app.get("/metrics/history")
async def get_metrics_history():
    """Get historical metrics data"""
    if not collector:
        raise HTTPException(status_code=503, detail="Collector not initialized")
    
    return {
        name: values[-50:]  # Last 50 points
        for name, values in collector.historical_data.items()
    }

# CLI Interface
async def main():
    """Run the metrics collector"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YTEmpire Metrics Collector")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    parser.add_argument("--host", default="0.0.0.0", help="API server host")
    
    args = parser.parse_args()
    
    # Run FastAPI server
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())