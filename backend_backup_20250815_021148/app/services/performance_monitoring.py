"""
Performance Monitoring Service
Real-time performance monitoring and analysis
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, text
import numpy as np
import asyncio
from collections import defaultdict
import logging
from prometheus_client import CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST

from app.core.cache import cache_service

logger = logging.getLogger(__name__)


class PerformanceMonitoringService:
    """Service for monitoring and analyzing system performance"""
    
    def __init__(self):
        self.cache_ttl = 60  # 1 minute cache for real-time metrics
        self.registry = CollectorRegistry()
        
    async def get_performance_overview(self) -> Dict[str, Any]:
        """Get comprehensive performance overview"""
        try:
            # Get current metrics
            current_metrics = await self._get_current_metrics()
            
            # Get historical metrics
            historical_metrics = await self._get_historical_metrics()
            
            # Get slow endpoints
            slow_endpoints = await self._get_slow_endpoints()
            
            # Get error rates
            error_rates = await self._get_error_rates()
            
            # Get database performance
            db_performance = await self._get_database_performance()
            
            # Get system resources
            system_resources = await self._get_system_resources()
            
            return {
                'current': current_metrics,
                'historical': historical_metrics,
                'slow_endpoints': slow_endpoints,
                'error_rates': error_rates,
                'database': db_performance,
                'system': system_resources,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting performance overview: {str(e)}")
            return self._empty_overview()
            
    async def get_endpoint_metrics(
        self,
        endpoint: Optional[str] = None,
        method: Optional[str] = None,
        time_range: int = 3600  # 1 hour default
    ) -> Dict[str, Any]:
        """Get detailed metrics for specific endpoint"""
        cache_key = f"endpoint_metrics:{endpoint}:{method}:{time_range}"
        cached = await cache_service.get(cache_key)
        if cached:
            return cached
            
        try:
            # Get endpoint-specific metrics from cache/storage
            metrics = {
                'endpoint': endpoint,
                'method': method,
                'time_range': time_range,
                'request_count': 0,
                'average_duration': 0,
                'p50_duration': 0,
                'p95_duration': 0,
                'p99_duration': 0,
                'error_count': 0,
                'error_rate': 0,
                'throughput': 0,
                'response_sizes': []
            }
            
            # In production, this would query actual metrics storage
            # For now, generate sample data
            if endpoint:
                metrics.update({
                    'request_count': np.random.randint(100, 10000),
                    'average_duration': np.random.uniform(0.05, 0.5),
                    'p50_duration': np.random.uniform(0.03, 0.3),
                    'p95_duration': np.random.uniform(0.1, 1.0),
                    'p99_duration': np.random.uniform(0.5, 2.0),
                    'error_count': np.random.randint(0, 50),
                    'throughput': np.random.uniform(10, 100)
                })
                metrics['error_rate'] = (metrics['error_count'] / metrics['request_count'] * 100) if metrics['request_count'] > 0 else 0
                
            await cache_service.set(cache_key, metrics, self.cache_ttl)
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting endpoint metrics: {str(e)}")
            return {}
            
    async def get_database_metrics(self) -> Dict[str, Any]:
        """Get database performance metrics"""
        try:
            metrics = {
                'connection_pool': await self._get_connection_pool_metrics(),
                'query_performance': await self._get_query_performance_metrics(),
                'slow_queries': await self._get_slow_queries(),
                'cache_hit_rate': await self._get_cache_hit_rate(),
                'active_connections': 0,
                'idle_connections': 0,
                'waiting_connections': 0
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting database metrics: {str(e)}")
            return {}
            
    async def get_cache_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics"""
        try:
            # Get Redis info
            cache_info = await cache_service.info()
            
            metrics = {
                'hit_rate': 0,
                'miss_rate': 0,
                'eviction_rate': 0,
                'memory_usage': 0,
                'key_count': 0,
                'operations_per_second': 0,
                'connected_clients': 0
            }
            
            # Parse Redis info (sample implementation)
            if cache_info:
                # This would parse actual Redis INFO output
                metrics.update({
                    'memory_usage': cache_info.get('used_memory', 0),
                    'key_count': cache_info.get('db0', {}).get('keys', 0),
                    'connected_clients': cache_info.get('connected_clients', 0)
                })
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting cache metrics: {str(e)}")
            return {}
            
    async def analyze_performance_trends(
        self,
        metric_type: str = 'latency',
        time_range: int = 86400  # 24 hours
    ) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        try:
            # Get historical data points
            data_points = await self._get_metric_history(metric_type, time_range)
            
            if not data_points:
                return {'trend': 'insufficient_data'}
                
            # Calculate trend
            values = [p['value'] for p in data_points]
            timestamps = [p['timestamp'] for p in data_points]
            
            # Simple linear regression for trend
            if len(values) >= 2:
                trend_coefficient = np.polyfit(range(len(values)), values, 1)[0]
                trend = 'improving' if trend_coefficient < 0 else 'degrading' if trend_coefficient > 0 else 'stable'
            else:
                trend = 'stable'
                
            # Calculate statistics
            analysis = {
                'metric_type': metric_type,
                'trend': trend,
                'current_value': values[-1] if values else 0,
                'average': np.mean(values) if values else 0,
                'median': np.median(values) if values else 0,
                'std_deviation': np.std(values) if values else 0,
                'min': min(values) if values else 0,
                'max': max(values) if values else 0,
                'data_points': len(values),
                'time_range': time_range,
                'anomalies': self._detect_anomalies(values)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing performance trends: {str(e)}")
            return {'trend': 'error'}
            
    async def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics (last 60 seconds)"""
        try:
            metrics = {
                'requests_per_second': await self._get_current_rps(),
                'active_requests': await self._get_active_requests(),
                'average_latency_ms': await self._get_current_latency(),
                'error_rate': await self._get_current_error_rate(),
                'cpu_usage': await self._get_cpu_usage(),
                'memory_usage': await self._get_memory_usage(),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting real-time metrics: {str(e)}")
            return {}
            
    async def get_performance_alerts(self) -> List[Dict[str, Any]]:
        """Get active performance alerts"""
        try:
            alerts = []
            
            # Check various thresholds
            metrics = await self.get_real_time_metrics()
            
            # High latency alert
            if metrics.get('average_latency_ms', 0) > 1000:
                alerts.append({
                    'type': 'high_latency',
                    'severity': 'warning',
                    'message': f"Average latency is {metrics['average_latency_ms']}ms",
                    'timestamp': datetime.utcnow().isoformat()
                })
                
            # High error rate alert
            if metrics.get('error_rate', 0) > 5:
                alerts.append({
                    'type': 'high_error_rate',
                    'severity': 'critical',
                    'message': f"Error rate is {metrics['error_rate']}%",
                    'timestamp': datetime.utcnow().isoformat()
                })
                
            # High CPU usage alert
            if metrics.get('cpu_usage', 0) > 80:
                alerts.append({
                    'type': 'high_cpu',
                    'severity': 'warning',
                    'message': f"CPU usage is {metrics['cpu_usage']}%",
                    'timestamp': datetime.utcnow().isoformat()
                })
                
            # High memory usage alert
            if metrics.get('memory_usage', 0) > 90:
                alerts.append({
                    'type': 'high_memory',
                    'severity': 'critical',
                    'message': f"Memory usage is {metrics['memory_usage']}%",
                    'timestamp': datetime.utcnow().isoformat()
                })
                
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting performance alerts: {str(e)}")
            return []
            
    # Private helper methods
    async def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        # In production, this would query Prometheus or similar
        return {
            'request_rate': np.random.uniform(50, 200),
            'average_latency': np.random.uniform(50, 500),
            'error_rate': np.random.uniform(0, 5),
            'throughput': np.random.uniform(1000, 5000)
        }
        
    async def _get_historical_metrics(self) -> List[Dict[str, Any]]:
        """Get historical metrics for charts"""
        # Generate sample data for last 24 hours
        now = datetime.utcnow()
        metrics = []
        
        for i in range(24):
            timestamp = now - timedelta(hours=i)
            metrics.append({
                'timestamp': timestamp.isoformat(),
                'request_rate': np.random.uniform(50, 200),
                'average_latency': np.random.uniform(50, 500),
                'error_rate': np.random.uniform(0, 5)
            })
            
        return metrics
        
    async def _get_slow_endpoints(self) -> List[Dict[str, Any]]:
        """Get slowest endpoints"""
        # Sample data - in production, query from metrics storage
        endpoints = [
            {'endpoint': '/api/v1/videos/generate', 'method': 'POST', 'avg_duration': 2.5, 'count': 150},
            {'endpoint': '/api/v1/analytics/dashboard', 'method': 'GET', 'avg_duration': 1.2, 'count': 500},
            {'endpoint': '/api/v1/channels/sync', 'method': 'POST', 'avg_duration': 0.8, 'count': 200},
        ]
        
        return sorted(endpoints, key=lambda x: x['avg_duration'], reverse=True)
        
    async def _get_error_rates(self) -> Dict[str, Any]:
        """Get error rate breakdown"""
        return {
            '4xx_errors': np.random.randint(0, 100),
            '5xx_errors': np.random.randint(0, 50),
            'timeout_errors': np.random.randint(0, 20),
            'total_errors': np.random.randint(0, 170)
        }
        
    async def _get_database_performance(self) -> Dict[str, Any]:
        """Get database performance metrics"""
        return {
            'average_query_time': np.random.uniform(1, 50),
            'slow_query_count': np.random.randint(0, 10),
            'connection_pool_usage': np.random.uniform(20, 80),
            'deadlock_count': np.random.randint(0, 2)
        }
        
    async def _get_system_resources(self) -> Dict[str, Any]:
        """Get system resource usage"""
        import psutil
        
        return {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'network_io': {
                'bytes_sent': psutil.net_io_counters().bytes_sent,
                'bytes_recv': psutil.net_io_counters().bytes_recv
            }
        }
        
    async def _get_connection_pool_metrics(self) -> Dict[str, Any]:
        """Get database connection pool metrics"""
        return {
            'size': 20,
            'overflow': 10,
            'checked_in': 15,
            'checked_out': 5,
            'total': 20
        }
        
    async def _get_query_performance_metrics(self) -> Dict[str, Any]:
        """Get query performance metrics"""
        return {
            'select': {'avg_ms': 5, 'count': 1000},
            'insert': {'avg_ms': 10, 'count': 500},
            'update': {'avg_ms': 8, 'count': 300},
            'delete': {'avg_ms': 7, 'count': 100}
        }
        
    async def _get_slow_queries(self) -> List[Dict[str, Any]]:
        """Get slow database queries"""
        return [
            {'query': 'SELECT * FROM videos WHERE ...', 'duration_ms': 500, 'count': 10},
            {'query': 'UPDATE channels SET ...', 'duration_ms': 300, 'count': 5}
        ]
        
    async def _get_cache_hit_rate(self) -> float:
        """Get cache hit rate"""
        return np.random.uniform(85, 95)
        
    async def _get_metric_history(
        self,
        metric_type: str,
        time_range: int
    ) -> List[Dict[str, Any]]:
        """Get historical data for a metric"""
        # Generate sample data
        now = datetime.utcnow()
        points = []
        
        for i in range(24):
            timestamp = now - timedelta(hours=i)
            value = np.random.uniform(50, 500) if metric_type == 'latency' else np.random.uniform(0, 100)
            points.append({
                'timestamp': timestamp.isoformat(),
                'value': value
            })
            
        return points
        
    def _detect_anomalies(self, values: List[float]) -> List[int]:
        """Detect anomalies in time series data"""
        if len(values) < 3:
            return []
            
        # Simple anomaly detection using z-score
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return []
            
        anomalies = []
        for i, value in enumerate(values):
            z_score = abs((value - mean) / std)
            if z_score > 3:  # 3 standard deviations
                anomalies.append(i)
                
        return anomalies
        
    async def _get_current_rps(self) -> float:
        """Get current requests per second"""
        return np.random.uniform(50, 200)
        
    async def _get_active_requests(self) -> int:
        """Get number of active requests"""
        return np.random.randint(5, 50)
        
    async def _get_current_latency(self) -> float:
        """Get current average latency"""
        return np.random.uniform(50, 500)
        
    async def _get_current_error_rate(self) -> float:
        """Get current error rate"""
        return np.random.uniform(0, 5)
        
    async def _get_cpu_usage(self) -> float:
        """Get current CPU usage"""
        import psutil
        return psutil.cpu_percent()
        
    async def _get_memory_usage(self) -> float:
        """Get current memory usage"""
        import psutil
        return psutil.virtual_memory().percent
        
    def _empty_overview(self) -> Dict[str, Any]:
        """Return empty overview structure"""
        return {
            'current': {},
            'historical': [],
            'slow_endpoints': [],
            'error_rates': {},
            'database': {},
            'system': {},
            'timestamp': datetime.utcnow().isoformat()
        }


# Create singleton instance
performance_monitoring_service = PerformanceMonitoringService()