#!/usr/bin/env python3
"""
Health Check and System Monitoring
Comprehensive health monitoring for disaster recovery preparedness
"""

import asyncio
import aiohttp
import psycopg2
from redis import Redis
import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import boto3
import subprocess

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class HealthCheckResult:
    component: str
    status: HealthStatus
    response_time_ms: float
    timestamp: datetime
    details: Dict[str, Any]
    error_message: Optional[str] = None

@dataclass
class SystemHealth:
    overall_status: HealthStatus
    timestamp: datetime
    components: List[HealthCheckResult]
    summary: Dict[str, Any]

class DatabaseHealthChecker:
    """PostgreSQL database health checker"""
    
    def __init__(self):
        self.db_config = {
            'host': os.getenv('DATABASE_HOST', 'localhost'),
            'port': int(os.getenv('DATABASE_PORT', '5432')),
            'database': os.getenv('DATABASE_NAME', 'ytempire'),
            'user': os.getenv('DATABASE_USER', 'postgres'),
            'password': os.getenv('DATABASE_PASSWORD', 'password')
        }
    
    async def check_health(self) -> HealthCheckResult:
        """Check database connectivity and performance"""
        start_time = time.time()
        
        try:
            # Test basic connectivity
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Check database version and basic query
            cursor.execute("SELECT version(), current_database(), current_user, now()")
            db_info = cursor.fetchone()
            
            # Check active connections
            cursor.execute("""
                SELECT count(*) as active_connections,
                       max(extract(epoch from now() - state_change)) as longest_idle_time
                FROM pg_stat_activity 
                WHERE state = 'active'
            """)
            connection_stats = cursor.fetchone()
            
            # Check database size
            cursor.execute("""
                SELECT pg_size_pretty(pg_database_size(current_database())) as db_size,
                       pg_database_size(current_database()) as db_size_bytes
            """)
            size_info = cursor.fetchone()
            
            # Check replication lag (if applicable)
            cursor.execute("SELECT pg_is_in_recovery()")
            is_replica = cursor.fetchone()[0]
            
            replication_lag = None
            if is_replica:
                cursor.execute("""
                    SELECT CASE 
                        WHEN pg_last_wal_receive_lsn() = pg_last_wal_replay_lsn() 
                        THEN 0 
                        ELSE EXTRACT(EPOCH FROM now() - pg_last_xact_replay_timestamp()) 
                    END as lag_seconds
                """)
                replication_lag = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            response_time = (time.time() - start_time) * 1000
            
            # Determine health status
            status = HealthStatus.HEALTHY
            if response_time > 1000:  # > 1 second
                status = HealthStatus.DEGRADED
            if connection_stats[0] > 90:  # > 90 active connections
                status = HealthStatus.DEGRADED
            if replication_lag and replication_lag > 60:  # > 1 minute lag
                status = HealthStatus.DEGRADED
            
            return HealthCheckResult(
                component="database",
                status=status,
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details={
                    "version": db_info[0].split(',')[0] if db_info[0] else "unknown",
                    "database": db_info[1],
                    "user": db_info[2],
                    "active_connections": connection_stats[0],
                    "longest_idle_time": connection_stats[1],
                    "database_size": size_info[0],
                    "database_size_bytes": size_info[1],
                    "is_replica": is_replica,
                    "replication_lag_seconds": replication_lag
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"Database health check failed: {e}")
            
            return HealthCheckResult(
                component="database",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details={},
                error_message=str(e)
            )

class RedisHealthChecker:
    """Redis health checker"""
    
    def __init__(self):
        self.redis_config = {
            'host': os.getenv('REDIS_HOST', 'localhost'),
            'port': int(os.getenv('REDIS_PORT', '6379')),
            'password': os.getenv('REDIS_PASSWORD') or None
        }
    
    async def check_health(self) -> HealthCheckResult:
        """Check Redis connectivity and performance"""
        start_time = time.time()
        
        try:
            redis_client = Redis(**self.redis_config, decode_responses=True)
            
            # Test basic connectivity
            pong = redis_client.ping()
            if not pong:
                raise Exception("Redis ping failed")
            
            # Get Redis info
            redis_info = redis_client.info()
            
            # Test read/write performance
            test_key = "health_check_test"
            test_value = f"test_{int(time.time())}"
            
            redis_client.set(test_key, test_value, ex=60)
            retrieved_value = redis_client.get(test_key)
            redis_client.delete(test_key)
            
            if retrieved_value != test_value:
                raise Exception("Redis read/write test failed")
            
            response_time = (time.time() - start_time) * 1000
            
            # Determine health status
            status = HealthStatus.HEALTHY
            memory_usage_ratio = redis_info.get('used_memory', 0) / redis_info.get('total_system_memory', 1)
            
            if response_time > 500:  # > 500ms
                status = HealthStatus.DEGRADED
            if memory_usage_ratio > 0.8:  # > 80% memory usage
                status = HealthStatus.DEGRADED
            if redis_info.get('connected_clients', 0) > 100:  # > 100 clients
                status = HealthStatus.DEGRADED
            
            return HealthCheckResult(
                component="redis",
                status=status,
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details={
                    "version": redis_info.get('redis_version'),
                    "uptime_seconds": redis_info.get('uptime_in_seconds'),
                    "connected_clients": redis_info.get('connected_clients'),
                    "used_memory": redis_info.get('used_memory'),
                    "used_memory_human": redis_info.get('used_memory_human'),
                    "memory_usage_ratio": round(memory_usage_ratio, 3),
                    "total_commands_processed": redis_info.get('total_commands_processed'),
                    "keyspace_hits": redis_info.get('keyspace_hits'),
                    "keyspace_misses": redis_info.get('keyspace_misses')
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"Redis health check failed: {e}")
            
            return HealthCheckResult(
                component="redis",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details={},
                error_message=str(e)
            )

class APIHealthChecker:
    """FastAPI backend health checker"""
    
    def __init__(self):
        self.api_base_url = os.getenv('API_BASE_URL', 'http://localhost:8000')
    
    async def check_health(self) -> HealthCheckResult:
        """Check API health endpoints"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                # Check main health endpoint
                async with session.get(f"{self.api_base_url}/health") as response:
                    health_data = await response.json()
                    
                    if response.status != 200:
                        raise Exception(f"API health endpoint returned {response.status}")
                
                # Check metrics endpoint
                async with session.get(f"{self.api_base_url}/metrics") as response:
                    if response.status != 200:
                        raise Exception(f"Metrics endpoint returned {response.status}")
                
                response_time = (time.time() - start_time) * 1000
                
                # Determine health status based on response time and health data
                status = HealthStatus.HEALTHY
                if response_time > 1000:  # > 1 second
                    status = HealthStatus.DEGRADED
                
                # Check if any dependent services are unhealthy
                if health_data.get('database', {}).get('status') != 'healthy':
                    status = HealthStatus.DEGRADED
                if health_data.get('redis', {}).get('status') != 'healthy':
                    status = HealthStatus.DEGRADED
                
                return HealthCheckResult(
                    component="api",
                    status=status,
                    response_time_ms=response_time,
                    timestamp=datetime.now(),
                    details=health_data
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"API health check failed: {e}")
            
            return HealthCheckResult(
                component="api",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details={},
                error_message=str(e)
            )

class StorageHealthChecker:
    """S3 and local storage health checker"""
    
    def __init__(self):
        self.s3_bucket = os.getenv('BACKUP_S3_BUCKET', 'ytempire-backups')
        self.local_paths = [
            '/app/uploads',
            '/app/generated_videos',
            '/var/backups/ytempire'
        ]
    
    async def check_health(self) -> HealthCheckResult:
        """Check storage systems health"""
        start_time = time.time()
        
        try:
            details = {}
            
            # Check local storage
            local_storage = {}
            for path in self.local_paths:
                if os.path.exists(path):
                    statvfs = os.statvfs(path)
                    free_bytes = statvfs.f_frsize * statvfs.f_bavail
                    total_bytes = statvfs.f_frsize * statvfs.f_blocks
                    used_bytes = total_bytes - free_bytes
                    usage_ratio = used_bytes / total_bytes if total_bytes > 0 else 0
                    
                    local_storage[path] = {
                        'free_bytes': free_bytes,
                        'total_bytes': total_bytes,
                        'used_bytes': used_bytes,
                        'usage_ratio': round(usage_ratio, 3),
                        'accessible': os.access(path, os.R_OK | os.W_OK)
                    }
                else:
                    local_storage[path] = {'exists': False}
            
            details['local_storage'] = local_storage
            
            # Check S3 connectivity
            try:
                s3_client = boto3.client('s3')
                
                # Test bucket access
                s3_client.head_bucket(Bucket=self.s3_bucket)
                
                # Test upload/download
                test_key = f"health-check/{int(time.time())}.txt"
                test_content = f"Health check at {datetime.now().isoformat()}"
                
                s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=test_key,
                    Body=test_content.encode()
                )
                
                # Verify the upload
                response = s3_client.get_object(Bucket=self.s3_bucket, Key=test_key)
                retrieved_content = response['Body'].read().decode()
                
                # Cleanup test object
                s3_client.delete_object(Bucket=self.s3_bucket, Key=test_key)
                
                if retrieved_content != test_content:
                    raise Exception("S3 read/write verification failed")
                
                details['s3'] = {
                    'accessible': True,
                    'bucket': self.s3_bucket,
                    'test_successful': True
                }
                
            except Exception as s3_error:
                details['s3'] = {
                    'accessible': False,
                    'error': str(s3_error)
                }
            
            response_time = (time.time() - start_time) * 1000
            
            # Determine overall storage health
            status = HealthStatus.HEALTHY
            
            # Check local storage usage
            for path_info in local_storage.values():
                if path_info.get('usage_ratio', 0) > 0.9:  # > 90% full
                    status = HealthStatus.DEGRADED
                if not path_info.get('accessible', True):
                    status = HealthStatus.DEGRADED
            
            # Check S3 accessibility
            if not details['s3'].get('accessible', False):
                status = HealthStatus.DEGRADED
            
            return HealthCheckResult(
                component="storage",
                status=status,
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details=details
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"Storage health check failed: {e}")
            
            return HealthCheckResult(
                component="storage",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details={},
                error_message=str(e)
            )

class SystemResourceChecker:
    """System resource health checker"""
    
    async def check_health(self) -> HealthCheckResult:
        """Check system resources (CPU, memory, disk)"""
        start_time = time.time()
        
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
            
            # Memory usage
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk usage
            disk_usage = {}
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_usage[partition.mountpoint] = {
                        'total': usage.total,
                        'used': usage.used,
                        'free': usage.free,
                        'usage_percent': round(usage.used / usage.total * 100, 2)
                    }
                except PermissionError:
                    continue
            
            # Network stats
            network = psutil.net_io_counters()
            
            # Process count
            process_count = len(psutil.pids())
            
            response_time = (time.time() - start_time) * 1000
            
            details = {
                'cpu': {
                    'usage_percent': cpu_percent,
                    'count': cpu_count,
                    'load_avg_1m': load_avg[0],
                    'load_avg_5m': load_avg[1],
                    'load_avg_15m': load_avg[2]
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'used': memory.used,
                    'usage_percent': memory.percent
                },
                'swap': {
                    'total': swap.total,
                    'used': swap.used,
                    'usage_percent': swap.percent
                },
                'disk': disk_usage,
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                },
                'processes': process_count
            }
            
            # Determine health status
            status = HealthStatus.HEALTHY
            
            if cpu_percent > 80:  # > 80% CPU
                status = HealthStatus.DEGRADED
            if memory.percent > 85:  # > 85% memory
                status = HealthStatus.DEGRADED
            if any(usage['usage_percent'] > 90 for usage in disk_usage.values()):  # > 90% disk
                status = HealthStatus.DEGRADED
            if load_avg[0] > cpu_count * 2:  # Load average > 2x CPU count
                status = HealthStatus.DEGRADED
            
            return HealthCheckResult(
                component="system_resources",
                status=status,
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details=details
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"System resources health check failed: {e}")
            
            return HealthCheckResult(
                component="system_resources",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details={},
                error_message=str(e)
            )

class ComprehensiveHealthMonitor:
    """Main health monitoring system"""
    
    def __init__(self):
        self.checkers = {
            'database': DatabaseHealthChecker(),
            'redis': RedisHealthChecker(),
            'api': APIHealthChecker(),
            'storage': StorageHealthChecker(),
            'system_resources': SystemResourceChecker()
        }
    
    async def run_all_checks(self) -> SystemHealth:
        """Run all health checks and return comprehensive system health"""
        results = []
        
        # Run all health checks concurrently
        tasks = []
        for name, checker in self.checkers.items():
            tasks.append(checker.check_health())
        
        check_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(check_results):
            if isinstance(result, Exception):
                # Handle checker that failed
                component_name = list(self.checkers.keys())[i]
                results.append(HealthCheckResult(
                    component=component_name,
                    status=HealthStatus.UNKNOWN,
                    response_time_ms=0,
                    timestamp=datetime.now(),
                    details={},
                    error_message=str(result)
                ))
            else:
                results.append(result)
        
        # Determine overall system health
        overall_status = self._determine_overall_status(results)
        
        # Generate summary
        summary = self._generate_summary(results)
        
        return SystemHealth(
            overall_status=overall_status,
            timestamp=datetime.now(),
            components=results,
            summary=summary
        )
    
    def _determine_overall_status(self, results: List[HealthCheckResult]) -> HealthStatus:
        """Determine overall system health status"""
        statuses = [result.status for result in results]
        
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif HealthStatus.UNKNOWN in statuses:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def _generate_summary(self, results: List[HealthCheckResult]) -> Dict[str, Any]:
        """Generate health summary"""
        status_counts = {}
        total_response_time = 0
        error_count = 0
        
        for result in results:
            status_counts[result.status.value] = status_counts.get(result.status.value, 0) + 1
            total_response_time += result.response_time_ms
            if result.error_message:
                error_count += 1
        
        return {
            'total_components': len(results),
            'status_distribution': status_counts,
            'average_response_time_ms': round(total_response_time / len(results), 2) if results else 0,
            'error_count': error_count,
            'check_timestamp': datetime.now().isoformat()
        }
    
    async def export_health_report(self, filepath: str):
        """Export comprehensive health report to JSON"""
        health = await self.run_all_checks()
        
        # Convert to serializable format
        report = {
            'overall_status': health.overall_status.value,
            'timestamp': health.timestamp.isoformat(),
            'summary': health.summary,
            'components': []
        }
        
        for component in health.components:
            report['components'].append({
                'component': component.component,
                'status': component.status.value,
                'response_time_ms': component.response_time_ms,
                'timestamp': component.timestamp.isoformat(),
                'details': component.details,
                'error_message': component.error_message
            })
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report

# CLI Interface
async def main():
    """Command-line interface for health monitoring"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YTEmpire System Health Monitor")
    parser.add_argument("--output", "-o", help="Output file for health report")
    parser.add_argument("--component", "-c", help="Check specific component only")
    parser.add_argument("--watch", "-w", action="store_true", help="Continuous monitoring")
    parser.add_argument("--interval", "-i", type=int, default=60, help="Watch interval in seconds")
    
    args = parser.parse_args()
    
    monitor = ComprehensiveHealthMonitor()
    
    if args.watch:
        print("Starting continuous health monitoring...")
        while True:
            try:
                health = await monitor.run_all_checks()
                print(f"\n[{health.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"Overall Status: {health.overall_status.value.upper()}")
                
                for component in health.components:
                    status_color = {
                        HealthStatus.HEALTHY: "✅",
                        HealthStatus.DEGRADED: "⚠️",
                        HealthStatus.UNHEALTHY: "❌",
                        HealthStatus.UNKNOWN: "❓"
                    }[component.status]
                    
                    print(f"  {status_color} {component.component}: {component.status.value} "
                          f"({component.response_time_ms:.1f}ms)")
                    
                    if component.error_message:
                        print(f"    Error: {component.error_message}")
                
                await asyncio.sleep(args.interval)
                
            except KeyboardInterrupt:
                print("\nMonitoring stopped.")
                break
            except Exception as e:
                print(f"Monitoring error: {e}")
                await asyncio.sleep(args.interval)
    
    else:
        # Single health check
        health = await monitor.run_all_checks()
        
        if args.output:
            await monitor.export_health_report(args.output)
            print(f"Health report saved to: {args.output}")
        else:
            print(f"Overall System Health: {health.overall_status.value.upper()}")
            print(f"Timestamp: {health.timestamp}")
            print(f"Components checked: {health.summary['total_components']}")
            print(f"Average response time: {health.summary['average_response_time_ms']:.1f}ms")
            
            if health.summary['error_count'] > 0:
                print(f"Errors detected: {health.summary['error_count']}")

if __name__ == "__main__":
    asyncio.run(main())