#!/usr/bin/env python3
"""
YTEmpire Health Checker
Comprehensive health monitoring for all services with alerting
"""

import os
import sys
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import requests
import psycopg2
import redis
import aiohttp
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceHealth:
    """Service health data"""
    name: str
    status: HealthStatus
    response_time: float
    error_message: Optional[str] = None
    metadata: Optional[Dict] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class HealthChecker:
    """Main health checking orchestrator"""
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.config = self._load_config()
        self.registry = CollectorRegistry()
        self._setup_metrics()
        self.alert_manager = AlertManager(self.config.get("alerts", {}))
        
    def _load_config(self) -> Dict:
        """Load health check configuration"""
        return {
            "services": {
                "backend": {
                    "url": os.getenv("BACKEND_URL", "http://localhost:8000"),
                    "endpoints": [
                        "/api/v1/health",
                        "/api/v1/health/ready",
                        "/api/v1/health/live"
                    ],
                    "timeout": 10,
                    "threshold_ms": 1000
                },
                "frontend": {
                    "url": os.getenv("FRONTEND_URL", "http://localhost:3000"),
                    "endpoints": ["/", "/health"],
                    "timeout": 10,
                    "threshold_ms": 500
                },
                "postgres": {
                    "host": os.getenv("POSTGRES_HOST", "localhost"),
                    "port": int(os.getenv("POSTGRES_PORT", "5432")),
                    "database": os.getenv("POSTGRES_DB", "ytempire_db"),
                    "user": os.getenv("POSTGRES_USER", "ytempire"),
                    "password": os.getenv("POSTGRES_PASSWORD", ""),
                    "timeout": 5
                },
                "redis": {
                    "host": os.getenv("REDIS_HOST", "localhost"),
                    "port": int(os.getenv("REDIS_PORT", "6379")),
                    "db": 0,
                    "timeout": 5
                },
                "celery": {
                    "broker_url": os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
                    "timeout": 10
                }
            },
            "prometheus": {
                "gateway": os.getenv("PROMETHEUS_GATEWAY", "localhost:9091"),
                "job": f"health_check_{self.environment}"
            },
            "alerts": {
                "slack_webhook": os.getenv("SLACK_WEBHOOK_URL"),
                "email_recipients": os.getenv("ALERT_EMAIL_RECIPIENTS", "").split(","),
                "pagerduty_key": os.getenv("PAGERDUTY_KEY"),
                "thresholds": {
                    "error_rate": 0.05,  # 5% error rate
                    "response_time_p95": 1000,  # 1 second
                    "consecutive_failures": 3
                }
            }
        }
    
    def _setup_metrics(self):
        """Setup Prometheus metrics"""
        self.metrics = {
            "health_status": Gauge(
                "service_health_status",
                "Service health status (1=healthy, 0=unhealthy)",
                ["service", "environment"],
                registry=self.registry
            ),
            "response_time": Gauge(
                "service_response_time_ms",
                "Service response time in milliseconds",
                ["service", "environment", "endpoint"],
                registry=self.registry
            ),
            "error_rate": Gauge(
                "service_error_rate",
                "Service error rate",
                ["service", "environment"],
                registry=self.registry
            )
        }
    
    async def check_all_services(self) -> Dict[str, ServiceHealth]:
        """Check health of all services"""
        results = {}
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._check_backend(session),
                self._check_frontend(session),
                self._check_postgres(),
                self._check_redis(),
                self._check_celery()
            ]
            
            health_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in health_results:
                if isinstance(result, ServiceHealth):
                    results[result.name] = result
                    self._update_metrics(result)
                elif isinstance(result, Exception):
                    logger.error(f"Health check error: {result}")
        
        # Push metrics to Prometheus
        self._push_metrics()
        
        # Check for alerts
        self._check_alerts(results)
        
        return results
    
    async def _check_backend(self, session: aiohttp.ClientSession) -> ServiceHealth:
        """Check backend API health"""
        config = self.config["services"]["backend"]
        base_url = config["url"]
        
        total_time = 0
        errors = []
        
        for endpoint in config["endpoints"]:
            url = f"{base_url}{endpoint}"
            start_time = time.time()
            
            try:
                async with session.get(url, timeout=config["timeout"]) as response:
                    response_time = (time.time() - start_time) * 1000
                    total_time += response_time
                    
                    if response.status != 200:
                        errors.append(f"{endpoint}: status {response.status}")
            except asyncio.TimeoutError:
                errors.append(f"{endpoint}: timeout")
            except Exception as e:
                errors.append(f"{endpoint}: {str(e)}")
        
        avg_time = total_time / len(config["endpoints"]) if config["endpoints"] else 0
        
        if errors:
            status = HealthStatus.UNHEALTHY if len(errors) == len(config["endpoints"]) else HealthStatus.DEGRADED
            error_msg = "; ".join(errors)
        else:
            status = HealthStatus.HEALTHY if avg_time < config["threshold_ms"] else HealthStatus.DEGRADED
            error_msg = None
        
        return ServiceHealth(
            name="backend",
            status=status,
            response_time=avg_time,
            error_message=error_msg,
            metadata={"endpoints_checked": len(config["endpoints"])}
        )
    
    async def _check_frontend(self, session: aiohttp.ClientSession) -> ServiceHealth:
        """Check frontend health"""
        config = self.config["services"]["frontend"]
        base_url = config["url"]
        
        start_time = time.time()
        
        try:
            async with session.get(base_url, timeout=config["timeout"]) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    # Check if HTML contains expected content
                    text = await response.text()
                    if "<div id=\"root\">" in text or "<!DOCTYPE html>" in text:
                        status = HealthStatus.HEALTHY
                        error_msg = None
                    else:
                        status = HealthStatus.DEGRADED
                        error_msg = "Unexpected content"
                else:
                    status = HealthStatus.UNHEALTHY
                    error_msg = f"HTTP {response.status}"
                    
        except asyncio.TimeoutError:
            status = HealthStatus.UNHEALTHY
            response_time = config["timeout"] * 1000
            error_msg = "Timeout"
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            response_time = 0
            error_msg = str(e)
        
        return ServiceHealth(
            name="frontend",
            status=status,
            response_time=response_time,
            error_message=error_msg
        )
    
    async def _check_postgres(self) -> ServiceHealth:
        """Check PostgreSQL health"""
        config = self.config["services"]["postgres"]
        start_time = time.time()
        
        try:
            conn = psycopg2.connect(
                host=config["host"],
                port=config["port"],
                database=config["database"],
                user=config["user"],
                password=config["password"],
                connect_timeout=config["timeout"]
            )
            
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            
            # Check replication lag if applicable
            cursor.execute("""
                SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp()))::int as lag
                WHERE pg_is_in_recovery()
            """)
            result = cursor.fetchone()
            replication_lag = result[0] if result and result[0] else 0
            
            cursor.close()
            conn.close()
            
            response_time = (time.time() - start_time) * 1000
            
            if replication_lag > 10:  # More than 10 seconds lag
                status = HealthStatus.DEGRADED
                error_msg = f"Replication lag: {replication_lag}s"
            else:
                status = HealthStatus.HEALTHY
                error_msg = None
            
            metadata = {
                "replication_lag_seconds": replication_lag,
                "connection_time_ms": response_time
            }
            
        except psycopg2.OperationalError as e:
            status = HealthStatus.UNHEALTHY
            response_time = (time.time() - start_time) * 1000
            error_msg = str(e)
            metadata = None
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            response_time = 0
            error_msg = str(e)
            metadata = None
        
        return ServiceHealth(
            name="postgres",
            status=status,
            response_time=response_time,
            error_message=error_msg,
            metadata=metadata
        )
    
    async def _check_redis(self) -> ServiceHealth:
        """Check Redis health"""
        config = self.config["services"]["redis"]
        start_time = time.time()
        
        try:
            r = redis.Redis(
                host=config["host"],
                port=config["port"],
                db=config["db"],
                socket_connect_timeout=config["timeout"],
                socket_timeout=config["timeout"]
            )
            
            # Ping Redis
            r.ping()
            
            # Get memory info
            info = r.info("memory")
            used_memory_mb = info.get("used_memory", 0) / 1024 / 1024
            max_memory_mb = info.get("maxmemory", 0) / 1024 / 1024
            
            response_time = (time.time() - start_time) * 1000
            
            # Check memory usage
            if max_memory_mb > 0 and used_memory_mb / max_memory_mb > 0.9:
                status = HealthStatus.DEGRADED
                error_msg = f"High memory usage: {used_memory_mb:.1f}/{max_memory_mb:.1f} MB"
            else:
                status = HealthStatus.HEALTHY
                error_msg = None
            
            metadata = {
                "used_memory_mb": used_memory_mb,
                "max_memory_mb": max_memory_mb,
                "connected_clients": info.get("connected_clients", 0)
            }
            
        except redis.ConnectionError as e:
            status = HealthStatus.UNHEALTHY
            response_time = (time.time() - start_time) * 1000
            error_msg = str(e)
            metadata = None
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            response_time = 0
            error_msg = str(e)
            metadata = None
        
        return ServiceHealth(
            name="redis",
            status=status,
            response_time=response_time,
            error_message=error_msg,
            metadata=metadata
        )
    
    async def _check_celery(self) -> ServiceHealth:
        """Check Celery workers health"""
        config = self.config["services"]["celery"]
        start_time = time.time()
        
        try:
            from celery import Celery
            
            app = Celery(broker=config["broker_url"])
            
            # Get worker stats
            stats = app.control.inspect().stats()
            active_tasks = app.control.inspect().active()
            
            response_time = (time.time() - start_time) * 1000
            
            if not stats:
                status = HealthStatus.UNHEALTHY
                error_msg = "No workers available"
                metadata = None
            else:
                worker_count = len(stats)
                total_tasks = sum(len(tasks) for tasks in (active_tasks or {}).values())
                
                status = HealthStatus.HEALTHY
                error_msg = None
                metadata = {
                    "worker_count": worker_count,
                    "active_tasks": total_tasks
                }
            
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            response_time = (time.time() - start_time) * 1000
            error_msg = str(e)
            metadata = None
        
        return ServiceHealth(
            name="celery",
            status=status,
            response_time=response_time,
            error_message=error_msg,
            metadata=metadata
        )
    
    def _update_metrics(self, health: ServiceHealth):
        """Update Prometheus metrics"""
        # Update health status
        self.metrics["health_status"].labels(
            service=health.name,
            environment=self.environment
        ).set(1 if health.status == HealthStatus.HEALTHY else 0)
        
        # Update response time
        self.metrics["response_time"].labels(
            service=health.name,
            environment=self.environment,
            endpoint="main"
        ).set(health.response_time)
    
    def _push_metrics(self):
        """Push metrics to Prometheus gateway"""
        if self.config["prometheus"]["gateway"]:
            try:
                push_to_gateway(
                    self.config["prometheus"]["gateway"],
                    job=self.config["prometheus"]["job"],
                    registry=self.registry
                )
            except Exception as e:
                logger.error(f"Failed to push metrics: {e}")
    
    def _check_alerts(self, results: Dict[str, ServiceHealth]):
        """Check if alerts should be triggered"""
        unhealthy_services = [
            name for name, health in results.items()
            if health.status == HealthStatus.UNHEALTHY
        ]
        
        degraded_services = [
            name for name, health in results.items()
            if health.status == HealthStatus.DEGRADED
        ]
        
        if unhealthy_services:
            self.alert_manager.send_critical_alert(
                f"Services DOWN: {', '.join(unhealthy_services)}",
                results
            )
        
        if degraded_services:
            self.alert_manager.send_warning_alert(
                f"Services DEGRADED: {', '.join(degraded_services)}",
                results
            )


class AlertManager:
    """Alert management system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.alert_history = []
        self.consecutive_failures = {}
    
    def send_critical_alert(self, message: str, health_data: Dict[str, ServiceHealth]):
        """Send critical alert"""
        logger.critical(f"CRITICAL ALERT: {message}")
        
        # Slack
        if self.config.get("slack_webhook"):
            self._send_slack_alert(message, health_data, "danger")
        
        # PagerDuty
        if self.config.get("pagerduty_key"):
            self._send_pagerduty_alert(message, health_data, "critical")
        
        # Email
        if self.config.get("email_recipients"):
            self._send_email_alert(message, health_data, "critical")
    
    def send_warning_alert(self, message: str, health_data: Dict[str, ServiceHealth]):
        """Send warning alert"""
        logger.warning(f"WARNING ALERT: {message}")
        
        # Only send to Slack for warnings
        if self.config.get("slack_webhook"):
            self._send_slack_alert(message, health_data, "warning")
    
    def _send_slack_alert(self, message: str, health_data: Dict, severity: str):
        """Send Slack alert"""
        color_map = {
            "danger": "#FF0000",
            "warning": "#FFA500",
            "good": "#00FF00"
        }
        
        fields = []
        for name, health in health_data.items():
            if health.status != HealthStatus.HEALTHY:
                fields.append({
                    "title": name.upper(),
                    "value": f"Status: {health.status.value}\nError: {health.error_message or 'N/A'}",
                    "short": True
                })
        
        payload = {
            "attachments": [{
                "color": color_map.get(severity, "#808080"),
                "title": f"Health Check Alert - {severity.upper()}",
                "text": message,
                "fields": fields[:10],  # Limit to 10 fields
                "footer": "YTEmpire Health Monitor",
                "ts": int(time.time())
            }]
        }
        
        try:
            requests.post(
                self.config["slack_webhook"],
                json=payload,
                timeout=10
            )
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    def _send_pagerduty_alert(self, message: str, health_data: Dict, severity: str):
        """Send PagerDuty alert"""
        # Implement PagerDuty integration
        pass
    
    def _send_email_alert(self, message: str, health_data: Dict, severity: str):
        """Send email alert"""
        # Implement email sending
        pass


async def continuous_monitoring(interval: int = 30):
    """Run continuous health monitoring"""
    checker = HealthChecker(
        environment=os.getenv("ENVIRONMENT", "production")
    )
    
    logger.info(f"Starting continuous health monitoring (interval: {interval}s)")
    
    while True:
        try:
            results = await checker.check_all_services()
            
            # Log summary
            healthy = sum(1 for h in results.values() if h.status == HealthStatus.HEALTHY)
            total = len(results)
            
            logger.info(f"Health check complete: {healthy}/{total} services healthy")
            
            # Save results to file
            with open("health_status.json", "w") as f:
                json.dump(
                    {name: asdict(health) for name, health in results.items()},
                    f,
                    indent=2,
                    default=str
                )
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
        
        await asyncio.sleep(interval)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YTEmpire Health Checker")
    parser.add_argument("--once", action="store_true",
                       help="Run health check once and exit")
    parser.add_argument("--interval", type=int, default=30,
                       help="Check interval in seconds (default: 30)")
    parser.add_argument("--environment", default="production",
                       help="Environment name")
    parser.add_argument("--json", action="store_true",
                       help="Output results as JSON")
    
    args = parser.parse_args()
    
    os.environ["ENVIRONMENT"] = args.environment
    
    if args.once:
        # Run once
        checker = HealthChecker(environment=args.environment)
        results = asyncio.run(checker.check_all_services())
        
        if args.json:
            print(json.dumps(
                {name: asdict(health) for name, health in results.items()},
                indent=2,
                default=str
            ))
        else:
            for name, health in results.items():
                status_symbol = "✅" if health.status == HealthStatus.HEALTHY else "❌"
                print(f"{status_symbol} {name}: {health.status.value}")
                if health.error_message:
                    print(f"   Error: {health.error_message}")
                print(f"   Response time: {health.response_time:.2f}ms")
        
        # Exit with error if any service is unhealthy
        unhealthy = any(h.status == HealthStatus.UNHEALTHY for h in results.values())
        sys.exit(1 if unhealthy else 0)
    else:
        # Run continuously
        asyncio.run(continuous_monitoring(interval=args.interval))


if __name__ == "__main__":
    main()