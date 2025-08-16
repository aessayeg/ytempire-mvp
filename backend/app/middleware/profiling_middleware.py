"""
Performance Profiling Middleware
Tracks and profiles API performance metrics
"""
import time
import asyncio
import tracemalloc
import psutil
import json
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from contextvars import ContextVar
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import logging
from prometheus_client import Counter, Histogram, Gauge
import cProfile
import pstats
import io
from functools import wraps

from app.core.cache import cache_service

logger = logging.getLogger(__name__)

# Context variables for request tracking
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
request_start_time_var: ContextVar[float] = ContextVar(
    "request_start_time", default=0.0
)

# Prometheus metrics
http_requests_total = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

http_request_size_bytes = Histogram(
    "http_request_size_bytes", "HTTP request size in bytes", ["method", "endpoint"]
)

http_response_size_bytes = Histogram(
    "http_response_size_bytes", "HTTP response size in bytes", ["method", "endpoint"]
)

database_query_duration_seconds = Histogram(
    "database_query_duration_seconds",
    "Database query duration in seconds",
    ["query_type"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

active_requests = Gauge("active_requests", "Number of active requests")

memory_usage_bytes = Gauge("memory_usage_bytes", "Memory usage in bytes", ["type"])

cpu_usage_percent = Gauge("cpu_usage_percent", "CPU usage percentage")


class PerformanceProfilingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for comprehensive performance profiling
    """

    def __init__(
        self,
        app: ASGIApp,
        enable_profiling: bool = True,
        enable_memory_tracking: bool = False,
        slow_request_threshold: float = 1.0,
        profile_sample_rate: float = 0.1,
    ):
        super().__init__(app)
        self.enable_profiling = enable_profiling
        self.enable_memory_tracking = enable_memory_tracking
        self.slow_request_threshold = slow_request_threshold
        self.profile_sample_rate = profile_sample_rate

        # Start memory tracking if enabled
        if self.enable_memory_tracking:
            tracemalloc.start()

        # Start system monitoring
        self._start_system_monitoring()

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with profiling"""
        # Generate request ID
        request_id = f"{datetime.utcnow().timestamp()}_{id(request)}"
        request_id_var.set(request_id)

        # Record start time
        start_time = time.perf_counter()
        request_start_time_var.set(start_time)

        # Increment active requests
        active_requests.inc()

        # Get request details
        method = request.method
        path = request.url.path

        # Initialize profiling
        profiler = None
        memory_before = None

        try:
            # Start CPU profiling for sampled requests
            if self._should_profile(path):
                profiler = cProfile.Profile()
                profiler.enable()

            # Track memory usage
            if self.enable_memory_tracking:
                memory_before = self._get_memory_usage()

            # Process request
            response = await call_next(request)

            # Calculate duration
            duration = time.perf_counter() - start_time

            # Stop profiling
            if profiler:
                profiler.disable()

            # Record metrics
            self._record_metrics(
                request=request,
                response=response,
                duration=duration,
                profiler=profiler,
                memory_before=memory_before,
            )

            # Add performance headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration:.3f}s"

            # Log slow requests
            if duration > self.slow_request_threshold:
                await self._log_slow_request(
                    request=request,
                    response=response,
                    duration=duration,
                    profiler=profiler,
                )

            return response

        except Exception as e:
            logger.error(f"Error in profiling middleware: {str(e)}")
            raise
        finally:
            # Decrement active requests
            active_requests.dec()

    def _should_profile(self, path: str) -> bool:
        """Determine if request should be profiled"""
        # Don't profile static files or health checks
        if path in ["/health", "/metrics", "/docs", "/redoc", "/openapi.json"]:
            return False

        # Sample requests based on rate
        import random

        return self.enable_profiling and random.random() < self.profile_sample_rate

    def _get_memory_usage(self) -> Dict[str, int]:
        """Get current memory usage"""
        process = psutil.Process()
        return {
            "rss": process.memory_info().rss,
            "vms": process.memory_info().vms,
            "percent": process.memory_percent(),
        }

    def _record_metrics(
        self,
        request: Request,
        response: Response,
        duration: float,
        profiler: Optional[cProfile.Profile] = None,
        memory_before: Optional[Dict[str, int]] = None,
    ):
        """Record performance metrics"""
        method = request.method
        path = request.url.path
        status = response.status_code

        # Prometheus metrics
        http_requests_total.labels(method=method, endpoint=path, status=status).inc()

        http_request_duration_seconds.labels(method=method, endpoint=path).observe(
            duration
        )

        # Request/Response sizes
        request_size = int(request.headers.get("content-length", 0))
        if request_size > 0:
            http_request_size_bytes.labels(method=method, endpoint=path).observe(
                request_size
            )

        response_size = int(response.headers.get("content-length", 0))
        if response_size > 0:
            http_response_size_bytes.labels(method=method, endpoint=path).observe(
                response_size
            )

        # Memory metrics
        if self.enable_memory_tracking and memory_before:
            memory_after = self._get_memory_usage()
            memory_delta = memory_after["rss"] - memory_before["rss"]

            memory_usage_bytes.labels(type="rss").set(memory_after["rss"])
            memory_usage_bytes.labels(type="vms").set(memory_after["vms"])

            # Log significant memory increases
            if memory_delta > 10 * 1024 * 1024:  # 10MB
                logger.warning(
                    f"High memory allocation in {method} {path}: "
                    f"{memory_delta / 1024 / 1024:.2f}MB"
                )

    async def _log_slow_request(
        self,
        request: Request,
        response: Response,
        duration: float,
        profiler: Optional[cProfile.Profile] = None,
    ):
        """Log details about slow requests"""
        method = request.method
        path = request.url.path
        status = response.status_code

        log_data = {
            "request_id": request_id_var.get(),
            "method": method,
            "path": path,
            "status": status,
            "duration": duration,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Add profiling data if available
        if profiler:
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
            ps.print_stats(20)  # Top 20 functions
            log_data["profile"] = s.getvalue()

        # Log to file/monitoring system
        logger.warning(f"Slow request detected: {json.dumps(log_data, indent=2)}")

        # Store in cache for analysis
        cache_key = f"slow_requests:{datetime.utcnow().strftime('%Y%m%d')}"
        await cache_service.lpush(cache_key, json.dumps(log_data))
        await cache_service.ltrim(cache_key, 0, 100)  # Keep last 100 slow requests
        await cache_service.expire(cache_key, 86400)  # 24 hours

    def _start_system_monitoring(self):
        """Start background system monitoring"""

        async def monitor_system():
            while True:
                try:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    cpu_usage_percent.set(cpu_percent)

                    # Memory usage
                    memory = psutil.virtual_memory()
                    memory_usage_bytes.labels(type="available").set(memory.available)
                    memory_usage_bytes.labels(type="used").set(memory.used)

                    # Disk usage
                    disk = psutil.disk_usage("/")
                    memory_usage_bytes.labels(type="disk_free").set(disk.free)

                    await asyncio.sleep(10)  # Update every 10 seconds
                except Exception as e:
                    logger.error(f"Error in system monitoring: {str(e)}")
                    await asyncio.sleep(60)

        # Start monitoring in background
        asyncio.create_task(monitor_system())


class DatabaseQueryProfiler:
    """Profile database queries"""

    @staticmethod
    def profile_query(query_type: str = "select"):
        """Decorator to profile database queries"""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.perf_counter()

                try:
                    result = await func(*args, **kwargs)
                    duration = time.perf_counter() - start_time

                    # Record metric
                    database_query_duration_seconds.labels(
                        query_type=query_type
                    ).observe(duration)

                    # Log slow queries
                    if duration > 0.1:  # 100ms
                        logger.warning(
                            f"Slow database query ({query_type}): {duration:.3f}s"
                        )

                    return result

                except Exception as e:
                    duration = time.perf_counter() - start_time
                    logger.error(
                        f"Database query error ({query_type}): {str(e)}, "
                        f"Duration: {duration:.3f}s"
                    )
                    raise

            return wrapper

        return decorator


class PerformanceProfiler:
    """Performance profiling utilities"""

    @staticmethod
    async def profile_function(func: Callable, *args, **kwargs):
        """Profile a specific function"""
        profiler = cProfile.Profile()
        profiler.enable()

        start_time = time.perf_counter()
        memory_before = psutil.Process().memory_info().rss

        try:
            result = (
                await func(*args, **kwargs)
                if asyncio.iscoroutinefunction(func)
                else func(*args, **kwargs)
            )

            duration = time.perf_counter() - start_time
            memory_after = psutil.Process().memory_info().rss
            memory_delta = memory_after - memory_before

            profiler.disable()

            # Generate profile report
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
            ps.print_stats(20)

            return {
                "result": result,
                "duration": duration,
                "memory_delta": memory_delta,
                "profile": s.getvalue(),
            }

        except Exception as e:
            profiler.disable()
            raise

    @staticmethod
    def analyze_slow_requests(date: Optional[str] = None) -> Dict[str, Any]:
        """Analyze slow requests from cache"""
        if not date:
            date = datetime.utcnow().strftime("%Y%m%d")

        cache_key = f"slow_requests:{date}"

        # This would be async in practice
        # slow_requests = await cache_service.lrange(cache_key, 0, -1)

        # For now, return placeholder
        return {
            "date": date,
            "total_slow_requests": 0,
            "average_duration": 0,
            "top_endpoints": [],
            "time_distribution": {},
        }


# Export middleware and utilities
__all__ = [
    "PerformanceProfilingMiddleware",
    "DatabaseQueryProfiler",
    "PerformanceProfiler",
    "http_requests_total",
    "http_request_duration_seconds",
    "database_query_duration_seconds",
    "active_requests",
    "memory_usage_bytes",
    "cpu_usage_percent",
]
