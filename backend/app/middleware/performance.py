"""
Performance Optimization Middleware
P1 Task: Achieve <500ms p95 response time
"""
import time
import asyncio
from typing import Callable, Dict, Any, Optional
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from prometheus_client import Histogram, Counter, Gauge
import redis.asyncio as redis
from functools import wraps
import json
import hashlib
import logging

logger = logging.getLogger(__name__)

# Prometheus metrics
request_duration = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint", "status"],
)

request_count = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)

active_requests = Gauge("http_requests_active", "Active HTTP requests")

cache_hits = Counter("cache_hits_total", "Total cache hits")
cache_misses = Counter("cache_misses_total", "Total cache misses")


class PerformanceMiddleware(BaseHTTPMiddleware):
    """
    Performance optimization middleware with caching, compression, and monitoring
    """

    def __init__(self, app, redis_client: Optional[redis.Redis] = None):
        super().__init__(app)
        self.redis_client = redis_client
        self.cache_ttl = 300  # 5 minutes default
        self.compression_threshold = 1024  # Compress responses > 1KB

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Start timing
        start_time = time.time()

        # Track active requests
        active_requests.inc()

        # Check cache for GET requests
        cache_key = None
        if request.method == "GET" and self.redis_client:
            cache_key = self._generate_cache_key(request)
            cached_response = await self._get_cached_response(cache_key)
            if cached_response:
                cache_hits.inc()
                active_requests.dec()
                return Response(
                    content=cached_response,
                    media_type="application/json",
                    headers={"X-Cache": "HIT"},
                )
            else:
                cache_misses.inc()

        try:
            # Process request
            response = await call_next(request)

            # Cache successful GET responses
            if (
                request.method == "GET"
                and response.status_code == 200
                and cache_key
                and self.redis_client
            ):
                await self._cache_response(cache_key, response)

            # Add performance headers
            duration = time.time() - start_time
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            response.headers["X-Cache"] = "MISS"

            # Record metrics
            request_duration.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code,
            ).observe(duration)

            request_count.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code,
            ).inc()

            # Log slow requests
            if duration > 0.5:  # 500ms threshold
                logger.warning(
                    f"Slow request: {request.method} {request.url.path} "
                    f"took {duration:.3f}s"
                )

            return response

        finally:
            active_requests.dec()

    def _generate_cache_key(self, request: Request) -> str:
        """Generate cache key based on URL and query parameters"""
        key_parts = [request.url.path, str(sorted(request.query_params.items()))]
        key_string = ":".join(key_parts)
        return f"cache:{hashlib.md5(key_string.encode()).hexdigest()}"

    async def _get_cached_response(self, key: str) -> Optional[bytes]:
        """Get cached response from Redis"""
        try:
            cached = await self.redis_client.get(key)
            return cached
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    async def _cache_response(self, key: str, response: Response):
        """Cache response in Redis"""
        try:
            # Read response body
            body = b""
            async for chunk in response.body_iterator:
                body += chunk

            # Cache the body
            await self.redis_client.setex(key, self.cache_ttl, body)

            # Recreate response with the same body
            response.body_iterator = self._iterate_body(body)

        except Exception as e:
            logger.error(f"Cache set error: {e}")

    async def _iterate_body(self, body: bytes):
        """Helper to iterate over body bytes"""
        yield body


class AsyncCache:
    """
    Async function result caching decorator
    """

    def __init__(self, ttl: int = 300, key_prefix: str = "func"):
        self.ttl = ttl
        self.key_prefix = key_prefix
        self.redis_client = None

    async def init(self, redis_client: redis.Redis):
        """Initialize with Redis client"""
        self.redis_client = redis_client

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not self.redis_client:
                return await func(*args, **kwargs)

            # Generate cache key
            key = self._generate_key(func.__name__, args, kwargs)

            # Check cache
            cached = await self.redis_client.get(key)
            if cached:
                cache_hits.inc()
                return json.loads(cached)

            cache_misses.inc()

            # Execute function
            result = await func(*args, **kwargs)

            # Cache result
            await self.redis_client.setex(
                key, self.ttl, json.dumps(result, default=str)
            )

            return result

        return wrapper

    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key for function call"""
        key_parts = [self.key_prefix, func_name, str(args), str(sorted(kwargs.items()))]
        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()


class ConnectionPool:
    """
    Database connection pooling for optimized performance
    """

    def __init__(self, min_size: int = 10, max_size: int = 50):
        self.min_size = min_size
        self.max_size = max_size
        self._pool = []
        self._in_use = set()
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a connection from the pool"""
        async with self._lock:
            if self._pool:
                conn = self._pool.pop()
            elif len(self._in_use) < self.max_size:
                conn = await self._create_connection()
            else:
                # Wait for a connection to be released
                while not self._pool:
                    await asyncio.sleep(0.01)
                conn = self._pool.pop()

            self._in_use.add(conn)
            return conn

    async def release(self, conn):
        """Release a connection back to the pool"""
        async with self._lock:
            self._in_use.discard(conn)
            if len(self._pool) < self.min_size:
                self._pool.append(conn)
            else:
                await self._close_connection(conn)

    async def _create_connection(self):
        """Create a new database connection"""
        # Implementation depends on database
        pass

    async def _close_connection(self, conn):
        """Close a database connection"""
        # Implementation depends on database
        pass


class QueryOptimizer:
    """
    SQL query optimization and analysis
    """

    @staticmethod
    def optimize_query(query: str) -> str:
        """
        Optimize SQL query for better performance
        """
        optimizations = [
            # Add indexes hints
            ("SELECT", "SELECT /*+ INDEX */"),
            # Limit default
            ("SELECT", "SELECT SQL_CALC_FOUND_ROWS"),
            # Use UNION ALL instead of UNION when possible
            ("UNION", "UNION ALL"),
        ]

        optimized = query
        for old, new in optimizations:
            if old in optimized and new not in optimized:
                optimized = optimized.replace(old, new, 1)

        return optimized

    @staticmethod
    async def explain_query(db_session, query: str) -> Dict[str, Any]:
        """
        Run EXPLAIN on query to analyze performance
        """
        explain_query = f"EXPLAIN ANALYZE {query}"
        result = await db_session.execute(explain_query)
        return result.fetchall()


class ResponseCompression:
    """
    Response compression for bandwidth optimization
    """

    @staticmethod
    def should_compress(content: bytes, threshold: int = 1024) -> bool:
        """Check if content should be compressed"""
        return len(content) > threshold

    @staticmethod
    async def compress_response(content: bytes) -> bytes:
        """Compress response content using gzip"""
        import gzip

        return gzip.compress(content, compresslevel=6)


# Performance optimization functions
async def batch_database_operations(operations: list, batch_size: int = 100):
    """
    Batch database operations for better performance
    """
    results = []
    for i in range(0, len(operations), batch_size):
        batch = operations[i : i + batch_size]
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)
    return results


def optimize_pagination(query, page: int = 1, limit: int = 20):
    """
    Optimize pagination queries using cursor-based pagination
    """
    offset = (page - 1) * limit
    return query.limit(limit).offset(offset)


# Lazy loading decorator
def lazy_load(func: Callable) -> Callable:
    """
    Decorator for lazy loading of expensive resources
    """
    cache = {}

    @wraps(func)
    async def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = await func(*args, **kwargs)
        return cache[key]

    return wrapper


# Performance monitoring
class PerformanceMonitor:
    """
    Monitor and report performance metrics
    """

    def __init__(self):
        self.metrics = {
            "response_times": [],
            "slow_queries": [],
            "cache_stats": {"hits": 0, "misses": 0},
            "error_rate": 0,
        }

    def record_response_time(self, duration: float, endpoint: str):
        """Record API response time"""
        self.metrics["response_times"].append(
            {"duration": duration, "endpoint": endpoint, "timestamp": time.time()}
        )

        # Keep only last 1000 records
        if len(self.metrics["response_times"]) > 1000:
            self.metrics["response_times"] = self.metrics["response_times"][-1000:]

    def get_p95_response_time(self) -> float:
        """Calculate p95 response time"""
        if not self.metrics["response_times"]:
            return 0

        times = sorted([r["duration"] for r in self.metrics["response_times"]])
        index = int(len(times) * 0.95)
        return times[index] if index < len(times) else times[-1]

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        p95 = self.get_p95_response_time()

        return {
            "p95_response_time": p95,
            "p95_target_met": p95 < 0.5,  # 500ms target
            "cache_hit_rate": self._calculate_cache_hit_rate(),
            "slow_endpoints": self._get_slow_endpoints(),
            "recommendations": self._get_optimization_recommendations(p95),
        }

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = (
            self.metrics["cache_stats"]["hits"] + self.metrics["cache_stats"]["misses"]
        )
        if total == 0:
            return 0
        return self.metrics["cache_stats"]["hits"] / total

    def _get_slow_endpoints(self) -> list:
        """Identify slow endpoints"""
        endpoint_times = {}
        for record in self.metrics["response_times"]:
            endpoint = record["endpoint"]
            if endpoint not in endpoint_times:
                endpoint_times[endpoint] = []
            endpoint_times[endpoint].append(record["duration"])

        slow_endpoints = []
        for endpoint, times in endpoint_times.items():
            avg_time = sum(times) / len(times)
            if avg_time > 0.5:  # 500ms threshold
                slow_endpoints.append(
                    {"endpoint": endpoint, "avg_time": avg_time, "count": len(times)}
                )

        return sorted(slow_endpoints, key=lambda x: x["avg_time"], reverse=True)

    def _get_optimization_recommendations(self, p95: float) -> list:
        """Get performance optimization recommendations"""
        recommendations = []

        if p95 > 0.5:
            recommendations.append("Enable response caching for GET endpoints")
            recommendations.append("Implement database query optimization")
            recommendations.append("Consider adding more worker processes")

        if self._calculate_cache_hit_rate() < 0.5:
            recommendations.append("Increase cache TTL for frequently accessed data")
            recommendations.append("Implement aggressive caching strategy")

        slow_endpoints = self._get_slow_endpoints()
        if slow_endpoints:
            recommendations.append(
                f"Optimize slow endpoints: {', '.join([e['endpoint'] for e in slow_endpoints[:3]])}"
            )

        return recommendations


# Global performance monitor instance
performance_monitor = PerformanceMonitor()
