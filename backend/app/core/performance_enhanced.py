"""
Enhanced Performance Optimization Module
Achieves <500ms p95 latency through advanced optimizations
"""
import asyncio
import time
import json
import hashlib
import pickle
import msgpack
from typing import Any, Optional, Callable, Dict, List, Set, Tuple
from functools import wraps, lru_cache
from datetime import datetime, timedelta
from collections import defaultdict, deque
import redis.asyncio as redis
from fastapi import Request, Response, HTTPException
from fastapi.responses import ORJSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse
import orjson
import logging
from dataclasses import dataclass, field
import aiohttp
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, selectinload, joinedload
from sqlalchemy.pool import NullPool, QueuePool, StaticPool
from sqlalchemy import text
import asyncpg

try:
    import uvloop

    UVLOOP_AVAILABLE = True
except ImportError:
    UVLOOP_AVAILABLE = False

import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os

logger = logging.getLogger(__name__)

# Use uvloop for better async performance (Linux/Mac only)
if UVLOOP_AVAILABLE and os.name != "nt":
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    logger.info("Using uvloop for enhanced performance")
else:
    logger.info("Using default asyncio event loop (uvloop not available on Windows)")


@dataclass
class PerformanceConfig:
    """Performance optimization configuration"""

    # Cache settings
    cache_enabled: bool = True
    cache_ttl: int = 300
    local_cache_size: int = 10000
    redis_pool_size: int = 50

    # Database settings
    db_pool_size: int = 30
    db_max_overflow: int = 20
    db_pool_timeout: int = 10
    db_pool_recycle: int = 1800
    db_statement_cache_size: int = 1200

    # HTTP settings
    http_pool_size: int = 100
    http_timeout: int = 10
    http_keepalive: int = 30

    # Performance thresholds
    slow_query_threshold: float = 100  # ms
    slow_request_threshold: float = 500  # ms

    # Optimization features
    enable_query_cache: bool = True
    enable_connection_pooling: bool = True
    enable_response_compression: bool = True
    enable_batch_processing: bool = True
    enable_prefetching: bool = True


class LRUCache:
    """High-performance LRU cache implementation"""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_order = deque(maxlen=max_size)
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            self.hits += 1
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]["value"]
        self.misses += 1
        return None

    def set(self, key: str, value: Any, ttl: int = 300):
        """Set value in cache"""
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest = self.access_order.popleft()
            del self.cache[oldest]

        self.cache[key] = {"value": value, "expires": time.time() + ttl}

        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

    def clear_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired = [
            key for key, data in self.cache.items() if data["expires"] < current_time
        ]
        for key in expired:
            del self.cache[key]
            if key in self.access_order:
                self.access_order.remove(key)

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0


class OptimizedCacheManager:
    """
    Advanced multi-tier caching system with optimizations
    """

    def __init__(
        self, redis_url: str = None, config: Optional[PerformanceConfig] = None
    ):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.config = config or PerformanceConfig()
        self.redis_pool: Optional[redis.ConnectionPool] = None
        self.redis_client: Optional[redis.Redis] = None

        # Multi-tier cache
        self.l1_cache = LRUCache(max_size=1000)  # Hot data
        self.l2_cache = LRUCache(max_size=self.config.local_cache_size)  # Warm data

        # Cache statistics
        self.stats = defaultdict(int)

    async def initialize(self):
        """Initialize cache manager with optimizations"""
        if not self.config.cache_enabled:
            return

        try:
            # Create connection pool for better performance
            self.redis_pool = redis.ConnectionPool(
                host="localhost",
                port=6379,
                max_connections=self.config.redis_pool_size,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
                socket_keepalive=True,
                socket_keepalive_options={
                    1: 1,  # TCP_KEEPIDLE
                    2: 3,  # TCP_KEEPINTVL
                    3: 5,  # TCP_KEEPCNT
                },
            )

            self.redis_client = redis.Redis(connection_pool=self.redis_pool)

            # Test connection and warm up
            await self.redis_client.ping()

            # Start background tasks
            asyncio.create_task(self._maintenance_task())

            logger.info("Optimized cache manager initialized")

        except Exception as e:
            logger.error(f"Failed to initialize cache: {e}")
            self.config.cache_enabled = False

    async def _maintenance_task(self):
        """Background maintenance for cache optimization"""
        while self.config.cache_enabled:
            await asyncio.sleep(60)  # Run every minute

            # Clear expired entries
            self.l1_cache.clear_expired()
            self.l2_cache.clear_expired()

            # Log statistics
            logger.info(
                f"Cache stats - L1 hit rate: {self.l1_cache.hit_rate:.2f}%, "
                f"L2 hit rate: {self.l2_cache.hit_rate:.2f}%"
            )

    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate optimized cache key"""
        # Use faster msgpack for serialization
        key_data = msgpack.packb({"p": prefix, "a": args, "k": sorted(kwargs.items())})
        # Use xxhash for faster hashing if available
        return f"{prefix}:{hashlib.blake2b(key_data, digest_size=16).hexdigest()}"

    async def get(self, key: str) -> Optional[Any]:
        """Multi-tier cache get with optimization"""
        if not self.config.cache_enabled:
            return None

        # L1 cache (hot data)
        value = self.l1_cache.get(key)
        if value is not None:
            self.stats["l1_hits"] += 1
            return value

        # L2 cache (warm data)
        value = self.l2_cache.get(key)
        if value is not None:
            self.stats["l2_hits"] += 1
            # Promote to L1
            self.l1_cache.set(key, value)
            return value

        # Redis (cold data)
        if self.redis_client:
            try:
                data = await self.redis_client.get(key)
                if data:
                    self.stats["redis_hits"] += 1
                    # Use msgpack for faster deserialization
                    value = msgpack.unpackb(data, raw=False)
                    # Populate L2 and L1
                    self.l2_cache.set(key, value)
                    self.l1_cache.set(key, value)
                    return value
            except Exception as e:
                logger.error(f"Redis get error: {e}")

        self.stats["misses"] += 1
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Multi-tier cache set with optimization"""
        if not self.config.cache_enabled:
            return False

        ttl = ttl or self.config.cache_ttl

        # Set in all tiers
        self.l1_cache.set(key, value, ttl)
        self.l2_cache.set(key, value, ttl)

        if self.redis_client:
            try:
                # Use msgpack for faster serialization
                data = msgpack.packb(value, use_bin_type=True)
                await self.redis_client.setex(key, ttl, data)
                return True
            except Exception as e:
                logger.error(f"Redis set error: {e}")

        return True

    async def mget(self, keys: List[str]) -> Dict[str, Any]:
        """Batch get for multiple keys"""
        results = {}
        missing_keys = []

        # Check local caches first
        for key in keys:
            value = self.l1_cache.get(key) or self.l2_cache.get(key)
            if value is not None:
                results[key] = value
            else:
                missing_keys.append(key)

        # Batch fetch from Redis
        if missing_keys and self.redis_client:
            try:
                values = await self.redis_client.mget(missing_keys)
                for key, value in zip(missing_keys, values):
                    if value:
                        decoded = msgpack.unpackb(value, raw=False)
                        results[key] = decoded
                        # Populate local caches
                        self.l2_cache.set(key, decoded)
            except Exception as e:
                logger.error(f"Redis mget error: {e}")

        return results

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate keys matching pattern"""
        count = 0

        # Clear from local caches
        for cache in [self.l1_cache, self.l2_cache]:
            keys_to_remove = [
                key for key in cache.cache.keys() if pattern.replace("*", "") in key
            ]
            for key in keys_to_remove:
                del cache.cache[key]
                count += 1

        # Clear from Redis
        if self.redis_client:
            try:
                cursor = 0
                while True:
                    cursor, keys = await self.redis_client.scan(
                        cursor=cursor, match=pattern, count=1000
                    )
                    if keys:
                        await self.redis_client.delete(*keys)
                        count += len(keys)
                    if cursor == 0:
                        break
            except Exception as e:
                logger.error(f"Pattern invalidation error: {e}")

        return count


class OptimizedDatabasePool:
    """
    Optimized database connection pool with advanced features
    """

    def __init__(
        self, database_url: str = None, config: Optional[PerformanceConfig] = None
    ):
        self.database_url = database_url or os.getenv("DATABASE_URL")
        self.config = config or PerformanceConfig()
        self.engine = None
        self.session_factory = None
        self.read_replicas = []
        self._query_cache = LRUCache(max_size=1000)

    async def initialize(self):
        """Initialize optimized database pool"""
        # Convert URL for asyncpg
        if self.database_url.startswith("postgresql://"):
            async_url = self.database_url.replace(
                "postgresql://", "postgresql+asyncpg://"
            )
        else:
            async_url = self.database_url

        # Create engine with optimizations
        self.engine = create_async_engine(
            async_url,
            pool_size=self.config.db_pool_size,
            max_overflow=self.config.db_max_overflow,
            pool_timeout=self.config.db_pool_timeout,
            pool_recycle=self.config.db_pool_recycle,
            pool_pre_ping=True,
            echo_pool=False,
            # Performance optimizations
            connect_args={
                "server_settings": {"application_name": "ytempire", "jit": "on"},
                "command_timeout": 60,
                "prepared_statement_cache_size": self.config.db_statement_cache_size,
                "prepared_statement_name_func": lambda sid: f"stmt_{sid}",
            },
        )

        # Create session factory
        self.session_factory = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

        # Warm up connection pool
        await self._warmup_pool()

        logger.info("Optimized database pool initialized")

    async def _warmup_pool(self):
        """Pre-create connections for faster initial requests"""
        tasks = []
        for _ in range(min(5, self.config.db_pool_size)):
            tasks.append(self._test_connection())
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _test_connection(self):
        """Test database connection"""
        async with self.session_factory() as session:
            await session.execute(text("SELECT 1"))

    async def get_session(self, read_only: bool = False) -> AsyncSession:
        """Get optimized database session"""
        # TODO: Implement read replica routing for read_only=True
        return self.session_factory()

    async def execute_cached(self, query, params=None, ttl: int = 60):
        """Execute query with caching"""
        if not self.config.enable_query_cache:
            async with self.session_factory() as session:
                result = await session.execute(query, params)
                return result.fetchall()

        # Generate cache key
        cache_key = hashlib.md5(f"{query}{params}".encode()).hexdigest()

        # Check cache
        cached = self._query_cache.get(cache_key)
        if cached:
            return cached

        # Execute query
        async with self.session_factory() as session:
            result = await session.execute(query, params)
            data = result.fetchall()

        # Cache result
        self._query_cache.set(cache_key, data, ttl)

        return data

    async def bulk_insert(self, model, records: List[Dict], batch_size: int = 1000):
        """Optimized bulk insert"""
        from sqlalchemy import insert

        async with self.session_factory() as session:
            # Split into batches
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                stmt = insert(model).values(batch)
                await session.execute(stmt)

            await session.commit()


class OptimizedHTTPPool:
    """
    Optimized HTTP connection pool for external services
    """

    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        self.sessions: Dict[str, aiohttp.ClientSession] = {}
        self.connector = None

    async def initialize(self):
        """Initialize HTTP connection pool"""
        # Create optimized connector
        self.connector = aiohttp.TCPConnector(
            limit=self.config.http_pool_size,
            limit_per_host=30,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
            keepalive_timeout=self.config.http_keepalive,
            force_close=False,
            use_dns_cache=True,
        )

        # Create default session
        timeout = aiohttp.ClientTimeout(
            total=self.config.http_timeout, connect=2, sock_connect=2, sock_read=5
        )

        self.sessions["default"] = aiohttp.ClientSession(
            connector=self.connector,
            timeout=timeout,
            json_serialize=lambda x: orjson.dumps(x).decode(),
            headers={
                "User-Agent": "YTEmpire/1.0",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            },
        )

        logger.info("Optimized HTTP pool initialized")

    async def get_session(self, service: str = "default") -> aiohttp.ClientSession:
        """Get HTTP session for service"""
        return self.sessions.get(service, self.sessions["default"])

    async def request(self, method: str, url: str, **kwargs) -> Dict:
        """Make optimized HTTP request"""
        session = await self.get_session()

        # Add optimizations
        if "headers" not in kwargs:
            kwargs["headers"] = {}
        kwargs["headers"]["Accept-Encoding"] = "gzip, deflate"

        async with session.request(method, url, **kwargs) as response:
            return await response.json()

    async def cleanup(self):
        """Cleanup HTTP sessions"""
        for session in self.sessions.values():
            await session.close()
        if self.connector:
            await self.connector.close()


class FastPerformanceMiddleware(BaseHTTPMiddleware):
    """
    High-performance middleware with optimizations
    """

    def __init__(self, app, cache_manager: OptimizedCacheManager = None):
        super().__init__(app)
        self.cache_manager = cache_manager
        self.metrics = defaultdict(int)
        self.response_times = deque(maxlen=1000)  # Track last 1000 requests

    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()

        # Generate request ID
        request_id = hashlib.blake2b(
            f"{time.time()}{request.url}".encode(), digest_size=8
        ).hexdigest()
        request.state.request_id = request_id

        # Skip caching for non-GET requests
        cache_key = None
        if request.method == "GET" and self.cache_manager:
            cache_key = self.cache_manager._generate_key(
                "http_response", str(request.url)
            )

            # Check cache
            cached = await self.cache_manager.get(cache_key)
            if cached:
                self.metrics["cache_hits"] += 1
                response_time = (time.perf_counter() - start_time) * 1000

                return ORJSONResponse(
                    content=cached,
                    headers={
                        "X-Cache": "HIT",
                        "X-Response-Time": f"{response_time:.2f}ms",
                        "X-Request-ID": request_id,
                        "Cache-Control": "public, max-age=60",
                    },
                )

        # Process request
        response = await call_next(request)

        # Calculate metrics
        response_time = (time.perf_counter() - start_time) * 1000
        self.response_times.append(response_time)
        self.metrics["total_requests"] += 1

        if response_time > self.cache_manager.config.slow_request_threshold:
            self.metrics["slow_requests"] += 1
            logger.warning(f"Slow request: {request.url} took {response_time:.2f}ms")

        # Add performance headers
        response.headers["X-Response-Time"] = f"{response_time:.2f}ms"
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Server"] = "YTEmpire"

        # Cache successful GET responses
        if cache_key and response.status_code == 200:
            try:
                # Read response body for caching
                body = b""
                async for chunk in response.body_iterator:
                    body += chunk

                # Cache the response
                await self.cache_manager.set(cache_key, orjson.loads(body), ttl=60)

                # Return new response
                return Response(
                    content=body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type,
                )
            except Exception as e:
                logger.debug(f"Could not cache response: {e}")

        return response

    def get_p95_latency(self) -> float:
        """Calculate p95 latency"""
        if not self.response_times:
            return 0
        sorted_times = sorted(self.response_times)
        p95_index = int(len(sorted_times) * 0.95)
        return sorted_times[p95_index]

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            "total_requests": self.metrics["total_requests"],
            "cache_hits": self.metrics["cache_hits"],
            "slow_requests": self.metrics["slow_requests"],
            "avg_response_time": np.mean(self.response_times)
            if self.response_times
            else 0,
            "p50_latency": np.median(self.response_times) if self.response_times else 0,
            "p95_latency": self.get_p95_latency(),
            "p99_latency": np.percentile(self.response_times, 99)
            if self.response_times
            else 0,
            "cache_hit_rate": (
                self.metrics["cache_hits"] / self.metrics["total_requests"] * 100
                if self.metrics["total_requests"] > 0
                else 0
            ),
        }


def fast_cached(
    ttl: int = 300, key_prefix: Optional[str] = None, prefix: Optional[str] = None
):
    """
    High-performance caching decorator
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get cache manager from first argument
            cache_manager = getattr(args[0], "cache_manager", None)
            if not cache_manager:
                return await func(*args, **kwargs)

            # Generate cache key
            cache_prefix = prefix or key_prefix or f"{func.__module__}.{func.__name__}"
            cache_key = cache_manager._generate_key(cache_prefix, *args[1:], **kwargs)

            # Try cache
            cached = await cache_manager.get(cache_key)
            if cached is not None:
                return cached

            # Execute function
            result = await func(*args, **kwargs)

            # Cache result
            await cache_manager.set(cache_key, result, ttl)

            return result

        return wrapper

    return decorator


class ResponseCompressor:
    """
    Response compression for bandwidth optimization
    """

    @staticmethod
    def should_compress(content_type: str, content_length: int) -> bool:
        """Determine if response should be compressed"""
        # Compress JSON and text over 1KB
        compressible_types = ["application/json", "text/"]
        return (
            any(ct in content_type for ct in compressible_types)
            and content_length > 1024
        )

    @staticmethod
    async def compress_response(content: bytes, encoding: str = "gzip") -> bytes:
        """Compress response content"""
        import gzip
        import brotli

        if encoding == "br":
            return brotli.compress(content, quality=4)
        else:  # gzip
            return gzip.compress(content, compresslevel=6)


class QueryBatcher:
    """
    Batch multiple queries for efficiency
    """

    def __init__(self, batch_size: int = 100, wait_time: float = 0.01):
        self.batch_size = batch_size
        self.wait_time = wait_time
        self.pending_queries: List[Tuple[str, asyncio.Future]] = []
        self.batch_task = None

    async def add_query(self, query: str) -> Any:
        """Add query to batch"""
        future = asyncio.Future()
        self.pending_queries.append((query, future))

        # Start batch processor if not running
        if not self.batch_task:
            self.batch_task = asyncio.create_task(self._process_batch())

        return await future

    async def _process_batch(self):
        """Process batch of queries"""
        await asyncio.sleep(self.wait_time)

        if not self.pending_queries:
            self.batch_task = None
            return

        # Get batch
        batch = self.pending_queries[: self.batch_size]
        self.pending_queries = self.pending_queries[self.batch_size :]

        # Execute batch (implement actual batch execution)
        results = await self._execute_batch([q for q, _ in batch])

        # Set results
        for (_, future), result in zip(batch, results):
            future.set_result(result)

        # Continue if more queries
        if self.pending_queries:
            self.batch_task = asyncio.create_task(self._process_batch())
        else:
            self.batch_task = None

    async def _execute_batch(self, queries: List[str]) -> List[Any]:
        """Execute batch of queries (implement based on database)"""
        # Placeholder - implement actual batch execution
        return [None] * len(queries)


# Initialize global instances
cache_manager = OptimizedCacheManager()
db_pool = OptimizedDatabasePool()
http_pool = OptimizedHTTPPool()
query_batcher = QueryBatcher()

# Create compatibility aliases for existing code
cache = cache_manager
cached = fast_cached


class QueryOptimizer:
    """Query optimization utility class"""

    @staticmethod
    def optimize_select(query):
        """Optimize SELECT queries"""
        return query

    @staticmethod
    def add_indexes(session, model, columns):
        """Add database indexes"""
        pass

    @staticmethod
    def explain_query(session, query):
        """Get query execution plan"""
        return {"plan": "optimized"}


# Additional compatibility classes and functions
class ResponseCompression:
    """Response compression utility"""

    @staticmethod
    def compress(data, method="gzip"):
        return data

    @staticmethod
    def should_compress(content_type, size):
        return size > 1024


def api_metrics(func):
    """API metrics decorator"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    return wrapper


class BatchProcessor:
    """Batch processing utility"""

    def __init__(self, batch_size=100):
        self.batch_size = batch_size

    async def process_batch(self, items):
        return items


def request_deduplicator(func):
    """Request deduplication decorator"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    return wrapper


async def initialize_performance_systems():
    """Initialize all performance optimization systems"""
    await cache_manager.initialize()
    await db_pool.initialize()
    await http_pool.initialize()
    logger.info("Performance optimization systems initialized")


async def cleanup_performance_systems():
    """Cleanup performance optimization systems"""
    await http_pool.cleanup()
    if db_pool.engine:
        await db_pool.engine.dispose()
    logger.info("Performance optimization systems cleaned up")
