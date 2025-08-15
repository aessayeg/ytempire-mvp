"""
Performance Optimization Module
Implements caching, connection pooling, and response optimization
"""
import asyncio
import os
import time
import json
import hashlib
import pickle
from typing import Any, Optional, Callable, Dict, List
from functools import wraps
from datetime import datetime, timedelta
import redis.asyncio as redis
from fastapi import Request, Response
from fastapi.responses import ORJSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse
import orjson
import logging
from dataclasses import dataclass
import aiohttp
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool, QueuePool

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Cache configuration"""
    enabled: bool = True
    default_ttl: int = 300  # 5 minutes
    max_connections: int = 100
    connection_timeout: int = 5
    socket_timeout: int = 5
    retry_on_timeout: bool = True
    health_check_interval: int = 30


class CacheManager:
    """
    Advanced caching system with multiple strategies
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379", config: Optional[CacheConfig] = None):
        self.redis_url = redis_url
        self.config = config or CacheConfig()
        self.redis_client: Optional[redis.Redis] = None
        self.local_cache: Dict[str, Any] = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "errors": 0,
            "evictions": 0
        }
        
    async def initialize(self):
        """Initialize cache manager"""
        if not self.config.enabled:
            logger.info("Cache disabled")
            return
            
        try:
            self.redis_client = await redis.from_url(
                self.redis_url,
                max_connections=self.config.max_connections,
                socket_connect_timeout=self.config.connection_timeout,
                socket_timeout=self.config.socket_timeout,
                retry_on_timeout=self.config.retry_on_timeout
            )
            
            # Test connection
            await self.redis_client.ping()
            
            # Start health check
            asyncio.create_task(self._health_check())
            
            logger.info("Cache manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize cache: {e}")
            self.config.enabled = False
            
    async def _health_check(self):
        """Periodic health check for Redis connection"""
        while self.config.enabled:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self.redis_client.ping()
            except Exception as e:
                logger.error(f"Cache health check failed: {e}")
                # Attempt reconnection
                try:
                    await self.initialize()
                except:
                    pass
                    
    def cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = {
            "prefix": prefix,
            "args": args,
            "kwargs": kwargs
        }
        key_hash = hashlib.md5(orjson.dumps(key_data, option=orjson.OPT_SORT_KEYS)).hexdigest()
        return f"{prefix}:{key_hash}"
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.config.enabled:
            return None
            
        try:
            # Check local cache first
            if key in self.local_cache:
                self.cache_stats["hits"] += 1
                return self.local_cache[key]["value"]
                
            # Check Redis
            if self.redis_client:
                value = await self.redis_client.get(key)
                if value:
                    self.cache_stats["hits"] += 1
                    # Store in local cache
                    deserialized = pickle.loads(value)
                    self.local_cache[key] = {
                        "value": deserialized,
                        "expires": datetime.utcnow() + timedelta(seconds=60)
                    }
                    return deserialized
                    
            self.cache_stats["misses"] += 1
            return None
            
        except Exception as e:
            self.cache_stats["errors"] += 1
            logger.error(f"Cache get error: {e}")
            return None
            
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        if not self.config.enabled:
            return False
            
        ttl = ttl or self.config.default_ttl
        
        try:
            # Store in local cache
            self.local_cache[key] = {
                "value": value,
                "expires": datetime.utcnow() + timedelta(seconds=ttl)
            }
            
            # Clean up expired local cache entries
            if len(self.local_cache) > 1000:
                self._cleanup_local_cache()
                
            # Store in Redis
            if self.redis_client:
                serialized = pickle.dumps(value)
                await self.redis_client.setex(key, ttl, serialized)
                
            return True
            
        except Exception as e:
            self.cache_stats["errors"] += 1
            logger.error(f"Cache set error: {e}")
            return False
            
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        if not self.config.enabled:
            return False
            
        try:
            # Remove from local cache
            if key in self.local_cache:
                del self.local_cache[key]
                
            # Remove from Redis
            if self.redis_client:
                await self.redis_client.delete(key)
                
            return True
            
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
            
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern"""
        if not self.config.enabled or not self.redis_client:
            return 0
            
        try:
            count = 0
            cursor = 0
            
            while True:
                cursor, keys = await self.redis_client.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100
                )
                
                if keys:
                    await self.redis_client.delete(*keys)
                    count += len(keys)
                    
                    # Also remove from local cache
                    for key in keys:
                        if key in self.local_cache:
                            del self.local_cache[key]
                            
                if cursor == 0:
                    break
                    
            return count
            
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
            return 0
            
    def _cleanup_local_cache(self):
        """Clean up expired entries from local cache"""
        now = datetime.utcnow()
        expired_keys = [
            key for key, data in self.local_cache.items()
            if data["expires"] < now
        ]
        
        for key in expired_keys:
            del self.local_cache[key]
            self.cache_stats["evictions"] += 1
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total * 100) if total > 0 else 0
        
        return {
            **self.cache_stats,
            "hit_rate": hit_rate,
            "local_cache_size": len(self.local_cache)
        }


def cached(ttl: int = 300, key_prefix: Optional[str] = None):
    """
    Decorator for caching function results
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get cache manager from first argument if it has one
            cache_manager = getattr(args[0], 'cache_manager', None) if args else None
            
            if not cache_manager:
                return await func(*args, **kwargs)
                
            # Generate cache key
            prefix = key_prefix or f"{func.__module__}.{func.__name__}"
            cache_key = cache_manager.cache_key(prefix, *args[1:], **kwargs)
            
            # Try to get from cache
            cached_value = await cache_manager.get(cache_key)
            if cached_value is not None:
                return cached_value
                
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            await cache_manager.set(cache_key, result, ttl)
            
            return result
            
        return wrapper
    return decorator


class ConnectionPoolManager:
    """
    Manages connection pools for various services
    """
    
    def __init__(self):
        self.pools: Dict[str, Any] = {}
        self.configs: Dict[str, Dict] = {}
        
    async def initialize(self):
        """Initialize connection pools"""
        # Database connection pool
        self.pools["database"] = await self.create_db_pool()
        
        # HTTP connection pool
        self.pools["http"] = await self.create_http_pool()
        
        logger.info("Connection pools initialized")
        
    async def create_db_pool(self) -> AsyncSession:
        """Create database connection pool"""
        engine = create_async_engine(
            os.getenv("DATABASE_URL", "postgresql+asyncpg://localhost/ytempire"),
            pool_size=20,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=3600,
            pool_pre_ping=True,
            echo_pool=False
        )
        
        SessionLocal = sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        return SessionLocal
        
    async def create_http_pool(self) -> aiohttp.ClientSession:
        """Create HTTP connection pool"""
        timeout = aiohttp.ClientTimeout(
            total=30,
            connect=5,
            sock_connect=5,
            sock_read=10
        )
        
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            ttl_dns_cache=300,
            enable_cleanup_closed=True
        )
        
        session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            json_serialize=lambda x: orjson.dumps(x).decode()
        )
        
        return session
        
    async def get_db_session(self) -> AsyncSession:
        """Get database session from pool"""
        SessionLocal = self.pools.get("database")
        if not SessionLocal:
            raise RuntimeError("Database pool not initialized")
        return SessionLocal()
        
    async def get_http_session(self) -> aiohttp.ClientSession:
        """Get HTTP session from pool"""
        session = self.pools.get("http")
        if not session:
            raise RuntimeError("HTTP pool not initialized")
        return session
        
    async def cleanup(self):
        """Cleanup all connection pools"""
        # Close HTTP session
        if "http" in self.pools:
            await self.pools["http"].close()
            
        # Close database engine
        if "database" in self.pools:
            # Engine cleanup would go here
            pass
            
        logger.info("Connection pools cleaned up")


class ResponseOptimizer:
    """
    Optimizes API responses for performance
    """
    
    @staticmethod
    def compress_response(data: Any) -> bytes:
        """Compress response data"""
        import gzip
        json_data = orjson.dumps(data)
        return gzip.compress(json_data)
        
    @staticmethod
    def create_fast_response(data: Any) -> ORJSONResponse:
        """Create optimized JSON response"""
        return ORJSONResponse(
            content=data,
            headers={
                "Cache-Control": "public, max-age=60",
                "X-Content-Type-Options": "nosniff"
            }
        )
        
    @staticmethod
    async def stream_large_response(data_generator):
        """Stream large responses"""
        async def generate():
            yield b'{"data":['
            first = True
            async for item in data_generator:
                if not first:
                    yield b','
                yield orjson.dumps(item)
                first = False
            yield b']}'
            
        return generate()


class PerformanceMiddleware(BaseHTTPMiddleware):
    """
    Middleware for performance monitoring and optimization
    """
    
    def __init__(self, app, cache_manager: Optional[CacheManager] = None):
        super().__init__(app)
        self.cache_manager = cache_manager
        self.metrics = {
            "request_count": 0,
            "total_time": 0,
            "slow_requests": 0,
            "errors": 0
        }
        
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Add request ID for tracing
        request_id = hashlib.md5(f"{time.time()}{request.url}".encode()).hexdigest()[:8]
        request.state.request_id = request_id
        
        try:
            # Check cache for GET requests
            if request.method == "GET" and self.cache_manager:
                cache_key = self.cache_manager.cache_key("response", str(request.url))
                cached_response = await self.cache_manager.get(cache_key)
                
                if cached_response:
                    return ORJSONResponse(
                        content=cached_response,
                        headers={
                            "X-Cache": "HIT",
                            "X-Request-ID": request_id
                        }
                    )
                    
            # Process request
            response = await call_next(request)
            
            # Calculate response time
            response_time = (time.time() - start_time) * 1000  # ms
            
            # Update metrics
            self.metrics["request_count"] += 1
            self.metrics["total_time"] += response_time
            
            if response_time > 500:  # Slow request threshold
                self.metrics["slow_requests"] += 1
                logger.warning(f"Slow request: {request.url} took {response_time:.2f}ms")
                
            # Add performance headers
            response.headers["X-Response-Time"] = f"{response_time:.2f}ms"
            response.headers["X-Request-ID"] = request_id
            
            # Cache successful GET responses
            if request.method == "GET" and response.status_code == 200 and self.cache_manager:
                # For caching, we need to read the response body
                # Check if it's a streaming response
                if isinstance(response, StreamingResponse):
                    # Collect the body from streaming response
                    body_chunks = []
                    async for chunk in response.body_iterator:
                        body_chunks.append(chunk)
                    body = b"".join(body_chunks)
                    
                    # Try to cache JSON responses
                    try:
                        # Cache the response
                        cache_key = self.cache_manager.cache_key("response", str(request.url))
                        await self.cache_manager.set(cache_key, orjson.loads(body), ttl=60)
                    except Exception as e:
                        logger.debug(f"Could not cache response: {e}")
                    
                    # Return new response with the collected body
                    return Response(
                        content=body,
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        media_type=response.media_type
                    )
                else:
                    # For regular responses, just pass through
                    pass
                
            return response
            
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"Request error: {e}")
            
            return ORJSONResponse(
                status_code=500,
                content={"error": "Internal server error", "request_id": request_id}
            )
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        avg_time = (
            self.metrics["total_time"] / self.metrics["request_count"]
            if self.metrics["request_count"] > 0
            else 0
        )
        
        return {
            **self.metrics,
            "avg_response_time": avg_time,
            "slow_request_rate": (
                self.metrics["slow_requests"] / self.metrics["request_count"] * 100
                if self.metrics["request_count"] > 0
                else 0
            )
        }


class QueryOptimizer:
    """
    Database query optimization utilities
    """
    
    @staticmethod
    def add_pagination(query, page: int = 1, per_page: int = 20):
        """Add efficient pagination to query"""
        offset = (page - 1) * per_page
        return query.limit(per_page).offset(offset)
        
    @staticmethod
    def add_select_columns(query, columns: List[str]):
        """Select only required columns"""
        from sqlalchemy import select
        return query.options(select(*columns))
        
    @staticmethod
    async def batch_insert(session: AsyncSession, model, records: List[Dict]):
        """Batch insert records efficiently"""
        from sqlalchemy import insert
        
        # Split into chunks
        chunk_size = 1000
        for i in range(0, len(records), chunk_size):
            chunk = records[i:i + chunk_size]
            stmt = insert(model).values(chunk)
            await session.execute(stmt)
            
        await session.commit()
        
    @staticmethod
    def create_index_hints(query, index_name: str):
        """Add index hints to query"""
        return query.with_hint(index_name)


# Global instances
cache_manager = CacheManager()
connection_pool = ConnectionPoolManager()
response_optimizer = ResponseOptimizer()