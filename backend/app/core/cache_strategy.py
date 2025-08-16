"""
Advanced Caching Strategy
Implements multi-layer caching for <500ms API response times
"""
import json
import hashlib
import pickle
from typing import Any, Optional, Callable, Union
from datetime import datetime, timedelta
from functools import wraps
import redis.asyncio as redis
from fastapi import Request, Response
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class CacheStrategy:
    """
    Multi-layer caching strategy with intelligent invalidation
    """

    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=False,  # Use binary for pickle
        )

        # Cache TTL configuration (in seconds)
        self.ttl_config = {
            # Static data - long cache
            "users": 3600,  # 1 hour
            "channels": 1800,  # 30 minutes
            # Dynamic data - medium cache
            "videos": 300,  # 5 minutes
            "analytics": 180,  # 3 minutes
            # Frequently changing - short cache
            "dashboard": 60,  # 1 minute
            "trending": 300,  # 5 minutes
            "costs": 120,  # 2 minutes
            # Real-time data - very short cache
            "queue_status": 10,  # 10 seconds
            "generation_status": 5,  # 5 seconds
        }

        # Cache key patterns for invalidation
        self.invalidation_patterns = {
            "user_update": ["users:*", "dashboard:*"],
            "channel_update": ["channels:*", "videos:*", "analytics:*"],
            "video_create": ["videos:*", "analytics:*", "costs:*", "dashboard:*"],
            "payment": ["users:*", "dashboard:*"],
        }

    def get_cache_key(self, prefix: str, identifier: Union[str, dict]) -> str:
        """Generate consistent cache key"""
        if isinstance(identifier, dict):
            # Sort dict keys for consistent hashing
            identifier = json.dumps(identifier, sort_keys=True)

        # Create hash for long identifiers
        if len(str(identifier)) > 50:
            identifier = hashlib.md5(str(identifier).encode()).hexdigest()

        return f"cache:{prefix}:{identifier}"

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            cached = await self.redis_client.get(key)
            if cached:
                return pickle.loads(cached)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        try:
            serialized = pickle.dumps(value)
            if ttl:
                await self.redis_client.setex(key, ttl, serialized)
            else:
                await self.redis_client.set(key, serialized)
        except Exception as e:
            logger.error(f"Cache set error: {e}")

    async def delete(self, pattern: str):
        """Delete cache entries matching pattern"""
        try:
            cursor = 0
            while True:
                cursor, keys = await self.redis_client.scan(
                    cursor, match=f"cache:{pattern}", count=100
                )
                if keys:
                    await self.redis_client.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:
            logger.error(f"Cache delete error: {e}")

    async def invalidate(self, event: str):
        """Invalidate related cache entries based on event"""
        patterns = self.invalidation_patterns.get(event, [])
        for pattern in patterns:
            await self.delete(pattern)
            logger.info(f"Invalidated cache pattern: {pattern}")

    def cache_endpoint(
        self,
        ttl: Optional[int] = None,
        key_prefix: Optional[str] = None,
        vary_on: list = None,
    ):
        """
        Decorator for caching endpoint responses

        Args:
            ttl: Cache time-to-live in seconds
            key_prefix: Custom cache key prefix
            vary_on: List of request attributes to vary cache on
        """

        def decorator(func):
            @wraps(func)
            async def wrapper(request: Request, *args, **kwargs):
                # Determine cache key prefix
                if key_prefix:
                    prefix = key_prefix
                else:
                    # Use endpoint path as prefix
                    prefix = request.url.path.replace("/", ":")

                # Build cache key components
                key_parts = [prefix]

                # Add vary_on parameters
                if vary_on:
                    for param in vary_on:
                        if param == "user":
                            # Add user ID if authenticated
                            if hasattr(request.state, "user_id"):
                                key_parts.append(f"user:{request.state.user_id}")
                        elif param == "query":
                            # Add query parameters
                            if request.url.query:
                                key_parts.append(f"query:{request.url.query}")
                        elif param in kwargs:
                            # Add path parameters
                            key_parts.append(f"{param}:{kwargs[param]}")

                # Generate cache key
                cache_key = ":".join(key_parts)

                # Try to get from cache
                cached = await self.get(cache_key)
                if cached is not None:
                    # Add cache hit header
                    response = Response(
                        content=json.dumps(cached),
                        media_type="application/json",
                        headers={"X-Cache": "HIT"},
                    )
                    return cached

                # Execute function
                result = await func(request, *args, **kwargs)

                # Determine TTL
                cache_ttl = ttl
                if not cache_ttl:
                    # Auto-determine based on endpoint
                    for pattern, pattern_ttl in self.ttl_config.items():
                        if pattern in prefix:
                            cache_ttl = pattern_ttl
                            break
                    else:
                        cache_ttl = 60  # Default 1 minute

                # Cache result
                await self.set(cache_key, result, cache_ttl)

                return result

            return wrapper

        return decorator

    async def warm_cache(self):
        """Pre-populate cache with frequently accessed data"""
        logger.info("Warming cache...")

        # This would typically load frequently accessed data
        # For example, trending topics, popular channels, etc.

        # Example: Cache trending topics
        trending_data = {"topics": ["AI", "Technology", "Gaming"]}  # Placeholder
        await self.set(
            self.get_cache_key("trending", "topics"),
            trending_data,
            self.ttl_config["trending"],
        )

        logger.info("Cache warming complete")

    async def get_stats(self) -> dict:
        """Get cache statistics"""
        try:
            info = await self.redis_client.info("stats")
            return {
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": info.get("keyspace_hits", 0)
                / max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1), 1),
                "used_memory": info.get("used_memory_human", "0"),
                "connected_clients": info.get("connected_clients", 0),
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}


# Global cache instance
cache_strategy = CacheStrategy()


# Convenient decorators
def cache_result(ttl: int = 300, key_prefix: Optional[str] = None):
    """Simple function result caching"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            cache_key = hashlib.md5(cache_key.encode()).hexdigest()

            if key_prefix:
                cache_key = f"{key_prefix}:{cache_key}"

            # Try cache
            cached = await cache_strategy.get(f"cache:func:{cache_key}")
            if cached is not None:
                return cached

            # Execute function
            result = await func(*args, **kwargs)

            # Cache result
            await cache_strategy.set(f"cache:func:{cache_key}", result, ttl)

            return result

        return wrapper

    return decorator


# Cache invalidation helpers
async def invalidate_user_cache(user_id: int):
    """Invalidate all cache entries for a user"""
    await cache_strategy.delete(f"*:user:{user_id}:*")


async def invalidate_channel_cache(channel_id: int):
    """Invalidate all cache entries for a channel"""
    await cache_strategy.delete(f"*:channel:{channel_id}:*")


async def invalidate_video_cache(video_id: int):
    """Invalidate all cache entries for a video"""
    await cache_strategy.delete(f"*:video:{video_id}:*")
