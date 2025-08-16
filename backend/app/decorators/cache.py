"""
Cache Decorators for YTEmpire
Provides centralized caching functionality with Redis backend
"""

import hashlib
import json
import logging
from functools import wraps
from typing import Any, Callable, Optional, Union
from datetime import timedelta
import asyncio
import pickle

from redis import Redis
from app.core.config import settings

logger = logging.getLogger(__name__)

# Redis client initialization
redis_client = Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    password=settings.REDIS_PASSWORD,
    decode_responses=False,  # Binary data for pickle
)


def cache_key_generator(prefix: str, *args, **kwargs) -> str:
    """
    Generate a unique cache key based on function arguments

    Args:
        prefix: Cache key prefix
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Unique cache key string
    """
    key_data = {"args": args, "kwargs": kwargs}
    key_hash = hashlib.md5(
        json.dumps(key_data, sort_keys=True, default=str).encode()
    ).hexdigest()
    return f"{prefix}:{key_hash}"


def cache(
    ttl: Union[int, timedelta] = 300,
    prefix: Optional[str] = None,
    key_builder: Optional[Callable] = None,
):
    """
    Cache decorator for synchronous functions

    Args:
        ttl: Time to live in seconds or timedelta
        prefix: Cache key prefix (defaults to function name)
        key_builder: Custom key builder function

    Example:
        @cache(ttl=3600, prefix="user_data")
        def get_user_data(user_id):
            return expensive_operation(user_id)
    """

    def decorator(func: Callable) -> Callable:
        cache_prefix = prefix or f"{func.__module__}.{func.__name__}"
        cache_ttl = ttl.total_seconds() if isinstance(ttl, timedelta) else ttl

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                cache_key = cache_key_generator(cache_prefix, *args, **kwargs)

            try:
                # Try to get from cache
                cached_value = redis_client.get(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache hit for key: {cache_key}")
                    return pickle.loads(cached_value)

                logger.debug(f"Cache miss for key: {cache_key}")
            except Exception as e:
                logger.warning(f"Cache get failed: {str(e)}")

            # Execute function
            result = func(*args, **kwargs)

            try:
                # Store in cache
                redis_client.setex(cache_key, int(cache_ttl), pickle.dumps(result))
                logger.debug(f"Cached result for key: {cache_key}")
            except Exception as e:
                logger.warning(f"Cache set failed: {str(e)}")

            return result

        wrapper.cache_clear = lambda: cache_clear(cache_prefix)
        wrapper.cache_prefix = cache_prefix
        return wrapper

    return decorator


def async_cache(
    ttl: Union[int, timedelta] = 300,
    prefix: Optional[str] = None,
    key_builder: Optional[Callable] = None,
):
    """
    Cache decorator for async functions

    Args:
        ttl: Time to live in seconds or timedelta
        prefix: Cache key prefix (defaults to function name)
        key_builder: Custom key builder function

    Example:
        @async_cache(ttl=timedelta(hours=1))
        async def get_analytics(channel_id):
            return await expensive_async_operation(channel_id)
    """

    def decorator(func: Callable) -> Callable:
        cache_prefix = prefix or f"{func.__module__}.{func.__name__}"
        cache_ttl = ttl.total_seconds() if isinstance(ttl, timedelta) else ttl

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                cache_key = cache_key_generator(cache_prefix, *args, **kwargs)

            try:
                # Try to get from cache
                cached_value = await asyncio.get_event_loop().run_in_executor(
                    None, redis_client.get, cache_key
                )
                if cached_value is not None:
                    logger.debug(f"Cache hit for key: {cache_key}")
                    return pickle.loads(cached_value)

                logger.debug(f"Cache miss for key: {cache_key}")
            except Exception as e:
                logger.warning(f"Cache get failed: {str(e)}")

            # Execute function
            result = await func(*args, **kwargs)

            try:
                # Store in cache
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    redis_client.setex,
                    cache_key,
                    int(cache_ttl),
                    pickle.dumps(result),
                )
                logger.debug(f"Cached result for key: {cache_key}")
            except Exception as e:
                logger.warning(f"Cache set failed: {str(e)}")

            return result

        wrapper.cache_clear = lambda: cache_clear(cache_prefix)
        wrapper.cache_prefix = cache_prefix
        return wrapper

    return decorator


def cache_clear(prefix: str):
    """
    Clear all cache entries with given prefix

    Args:
        prefix: Cache key prefix to clear
    """
    try:
        pattern = f"{prefix}:*"
        keys = redis_client.keys(pattern)
        if keys:
            redis_client.delete(*keys)
            logger.info(f"Cleared {len(keys)} cache entries for prefix: {prefix}")
    except Exception as e:
        logger.error(f"Failed to clear cache: {str(e)}")


def invalidate_cache(pattern: str):
    """
    Invalidate cache entries matching pattern

    Args:
        pattern: Redis key pattern (e.g., "user:*" or "*analytics*")
    """
    try:
        keys = redis_client.keys(pattern)
        if keys:
            redis_client.delete(*keys)
            logger.info(f"Invalidated {len(keys)} cache entries matching: {pattern}")
    except Exception as e:
        logger.error(f"Failed to invalidate cache: {str(e)}")


# Specialized cache decorators


def trending_cache(ttl: int = 3600):
    """
    Cache decorator for trending topics (1 hour default)

    Example:
        @trending_cache()
        def get_trending_topics():
            return fetch_trending_from_youtube()
    """
    return cache(ttl=ttl, prefix="trending")


def analytics_cache(ttl: int = 900):
    """
    Cache decorator for analytics data (15 minutes default)

    Example:
        @analytics_cache()
        async def get_channel_analytics(channel_id):
            return await fetch_analytics(channel_id)
    """
    return async_cache(ttl=ttl, prefix="analytics")


def user_cache(ttl: int = 1800):
    """
    Cache decorator for user data (30 minutes default)

    Example:
        @user_cache()
        def get_user_profile(user_id):
            return fetch_user_from_db(user_id)
    """
    return cache(ttl=ttl, prefix="user")


def video_cache(ttl: int = 3600):
    """
    Cache decorator for video data (1 hour default)

    Example:
        @video_cache()
        async def get_video_details(video_id):
            return await fetch_video_from_db(video_id)
    """
    return async_cache(ttl=ttl, prefix="video")


class CacheManager:
    """
    Central cache management class
    """

    @staticmethod
    def get_cache_stats() -> dict:
        """Get cache statistics"""
        try:
            info = redis_client.info()
            return {
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses"),
                "hit_rate": (
                    info.get("keyspace_hits", 0)
                    / (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1))
                )
                * 100,
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {str(e)}")
            return {}

    @staticmethod
    def flush_all():
        """Flush all cache entries (use with caution)"""
        try:
            redis_client.flushdb()
            logger.warning("Flushed all cache entries")
        except Exception as e:
            logger.error(f"Failed to flush cache: {str(e)}")

    @staticmethod
    def set_with_ttl(key: str, value: Any, ttl: int):
        """Set a value with TTL"""
        try:
            redis_client.setex(key, ttl, pickle.dumps(value))
        except Exception as e:
            logger.error(f"Failed to set cache value: {str(e)}")

    @staticmethod
    def get(key: str) -> Optional[Any]:
        """Get a value from cache"""
        try:
            value = redis_client.get(key)
            return pickle.loads(value) if value else None
        except Exception as e:
            logger.error(f"Failed to get cache value: {str(e)}")
            return None


# Export main decorators and utilities
__all__ = [
    "cache",
    "async_cache",
    "cache_clear",
    "invalidate_cache",
    "trending_cache",
    "analytics_cache",
    "user_cache",
    "video_cache",
    "CacheManager",
]
