"""
Cache Module for YTEmpire Services
Provides caching functionality for services that require app.core.cache
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import redis.asyncio as redis
from functools import wraps

from app.core.config import settings

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Central cache manager for YTEmpire services
    """

    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.REDIS_URL or "redis://localhost:6379/0"
        self.redis_client: Optional[redis.Redis] = None
        self.default_ttl = 3600  # 1 hour default TTL

    async def initialize(self):
        """Initialize cache connection"""
        try:
            self.redis_client = await redis.from_url(
                self.redis_url, encoding="utf-8", decode_responses=True
            )
            # Test connection
            await self.redis_client.ping()
            logger.info("Cache manager initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize cache: {e}. Using fallback cache.")
            self.redis_client = None

    async def shutdown(self):
        """Shutdown cache connection"""
        if self.redis_client:
            await self.redis_client.close()

    async def get(self, key: str) -> Any:
        """Get value from cache"""
        try:
            if not self.redis_client:
                return None

            value = await self.redis_client.get(key)
            if value is None:
                return None

            # Try to parse JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value

        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache"""
        try:
            if not self.redis_client:
                return False

            ttl = ttl or self.default_ttl

            # Serialize value
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value)
            else:
                serialized_value = str(value)

            await self.redis_client.setex(key, ttl, serialized_value)
            return True

        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            if not self.redis_client:
                return False

            await self.redis_client.delete(key)
            return True

        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            if not self.redis_client:
                return False

            return bool(await self.redis_client.exists(key))

        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False

    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern"""
        try:
            if not self.redis_client:
                return 0

            keys = await self.redis_client.keys(pattern)
            if keys:
                return await self.redis_client.delete(*keys)
            return 0

        except Exception as e:
            logger.error(f"Cache clear pattern error for {pattern}: {e}")
            return 0

    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter in cache"""
        try:
            if not self.redis_client:
                return 0

            return await self.redis_client.incrby(key, amount)

        except Exception as e:
            logger.error(f"Cache increment error for key {key}: {e}")
            return 0

    async def set_hash(
        self, key: str, mapping: Dict[str, Any], ttl: int = None
    ) -> bool:
        """Set hash in cache"""
        try:
            if not self.redis_client:
                return False

            # Serialize values in mapping
            serialized_mapping = {}
            for k, v in mapping.items():
                if isinstance(v, (dict, list)):
                    serialized_mapping[k] = json.dumps(v)
                else:
                    serialized_mapping[k] = str(v)

            await self.redis_client.hset(key, mapping=serialized_mapping)

            if ttl:
                await self.redis_client.expire(key, ttl)

            return True

        except Exception as e:
            logger.error(f"Cache set hash error for key {key}: {e}")
            return False

    async def get_hash(self, key: str, field: str = None) -> Any:
        """Get hash or field from cache"""
        try:
            if not self.redis_client:
                return None

            if field:
                value = await self.redis_client.hget(key, field)
                if value is None:
                    return None
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value
            else:
                hash_data = await self.redis_client.hgetall(key)
                if not hash_data:
                    return None

                # Deserialize values
                result = {}
                for k, v in hash_data.items():
                    try:
                        result[k] = json.loads(v)
                    except (json.JSONDecodeError, TypeError):
                        result[k] = v

                return result

        except Exception as e:
            logger.error(f"Cache get hash error for key {key}: {e}")
            return None


# Global cache manager instance
cache_manager = CacheManager()
cache_service = cache_manager  # Alias for backward compatibility


def cached(ttl: int = 3600, key_prefix: str = ""):
    """
    Decorator for caching function results
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}"

            # Try to get from cache
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function
            result = await func(*args, **kwargs)

            # Cache result
            await cache_manager.set(cache_key, result, ttl)

            return result

        return wrapper

    return decorator


# Convenience functions for backward compatibility
async def get(key: str) -> Any:
    """Get value from cache"""
    return await cache_manager.get(key)


async def set(key: str, value: Any, ttl: int = None) -> bool:
    """Set value in cache"""
    return await cache_manager.set(key, value, ttl)


async def delete(key: str) -> bool:
    """Delete key from cache"""
    return await cache_manager.delete(key)


async def clear_pattern(pattern: str) -> int:
    """Clear all keys matching pattern"""
    return await cache_manager.clear_pattern(pattern)


# Export commonly used items
__all__ = [
    "CacheManager",
    "cache_manager",
    "cache_service",
    "cached",
    "get",
    "set",
    "delete",
    "clear_pattern",
]
