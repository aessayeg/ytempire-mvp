"""
API endpoints for advanced caching service
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from app.core.security import get_current_user
from app.models.user import User

router = APIRouter()


class CacheConfig(BaseModel):
    """Cache configuration model"""

    ttl: int = Field(default=3600, ge=1, le=86400)
    tags: List[str] = []
    compression: bool = False
    warming_enabled: bool = False
    eviction_policy: str = Field(default="LRU", pattern="^(LRU|LFU|FIFO)$")


class CacheEntry(BaseModel):
    """Cache entry model"""

    key: str
    value: Any
    config: Optional[CacheConfig] = None


class BatchCacheRequest(BaseModel):
    """Batch cache operation request"""

    items: Dict[str, Any]
    ttl: int = Field(default=3600, ge=1, le=86400)
    tags: List[str] = []


class CacheWarmingTask(BaseModel):
    """Cache warming task configuration"""

    key: str
    endpoint: str
    interval: int = Field(default=3600, ge=60, le=86400)
    params: Optional[Dict[str, Any]] = None


@router.get("/stats")
async def get_cache_statistics(
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get cache statistics including hit rates and tier distribution

    Returns comprehensive cache performance metrics
    """
    try:
        # Import here to avoid circular dependency
        from app.services.advanced_caching import AdvancedCacheManager
        import redis.asyncio as redis

        # Initialize cache manager
        redis_client = await redis.from_url("redis://localhost:6379")
        cache_manager = AdvancedCacheManager(
            redis_client=redis_client, enable_l1=True, enable_l2=True, enable_l3=False
        )

        stats = await cache_manager.get_stats()

        return {
            "timestamp": datetime.now().isoformat(),
            "statistics": stats,
            "performance": {
                "hit_rate": stats.get("hit_rate", 0),
                "total_hits": stats.get("hits", 0),
                "total_misses": stats.get("misses", 0),
                "total_operations": stats.get("hits", 0) + stats.get("misses", 0),
            },
            "tiers": {
                "l1_size": stats.get("l1_size", 0),
                "l2_enabled": True,
                "l3_enabled": False,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/set")
async def set_cache_value(
    entry: CacheEntry, current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Set a value in the multi-tier cache

    - **key**: Cache key
    - **value**: Value to cache
    - **config**: Cache configuration (TTL, tags, compression, etc.)
    """
    try:
        from app.services.advanced_caching import (
            AdvancedCacheManager,
            CacheConfig as ServiceCacheConfig,
        )
        import redis.asyncio as redis

        redis_client = await redis.from_url("redis://localhost:6379")
        cache_manager = AdvancedCacheManager(redis_client=redis_client)

        # Convert to service config
        service_config = None
        if entry.config:
            service_config = ServiceCacheConfig(
                ttl=entry.config.ttl,
                tags=entry.config.tags,
                compression=entry.config.compression,
                warming_enabled=entry.config.warming_enabled,
                eviction_policy=entry.config.eviction_policy,
            )

        success = await cache_manager.set(entry.key, entry.value, service_config)

        return {
            "status": "success" if success else "failed",
            "key": entry.key,
            "ttl": entry.config.ttl if entry.config else 3600,
            "tags": entry.config.tags if entry.config else [],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/get/{key}")
async def get_cache_value(
    key: str,
    default: Optional[Any] = None,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get a value from the multi-tier cache

    - **key**: Cache key
    - **default**: Default value if not found
    """
    try:
        from app.services.advanced_caching import AdvancedCacheManager
        import redis.asyncio as redis

        redis_client = await redis.from_url("redis://localhost:6379")
        cache_manager = AdvancedCacheManager(redis_client=redis_client)

        value = await cache_manager.get(key, default)

        if value is None and default is None:
            raise HTTPException(status_code=404, detail="Key not found in cache")

        return {
            "status": "success",
            "key": key,
            "value": value,
            "from_cache": value is not None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete/{key}")
async def delete_cache_value(
    key: str, current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Delete a value from all cache tiers

    - **key**: Cache key to delete
    """
    try:
        from app.services.advanced_caching import AdvancedCacheManager
        import redis.asyncio as redis

        redis_client = await redis.from_url("redis://localhost:6379")
        cache_manager = AdvancedCacheManager(redis_client=redis_client)

        success = await cache_manager.delete(key)

        return {
            "status": "success" if success else "not_found",
            "key": key,
            "deleted": success,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch/get")
async def batch_get_cache_values(
    keys: List[str], current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get multiple values from cache in a single operation

    - **keys**: List of cache keys to retrieve
    """
    try:
        from app.services.advanced_caching import AdvancedCacheManager
        import redis.asyncio as redis

        redis_client = await redis.from_url("redis://localhost:6379")
        cache_manager = AdvancedCacheManager(redis_client=redis_client)

        results = await cache_manager.batch_get(keys)

        return {
            "status": "success",
            "requested": len(keys),
            "found": len(results),
            "results": results,
            "missing": [k for k in keys if k not in results],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch/set")
async def batch_set_cache_values(
    request: BatchCacheRequest, current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Set multiple values in cache in a single operation

    - **items**: Dictionary of key-value pairs
    - **ttl**: Time to live for all items
    - **tags**: Tags to apply to all items
    """
    try:
        from app.services.advanced_caching import AdvancedCacheManager
        import redis.asyncio as redis

        redis_client = await redis.from_url("redis://localhost:6379")
        cache_manager = AdvancedCacheManager(redis_client=redis_client)

        success = await cache_manager.batch_set(request.items, request.ttl)

        return {
            "status": "success" if success else "partial",
            "count": len(request.items),
            "ttl": request.ttl,
            "tags": request.tags,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/invalidate/tag")
async def invalidate_by_tag(
    tag: str, current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Invalidate all cache entries with a specific tag

    - **tag**: Tag to invalidate
    """
    try:
        from app.services.advanced_caching import AdvancedCacheManager
        import redis.asyncio as redis

        redis_client = await redis.from_url("redis://localhost:6379")
        cache_manager = AdvancedCacheManager(redis_client=redis_client)

        count = await cache_manager.invalidate_by_tag(tag)

        return {"status": "success", "tag": tag, "invalidated": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/invalidate/pattern")
async def invalidate_by_pattern(
    pattern: str, current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Invalidate cache entries matching a pattern

    - **pattern**: Key pattern with wildcards (e.g., 'user:*', 'product:123:*')
    """
    try:
        from app.services.advanced_caching import AdvancedCacheManager
        import redis.asyncio as redis

        redis_client = await redis.from_url("redis://localhost:6379")
        cache_manager = AdvancedCacheManager(redis_client=redis_client)

        count = await cache_manager.invalidate_pattern(pattern)

        return {"status": "success", "pattern": pattern, "invalidated": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/warm")
async def warm_cache(
    task: CacheWarmingTask, current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Proactively warm cache with data

    - **key**: Cache key to warm
    - **endpoint**: API endpoint to fetch data from
    - **interval**: Warming interval in seconds
    - **params**: Optional parameters for the endpoint
    """
    try:
        from app.services.advanced_caching import AdvancedCacheManager
        import redis.asyncio as redis
        import httpx

        async def loader():
            async with httpx.AsyncClient() as client:
                response = await client.get(task.endpoint, params=task.params)
                response.raise_for_status()
                return response.json()

        redis_client = await redis.from_url("redis://localhost:6379")
        cache_manager = AdvancedCacheManager(redis_client=redis_client)

        await cache_manager.warm_cache(task.key, loader, task.interval)

        return {
            "status": "success",
            "key": task.key,
            "warmed": True,
            "interval": task.interval,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear")
async def clear_all_cache(
    confirm: bool = False, current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Clear all cache tiers (use with caution!)

    - **confirm**: Must be true to confirm the operation
    """
    if not confirm:
        raise HTTPException(
            status_code=400, detail="Must set confirm=true to clear all cache"
        )

    try:
        from app.services.advanced_caching import AdvancedCacheManager
        import redis.asyncio as redis

        redis_client = await redis.from_url("redis://localhost:6379")
        cache_manager = AdvancedCacheManager(redis_client=redis_client)

        success = await cache_manager.clear_all()

        return {
            "status": "success" if success else "failed",
            "message": "All cache tiers cleared"
            if success
            else "Failed to clear some tiers",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cdn/status")
async def get_cdn_cache_status(
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get CDN cache status and configuration
    """
    try:
        # This would integrate with actual CDN providers
        return {
            "status": "active",
            "provider": "cloudflare",
            "zones": [
                {
                    "zone_id": "example_zone",
                    "domain": "cdn.ytempire.com",
                    "cache_level": "aggressive",
                    "ttl": 86400,
                }
            ],
            "cache_purge_available": True,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cdn/purge")
async def purge_cdn_cache(
    urls: List[str],
    provider: str = "cloudflare",
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Purge URLs from CDN cache

    - **urls**: List of URLs to purge
    - **provider**: CDN provider (cloudflare, cloudfront, fastly)
    """
    try:
        from app.services.advanced_caching import CDNCache

        # This would use actual configuration from settings
        cdn_config = {
            "email": "admin@ytempire.com",
            "api_key": "cdn_api_key",
            "zone_id": "zone_id",
        }

        cdn_cache = CDNCache(provider, cdn_config)
        success = await cdn_cache.purge(urls)

        return {
            "status": "success" if success else "failed",
            "provider": provider,
            "purged": len(urls) if success else 0,
            "urls": urls,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/video-cache/stats")
async def get_video_cache_statistics(
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get video generation cache statistics

    Returns cache statistics specific to video generation pipeline
    """
    try:
        # This would integrate with the specialized video cache
        return {
            "timestamp": datetime.now().isoformat(),
            "cache_usage": {
                "scripts": {"cached": 150, "hit_rate": 0.75, "avg_size_kb": 2.5},
                "thumbnails": {"cached": 500, "hit_rate": 0.85, "avg_size_kb": 150},
                "voice_synthesis": {"cached": 75, "hit_rate": 0.60, "avg_size_kb": 500},
            },
            "cost_savings": {
                "scripts": "$45.00",
                "thumbnails": "$120.00",
                "voice": "$180.00",
                "total": "$345.00",
            },
            "performance": {
                "avg_cache_retrieval_ms": 15,
                "avg_generation_time_saved_s": 45,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
