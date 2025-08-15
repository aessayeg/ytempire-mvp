"""
Advanced Caching Layer
Multi-tier caching with Redis, Memcached, and CDN integration
"""

import asyncio
import json
import hashlib
import pickle
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import redis.asyncio as redis
import aiomcache
from functools import wraps
import logging
from prometheus_client import Counter, Histogram, Gauge

# Metrics
cache_hits = Counter('cache_hits_total', 'Total cache hits', ['cache_tier', 'cache_key_type'])
cache_misses = Counter('cache_misses_total', 'Total cache misses', ['cache_tier', 'cache_key_type'])
cache_latency = Histogram('cache_operation_latency', 'Cache operation latency', ['operation', 'cache_tier'])
cache_memory_usage = Gauge('cache_memory_usage_bytes', 'Cache memory usage', ['cache_tier'])
cache_evictions = Counter('cache_evictions_total', 'Total cache evictions', ['cache_tier', 'reason'])

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Cache configuration"""
    ttl: int = 3600  # Default TTL in seconds
    max_size: int = 10000  # Maximum number of entries
    eviction_policy: str = 'LRU'  # LRU, LFU, FIFO
    compression: bool = False
    serialization: str = 'json'  # json, pickle, msgpack
    tags: List[str] = field(default_factory=list)
    warming_enabled: bool = False
    invalidation_strategy: str = 'immediate'  # immediate, lazy, scheduled

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    ttl: int
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    size: int = 0
    tags: List[str] = field(default_factory=list)
    version: int = 1

class CacheTier:
    """Base cache tier interface"""
    
    async def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        raise NotImplementedError
    
    async def delete(self, key: str) -> bool:
        raise NotImplementedError
    
    async def exists(self, key: str) -> bool:
        raise NotImplementedError
    
    async def clear(self) -> bool:
        raise NotImplementedError

class L1Cache(CacheTier):
    """In-memory L1 cache (local)"""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.access_order: List[str] = []
    
    async def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            entry = self.cache[key]
            
            # Check TTL
            if datetime.now() > entry.created_at + timedelta(seconds=entry.ttl):
                await self.delete(key)
                return None
            
            # Update access info
            entry.accessed_at = datetime.now()
            entry.access_count += 1
            
            # Update LRU order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            cache_hits.labels(cache_tier='L1', cache_key_type='general').inc()
            return entry.value
        
        cache_misses.labels(cache_tier='L1', cache_key_type='general').inc()
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        # Evict if at capacity
        if len(self.cache) >= self.max_size:
            await self._evict()
        
        # Calculate size
        size = len(json.dumps(value) if isinstance(value, (dict, list)) else str(value))
        
        self.cache[key] = CacheEntry(
            key=key,
            value=value,
            ttl=ttl,
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            size=size
        )
        
        self.access_order.append(key)
        
        # Update metrics
        total_size = sum(entry.size for entry in self.cache.values())
        cache_memory_usage.labels(cache_tier='L1').set(total_size)
        
        return True
    
    async def delete(self, key: str) -> bool:
        if key in self.cache:
            del self.cache[key]
            if key in self.access_order:
                self.access_order.remove(key)
            return True
        return False
    
    async def exists(self, key: str) -> bool:
        return key in self.cache
    
    async def clear(self) -> bool:
        self.cache.clear()
        self.access_order.clear()
        cache_memory_usage.labels(cache_tier='L1').set(0)
        return True
    
    async def _evict(self):
        """Evict least recently used entry"""
        if self.access_order:
            lru_key = self.access_order[0]
            await self.delete(lru_key)
            cache_evictions.labels(cache_tier='L1', reason='capacity').inc()

class L2Cache(CacheTier):
    """Redis L2 cache (distributed)"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def get(self, key: str) -> Optional[Any]:
        try:
            value = await self.redis.get(key)
            if value:
                cache_hits.labels(cache_tier='L2', cache_key_type='general').inc()
                return json.loads(value) if value else None
            else:
                cache_misses.labels(cache_tier='L2', cache_key_type='general').inc()
                return None
        except Exception as e:
            logger.error(f"L2 cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        try:
            serialized = json.dumps(value)
            await self.redis.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.error(f"L2 cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        try:
            result = await self.redis.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"L2 cache delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        try:
            return await self.redis.exists(key) > 0
        except Exception as e:
            logger.error(f"L2 cache exists error: {e}")
            return False
    
    async def clear(self) -> bool:
        try:
            await self.redis.flushdb()
            return True
        except Exception as e:
            logger.error(f"L2 cache clear error: {e}")
            return False

class L3Cache(CacheTier):
    """Memcached L3 cache"""
    
    def __init__(self, memcached_host: str = 'localhost', memcached_port: int = 11211):
        self.client = aiomcache.Client(memcached_host, memcached_port)
    
    async def get(self, key: str) -> Optional[Any]:
        try:
            value = await self.client.get(key.encode())
            if value:
                cache_hits.labels(cache_tier='L3', cache_key_type='general').inc()
                return json.loads(value.decode())
            else:
                cache_misses.labels(cache_tier='L3', cache_key_type='general').inc()
                return None
        except Exception as e:
            logger.error(f"L3 cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        try:
            serialized = json.dumps(value).encode()
            await self.client.set(key.encode(), serialized, exptime=ttl)
            return True
        except Exception as e:
            logger.error(f"L3 cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        try:
            await self.client.delete(key.encode())
            return True
        except Exception as e:
            logger.error(f"L3 cache delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        try:
            value = await self.client.get(key.encode())
            return value is not None
        except Exception as e:
            logger.error(f"L3 cache exists error: {e}")
            return False
    
    async def clear(self) -> bool:
        try:
            await self.client.flush_all()
            return True
        except Exception as e:
            logger.error(f"L3 cache clear error: {e}")
            return False

class AdvancedCacheManager:
    """Multi-tier cache manager with advanced features"""
    
    def __init__(self,
                 redis_client: redis.Redis = None,
                 memcached_host: str = 'localhost',
                 memcached_port: int = 11211,
                 enable_l1: bool = True,
                 enable_l2: bool = True,
                 enable_l3: bool = False):
        
        self.tiers = []
        
        # Initialize cache tiers
        if enable_l1:
            self.l1_cache = L1Cache()
            self.tiers.append(self.l1_cache)
        
        if enable_l2 and redis_client:
            self.l2_cache = L2Cache(redis_client)
            self.tiers.append(self.l2_cache)
        
        if enable_l3:
            self.l3_cache = L3Cache(memcached_host, memcached_port)
            self.tiers.append(self.l3_cache)
        
        # Cache configurations
        self.configs: Dict[str, CacheConfig] = {}
        
        # Tag-based invalidation
        self.tag_keys: Dict[str, List[str]] = {}
        
        # Cache warming queue
        self.warming_queue: List[Dict[str, Any]] = []
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0
        }
    
    async def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache with multi-tier fallback
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        start_time = datetime.now()
        
        # Try each tier
        for i, tier in enumerate(self.tiers):
            value = await tier.get(key)
            
            if value is not None:
                # Backfill to higher tiers
                for j in range(i):
                    await self.tiers[j].set(key, value)
                
                # Record latency
                latency = (datetime.now() - start_time).total_seconds()
                cache_latency.labels(operation='get', cache_tier=f'L{i+1}').observe(latency)
                
                self.stats['hits'] += 1
                return value
        
        self.stats['misses'] += 1
        return default
    
    async def set(self, key: str, value: Any, config: CacheConfig = None) -> bool:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            config: Cache configuration
            
        Returns:
            Success status
        """
        if config is None:
            config = CacheConfig()
        
        start_time = datetime.now()
        
        # Store configuration
        self.configs[key] = config
        
        # Apply compression if enabled
        if config.compression:
            value = self._compress(value)
        
        # Set in all tiers
        success = True
        for tier in self.tiers:
            tier_success = await tier.set(key, value, config.ttl)
            success = success and tier_success
        
        # Handle tags
        if config.tags:
            for tag in config.tags:
                if tag not in self.tag_keys:
                    self.tag_keys[tag] = []
                self.tag_keys[tag].append(key)
        
        # Record latency
        latency = (datetime.now() - start_time).total_seconds()
        cache_latency.labels(operation='set', cache_tier='all').observe(latency)
        
        self.stats['sets'] += 1
        return success
    
    async def delete(self, key: str) -> bool:
        """Delete value from all cache tiers"""
        success = True
        for tier in self.tiers:
            tier_success = await tier.delete(key)
            success = success and tier_success
        
        # Remove from tag mapping
        if key in self.configs and self.configs[key].tags:
            for tag in self.configs[key].tags:
                if tag in self.tag_keys and key in self.tag_keys[tag]:
                    self.tag_keys[tag].remove(key)
        
        self.stats['deletes'] += 1
        return success
    
    async def invalidate_by_tag(self, tag: str) -> int:
        """
        Invalidate all cache entries with a specific tag
        
        Args:
            tag: Tag to invalidate
            
        Returns:
            Number of entries invalidated
        """
        if tag not in self.tag_keys:
            return 0
        
        keys_to_invalidate = self.tag_keys[tag].copy()
        count = 0
        
        for key in keys_to_invalidate:
            if await self.delete(key):
                count += 1
        
        # Clear tag mapping
        del self.tag_keys[tag]
        
        logger.info(f"Invalidated {count} cache entries with tag '{tag}'")
        return count
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate cache entries matching a pattern
        
        Args:
            pattern: Key pattern (supports wildcards)
            
        Returns:
            Number of entries invalidated
        """
        count = 0
        
        # Only works with Redis L2 cache
        if hasattr(self, 'l2_cache'):
            async for key in self.l2_cache.redis.scan_iter(match=pattern):
                if await self.delete(key.decode()):
                    count += 1
        
        logger.info(f"Invalidated {count} cache entries matching pattern '{pattern}'")
        return count
    
    async def warm_cache(self, key: str, loader: Callable, ttl: int = 3600):
        """
        Proactively warm cache with data
        
        Args:
            key: Cache key
            loader: Function to load data
            ttl: Time to live
        """
        try:
            value = await loader() if asyncio.iscoroutinefunction(loader) else loader()
            await self.set(key, value, CacheConfig(ttl=ttl, warming_enabled=True))
            logger.info(f"Warmed cache for key '{key}'")
        except Exception as e:
            logger.error(f"Failed to warm cache for key '{key}': {e}")
    
    async def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values from cache
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary of key-value pairs
        """
        results = {}
        
        # Try to use Redis MGET for efficiency
        if hasattr(self, 'l2_cache'):
            values = await self.l2_cache.redis.mget(keys)
            for key, value in zip(keys, values):
                if value:
                    results[key] = json.loads(value)
        else:
            # Fallback to individual gets
            for key in keys:
                value = await self.get(key)
                if value is not None:
                    results[key] = value
        
        return results
    
    async def batch_set(self, items: Dict[str, Any], ttl: int = 3600) -> bool:
        """
        Set multiple values in cache
        
        Args:
            items: Dictionary of key-value pairs
            ttl: Time to live for all items
            
        Returns:
            Success status
        """
        success = True
        
        # Use pipeline for Redis
        if hasattr(self, 'l2_cache'):
            pipe = self.l2_cache.redis.pipeline()
            for key, value in items.items():
                pipe.setex(key, ttl, json.dumps(value))
            
            results = await pipe.execute()
            success = all(results)
        else:
            # Fallback to individual sets
            for key, value in items.items():
                item_success = await self.set(key, value, CacheConfig(ttl=ttl))
                success = success and item_success
        
        return success
    
    def _compress(self, value: Any) -> bytes:
        """Compress value for storage"""
        import zlib
        serialized = pickle.dumps(value)
        return zlib.compress(serialized)
    
    def _decompress(self, data: bytes) -> Any:
        """Decompress value from storage"""
        import zlib
        decompressed = zlib.decompress(data)
        return pickle.loads(decompressed)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = self.stats['hits'] / (self.stats['hits'] + self.stats['misses']) if (self.stats['hits'] + self.stats['misses']) > 0 else 0
        
        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': hit_rate,
            'sets': self.stats['sets'],
            'deletes': self.stats['deletes'],
            'l1_size': len(self.l1_cache.cache) if hasattr(self, 'l1_cache') else 0,
            'tag_count': len(self.tag_keys),
            'total_tagged_keys': sum(len(keys) for keys in self.tag_keys.values())
        }
    
    async def clear_all(self) -> bool:
        """Clear all cache tiers"""
        success = True
        for tier in self.tiers:
            tier_success = await tier.clear()
            success = success and tier_success
        
        self.configs.clear()
        self.tag_keys.clear()
        self.stats = {'hits': 0, 'misses': 0, 'sets': 0, 'deletes': 0}
        
        return success

def cache_decorator(ttl: int = 3600, tags: List[str] = None, key_prefix: str = None):
    """
    Decorator for caching function results
    
    Args:
        ttl: Time to live in seconds
        tags: Cache tags for invalidation
        key_prefix: Optional key prefix
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix or func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}_{v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)
            
            # Try to get from cache
            cache_manager = kwargs.pop('_cache_manager', None)
            if cache_manager:
                cached_value = await cache_manager.get(cache_key)
                if cached_value is not None:
                    return cached_value
            
            # Execute function
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            # Store in cache
            if cache_manager and result is not None:
                config = CacheConfig(ttl=ttl, tags=tags or [])
                await cache_manager.set(cache_key, result, config)
            
            return result
        
        return wrapper
    return decorator

class CacheWarmer:
    """Background cache warming service"""
    
    def __init__(self, cache_manager: AdvancedCacheManager):
        self.cache_manager = cache_manager
        self.warming_tasks: List[Dict[str, Any]] = []
        self.running = False
    
    def add_warming_task(self, key: str, loader: Callable, interval: int = 3600):
        """Add a cache warming task"""
        self.warming_tasks.append({
            'key': key,
            'loader': loader,
            'interval': interval,
            'last_run': None
        })
    
    async def start(self):
        """Start cache warming service"""
        self.running = True
        
        while self.running:
            for task in self.warming_tasks:
                if task['last_run'] is None or \
                   datetime.now() - task['last_run'] > timedelta(seconds=task['interval']):
                    
                    await self.cache_manager.warm_cache(
                        task['key'],
                        task['loader'],
                        task['interval']
                    )
                    task['last_run'] = datetime.now()
            
            await asyncio.sleep(60)  # Check every minute
    
    def stop(self):
        """Stop cache warming service"""
        self.running = False

# CDN Integration
class CDNCache:
    """CDN cache integration for static content"""
    
    def __init__(self, cdn_provider: str, config: Dict[str, Any]):
        self.provider = cdn_provider
        self.config = config
    
    async def purge(self, urls: List[str]) -> bool:
        """Purge URLs from CDN cache"""
        if self.provider == 'cloudflare':
            return await self._purge_cloudflare(urls)
        elif self.provider == 'cloudfront':
            return await self._purge_cloudfront(urls)
        elif self.provider == 'fastly':
            return await self._purge_fastly(urls)
        else:
            logger.warning(f"Unknown CDN provider: {self.provider}")
            return False
    
    async def _purge_cloudflare(self, urls: List[str]) -> bool:
        """Purge Cloudflare cache"""
        import aiohttp
        
        headers = {
            'X-Auth-Email': self.config['email'],
            'X-Auth-Key': self.config['api_key'],
            'Content-Type': 'application/json'
        }
        
        data = {'files': urls}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"https://api.cloudflare.com/client/v4/zones/{self.config['zone_id']}/purge_cache",
                headers=headers,
                json=data
            ) as response:
                return response.status == 200
    
    async def _purge_cloudfront(self, urls: List[str]) -> bool:
        """Purge CloudFront cache"""
        import boto3
        
        client = boto3.client('cloudfront')
        
        try:
            response = client.create_invalidation(
                DistributionId=self.config['distribution_id'],
                InvalidationBatch={
                    'Paths': {
                        'Quantity': len(urls),
                        'Items': urls
                    },
                    'CallerReference': str(datetime.now().timestamp())
                }
            )
            return response['ResponseMetadata']['HTTPStatusCode'] == 201
        except Exception as e:
            logger.error(f"CloudFront purge error: {e}")
            return False
    
    async def _purge_fastly(self, urls: List[str]) -> bool:
        """Purge Fastly cache"""
        import aiohttp
        
        headers = {
            'Fastly-Key': self.config['api_key']
        }
        
        success = True
        async with aiohttp.ClientSession() as session:
            for url in urls:
                async with session.post(
                    f"https://api.fastly.com/purge/{url}",
                    headers=headers
                ) as response:
                    if response.status != 200:
                        success = False
        
        return success

# Example usage
async def main():
    # Initialize Redis client
    redis_client = await redis.Redis(host='localhost', port=6379)
    
    # Create cache manager
    cache_manager = AdvancedCacheManager(
        redis_client=redis_client,
        enable_l1=True,
        enable_l2=True,
        enable_l3=False
    )
    
    # Set value with tags
    config = CacheConfig(
        ttl=3600,
        tags=['user_data', 'profile'],
        compression=True
    )
    
    await cache_manager.set('user:123', {'name': 'John', 'age': 30}, config)
    
    # Get value
    user_data = await cache_manager.get('user:123')
    print(f"User data: {user_data}")
    
    # Batch operations
    items = {
        'product:1': {'name': 'Product 1', 'price': 100},
        'product:2': {'name': 'Product 2', 'price': 200}
    }
    await cache_manager.batch_set(items)
    
    products = await cache_manager.batch_get(['product:1', 'product:2'])
    print(f"Products: {products}")
    
    # Invalidate by tag
    invalidated = await cache_manager.invalidate_by_tag('user_data')
    print(f"Invalidated {invalidated} entries")
    
    # Get statistics
    stats = await cache_manager.get_stats()
    print(f"Cache stats: {stats}")
    
    # Cache warming
    warmer = CacheWarmer(cache_manager)
    
    async def load_expensive_data():
        # Simulate expensive operation
        await asyncio.sleep(1)
        return {'expensive': 'data'}
    
    warmer.add_warming_task('expensive_data', load_expensive_data, interval=300)
    
    # Start warmer in background
    # asyncio.create_task(warmer.start())

if __name__ == "__main__":
    asyncio.run(main())