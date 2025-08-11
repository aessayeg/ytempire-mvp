"""
Third-Party API Optimization Service
Handles caching, rate limiting, fallback strategies, and cost optimization for external APIs
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import hashlib
import time
from contextlib import asynccontextmanager
import aiohttp
from collections import defaultdict

import redis.asyncio as redis
from app.core.config import settings
from app.services.cost_tracking import cost_tracker

logger = logging.getLogger(__name__)

class APIProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic" 
    ELEVENLABS = "elevenlabs"
    GOOGLE_TTS = "google_tts"
    YOUTUBE = "youtube"
    STRIPE = "stripe"

class CacheStrategy(str, Enum):
    NONE = "none"
    SIMPLE = "simple"
    SMART = "smart"
    AGGRESSIVE = "aggressive"

@dataclass
class APIEndpointConfig:
    provider: APIProvider
    endpoint: str
    method: str
    rate_limit_per_minute: int
    rate_limit_per_hour: int
    cache_strategy: CacheStrategy
    cache_ttl_seconds: int
    cost_per_request: float
    timeout_seconds: int
    max_retries: int
    fallback_providers: List[APIProvider] = None
    priority_level: int = 5  # 1-10, higher = more important

@dataclass
class APICallMetrics:
    provider: APIProvider
    endpoint: str
    timestamp: datetime
    response_time_ms: float
    status_code: int
    cost: float
    cache_hit: bool
    retry_count: int

class APIOptimizer:
    def __init__(self):
        self.redis_client = None
        self.session_pool = {}
        self.rate_limits = defaultdict(lambda: defaultdict(int))
        self.rate_limit_reset = defaultdict(lambda: defaultdict(datetime))
        self.metrics = []
        self.circuit_breakers = defaultdict(lambda: {"failures": 0, "last_failure": None})
        
        # Load API configurations
        self.api_configs = self._load_api_configs()
        
    async def _get_redis_client(self):
        """Get Redis client for caching"""
        if not self.redis_client:
            self.redis_client = redis.from_url(
                f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/1"
            )
        return self.redis_client

    def _load_api_configs(self) -> Dict[str, APIEndpointConfig]:
        """Load API endpoint configurations"""
        return {
            "openai_completions": APIEndpointConfig(
                provider=APIProvider.OPENAI,
                endpoint="chat/completions",
                method="POST",
                rate_limit_per_minute=500,
                rate_limit_per_hour=10000,
                cache_strategy=CacheStrategy.SMART,
                cache_ttl_seconds=1800,  # 30 minutes
                cost_per_request=0.002,
                timeout_seconds=30,
                max_retries=3,
                fallback_providers=[APIProvider.ANTHROPIC],
                priority_level=8
            ),
            "anthropic_completions": APIEndpointConfig(
                provider=APIProvider.ANTHROPIC,
                endpoint="messages",
                method="POST",
                rate_limit_per_minute=300,
                rate_limit_per_hour=5000,
                cache_strategy=CacheStrategy.SMART,
                cache_ttl_seconds=1800,
                cost_per_request=0.003,
                timeout_seconds=45,
                max_retries=3,
                priority_level=7
            ),
            "elevenlabs_tts": APIEndpointConfig(
                provider=APIProvider.ELEVENLABS,
                endpoint="text-to-speech",
                method="POST",
                rate_limit_per_minute=100,
                rate_limit_per_hour=1000,
                cache_strategy=CacheStrategy.AGGRESSIVE,
                cache_ttl_seconds=86400,  # 24 hours
                cost_per_request=0.005,
                timeout_seconds=60,
                max_retries=2,
                fallback_providers=[APIProvider.GOOGLE_TTS],
                priority_level=6
            ),
            "google_tts": APIEndpointConfig(
                provider=APIProvider.GOOGLE_TTS,
                endpoint="synthesize",
                method="POST",
                rate_limit_per_minute=200,
                rate_limit_per_hour=10000,
                cache_strategy=CacheStrategy.AGGRESSIVE,
                cache_ttl_seconds=86400,
                cost_per_request=0.001,
                timeout_seconds=30,
                max_retries=3,
                priority_level=5
            ),
            "youtube_api": APIEndpointConfig(
                provider=APIProvider.YOUTUBE,
                endpoint="videos",
                method="POST",
                rate_limit_per_minute=100,
                rate_limit_per_hour=10000,
                cache_strategy=CacheStrategy.SIMPLE,
                cache_ttl_seconds=300,  # 5 minutes
                cost_per_request=0.0,  # Quota-based, not cost
                timeout_seconds=120,
                max_retries=3,
                priority_level=9
            )
        }

    async def make_optimized_request(
        self,
        config_key: str,
        request_data: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make an optimized API request with caching, rate limiting, and fallbacks"""
        
        if config_key not in self.api_configs:
            raise ValueError(f"Unknown API config: {config_key}")
        
        config = self.api_configs[config_key]
        start_time = time.time()
        
        try:
            # 1. Check cache first
            cached_result = await self._check_cache(config, request_data)
            if cached_result:
                self._record_metrics(config, start_time, 200, 0.0, True, 0)
                return cached_result
            
            # 2. Check rate limits
            if not await self._check_rate_limit(config, user_id):
                raise Exception(f"Rate limit exceeded for {config.provider.value}")
            
            # 3. Check circuit breaker
            if self._is_circuit_breaker_open(config.provider):
                if config.fallback_providers:
                    return await self._try_fallback_providers(config, request_data, headers, user_id)
                else:
                    raise Exception(f"Circuit breaker open for {config.provider.value}")
            
            # 4. Make the actual request
            result = await self._make_request(config, request_data, headers)
            
            # 5. Cache the result
            await self._cache_result(config, request_data, result)
            
            # 6. Record success metrics
            response_time = (time.time() - start_time) * 1000
            self._record_metrics(config, start_time, 200, config.cost_per_request, False, 0)
            
            # 7. Track costs
            if config.cost_per_request > 0 and user_id:
                await cost_tracker.track_api_call(
                    user_id=user_id,
                    service=config.provider.value,
                    operation=config.endpoint,
                    cost=config.cost_per_request,
                    metadata={
                        "response_time_ms": response_time,
                        "cache_hit": False
                    }
                )
            
            # 8. Reset circuit breaker on success
            self._reset_circuit_breaker(config.provider)
            
            return result
            
        except Exception as e:
            # Record failure
            response_time = (time.time() - start_time) * 1000
            self._record_metrics(config, start_time, 500, 0, False, 1)
            
            # Update circuit breaker
            self._record_failure(config.provider)
            
            # Try fallback if available
            if config.fallback_providers and not isinstance(e, Exception):
                logger.warning(f"Primary API failed, trying fallbacks: {e}")
                return await self._try_fallback_providers(config, request_data, headers, user_id)
            
            raise e

    async def _check_cache(
        self, 
        config: APIEndpointConfig, 
        request_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Check if request result is cached"""
        if config.cache_strategy == CacheStrategy.NONE:
            return None
        
        try:
            redis_client = await self._get_redis_client()
            cache_key = self._generate_cache_key(config, request_data)
            
            cached_data = await redis_client.get(cache_key)
            if cached_data:
                logger.debug(f"Cache hit for {config.provider.value}:{config.endpoint}")
                return json.loads(cached_data)
            
        except Exception as e:
            logger.warning(f"Cache check failed: {e}")
        
        return None

    async def _cache_result(
        self, 
        config: APIEndpointConfig, 
        request_data: Dict[str, Any], 
        result: Dict[str, Any]
    ):
        """Cache API response result"""
        if config.cache_strategy == CacheStrategy.NONE:
            return
        
        try:
            redis_client = await self._get_redis_client()
            cache_key = self._generate_cache_key(config, request_data)
            
            await redis_client.setex(
                cache_key,
                config.cache_ttl_seconds,
                json.dumps(result)
            )
            
            logger.debug(f"Cached result for {config.provider.value}:{config.endpoint}")
            
        except Exception as e:
            logger.warning(f"Cache store failed: {e}")

    def _generate_cache_key(
        self, 
        config: APIEndpointConfig, 
        request_data: Dict[str, Any]
    ) -> str:
        """Generate cache key for request"""
        # Create deterministic cache key from request data
        key_data = {
            "provider": config.provider.value,
            "endpoint": config.endpoint,
            "data": request_data
        }
        
        # For smart caching, include only relevant fields
        if config.cache_strategy == CacheStrategy.SMART:
            # Remove timestamp-sensitive fields
            filtered_data = {k: v for k, v in request_data.items() 
                           if k not in ["timestamp", "request_id", "user_id"]}
            key_data["data"] = filtered_data
        
        key_string = json.dumps(key_data, sort_keys=True)
        cache_key = f"api_cache:{hashlib.md5(key_string.encode()).hexdigest()}"
        
        return cache_key

    async def _check_rate_limit(
        self, 
        config: APIEndpointConfig, 
        user_id: Optional[str]
    ) -> bool:
        """Check if request is within rate limits"""
        now = datetime.utcnow()
        provider_key = f"{config.provider.value}:{config.endpoint}"
        
        # Check per-minute limit
        minute_key = f"{provider_key}:minute:{now.strftime('%Y%m%d%H%M')}"
        if self.rate_limits[minute_key]["count"] >= config.rate_limit_per_minute:
            return False
        
        # Check per-hour limit
        hour_key = f"{provider_key}:hour:{now.strftime('%Y%m%d%H')}"
        if self.rate_limits[hour_key]["count"] >= config.rate_limit_per_hour:
            return False
        
        # Increment counters
        self.rate_limits[minute_key]["count"] += 1
        self.rate_limits[hour_key]["count"] += 1
        
        return True

    async def _make_request(
        self,
        config: APIEndpointConfig,
        request_data: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make the actual HTTP request"""
        session = await self._get_session(config.provider)
        
        request_headers = headers or {}
        request_headers.update({
            "Content-Type": "application/json",
            "User-Agent": "YTEmpire-API/1.0"
        })
        
        # Build full URL
        base_urls = {
            APIProvider.OPENAI: "https://api.openai.com/v1/",
            APIProvider.ANTHROPIC: "https://api.anthropic.com/v1/",
            APIProvider.ELEVENLABS: "https://api.elevenlabs.io/v1/",
            APIProvider.GOOGLE_TTS: "https://texttospeech.googleapis.com/v1/",
            APIProvider.YOUTUBE: "https://www.googleapis.com/youtube/v3/",
            APIProvider.STRIPE: "https://api.stripe.com/v1/"
        }
        
        url = f"{base_urls[config.provider]}{config.endpoint}"
        
        timeout = aiohttp.ClientTimeout(total=config.timeout_seconds)
        
        async with session.request(
            method=config.method,
            url=url,
            json=request_data if config.method in ["POST", "PUT"] else None,
            params=request_data if config.method == "GET" else None,
            headers=request_headers,
            timeout=timeout
        ) as response:
            if response.status >= 400:
                error_text = await response.text()
                raise Exception(f"API request failed: {response.status} {error_text}")
            
            result = await response.json()
            return result

    async def _get_session(self, provider: APIProvider) -> aiohttp.ClientSession:
        """Get or create HTTP session for provider"""
        if provider not in self.session_pool:
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=20,
                ttl_dns_cache=300,
                use_dns_cache=True
            )
            
            self.session_pool[provider] = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=60)
            )
        
        return self.session_pool[provider]

    def _is_circuit_breaker_open(self, provider: APIProvider) -> bool:
        """Check if circuit breaker is open for provider"""
        breaker = self.circuit_breakers[provider]
        
        # Circuit breaker is open if more than 5 failures in last 5 minutes
        if breaker["failures"] >= 5:
            if breaker["last_failure"]:
                time_since_failure = datetime.utcnow() - breaker["last_failure"]
                if time_since_failure < timedelta(minutes=5):
                    return True
                else:
                    # Reset circuit breaker after 5 minutes
                    self._reset_circuit_breaker(provider)
        
        return False

    def _record_failure(self, provider: APIProvider):
        """Record API failure for circuit breaker"""
        self.circuit_breakers[provider]["failures"] += 1
        self.circuit_breakers[provider]["last_failure"] = datetime.utcnow()

    def _reset_circuit_breaker(self, provider: APIProvider):
        """Reset circuit breaker on successful request"""
        self.circuit_breakers[provider]["failures"] = 0
        self.circuit_breakers[provider]["last_failure"] = None

    async def _try_fallback_providers(
        self,
        primary_config: APIEndpointConfig,
        request_data: Dict[str, Any],
        headers: Optional[Dict[str, str]],
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Try fallback providers in order"""
        for fallback_provider in primary_config.fallback_providers:
            try:
                # Find fallback config
                fallback_config_key = None
                for key, config in self.api_configs.items():
                    if config.provider == fallback_provider:
                        fallback_config_key = key
                        break
                
                if fallback_config_key:
                    logger.info(f"Trying fallback provider: {fallback_provider.value}")
                    return await self.make_optimized_request(
                        fallback_config_key, request_data, headers, user_id
                    )
                    
            except Exception as e:
                logger.warning(f"Fallback provider {fallback_provider.value} failed: {e}")
                continue
        
        raise Exception("All providers (primary and fallbacks) failed")

    def _record_metrics(
        self,
        config: APIEndpointConfig,
        start_time: float,
        status_code: int,
        cost: float,
        cache_hit: bool,
        retry_count: int
    ):
        """Record API call metrics"""
        response_time = (time.time() - start_time) * 1000
        
        metric = APICallMetrics(
            provider=config.provider,
            endpoint=config.endpoint,
            timestamp=datetime.utcnow(),
            response_time_ms=response_time,
            status_code=status_code,
            cost=cost,
            cache_hit=cache_hit,
            retry_count=retry_count
        )
        
        self.metrics.append(metric)
        
        # Keep only last 1000 metrics
        if len(self.metrics) > 1000:
            self.metrics = self.metrics[-1000:]

    async def get_optimization_stats(self) -> Dict[str, Any]:
        """Get API optimization statistics"""
        if not self.metrics:
            return {"error": "No metrics available"}
        
        # Calculate stats
        recent_metrics = [m for m in self.metrics 
                         if (datetime.utcnow() - m.timestamp).total_seconds() < 3600]
        
        stats = {
            "total_requests_hour": len(recent_metrics),
            "cache_hit_rate": sum(1 for m in recent_metrics if m.cache_hit) / len(recent_metrics) * 100,
            "average_response_time": sum(m.response_time_ms for m in recent_metrics) / len(recent_metrics),
            "total_cost_hour": sum(m.cost for m in recent_metrics),
            "error_rate": sum(1 for m in recent_metrics if m.status_code >= 400) / len(recent_metrics) * 100
        }
        
        # Provider-specific stats
        provider_stats = {}
        for provider in APIProvider:
            provider_metrics = [m for m in recent_metrics if m.provider == provider]
            if provider_metrics:
                provider_stats[provider.value] = {
                    "requests": len(provider_metrics),
                    "avg_response_time": sum(m.response_time_ms for m in provider_metrics) / len(provider_metrics),
                    "cache_hit_rate": sum(1 for m in provider_metrics if m.cache_hit) / len(provider_metrics) * 100,
                    "error_rate": sum(1 for m in provider_metrics if m.status_code >= 400) / len(provider_metrics) * 100
                }
        
        return {
            "overall_stats": stats,
            "provider_stats": provider_stats,
            "circuit_breakers": {
                provider.value: {
                    "failures": self.circuit_breakers[provider]["failures"],
                    "is_open": self._is_circuit_breaker_open(provider)
                }
                for provider in APIProvider
            }
        }

    async def clear_cache(self, provider: Optional[APIProvider] = None):
        """Clear API cache for specific provider or all"""
        try:
            redis_client = await self._get_redis_client()
            
            if provider:
                # Clear cache for specific provider
                pattern = f"api_cache:*{provider.value}*"
            else:
                # Clear all API cache
                pattern = "api_cache:*"
            
            keys = await redis_client.keys(pattern)
            if keys:
                await redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} cache entries for {provider.value if provider else 'all providers'}")
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")

    async def update_rate_limits(
        self, 
        provider: APIProvider, 
        per_minute: int, 
        per_hour: int
    ):
        """Update rate limits for a provider"""
        for config in self.api_configs.values():
            if config.provider == provider:
                config.rate_limit_per_minute = per_minute
                config.rate_limit_per_hour = per_hour
        
        logger.info(f"Updated rate limits for {provider.value}: {per_minute}/min, {per_hour}/hour")

    async def close_sessions(self):
        """Clean up HTTP sessions"""
        for session in self.session_pool.values():
            await session.close()
        
        if self.redis_client:
            await self.redis_client.close()

# Global API optimizer instance
api_optimizer = APIOptimizer()