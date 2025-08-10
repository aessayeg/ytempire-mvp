"""
Rate limiting middleware for API endpoints
"""
from typing import Callable, Optional
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import redis.asyncio as redis
import time
import json
import hashlib
import logging
from datetime import datetime, timedelta

from app.core.config import settings

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using Redis
    
    Implements multiple rate limiting strategies:
    - Per-user rate limiting
    - Per-IP rate limiting
    - Per-endpoint rate limiting
    - Subscription tier-based limits
    """
    
    def __init__(self, app, redis_url: str = None):
        super().__init__(app)
        self.redis_url = redis_url or settings.REDIS_URL
        self.redis_client = None
        self._connect()
    
    def _connect(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with rate limiting
        """
        # Skip rate limiting for certain paths
        skip_paths = ["/docs", "/redoc", "/openapi.json", "/health", "/metrics"]
        if any(request.url.path.startswith(path) for path in skip_paths):
            return await call_next(request)
        
        # Get client identifier
        client_id = await self._get_client_id(request)
        
        # Get rate limit configuration
        rate_config = await self._get_rate_config(request)
        
        # Check rate limit
        is_allowed, retry_after = await self._check_rate_limit(
            client_id,
            rate_config["max_requests"],
            rate_config["window_seconds"]
        )
        
        if not is_allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after": retry_after
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(rate_config["max_requests"]),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + retry_after)
                }
            )
        
        # Get remaining requests
        remaining = await self._get_remaining_requests(
            client_id,
            rate_config["max_requests"],
            rate_config["window_seconds"]
        )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(rate_config["max_requests"])
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(
            int(time.time()) + rate_config["window_seconds"]
        )
        
        return response
    
    async def _get_client_id(self, request: Request) -> str:
        """
        Get unique client identifier
        
        Priority:
        1. User ID (if authenticated)
        2. API key
        3. IP address
        """
        # Try to get user ID from JWT token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            try:
                from jose import jwt
                token = auth_header.split(" ")[1]
                payload = jwt.decode(
                    token,
                    settings.SECRET_KEY,
                    algorithms=[settings.ALGORITHM]
                )
                user_id = payload.get("sub")
                if user_id:
                    return f"user:{user_id}"
            except:
                pass
        
        # Try to get API key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key:{api_key}"
        
        # Fall back to IP address
        client_ip = request.client.host
        return f"ip:{client_ip}"
    
    async def _get_rate_config(self, request: Request) -> dict:
        """
        Get rate limit configuration based on endpoint and user tier
        """
        path = request.url.path
        method = request.method
        
        # Default limits
        default_config = {
            "max_requests": 100,
            "window_seconds": 60
        }
        
        # Endpoint-specific limits
        endpoint_limits = {
            # Auth endpoints - stricter limits
            "POST:/api/v1/auth/register": {"max_requests": 3, "window_seconds": 3600},
            "POST:/api/v1/auth/login": {"max_requests": 5, "window_seconds": 900},
            "POST:/api/v1/auth/password-reset": {"max_requests": 3, "window_seconds": 3600},
            
            # Video generation - resource intensive
            "POST:/api/v1/videos/generate": {"max_requests": 10, "window_seconds": 3600},
            "POST:/api/v1/videos/bulk-generate": {"max_requests": 2, "window_seconds": 3600},
            
            # Channel operations
            "POST:/api/v1/channels": {"max_requests": 5, "window_seconds": 3600},
            "DELETE:/api/v1/channels": {"max_requests": 5, "window_seconds": 3600},
            
            # Read operations - more lenient
            "GET:/api/v1/videos": {"max_requests": 100, "window_seconds": 60},
            "GET:/api/v1/channels": {"max_requests": 100, "window_seconds": 60},
            "GET:/api/v1/users/profile": {"max_requests": 60, "window_seconds": 60}
        }
        
        # Check for exact match
        endpoint_key = f"{method}:{path}"
        for pattern, config in endpoint_limits.items():
            if endpoint_key.startswith(pattern.replace("/api/v1/", "/api/v1/")):
                return config
        
        # Check user tier for enhanced limits
        client_id = await self._get_client_id(request)
        if client_id.startswith("user:"):
            user_tier = await self._get_user_tier(client_id.split(":")[1])
            
            tier_multipliers = {
                "free": 1.0,
                "starter": 2.0,
                "pro": 5.0,
                "enterprise": 10.0
            }
            
            multiplier = tier_multipliers.get(user_tier, 1.0)
            default_config["max_requests"] = int(default_config["max_requests"] * multiplier)
        
        return default_config
    
    async def _get_user_tier(self, user_id: str) -> str:
        """
        Get user subscription tier from cache or database
        """
        if not self.redis_client:
            return "free"
        
        try:
            # Check cache
            cache_key = f"user_tier:{user_id}"
            cached_tier = await self.redis_client.get(cache_key)
            
            if cached_tier:
                return cached_tier
            
            # Get from database
            from app.db.session import get_db
            from app.models.user import User
            from sqlalchemy import select
            
            async with get_db() as db:
                result = await db.execute(
                    select(User.subscription_tier).filter(User.id == user_id)
                )
                user = result.scalar_one_or_none()
                
                if user:
                    tier = user.subscription_tier
                    # Cache for 5 minutes
                    await self.redis_client.setex(cache_key, 300, tier)
                    return tier
            
            return "free"
            
        except Exception as e:
            logger.error(f"Error getting user tier: {str(e)}")
            return "free"
    
    async def _check_rate_limit(
        self,
        client_id: str,
        max_requests: int,
        window_seconds: int
    ) -> tuple[bool, int]:
        """
        Check if request is within rate limit
        
        Returns:
            (is_allowed, retry_after_seconds)
        """
        if not self.redis_client:
            # If Redis is not available, allow the request
            return (True, 0)
        
        try:
            key = f"rate_limit:{client_id}"
            current_time = int(time.time())
            window_start = current_time - window_seconds
            
            # Use Redis sorted set for sliding window
            pipe = self.redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count requests in current window
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(current_time): current_time})
            
            # Set expiration
            pipe.expire(key, window_seconds)
            
            results = await pipe.execute()
            request_count = results[1]
            
            if request_count >= max_requests:
                # Get oldest request in window to calculate retry time
                oldest = await self.redis_client.zrange(key, 0, 0, withscores=True)
                if oldest:
                    oldest_time = int(oldest[0][1])
                    retry_after = window_seconds - (current_time - oldest_time)
                    return (False, retry_after)
                return (False, window_seconds)
            
            return (True, 0)
            
        except Exception as e:
            logger.error(f"Rate limit check error: {str(e)}")
            # On error, allow the request
            return (True, 0)
    
    async def _get_remaining_requests(
        self,
        client_id: str,
        max_requests: int,
        window_seconds: int
    ) -> int:
        """
        Get number of remaining requests in current window
        """
        if not self.redis_client:
            return max_requests
        
        try:
            key = f"rate_limit:{client_id}"
            current_time = int(time.time())
            window_start = current_time - window_seconds
            
            # Count requests in current window
            count = await self.redis_client.zcount(key, window_start, current_time)
            
            return max(0, max_requests - count)
            
        except Exception as e:
            logger.error(f"Error getting remaining requests: {str(e)}")
            return max_requests

class APIKeyRateLimiter:
    """
    Specific rate limiter for API key authentication
    """
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.REDIS_URL
        self.redis_client = None
        self._connect()
    
    def _connect(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
    
    async def check_api_key_limit(
        self,
        api_key: str,
        endpoint: str
    ) -> tuple[bool, dict]:
        """
        Check API key rate limit
        
        Returns:
            (is_allowed, limit_info)
        """
        if not self.redis_client:
            return (True, {})
        
        try:
            # Get API key configuration
            from app.db.session import get_db
            from app.models.api_key import APIKey
            from sqlalchemy import select
            
            async with get_db() as db:
                result = await db.execute(
                    select(APIKey).filter(
                        APIKey.key == api_key,
                        APIKey.is_active == True
                    )
                )
                api_key_obj = result.scalar_one_or_none()
                
                if not api_key_obj:
                    return (False, {"error": "Invalid API key"})
                
                # Check if key has expired
                if api_key_obj.expires_at and api_key_obj.expires_at < datetime.utcnow():
                    return (False, {"error": "API key expired"})
                
                # Get rate limits for this key
                max_requests = api_key_obj.rate_limit or 1000
                window_seconds = 3600  # 1 hour window
                
                # Check rate limit
                key = f"api_key_limit:{api_key}:{endpoint}"
                current = await self.redis_client.get(key)
                
                if current is None:
                    # First request
                    await self.redis_client.setex(key, window_seconds, 1)
                    remaining = max_requests - 1
                else:
                    current_count = int(current)
                    if current_count >= max_requests:
                        ttl = await self.redis_client.ttl(key)
                        return (False, {
                            "error": "API key rate limit exceeded",
                            "retry_after": ttl,
                            "limit": max_requests,
                            "remaining": 0
                        })
                    
                    # Increment counter
                    await self.redis_client.incr(key)
                    remaining = max_requests - current_count - 1
                
                # Update last used timestamp
                api_key_obj.last_used_at = datetime.utcnow()
                api_key_obj.request_count += 1
                await db.commit()
                
                return (True, {
                    "limit": max_requests,
                    "remaining": remaining,
                    "reset": int(time.time()) + window_seconds
                })
                
        except Exception as e:
            logger.error(f"API key rate limit check error: {str(e)}")
            return (True, {})

# Dependency for API key authentication
async def require_api_key(
    request: Request,
    api_key: str = None
) -> dict:
    """
    Validate API key and check rate limits
    """
    # Get API key from header or query parameter
    if not api_key:
        api_key = request.headers.get("X-API-Key")
    
    if not api_key:
        api_key = request.query_params.get("api_key")
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    # Check rate limit
    rate_limiter = APIKeyRateLimiter()
    is_allowed, limit_info = await rate_limiter.check_api_key_limit(
        api_key,
        request.url.path
    )
    
    if not is_allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=limit_info.get("error", "Rate limit exceeded"),
            headers={
                "Retry-After": str(limit_info.get("retry_after", 60)),
                "X-RateLimit-Limit": str(limit_info.get("limit", 0)),
                "X-RateLimit-Remaining": str(limit_info.get("remaining", 0))
            }
        )
    
    return limit_info