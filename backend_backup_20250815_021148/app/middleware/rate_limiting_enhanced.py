"""
Enhanced API Rate Limiting for YTEmpire
P2 Task: [BACKEND] API Rate Limiting Enhancement
Advanced rate limiting with tiered limits and intelligent throttling
"""

import time
import asyncio
from typing import Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import redis
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
import hashlib
import json
import logging

from ..core.config import settings

logger = logging.getLogger(__name__)

class RateLimitTier(str):
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"

class EnhancedRateLimiter:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=True
        )
        
        # Tiered rate limits (requests per hour)
        self.tier_limits = {
            RateLimitTier.FREE: {
                "requests_per_hour": 100,
                "requests_per_minute": 10,
                "burst_size": 20,
                "video_generation_per_day": 5
            },
            RateLimitTier.BASIC: {
                "requests_per_hour": 1000,
                "requests_per_minute": 50,
                "burst_size": 100,
                "video_generation_per_day": 50
            },
            RateLimitTier.PRO: {
                "requests_per_hour": 5000,
                "requests_per_minute": 200,
                "burst_size": 500,
                "video_generation_per_day": 200
            },
            RateLimitTier.ENTERPRISE: {
                "requests_per_hour": 50000,
                "requests_per_minute": 1000,
                "burst_size": 2000,
                "video_generation_per_day": 1000
            }
        }
        
        # Endpoint-specific limits
        self.endpoint_limits = {
            "/api/v1/videos/generate": {"multiplier": 10, "cooldown": 60},
            "/api/v1/analytics/export": {"multiplier": 5, "cooldown": 30},
            "/api/v1/channels/bulk": {"multiplier": 3, "cooldown": 10}
        }
        
    def get_client_id(self, request: Request) -> str:
        """Get unique client identifier"""
        
        # Try to get authenticated user ID
        if hasattr(request.state, "user_id"):
            return f"user:{request.state.user_id}"
        
        # Fall back to IP address
        client_ip = request.client.host
        
        # Handle proxy headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        return f"ip:{client_ip}"
    
    def get_user_tier(self, client_id: str) -> RateLimitTier:
        """Get user's subscription tier"""
        
        # Check cache
        tier = self.redis_client.get(f"user_tier:{client_id}")
        if tier:
            return RateLimitTier(tier)
        
        # In production, query from database
        # For now, return default
        if "user:" in client_id:
            # Authenticated users get basic tier
            return RateLimitTier.BASIC
        
        return RateLimitTier.FREE
    
    async def check_rate_limit(
        self,
        request: Request,
        endpoint: Optional[str] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is within rate limits"""
        
        client_id = self.get_client_id(request)
        tier = self.get_user_tier(client_id)
        limits = self.tier_limits[tier]
        
        # Get endpoint path
        endpoint = endpoint or request.url.path
        
        # Apply endpoint-specific multipliers
        if endpoint in self.endpoint_limits:
            endpoint_config = self.endpoint_limits[endpoint]
            limits = self._apply_endpoint_limits(limits, endpoint_config)
        
        # Check multiple time windows
        checks = [
            self._check_window(client_id, "minute", 60, limits["requests_per_minute"]),
            self._check_window(client_id, "hour", 3600, limits["requests_per_hour"]),
            self._check_burst(client_id, limits["burst_size"])
        ]
        
        # Special check for video generation
        if "videos/generate" in endpoint:
            checks.append(
                self._check_daily_video_limit(client_id, limits["video_generation_per_day"])
            )
        
        # Run all checks
        results = await asyncio.gather(*checks)
        
        # Aggregate results
        allowed = all(r[0] for r in results)
        
        # Calculate retry after
        retry_after = 0
        if not allowed:
            retry_after = max(r[1].get("retry_after", 0) for r in results)
        
        # Prepare response headers
        headers = {
            "X-RateLimit-Limit": str(limits["requests_per_hour"]),
            "X-RateLimit-Remaining": str(min(r[1].get("remaining", 0) for r in results)),
            "X-RateLimit-Reset": str(int(time.time()) + 3600),
            "X-RateLimit-Tier": tier
        }
        
        if retry_after > 0:
            headers["Retry-After"] = str(retry_after)
        
        return allowed, headers
    
    async def _check_window(
        self,
        client_id: str,
        window: str,
        duration: int,
        limit: int
    ) -> Tuple[bool, Dict]:
        """Check rate limit for a time window"""
        
        key = f"rate_limit:{client_id}:{window}"
        current_time = int(time.time())
        window_start = current_time - duration
        
        # Remove old entries
        self.redis_client.zremrangebyscore(key, 0, window_start)
        
        # Count requests in window
        request_count = self.redis_client.zcard(key)
        
        if request_count >= limit:
            # Calculate retry after
            oldest_request = self.redis_client.zrange(key, 0, 0, withscores=True)
            if oldest_request:
                retry_after = int(oldest_request[0][1]) + duration - current_time
            else:
                retry_after = duration
            
            return False, {"remaining": 0, "retry_after": retry_after}
        
        # Add current request
        self.redis_client.zadd(key, {str(current_time): current_time})
        self.redis_client.expire(key, duration)
        
        return True, {"remaining": limit - request_count - 1, "retry_after": 0}
    
    async def _check_burst(self, client_id: str, burst_size: int) -> Tuple[bool, Dict]:
        """Check burst limit using token bucket algorithm"""
        
        key = f"burst:{client_id}"
        current_time = time.time()
        
        # Get current tokens
        bucket_data = self.redis_client.get(key)
        
        if bucket_data:
            bucket = json.loads(bucket_data)
            tokens = bucket["tokens"]
            last_refill = bucket["last_refill"]
        else:
            tokens = burst_size
            last_refill = current_time
        
        # Refill tokens
        time_passed = current_time - last_refill
        tokens_to_add = time_passed * (burst_size / 3600)  # Refill rate: burst_size per hour
        tokens = min(burst_size, tokens + tokens_to_add)
        
        if tokens < 1:
            retry_after = int((1 - tokens) * (3600 / burst_size))
            return False, {"remaining": 0, "retry_after": retry_after}
        
        # Consume token
        tokens -= 1
        
        # Save bucket state
        bucket_data = json.dumps({
            "tokens": tokens,
            "last_refill": current_time
        })
        self.redis_client.setex(key, 3600, bucket_data)
        
        return True, {"remaining": int(tokens), "retry_after": 0}
    
    async def _check_daily_video_limit(
        self,
        client_id: str,
        daily_limit: int
    ) -> Tuple[bool, Dict]:
        """Check daily video generation limit"""
        
        today = datetime.utcnow().strftime("%Y%m%d")
        key = f"video_limit:{client_id}:{today}"
        
        count = self.redis_client.get(key)
        count = int(count) if count else 0
        
        if count >= daily_limit:
            # Calculate seconds until midnight UTC
            now = datetime.utcnow()
            midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            retry_after = int((midnight - now).total_seconds())
            
            return False, {"remaining": 0, "retry_after": retry_after}
        
        # Increment counter
        self.redis_client.incr(key)
        self.redis_client.expire(key, 86400)  # Expire after 24 hours
        
        return True, {"remaining": daily_limit - count - 1, "retry_after": 0}
    
    def _apply_endpoint_limits(self, base_limits: Dict, endpoint_config: Dict) -> Dict:
        """Apply endpoint-specific limit modifications"""
        
        modified_limits = base_limits.copy()
        multiplier = endpoint_config.get("multiplier", 1)
        
        # Reduce limits for expensive endpoints
        modified_limits["requests_per_minute"] = max(
            1,
            modified_limits["requests_per_minute"] // multiplier
        )
        modified_limits["requests_per_hour"] = max(
            1,
            modified_limits["requests_per_hour"] // multiplier
        )
        
        return modified_limits
    
    async def handle_rate_limit_exceeded(
        self,
        request: Request,
        headers: Dict[str, str]
    ) -> JSONResponse:
        """Handle rate limit exceeded response"""
        
        client_id = self.get_client_id(request)
        tier = self.get_user_tier(client_id)
        
        # Log rate limit violation
        logger.warning(f"Rate limit exceeded for {client_id} (tier: {tier})")
        
        # Increment violation counter
        violation_key = f"violations:{client_id}"
        violations = self.redis_client.incr(violation_key)
        self.redis_client.expire(violation_key, 3600)
        
        # Check for abuse
        if violations > 10:
            # Temporary ban for repeated violations
            ban_key = f"banned:{client_id}"
            self.redis_client.setex(ban_key, 3600, "1")  # 1 hour ban
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests",
                    "message": "Temporarily banned due to repeated violations",
                    "retry_after": 3600
                },
                headers=headers
            )
        
        # Prepare response
        retry_after = int(headers.get("Retry-After", 60))
        
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate Limit Exceeded",
                "message": f"You have exceeded the rate limit for tier {tier}",
                "tier": tier,
                "retry_after": retry_after,
                "upgrade_url": "/pricing" if tier != RateLimitTier.ENTERPRISE else None
            },
            headers=headers
        )
    
    def reset_limits(self, client_id: str):
        """Reset rate limits for a client (admin function)"""
        
        patterns = [
            f"rate_limit:{client_id}:*",
            f"burst:{client_id}",
            f"video_limit:{client_id}:*",
            f"violations:{client_id}"
        ]
        
        for pattern in patterns:
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
    
    def get_usage_stats(self, client_id: str) -> Dict:
        """Get usage statistics for a client"""
        
        tier = self.get_user_tier(client_id)
        limits = self.tier_limits[tier]
        
        # Get current usage
        minute_key = f"rate_limit:{client_id}:minute"
        hour_key = f"rate_limit:{client_id}:hour"
        
        minute_count = self.redis_client.zcard(minute_key)
        hour_count = self.redis_client.zcard(hour_key)
        
        # Get video usage
        today = datetime.utcnow().strftime("%Y%m%d")
        video_key = f"video_limit:{client_id}:{today}"
        video_count = int(self.redis_client.get(video_key) or 0)
        
        return {
            "tier": tier,
            "usage": {
                "requests_per_minute": {
                    "used": minute_count,
                    "limit": limits["requests_per_minute"],
                    "percentage": (minute_count / limits["requests_per_minute"]) * 100
                },
                "requests_per_hour": {
                    "used": hour_count,
                    "limit": limits["requests_per_hour"],
                    "percentage": (hour_count / limits["requests_per_hour"]) * 100
                },
                "videos_today": {
                    "used": video_count,
                    "limit": limits["video_generation_per_day"],
                    "percentage": (video_count / limits["video_generation_per_day"]) * 100
                }
            }
        }

# Middleware function
async def rate_limit_middleware(request: Request, call_next):
    """FastAPI middleware for rate limiting"""
    
    limiter = EnhancedRateLimiter()
    
    # Check if banned
    client_id = limiter.get_client_id(request)
    if limiter.redis_client.get(f"banned:{client_id}"):
        return JSONResponse(
            status_code=429,
            content={"error": "Temporarily banned due to rate limit violations"}
        )
    
    # Check rate limits
    allowed, headers = await limiter.check_rate_limit(request)
    
    if not allowed:
        return await limiter.handle_rate_limit_exceeded(request, headers)
    
    # Process request
    response = await call_next(request)
    
    # Add rate limit headers to response
    for key, value in headers.items():
        response.headers[key] = value
    
    return response