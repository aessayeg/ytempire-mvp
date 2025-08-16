"""
Rate limiting service for API endpoints
"""
import redis.asyncio as redis
from typing import Optional
from datetime import datetime, timedelta
import hashlib
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Redis-based rate limiter for API endpoints
    """

    def __init__(self):
        self.redis_client = None
        self._connect()

    def _connect(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(
                settings.REDIS_URL, encoding="utf-8", decode_responses=True
            )
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")

    async def check_rate_limit(
        self, key: str, max_attempts: int = 10, window: int = 60
    ) -> bool:
        """
        Check if rate limit is exceeded

        Args:
            key: Unique identifier for the rate limit (e.g., "login:user@example.com")
            max_attempts: Maximum number of attempts allowed
            window: Time window in seconds

        Returns:
            True if within rate limit, False if exceeded
        """
        if not self.redis_client:
            # If Redis is not available, allow the request
            return True

        try:
            # Create a unique key with prefix
            rate_key = f"rate_limit:{key}"

            # Get current count
            current = await self.redis_client.get(rate_key)

            if current is None:
                # First attempt, set the key with expiration
                await self.redis_client.setex(rate_key, window, 1)
                return True

            current_count = int(current)

            if current_count >= max_attempts:
                # Rate limit exceeded
                ttl = await self.redis_client.ttl(rate_key)
                logger.warning(f"Rate limit exceeded for {key}. TTL: {ttl} seconds")
                return False

            # Increment the counter
            await self.redis_client.incr(rate_key)
            return True

        except Exception as e:
            logger.error(f"Rate limiter error: {str(e)}")
            # On error, allow the request
            return True

    async def log_failed_attempt(self, key: str, window: int = 3600) -> None:
        """
        Log a failed attempt (e.g., failed login)

        Args:
            key: Unique identifier
            window: Time window for tracking failed attempts
        """
        if not self.redis_client:
            return

        try:
            failed_key = f"failed_attempts:{key}"

            # Increment failed attempts counter
            current = await self.redis_client.get(failed_key)

            if current is None:
                await self.redis_client.setex(failed_key, window, 1)
            else:
                await self.redis_client.incr(failed_key)

                # Check if we should trigger additional security measures
                failed_count = int(await self.redis_client.get(failed_key))

                if failed_count >= 10:
                    # Log security event
                    logger.warning(
                        f"Multiple failed attempts detected for {key}: {failed_count} attempts"
                    )

                    # Could trigger additional security measures here
                    # e.g., send alert email, temporary IP block, etc.

        except Exception as e:
            logger.error(f"Failed to log failed attempt: {str(e)}")

    async def reset_limit(self, key: str) -> bool:
        """
        Reset rate limit for a specific key

        Args:
            key: Unique identifier to reset

        Returns:
            True if reset successful
        """
        if not self.redis_client:
            return False

        try:
            rate_key = f"rate_limit:{key}"
            failed_key = f"failed_attempts:{key}"

            # Delete both rate limit and failed attempts
            await self.redis_client.delete(rate_key)
            await self.redis_client.delete(failed_key)

            logger.info(f"Rate limit reset for {key}")
            return True

        except Exception as e:
            logger.error(f"Failed to reset rate limit: {str(e)}")
            return False

    async def get_remaining_attempts(
        self, key: str, max_attempts: int = 10
    ) -> Optional[int]:
        """
        Get remaining attempts for a key

        Args:
            key: Unique identifier
            max_attempts: Maximum attempts allowed

        Returns:
            Number of remaining attempts, or None if no limit set
        """
        if not self.redis_client:
            return max_attempts

        try:
            rate_key = f"rate_limit:{key}"
            current = await self.redis_client.get(rate_key)

            if current is None:
                return max_attempts

            used_attempts = int(current)
            return max(0, max_attempts - used_attempts)

        except Exception as e:
            logger.error(f"Failed to get remaining attempts: {str(e)}")
            return max_attempts

    async def get_ttl(self, key: str) -> Optional[int]:
        """
        Get time until rate limit resets

        Args:
            key: Unique identifier

        Returns:
            Seconds until reset, or None if no limit set
        """
        if not self.redis_client:
            return None

        try:
            rate_key = f"rate_limit:{key}"
            ttl = await self.redis_client.ttl(rate_key)

            if ttl < 0:
                return None

            return ttl

        except Exception as e:
            logger.error(f"Failed to get TTL: {str(e)}")
            return None

    async def apply_sliding_window_limit(
        self, key: str, max_requests: int = 100, window_minutes: int = 60
    ) -> bool:
        """
        Apply sliding window rate limiting

        Args:
            key: Unique identifier
            max_requests: Maximum requests in the window
            window_minutes: Time window in minutes

        Returns:
            True if within limit, False if exceeded
        """
        if not self.redis_client:
            return True

        try:
            now = datetime.utcnow()
            window_start = now - timedelta(minutes=window_minutes)

            # Use Redis sorted set for sliding window
            zset_key = f"sliding_window:{key}"

            # Remove old entries outside the window
            await self.redis_client.zremrangebyscore(
                zset_key, 0, window_start.timestamp()
            )

            # Count requests in current window
            count = await self.redis_client.zcard(zset_key)

            if count >= max_requests:
                return False

            # Add current request
            await self.redis_client.zadd(
                zset_key, {str(now.timestamp()): now.timestamp()}
            )

            # Set expiration on the sorted set
            await self.redis_client.expire(zset_key, window_minutes * 60)

            return True

        except Exception as e:
            logger.error(f"Sliding window rate limit error: {str(e)}")
            return True

    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()


# Global rate limiter instance
rate_limiter = RateLimiter()
