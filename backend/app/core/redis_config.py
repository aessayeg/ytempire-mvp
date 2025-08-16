"""
Redis Configuration for YTEmpire
Centralized Redis client management and connection pooling
"""

import logging
from typing import Optional, Dict, Any
import redis
from redis import asyncio as aioredis
from redis.sentinel import Sentinel
from redis.exceptions import RedisError
from app.core.config import settings

logger = logging.getLogger(__name__)


class RedisConfig:
    """
    Redis configuration and connection management
    """

    # Connection pool settings
    POOL_MAX_CONNECTIONS = 100
    POOL_MIN_IDLE = 10
    SOCKET_KEEPALIVE = True
    SOCKET_KEEPALIVE_OPTIONS = {
        1: 1,  # TCP_KEEPIDLE
        2: 60,  # TCP_KEEPINTVL
        3: 20,  # TCP_KEEPCNT
    }

    # Retry settings
    MAX_RETRIES = 3
    RETRY_ON_TIMEOUT = True
    RETRY_ON_ERROR = [ConnectionError, TimeoutError]

    # Timeout settings
    SOCKET_CONNECT_TIMEOUT = 5
    SOCKET_TIMEOUT = 5

    # Redis databases
    DATABASES = {
        "default": 0,  # General purpose
        "cache": 1,  # Caching
        "sessions": 2,  # User sessions
        "websocket": 3,  # WebSocket connections
        "celery": 4,  # Celery broker
        "analytics": 5,  # Analytics data
        "rate_limit": 6,  # Rate limiting
        "pubsub": 7,  # Pub/Sub channels
    }

    @classmethod
    def get_connection_pool(cls, db: int = 0) -> redis.ConnectionPool:
        """
        Get Redis connection pool

        Args:
            db: Database number

        Returns:
            Connection pool instance
        """
        return redis.ConnectionPool(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=db,
            password=settings.REDIS_PASSWORD,
            max_connections=cls.POOL_MAX_CONNECTIONS,
            socket_keepalive=cls.SOCKET_KEEPALIVE,
            socket_keepalive_options=cls.SOCKET_KEEPALIVE_OPTIONS,
            socket_connect_timeout=cls.SOCKET_CONNECT_TIMEOUT,
            socket_timeout=cls.SOCKET_TIMEOUT,
            retry_on_timeout=cls.RETRY_ON_TIMEOUT,
            retry_on_error=cls.RETRY_ON_ERROR,
            max_retries=cls.MAX_RETRIES,
            decode_responses=True,
        )

    @classmethod
    def get_async_connection_pool(cls, db: int = 0) -> aioredis.ConnectionPool:
        """
        Get async Redis connection pool

        Args:
            db: Database number

        Returns:
            Async connection pool instance
        """
        return aioredis.ConnectionPool(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=db,
            password=settings.REDIS_PASSWORD,
            max_connections=cls.POOL_MAX_CONNECTIONS,
            socket_keepalive=cls.SOCKET_KEEPALIVE,
            socket_keepalive_options=cls.SOCKET_KEEPALIVE_OPTIONS,
            socket_connect_timeout=cls.SOCKET_CONNECT_TIMEOUT,
            socket_timeout=cls.SOCKET_TIMEOUT,
            retry_on_timeout=cls.RETRY_ON_TIMEOUT,
            retry_on_error=cls.RETRY_ON_ERROR,
            max_retries=cls.MAX_RETRIES,
            decode_responses=True,
        )


class RedisClient:
    """
    Singleton Redis client manager
    """

    _instance = None
    _clients: Dict[str, redis.Redis] = {}
    _async_clients: Dict[str, aioredis.Redis] = {}
    _pools: Dict[str, redis.ConnectionPool] = {}
    _async_pools: Dict[str, aioredis.ConnectionPool] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisClient, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._setup_clients()

    def _setup_clients(self):
        """Setup Redis clients for each database"""
        for name, db in RedisConfig.DATABASES.items():
            # Create connection pools
            self._pools[name] = RedisConfig.get_connection_pool(db)
            self._async_pools[name] = RedisConfig.get_async_connection_pool(db)

            # Create clients
            self._clients[name] = redis.Redis(connection_pool=self._pools[name])
            self._async_clients[name] = aioredis.Redis(
                connection_pool=self._async_pools[name]
            )

            logger.info(f"Redis client '{name}' initialized on database {db}")

    def get_client(self, name: str = "default") -> redis.Redis:
        """
        Get sync Redis client

        Args:
            name: Client name (default, cache, sessions, etc.)

        Returns:
            Redis client instance
        """
        if name not in self._clients:
            raise ValueError(f"Unknown Redis client: {name}")
        return self._clients[name]

    def get_async_client(self, name: str = "default") -> aioredis.Redis:
        """
        Get async Redis client

        Args:
            name: Client name (default, cache, sessions, etc.)

        Returns:
            Async Redis client instance
        """
        if name not in self._async_clients:
            raise ValueError(f"Unknown Redis client: {name}")
        return self._async_clients[name]

    async def health_check(self) -> Dict[str, bool]:
        """
        Check health of all Redis connections

        Returns:
            Dictionary of client health status
        """
        health = {}

        for name, client in self._async_clients.items():
            try:
                await client.ping()
                health[name] = True
            except Exception as e:
                logger.error(f"Redis client '{name}' health check failed: {str(e)}")
                health[name] = False

        return health

    async def close_all(self):
        """Close all Redis connections"""
        for client in self._async_clients.values():
            await client.close()

        for client in self._clients.values():
            client.close()

        logger.info("All Redis connections closed")


class RedisSentinelClient:
    """
    Redis Sentinel client for high availability
    """

    def __init__(self, sentinels: list, service_name: str):
        """
        Initialize Sentinel client

        Args:
            sentinels: List of (host, port) tuples for sentinels
            service_name: Name of the Redis service
        """
        self.sentinel = Sentinel(sentinels)
        self.service_name = service_name
        self._master = None
        self._slaves = []

    def get_master(self) -> redis.Redis:
        """Get Redis master instance"""
        if not self._master:
            self._master = self.sentinel.master_for(
                self.service_name,
                socket_timeout=RedisConfig.SOCKET_TIMEOUT,
                socket_connect_timeout=RedisConfig.SOCKET_CONNECT_TIMEOUT,
                retry_on_timeout=RedisConfig.RETRY_ON_TIMEOUT,
                max_retries=RedisConfig.MAX_RETRIES,
            )
        return self._master

    def get_slave(self) -> redis.Redis:
        """Get Redis slave instance for read operations"""
        slave = self.sentinel.slave_for(
            self.service_name,
            socket_timeout=RedisConfig.SOCKET_TIMEOUT,
            socket_connect_timeout=RedisConfig.SOCKET_CONNECT_TIMEOUT,
            retry_on_timeout=RedisConfig.RETRY_ON_TIMEOUT,
            max_retries=RedisConfig.MAX_RETRIES,
        )
        self._slaves.append(slave)
        return slave


class RedisLock:
    """
    Distributed lock implementation using Redis
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        key: str,
        timeout: int = 10,
        blocking_timeout: int = 5,
    ):
        """
        Initialize Redis lock

        Args:
            redis_client: Redis client instance
            key: Lock key
            timeout: Lock timeout in seconds
            blocking_timeout: Blocking timeout for acquire
        """
        self.redis_client = redis_client
        self.key = f"lock:{key}"
        self.timeout = timeout
        self.blocking_timeout = blocking_timeout
        self.lock = None

    def __enter__(self):
        """Context manager entry"""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()

    def acquire(self) -> bool:
        """Acquire lock"""
        self.lock = self.redis_client.lock(
            self.key, timeout=self.timeout, blocking_timeout=self.blocking_timeout
        )
        return self.lock.acquire()

    def release(self):
        """Release lock"""
        if self.lock:
            try:
                self.lock.release()
            except Exception as e:
                logger.warning(f"Failed to release lock {self.key}: {str(e)}")

    def extend(self, additional_time: int):
        """Extend lock timeout"""
        if self.lock:
            self.lock.extend(additional_time)


# Singleton instances
redis_client = RedisClient()


# Convenience functions
def get_redis(name: str = "default") -> redis.Redis:
    """Get sync Redis client"""
    return redis_client.get_client(name)


def get_async_redis(name: str = "default") -> aioredis.Redis:
    """Get async Redis client"""
    return redis_client.get_async_client(name)


async def redis_health_check() -> Dict[str, bool]:
    """Check Redis health"""
    return await redis_client.health_check()


# Export main components
__all__ = [
    "RedisConfig",
    "RedisClient",
    "RedisSentinelClient",
    "RedisLock",
    "redis_client",
    "get_redis",
    "get_async_redis",
    "redis_health_check",
]
