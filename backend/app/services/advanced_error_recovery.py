"""
Advanced Error Recovery Service
Implements sophisticated error recovery mechanisms with self-healing capabilities
"""

import asyncio
import logging
import traceback
from typing import Any, Dict, List, Optional, Callable, TypeVar, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
import json
import hashlib

import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

# Type variable for generic functions
T = TypeVar("T")

# Metrics
error_recovery_counter = Counter(
    "error_recovery_attempts",
    "Error recovery attempts",
    ["service", "error_type", "recovery_strategy"],
)
recovery_success_counter = Counter(
    "error_recovery_success",
    "Successful error recoveries",
    ["service", "recovery_strategy"],
)
circuit_breaker_state = Gauge(
    "circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 2=half-open)",
    ["service"],
)
recovery_time_histogram = Histogram(
    "error_recovery_duration_seconds",
    "Time taken for error recovery",
    ["service", "strategy"],
)


class RecoveryStrategy(Enum):
    """Available recovery strategies"""

    RETRY = "retry"
    CIRCUIT_BREAKER = "circuit_breaker"
    FALLBACK = "fallback"
    CACHE_FALLBACK = "cache_fallback"
    DEGRADED_MODE = "degraded_mode"
    COMPENSATION = "compensation"
    BULKHEAD = "bulkhead"
    TIMEOUT = "timeout"
    HEDGE_REQUEST = "hedge_request"


class CircuitState(Enum):
    """Circuit breaker states"""

    CLOSED = 0  # Normal operation
    OPEN = 1  # Failing, reject requests
    HALF_OPEN = 2  # Testing recovery


@dataclass
class ErrorContext:
    """Context for error recovery"""

    service_name: str
    operation: str
    error_type: str
    error_message: str
    timestamp: datetime
    attempt_number: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    traceback: Optional[str] = None


@dataclass
class RecoveryConfig:
    """Configuration for recovery strategies"""

    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    retry_jitter: bool = True

    # Circuit breaker configuration
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_requests: int = 3

    # Timeout configuration
    operation_timeout: float = 30.0

    # Bulkhead configuration
    max_concurrent_calls: int = 10
    queue_size: int = 50

    # Cache fallback
    cache_ttl: int = 3600
    stale_cache_acceptable: bool = True

    # Degraded mode
    degraded_features: List[str] = field(default_factory=list)

    # Hedge requests
    hedge_delay: float = 2.0
    max_hedge_requests: int = 2


class CircuitBreaker:
    """Circuit breaker implementation"""

    def __init__(self, name: str, config: RecoveryConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.success_count = 0
        self._lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if await self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    circuit_breaker_state.labels(service=self.name).set(2)
                else:
                    raise Exception(f"Circuit breaker is OPEN for {self.name}")

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise

    async def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if self.last_failure_time:
            time_since_failure = (
                datetime.now() - self.last_failure_time
            ).total_seconds()
            return time_since_failure >= self.config.recovery_timeout
        return False

    async def _on_success(self):
        """Handle successful call"""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.half_open_requests:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    circuit_breaker_state.labels(service=self.name).set(0)
                    logger.info(f"Circuit breaker {self.name} is now CLOSED")
            else:
                self.failure_count = 0

    async def _on_failure(self):
        """Handle failed call"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                circuit_breaker_state.labels(service=self.name).set(1)
                logger.warning(f"Circuit breaker {self.name} is now OPEN")

            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.success_count = 0
                circuit_breaker_state.labels(service=self.name).set(1)


class Bulkhead:
    """Bulkhead isolation pattern implementation"""

    def __init__(self, name: str, config: RecoveryConfig):
        self.name = name
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrent_calls)
        self.queue: deque = deque(maxlen=config.queue_size)
        self.active_calls = 0

    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with bulkhead isolation"""
        if self.active_calls >= self.config.max_concurrent_calls:
            if len(self.queue) >= self.config.queue_size:
                raise Exception(f"Bulkhead queue full for {self.name}")

            # Queue the request
            future = asyncio.Future()
            self.queue.append((func, args, kwargs, future))
            return await future

        async with self.semaphore:
            self.active_calls += 1
            try:
                result = await func(*args, **kwargs)

                # Process queued requests
                if self.queue:
                    next_request = self.queue.popleft()
                    asyncio.create_task(self._process_queued(*next_request))

                return result
            finally:
                self.active_calls -= 1

    async def _process_queued(self, func, args, kwargs, future):
        """Process queued request"""
        try:
            result = await self.call(func, *args, **kwargs)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)


class AdvancedErrorRecovery:
    """Advanced error recovery service with multiple strategies"""

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.bulkheads: Dict[str, Bulkhead] = {}
        self.fallback_chains: Dict[str, List[Callable]] = {}
        self.compensation_handlers: Dict[str, Callable] = {}
        self.cache_store: Dict[str, Any] = {}
        self.is_initialized = False

    async def initialize(self):
        """Initialize the error recovery service"""
        if self.redis_url:
            try:
                self.redis_client = await redis.from_url(self.redis_url)
                await self.redis_client.ping()
                logger.info("Advanced error recovery service initialized with Redis")
            except Exception as e:
                logger.warning(f"Redis connection failed, using in-memory cache: {e}")
                self.redis_client = None

        self.is_initialized = True

    async def shutdown(self):
        """Shutdown the service"""
        if self.redis_client:
            await self.redis_client.close()

    def register_circuit_breaker(
        self, service_name: str, config: Optional[RecoveryConfig] = None
    ):
        """Register a circuit breaker for a service"""
        if service_name not in self.circuit_breakers:
            config = config or RecoveryConfig()
            self.circuit_breakers[service_name] = CircuitBreaker(service_name, config)
            logger.info(f"Circuit breaker registered for {service_name}")

    def register_bulkhead(
        self, service_name: str, config: Optional[RecoveryConfig] = None
    ):
        """Register a bulkhead for a service"""
        if service_name not in self.bulkheads:
            config = config or RecoveryConfig()
            self.bulkheads[service_name] = Bulkhead(service_name, config)
            logger.info(f"Bulkhead registered for {service_name}")

    def register_fallback_chain(self, service_name: str, fallbacks: List[Callable]):
        """Register a fallback chain for a service"""
        self.fallback_chains[service_name] = fallbacks
        logger.info(
            f"Fallback chain registered for {service_name} with {len(fallbacks)} fallbacks"
        )

    def register_compensation_handler(self, operation: str, handler: Callable):
        """Register a compensation handler for an operation"""
        self.compensation_handlers[operation] = handler
        logger.info(f"Compensation handler registered for {operation}")

    async def with_retry(
        self,
        func: Callable[..., T],
        context: ErrorContext,
        config: Optional[RecoveryConfig] = None,
        *args,
        **kwargs,
    ) -> T:
        """Execute function with retry logic"""
        config = config or RecoveryConfig()
        last_exception = None

        for attempt in range(config.max_retries):
            try:
                context.attempt_number = attempt + 1
                error_recovery_counter.labels(
                    service=context.service_name,
                    error_type=context.error_type,
                    recovery_strategy=RecoveryStrategy.RETRY.value,
                ).inc()

                result = await func(*args, **kwargs)

                recovery_success_counter.labels(
                    service=context.service_name,
                    recovery_strategy=RecoveryStrategy.RETRY.value,
                ).inc()

                return result

            except Exception as e:
                last_exception = e
                context.error_message = str(e)
                context.traceback = traceback.format_exc()

                if attempt < config.max_retries - 1:
                    delay = config.retry_delay * (config.retry_backoff**attempt)
                    if config.retry_jitter:
                        import random

                        delay *= 0.5 + random.random()

                    logger.warning(
                        f"Retry {attempt + 1}/{config.max_retries} for {context.service_name}::"
                        f"{context.operation} after {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"All retries exhausted for {context.service_name}::{context.operation}"
                    )

        raise last_exception

    async def with_circuit_breaker(
        self, func: Callable[..., T], service_name: str, *args, **kwargs
    ) -> T:
        """Execute function with circuit breaker protection"""
        if service_name not in self.circuit_breakers:
            self.register_circuit_breaker(service_name)

        breaker = self.circuit_breakers[service_name]

        error_recovery_counter.labels(
            service=service_name,
            error_type="circuit_breaker",
            recovery_strategy=RecoveryStrategy.CIRCUIT_BREAKER.value,
        ).inc()

        try:
            result = await breaker.call(func, *args, **kwargs)

            recovery_success_counter.labels(
                service=service_name,
                recovery_strategy=RecoveryStrategy.CIRCUIT_BREAKER.value,
            ).inc()

            return result
        except Exception as e:
            logger.error(f"Circuit breaker failed for {service_name}: {e}")
            raise

    async def with_fallback(
        self, primary_func: Callable[..., T], service_name: str, *args, **kwargs
    ) -> T:
        """Execute function with fallback chain"""
        fallbacks = self.fallback_chains.get(service_name, [])
        functions = [primary_func] + fallbacks

        last_exception = None

        for i, func in enumerate(functions):
            try:
                error_recovery_counter.labels(
                    service=service_name,
                    error_type="fallback",
                    recovery_strategy=RecoveryStrategy.FALLBACK.value,
                ).inc()

                result = await func(*args, **kwargs)

                if i > 0:
                    logger.info(f"Fallback {i} succeeded for {service_name}")

                recovery_success_counter.labels(
                    service=service_name,
                    recovery_strategy=RecoveryStrategy.FALLBACK.value,
                ).inc()

                return result

            except Exception as e:
                last_exception = e
                logger.warning(f"Function {i} failed for {service_name}: {e}")

                if i < len(functions) - 1:
                    logger.info(f"Trying fallback {i + 1} for {service_name}")

        logger.error(f"All fallbacks exhausted for {service_name}")
        raise last_exception

    async def with_cache_fallback(
        self, func: Callable[..., T], cache_key: str, ttl: int = 3600, *args, **kwargs
    ) -> T:
        """Execute function with cache fallback"""
        # Try to get from cache first
        cached_value = await self._get_from_cache(cache_key)
        if cached_value is not None:
            logger.info(f"Returning cached value for {cache_key}")
            return cached_value

        try:
            result = await func(*args, **kwargs)

            # Cache the result
            await self._set_in_cache(cache_key, result, ttl)

            return result

        except Exception as e:
            # Try stale cache
            stale_value = await self._get_from_cache(cache_key, accept_stale=True)
            if stale_value is not None:
                logger.warning(
                    f"Returning stale cached value for {cache_key} due to error: {e}"
                )
                return stale_value

            raise

    async def with_bulkhead(
        self, func: Callable[..., T], service_name: str, *args, **kwargs
    ) -> T:
        """Execute function with bulkhead isolation"""
        if service_name not in self.bulkheads:
            self.register_bulkhead(service_name)

        bulkhead = self.bulkheads[service_name]

        error_recovery_counter.labels(
            service=service_name,
            error_type="bulkhead",
            recovery_strategy=RecoveryStrategy.BULKHEAD.value,
        ).inc()

        try:
            result = await bulkhead.call(func, *args, **kwargs)

            recovery_success_counter.labels(
                service=service_name, recovery_strategy=RecoveryStrategy.BULKHEAD.value
            ).inc()

            return result
        except Exception as e:
            logger.error(f"Bulkhead failed for {service_name}: {e}")
            raise

    async def with_timeout(
        self, func: Callable[..., T], timeout: float, service_name: str, *args, **kwargs
    ) -> T:
        """Execute function with timeout"""
        error_recovery_counter.labels(
            service=service_name,
            error_type="timeout",
            recovery_strategy=RecoveryStrategy.TIMEOUT.value,
        ).inc()

        try:
            result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)

            recovery_success_counter.labels(
                service=service_name, recovery_strategy=RecoveryStrategy.TIMEOUT.value
            ).inc()

            return result

        except asyncio.TimeoutError:
            logger.error(f"Timeout after {timeout}s for {service_name}")
            raise

    async def with_hedge_request(
        self,
        func: Callable[..., T],
        service_name: str,
        hedge_delay: float = 2.0,
        max_hedges: int = 2,
        *args,
        **kwargs,
    ) -> T:
        """Execute function with hedge requests for lower latency"""
        tasks = []

        # Start primary request
        primary_task = asyncio.create_task(func(*args, **kwargs))
        tasks.append(primary_task)

        # Schedule hedge requests
        for i in range(max_hedges):
            await asyncio.sleep(hedge_delay)

            # Check if primary completed
            if primary_task.done():
                try:
                    return primary_task.result()
                except Exception:
                    pass

            # Start hedge request
            logger.info(f"Starting hedge request {i + 1} for {service_name}")
            hedge_task = asyncio.create_task(func(*args, **kwargs))
            tasks.append(hedge_task)

        # Wait for first successful response
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        # Cancel pending tasks
        for task in pending:
            task.cancel()

        # Return first successful result
        for task in done:
            try:
                return task.result()
            except Exception:
                continue

        raise Exception(f"All hedge requests failed for {service_name}")

    async def with_compensation(
        self,
        func: Callable[..., T],
        operation: str,
        compensation_data: Dict[str, Any],
        *args,
        **kwargs,
    ) -> T:
        """Execute function with compensation handler for rollback"""
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Operation {operation} failed, executing compensation")

            if operation in self.compensation_handlers:
                try:
                    await self.compensation_handlers[operation](compensation_data)
                    logger.info(f"Compensation successful for {operation}")
                except Exception as comp_error:
                    logger.error(f"Compensation failed for {operation}: {comp_error}")

            raise

    async def with_degraded_mode(
        self,
        func: Callable[..., T],
        degraded_func: Callable[..., T],
        service_name: str,
        *args,
        **kwargs,
    ) -> T:
        """Execute function with degraded mode fallback"""
        try:
            # Try normal operation
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            logger.warning(f"Switching to degraded mode for {service_name}: {e}")

            error_recovery_counter.labels(
                service=service_name,
                error_type="degraded",
                recovery_strategy=RecoveryStrategy.DEGRADED_MODE.value,
            ).inc()

            # Execute degraded version
            result = await degraded_func(*args, **kwargs)

            recovery_success_counter.labels(
                service=service_name,
                recovery_strategy=RecoveryStrategy.DEGRADED_MODE.value,
            ).inc()

            return result

    async def _get_from_cache(
        self, key: str, accept_stale: bool = False
    ) -> Optional[Any]:
        """Get value from cache"""
        if self.redis_client:
            try:
                value = await self.redis_client.get(key)
                if value:
                    return json.loads(value)
            except Exception as e:
                logger.warning(f"Cache get failed: {e}")

        # Fallback to in-memory cache
        if key in self.cache_store:
            return self.cache_store[key]

        return None

    async def _set_in_cache(self, key: str, value: Any, ttl: int):
        """Set value in cache"""
        if self.redis_client:
            try:
                await self.redis_client.set(key, json.dumps(value), ex=ttl)
            except Exception as e:
                logger.warning(f"Cache set failed: {e}")

        # Fallback to in-memory cache
        self.cache_store[key] = value

    def get_circuit_breaker_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers"""
        status = {}
        for name, breaker in self.circuit_breakers.items():
            status[name] = {
                "state": breaker.state.name,
                "failure_count": breaker.failure_count,
                "last_failure": breaker.last_failure_time.isoformat()
                if breaker.last_failure_time
                else None,
            }
        return status

    def get_bulkhead_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all bulkheads"""
        status = {}
        for name, bulkhead in self.bulkheads.items():
            status[name] = {
                "active_calls": bulkhead.active_calls,
                "queue_size": len(bulkhead.queue),
                "max_concurrent": bulkhead.config.max_concurrent_calls,
            }
        return status


# Singleton instance
advanced_recovery = AdvancedErrorRecovery()


# Decorator functions for easy use
def with_recovery(
    strategies: List[RecoveryStrategy], service_name: str, **config_kwargs
):
    """Decorator to apply recovery strategies to a function"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            result = func

            # Apply strategies in order
            for strategy in strategies:
                if strategy == RecoveryStrategy.RETRY:
                    context = ErrorContext(
                        service_name=service_name,
                        operation=func.__name__,
                        error_type="unknown",
                        error_message="",
                        timestamp=datetime.now(),
                    )
                    config = RecoveryConfig(**config_kwargs)
                    result = await advanced_recovery.with_retry(
                        result, context, config, *args, **kwargs
                    )

                elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                    result = await advanced_recovery.with_circuit_breaker(
                        result, service_name, *args, **kwargs
                    )

                elif strategy == RecoveryStrategy.BULKHEAD:
                    result = await advanced_recovery.with_bulkhead(
                        result, service_name, *args, **kwargs
                    )

                elif strategy == RecoveryStrategy.TIMEOUT:
                    timeout = config_kwargs.get("operation_timeout", 30.0)
                    result = await advanced_recovery.with_timeout(
                        result, timeout, service_name, *args, **kwargs
                    )

            return result

        return wrapper

    return decorator
