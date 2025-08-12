#!/usr/bin/env python3
"""
Comprehensive Error Handling Framework for YTEmpire
Implements circuit breakers, retry logic, and error recovery strategies
"""

import asyncio
import functools
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass, field
import traceback
import json

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"          # Can be ignored or logged
    MEDIUM = "medium"    # Should be handled but not critical
    HIGH = "high"        # Requires immediate handling
    CRITICAL = "critical" # System failure, requires immediate action

class RecoveryStrategy(Enum):
    """Error recovery strategies"""
    RETRY = "retry"                  # Retry the operation
    FALLBACK = "fallback"            # Use fallback service/method
    CIRCUIT_BREAK = "circuit_break"  # Stop trying temporarily
    COMPENSATE = "compensate"        # Run compensation logic
    ESCALATE = "escalate"            # Escalate to higher level
    IGNORE = "ignore"                # Log and continue

@dataclass
class ErrorContext:
    """Context information for error handling"""
    error_type: Type[Exception]
    error_message: str
    severity: ErrorSeverity
    service: str
    operation: str
    timestamp: datetime
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None

@dataclass
class CircuitBreakerState:
    """Circuit breaker state management"""
    is_open: bool = False
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    consecutive_failures: int = 0
    total_requests: int = 0
    total_failures: int = 0

class CircuitBreaker:
    """
    Circuit breaker pattern implementation
    Prevents cascading failures by temporarily blocking calls to failing services
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.state = CircuitBreakerState()
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        async with self._lock:
            # Check if circuit is open
            if self.state.is_open:
                if self._should_attempt_reset():
                    self.state.is_open = False
                    self.state.consecutive_failures = 0
                else:
                    raise CircuitBreakerOpenException(
                        f"Circuit breaker is open. Service unavailable for {self.recovery_timeout}s"
                    )
        
        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Reset on success
            async with self._lock:
                self.state.last_success_time = datetime.now()
                self.state.consecutive_failures = 0
                self.state.total_requests += 1
            
            return result
            
        except self.expected_exception as e:
            async with self._lock:
                self.state.consecutive_failures += 1
                self.state.total_failures += 1
                self.state.total_requests += 1
                self.state.last_failure_time = datetime.now()
                
                if self.state.consecutive_failures >= self.failure_threshold:
                    self.state.is_open = True
                    logger.error(f"Circuit breaker opened after {self.failure_threshold} failures")
            
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if not self.state.last_failure_time:
            return True
        
        time_since_failure = (datetime.now() - self.state.last_failure_time).seconds
        return time_since_failure >= self.recovery_timeout
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state for monitoring"""
        return {
            "is_open": self.state.is_open,
            "consecutive_failures": self.state.consecutive_failures,
            "total_requests": self.state.total_requests,
            "total_failures": self.state.total_failures,
            "failure_rate": (
                self.state.total_failures / self.state.total_requests 
                if self.state.total_requests > 0 else 0
            )
        }

class RetryPolicy:
    """
    Retry policy with exponential backoff
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_delay(self, retry_count: int) -> float:
        """Calculate delay for retry attempt"""
        delay = min(
            self.initial_delay * (self.exponential_base ** retry_count),
            self.max_delay
        )
        
        if self.jitter:
            import random
            delay = delay * (0.5 + random.random())
        
        return delay

class ErrorHandler:
    """
    Central error handling with recovery strategies
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_policies: Dict[str, RetryPolicy] = {}
        self.fallback_handlers: Dict[str, Callable] = {}
        self.error_listeners: List[Callable] = []
        self.metrics: Dict[str, int] = {
            "total_errors": 0,
            "recovered_errors": 0,
            "circuit_breaks": 0,
            "fallbacks_used": 0
        }
    
    def register_circuit_breaker(
        self,
        service: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60
    ):
        """Register a circuit breaker for a service"""
        self.circuit_breakers[service] = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
    
    def register_retry_policy(
        self,
        service: str,
        max_retries: int = 3,
        initial_delay: float = 1.0
    ):
        """Register a retry policy for a service"""
        self.retry_policies[service] = RetryPolicy(
            max_retries=max_retries,
            initial_delay=initial_delay
        )
    
    def register_fallback(self, service: str, fallback_handler: Callable):
        """Register a fallback handler for a service"""
        self.fallback_handlers[service] = fallback_handler
    
    def add_error_listener(self, listener: Callable):
        """Add an error event listener for monitoring"""
        self.error_listeners.append(listener)
    
    async def handle_error(
        self,
        error: Exception,
        context: ErrorContext,
        strategy: RecoveryStrategy = RecoveryStrategy.RETRY
    ) -> Any:
        """Handle error with specified recovery strategy"""
        
        self.metrics["total_errors"] += 1
        
        # Notify listeners
        for listener in self.error_listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(error, context)
                else:
                    listener(error, context)
            except Exception as e:
                logger.error(f"Error listener failed: {e}")
        
        # Apply recovery strategy
        if strategy == RecoveryStrategy.RETRY:
            return await self._handle_retry(error, context)
        elif strategy == RecoveryStrategy.FALLBACK:
            return await self._handle_fallback(error, context)
        elif strategy == RecoveryStrategy.CIRCUIT_BREAK:
            return await self._handle_circuit_break(error, context)
        elif strategy == RecoveryStrategy.COMPENSATE:
            return await self._handle_compensation(error, context)
        elif strategy == RecoveryStrategy.ESCALATE:
            return await self._handle_escalation(error, context)
        else:
            logger.warning(f"Ignoring error: {error}")
            return None
    
    async def _handle_retry(self, error: Exception, context: ErrorContext) -> Any:
        """Handle retry with exponential backoff"""
        service = context.service
        
        if service not in self.retry_policies:
            self.retry_policies[service] = RetryPolicy()
        
        policy = self.retry_policies[service]
        
        if context.retry_count >= policy.max_retries:
            logger.error(f"Max retries ({policy.max_retries}) exceeded for {service}")
            raise error
        
        delay = policy.get_delay(context.retry_count)
        logger.info(f"Retrying {service} after {delay:.2f}s (attempt {context.retry_count + 1})")
        
        await asyncio.sleep(delay)
        context.retry_count += 1
        
        # Retry logic should be implemented by the caller
        self.metrics["recovered_errors"] += 1
        return None
    
    async def _handle_fallback(self, error: Exception, context: ErrorContext) -> Any:
        """Handle fallback to alternative service"""
        service = context.service
        
        if service not in self.fallback_handlers:
            logger.error(f"No fallback handler for {service}")
            raise error
        
        logger.info(f"Using fallback for {service}")
        self.metrics["fallbacks_used"] += 1
        
        fallback = self.fallback_handlers[service]
        if asyncio.iscoroutinefunction(fallback):
            return await fallback(context)
        return fallback(context)
    
    async def _handle_circuit_break(self, error: Exception, context: ErrorContext) -> Any:
        """Handle circuit breaker activation"""
        service = context.service
        
        if service not in self.circuit_breakers:
            self.register_circuit_breaker(service)
        
        breaker = self.circuit_breakers[service]
        self.metrics["circuit_breaks"] += 1
        
        # Circuit breaker will handle the error
        raise error
    
    async def _handle_compensation(self, error: Exception, context: ErrorContext) -> Any:
        """Handle compensation/rollback logic"""
        logger.info(f"Running compensation for {context.service}")
        
        # Compensation logic should be implemented by specific services
        # This is a placeholder for the pattern
        if "compensation_handler" in context.metadata:
            handler = context.metadata["compensation_handler"]
            if asyncio.iscoroutinefunction(handler):
                return await handler(context)
            return handler(context)
        
        raise error
    
    async def _handle_escalation(self, error: Exception, context: ErrorContext) -> Any:
        """Escalate error to higher level"""
        logger.critical(f"Escalating error from {context.service}: {error}")
        
        # Send alerts, notifications, etc.
        # This could trigger PagerDuty, send emails, etc.
        if context.severity == ErrorSeverity.CRITICAL:
            # Trigger immediate alert
            await self._send_critical_alert(error, context)
        
        raise error
    
    async def _send_critical_alert(self, error: Exception, context: ErrorContext):
        """Send critical error alert"""
        alert_data = {
            "timestamp": context.timestamp.isoformat(),
            "service": context.service,
            "operation": context.operation,
            "error": str(error),
            "severity": context.severity.value,
            "metadata": context.metadata
        }
        
        logger.critical(f"CRITICAL ALERT: {json.dumps(alert_data)}")
        
        # Implement actual alerting (email, Slack, PagerDuty, etc.)
        # This is a placeholder
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get error handling metrics"""
        metrics = self.metrics.copy()
        
        # Add circuit breaker states
        metrics["circuit_breakers"] = {}
        for service, breaker in self.circuit_breakers.items():
            metrics["circuit_breakers"][service] = breaker.get_state()
        
        return metrics

class ServiceErrorHandler:
    """
    Service-specific error handling
    """
    
    def __init__(self, service_name: str, error_handler: ErrorHandler):
        self.service_name = service_name
        self.error_handler = error_handler
        self.operation_handlers: Dict[str, Callable] = {}
    
    def register_operation_handler(self, operation: str, handler: Callable):
        """Register error handler for specific operation"""
        self.operation_handlers[operation] = handler
    
    async def handle(
        self,
        error: Exception,
        operation: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        metadata: Optional[Dict] = None
    ) -> Any:
        """Handle service error"""
        
        context = ErrorContext(
            error_type=type(error),
            error_message=str(error),
            severity=severity,
            service=self.service_name,
            operation=operation,
            timestamp=datetime.now(),
            metadata=metadata or {},
            stack_trace=traceback.format_exc()
        )
        
        # Check for operation-specific handler
        if operation in self.operation_handlers:
            handler = self.operation_handlers[operation]
            if asyncio.iscoroutinefunction(handler):
                return await handler(error, context)
            return handler(error, context)
        
        # Determine recovery strategy based on error type and severity
        strategy = self._determine_strategy(error, context)
        
        return await self.error_handler.handle_error(error, context, strategy)
    
    def _determine_strategy(
        self,
        error: Exception,
        context: ErrorContext
    ) -> RecoveryStrategy:
        """Determine recovery strategy based on error and context"""
        
        # Critical errors should escalate
        if context.severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.ESCALATE
        
        # Transient errors should retry
        transient_errors = (
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError
        )
        if isinstance(error, transient_errors):
            return RecoveryStrategy.RETRY
        
        # Service-specific strategies
        if self.service_name in ["openai", "anthropic", "elevenlabs"]:
            # AI services should use fallback
            return RecoveryStrategy.FALLBACK
        
        # Default to retry for medium severity
        if context.severity == ErrorSeverity.MEDIUM:
            return RecoveryStrategy.RETRY
        
        # Low severity can be ignored
        if context.severity == ErrorSeverity.LOW:
            return RecoveryStrategy.IGNORE
        
        return RecoveryStrategy.ESCALATE

# Decorators for easy error handling

def with_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    exceptions: tuple = (Exception,)
):
    """Decorator for automatic retry with exponential backoff"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            policy = RetryPolicy(max_retries=max_retries, initial_delay=initial_delay)
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = policy.get_delay(attempt)
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                            f"after {delay:.2f}s: {e}"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Max retries exceeded for {func.__name__}: {e}")
            
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            policy = RetryPolicy(max_retries=max_retries, initial_delay=initial_delay)
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = policy.get_delay(attempt)
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                            f"after {delay:.2f}s: {e}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"Max retries exceeded for {func.__name__}: {e}")
            
            raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator

def with_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    expected_exception: Type[Exception] = Exception
):
    """Decorator for circuit breaker pattern"""
    breaker = CircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exception=expected_exception
    )
    
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we need to run in an event loop
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(breaker.call(func, *args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator

def with_fallback(fallback_func: Callable):
    """Decorator for fallback on error"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Falling back from {func.__name__}: {e}")
                if asyncio.iscoroutinefunction(fallback_func):
                    return await fallback_func(*args, **kwargs)
                return fallback_func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Falling back from {func.__name__}: {e}")
                return fallback_func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator

# Custom Exceptions

class CircuitBreakerOpenException(Exception):
    """Raised when circuit breaker is open"""
    pass

class RetryExhaustedException(Exception):
    """Raised when all retries are exhausted"""
    pass

class FallbackException(Exception):
    """Raised when fallback also fails"""
    pass

# Global error handler instance
error_handler = ErrorHandler()

# Initialize default circuit breakers and retry policies
def initialize_error_handling():
    """Initialize error handling for all services"""
    
    # AI Services
    error_handler.register_circuit_breaker("openai", failure_threshold=5, recovery_timeout=60)
    error_handler.register_circuit_breaker("anthropic", failure_threshold=5, recovery_timeout=60)
    error_handler.register_circuit_breaker("elevenlabs", failure_threshold=3, recovery_timeout=30)
    
    # YouTube Service
    error_handler.register_circuit_breaker("youtube", failure_threshold=3, recovery_timeout=120)
    
    # Payment Services
    error_handler.register_circuit_breaker("stripe", failure_threshold=2, recovery_timeout=180)
    
    # Database
    error_handler.register_circuit_breaker("database", failure_threshold=10, recovery_timeout=30)
    
    # Redis
    error_handler.register_circuit_breaker("redis", failure_threshold=10, recovery_timeout=20)
    
    # Retry policies
    error_handler.register_retry_policy("openai", max_retries=3, initial_delay=2.0)
    error_handler.register_retry_policy("youtube", max_retries=5, initial_delay=1.0)
    error_handler.register_retry_policy("database", max_retries=3, initial_delay=0.5)
    
    logger.info("Error handling framework initialized")

if __name__ == "__main__":
    # Example usage
    initialize_error_handling()
    
    # Example with decorators
    @with_retry(max_retries=3)
    @with_circuit_breaker(failure_threshold=5)
    async def risky_operation():
        """Example operation that might fail"""
        import random
        if random.random() < 0.7:
            raise ConnectionError("Service unavailable")
        return "Success!"
    
    # Run example
    async def main():
        for i in range(10):
            try:
                result = await risky_operation()
                print(f"Attempt {i+1}: {result}")
            except Exception as e:
                print(f"Attempt {i+1} failed: {e}")
            await asyncio.sleep(1)
        
        # Print metrics
        print("\nMetrics:", error_handler.get_metrics())
    
    asyncio.run(main())