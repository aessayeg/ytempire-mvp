"""
Error Handling Framework
Comprehensive error handling, logging, and recovery system
"""
from typing import Optional, Dict, Any, Type, Callable, Union
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
import traceback
import sys
from datetime import datetime
from enum import Enum
import asyncio
from functools import wraps
import json

logger = logging.getLogger(__name__)

class ErrorCode(Enum):
    """Application error codes"""
    # Authentication & Authorization (1000-1999)
    INVALID_CREDENTIALS = "AUTH_1001"
    TOKEN_EXPIRED = "AUTH_1002"
    UNAUTHORIZED = "AUTH_1003"
    FORBIDDEN = "AUTH_1004"
    
    # User & Account (2000-2999)
    USER_NOT_FOUND = "USER_2001"
    USER_ALREADY_EXISTS = "USER_2002"
    INVALID_USER_DATA = "USER_2003"
    ACCOUNT_SUSPENDED = "USER_2004"
    
    # Channel Management (3000-3999)
    CHANNEL_NOT_FOUND = "CHANNEL_3001"
    CHANNEL_LIMIT_EXCEEDED = "CHANNEL_3002"
    CHANNEL_NAME_TAKEN = "CHANNEL_3003"
    
    # Video Generation (4000-4999)
    VIDEO_GENERATION_FAILED = "VIDEO_4001"
    VIDEO_QUOTA_EXCEEDED = "VIDEO_4002"
    VIDEO_PROCESSING_ERROR = "VIDEO_4003"
    INVALID_VIDEO_FORMAT = "VIDEO_4004"
    
    # AI Services (5000-5999)
    AI_SERVICE_ERROR = "AI_5001"
    AI_QUOTA_EXCEEDED = "AI_5002"
    AI_RATE_LIMITED = "AI_5003"
    AI_INVALID_PROMPT = "AI_5004"
    
    # Payment & Billing (6000-6999)
    PAYMENT_FAILED = "PAYMENT_6001"
    SUBSCRIPTION_EXPIRED = "PAYMENT_6002"
    INVALID_PAYMENT_METHOD = "PAYMENT_6003"
    BILLING_ERROR = "PAYMENT_6004"
    
    # System & Infrastructure (7000-7999)
    DATABASE_ERROR = "SYSTEM_7001"
    CACHE_ERROR = "SYSTEM_7002"
    QUEUE_ERROR = "SYSTEM_7003"
    STORAGE_ERROR = "SYSTEM_7004"
    
    # External APIs (8000-8999)
    YOUTUBE_API_ERROR = "EXTERNAL_8001"
    STRIPE_API_ERROR = "EXTERNAL_8002"
    OPENAI_API_ERROR = "EXTERNAL_8003"
    THIRD_PARTY_ERROR = "EXTERNAL_8004"
    
    # Validation & Business Logic (9000-9999)
    VALIDATION_ERROR = "VALIDATION_9001"
    BUSINESS_RULE_VIOLATION = "VALIDATION_9002"
    RATE_LIMIT_EXCEEDED = "VALIDATION_9003"
    INVALID_REQUEST = "VALIDATION_9004"

class AppException(Exception):
    """Base application exception"""
    
    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: Optional[Dict[str, Any]] = None,
        internal_message: Optional[str] = None
    ):
        self.error_code = error_code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        self.internal_message = internal_message or message
        self.timestamp = datetime.utcnow().isoformat()
        super().__init__(self.message)

class AuthenticationError(AppException):
    """Authentication-related errors"""
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(
            error_code=ErrorCode.INVALID_CREDENTIALS,
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            **kwargs
        )

class AuthorizationError(AppException):
    """Authorization-related errors"""
    def __init__(self, message: str = "Access forbidden", **kwargs):
        super().__init__(
            error_code=ErrorCode.FORBIDDEN,
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            **kwargs
        )

class ValidationError(AppException):
    """Validation errors"""
    def __init__(self, message: str = "Validation failed", **kwargs):
        super().__init__(
            error_code=ErrorCode.VALIDATION_ERROR,
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            **kwargs
        )

class ResourceNotFoundError(AppException):
    """Resource not found errors"""
    def __init__(self, resource: str, identifier: str, **kwargs):
        super().__init__(
            error_code=ErrorCode.USER_NOT_FOUND,
            message=f"{resource} with identifier {identifier} not found",
            status_code=status.HTTP_404_NOT_FOUND,
            **kwargs
        )

class RateLimitError(AppException):
    """Rate limiting errors"""
    def __init__(self, message: str = "Rate limit exceeded", **kwargs):
        super().__init__(
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            **kwargs
        )

class ExternalServiceError(AppException):
    """External service errors"""
    def __init__(self, service: str, message: str = "External service error", **kwargs):
        super().__init__(
            error_code=ErrorCode.THIRD_PARTY_ERROR,
            message=f"{service}: {message}",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            **kwargs
        )

class ErrorHandler:
    """Central error handling and recovery"""
    
    def __init__(self):
        self.error_callbacks: Dict[ErrorCode, Callable] = {}
        self.recovery_strategies: Dict[ErrorCode, Callable] = {}
        self.error_metrics: Dict[str, int] = {}
        
    def register_error_callback(self, error_code: ErrorCode, callback: Callable):
        """Register callback for specific error code"""
        self.error_callbacks[error_code] = callback
        
    def register_recovery_strategy(self, error_code: ErrorCode, strategy: Callable):
        """Register recovery strategy for error code"""
        self.recovery_strategies[error_code] = strategy
        
    async def handle_error(self, error: AppException) -> Optional[Any]:
        """Handle application error with recovery"""
        # Log error
        logger.error(
            f"Error {error.error_code.value}: {error.internal_message}",
            extra={
                "error_code": error.error_code.value,
                "status_code": error.status_code,
                "details": error.details,
                "timestamp": error.timestamp
            }
        )
        
        # Update metrics
        self.error_metrics[error.error_code.value] = \
            self.error_metrics.get(error.error_code.value, 0) + 1
        
        # Execute callback if registered
        if error.error_code in self.error_callbacks:
            try:
                await self.error_callbacks[error.error_code](error)
            except Exception as e:
                logger.error(f"Error in callback for {error.error_code}: {e}")
        
        # Attempt recovery if strategy exists
        if error.error_code in self.recovery_strategies:
            try:
                return await self.recovery_strategies[error.error_code](error)
            except Exception as e:
                logger.error(f"Recovery failed for {error.error_code}: {e}")
                
        return None
        
    def get_error_response(self, error: AppException) -> JSONResponse:
        """Format error response"""
        return JSONResponse(
            status_code=error.status_code,
            content={
                "error": {
                    "code": error.error_code.value,
                    "message": error.message,
                    "details": error.details,
                    "timestamp": error.timestamp
                }
            }
        )

# Global error handler instance
error_handler = ErrorHandler()

def handle_errors(func):
    """Decorator for error handling"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except AppException as e:
            # Try recovery
            result = await error_handler.handle_error(e)
            if result is not None:
                return result
            raise
        except Exception as e:
            # Convert to AppException
            app_error = AppException(
                error_code=ErrorCode.SYSTEM_7001,
                message="An unexpected error occurred",
                internal_message=str(e),
                details={"traceback": traceback.format_exc()}
            )
            await error_handler.handle_error(app_error)
            raise app_error
    return wrapper

class CircuitBreaker:
    """Circuit breaker for external services"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker"""
        if self.state == "open":
            if self.last_failure_time and \
               (datetime.utcnow() - self.last_failure_time).seconds > self.recovery_timeout:
                self.state = "half-open"
            else:
                raise ExternalServiceError(
                    service="Circuit Breaker",
                    message="Service is temporarily unavailable"
                )
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
            
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
                
            raise

class RetryPolicy:
    """Retry policy for transient failures"""
    
    def __init__(
        self,
        max_retries: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        max_delay: float = 60.0
    ):
        self.max_retries = max_retries
        self.delay = delay
        self.backoff = backoff
        self.max_delay = max_delay
        
    async def execute(self, func: Callable, *args, **kwargs):
        """Execute function with retry policy"""
        last_exception = None
        delay = self.delay
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay} seconds..."
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * self.backoff, self.max_delay)
                    
        raise last_exception

async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for FastAPI"""
    if isinstance(exc, AppException):
        return error_handler.get_error_response(exc)
        
    elif isinstance(exc, StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": "HTTP_ERROR",
                    "message": exc.detail,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        )
        
    elif isinstance(exc, RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": ErrorCode.VALIDATION_ERROR.value,
                    "message": "Request validation failed",
                    "details": exc.errors(),
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        )
        
    else:
        # Log unexpected errors
        logger.error(
            f"Unexpected error: {exc}",
            exc_info=True,
            extra={"traceback": traceback.format_exc()}
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An internal error occurred",
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        )

# Recovery strategies
async def retry_database_operation(error: AppException):
    """Retry database operations on failure"""
    logger.info("Attempting database operation recovery")
    # Implement database retry logic
    return None

async def fallback_to_cache(error: AppException):
    """Fallback to cache on service failure"""
    logger.info("Falling back to cache")
    # Implement cache fallback logic
    return None

# Register recovery strategies
error_handler.register_recovery_strategy(ErrorCode.DATABASE_ERROR, retry_database_operation)
error_handler.register_recovery_strategy(ErrorCode.EXTERNAL_8001, fallback_to_cache)