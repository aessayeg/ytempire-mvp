"""
Error Handling Framework for YTEmpire Services
Provides comprehensive error handling and recovery patterns
"""

import logging
import traceback
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass
from functools import wraps

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories"""
    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_API = "external_api"
    DATABASE = "database"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RESOURCE = "resource"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for errors"""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    details: Dict[str, Any]
    traceback: Optional[str]
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    service_name: Optional[str] = None


class ErrorHandler:
    """
    Central error handling system
    """
    
    def __init__(self):
        self.error_handlers: Dict[str, List[Callable]] = {}
        self.error_log: List[ErrorContext] = []
        self.max_log_size = 1000
        
    def register_handler(self, category: ErrorCategory, handler: Callable):
        """Register error handler for specific category"""
        category_str = category.value
        if category_str not in self.error_handlers:
            self.error_handlers[category_str] = []
        self.error_handlers[category_str].append(handler)
    
    async def handle_error(
        self,
        error: Exception,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        context: Dict[str, Any] = None,
        user_id: str = None,
        request_id: str = None,
        service_name: str = None
    ) -> ErrorContext:
        """Handle error with context and severity"""
        
        import uuid
        error_id = str(uuid.uuid4())
        
        error_context = ErrorContext(
            error_id=error_id,
            timestamp=datetime.utcnow(),
            severity=severity,
            category=category,
            message=str(error),
            details=context or {},
            traceback=traceback.format_exc(),
            user_id=user_id,
            request_id=request_id,
            service_name=service_name
        )
        
        # Add to error log
        self.error_log.append(error_context)
        if len(self.error_log) > self.max_log_size:
            self.error_log.pop(0)
        
        # Log error
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(severity, logging.ERROR)
        
        logger.log(
            log_level,
            f"Error {error_id} [{category.value}:{severity.value}]: {error_context.message}",
            extra={
                'error_id': error_id,
                'category': category.value,
                'severity': severity.value,
                'context': context,
                'user_id': user_id,
                'request_id': request_id,
                'service_name': service_name
            }
        )
        
        # Execute registered handlers
        category_handlers = self.error_handlers.get(category.value, [])
        for handler in category_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(error_context)
                else:
                    handler(error_context)
            except Exception as handler_error:
                logger.error(f"Error in error handler: {handler_error}")
        
        return error_context
    
    def get_recent_errors(self, limit: int = 50) -> List[ErrorContext]:
        """Get recent errors"""
        return self.error_log[-limit:]
    
    def get_errors_by_severity(self, severity: ErrorSeverity) -> List[ErrorContext]:
        """Get errors by severity"""
        return [error for error in self.error_log if error.severity == severity]
    
    def get_errors_by_category(self, category: ErrorCategory) -> List[ErrorContext]:
        """Get errors by category"""
        return [error for error in self.error_log if error.category == category]


# Global error handler
error_handler = ErrorHandler()

# Alias for backward compatibility 
ServiceErrorHandler = ErrorHandler


def handle_errors(
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    reraise: bool = True
):
    """Decorator for automatic error handling"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                await error_handler.handle_error(
                    error=e,
                    severity=severity,
                    category=category,
                    service_name=func.__module__
                )
                if reraise:
                    raise
                return None
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                asyncio.create_task(error_handler.handle_error(
                    error=e,
                    severity=severity,
                    category=category,
                    service_name=func.__module__
                ))
                if reraise:
                    raise
                return None
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def safe_execute(
    func: Callable,
    *args,
    default_return: Any = None,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    **kwargs
) -> Any:
    """Safely execute function with error handling"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        asyncio.create_task(error_handler.handle_error(
            error=e,
            severity=severity,
            category=category
        ))
        return default_return


async def safe_execute_async(
    func: Callable,
    *args,
    default_return: Any = None,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    **kwargs
) -> Any:
    """Safely execute async function with error handling"""
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        await error_handler.handle_error(
            error=e,
            severity=severity,
            category=category
        )
        return default_return


# Export commonly used items
__all__ = [
    'ErrorSeverity',
    'ErrorCategory',
    'ErrorContext',
    'ErrorHandler',
    'ServiceErrorHandler',
    'error_handler',
    'handle_errors',
    'safe_execute',
    'safe_execute_async'
]