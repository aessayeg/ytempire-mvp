"""
Global Error Middleware for FastAPI
Centralized error catching, formatting, and reporting
"""

import time
import json
import logging
import traceback
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError, HTTPException
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy.exc import SQLAlchemyError
import redis.exceptions

from app.core.exceptions import (
    YTEmpireException,
    APIException,
    AuthenticationException,
    ResourceNotFoundException,
    ValidationException,
    QuotaExceededException,
    ThresholdExceededException,
    ExternalServiceException,
    DatabaseException,
    VideoProcessingException,
    RateLimitException,
)

logger = logging.getLogger(__name__)


class ErrorMetrics:
    """Track error metrics for monitoring"""

    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.error_rates: Dict[str, list] = {}
        self.last_errors: Dict[str, Any] = {}
        self.start_time = time.time()

    def record_error(self, error_type: str, error_code: str, path: str):
        """Record an error occurrence"""
        key = f"{error_type}:{error_code}"

        # Increment count
        self.error_counts[key] = self.error_counts.get(key, 0) + 1

        # Track rate (errors per minute)
        current_minute = int(time.time() / 60)
        if key not in self.error_rates:
            self.error_rates[key] = []

        # Keep only last 5 minutes
        self.error_rates[key] = [
            (t, c) for t, c in self.error_rates[key] if current_minute - t < 5
        ]

        # Add current error
        if self.error_rates[key] and self.error_rates[key][-1][0] == current_minute:
            self.error_rates[key][-1] = (
                current_minute,
                self.error_rates[key][-1][1] + 1,
            )
        else:
            self.error_rates[key].append((current_minute, 1))

        # Store last error
        self.last_errors[key] = {
            "timestamp": datetime.now().isoformat(),
            "path": path,
            "count": self.error_counts[key],
        }

    def get_error_rate(self, error_type: str, error_code: str) -> float:
        """Get current error rate per minute"""
        key = f"{error_type}:{error_code}"
        if key not in self.error_rates or not self.error_rates[key]:
            return 0.0

        total_errors = sum(count for _, count in self.error_rates[key])
        minutes = len(self.error_rates[key])
        return total_errors / minutes if minutes > 0 else 0.0

    def get_metrics(self) -> Dict[str, Any]:
        """Get all error metrics"""
        uptime = time.time() - self.start_time

        return {
            "uptime_seconds": uptime,
            "total_errors": sum(self.error_counts.values()),
            "error_counts": self.error_counts,
            "error_rates": {
                key: self.get_error_rate(*key.split(":"))
                for key in self.error_counts.keys()
            },
            "last_errors": self.last_errors,
        }


# Global error metrics instance
error_metrics = ErrorMetrics()


class GlobalErrorMiddleware(BaseHTTPMiddleware):
    """
    Global error handling middleware
    Catches all exceptions and returns consistent error responses
    """

    async def dispatch(self, request: Request, call_next):
        """Process request and handle any errors"""
        try:
            # Add request ID for tracing
            request_id = request.headers.get("X-Request-ID", str(time.time()))
            request.state.request_id = request_id

            # Process request
            response = await call_next(request)

            return response

        except Exception as exc:
            # Handle the error
            return await self.handle_error(exc, request)

    async def handle_error(self, exc: Exception, request: Request) -> JSONResponse:
        """Handle different types of errors"""

        request_id = getattr(request.state, "request_id", "unknown")
        path = request.url.path
        method = request.method

        # Log the error with context
        logger.error(
            f"Error handling {method} {path} (Request ID: {request_id})",
            exc_info=True,
            extra={
                "request_id": request_id,
                "path": path,
                "method": method,
                "error_type": type(exc).__name__,
            },
        )

        # Determine error response based on exception type
        if isinstance(exc, APIException):
            return await self._handle_api_exception(exc, request_id, path)

        elif isinstance(exc, YTEmpireException):
            return await self._handle_ytempire_exception(exc, request_id, path)

        elif isinstance(exc, HTTPException):
            return await self._handle_http_exception(exc, request_id, path)

        elif isinstance(exc, RequestValidationError):
            return await self._handle_validation_error(exc, request_id, path)

        elif isinstance(exc, SQLAlchemyError):
            return await self._handle_database_error(exc, request_id, path)

        elif isinstance(exc, redis.exceptions.RedisError):
            return await self._handle_redis_error(exc, request_id, path)

        elif isinstance(exc, (ConnectionError, TimeoutError)):
            return await self._handle_connection_error(exc, request_id, path)

        else:
            return await self._handle_unexpected_error(exc, request_id, path)

    async def _handle_api_exception(
        self, exc: APIException, request_id: str, path: str
    ) -> JSONResponse:
        """Handle API exceptions"""

        error_metrics.record_error("api", exc.error_code, path)

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.error_code,
                    "message": exc.message,
                    "details": exc.details,
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat(),
                }
            },
        )

    async def _handle_ytempire_exception(
        self, exc: YTEmpireException, request_id: str, path: str
    ) -> JSONResponse:
        """Handle YTEmpire custom exceptions"""

        error_metrics.record_error("ytempire", exc.error_code, path)

        # Map to appropriate HTTP status code
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

        if isinstance(exc, ExternalServiceException):
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        elif isinstance(exc, DatabaseException):
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        elif isinstance(exc, VideoProcessingException):
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY

        return JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "code": exc.error_code,
                    "message": exc.message,
                    "details": exc.details,
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat(),
                }
            },
        )

    async def _handle_http_exception(
        self, exc: HTTPException, request_id: str, path: str
    ) -> JSONResponse:
        """Handle HTTP exceptions"""

        error_metrics.record_error("http", str(exc.status_code), path)

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": f"HTTP_{exc.status_code}",
                    "message": exc.detail,
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat(),
                }
            },
        )

    async def _handle_validation_error(
        self, exc: RequestValidationError, request_id: str, path: str
    ) -> JSONResponse:
        """Handle request validation errors"""

        error_metrics.record_error("validation", "VALIDATION_ERROR", path)

        # Format validation errors
        errors = []
        for error in exc.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            errors.append(
                {"field": field, "message": error["msg"], "type": error["type"]}
            )

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed",
                    "details": {"validation_errors": errors},
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat(),
                }
            },
        )

    async def _handle_database_error(
        self, exc: SQLAlchemyError, request_id: str, path: str
    ) -> JSONResponse:
        """Handle database errors"""

        error_metrics.record_error("database", "DATABASE_ERROR", path)

        # Don't expose internal database errors
        message = "Database operation failed"

        # Log the actual error
        logger.error(f"Database error: {exc}", exc_info=True)

        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": {
                    "code": "DATABASE_ERROR",
                    "message": message,
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat(),
                }
            },
        )

    async def _handle_redis_error(
        self, exc: redis.exceptions.RedisError, request_id: str, path: str
    ) -> JSONResponse:
        """Handle Redis errors"""

        error_metrics.record_error("redis", "CACHE_ERROR", path)

        logger.error(f"Redis error: {exc}", exc_info=True)

        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": {
                    "code": "CACHE_ERROR",
                    "message": "Cache service temporarily unavailable",
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat(),
                }
            },
        )

    async def _handle_connection_error(
        self, exc: Exception, request_id: str, path: str
    ) -> JSONResponse:
        """Handle connection errors"""

        error_metrics.record_error("connection", "CONNECTION_ERROR", path)

        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": {
                    "code": "CONNECTION_ERROR",
                    "message": "Service temporarily unavailable",
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat(),
                }
            },
        )

    async def _handle_unexpected_error(
        self, exc: Exception, request_id: str, path: str
    ) -> JSONResponse:
        """Handle unexpected errors"""

        error_metrics.record_error("unexpected", "INTERNAL_ERROR", path)

        # Log full traceback for unexpected errors
        logger.critical(
            f"Unexpected error: {exc}",
            exc_info=True,
            extra={
                "request_id": request_id,
                "path": path,
                "traceback": traceback.format_exc(),
            },
        )

        # Check if we should send alerts for critical errors
        error_rate = error_metrics.get_error_rate("unexpected", "INTERNAL_ERROR")
        if error_rate > 5:  # More than 5 unexpected errors per minute
            await self._send_critical_alert(exc, request_id, path, error_rate)

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred",
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat(),
                }
            },
        )

    async def _send_critical_alert(
        self, exc: Exception, request_id: str, path: str, error_rate: float
    ):
        """Send alert for critical errors"""

        alert_data = {
            "level": "CRITICAL",
            "service": "ytempire-backend",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "request_id": request_id,
            "path": path,
            "error_rate_per_minute": error_rate,
            "timestamp": datetime.now().isoformat(),
        }

        logger.critical(f"CRITICAL ALERT: {json.dumps(alert_data)}")

        # TODO: Integrate with actual alerting service
        # - Send to Slack
        # - Send to PagerDuty
        # - Send email to ops team


def create_error_handlers(app):
    """Create error handlers for FastAPI app"""

    @app.exception_handler(APIException)
    async def api_exception_handler(request: Request, exc: APIException):
        """Handle API exceptions"""
        middleware = GlobalErrorMiddleware(app)
        return await middleware._handle_api_exception(
            exc, getattr(request.state, "request_id", "unknown"), request.url.path
        )

    @app.exception_handler(YTEmpireException)
    async def ytempire_exception_handler(request: Request, exc: YTEmpireException):
        """Handle YTEmpire exceptions"""
        middleware = GlobalErrorMiddleware(app)
        return await middleware._handle_ytempire_exception(
            exc, getattr(request.state, "request_id", "unknown"), request.url.path
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        """Handle validation errors"""
        middleware = GlobalErrorMiddleware(app)
        return await middleware._handle_validation_error(
            exc, getattr(request.state, "request_id", "unknown"), request.url.path
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions"""
        middleware = GlobalErrorMiddleware(app)
        return await middleware._handle_http_exception(
            exc, getattr(request.state, "request_id", "unknown"), request.url.path
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle all other exceptions"""
        middleware = GlobalErrorMiddleware(app)
        return await middleware.handle_error(exc, request)

    # Add metrics endpoint
    @app.get("/api/v1/errors/metrics")
    async def get_error_metrics():
        """Get error metrics for monitoring"""
        return error_metrics.get_metrics()

    logger.info("Error handlers registered")


# Export for use in main app
__all__ = ["GlobalErrorMiddleware", "create_error_handlers", "error_metrics"]
