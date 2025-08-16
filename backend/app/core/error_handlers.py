"""
Global error handlers for FastAPI application
"""
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy.exc import SQLAlchemyError
import traceback
import logging
from typing import Union
from datetime import datetime
import json

from app.core.exceptions import (
    YTEmpireException,
    APIException,
    ExternalServiceException,
    DatabaseException,
)
from app.core.config import settings

logger = logging.getLogger(__name__)


async def ytempire_exception_handler(
    request: Request, exc: YTEmpireException
) -> JSONResponse:
    """Handle custom YTEmpire exceptions"""
    # Log the error
    logger.error(
        f"YTEmpire Exception: {exc.error_code} - {exc.message}",
        extra={
            "error_code": exc.error_code,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method,
            "client": request.client.host if request.client else None,
        },
    )

    # Determine status code
    if isinstance(exc, APIException):
        status_code = exc.status_code
    else:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    # Build response
    response_content = {
        "error": {
            "code": exc.error_code,
            "message": exc.message,
            "details": exc.details,
            "timestamp": datetime.utcnow().isoformat(),
            "path": request.url.path,
        }
    }

    # Add trace ID if available
    if hasattr(request.state, "trace_id"):
        response_content["error"]["trace_id"] = request.state.trace_id

    return JSONResponse(status_code=status_code, content=response_content)


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle request validation errors"""
    # Log the validation error
    logger.warning(
        f"Validation Error: {request.url.path}",
        extra={
            "errors": exc.errors(),
            "body": exc.body,
            "path": request.url.path,
            "method": request.method,
        },
    )

    # Format validation errors
    errors = []
    for error in exc.errors():
        field_path = " -> ".join(str(loc) for loc in error["loc"])
        errors.append(
            {"field": field_path, "message": error["msg"], "type": error["type"]}
        )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": {"validation_errors": errors},
                "timestamp": datetime.utcnow().isoformat(),
                "path": request.url.path,
            }
        },
    )


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Handle HTTP exceptions"""
    # Log HTTP errors (excluding 4xx client errors)
    if exc.status_code >= 500:
        logger.error(
            f"HTTP Exception: {exc.status_code} - {exc.detail}",
            extra={
                "status_code": exc.status_code,
                "path": request.url.path,
                "method": request.method,
            },
        )
    elif exc.status_code >= 400:
        logger.warning(
            f"HTTP Exception: {exc.status_code} - {exc.detail}",
            extra={
                "status_code": exc.status_code,
                "path": request.url.path,
                "method": request.method,
            },
        )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": exc.detail or "An error occurred",
                "timestamp": datetime.utcnow().isoformat(),
                "path": request.url.path,
            }
        },
    )


async def database_exception_handler(
    request: Request, exc: SQLAlchemyError
) -> JSONResponse:
    """Handle database exceptions"""
    # Log the database error
    logger.error(
        f"Database Error: {str(exc)}",
        extra={
            "exception_type": type(exc).__name__,
            "path": request.url.path,
            "method": request.method,
            "traceback": traceback.format_exc(),
        },
    )

    # Don't expose internal database errors in production
    if settings.ENVIRONMENT == "production":
        message = "A database error occurred"
        details = {}
    else:
        message = f"Database error: {str(exc)}"
        details = {"exception_type": type(exc).__name__}

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "DATABASE_ERROR",
                "message": message,
                "details": details,
                "timestamp": datetime.utcnow().isoformat(),
                "path": request.url.path,
            }
        },
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all other uncaught exceptions"""
    # Log the unexpected error
    logger.critical(
        f"Uncaught Exception: {str(exc)}",
        extra={
            "exception_type": type(exc).__name__,
            "path": request.url.path,
            "method": request.method,
            "traceback": traceback.format_exc(),
        },
    )

    # Don't expose internal errors in production
    if settings.ENVIRONMENT == "production":
        message = "An unexpected error occurred"
        details = {}
    else:
        message = f"Unexpected error: {str(exc)}"
        details = {
            "exception_type": type(exc).__name__,
            "traceback": traceback.format_exc().split("\n")[
                -5:
            ],  # Last 5 lines of traceback
        }

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": message,
                "details": details,
                "timestamp": datetime.utcnow().isoformat(),
                "path": request.url.path,
            }
        },
    )


def register_error_handlers(app):
    """Register all error handlers with the FastAPI app"""
    # Custom exceptions
    app.add_exception_handler(YTEmpireException, ytempire_exception_handler)
    app.add_exception_handler(APIException, ytempire_exception_handler)
    app.add_exception_handler(ExternalServiceException, ytempire_exception_handler)
    app.add_exception_handler(DatabaseException, ytempire_exception_handler)

    # Framework exceptions
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)

    # Database exceptions
    app.add_exception_handler(SQLAlchemyError, database_exception_handler)

    # Catch-all for unexpected exceptions
    app.add_exception_handler(Exception, general_exception_handler)

    logger.info("Error handlers registered successfully")


class ErrorResponseMiddleware:
    """Middleware to ensure consistent error responses"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            # Log unhandled exceptions that somehow bypass handlers
            logger.error(
                f"Unhandled exception in middleware: {str(exc)}",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "traceback": traceback.format_exc(),
                },
            )

            # Return a generic error response
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": {
                        "code": "INTERNAL_SERVER_ERROR",
                        "message": "An unexpected error occurred",
                        "timestamp": datetime.utcnow().isoformat(),
                        "path": request.url.path,
                    }
                },
            )
