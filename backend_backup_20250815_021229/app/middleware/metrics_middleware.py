"""
Metrics Middleware for Request Tracking
Automatically tracks HTTP request metrics for Prometheus
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from backend.app.core.metrics import (
    http_requests_total,
    http_request_duration_seconds,
    http_request_size_bytes,
    http_response_size_bytes
)

logger = logging.getLogger(__name__)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to track HTTP request metrics"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.excluded_paths = {
            "/metrics",  # Don't track metrics endpoint itself
            "/health",
            "/api/v1/health",
            "/api/v1/health/ready",
            "/api/v1/health/live",
            "/docs",
            "/redoc",
            "/openapi.json"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and track metrics"""
        
        # Skip metrics for excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)
        
        # Skip metrics for static files
        if request.url.path.startswith("/static/"):
            return await call_next(request)
        
        # Start timing
        start_time = time.time()
        
        # Get request size
        request_size = 0
        if request.headers.get("content-length"):
            try:
                request_size = int(request.headers["content-length"])
            except (ValueError, TypeError):
                pass
        
        # Process request
        response = None
        status_code = 500
        
        try:
            response = await call_next(request)
            status_code = response.status_code
            
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
            
        finally:
            # Calculate duration
            duration = time.time() - start_time
            
            # Determine endpoint (group by route pattern)
            endpoint = self._get_endpoint_pattern(request)
            
            # Track request count
            http_requests_total.labels(
                method=request.method,
                endpoint=endpoint,
                status=self._get_status_group(status_code)
            ).inc()
            
            # Track request duration
            http_request_duration_seconds.labels(
                method=request.method,
                endpoint=endpoint
            ).observe(duration)
            
            # Track request size
            if request_size > 0:
                http_request_size_bytes.labels(
                    method=request.method,
                    endpoint=endpoint
                ).observe(request_size)
            
            # Track response size
            if response and response.headers.get("content-length"):
                try:
                    response_size = int(response.headers["content-length"])
                    http_response_size_bytes.labels(
                        method=request.method,
                        endpoint=endpoint
                    ).observe(response_size)
                except (ValueError, TypeError):
                    pass
            
            # Log slow requests
            if duration > 1.0:
                logger.warning(
                    f"Slow request: {request.method} {endpoint} "
                    f"took {duration:.2f}s (status: {status_code})"
                )
        
        return response
    
    def _get_endpoint_pattern(self, request: Request) -> str:
        """Get endpoint pattern for grouping metrics"""
        path = request.url.path
        
        # Common patterns to group
        patterns = [
            ("/api/v1/channels/", "/api/v1/channels/{id}"),
            ("/api/v1/videos/", "/api/v1/videos/{id}"),
            ("/api/v1/users/", "/api/v1/users/{id}"),
            ("/api/v1/analytics/", "/api/v1/analytics/{type}"),
            ("/api/v1/youtube/", "/api/v1/youtube/{action}"),
            ("/ws/", "/ws/{client_id}"),
        ]
        
        for prefix, pattern in patterns:
            if path.startswith(prefix) and path != prefix:
                return pattern
        
        return path
    
    def _get_status_group(self, status_code: int) -> str:
        """Group status codes for metrics"""
        if status_code < 200:
            return "1xx"
        elif status_code < 300:
            return "2xx"
        elif status_code < 400:
            return "3xx"
        elif status_code < 500:
            return "4xx"
        else:
            return "5xx"


class DatabaseMetricsMiddleware:
    """Middleware to track database query metrics"""
    
    def __init__(self):
        self.query_patterns = {
            "SELECT": "select",
            "INSERT": "insert",
            "UPDATE": "update",
            "DELETE": "delete",
            "BEGIN": "transaction",
            "COMMIT": "transaction",
            "ROLLBACK": "transaction"
        }
    
    async def __call__(self, execute, query, *args, **kwargs):
        """Track database query execution"""
        from backend.app.core.metrics import db_query_duration_seconds, db_query_errors_total
        
        start_time = time.time()
        query_type = self._get_query_type(str(query))
        table = self._extract_table_name(str(query))
        
        try:
            result = await execute(query, *args, **kwargs)
            return result
            
        except Exception as e:
            db_query_errors_total.labels(
                error_type=type(e).__name__
            ).inc()
            raise
            
        finally:
            duration = time.time() - start_time
            db_query_duration_seconds.labels(
                query_type=query_type,
                table=table
            ).observe(duration)
            
            # Log slow queries
            if duration > 0.1:  # 100ms
                logger.warning(
                    f"Slow query ({duration:.3f}s): {query_type} on {table}"
                )
    
    def _get_query_type(self, query: str) -> str:
        """Extract query type from SQL"""
        query_upper = query.upper().strip()
        
        for keyword, query_type in self.query_patterns.items():
            if query_upper.startswith(keyword):
                return query_type
        
        return "other"
    
    def _extract_table_name(self, query: str) -> str:
        """Extract table name from SQL query"""
        query_upper = query.upper()
        
        # Simple extraction for common patterns
        if "FROM" in query_upper:
            parts = query_upper.split("FROM")
            if len(parts) > 1:
                table_part = parts[1].strip().split()[0]
                return table_part.lower().strip('"').strip("'")
        
        elif "INTO" in query_upper:
            parts = query_upper.split("INTO")
            if len(parts) > 1:
                table_part = parts[1].strip().split()[0]
                return table_part.lower().strip('"').strip("'")
        
        elif "UPDATE" in query_upper:
            parts = query_upper.split("UPDATE")
            if len(parts) > 1:
                table_part = parts[1].strip().split()[0]
                return table_part.lower().strip('"').strip("'")
        
        return "unknown"


class CacheMetricsMiddleware:
    """Middleware to track cache operations"""
    
    def __init__(self, metrics_collector):
        self.metrics_collector = metrics_collector
    
    async def track_cache_operation(self, operation: str, key: str, cache_type: str = "redis"):
        """Track cache operation metrics"""
        if operation == "hit":
            self.metrics_collector.track_cache_hit(cache_type)
        elif operation == "miss":
            self.metrics_collector.track_cache_miss(cache_type)
    
    async def __call__(self, func, *args, **kwargs):
        """Wrap cache operations"""
        key = args[0] if args else kwargs.get("key", "unknown")
        
        result = await func(*args, **kwargs)
        
        if result is not None:
            await self.track_cache_operation("hit", key)
        else:
            await self.track_cache_operation("miss", key)
        
        return result