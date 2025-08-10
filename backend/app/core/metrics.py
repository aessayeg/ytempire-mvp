"""
Prometheus Metrics Configuration
Owner: Platform Ops Lead
"""

import time
from typing import Dict, Optional
from prometheus_client import Counter, Histogram, Gauge, Info, REGISTRY
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# HTTP Metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

http_requests_in_progress = Gauge(
    'http_requests_in_progress',
    'Current HTTP requests in progress'
)

# Application Metrics
active_users = Gauge(
    'ytempire_active_users_total',
    'Number of active users'
)

videos_generated_total = Counter(
    'ytempire_videos_generated_total',
    'Total videos generated',
    ['user_id', 'channel_id']
)

videos_uploaded_total = Counter(
    'ytempire_videos_uploaded_total',
    'Total videos uploaded to YouTube',
    ['user_id', 'channel_id']
)

api_costs_total = Counter(
    'ytempire_api_costs_total',
    'Total API costs incurred',
    ['service', 'user_id']
)

database_connections = Gauge(
    'ytempire_database_connections',
    'Current database connections'
)

# Authentication Metrics
auth_attempts_total = Counter(
    'ytempire_auth_attempts_total',
    'Total authentication attempts',
    ['status']  # success, failed
)

jwt_tokens_issued_total = Counter(
    'ytempire_jwt_tokens_issued_total',
    'Total JWT tokens issued',
    ['token_type']  # access, refresh
)

# Rate Limiting Metrics
rate_limit_hits_total = Counter(
    'ytempire_rate_limit_hits_total',
    'Total rate limit hits',
    ['client_ip']
)

# YouTube API Metrics
youtube_api_calls_total = Counter(
    'ytempire_youtube_api_calls_total',
    'Total YouTube API calls',
    ['operation', 'status']
)

youtube_quota_usage = Gauge(
    'ytempire_youtube_quota_usage',
    'Current YouTube API quota usage'
)

# Celery Metrics
celery_tasks_total = Counter(
    'ytempire_celery_tasks_total',
    'Total Celery tasks',
    ['task_name', 'status']
)

celery_task_duration_seconds = Histogram(
    'ytempire_celery_task_duration_seconds',
    'Celery task duration in seconds',
    ['task_name']
)

# Application Info
app_info = Info(
    'ytempire_app',
    'YTEmpire application information'
)

# Set application info
app_info.info({
    'version': '1.0.0',
    'environment': 'development',
    'build_date': time.strftime('%Y-%m-%d %H:%M:%S')
})


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to collect HTTP metrics for Prometheus."""

    async def dispatch(self, request: Request, call_next) -> Response:
        """Collect metrics for each HTTP request."""
        # Skip metrics collection for the metrics endpoint itself
        if request.url.path == "/metrics":
            return await call_next(request)

        # Extract method and endpoint
        method = request.method
        endpoint = request.url.path

        # Increment in-progress requests
        http_requests_in_progress.inc()

        # Start timer
        start_time = time.time()

        try:
            # Process request
            response = await call_next(request)
            
            # Record metrics
            status_code = str(response.status_code)
            
            # Increment total requests counter
            http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code
            ).inc()
            
            # Record request duration
            duration = time.time() - start_time
            http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            return response

        except Exception as e:
            # Record error
            http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status_code="500"
            ).inc()
            
            duration = time.time() - start_time
            http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            raise e

        finally:
            # Decrement in-progress requests
            http_requests_in_progress.dec()


class MetricsCollector:
    """Utility class for collecting application metrics."""

    @staticmethod
    def record_user_auth(success: bool):
        """Record user authentication attempt."""
        status = "success" if success else "failed"
        auth_attempts_total.labels(status=status).inc()

    @staticmethod
    def record_jwt_token_issued(token_type: str):
        """Record JWT token issuance."""
        jwt_tokens_issued_total.labels(token_type=token_type).inc()

    @staticmethod
    def record_rate_limit_hit(client_ip: str):
        """Record rate limit hit."""
        rate_limit_hits_total.labels(client_ip=client_ip).inc()

    @staticmethod
    def record_video_generated(user_id: str, channel_id: str):
        """Record video generation."""
        videos_generated_total.labels(
            user_id=user_id,
            channel_id=channel_id
        ).inc()

    @staticmethod
    def record_video_uploaded(user_id: str, channel_id: str):
        """Record video upload."""
        videos_uploaded_total.labels(
            user_id=user_id,
            channel_id=channel_id
        ).inc()

    @staticmethod
    def record_api_cost(service: str, user_id: str, cost: float):
        """Record API cost."""
        api_costs_total.labels(
            service=service,
            user_id=user_id
        ).inc(cost)

    @staticmethod
    def record_youtube_api_call(operation: str, success: bool):
        """Record YouTube API call."""
        status = "success" if success else "failed"
        youtube_api_calls_total.labels(
            operation=operation,
            status=status
        ).inc()

    @staticmethod
    def update_youtube_quota_usage(quota_used: int):
        """Update YouTube API quota usage."""
        youtube_quota_usage.set(quota_used)

    @staticmethod
    def record_celery_task(task_name: str, status: str, duration: Optional[float] = None):
        """Record Celery task execution."""
        celery_tasks_total.labels(
            task_name=task_name,
            status=status
        ).inc()
        
        if duration is not None:
            celery_task_duration_seconds.labels(task_name=task_name).observe(duration)

    @staticmethod
    def update_active_users(count: int):
        """Update active users count."""
        active_users.set(count)

    @staticmethod
    def update_database_connections(count: int):
        """Update database connections count."""
        database_connections.set(count)


# Export metrics collector instance
metrics = MetricsCollector()