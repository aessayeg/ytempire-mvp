"""
Distributed tracing configuration for YTEmpire backend.
Implements OpenTelemetry with Jaeger integration.
"""

import os
import logging
from typing import Optional, Dict, Any
from functools import wraps
import time

from opentelemetry import trace, metrics, baggage
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.celery import CeleryInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.system_metrics import SystemMetricsInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.b3 import B3MultiFormat
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.sdk.trace import TracerProvider, Status, StatusCode
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased, ParentBased
from opentelemetry.semconv.trace import SpanAttributes
from opentelemetry.trace import Span, Tracer
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest
from fastapi import Request
from fastapi.responses import Response

logger = logging.getLogger(__name__)

# Configuration
JAEGER_ENDPOINT = os.getenv("JAEGER_ENDPOINT", "http://localhost:14268/api/traces")
OTLP_ENDPOINT = os.getenv("OTLP_ENDPOINT", "localhost:4317")
SERVICE_NAME_ENV = os.getenv("SERVICE_NAME", "ytempire-backend")
SERVICE_VERSION_ENV = os.getenv("SERVICE_VERSION", "1.0.0")
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
TRACE_SAMPLING_RATE = float(os.getenv("TRACE_SAMPLING_RATE", "0.1"))
ENABLE_TRACING = os.getenv("ENABLE_TRACING", "true").lower() == "true"
ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"

# Metrics
request_counter = Counter(
    "ytempire_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

request_duration = Histogram(
    "ytempire_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)

active_requests = Gauge(
    "ytempire_http_requests_active", "Active HTTP requests", ["method", "endpoint"]
)

database_queries = Counter(
    "ytempire_database_queries_total", "Total database queries", ["operation", "table"]
)

cache_operations = Counter(
    "ytempire_cache_operations_total", "Total cache operations", ["operation", "status"]
)

ai_service_calls = Counter(
    "ytempire_ai_service_calls_total",
    "Total AI service API calls",
    ["service", "model", "status"],
)

video_generation_duration = Histogram(
    "ytempire_video_generation_duration_seconds",
    "Video generation duration in seconds",
    ["stage"],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600),
)

service_info = Info("ytempire_service", "Service information")
service_info.info(
    {
        "version": SERVICE_VERSION_ENV,
        "environment": ENVIRONMENT,
        "service": SERVICE_NAME_ENV,
    }
)


class TracingManager:
    """Manages distributed tracing configuration and initialization."""

    def __init__(self):
        self.tracer_provider: Optional[TracerProvider] = None
        self.meter_provider: Optional[MeterProvider] = None
        self.tracer: Optional[Tracer] = None
        self.meter = None
        self.is_initialized = False

    def initialize(self, app=None):
        """Initialize tracing and metrics collection."""
        if self.is_initialized:
            logger.info("Tracing already initialized")
            return

        try:
            # Create resource
            resource = Resource.create(
                {
                    SERVICE_NAME: SERVICE_NAME_ENV,
                    SERVICE_VERSION: SERVICE_VERSION_ENV,
                    "environment": ENVIRONMENT,
                    "deployment.environment": ENVIRONMENT,
                    "service.namespace": "ytempire",
                    "telemetry.sdk.language": "python",
                    "telemetry.sdk.name": "opentelemetry",
                }
            )

            if ENABLE_TRACING:
                self._setup_tracing(resource)

            if ENABLE_METRICS:
                self._setup_metrics(resource)

            # Instrument libraries
            self._instrument_libraries()

            # Set propagator for distributed context
            set_global_textmap(B3MultiFormat())

            self.is_initialized = True
            logger.info("Tracing and metrics initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize tracing: {e}")

    def _setup_tracing(self, resource: Resource):
        """Setup distributed tracing with Jaeger and OTLP."""
        # Create sampler
        sampler = ParentBased(root=TraceIdRatioBased(TRACE_SAMPLING_RATE))

        # Create tracer provider
        self.tracer_provider = TracerProvider(resource=resource, sampler=sampler)

        # Add Jaeger exporter
        jaeger_exporter = JaegerExporter(
            collector_endpoint=JAEGER_ENDPOINT,
            max_tag_value_length=2048,
        )
        self.tracer_provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))

        # Add OTLP exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint=OTLP_ENDPOINT,
            insecure=True,
        )
        self.tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

        # Add console exporter for debugging
        if ENVIRONMENT == "development":
            console_exporter = ConsoleSpanExporter()
            self.tracer_provider.add_span_processor(
                BatchSpanProcessor(console_exporter)
            )

        # Set global tracer provider
        trace.set_tracer_provider(self.tracer_provider)
        self.tracer = trace.get_tracer(__name__)

    def _setup_metrics(self, resource: Resource):
        """Setup metrics collection with Prometheus and OTLP."""
        # Create meter provider
        prometheus_reader = PrometheusMetricReader()
        otlp_metric_exporter = OTLPMetricExporter(
            endpoint=OTLP_ENDPOINT,
            insecure=True,
        )

        self.meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[prometheus_reader],
            metric_exporters=[otlp_metric_exporter],
        )

        # Set global meter provider
        metrics.set_meter_provider(self.meter_provider)
        self.meter = metrics.get_meter(__name__)

    def _instrument_libraries(self):
        """Instrument various libraries for automatic tracing."""
        # FastAPI
        FastAPIInstrumentor.instrument(
            tracer_provider=self.tracer_provider,
            excluded_urls="health,metrics,docs,openapi.json",
        )

        # HTTP requests
        RequestsInstrumentor().instrument(tracer_provider=self.tracer_provider)

        # SQLAlchemy
        SQLAlchemyInstrumentor().instrument(
            tracer_provider=self.tracer_provider,
            enable_commenter=True,
        )

        # PostgreSQL
        Psycopg2Instrumentor().instrument(tracer_provider=self.tracer_provider)

        # Redis
        RedisInstrumentor().instrument(tracer_provider=self.tracer_provider)

        # Celery
        CeleryInstrumentor().instrument(tracer_provider=self.tracer_provider)

        # Logging
        LoggingInstrumentor().instrument(
            tracer_provider=self.tracer_provider, set_logging_format=True
        )

        # System metrics
        SystemMetricsInstrumentor().instrument()

    def shutdown(self):
        """Shutdown tracing and metrics providers."""
        if self.tracer_provider:
            self.tracer_provider.shutdown()
        if self.meter_provider:
            self.meter_provider.shutdown()
        self.is_initialized = False
        logger.info("Tracing and metrics shutdown complete")


# Global tracing manager instance
tracing_manager = TracingManager()


def trace_span(
    name: str, kind=trace.SpanKind.INTERNAL, attributes: Dict[str, Any] = None
):
    """Decorator to create a traced span for a function."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(
                name, kind=kind, attributes=attributes or {}
            ) as span:
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(
                name, kind=kind, attributes=attributes or {}
            ) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


class TracingMiddleware:
    """Custom middleware for enhanced request tracing."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, request: Request, call_next):
        # Start timing
        start_time = time.time()

        # Get or create trace context
        tracer = trace.get_tracer(__name__)

        # Extract trace context from headers
        context = {}
        if "traceparent" in request.headers:
            context["traceparent"] = request.headers["traceparent"]
        if "tracestate" in request.headers:
            context["tracestate"] = request.headers["tracestate"]

        # Create span
        with tracer.start_as_current_span(
            f"{request.method} {request.url.path}",
            kind=trace.SpanKind.SERVER,
            attributes={
                SpanAttributes.HTTP_METHOD: request.method,
                SpanAttributes.HTTP_URL: str(request.url),
                SpanAttributes.HTTP_TARGET: request.url.path,
                SpanAttributes.HTTP_HOST: request.url.hostname,
                SpanAttributes.HTTP_SCHEME: request.url.scheme,
                SpanAttributes.HTTP_USER_AGENT: request.headers.get("user-agent", ""),
                SpanAttributes.NET_HOST_NAME: request.url.hostname,
                SpanAttributes.NET_HOST_PORT: request.url.port or 80,
                "client.address": request.client.host if request.client else "unknown",
            },
        ) as span:
            # Add baggage
            baggage.set_baggage(
                "user.id", request.headers.get("x-user-id", "anonymous")
            )
            baggage.set_baggage("request.id", request.headers.get("x-request-id", ""))

            # Track active requests
            active_requests.labels(
                method=request.method, endpoint=request.url.path
            ).inc()

            try:
                # Process request
                response = await call_next(request)

                # Add response attributes
                span.set_attribute(
                    SpanAttributes.HTTP_STATUS_CODE, response.status_code
                )

                # Set span status based on HTTP status
                if response.status_code >= 400:
                    span.set_status(Status(StatusCode.ERROR))
                else:
                    span.set_status(Status(StatusCode.OK))

                # Record metrics
                duration = time.time() - start_time
                request_counter.labels(
                    method=request.method,
                    endpoint=request.url.path,
                    status_code=response.status_code,
                ).inc()
                request_duration.labels(
                    method=request.method, endpoint=request.url.path
                ).observe(duration)

                # Add trace ID to response headers
                trace_id = format(span.get_span_context().trace_id, "032x")
                response.headers["X-Trace-Id"] = trace_id

                return response

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

            finally:
                active_requests.labels(
                    method=request.method, endpoint=request.url.path
                ).dec()


def record_ai_service_call(service: str, model: str, success: bool = True):
    """Record an AI service API call metric."""
    ai_service_calls.labels(
        service=service, model=model, status="success" if success else "failure"
    ).inc()


def record_database_query(operation: str, table: str):
    """Record a database query metric."""
    database_queries.labels(operation=operation, table=table).inc()


def record_cache_operation(operation: str, hit: bool = True):
    """Record a cache operation metric."""
    cache_operations.labels(operation=operation, status="hit" if hit else "miss").inc()


def record_video_generation_stage(stage: str, duration: float):
    """Record video generation stage duration."""
    video_generation_duration.labels(stage=stage).observe(duration)


async def get_metrics():
    """Generate Prometheus metrics."""
    return Response(content=generate_latest(), media_type="text/plain")


# Custom span attributes
class CustomAttributes:
    """Custom span attributes for YTEmpire."""

    USER_ID = "ytempire.user.id"
    CHANNEL_ID = "ytempire.channel.id"
    VIDEO_ID = "ytempire.video.id"
    PIPELINE_STAGE = "ytempire.pipeline.stage"
    AI_MODEL = "ytempire.ai.model"
    AI_SERVICE = "ytempire.ai.service"
    COST_USD = "ytempire.cost.usd"
    CACHE_HIT = "ytempire.cache.hit"
    QUEUE_NAME = "ytempire.queue.name"
    JOB_ID = "ytempire.job.id"
