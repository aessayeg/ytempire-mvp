"""
YTEmpire MVP - Main FastAPI Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
import logging
from contextlib import asynccontextmanager

from app.core.config import settings
from app.api.v1.api import api_router
from app.core.logging import setup_logging
from app.db.session import engine
from app.db.base import Base
from app.services.websocket_manager import ConnectionManager
from app.core.performance_enhanced import (
    cache_manager, db_pool, http_pool, 
    FastPerformanceMiddleware, initialize_performance_systems,
    cleanup_performance_systems
)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize WebSocket manager
ws_manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle
    """
    # Startup
    logger.info("Starting YTEmpire API...")
    
    # Initialize enhanced performance components
    await initialize_performance_systems()
    logger.info("Enhanced performance optimization initialized")
    
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("Database initialized")
    logger.info(f"API running in {settings.ENVIRONMENT} mode")
    
    yield
    
    # Shutdown
    logger.info("Shutting down YTEmpire API...")
    await cleanup_performance_systems()
    await engine.dispose()


# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    description="YTEmpire - AI-Powered YouTube Content Automation Platform",
    lifespan=lifespan
)

# Security Headers Middleware - MUST BE FIRST
from app.middleware.security_headers import setup_security_headers
app = setup_security_headers(app)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

# Compression Middleware - Add before other middleware for best performance
from app.middleware.compression import CompressionMiddleware
app.add_middleware(CompressionMiddleware, minimum_size=500, compression_level=6)

# Enhanced performance middleware
app.add_middleware(FastPerformanceMiddleware, cache_manager=cache_manager)

# Enhanced Rate Limiting Middleware
from app.middleware.rate_limiting_enhanced import RateLimitMiddleware
app.add_middleware(RateLimitMiddleware)

# Input Validation and Sanitization Middleware
from app.middleware.input_validation import InputValidationMiddleware
app.add_middleware(InputValidationMiddleware)

# Global error handling middleware
from app.middleware.global_error_handler import GlobalErrorMiddleware, create_error_handlers
app.add_middleware(GlobalErrorMiddleware)

# Register error handlers
create_error_handlers(app)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Prometheus metrics
if settings.ENABLE_METRICS:
    Instrumentator().instrument(app).expose(app)

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint for monitoring
    """
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT
    }

# Performance metrics endpoint
@app.get("/metrics/performance", tags=["Metrics"])
async def performance_metrics():
    """
    Performance metrics endpoint for monitoring
    """
    # Get middleware metrics
    middleware = None
    for middleware_item in app.user_middleware:
        if isinstance(middleware_item.cls, type) and issubclass(middleware_item.cls, FastPerformanceMiddleware):
            middleware = middleware_item
            break
    
    metrics = {}
    if middleware and hasattr(middleware, 'get_metrics'):
        metrics = middleware.get_metrics()
    
    # Get cache metrics
    cache_stats = cache_manager.stats if cache_manager else {}
    
    return {
        "performance": metrics,
        "cache": cache_stats,
        "timestamp": str(datetime.utcnow())
    }

from datetime import datetime

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint
    """
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "description": "YTEmpire API - Automating YouTube Success",
        "documentation": "/docs"
    }

# WebSocket endpoints
from fastapi import WebSocket, WebSocketDisconnect
from typing import Optional
import json

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time updates
    """
    await ws_manager.connect(websocket, client_id)
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_json()
            # Process incoming message
            await ws_manager.handle_incoming_message(websocket, client_id, data)
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket, client_id)
        await ws_manager.broadcast({
            "type": "notification",
            "data": {
                "message": f"Client {client_id} disconnected"
            }
        })
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        await ws_manager.disconnect(websocket, client_id)

@app.websocket("/ws/video-updates/{channel_id}")
async def video_updates_websocket(websocket: WebSocket, channel_id: str):
    """
    WebSocket endpoint for video generation updates
    """
    room_id = f"channel:{channel_id}"
    await ws_manager.connect(websocket, channel_id, metadata={"channel_id": channel_id})
    await ws_manager.join_room(channel_id, room_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            await ws_manager.handle_incoming_message(websocket, channel_id, data)
    except WebSocketDisconnect:
        await ws_manager.leave_room(channel_id, room_id)
        await ws_manager.disconnect(websocket, channel_id)
    except Exception as e:
        logger.error(f"WebSocket error for channel {channel_id}: {e}")
        await ws_manager.leave_room(channel_id, room_id)
        await ws_manager.disconnect(websocket, channel_id)