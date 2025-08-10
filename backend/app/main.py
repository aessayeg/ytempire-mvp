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
from app.services.websocket_manager import WebSocketManager
from app.core.performance import (
    cache_manager, connection_pool, PerformanceMiddleware
)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize WebSocket manager
ws_manager = WebSocketManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle
    """
    # Startup
    logger.info("Starting YTEmpire API...")
    
    # Initialize performance components
    await cache_manager.initialize()
    await connection_pool.initialize()
    logger.info("Performance optimization initialized")
    
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("Database initialized")
    logger.info(f"API running in {settings.ENVIRONMENT} mode")
    
    yield
    
    # Shutdown
    logger.info("Shutting down YTEmpire API...")
    await connection_pool.cleanup()
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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

# Performance middleware
app.add_middleware(PerformanceMiddleware, cache_manager=cache_manager)

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

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time updates
    """
    await ws_manager.connect(websocket, client_id)
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            # Echo back to sender
            await ws_manager.send_personal_message(f"Echo: {data}", client_id)
    except WebSocketDisconnect:
        ws_manager.disconnect(client_id)
        await ws_manager.broadcast(f"Client {client_id} left")

@app.websocket("/ws/video-updates/{channel_id}")
async def video_updates_websocket(websocket: WebSocket, channel_id: str):
    """
    WebSocket endpoint for video generation updates
    """
    await ws_manager.connect(websocket, f"channel_{channel_id}")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(f"channel_{channel_id}")