"""
Main API Router - YTEmpire v1 API
Consolidates all endpoint routers
"""
from fastapi import APIRouter

from app.api.v1.endpoints import (
    auth, channels, youtube_accounts, script_generation,
    payment, dashboard, video_queue, webhooks, analytics
)
from app.api.v1 import cost_tracking

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    auth.router, 
    prefix="/auth", 
    tags=["authentication"]
)

api_router.include_router(
    channels.router,
    prefix="/channels",
    tags=["channels"]
)

api_router.include_router(
    cost_tracking.router,
    prefix="/costs",
    tags=["cost-tracking"]
)

api_router.include_router(
    youtube_accounts.router,
    prefix="/youtube",
    tags=["youtube-accounts"]
)

api_router.include_router(
    script_generation.router,
    prefix="/scripts",
    tags=["script-generation"]
)

api_router.include_router(
    payment.router,
    prefix="/payments",
    tags=["payments"]
)

api_router.include_router(
    dashboard.router,
    prefix="/dashboard",
    tags=["dashboard"]
)

api_router.include_router(
    video_queue.router,
    prefix="/queue",
    tags=["video-queue"]
)

api_router.include_router(
    webhooks.router,
    prefix="/webhooks",
    tags=["webhooks"]
)

api_router.include_router(
    analytics.router,
    prefix="/analytics",
    tags=["analytics"]
)

# Health check endpoint
@api_router.get("/health", tags=["system"])
async def health_check():
    """System health check endpoint"""
    return {
        "status": "healthy",
        "service": "ytempire-api",
        "version": "1.0.0"
    }

# API info endpoint
@api_router.get("/", tags=["system"])
async def api_info():
    """API information and documentation links"""
    return {
        "service": "YTEmpire API",
        "version": "1.0.0",
        "documentation": "/docs",
        "openapi": "/openapi.json",
        "endpoints": {
            "auth": "/api/v1/auth",
            "channels": "/api/v1/channels",
            "costs": "/api/v1/costs",
            "youtube": "/api/v1/youtube",
            "health": "/api/v1/health"
        }
    }