"""
Main API Router - YTEmpire v1 API
Consolidates all endpoint routers
"""
from fastapi import APIRouter

from app.api.v1.endpoints import (
    auth, channels, youtube_accounts, script_generation,
    payment, dashboard, video_queue, webhooks, analytics, notifications, batch, api_optimization, data_quality,
    videos, users, test_generation, video_generation, cost_optimization, beta_users, revenue, collaboration, video_processing, advanced_analytics, business_intelligence, system_monitoring,
    behavior_analytics, channels_optimized, experiments, gpu_resources, quality_dashboard, youtube_oauth, youtube_advanced, content_library, cost_intelligence, ai_multi_provider
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

api_router.include_router(
    notifications.router,
    prefix="/notifications",
    tags=["notifications"]
)

api_router.include_router(
    batch.router,
    prefix="/batch",
    tags=["batch-processing"]
)

api_router.include_router(
    api_optimization.router,
    prefix="/api-optimization",
    tags=["api-optimization"]
)

api_router.include_router(
    data_quality.router,
    prefix="/data-quality",
    tags=["data-quality"]
)

api_router.include_router(
    videos.router,
    prefix="/videos",
    tags=["videos"]
)

api_router.include_router(
    users.router,
    prefix="/users",
    tags=["users"]
)

api_router.include_router(
    test_generation.router,
    prefix="/test",
    tags=["test-generation"]
)

api_router.include_router(
    video_generation.router,
    prefix="/video-generation",
    tags=["video-generation"]
)

api_router.include_router(
    cost_optimization.router,
    prefix="/cost-optimization",
    tags=["cost-optimization"]
)

api_router.include_router(
    beta_users.router,
    prefix="/beta",
    tags=["beta-users"]
)

api_router.include_router(
    revenue.router,
    prefix="/revenue",
    tags=["revenue-tracking"]
)

api_router.include_router(
    collaboration.router,
    prefix="/collaboration",
    tags=["real-time-collaboration"]
)

api_router.include_router(
    video_processing.router,
    prefix="/video-processing",
    tags=["advanced-video-processing"]
)

api_router.include_router(
    advanced_analytics.router,
    prefix="/advanced-analytics",
    tags=["advanced-analytics"]
)

api_router.include_router(
    business_intelligence.router,
    prefix="/bi",
    tags=["business-intelligence"]
)

api_router.include_router(
    system_monitoring.router,
    prefix="/system",
    tags=["system-monitoring"]
)

# Missing endpoint routers - now registered
api_router.include_router(
    behavior_analytics.router,
    prefix="/behavior-analytics",
    tags=["behavior-analytics"]
)

api_router.include_router(
    channels_optimized.router,
    prefix="/channels-optimized",
    tags=["channels-optimization"]
)

api_router.include_router(
    experiments.router,
    prefix="/experiments",
    tags=["experiments"]
)

api_router.include_router(
    gpu_resources.router,
    prefix="/gpu-resources",
    tags=["gpu-resources"]
)

api_router.include_router(
    quality_dashboard.router,
    prefix="/quality",
    tags=["quality-dashboard"]
)

api_router.include_router(
    youtube_oauth.router,
    prefix="/youtube-oauth",
    tags=["youtube-oauth"]
)

api_router.include_router(
    youtube_advanced.router,
    prefix="/youtube-advanced",
    tags=["youtube-advanced"]
)

api_router.include_router(
    content_library.router,
    prefix="/content-library",
    tags=["content-library"]
)

api_router.include_router(
    cost_intelligence.router,
    prefix="/cost-intelligence",
    tags=["cost-intelligence"]
)

api_router.include_router(
    ai_multi_provider.router,
    prefix="/ai-providers",
    tags=["ai-multi-provider"]
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