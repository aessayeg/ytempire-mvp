"""
API Router Configuration
Owner: API Developer
"""

from fastapi import APIRouter
from app.api.v1.endpoints import auth, users, channels, videos, analytics, webhooks

api_router = APIRouter()

# Authentication endpoints
api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["authentication"]
)

# User management
api_router.include_router(
    users.router,
    prefix="/users",
    tags=["users"]
)

# Channel management
api_router.include_router(
    channels.router,
    prefix="/channels",
    tags=["channels"]
)

# Video generation and management
api_router.include_router(
    videos.router,
    prefix="/videos",
    tags=["videos"]
)

# Analytics and metrics
api_router.include_router(
    analytics.router,
    prefix="/analytics",
    tags=["analytics"]
)

# Webhook endpoints for N8N
api_router.include_router(
    webhooks.router,
    prefix="/webhooks",
    tags=["webhooks"]
)