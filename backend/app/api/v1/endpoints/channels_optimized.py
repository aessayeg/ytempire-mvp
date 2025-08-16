"""
Optimized Channel Endpoints - No N+1 Queries
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.user import User
from app.api.v1.endpoints.auth import get_current_user
from app.services.optimized_queries import optimized_queries
from app.core.cache_strategy import cache_strategy

router = APIRouter()


@router.get("/optimized/channels")
async def get_user_channels_optimized(
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    """Get user channels with videos - optimized without N+1"""
    # Use optimized query with eager loading
    channels = await optimized_queries.get_channels_with_videos(db, current_user.id)

    # Transform to response
    return {
        "channels": [
            {
                "id": channel.id,
                "name": channel.name,
                "youtube_channel_id": channel.youtube_channel_id,
                "video_count": len(channel.videos),
                "total_views": sum(v.views or 0 for v in channel.videos),
                "videos": [
                    {
                        "id": v.id,
                        "title": v.title,
                        "views": v.views,
                        "likes": v.likes,
                        "cost": sum(c.amount for c in v.costs) if v.costs else 0,
                    }
                    for v in channel.videos[:10]  # Latest 10 videos
                ],
            }
            for channel in channels
        ]
    }


@router.get("/optimized/dashboard")
async def get_dashboard_optimized(
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    """Get dashboard data - single optimized query"""
    # Check cache first
    cache_key = cache_strategy.get_cache_key("dashboard", f"user:{current_user.id}")
    cached = await cache_strategy.get(cache_key)
    if cached:
        return cached

    # Get optimized dashboard data
    data = await optimized_queries.get_dashboard_data(db, current_user.id)

    if not data:
        raise HTTPException(status_code=404, detail="User data not found")

    response = {
        "user_id": current_user.id,
        "channels": data["channel_count"],
        "videos": data["video_count"],
        "total_views": data["total_views"],
        "total_likes": data["total_likes"],
        "total_cost": data["total_cost"],
        "avg_cost_per_video": data["total_cost"] / max(data["video_count"], 1),
    }

    # Cache for 1 minute
    await cache_strategy.set(cache_key, response, 60)

    return response


@router.get("/optimized/channel/{channel_id}/performance")
async def get_channel_performance_optimized(
    channel_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get channel performance metrics - optimized single query"""
    metrics = await optimized_queries.get_channel_performance_metrics(db, channel_id)

    if not metrics:
        raise HTTPException(status_code=404, detail="Channel not found")

    # Calculate additional metrics
    metrics["engagement_rate"] = (
        (metrics["total_likes"] + metrics["total_comments"])
        / max(metrics["total_views"], 1)
        * 100
    )
    metrics["cost_per_view"] = metrics["total_cost"] / max(metrics["total_views"], 1)

    return metrics
