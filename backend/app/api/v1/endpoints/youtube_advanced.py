"""
Advanced YouTube Integration API Endpoints
Provides enhanced YouTube features: Analytics, Playlists, Comment Moderation, Thumbnail Management
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from datetime import datetime, date, timedelta
from enum import Enum

from app.db.session import get_db
from app.api.v1.endpoints.auth import get_current_verified_user
from app.models.user import User
from app.services.youtube_service import YouTubeService
from app.services.cost_tracking import cost_tracker

import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class CommentModerationAction(str, Enum):
    """Comment moderation actions"""

    APPROVE = "approve"
    REJECT = "reject"
    HOLD_FOR_REVIEW = "hold_for_review"


class PlaylistPrivacyStatus(str, Enum):
    """Playlist privacy options"""

    PRIVATE = "private"
    PUBLIC = "public"
    UNLISTED = "unlisted"


class AnalyticsRequest(BaseModel):
    """Request for YouTube Analytics data"""

    channel_id: str
    start_date: date
    end_date: date
    metrics: Optional[List[str]] = None


class PlaylistCreateRequest(BaseModel):
    """Request to create a playlist"""

    title: str = Field(..., min_length=1, max_length=150)
    description: str = Field("", max_length=5000)
    privacy_status: PlaylistPrivacyStatus = PlaylistPrivacyStatus.PRIVATE


class AddToPlaylistRequest(BaseModel):
    """Request to add video to playlist"""

    playlist_id: str
    video_id: str
    position: Optional[int] = None


class CommentModerationRequest(BaseModel):
    """Request to moderate comments"""

    comment_id: str
    action: CommentModerationAction


class ThumbnailSelectionRequest(BaseModel):
    """Request to select thumbnail"""

    video_id: str
    thumbnail_option: str = Field(
        "default", description="Thumbnail option: default, 1, 2, 3, or custom URL"
    )


@router.post("/analytics/query")
async def get_youtube_analytics(
    request: AnalyticsRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
) -> Dict[str, Any]:
    """
    Get YouTube Analytics data for a channel
    """
    try:
        youtube_service = YouTubeService()

        # Track cost
        background_tasks.add_task(
            cost_tracker.track_cost,
            user_id=str(current_user.id),
            service="youtube_analytics",
            operation="analytics_query",
            cost=0.01,  # Estimated cost for analytics API call
            metadata={"channel_id": request.channel_id},
        )

        analytics_data = await youtube_service.get_analytics(
            channel_id=request.channel_id,
            start_date=datetime.combine(request.start_date, datetime.min.time()),
            end_date=datetime.combine(request.end_date, datetime.max.time()),
            metrics=request.metrics,
        )

        return {
            "status": "success",
            "data": analytics_data,
            "quota_usage": youtube_service.get_quota_usage(),
        }

    except Exception as e:
        logger.error(f"Failed to get YouTube analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get analytics: {str(e)}",
        )


@router.get("/analytics/metrics/available")
async def get_available_metrics(
    current_user: User = Depends(get_current_verified_user),
) -> Dict[str, Any]:
    """
    Get list of available analytics metrics
    """
    return {
        "metrics": {
            "views": {
                "name": "Views",
                "description": "Number of times the video was viewed",
                "type": "integer",
            },
            "likes": {
                "name": "Likes",
                "description": "Number of likes the video received",
                "type": "integer",
            },
            "comments": {
                "name": "Comments",
                "description": "Number of comments on the video",
                "type": "integer",
            },
            "shares": {
                "name": "Shares",
                "description": "Number of times the video was shared",
                "type": "integer",
            },
            "estimatedMinutesWatched": {
                "name": "Estimated Minutes Watched",
                "description": "Total estimated minutes watched",
                "type": "float",
            },
            "averageViewDuration": {
                "name": "Average View Duration",
                "description": "Average time viewers spend watching",
                "type": "float",
            },
            "subscribersGained": {
                "name": "Subscribers Gained",
                "description": "New subscribers gained from this content",
                "type": "integer",
            },
            "subscribersLost": {
                "name": "Subscribers Lost",
                "description": "Subscribers lost from this content",
                "type": "integer",
            },
        }
    }


@router.post("/playlists")
async def create_playlist(
    request: PlaylistCreateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
) -> Dict[str, Any]:
    """
    Create a new YouTube playlist
    """
    try:
        youtube_service = YouTubeService()

        # Track cost
        background_tasks.add_task(
            cost_tracker.track_cost,
            user_id=str(current_user.id),
            service="youtube_api",
            operation="create_playlist",
            cost=0.005,  # Small cost for playlist creation
            metadata={"title": request.title},
        )

        playlist_data = await youtube_service.create_playlist(
            title=request.title,
            description=request.description,
            privacy_status=request.privacy_status.value,
        )

        return {
            "status": "success",
            "playlist": playlist_data,
            "quota_usage": youtube_service.get_quota_usage(),
        }

    except Exception as e:
        logger.error(f"Failed to create playlist: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create playlist: {str(e)}",
        )


@router.post("/playlists/add-video")
async def add_video_to_playlist(
    request: AddToPlaylistRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
) -> Dict[str, str]:
    """
    Add a video to a playlist
    """
    try:
        youtube_service = YouTubeService()

        # Track cost
        background_tasks.add_task(
            cost_tracker.track_cost,
            user_id=str(current_user.id),
            service="youtube_api",
            operation="add_to_playlist",
            cost=0.002,
            metadata={"playlist_id": request.playlist_id, "video_id": request.video_id},
        )

        success = await youtube_service.add_video_to_playlist(
            playlist_id=request.playlist_id,
            video_id=request.video_id,
            position=request.position,
        )

        if success:
            return {
                "status": "success",
                "message": "Video added to playlist successfully",
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to add video to playlist",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add video to playlist: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add video to playlist: {str(e)}",
        )


@router.get("/videos/{video_id}/comments")
async def get_video_comments(
    video_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    max_results: int = Query(100, ge=1, le=1000),
    order: str = Query("relevance", regex="^(relevance|time)$"),
) -> Dict[str, Any]:
    """
    Get comments for a specific video
    """
    try:
        youtube_service = YouTubeService()

        # Track cost
        background_tasks.add_task(
            cost_tracker.track_cost,
            user_id=str(current_user.id),
            service="youtube_api",
            operation="get_comments",
            cost=0.01,  # Cost scales with number of API calls needed
            metadata={"video_id": video_id, "max_results": max_results},
        )

        comments = await youtube_service.get_video_comments(
            video_id=video_id, max_results=max_results, order=order
        )

        return {
            "status": "success",
            "video_id": video_id,
            "comments": comments,
            "count": len(comments),
            "quota_usage": youtube_service.get_quota_usage(),
        }

    except Exception as e:
        logger.error(f"Failed to get video comments: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get comments: {str(e)}",
        )


@router.post("/comments/moderate")
async def moderate_comment(
    request: CommentModerationRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
) -> Dict[str, str]:
    """
    Moderate a comment (approve/reject/hold for review)
    """
    try:
        youtube_service = YouTubeService()

        # Track cost
        background_tasks.add_task(
            cost_tracker.track_cost,
            user_id=str(current_user.id),
            service="youtube_api",
            operation="moderate_comment",
            cost=0.005,
            metadata={"comment_id": request.comment_id, "action": request.action.value},
        )

        success = await youtube_service.moderate_comment(
            comment_id=request.comment_id, action=request.action.value
        )

        if success:
            return {
                "status": "success",
                "message": f"Comment {request.action.value}d successfully",
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to moderate comment",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to moderate comment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to moderate comment: {str(e)}",
        )


@router.get("/videos/{video_id}/thumbnails")
async def get_thumbnail_options(
    video_id: str, current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Get available thumbnail options for a video
    """
    try:
        youtube_service = YouTubeService()

        thumbnail_options = await youtube_service.get_video_thumbnail_options(video_id)

        return {
            "status": "success",
            "video_id": video_id,
            "thumbnail_options": thumbnail_options,
        }

    except Exception as e:
        logger.error(f"Failed to get thumbnail options: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get thumbnail options: {str(e)}",
        )


@router.post("/videos/thumbnails/select")
async def select_thumbnail(
    request: ThumbnailSelectionRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
) -> Dict[str, str]:
    """
    Select thumbnail for a video from available options
    """
    try:
        youtube_service = YouTubeService()

        # Track cost
        background_tasks.add_task(
            cost_tracker.track_cost,
            user_id=str(current_user.id),
            service="youtube_api",
            operation="select_thumbnail",
            cost=0.001,
            metadata={
                "video_id": request.video_id,
                "thumbnail_option": request.thumbnail_option,
            },
        )

        success = await youtube_service.set_video_thumbnail_from_options(
            video_id=request.video_id, thumbnail_option=request.thumbnail_option
        )

        if success:
            return {"status": "success", "message": "Thumbnail selection completed"}
        else:
            return {
                "status": "info",
                "message": "Thumbnail selection noted (custom upload may be required)",
            }

    except Exception as e:
        logger.error(f"Failed to select thumbnail: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to select thumbnail: {str(e)}",
        )


@router.get("/quota/status")
async def get_quota_status(
    current_user: User = Depends(get_current_verified_user),
) -> Dict[str, Any]:
    """
    Get current YouTube API quota usage status
    """
    try:
        youtube_service = YouTubeService()
        quota_usage = youtube_service.get_quota_usage()

        # Calculate health status
        percentage_used = quota_usage.get("percentage_used", 0)
        if percentage_used < 70:
            health_status = "healthy"
        elif percentage_used < 85:
            health_status = "warning"
        else:
            health_status = "critical"

        return {
            "status": "success",
            "quota_usage": quota_usage,
            "health_status": health_status,
            "recommendations": get_quota_recommendations(percentage_used),
        }

    except Exception as e:
        logger.error(f"Failed to get quota status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get quota status: {str(e)}",
        )


@router.get("/trending/analyze")
async def analyze_trending_content(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    region_code: str = Query("US", description="Country code for trending analysis"),
    category_id: Optional[str] = Query(None, description="Video category ID"),
    max_results: int = Query(50, ge=1, le=200),
) -> Dict[str, Any]:
    """
    Analyze trending videos for content insights
    """
    try:
        youtube_service = YouTubeService()

        # Track cost
        background_tasks.add_task(
            cost_tracker.track_cost,
            user_id=str(current_user.id),
            service="youtube_api",
            operation="analyze_trending",
            cost=0.02,
            metadata={
                "region_code": region_code,
                "category_id": category_id,
                "max_results": max_results,
            },
        )

        trending_videos = await youtube_service.get_trending_videos(
            region_code=region_code, category_id=category_id, max_results=max_results
        )

        # Analyze the trending data
        analysis = analyze_trending_data(trending_videos)

        return {
            "status": "success",
            "region": region_code,
            "category_id": category_id,
            "trending_videos": trending_videos,
            "analysis": analysis,
            "quota_usage": youtube_service.get_quota_usage(),
        }

    except Exception as e:
        logger.error(f"Failed to analyze trending content: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze trending content: {str(e)}",
        )


def get_quota_recommendations(percentage_used: float) -> List[str]:
    """Generate quota usage recommendations"""
    recommendations = []

    if percentage_used < 50:
        recommendations.append(
            "Quota usage is healthy. You can increase API operations if needed."
        )
    elif percentage_used < 70:
        recommendations.append(
            "Monitor quota usage closely. Consider optimizing API calls."
        )
    elif percentage_used < 85:
        recommendations.append(
            "Quota usage is high. Implement caching to reduce API calls."
        )
        recommendations.append("Consider spreading operations throughout the day.")
    else:
        recommendations.append(
            "Quota usage is critical! Reduce non-essential API calls."
        )
        recommendations.append("Implement aggressive caching and batch operations.")
        recommendations.append("Consider using multiple API keys or accounts.")

    return recommendations


def analyze_trending_data(trending_videos: List[Dict]) -> Dict[str, Any]:
    """Analyze trending videos for insights"""
    if not trending_videos:
        return {"error": "No trending videos to analyze"}

    # Calculate metrics
    total_views = sum(video.get("view_count", 0) for video in trending_videos)
    total_likes = sum(video.get("like_count", 0) for video in trending_videos)

    avg_views = total_views / len(trending_videos) if trending_videos else 0
    avg_likes = total_likes / len(trending_videos) if trending_videos else 0

    # Analyze titles for common patterns
    all_titles = [video.get("title", "") for video in trending_videos]
    common_words = extract_common_words(all_titles)

    # Analyze channels
    channels = {}
    for video in trending_videos:
        channel = video.get("channel_title", "Unknown")
        if channel not in channels:
            channels[channel] = 0
        channels[channel] += 1

    top_channels = sorted(channels.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "total_videos": len(trending_videos),
        "avg_views": int(avg_views),
        "avg_likes": int(avg_likes),
        "engagement_rate": (avg_likes / avg_views * 100) if avg_views > 0 else 0,
        "common_title_words": common_words[:20],
        "top_channels": top_channels,
        "insights": generate_content_insights(trending_videos),
    }


def extract_common_words(titles: List[str]) -> List[str]:
    """Extract common words from video titles"""
    import re
    from collections import Counter

    # Combine all titles and extract words
    all_text = " ".join(titles).lower()
    words = re.findall(r"\b[a-z]{3,}\b", all_text)  # Words 3+ chars

    # Filter out common stop words
    stop_words = {
        "the",
        "and",
        "you",
        "for",
        "are",
        "with",
        "this",
        "that",
        "how",
        "not",
        "but",
    }
    words = [word for word in words if word not in stop_words]

    # Count frequency and return most common
    word_counts = Counter(words)
    return [word for word, count in word_counts.most_common(20)]


def generate_content_insights(trending_videos: List[Dict]) -> List[str]:
    """Generate actionable content insights"""
    insights = []

    if not trending_videos:
        return ["No trending videos available for analysis"]

    # View count analysis
    view_counts = [v.get("view_count", 0) for v in trending_videos]
    min_views = min(view_counts)
    max_views = max(view_counts)

    insights.append(f"Trending videos range from {min_views:,} to {max_views:,} views")

    # Category analysis
    categories = {}
    for video in trending_videos:
        cat_id = video.get("category_id", "Unknown")
        categories[cat_id] = categories.get(cat_id, 0) + 1

    if categories:
        top_category = max(categories, key=categories.get)
        insights.append(
            f"Most popular category: {top_category} ({categories[top_category]} videos)"
        )

    # Upload timing (would need more data in practice)
    insights.append("Consider uploading during peak engagement hours (7-9 PM)")

    # Engagement insights
    avg_engagement = sum(
        v.get("like_count", 0) / max(v.get("view_count", 1), 1) for v in trending_videos
    ) / len(trending_videos)
    insights.append(f"Average engagement rate: {avg_engagement:.1%}")

    return insights
