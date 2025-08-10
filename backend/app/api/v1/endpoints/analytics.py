"""
Analytics Endpoints
Owner: Analytics Engineer
"""

from fastapi import APIRouter, Depends, Query, HTTPException, status
from datetime import date, datetime, timedelta
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.api.v1.endpoints.auth import get_current_user
from app.models.user import User
from app.repositories.analytics_repository import AnalyticsRepository
from app.repositories.channel_repository import ChannelRepository

router = APIRouter()


def get_analytics_repo(db: AsyncSession = Depends(get_db)) -> AnalyticsRepository:
    """Get analytics repository instance."""
    return AnalyticsRepository(db)


def get_channel_repo(db: AsyncSession = Depends(get_db)) -> ChannelRepository:
    """Get channel repository instance."""
    return ChannelRepository(db)


@router.get("/dashboard")
async def get_dashboard_metrics(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    current_user: User = Depends(get_current_user),
    analytics_repo: AnalyticsRepository = Depends(get_analytics_repo)
):
    """Get comprehensive dashboard overview metrics for the current user."""
    metrics = await analytics_repo.get_user_dashboard_metrics(current_user.id, days)
    
    return {
        "user_id": current_user.id,
        "metrics": metrics,
        "generated_at": datetime.utcnow().isoformat()
    }


@router.get("/dashboard/comparison")
async def get_performance_comparison(
    period_days: int = Query(30, ge=7, le=365, description="Period length in days"),
    current_user: User = Depends(get_current_user),
    analytics_repo: AnalyticsRepository = Depends(get_analytics_repo)
):
    """Compare current period performance with previous period."""
    comparison = await analytics_repo.get_performance_comparison(current_user.id, period_days)
    
    return {
        "user_id": current_user.id,
        "comparison": comparison,
        "generated_at": datetime.utcnow().isoformat()
    }


@router.get("/channels/{channel_id}")
async def get_channel_analytics(
    channel_id: str,
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD)"),
    current_user: User = Depends(get_current_user),
    analytics_repo: AnalyticsRepository = Depends(get_analytics_repo),
    channel_repo: ChannelRepository = Depends(get_channel_repo)
):
    """Get detailed analytics for a specific channel."""
    # Verify channel ownership
    if not await channel_repo.check_ownership(channel_id, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Channel not found or access denied"
        )
    
    # Set default date range if not provided
    if not end_date:
        end_date = datetime.utcnow().date()
    if not start_date:
        start_date = end_date - timedelta(days=30)
    
    # Validate date range
    if start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Start date must be before end date"
        )
    
    if (end_date - start_date).days > 365:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Date range cannot exceed 365 days"
        )
    
    analytics = await analytics_repo.get_channel_analytics(channel_id, start_date, end_date)
    
    return {
        "channel_id": channel_id,
        "period": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "days": (end_date - start_date).days + 1
        },
        "analytics": analytics,
        "data_points": len(analytics)
    }


@router.get("/videos/{video_id}")
async def get_video_analytics(
    video_id: str,
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD)"),
    current_user: User = Depends(get_current_user),
    analytics_repo: AnalyticsRepository = Depends(get_analytics_repo)
):
    """Get detailed analytics for a specific video."""
    from app.repositories.video_repository import VideoRepository
    video_repo = VideoRepository(analytics_repo.db)
    
    # Verify video ownership
    if not await video_repo.check_ownership(video_id, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found or access denied"
        )
    
    # Set default date range if not provided
    if not end_date:
        end_date = datetime.utcnow().date()
    if not start_date:
        start_date = end_date - timedelta(days=30)
    
    analytics = await analytics_repo.get_video_analytics(video_id, start_date, end_date)
    
    return {
        "video_id": video_id,
        "period": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "days": (end_date - start_date).days + 1
        },
        "analytics": analytics,
        "data_points": len(analytics)
    }


@router.get("/costs/breakdown")
async def get_cost_breakdown(
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD)"),
    video_id: Optional[str] = Query(None, description="Filter by specific video"),
    current_user: User = Depends(get_current_user),
    analytics_repo: AnalyticsRepository = Depends(get_analytics_repo)
):
    """Get detailed cost breakdown by service type."""
    # Set default date range if not provided
    if not end_date:
        end_date = datetime.utcnow().date()
    if not start_date:
        start_date = end_date - timedelta(days=30)
    
    # If video_id provided, verify ownership
    if video_id:
        from app.repositories.video_repository import VideoRepository
        video_repo = VideoRepository(analytics_repo.db)
        
        if not await video_repo.check_ownership(video_id, current_user.id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found or access denied"
            )
    
    breakdown = await analytics_repo.get_cost_breakdown(
        current_user.id, start_date, end_date, video_id
    )
    
    return {
        "user_id": current_user.id,
        "cost_breakdown": breakdown,
        "generated_at": datetime.utcnow().isoformat()
    }


@router.get("/trending")
async def get_trending_content(
    limit: int = Query(10, ge=1, le=50, description="Number of trending items"),
    current_user: User = Depends(get_current_user),
    analytics_repo: AnalyticsRepository = Depends(get_analytics_repo)
):
    """Get user's trending content based on engagement metrics."""
    trending = await analytics_repo.get_trending_content(current_user.id, limit)
    
    return {
        "user_id": current_user.id,
        "trending_content": trending,
        "count": len(trending),
        "generated_at": datetime.utcnow().isoformat()
    }


@router.get("/summary/monthly")
async def get_monthly_summary(
    year: int = Query(..., ge=2020, le=2030, description="Year"),
    month: int = Query(..., ge=1, le=12, description="Month"),
    current_user: User = Depends(get_current_user),
    analytics_repo: AnalyticsRepository = Depends(get_analytics_repo)
):
    """Get monthly summary for specified month and year."""
    # Calculate date range for the month
    start_date = date(year, month, 1)
    if month == 12:
        end_date = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = date(year, month + 1, 1) - timedelta(days=1)
    
    # Get metrics for the month
    days_in_month = (end_date - start_date).days + 1
    metrics = await analytics_repo.get_user_dashboard_metrics(current_user.id, days_in_month)
    
    # Get cost breakdown for the month
    cost_breakdown = await analytics_repo.get_cost_breakdown(
        current_user.id, start_date, end_date
    )
    
    return {
        "user_id": current_user.id,
        "period": {
            "year": year,
            "month": month,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "days": days_in_month
        },
        "summary": {
            "metrics": metrics,
            "cost_breakdown": cost_breakdown
        },
        "generated_at": datetime.utcnow().isoformat()
    }


@router.get("/export/csv")
async def export_analytics_csv(
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD)"),
    include_costs: bool = Query(True, description="Include cost data"),
    current_user: User = Depends(get_current_user),
    analytics_repo: AnalyticsRepository = Depends(get_analytics_repo)
):
    """Export analytics data as CSV (returns data structure for CSV generation)."""
    # Set default date range
    if not end_date:
        end_date = datetime.utcnow().date()
    if not start_date:
        start_date = end_date - timedelta(days=30)
    
    # Get dashboard metrics
    days = (end_date - start_date).days + 1
    metrics = await analytics_repo.get_user_dashboard_metrics(current_user.id, days)
    
    export_data = {
        "user_id": current_user.id,
        "export_type": "csv",
        "period": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "days": days
        },
        "metrics": metrics
    }
    
    # Include cost breakdown if requested
    if include_costs:
        cost_breakdown = await analytics_repo.get_cost_breakdown(
            current_user.id, start_date, end_date
        )
        export_data["cost_breakdown"] = cost_breakdown
    
    # In a real implementation, this would generate and return actual CSV data
    # For now, return the data structure that would be used to generate CSV
    return {
        "message": "CSV export data prepared",
        "download_url": f"/api/v1/analytics/download/csv/{current_user.id}",
        "data": export_data,
        "generated_at": datetime.utcnow().isoformat()
    }


@router.get("/health")
async def analytics_health_check(
    current_user: User = Depends(get_current_user),
    analytics_repo: AnalyticsRepository = Depends(get_analytics_repo)
):
    """Health check endpoint for analytics service."""
    try:
        # Quick test query
        metrics = await analytics_repo.get_user_dashboard_metrics(current_user.id, 1)
        
        return {
            "status": "healthy",
            "analytics_service": "operational",
            "database_connection": "active",
            "user_data_available": len(metrics.get('channels_performance', [])) > 0,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analytics service unhealthy: {str(e)}"
        )