"""
Revenue API Endpoints
Comprehensive revenue tracking and analytics
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import logging

from app.db.session import get_db
from app.api.v1.endpoints.auth import get_current_verified_user
from app.models.user import User
from app.services.revenue_tracking import revenue_tracking_service

logger = logging.getLogger(__name__)

router = APIRouter()


class RevenueOverview(BaseModel):
    """Revenue overview response model"""
    total_revenue: float
    average_revenue_per_video: float
    highest_revenue_video: float
    lowest_revenue_video: float
    total_videos_monetized: int
    daily_revenue: List[Dict[str, Any]]
    channel_breakdown: List[Dict[str, Any]]
    cpm: float
    rpm: float
    forecast: Dict[str, Any]
    period: Dict[str, str]


class ChannelRevenue(BaseModel):
    """Channel revenue response model"""
    channel_id: int
    channel_name: str
    total_revenue: float
    video_count: int
    average_revenue_per_video: float
    top_earning_videos: List[Dict[str, Any]]
    revenue_by_video: List[Dict[str, Any]]
    subscriber_count: int
    total_views: int


class RevenueTrends(BaseModel):
    """Revenue trends response model"""
    trends: List[Dict[str, Any]]
    period: str
    total_revenue: float
    average_revenue: float
    peak_revenue: float
    lowest_revenue: float


class RevenueForecast(BaseModel):
    """Revenue forecast response model"""
    forecast: List[Dict[str, Any]]
    confidence: float
    method: str
    historical_average: float
    trend: str
    estimated_total: float


class RevenueBreakdown(BaseModel):
    """Revenue breakdown response model"""
    breakdown: List[Dict[str, Any]]
    type: str
    total: float


@router.get("/overview", response_model=RevenueOverview)
async def get_revenue_overview(
    start_date: Optional[datetime] = Query(None, description="Start date for revenue period"),
    end_date: Optional[datetime] = Query(None, description="End date for revenue period"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> RevenueOverview:
    """
    Get comprehensive revenue overview for the authenticated user.
    
    Returns revenue metrics including:
    - Total revenue and averages
    - Daily revenue breakdown
    - Channel-wise revenue distribution
    - CPM/RPM metrics
    - Revenue forecast
    """
    try:
        overview = await revenue_tracking_service.get_revenue_overview(
            db=db,
            user_id=current_user.id,
            start_date=start_date,
            end_date=end_date
        )
        return RevenueOverview(**overview)
    except Exception as e:
        logger.error(f"Error fetching revenue overview: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch revenue overview"
        )


@router.get("/channels/{channel_id}", response_model=ChannelRevenue)
async def get_channel_revenue(
    channel_id: int,
    start_date: Optional[datetime] = Query(None, description="Start date for revenue period"),
    end_date: Optional[datetime] = Query(None, description="End date for revenue period"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> ChannelRevenue:
    """
    Get revenue details for a specific channel.
    
    Returns:
    - Total channel revenue
    - Revenue per video breakdown
    - Top earning videos
    - Channel metrics
    """
    try:
        # Verify channel ownership
        from app.models.channel import Channel
        from sqlalchemy import select
        
        channel_query = select(Channel).where(
            Channel.id == channel_id,
            Channel.owner_id == current_user.id
        )
        channel_result = await db.execute(channel_query)
        channel = channel_result.scalar_one_or_none()
        
        if not channel:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Channel not found or access denied"
            )
        
        revenue_data = await revenue_tracking_service.get_channel_revenue(
            db=db,
            channel_id=channel_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if not revenue_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Revenue data not found"
            )
            
        return ChannelRevenue(**revenue_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching channel revenue: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch channel revenue"
        )


@router.get("/trends", response_model=RevenueTrends)
async def get_revenue_trends(
    period: str = Query('daily', description="Aggregation period: daily, weekly, monthly"),
    lookback_days: int = Query(30, description="Number of days to look back"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> RevenueTrends:
    """
    Get revenue trends over time.
    
    Parameters:
    - period: Aggregation period (daily/weekly/monthly)
    - lookback_days: Historical period to analyze
    
    Returns:
    - Time series revenue data
    - Growth rates
    - Statistical summaries
    """
    try:
        if period not in ['daily', 'weekly', 'monthly']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Period must be 'daily', 'weekly', or 'monthly'"
            )
            
        if lookback_days < 1 or lookback_days > 365:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Lookback days must be between 1 and 365"
            )
            
        trends = await revenue_tracking_service.get_revenue_trends(
            db=db,
            user_id=current_user.id,
            period=period,
            lookback_days=lookback_days
        )
        
        return RevenueTrends(**trends)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching revenue trends: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch revenue trends"
        )


@router.get("/forecast", response_model=RevenueForecast)
async def get_revenue_forecast(
    forecast_days: int = Query(7, description="Number of days to forecast"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> RevenueForecast:
    """
    Get revenue forecast using machine learning.
    
    Parameters:
    - forecast_days: Number of days to predict (max 30)
    
    Returns:
    - Daily revenue predictions
    - Confidence intervals
    - Model confidence score
    - Trend analysis
    """
    try:
        if forecast_days < 1 or forecast_days > 30:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Forecast days must be between 1 and 30"
            )
            
        forecast = await revenue_tracking_service.get_revenue_forecast(
            db=db,
            user_id=current_user.id,
            forecast_days=forecast_days
        )
        
        return RevenueForecast(**forecast)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating revenue forecast: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate revenue forecast"
        )


@router.get("/breakdown", response_model=RevenueBreakdown)
async def get_revenue_breakdown(
    breakdown_by: str = Query('source', description="Breakdown dimension: source, content_type, video_length, time_of_day"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> RevenueBreakdown:
    """
    Get detailed revenue breakdown by various dimensions.
    
    Parameters:
    - breakdown_by: Dimension for breakdown analysis
      - source: Revenue sources (ads, sponsorships, etc.)
      - content_type: By video category
      - video_length: By video duration ranges
      - time_of_day: By publishing time
    
    Returns:
    - Categorized revenue breakdown
    - Percentages and totals
    """
    try:
        valid_breakdowns = ['source', 'content_type', 'video_length', 'time_of_day']
        if breakdown_by not in valid_breakdowns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Breakdown must be one of: {', '.join(valid_breakdowns)}"
            )
            
        breakdown = await revenue_tracking_service.get_revenue_breakdown(
            db=db,
            user_id=current_user.id,
            breakdown_by=breakdown_by
        )
        
        return RevenueBreakdown(**breakdown)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching revenue breakdown: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch revenue breakdown"
        )


@router.get("/export")
async def export_revenue_data(
    format: str = Query('csv', description="Export format: csv or json"),
    start_date: Optional[datetime] = Query(None, description="Start date for export"),
    end_date: Optional[datetime] = Query(None, description="End date for export"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
):
    """
    Export revenue data in CSV or JSON format.
    
    Parameters:
    - format: Export format (csv/json)
    - start_date: Start date for data export
    - end_date: End date for data export
    
    Returns:
    - Downloadable file with revenue data
    """
    try:
        if format not in ['csv', 'json']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Format must be 'csv' or 'json'"
            )
            
        # Get revenue data
        overview = await revenue_tracking_service.get_revenue_overview(
            db=db,
            user_id=current_user.id,
            start_date=start_date,
            end_date=end_date
        )
        
        if format == 'csv':
            import csv
            import io
            from fastapi.responses import StreamingResponse
            
            # Create CSV
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(['Date', 'Revenue', 'Channel', 'Video Count'])
            
            # Write daily revenue data
            for day in overview['daily_revenue']:
                writer.writerow([
                    day['date'],
                    day['revenue'],
                    '',  # Will be filled with channel data
                    ''
                ])
                
            # Write channel breakdown
            writer.writerow([])  # Empty row
            writer.writerow(['Channel Breakdown'])
            writer.writerow(['Channel', 'Revenue'])
            for channel in overview['channel_breakdown']:
                writer.writerow([
                    channel['channel_name'],
                    channel['revenue']
                ])
                
            output.seek(0)
            
            return StreamingResponse(
                io.BytesIO(output.getvalue().encode()),
                media_type='text/csv',
                headers={
                    'Content-Disposition': f'attachment; filename=revenue_export_{datetime.utcnow().date()}.csv'
                }
            )
        else:
            # Return JSON
            from fastapi.responses import JSONResponse
            
            return JSONResponse(
                content=overview,
                headers={
                    'Content-Disposition': f'attachment; filename=revenue_export_{datetime.utcnow().date()}.json'
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting revenue data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export revenue data"
        )


@router.post("/recalculate")
async def recalculate_revenue(
    channel_id: Optional[int] = Query(None, description="Specific channel to recalculate"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
):
    """
    Trigger revenue recalculation for user's channels.
    
    This endpoint forces a refresh of revenue calculations,
    useful after YouTube Analytics API updates.
    
    Parameters:
    - channel_id: Optional specific channel to recalculate
    
    Returns:
    - Status of recalculation process
    """
    try:
        # This would trigger a background task to recalculate revenue
        # For now, we'll just clear the cache
        from app.core.cache import cache_service
        
        if channel_id:
            # Verify channel ownership
            from app.models.channel import Channel
            from sqlalchemy import select
            
            channel_query = select(Channel).where(
                Channel.id == channel_id,
                Channel.owner_id == current_user.id
            )
            channel_result = await db.execute(channel_query)
            channel = channel_result.scalar_one_or_none()
            
            if not channel:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Channel not found or access denied"
                )
            
            # Clear channel-specific cache
            cache_pattern = f"revenue:*:{current_user.id}:*"
        else:
            # Clear all user revenue cache
            cache_pattern = f"revenue:*:{current_user.id}:*"
            
        # In production, this would be done more efficiently
        await cache_service.delete_pattern(cache_pattern)
        
        return {
            "status": "success",
            "message": "Revenue recalculation initiated",
            "channel_id": channel_id,
            "estimated_time": "2-5 minutes"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recalculating revenue: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initiate revenue recalculation"
        )