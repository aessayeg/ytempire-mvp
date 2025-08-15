"""
Analytics API Endpoints
Provides access to analytics data and reports
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
from app.services.analytics_service import (
    analytics_pipeline,
    MetricType,
    AggregationLevel,
    DataSource,
    MetricPoint
)
from app.services.roi_calculator import roi_calculator, ROIMetric

import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class TimeRange(str, Enum):
    """Predefined time ranges"""
    LAST_HOUR = "last_hour"
    LAST_24_HOURS = "last_24_hours"
    LAST_7_DAYS = "last_7_days"
    LAST_30_DAYS = "last_30_days"
    LAST_90_DAYS = "last_90_days"
    CUSTOM = "custom"


class MetricRequest(BaseModel):
    """Request to ingest metrics"""
    metric_type: MetricType
    value: float
    source: DataSource = DataSource.INTERNAL
    channel_id: Optional[str] = None
    video_id: Optional[str] = None
    metadata: Dict[str, Any] = {}


class BatchMetricRequest(BaseModel):
    """Batch metric ingestion request"""
    metrics: List[MetricRequest]


class AnalyticsQuery(BaseModel):
    """Analytics query parameters"""
    metric_types: List[MetricType]
    time_range: TimeRange = TimeRange.LAST_7_DAYS
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    aggregation_level: Optional[AggregationLevel] = AggregationLevel.DAY
    channel_id: Optional[str] = None
    video_id: Optional[str] = None


class ChannelAnalyticsResponse(BaseModel):
    """Channel analytics response"""
    channel_id: str
    period: Dict[str, str]
    metrics: Dict[str, Any]
    trends: Dict[str, str]
    top_videos: List[Dict[str, Any]]
    engagement: Dict[str, float]


class VideoPerformanceResponse(BaseModel):
    """Video performance response"""
    video_id: str
    period_days: int
    metrics: Dict[str, Any]
    hourly_views: List[Dict[str, Any]]
    retention: Dict[str, Any]
    traffic_sources: Dict[str, Any]


class RealtimeMetrics(BaseModel):
    """Real-time metrics response"""
    timestamp: datetime
    active_viewers: int
    videos_processing: int
    api_calls_per_minute: int
    error_rate: float
    average_response_time: float


@router.post("/ingest")
async def ingest_metric(
    request: MetricRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, str]:
    """
    Ingest a single metric into the analytics pipeline
    """
    try:
        metric = MetricPoint(
            timestamp=datetime.utcnow(),
            metric_type=request.metric_type,
            value=request.value,
            source=request.source,
            channel_id=request.channel_id,
            video_id=request.video_id,
            user_id=str(current_user.id),
            metadata=request.metadata
        )
        
        background_tasks.add_task(analytics_pipeline.ingest_metric, metric)
        
        return {"status": "accepted", "timestamp": metric.timestamp.isoformat()}
        
    except Exception as e:
        logger.error(f"Failed to ingest metric: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest metric: {str(e)}"
        )


@router.post("/ingest/batch")
async def ingest_batch_metrics(
    request: BatchMetricRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Ingest a batch of metrics
    """
    try:
        metrics = []
        for metric_req in request.metrics:
            metric = MetricPoint(
                timestamp=datetime.utcnow(),
                metric_type=metric_req.metric_type,
                value=metric_req.value,
                source=metric_req.source,
                channel_id=metric_req.channel_id,
                video_id=metric_req.video_id,
                user_id=str(current_user.id),
                metadata=metric_req.metadata
            )
            metrics.append(metric)
        
        background_tasks.add_task(analytics_pipeline.ingest_batch, metrics)
        
        return {
            "status": "accepted",
            "count": len(metrics),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to ingest batch metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest batch: {str(e)}"
        )


@router.post("/query")
async def query_analytics(
    query: AnalyticsQuery,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Query analytics data with flexible parameters
    """
    try:
        # Determine time range
        if query.time_range == TimeRange.CUSTOM:
            if not query.start_date or not query.end_date:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Start and end dates required for custom range"
                )
            start_time = datetime.combine(query.start_date, datetime.min.time())
            end_time = datetime.combine(query.end_date, datetime.max.time())
        else:
            end_time = datetime.utcnow()
            if query.time_range == TimeRange.LAST_HOUR:
                start_time = end_time - timedelta(hours=1)
            elif query.time_range == TimeRange.LAST_24_HOURS:
                start_time = end_time - timedelta(days=1)
            elif query.time_range == TimeRange.LAST_7_DAYS:
                start_time = end_time - timedelta(days=7)
            elif query.time_range == TimeRange.LAST_30_DAYS:
                start_time = end_time - timedelta(days=30)
            else:  # LAST_90_DAYS
                start_time = end_time - timedelta(days=90)
        
        # Query metrics for each type
        results = {}
        for metric_type in query.metric_types:
            metrics = await analytics_pipeline.query_metrics(
                metric_type=metric_type,
                start_time=start_time,
                end_time=end_time,
                aggregation_level=query.aggregation_level,
                channel_id=query.channel_id,
                video_id=query.video_id
            )
            results[metric_type.value] = metrics
        
        return {
            "query": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "aggregation_level": query.aggregation_level.value if query.aggregation_level else None
            },
            "results": results,
            "count": sum(len(m) for m in results.values())
        }
        
    except Exception as e:
        logger.error(f"Analytics query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )


@router.get("/channels/{channel_id}", response_model=ChannelAnalyticsResponse)
async def get_channel_analytics(
    channel_id: str,
    start_date: date = Query(default=None),
    end_date: date = Query(default=None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> ChannelAnalyticsResponse:
    """
    Get comprehensive analytics for a channel
    """
    try:
        # Default to last 30 days if not specified
        if not end_date:
            end_date = date.today()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        analytics = await analytics_pipeline.get_channel_analytics(
            channel_id=channel_id,
            start_date=start_date,
            end_date=end_date
        )
        
        return ChannelAnalyticsResponse(**analytics)
        
    except Exception as e:
        logger.error(f"Failed to get channel analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get analytics: {str(e)}"
        )


@router.get("/videos/{video_id}/performance", response_model=VideoPerformanceResponse)
async def get_video_performance(
    video_id: str,
    days: int = Query(default=7, ge=1, le=90),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> VideoPerformanceResponse:
    """
    Get performance metrics for a specific video
    """
    try:
        performance = await analytics_pipeline.get_video_performance(
            video_id=video_id,
            days=days
        )
        
        return VideoPerformanceResponse(**performance)
        
    except Exception as e:
        logger.error(f"Failed to get video performance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance: {str(e)}"
        )


@router.get("/realtime", response_model=RealtimeMetrics)
async def get_realtime_metrics(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> RealtimeMetrics:
    """
    Get real-time system metrics
    """
    try:
        # Get metrics from last minute
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=1)
        
        # Query various real-time metrics
        # In production, these would come from monitoring systems
        
        return RealtimeMetrics(
            timestamp=end_time,
            active_viewers=42,  # Mock data
            videos_processing=3,
            api_calls_per_minute=127,
            error_rate=0.002,
            average_response_time=145.3
        )
        
    except Exception as e:
        logger.error(f"Failed to get realtime metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get realtime metrics"
        )


@router.get("/reports")
async def list_reports(
    report_type: Optional[str] = Query(default=None),
    limit: int = Query(default=10, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> List[Dict[str, Any]]:
    """
    List available analytics reports
    """
    try:
        # In production, query from database
        reports = [
            {
                "report_id": "daily_2025-01-09",
                "report_type": "daily",
                "generated_at": "2025-01-10T00:00:00",
                "status": "completed"
            },
            {
                "report_id": "weekly_2025-01-03_2025-01-09",
                "report_type": "weekly",
                "generated_at": "2025-01-10T00:00:00",
                "status": "completed"
            }
        ]
        
        if report_type:
            reports = [r for r in reports if r["report_type"] == report_type]
        
        return reports[:limit]
        
    except Exception as e:
        logger.error(f"Failed to list reports: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list reports"
        )


@router.get("/reports/{report_id}")
async def get_report(
    report_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Get a specific analytics report
    """
    try:
        # In production, fetch from storage
        if not report_id.startswith(("daily_", "weekly_", "monthly_")):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Report not found"
            )
        
        return {
            "report_id": report_id,
            "report_type": report_id.split("_")[0],
            "generated_at": datetime.utcnow().isoformat(),
            "metrics": {
                "total_views": 15234,
                "total_watch_hours": 892.5,
                "new_subscribers": 127,
                "revenue": 456.78,
                "costs": 123.45
            },
            "insights": [
                "View count increased by 23% compared to previous period",
                "Engagement rate is above industry average",
                "Peak viewing hours are between 8-10 PM EST"
            ],
            "recommendations": [
                "Consider posting more content during peak hours",
                "Increase video frequency to maintain growth momentum",
                "Focus on topics that generated highest engagement"
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get report"
        )


@router.get("/export")
async def export_analytics(
    format: str = Query(default="csv", pattern="^(csv|json|excel)$"),
    time_range: TimeRange = Query(default=TimeRange.LAST_7_DAYS),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Export analytics data in various formats
    """
    try:
        # In production, generate actual export file
        export_url = f"https://storage.ytempire.com/exports/{current_user.id}/analytics_{time_range.value}.{format}"
        
        return {
            "status": "generating",
            "format": format,
            "time_range": time_range.value,
            "export_url": export_url,
            "expires_at": (datetime.utcnow() + timedelta(hours=24)).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to export analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export analytics"
        )


@router.get("/benchmarks")
async def get_industry_benchmarks(
    category: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Get industry benchmark data for comparison
    """
    try:
        benchmarks = {
            "general": {
                "avg_view_duration": 2.5,
                "engagement_rate": 0.04,
                "ctr": 0.02,
                "subscriber_conversion": 0.01
            },
            "education": {
                "avg_view_duration": 4.2,
                "engagement_rate": 0.06,
                "ctr": 0.025,
                "subscriber_conversion": 0.015
            },
            "entertainment": {
                "avg_view_duration": 3.1,
                "engagement_rate": 0.05,
                "ctr": 0.03,
                "subscriber_conversion": 0.012
            }
        }
        
        if category and category in benchmarks:
            return {
                "category": category,
                "benchmarks": benchmarks[category],
                "updated_at": datetime.utcnow().isoformat()
            }
        
        return {
            "categories": list(benchmarks.keys()),
            "benchmarks": benchmarks,
            "updated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get benchmarks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get benchmarks"
        )


@router.get("/roi/overall")
async def get_overall_roi(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Calculate overall ROI metrics for the platform or user
    """
    try:
        start_datetime = datetime.combine(start_date, datetime.min.time()) if start_date else None
        end_datetime = datetime.combine(end_date, datetime.max.time()) if end_date else None
        
        roi_data = await roi_calculator.calculate_overall_roi(
            db=db,
            start_date=start_datetime,
            end_date=end_datetime,
            user_id=str(current_user.id)
        )
        
        return roi_data
        
    except Exception as e:
        logger.error(f"Failed to calculate overall ROI: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate ROI"
        )


@router.get("/roi/channel/{channel_id}")
async def get_channel_roi(
    channel_id: str,
    period_days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Calculate ROI metrics for a specific channel
    """
    try:
        channel_roi = await roi_calculator.calculate_channel_roi(
            db=db,
            channel_id=channel_id,
            period_days=period_days
        )
        
        return channel_roi.__dict__
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to calculate channel ROI: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate channel ROI"
        )


@router.get("/roi/video/{video_id}")
async def get_video_roi(
    video_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Calculate ROI metrics for a specific video
    """
    try:
        video_roi = await roi_calculator.calculate_video_roi(
            db=db,
            video_id=video_id
        )
        
        return video_roi.__dict__
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to calculate video ROI: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate video ROI"
        )