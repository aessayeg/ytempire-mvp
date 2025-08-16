"""
Reports Generation Endpoints
Handles analytics reports, performance summaries, and data exports
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, date
from decimal import Decimal
import json
import csv
import io
import logging
from pydantic import BaseModel, Field

from app.db.session import get_db
from app.core.auth import get_current_user
from app.models.user import User
from app.models.video import Video
from app.models.channel import Channel
from app.models.analytics import Analytics
from app.models.cost import Cost
from app.services.analytics_service import analytics_service
from app.services.cost_tracking import cost_tracker
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models for requests/responses
class ReportRequest(BaseModel):
    """Report generation request"""

    report_type: str = Field(
        ..., description="Type of report (performance, cost, analytics, summary)"
    )
    start_date: date = Field(..., description="Report start date")
    end_date: date = Field(..., description="Report end date")
    channel_ids: Optional[List[str]] = Field(None, description="Filter by channel IDs")
    format: str = Field("json", description="Output format (json, csv, pdf)")
    include_details: bool = Field(True, description="Include detailed breakdowns")


class PerformanceReport(BaseModel):
    """Performance report data"""

    period: Dict[str, str]
    summary: Dict[str, Any]
    video_metrics: List[Dict[str, Any]]
    channel_metrics: List[Dict[str, Any]]
    trend_analysis: Dict[str, Any]
    recommendations: List[str]


class CostReport(BaseModel):
    """Cost analysis report"""

    period: Dict[str, str]
    total_cost: float
    cost_breakdown: Dict[str, float]
    cost_per_video: float
    savings: Dict[str, float]
    projections: Dict[str, float]


@router.post("/generate", response_model=Dict[str, Any])
async def generate_report(
    request: ReportRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Generate a comprehensive report

    Args:
        request: Report parameters

    Returns:
        Report data or download link
    """
    try:
        # Validate date range
        if request.end_date < request.start_date:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="End date must be after start date",
            )

        # Generate report based on type
        if request.report_type == "performance":
            report_data = await _generate_performance_report(
                db,
                current_user.id,
                request.start_date,
                request.end_date,
                request.channel_ids,
                request.include_details,
            )

        elif request.report_type == "cost":
            report_data = await _generate_cost_report(
                db,
                current_user.id,
                request.start_date,
                request.end_date,
                request.channel_ids,
                request.include_details,
            )

        elif request.report_type == "analytics":
            report_data = await _generate_analytics_report(
                db,
                current_user.id,
                request.start_date,
                request.end_date,
                request.channel_ids,
                request.include_details,
            )

        elif request.report_type == "summary":
            report_data = await _generate_summary_report(
                db,
                current_user.id,
                request.start_date,
                request.end_date,
                request.channel_ids,
            )

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown report type: {request.report_type}",
            )

        # Format output
        if request.format == "csv":
            # Convert to CSV for download
            csv_content = _convert_to_csv(report_data)
            filename = (
                f"report_{request.report_type}_{datetime.now().strftime('%Y%m%d')}.csv"
            )

            # Schedule background cleanup if needed
            background_tasks.add_task(_cleanup_temp_files, filename)

            return {
                "status": "success",
                "format": "csv",
                "filename": filename,
                "download_url": f"/api/v1/reports/download/{filename}",
            }

        elif request.format == "pdf":
            # Generate PDF (placeholder - would use reportlab or similar)
            pdf_filename = (
                f"report_{request.report_type}_{datetime.now().strftime('%Y%m%d')}.pdf"
            )

            return {
                "status": "success",
                "format": "pdf",
                "filename": pdf_filename,
                "download_url": f"/api/v1/reports/download/{pdf_filename}",
            }

        else:
            # Return JSON data
            return {"status": "success", "format": "json", "data": report_data}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation failed: {str(e)}",
        )


@router.get("/performance", response_model=PerformanceReport)
async def get_performance_report(
    start_date: date = Query(..., description="Start date"),
    end_date: date = Query(..., description="End date"),
    channel_id: Optional[str] = Query(None, description="Filter by channel"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get performance report for specified period

    Args:
        start_date: Report start date
        end_date: Report end date
        channel_id: Optional channel filter

    Returns:
        Performance metrics and analysis
    """
    try:
        channel_ids = [channel_id] if channel_id else None
        report_data = await _generate_performance_report(
            db, current_user.id, start_date, end_date, channel_ids, True
        )

        return PerformanceReport(**report_data)

    except Exception as e:
        logger.error(f"Error getting performance report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate performance report: {str(e)}",
        )


@router.get("/cost", response_model=CostReport)
async def get_cost_report(
    start_date: date = Query(..., description="Start date"),
    end_date: date = Query(..., description="End date"),
    channel_id: Optional[str] = Query(None, description="Filter by channel"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get cost analysis report

    Args:
        start_date: Report start date
        end_date: Report end date
        channel_id: Optional channel filter

    Returns:
        Cost breakdown and analysis
    """
    try:
        channel_ids = [channel_id] if channel_id else None
        report_data = await _generate_cost_report(
            db, current_user.id, start_date, end_date, channel_ids, True
        )

        return CostReport(**report_data)

    except Exception as e:
        logger.error(f"Error getting cost report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate cost report: {str(e)}",
        )


@router.get("/weekly")
async def get_weekly_report(
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    """
    Get weekly summary report

    Returns:
        Weekly performance summary
    """
    try:
        end_date = date.today()
        start_date = end_date - timedelta(days=7)

        report = await _generate_summary_report(
            db, current_user.id, start_date, end_date, None
        )

        return report

    except Exception as e:
        logger.error(f"Error getting weekly report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate weekly report: {str(e)}",
        )


@router.get("/monthly")
async def get_monthly_report(
    month: Optional[int] = Query(None, ge=1, le=12, description="Month (1-12)"),
    year: Optional[int] = Query(None, ge=2024, description="Year"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get monthly summary report

    Args:
        month: Month number (defaults to current month)
        year: Year (defaults to current year)

    Returns:
        Monthly performance summary
    """
    try:
        # Default to current month if not specified
        if not month or not year:
            today = date.today()
            month = month or today.month
            year = year or today.year

        # Calculate date range
        start_date = date(year, month, 1)
        if month == 12:
            end_date = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = date(year, month + 1, 1) - timedelta(days=1)

        report = await _generate_summary_report(
            db, current_user.id, start_date, end_date, None
        )

        return report

    except Exception as e:
        logger.error(f"Error getting monthly report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate monthly report: {str(e)}",
        )


@router.get("/download/{filename}")
async def download_report(
    filename: str, current_user: User = Depends(get_current_user)
):
    """
    Download generated report file

    Args:
        filename: Report filename

    Returns:
        File download response
    """
    try:
        # Validate filename (security check)
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid filename"
            )

        # Check if file exists
        file_path = f"/tmp/reports/{filename}"  # Use appropriate temp directory

        # Return file
        return FileResponse(
            path=file_path, filename=filename, media_type="application/octet-stream"
        )

    except Exception as e:
        logger.error(f"Error downloading report: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Report file not found"
        )


# Helper functions for report generation
async def _generate_performance_report(
    db: AsyncSession,
    user_id: str,
    start_date: date,
    end_date: date,
    channel_ids: Optional[List[str]],
    include_details: bool,
) -> Dict[str, Any]:
    """Generate performance report data"""

    # Query videos in date range
    query = select(Video).where(
        and_(
            Video.user_id == user_id,
            Video.created_at >= start_date,
            Video.created_at <= end_date,
        )
    )

    if channel_ids:
        query = query.where(Video.channel_id.in_(channel_ids))

    result = await db.execute(query)
    videos = result.scalars().all()

    # Calculate metrics
    total_videos = len(videos)
    total_views = sum(v.view_count or 0 for v in videos)
    total_likes = sum(v.like_count or 0 for v in videos)
    avg_engagement = (total_likes / total_views * 100) if total_views > 0 else 0

    # Video metrics
    video_metrics = []
    if include_details:
        for video in videos[:50]:  # Limit to top 50
            video_metrics.append(
                {
                    "id": video.id,
                    "title": video.title,
                    "views": video.view_count,
                    "likes": video.like_count,
                    "engagement_rate": (video.like_count / video.view_count * 100)
                    if video.view_count
                    else 0,
                    "created_at": video.created_at.isoformat(),
                }
            )

    # Channel metrics
    channel_metrics = []
    if channel_ids:
        for channel_id in channel_ids:
            channel_videos = [v for v in videos if v.channel_id == channel_id]
            channel_metrics.append(
                {
                    "channel_id": channel_id,
                    "total_videos": len(channel_videos),
                    "total_views": sum(v.view_count or 0 for v in channel_videos),
                    "avg_views": sum(v.view_count or 0 for v in channel_videos)
                    / len(channel_videos)
                    if channel_videos
                    else 0,
                }
            )

    return {
        "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "summary": {
            "total_videos": total_videos,
            "total_views": total_views,
            "total_likes": total_likes,
            "avg_engagement": round(avg_engagement, 2),
            "avg_views_per_video": round(total_views / total_videos, 0)
            if total_videos
            else 0,
        },
        "video_metrics": video_metrics,
        "channel_metrics": channel_metrics,
        "trend_analysis": {
            "growth_rate": "15%",  # Placeholder - calculate actual growth
            "best_performing_day": "Monday",
            "peak_hour": "3 PM EST",
        },
        "recommendations": [
            "Focus on technology content for higher engagement",
            "Post more frequently on weekdays",
            "Optimize thumbnails for mobile viewing",
        ],
    }


async def _generate_cost_report(
    db: AsyncSession,
    user_id: str,
    start_date: date,
    end_date: date,
    channel_ids: Optional[List[str]],
    include_details: bool,
) -> Dict[str, Any]:
    """Generate cost analysis report"""

    # Query costs in date range
    query = select(Cost).where(
        and_(
            Cost.user_id == user_id,
            Cost.created_at >= start_date,
            Cost.created_at <= end_date,
        )
    )

    result = await db.execute(query)
    costs = result.scalars().all()

    # Calculate totals
    total_cost = sum(float(c.amount) for c in costs)

    # Breakdown by service
    cost_breakdown = {}
    for cost in costs:
        service = cost.service
        if service not in cost_breakdown:
            cost_breakdown[service] = 0
        cost_breakdown[service] += float(cost.amount)

    # Calculate per-video cost
    video_count_query = select(func.count(Video.id)).where(
        and_(
            Video.user_id == user_id,
            Video.created_at >= start_date,
            Video.created_at <= end_date,
        )
    )

    if channel_ids:
        video_count_query = video_count_query.where(Video.channel_id.in_(channel_ids))

    result = await db.execute(video_count_query)
    video_count = result.scalar() or 1

    cost_per_video = total_cost / video_count if video_count else 0

    return {
        "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "total_cost": round(total_cost, 2),
        "cost_breakdown": {k: round(v, 2) for k, v in cost_breakdown.items()},
        "cost_per_video": round(cost_per_video, 2),
        "savings": {
            "from_caching": round(total_cost * 0.15, 2),  # Estimate 15% savings
            "from_optimization": round(total_cost * 0.10, 2),  # Estimate 10% savings
        },
        "projections": {
            "next_month": round(total_cost * 1.2, 2),  # 20% growth projection
            "next_quarter": round(total_cost * 3.5, 2),
        },
    }


async def _generate_analytics_report(
    db: AsyncSession,
    user_id: str,
    start_date: date,
    end_date: date,
    channel_ids: Optional[List[str]],
    include_details: bool,
) -> Dict[str, Any]:
    """Generate analytics report"""

    # Get analytics data
    analytics_data = await analytics_service.get_analytics_summary(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        channel_ids=channel_ids,
    )

    return analytics_data


async def _generate_summary_report(
    db: AsyncSession,
    user_id: str,
    start_date: date,
    end_date: date,
    channel_ids: Optional[List[str]],
) -> Dict[str, Any]:
    """Generate summary report combining all metrics"""

    # Get performance data
    performance = await _generate_performance_report(
        db, user_id, start_date, end_date, channel_ids, False
    )

    # Get cost data
    costs = await _generate_cost_report(
        db, user_id, start_date, end_date, channel_ids, False
    )

    return {
        "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "performance": performance["summary"],
        "costs": {"total": costs["total_cost"], "per_video": costs["cost_per_video"]},
        "roi": {
            "revenue_estimate": performance["summary"]["total_views"]
            * 0.002,  # $2 CPM estimate
            "profit_margin": (
                (performance["summary"]["total_views"] * 0.002) - costs["total_cost"]
            )
            / costs["total_cost"]
            * 100
            if costs["total_cost"]
            else 0,
        },
        "highlights": [
            f"Generated {performance['summary']['total_videos']} videos",
            f"Achieved {performance['summary']['total_views']:,} total views",
            f"Maintained ${costs['cost_per_video']:.2f} cost per video",
            f"Engagement rate: {performance['summary']['avg_engagement']:.1f}%",
        ],
    }


def _convert_to_csv(data: Dict[str, Any]) -> str:
    """Convert report data to CSV format"""
    output = io.StringIO()
    writer = csv.writer(output)

    # Flatten nested data structure for CSV
    if "summary" in data:
        writer.writerow(["Metric", "Value"])
        for key, value in data["summary"].items():
            writer.writerow([key, value])

    return output.getvalue()


async def _cleanup_temp_files(filename: str):
    """Clean up temporary report files"""
    # Implement cleanup logic
    pass
