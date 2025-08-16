"""
Cost tracking API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from datetime import datetime, timedelta
from decimal import Decimal

from app.db.session import get_db
from app.services.cost_tracking import cost_tracker, CostMetrics
from app.api.v1.endpoints.auth import get_current_verified_user
from app.models.user import User
from app.schemas.cost import (
    CostRecordCreate,
    CostRecordResponse,
    CostAggregationResponse,
    ThresholdCreate,
    ThresholdResponse,
    VideoCostResponse,
)

router = APIRouter(prefix="/cost", tags=["cost-tracking"])


@router.post("/track")
async def track_cost(
    cost_data: CostRecordCreate,
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db),
) -> CostRecordResponse:
    """Track API usage cost"""
    try:
        # Add user context to metadata
        metadata = cost_data.metadata or {}
        metadata["user_id"] = current_user.id

        # Track the cost
        total_cost = await cost_tracker.track_api_call(
            service=cost_data.service,
            operation=cost_data.operation,
            units=cost_data.units,
            metadata=metadata,
            db=db,
        )

        return CostRecordResponse(
            service=cost_data.service,
            operation=cost_data.operation,
            units=cost_data.units,
            total_cost=float(total_cost),
            timestamp=datetime.utcnow(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/realtime")
async def get_realtime_costs(
    current_user: User = Depends(get_current_verified_user),
) -> CostMetrics:
    """Get real-time cost metrics"""
    try:
        metrics = await cost_tracker.get_real_time_costs()
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/video/{video_id}")
async def get_video_cost(
    video_id: str,
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db),
) -> VideoCostResponse:
    """Get cost breakdown for a specific video"""
    try:
        cost_data = await cost_tracker.get_video_cost(video_id, db)
        if not cost_data:
            raise HTTPException(status_code=404, detail="Video cost data not found")
        return VideoCostResponse(**cost_data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/aggregations")
async def get_cost_aggregations(
    start_date: datetime = Query(..., description="Start date for aggregation"),
    end_date: datetime = Query(..., description="End date for aggregation"),
    granularity: str = Query("day", pattern="^(hour|day|week|month)$"),
    service: Optional[str] = Query(None, description="Filter by service"),
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db),
) -> List[CostAggregationResponse]:
    """Get aggregated cost data"""
    try:
        aggregations = await cost_tracker.get_cost_aggregations(
            start_date=start_date, end_date=end_date, granularity=granularity, db=db
        )

        # Filter by service if specified
        if service:
            aggregations = [a for a in aggregations if a["service"] == service]

        return [CostAggregationResponse(**agg) for agg in aggregations]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/threshold")
async def set_cost_threshold(
    threshold_data: ThresholdCreate,
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db),
) -> ThresholdResponse:
    """Set a cost threshold alert"""
    try:
        # Only admins can set thresholds
        if not current_user.is_admin:
            raise HTTPException(status_code=403, detail="Admin access required")

        await cost_tracker.set_threshold(
            threshold_type=threshold_data.threshold_type,
            value=Decimal(str(threshold_data.value)),
            service=threshold_data.service,
            alert_email=threshold_data.alert_email,
            db=db,
        )

        return ThresholdResponse(
            threshold_type=threshold_data.threshold_type,
            value=threshold_data.value,
            service=threshold_data.service,
            alert_email=threshold_data.alert_email,
            is_active=True,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/daily-summary")
async def get_daily_cost_summary(
    date: Optional[datetime] = Query(
        None, description="Date for summary (defaults to today)"
    ),
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db),
):
    """Get daily cost summary"""
    try:
        # Default to today if no date specified
        if not date:
            date = datetime.utcnow().date()
        else:
            date = date.date()

        start_of_day = datetime.combine(date, datetime.min.time())
        end_of_day = datetime.combine(date, datetime.max.time())

        # Get aggregations for the day
        aggregations = await cost_tracker.get_cost_aggregations(
            start_date=start_of_day, end_date=end_of_day, granularity="hour", db=db
        )

        # Calculate summary metrics
        total_cost = sum(agg["total_cost"] for agg in aggregations)
        by_service = {}
        for agg in aggregations:
            service = agg["service"]
            if service not in by_service:
                by_service[service] = 0
            by_service[service] += agg["total_cost"]

        # Get current metrics
        metrics = await cost_tracker.get_real_time_costs()

        return {
            "date": date.isoformat(),
            "total_cost": total_cost,
            "cost_by_service": by_service,
            "hourly_breakdown": aggregations,
            "per_video_cost": float(metrics.per_video_cost)
            if metrics.per_video_cost
            else None,
            "threshold_status": metrics.threshold_status,
            "projections": {
                "daily": float(metrics.daily_cost),
                "monthly": float(metrics.monthly_projection),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends")
async def get_cost_trends(
    days: int = Query(7, ge=1, le=90, description="Number of days to analyze"),
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db),
):
    """Get cost trends over time"""
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        # Get daily aggregations
        aggregations = await cost_tracker.get_cost_aggregations(
            start_date=start_date, end_date=end_date, granularity="day", db=db
        )

        # Calculate trends
        daily_costs = {}
        for agg in aggregations:
            day = agg["period"].split("T")[0]
            if day not in daily_costs:
                daily_costs[day] = 0
            daily_costs[day] += agg["total_cost"]

        # Calculate moving average
        costs_list = list(daily_costs.values())
        if len(costs_list) > 1:
            avg_cost = sum(costs_list) / len(costs_list)
            trend = "increasing" if costs_list[-1] > avg_cost else "decreasing"
            change_pct = (
                ((costs_list[-1] - costs_list[0]) / costs_list[0] * 100)
                if costs_list[0] > 0
                else 0
            )
        else:
            avg_cost = costs_list[0] if costs_list else 0
            trend = "stable"
            change_pct = 0

        return {
            "period_days": days,
            "daily_costs": daily_costs,
            "average_daily_cost": avg_cost,
            "trend": trend,
            "change_percentage": change_pct,
            "total_period_cost": sum(daily_costs.values()),
            "highest_day": max(daily_costs.items(), key=lambda x: x[1])
            if daily_costs
            else None,
            "lowest_day": min(daily_costs.items(), key=lambda x: x[1])
            if daily_costs
            else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
