"""
User Behavior Analytics API Endpoints
Track and analyze user behavior patterns
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import logging

from app.db.session import get_db
from app.api.v1.endpoints.auth import get_current_user
from app.models.user import User
from app.services.user_behavior_analytics import user_behavior_analytics_service, EventType

logger = logging.getLogger(__name__)

router = APIRouter()


class EventTrackingRequest(BaseModel):
    """Request model for tracking events"""
    event_type: str
    event_data: Dict[str, Any]
    session_id: Optional[str] = None
    client_timestamp: Optional[datetime] = None
    page_url: Optional[str] = None
    referrer: Optional[str] = None


class BehaviorOverviewResponse(BaseModel):
    """Response model for behavior overview"""
    total_events: int
    unique_users: int
    event_breakdown: List[Dict[str, Any]]
    journey_stats: Dict[str, Any]
    feature_usage: List[Dict[str, Any]]
    session_stats: Dict[str, Any]
    period: Dict[str, str]


class FunnelStep(BaseModel):
    """Funnel step definition"""
    step: str
    step_number: int
    users: int
    conversion_rate: float
    drop_off_rate: float


class FunnelAnalysisResponse(BaseModel):
    """Response model for funnel analysis"""
    funnel_name: str
    steps: List[FunnelStep]
    overall_conversion: float
    total_completions: int
    period: Dict[str, str]


class CohortAnalysisResponse(BaseModel):
    """Response model for cohort analysis"""
    cohort_type: str
    metric: str
    cohorts: List[Dict[str, Any]]
    periods: int


class HeatmapResponse(BaseModel):
    """Response model for feature heatmap"""
    heatmap: List[Dict[str, Any]]
    max_value: int
    period: Dict[str, str]


class UserSegmentsResponse(BaseModel):
    """Response model for user segments"""
    segments: Dict[str, Dict[str, Any]]
    total_users: int
    criteria: Dict[str, Any]


@router.post("/events", response_model=Dict[str, Any])
async def track_event(
    request: EventTrackingRequest,
    req: Request,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Track a user behavior event.
    
    Events can be tracked for authenticated users or anonymous sessions.
    Common event types:
    - page_view, click, form_submit, feature_use
    - video_generate, channel_create
    - error, session_start, session_end
    """
    try:
        # Get user ID (authenticated or anonymous)
        user_id = current_user.id if current_user else None
        
        # Get client info from request
        user_agent = req.headers.get("user-agent", "")
        ip_address = req.client.host if req.client else None
        
        # Enhance event data with client info
        enhanced_data = {
            **request.event_data,
            "user_agent": user_agent,
            "ip_address": ip_address,
            "page_url": request.page_url,
            "referrer": request.referrer
        }
        
        # Track the event
        result = await user_behavior_analytics_service.track_event(
            db=db,
            user_id=user_id,
            event_type=request.event_type,
            event_data=enhanced_data,
            session_id=request.session_id,
            timestamp=request.client_timestamp or datetime.utcnow()
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error tracking event: {str(e)}")
        # Don't fail the request for tracking errors
        return {"tracked": False, "error": str(e)}


@router.post("/events/batch", response_model=Dict[str, Any])
async def track_batch_events(
    events: List[EventTrackingRequest],
    req: Request,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Track multiple events in a single request.
    Useful for reducing network calls from the frontend.
    """
    try:
        user_id = current_user.id if current_user else None
        user_agent = req.headers.get("user-agent", "")
        ip_address = req.client.host if req.client else None
        
        tracked_count = 0
        failed_count = 0
        
        for event in events:
            try:
                enhanced_data = {
                    **event.event_data,
                    "user_agent": user_agent,
                    "ip_address": ip_address,
                    "page_url": event.page_url,
                    "referrer": event.referrer
                }
                
                await user_behavior_analytics_service.track_event(
                    db=db,
                    user_id=user_id,
                    event_type=event.event_type,
                    event_data=enhanced_data,
                    session_id=event.session_id,
                    timestamp=event.client_timestamp or datetime.utcnow()
                )
                tracked_count += 1
            except Exception as e:
                logger.error(f"Error tracking batch event: {str(e)}")
                failed_count += 1
                
        return {
            "tracked": tracked_count,
            "failed": failed_count,
            "total": len(events)
        }
        
    except Exception as e:
        logger.error(f"Error tracking batch events: {str(e)}")
        return {"tracked": 0, "failed": len(events), "error": str(e)}


@router.get("/behavior/overview", response_model=BehaviorOverviewResponse)
async def get_behavior_overview(
    start_date: Optional[datetime] = Query(None, description="Start date for analysis"),
    end_date: Optional[datetime] = Query(None, description="End date for analysis"),
    user_id: Optional[int] = Query(None, description="Filter by specific user"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> BehaviorOverviewResponse:
    """
    Get comprehensive behavior analytics overview.
    
    Returns:
    - Total events and unique users
    - Event type breakdown
    - User journey statistics
    - Feature usage metrics
    - Session statistics
    """
    try:
        # Admin can view any user, others only their own data
        if user_id and not current_user.is_superuser:
            if user_id != current_user.id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not authorized to view other users' analytics"
                )
                
        overview = await user_behavior_analytics_service.get_behavior_overview(
            db=db,
            user_id=user_id,
            start_date=start_date,
            end_date=end_date
        )
        
        return BehaviorOverviewResponse(**overview)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching behavior overview: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch behavior overview"
        )


@router.post("/funnels", response_model=FunnelAnalysisResponse)
async def analyze_funnel(
    funnel_steps: List[str] = Query(..., description="Ordered list of funnel steps"),
    start_date: Optional[datetime] = Query(None, description="Start date for analysis"),
    end_date: Optional[datetime] = Query(None, description="End date for analysis"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> FunnelAnalysisResponse:
    """
    Analyze conversion funnel for specified steps.
    
    Example funnel steps:
    - ["page_view", "signup", "video_generate", "channel_create"]
    - ["landing_page", "pricing_view", "checkout", "payment_complete"]
    
    Returns conversion rates and drop-off at each step.
    """
    try:
        if len(funnel_steps) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Funnel must have at least 2 steps"
            )
            
        funnel_analysis = await user_behavior_analytics_service.get_conversion_funnels(
            db=db,
            funnel_steps=funnel_steps,
            start_date=start_date,
            end_date=end_date
        )
        
        return FunnelAnalysisResponse(**funnel_analysis)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing funnel: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze funnel"
        )


@router.get("/cohorts", response_model=CohortAnalysisResponse)
async def get_cohort_analysis(
    cohort_type: str = Query('signup', description="Type of cohort: signup, first_video, etc"),
    metric: str = Query('retention', description="Metric to analyze: retention, revenue, engagement"),
    periods: int = Query(6, description="Number of periods to analyze"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> CohortAnalysisResponse:
    """
    Perform cohort analysis for user retention and behavior.
    
    Analyzes user cohorts based on signup date or other events,
    tracking their behavior over subsequent time periods.
    """
    try:
        if not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cohort analysis requires admin access"
            )
            
        if periods < 1 or periods > 12:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Periods must be between 1 and 12"
            )
            
        cohort_analysis = await user_behavior_analytics_service.get_cohort_analysis(
            db=db,
            cohort_type=cohort_type,
            metric=metric,
            periods=periods
        )
        
        return CohortAnalysisResponse(**cohort_analysis)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error performing cohort analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform cohort analysis"
        )


@router.get("/heatmaps", response_model=HeatmapResponse)
async def get_feature_heatmap(
    start_date: Optional[datetime] = Query(None, description="Start date for heatmap"),
    end_date: Optional[datetime] = Query(None, description="End date for heatmap"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> HeatmapResponse:
    """
    Generate feature usage heatmap showing activity patterns.
    
    Returns hourly usage patterns for each day in the period,
    useful for identifying peak usage times and patterns.
    """
    try:
        heatmap_data = await user_behavior_analytics_service.get_feature_heatmap(
            db=db,
            start_date=start_date,
            end_date=end_date
        )
        
        return HeatmapResponse(**heatmap_data)
        
    except Exception as e:
        logger.error(f"Error generating heatmap: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate heatmap"
        )


@router.post("/segments", response_model=UserSegmentsResponse)
async def get_user_segments(
    criteria: Dict[str, Any] = Query({}, description="Segmentation criteria"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> UserSegmentsResponse:
    """
    Segment users based on behavior patterns.
    
    Identifies user segments such as:
    - Power users (high activity)
    - At-risk users (declining activity)
    - New users (recent signups)
    - Dormant users (no recent activity)
    
    Useful for targeted marketing and retention strategies.
    """
    try:
        if not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User segmentation requires admin access"
            )
            
        segments = await user_behavior_analytics_service.get_user_segments(
            db=db,
            segmentation_criteria=criteria
        )
        
        return UserSegmentsResponse(**segments)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error segmenting users: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to segment users"
        )


@router.get("/sessions/{session_id}")
async def get_session_details(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get detailed information about a specific session.
    
    Returns all events and metadata for the session.
    """
    try:
        from app.models.user_session import UserSession
        from app.models.user_event import UserEvent
        from sqlalchemy import select
        
        # Get session
        session_query = select(UserSession).where(UserSession.session_id == session_id)
        session_result = await db.execute(session_query)
        session = session_result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
            
        # Check authorization
        if not current_user.is_superuser and session.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to view this session"
            )
            
        # Get session events
        events_query = (
            select(UserEvent)
            .where(UserEvent.session_id == session_id)
            .order_by(UserEvent.timestamp)
        )
        events_result = await db.execute(events_query)
        events = events_result.scalars().all()
        
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "start_time": session.start_time.isoformat() if session.start_time else None,
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "duration_seconds": session.duration_seconds,
            "event_count": session.event_count,
            "is_active": session.is_active,
            "device_info": {
                "browser": session.browser,
                "device_type": session.device_type,
                "os": session.os,
                "screen_resolution": session.screen_resolution
            },
            "events": [
                {
                    "event_type": e.event_type,
                    "timestamp": e.timestamp.isoformat(),
                    "page_url": e.page_url,
                    "event_data": e.event_data
                }
                for e in events
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching session details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch session details"
        )