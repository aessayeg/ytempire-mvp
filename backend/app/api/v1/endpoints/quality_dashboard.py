"""
Quality Dashboard API Endpoints
Provides real-time quality metrics and dashboards
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
import json

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from app.services.analytics_service import (
    metrics_collector,
    quality_dashboard, 
    quality_monitor,
    MetricType,
    MetricCategory,
    QUALITY_METRICS
)
# Removed defect_tracking (unrelated) import defect_tracker, DefectSeverity, DefectStatus
from app.services.automated_reporting import setup_automated_reporting
from app.core.deps import get_current_user

router = APIRouter()

# ============================================================================
# Request/Response Models
# ============================================================================

class MetricValueRequest(BaseModel):
    """Request to record a metric value"""
    metric_id: str
    value: float
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[Dict[str, str]] = None

class DashboardFilters(BaseModel):
    """Dashboard filtering options"""
    time_range: str = Field(default="24h", description="Time range: 1h, 24h, 7d, 30d")
    metric_types: Optional[List[MetricType]] = None
    categories: Optional[List[MetricCategory]] = None
    include_trends: bool = True
    include_charts: bool = False

class DefectFilters(BaseModel):
    """Defect filtering options"""
    severity: Optional[List[DefectSeverity]] = None
    status: Optional[List[DefectStatus]] = None
    assignee_id: Optional[str] = None
    component: Optional[str] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None

# ============================================================================
# Quality Metrics Endpoints
# ============================================================================

@router.get("/overview", summary="Get quality overview")
async def get_quality_overview(
    current_user = Depends(get_current_user)
):
    """Get overall quality dashboard overview"""
    try:
        overview = await quality_dashboard.generate_quality_overview()
        return {
            "status": "success",
            "data": overview
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics", summary="Get latest metrics")
async def get_latest_metrics(
    metric_types: Optional[List[MetricType]] = Query(None),
    categories: Optional[List[MetricCategory]] = Query(None),
    current_user = Depends(get_current_user)
):
    """Get latest metric values with optional filtering"""
    try:
        latest_metrics = await metrics_collector.get_latest_metrics()
        
        # Filter by type and category if specified
        if metric_types or categories:
            filtered_metrics = {}
            for metric_id, metric_value in latest_metrics.items():
                metric_def = QUALITY_METRICS.get(metric_id)
                if metric_def:
                    if metric_types and metric_def.type not in metric_types:
                        continue
                    if categories and metric_def.category not in categories:
                        continue
                    filtered_metrics[metric_id] = metric_value
            latest_metrics = filtered_metrics
        
        # Convert to API response format
        metrics_data = {}
        for metric_id, metric_value in latest_metrics.items():
            metric_def = QUALITY_METRICS.get(metric_id)
            metrics_data[metric_id] = {
                "name": metric_def.name if metric_def else metric_id,
                "value": metric_value.value,
                "unit": metric_def.unit if metric_def else "",
                "status": metric_value.status.value,
                "timestamp": metric_value.timestamp.isoformat(),
                "type": metric_def.type.value if metric_def else None,
                "category": metric_def.category.value if metric_def else None,
                "target": metric_def.threshold.target_value if metric_def else None
            }
        
        return {
            "status": "success",
            "data": metrics_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/metrics", summary="Record metric value")
async def record_metric(
    metric_request: MetricValueRequest,
    current_user = Depends(get_current_user)
):
    """Record a new metric value"""
    try:
        metric_value = await metrics_collector.collect_metric(
            metric_id=metric_request.metric_id,
            value=metric_request.value,
            timestamp=metric_request.timestamp,
            metadata=metric_request.metadata,
            tags=metric_request.tags
        )
        
        if metric_value:
            return {
                "status": "success",
                "message": "Metric recorded successfully",
                "data": {
                    "metric_id": metric_value.metric_id,
                    "value": metric_value.value,
                    "status": metric_value.status.value,
                    "timestamp": metric_value.timestamp.isoformat()
                }
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to record metric")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/{metric_id}/history", summary="Get metric history")
async def get_metric_history(
    metric_id: str,
    start_time: datetime = Query(..., description="Start time for history"),
    end_time: datetime = Query(..., description="End time for history"),
    aggregation: str = Query("raw", description="Aggregation type: raw, hourly, daily"),
    current_user = Depends(get_current_user)
):
    """Get historical data for a specific metric"""
    try:
        if metric_id not in QUALITY_METRICS:
            raise HTTPException(status_code=404, detail="Metric not found")
        
        history = await metrics_collector.get_metric_history(
            metric_id=metric_id,
            start_time=start_time,
            end_time=end_time,
            aggregation=aggregation
        )
        
        history_data = [
            {
                "timestamp": mv.timestamp.isoformat(),
                "value": mv.value,
                "status": mv.status.value,
                "metadata": mv.metadata,
                "tags": mv.tags
            }
            for mv in history
        ]
        
        metric_def = QUALITY_METRICS[metric_id]
        return {
            "status": "success",
            "data": {
                "metric_info": {
                    "id": metric_id,
                    "name": metric_def.name,
                    "type": metric_def.type.value,
                    "unit": metric_def.unit,
                    "target": metric_def.threshold.target_value
                },
                "history": history_data,
                "count": len(history_data)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trends", summary="Get metric trends")
async def get_metric_trends(
    days: int = Query(7, ge=1, le=90, description="Number of days for trend analysis"),
    current_user = Depends(get_current_user)
):
    """Get trend analysis for all metrics"""
    try:
        trends = await quality_dashboard.generate_trend_analysis(days=days)
        return {
            "status": "success",
            "data": trends
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Defect Tracking Endpoints
# ============================================================================

@router.get("/defects", summary="Get defects")
async def get_defects(
    severity: Optional[List[DefectSeverity]] = Query(None),
    status: Optional[List[DefectStatus]] = Query(None),
    assignee_id: Optional[str] = Query(None),
    component: Optional[str] = Query(None),
    created_after: Optional[datetime] = Query(None),
    created_before: Optional[datetime] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user = Depends(get_current_user)
):
    """Get defects with filtering and pagination"""
    try:
        filters = {}
        if severity:
            filters["severity"] = [s.value for s in severity]
        if status:
            filters["status"] = [s.value for s in status]
        if assignee_id:
            filters["assignee_id"] = assignee_id
        if component:
            filters["component"] = component
        if created_after:
            filters["created_after"] = created_after
        if created_before:
            filters["created_before"] = created_before
        
        defects = await defect_tracker.get_defects(
            filters=filters,
            limit=limit,
            offset=offset
        )
        
        defects_data = [
            {
                "id": d.id,
                "title": d.title,
                "description": d.description,
                "severity": d.severity.value,
                "priority": d.priority.value,
                "status": d.status.value,
                "type": d.type.value,
                "source": d.source.value,
                "reporter_id": d.reporter_id,
                "assignee_id": d.assignee_id,
                "component": d.component,
                "tags": d.tags,
                "created_at": d.created_at.isoformat(),
                "updated_at": d.updated_at.isoformat(),
                "resolved_at": d.resolved_at.isoformat() if d.resolved_at else None,
                "time_to_resolution": d.time_to_resolution,
                "reopened_count": d.reopened_count
            }
            for d in defects
        ]
        
        return {
            "status": "success",
            "data": {
                "defects": defects_data,
                "count": len(defects_data),
                "has_more": len(defects_data) == limit
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/defects/statistics", summary="Get defect statistics")
async def get_defect_statistics(
    current_user = Depends(get_current_user)
):
    """Get comprehensive defect statistics"""
    try:
        stats = await defect_tracker.get_defect_statistics()
        return {
            "status": "success",
            "data": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/defects/{defect_id}", summary="Get defect details")
async def get_defect(
    defect_id: str,
    current_user = Depends(get_current_user)
):
    """Get detailed information about a specific defect"""
    try:
        defect = await defect_tracker.get_defect(defect_id)
        if not defect:
            raise HTTPException(status_code=404, detail="Defect not found")
        
        return {
            "status": "success",
            "data": {
                "id": defect.id,
                "title": defect.title,
                "description": defect.description,
                "severity": defect.severity.value,
                "priority": defect.priority.value,
                "status": defect.status.value,
                "type": defect.type.value,
                "source": defect.source.value,
                "reporter_id": defect.reporter_id,
                "assignee_id": defect.assignee_id,
                "component": defect.component,
                "tags": defect.tags,
                "created_at": defect.created_at.isoformat(),
                "updated_at": defect.updated_at.isoformat(),
                "resolved_at": defect.resolved_at.isoformat() if defect.resolved_at else None,
                "closed_at": defect.closed_at.isoformat() if defect.closed_at else None,
                "resolution": defect.resolution,
                "resolution_notes": defect.resolution_notes,
                "fix_version": defect.fix_version,
                "time_to_resolution": defect.time_to_resolution,
                "reopened_count": defect.reopened_count,
                "duplicate_of": defect.duplicate_of,
                "metadata": defect.metadata.__dict__ if defect.metadata else {},
                "attachments": defect.attachments,
                "comments": defect.comments
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Dashboard and Reporting Endpoints
# ============================================================================

@router.get("/dashboard", summary="Get quality dashboard")
async def get_quality_dashboard(
    time_range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"),
    include_trends: bool = Query(True),
    include_charts: bool = Query(False),
    current_user = Depends(get_current_user)
):
    """Get comprehensive quality dashboard"""
    try:
        # Get overview
        overview = await quality_dashboard.generate_quality_overview()
        
        # Get trends if requested
        trends = None
        if include_trends:
            days_map = {"1h": 1, "24h": 1, "7d": 7, "30d": 30}
            days = days_map.get(time_range, 7)
            trends = await quality_dashboard.generate_trend_analysis(days=days)
        
        # Get recent defects
        recent_defects = await defect_tracker.get_defects(
            filters={"created_after": datetime.now(timezone.utc) - timedelta(days=7)},
            limit=10
        )
        
        dashboard_data = {
            "overview": overview,
            "trends": trends,
            "recent_defects": [
                {
                    "id": d.id,
                    "title": d.title,
                    "severity": d.severity.value,
                    "status": d.status.value,
                    "created_at": d.created_at.isoformat()
                }
                for d in recent_defects
            ],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "time_range": time_range
        }
        
        return {
            "status": "success",
            "data": dashboard_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reports", summary="Get available reports")
async def get_available_reports(
    current_user = Depends(get_current_user)
):
    """Get list of available quality reports"""
    try:
        reports = [
            {
                "id": "daily_summary",
                "name": "Daily Summary",
                "description": "Daily overview of system health and metrics",
                "schedule": "Daily at 8:00 AM"
            },
            {
                "id": "weekly_quality",
                "name": "Weekly Quality Report",
                "description": "Comprehensive weekly quality analysis",
                "schedule": "Weekly on Monday at 9:00 AM"
            },
            {
                "id": "monthly_executive",
                "name": "Monthly Executive Report",
                "description": "Executive summary for business stakeholders",
                "schedule": "Monthly on 1st at 10:00 AM"
            }
        ]
        
        return {
            "status": "success",
            "data": {
                "reports": reports,
                "count": len(reports)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reports/{report_type}/generate", summary="Generate report on-demand")
async def generate_report(
    report_type: str,
    format: str = Query("json", description="Report format: json, html, pdf"),
    background_tasks: BackgroundTasks = None,
    current_user = Depends(get_current_user)
):
    """Generate a quality report on-demand"""
    try:
        # Set up reporting system
        reporting_system = setup_automated_reporting(
            metrics_collector,
            quality_dashboard,
            defect_tracker
        )
        
        # Generate report based on type
        if report_type == "daily_summary":
            report_data = await reporting_system.generator.generate_daily_summary(datetime.now())
        elif report_type == "weekly_quality":
            week_start = datetime.now() - timedelta(days=7)
            report_data = await reporting_system.generator.generate_weekly_quality_report(week_start)
        elif report_type == "monthly_executive":
            month_start = datetime.now().replace(day=1)
            report_data = await reporting_system.generator.generate_monthly_executive_report(month_start)
        else:
            raise HTTPException(status_code=404, detail="Report type not found")
        
        # Format report based on requested format
        if format == "html" and report_type in reporting_system.generator.templates:
            template = reporting_system.generator.templates[report_type]
            from jinja2 import Template
            jinja_template = Template(template.html_template)
            html_content = jinja_template.render(
                css_styles=template.css_styles,
                **report_data.get('data', {})
            )
            report_data['html_content'] = html_content
        
        return {
            "status": "success",
            "message": f"Report generated successfully",
            "data": report_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", summary="Quality system health check")
async def quality_health_check():
    """Check health of quality monitoring system"""
    try:
        # Check if monitoring is running
        monitoring_status = "running" if quality_monitor.monitoring_tasks else "stopped"
        
        # Get latest metrics count
        latest_metrics = await metrics_collector.get_latest_metrics()
        metrics_count = len(latest_metrics)
        
        # Check data freshness
        newest_metric_time = None
        if latest_metrics:
            newest_metric_time = max(mv.timestamp for mv in latest_metrics.values())
            data_freshness_minutes = (datetime.now(timezone.utc) - newest_metric_time).total_seconds() / 60
        else:
            data_freshness_minutes = float('inf')
        
        health_status = {
            "status": "healthy",
            "monitoring_status": monitoring_status,
            "metrics_count": metrics_count,
            "data_freshness_minutes": data_freshness_minutes,
            "checks": {
                "monitoring_running": monitoring_status == "running",
                "metrics_available": metrics_count > 0,
                "data_fresh": data_freshness_minutes < 60  # Less than 1 hour old
            }
        }
        
        # Overall health
        if not all(health_status["checks"].values()):
            health_status["status"] = "unhealthy"
        
        return {
            "status": "success",
            "data": health_status
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "data": {"status": "unhealthy"}
        }

# ============================================================================
# Control Endpoints
# ============================================================================

@router.post("/monitoring/start", summary="Start quality monitoring")
async def start_monitoring(
    current_user = Depends(get_current_user)
):
    """Start automated quality monitoring"""
    try:
        await quality_monitor.start_monitoring()
        return {
            "status": "success",
            "message": "Quality monitoring started"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/monitoring/stop", summary="Stop quality monitoring")
async def stop_monitoring(
    current_user = Depends(get_current_user)
):
    """Stop automated quality monitoring"""
    try:
        await quality_monitor.stop_monitoring()
        return {
            "status": "success",
            "message": "Quality monitoring stopped"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))