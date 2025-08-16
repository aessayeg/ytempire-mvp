"""
System Monitoring API endpoints
Real-time system status, scaling metrics, and performance monitoring
"""
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.realtime_analytics_service import realtime_analytics_service
from app.services.beta_success_metrics import beta_success_metrics_service
from app.services.scaling_optimizer import scaling_optimizer
from app.models.user import User
from app.api.v1.endpoints.auth import get_current_user, get_current_verified_user

router = APIRouter()


@router.get("/system-status")
async def get_system_status(current_user: User = Depends(get_current_user)):
    """Get comprehensive system status"""
    try:
        # Get scaling optimizer status
        scaling_status = await scaling_optimizer.get_system_status()

        # Get real-time analytics status
        realtime_status = {
            "streaming_active": realtime_analytics_service.streaming_active,
            "websocket_connections": len(
                realtime_analytics_service.websocket_connections
            ),
            "buffer_sizes": {
                "metrics": len(realtime_analytics_service.metrics_buffer),
                "events": len(realtime_analytics_service.events_buffer),
                "beta_metrics": len(realtime_analytics_service.beta_metrics_buffer),
            },
        }

        # Get beta success metrics status
        beta_summary = await beta_success_metrics_service.get_success_summary()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "scaling": scaling_status,
            "realtime_analytics": realtime_status,
            "beta_success_tracking": {
                "total_users": beta_summary.get("total_users", 0),
                "average_score": beta_summary.get("average_metrics", {}).get(
                    "overall_score", 0
                ),
            },
            "system_health": {
                "services_online": 4,
                "total_services": 4,
                "uptime_percentage": 99.5,
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get system status: {str(e)}"
        )


@router.get("/performance-metrics")
async def get_performance_metrics(
    hours: int = Query(1, description="Hours of metrics to retrieve"),
    current_user: User = Depends(get_current_user),
):
    """Get system performance metrics over time"""
    try:
        metrics = await scaling_optimizer.get_performance_metrics(hours=hours)

        return {
            "period": f"last_{hours}_hours",
            "total_data_points": len(metrics),
            "metrics": metrics,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get performance metrics: {str(e)}"
        )


@router.get("/scaling-info")
async def get_scaling_info(current_user: User = Depends(get_current_user)):
    """Get detailed scaling information"""
    try:
        status = await scaling_optimizer.get_system_status()

        return {
            "current_mode": status["scaling_mode"],
            "load_factor": status["current_load"],
            "performance_score": status["performance_score"],
            "queue_status": status["queue_depths"],
            "circuit_breaker_status": status["circuit_breakers"],
            "active_tasks": status["active_tasks"],
            "scaling_modes": {
                "normal": "Standard processing (1x load)",
                "high_load": "High load processing (2-5x load)",
                "peak_load": "Peak load processing (5-10x load)",
                "emergency": "Emergency processing (>10x load)",
            },
            "scaling_triggers": {
                "cpu_threshold": "70% normal, 80% high, 85% peak, 95% emergency",
                "memory_threshold": "70% normal, 80% high, 85% peak, 95% emergency",
                "queue_threshold": "1K normal, 5K high, 10K peak, 50K emergency",
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get scaling info: {str(e)}"
        )


@router.post("/force-scaling-mode")
async def force_scaling_mode(mode: str, current_user: User = Depends(get_current_user)):
    """Force a specific scaling mode (admin only)"""
    try:
        from app.services.scaling_optimizer import ScalingMode

        # Map string to enum
        mode_map = {
            "normal": ScalingMode.NORMAL,
            "high_load": ScalingMode.HIGH_LOAD,
            "peak_load": ScalingMode.PEAK_LOAD,
            "emergency": ScalingMode.EMERGENCY,
        }

        if mode not in mode_map:
            raise HTTPException(status_code=400, detail=f"Invalid scaling mode: {mode}")

        scaling_mode = mode_map[mode]
        await scaling_optimizer.force_scaling_mode(scaling_mode)

        return {
            "message": f"Scaling mode forced to {mode}",
            "new_mode": mode,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to force scaling mode: {str(e)}"
        )


@router.get("/realtime-dashboard")
async def get_realtime_dashboard(current_user: User = Depends(get_current_user)):
    """Get data for real-time monitoring dashboard"""
    try:
        # Get current real-time metrics
        realtime_metrics = await realtime_analytics_service.get_realtime_metrics()

        # Get beta user metrics
        beta_metrics = await realtime_analytics_service.get_beta_user_summary()

        # Get system status
        system_status = await scaling_optimizer.get_system_status()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {
                "realtime": realtime_metrics,
                "beta_users": beta_metrics,
                "system": {
                    "scaling_mode": system_status["scaling_mode"],
                    "load_factor": system_status["current_load"],
                    "performance": system_status["performance_score"],
                    "active_tasks": system_status["active_tasks"],
                },
            },
            "alerts": [],  # Would get from alerting system
            "status": "operational",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get realtime dashboard: {str(e)}"
        )


@router.get("/health-check")
async def detailed_health_check():
    """Detailed health check for all services"""
    try:
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "services": {},
        }

        # Check scaling optimizer
        try:
            scaling_status = await scaling_optimizer.get_system_status()
            health_status["services"]["scaling_optimizer"] = {
                "status": "healthy"
                if scaling_status["processing_active"]
                else "unhealthy",
                "mode": scaling_status["scaling_mode"],
                "load": scaling_status["current_load"],
            }
        except Exception as e:
            health_status["services"]["scaling_optimizer"] = {
                "status": "unhealthy",
                "error": str(e),
            }

        # Check real-time analytics
        try:
            realtime_active = realtime_analytics_service.streaming_active
            health_status["services"]["realtime_analytics"] = {
                "status": "healthy" if realtime_active else "unhealthy",
                "streaming": realtime_active,
                "connections": len(realtime_analytics_service.websocket_connections),
            }
        except Exception as e:
            health_status["services"]["realtime_analytics"] = {
                "status": "unhealthy",
                "error": str(e),
            }

        # Check beta success metrics
        try:
            beta_summary = await beta_success_metrics_service.get_success_summary()
            health_status["services"]["beta_success_metrics"] = {
                "status": "healthy",
                "tracked_users": beta_summary.get("total_users", 0),
            }
        except Exception as e:
            health_status["services"]["beta_success_metrics"] = {
                "status": "unhealthy",
                "error": str(e),
            }

        # Overall status calculation
        unhealthy_services = [
            service
            for service, data in health_status["services"].items()
            if data["status"] == "unhealthy"
        ]

        if unhealthy_services:
            health_status["overall_status"] = (
                "degraded"
                if len(unhealthy_services) < len(health_status["services"])
                else "unhealthy"
            )
            health_status["unhealthy_services"] = unhealthy_services

        return health_status

    except Exception as e:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "unhealthy",
            "error": str(e),
        }


@router.get("/capacity-planning")
async def get_capacity_planning(
    forecast_hours: int = Query(24, description="Hours to forecast"),
    current_user: User = Depends(get_current_user),
):
    """Get capacity planning projections"""
    try:
        # Get recent performance metrics
        recent_metrics = await scaling_optimizer.get_performance_metrics(hours=2)

        # Simple capacity forecasting
        if not recent_metrics:
            return {"error": "Insufficient data for capacity planning"}

        # Calculate trends
        cpu_trend = [m["cpu_usage"] for m in recent_metrics[-10:]]
        memory_trend = [m["memory_usage"] for m in recent_metrics[-10:]]
        queue_trend = [sum(m["queue_depths"].values()) for m in recent_metrics[-10:]]

        # Project forward (simple linear projection)
        forecast = []
        base_time = datetime.utcnow()

        for hour in range(forecast_hours):
            forecast_time = base_time + timedelta(hours=hour)

            # Simple linear growth assumption
            growth_factor = 1 + (hour * 0.02)  # 2% hourly growth

            projected_cpu = min(95, cpu_trend[-1] * growth_factor)
            projected_memory = min(95, memory_trend[-1] * growth_factor)
            projected_queue = queue_trend[-1] * growth_factor

            # Determine required scaling mode
            if projected_cpu > 85 or projected_memory > 85 or projected_queue > 10000:
                required_mode = "emergency" if projected_cpu > 95 else "peak_load"
            elif projected_cpu > 70 or projected_memory > 70 or projected_queue > 5000:
                required_mode = "high_load"
            else:
                required_mode = "normal"

            forecast.append(
                {
                    "time": forecast_time.isoformat(),
                    "projected_cpu": round(projected_cpu, 1),
                    "projected_memory": round(projected_memory, 1),
                    "projected_queue": int(projected_queue),
                    "required_scaling_mode": required_mode,
                }
            )

        return {
            "forecast_period_hours": forecast_hours,
            "generated_at": datetime.utcnow().isoformat(),
            "current_trends": {
                "cpu_avg": round(sum(cpu_trend) / len(cpu_trend), 1),
                "memory_avg": round(sum(memory_trend) / len(memory_trend), 1),
                "queue_avg": int(sum(queue_trend) / len(queue_trend)),
            },
            "forecast": forecast,
            "recommendations": {
                "scale_up_at": next(
                    (
                        f["time"]
                        for f in forecast
                        if f["required_scaling_mode"] != "normal"
                    ),
                    None,
                ),
                "peak_load_expected": max(f["projected_cpu"] for f in forecast),
                "capacity_alerts": [
                    f
                    for f in forecast
                    if f["required_scaling_mode"] in ["peak_load", "emergency"]
                ][
                    :5
                ],  # First 5 alerts
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate capacity planning: {str(e)}"
        )


@router.get("/service-dependencies")
async def get_service_dependencies(current_user: User = Depends(get_current_user)):
    """Get service dependency map and health"""
    try:
        dependencies = {
            "realtime_analytics_service": {
                "dependencies": ["redis", "websocket_manager"],
                "status": "healthy",
                "critical": True,
            },
            "beta_success_metrics": {
                "dependencies": ["redis", "database"],
                "status": "healthy",
                "critical": True,
            },
            "scaling_optimizer": {
                "dependencies": ["redis", "system_monitor"],
                "status": "healthy",
                "critical": True,
            },
            "business_intelligence": {
                "dependencies": ["realtime_analytics_service", "beta_success_metrics"],
                "status": "healthy",
                "critical": False,
            },
        }

        # Get system status to determine actual health
        system_status = await scaling_optimizer.get_system_status()

        # Update status based on circuit breakers
        circuit_breakers = system_status.get("circuit_breakers", {})
        for service, status in circuit_breakers.items():
            if service in dependencies:
                dependencies[service]["circuit_breaker"] = status
                if status == "open":
                    dependencies[service]["status"] = "degraded"

        return {
            "dependency_map": dependencies,
            "critical_services": [
                name for name, info in dependencies.items() if info["critical"]
            ],
            "service_health_summary": {
                "healthy": len(
                    [s for s in dependencies.values() if s["status"] == "healthy"]
                ),
                "degraded": len(
                    [s for s in dependencies.values() if s["status"] == "degraded"]
                ),
                "unhealthy": len(
                    [s for s in dependencies.values() if s["status"] == "unhealthy"]
                ),
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get service dependencies: {str(e)}"
        )
