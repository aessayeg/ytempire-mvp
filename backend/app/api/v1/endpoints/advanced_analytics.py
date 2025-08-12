"""
Advanced Analytics Pipeline API Endpoints
Real-time streaming, predictive models, and GDPR compliance
"""

from typing import List, Dict, Any, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from enum import Enum
import uuid
import asyncio

from app.api.v1.endpoints.auth import get_current_verified_user
from app.models.user import User
from app.services.analytics_pipeline import (
    AdvancedAnalyticsPipeline, analytics_pipeline,
    StreamProcessor, MetricsAggregator, PredictiveModels,
    GDPRComplianceManager
)
from app.db.session import get_db

router = APIRouter()


class AnalyticsRequestType(str, Enum):
    TREND_ANALYSIS = "trend_analysis"
    PERFORMANCE_METRICS = "performance_metrics" 
    AUDIENCE_INSIGHTS = "audience_insights"
    REVENUE_FORECASTING = "revenue_forecasting"
    ANOMALY_DETECTION = "anomaly_detection"
    COMPETITIVE_ANALYSIS = "competitive_analysis"


class StreamingMode(str, Enum):
    REAL_TIME = "real_time"
    BATCH = "batch"
    MICRO_BATCH = "micro_batch"


class AnalyticsRequest(BaseModel):
    """Request for analytics processing"""
    request_type: AnalyticsRequestType = Field(..., description="Type of analytics request")
    channel_ids: List[str] = Field(..., description="Channel IDs to analyze")
    date_range: Dict[str, str] = Field(..., description="Start and end dates")
    metrics: List[str] = Field(default=[], description="Specific metrics to include")
    streaming_mode: StreamingMode = Field(StreamingMode.BATCH, description="Processing mode")
    enable_predictions: bool = Field(False, description="Enable predictive analysis")
    anonymize_data: bool = Field(True, description="Apply data anonymization")
    aggregation_level: str = Field("daily", description="Aggregation level: hourly, daily, weekly")
    
    @validator('date_range')
    def validate_date_range(cls, v):
        if 'start_date' not in v or 'end_date' not in v:
            raise ValueError('Date range must include start_date and end_date')
        return v


class StreamingAnalyticsRequest(BaseModel):
    """Request for real-time streaming analytics"""
    stream_id: str = Field(..., description="Unique stream identifier")
    channel_ids: List[str] = Field(..., description="Channels to stream")
    metrics: List[str] = Field(..., description="Metrics to stream")
    stream_duration_minutes: int = Field(60, ge=1, le=1440, description="Stream duration")
    update_interval_seconds: int = Field(30, ge=10, le=300, description="Update interval")
    enable_alerts: bool = Field(True, description="Enable anomaly alerts")
    alert_thresholds: Dict[str, float] = Field(default={}, description="Custom alert thresholds")


class PredictiveAnalyticsRequest(BaseModel):
    """Request for predictive analytics"""
    model_type: str = Field(..., description="Model type: trend, growth, anomaly")
    historical_days: int = Field(90, ge=30, le=365, description="Historical data days")
    forecast_days: int = Field(30, ge=7, le=90, description="Forecast period days")
    confidence_level: float = Field(0.95, ge=0.8, le=0.99, description="Confidence level")
    include_seasonality: bool = Field(True, description="Include seasonal patterns")
    external_factors: Dict[str, Any] = Field(default={}, description="External factors")


class GDPRDataRequest(BaseModel):
    """GDPR data request"""
    request_type: str = Field(..., description="export, delete, anonymize")
    user_email: str = Field(..., description="User email for GDPR request")
    data_categories: List[str] = Field(default=[], description="Specific data categories")
    reason: str = Field(..., description="Reason for request")


@router.post("/process")
async def process_analytics(
    request: AnalyticsRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Process advanced analytics request with real-time streaming support
    """
    try:
        # Generate processing ID
        processing_id = f"analytics_{uuid.uuid4().hex[:8]}"
        
        # Process analytics based on mode
        if request.streaming_mode == StreamingMode.REAL_TIME:
            # Start real-time streaming
            result = await analytics_pipeline.start_real_time_stream(
                processing_id=processing_id,
                channel_ids=request.channel_ids,
                metrics=request.metrics,
                user_id=str(current_user.id)
            )
        else:
            # Process in batch mode
            result = await analytics_pipeline.process_analytics_request(
                request_type=request.request_type,
                channel_ids=request.channel_ids,
                date_range=request.date_range,
                metrics=request.metrics,
                aggregation_level=request.aggregation_level,
                enable_predictions=request.enable_predictions,
                anonymize_data=request.anonymize_data,
                user_id=str(current_user.id)
            )
        
        return {
            "success": True,
            "processing_id": processing_id,
            "request_type": request.request_type,
            "streaming_mode": request.streaming_mode,
            "estimated_completion": datetime.utcnow() + timedelta(minutes=5),
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analytics processing failed: {str(e)}"
        )


@router.post("/stream/start")
async def start_streaming_analytics(
    request: StreamingAnalyticsRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_verified_user)
):
    """
    Start real-time analytics streaming with anomaly detection
    """
    try:
        # Start streaming session
        stream_config = {
            "stream_id": request.stream_id,
            "channel_ids": request.channel_ids,
            "metrics": request.metrics,
            "update_interval": request.update_interval_seconds,
            "duration": request.stream_duration_minutes,
            "user_id": str(current_user.id),
            "enable_alerts": request.enable_alerts,
            "alert_thresholds": request.alert_thresholds
        }
        
        stream_session = await analytics_pipeline.stream_processor.start_stream(stream_config)
        
        return {
            "success": True,
            "stream_id": request.stream_id,
            "message": "Real-time analytics streaming started",
            "stream_session": stream_session,
            "websocket_endpoint": f"/ws/analytics/{request.stream_id}",
            "estimated_end_time": datetime.utcnow() + timedelta(minutes=request.stream_duration_minutes)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start streaming: {str(e)}"
        )


@router.post("/stream/stop/{stream_id}")
async def stop_streaming_analytics(
    stream_id: str,
    current_user: User = Depends(get_current_verified_user)
):
    """
    Stop real-time analytics streaming
    """
    try:
        result = await analytics_pipeline.stream_processor.stop_stream(
            stream_id=stream_id,
            user_id=str(current_user.id)
        )
        
        return {
            "success": True,
            "stream_id": stream_id,
            "message": "Analytics streaming stopped",
            "final_metrics": result
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop streaming: {str(e)}"
        )


@router.post("/predict")
async def run_predictive_analytics(
    request: PredictiveAnalyticsRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_verified_user)
):
    """
    Run predictive analytics models for trend forecasting
    """
    try:
        # Run predictive models
        predictions = await analytics_pipeline.predictive_models.generate_predictions(
            model_type=request.model_type,
            historical_days=request.historical_days,
            forecast_days=request.forecast_days,
            confidence_level=request.confidence_level,
            include_seasonality=request.include_seasonality,
            external_factors=request.external_factors,
            user_id=str(current_user.id)
        )
        
        return {
            "success": True,
            "model_type": request.model_type,
            "forecast_period": f"{request.forecast_days} days",
            "confidence_level": request.confidence_level,
            "predictions": predictions,
            "model_accuracy": predictions.get("accuracy_metrics", {}),
            "trend_analysis": predictions.get("trend_indicators", {})
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Predictive analytics failed: {str(e)}"
        )


@router.get("/anomalies")
async def detect_anomalies(
    channel_ids: List[str] = Query(..., description="Channel IDs to check"),
    hours_back: int = Query(24, ge=1, le=168, description="Hours to look back"),
    sensitivity: float = Query(0.95, ge=0.8, le=0.99, description="Detection sensitivity"),
    current_user: User = Depends(get_current_verified_user)
):
    """
    Detect anomalies in channel performance using ML models
    """
    try:
        anomalies = await analytics_pipeline.anomaly_detector.detect_anomalies(
            channel_ids=channel_ids,
            lookback_hours=hours_back,
            sensitivity=sensitivity,
            user_id=str(current_user.id)
        )
        
        return {
            "success": True,
            "detection_period": f"{hours_back} hours",
            "sensitivity_level": sensitivity,
            "anomalies_found": len(anomalies),
            "anomalies": anomalies,
            "summary": {
                "critical": len([a for a in anomalies if a.get("severity") == "critical"]),
                "warning": len([a for a in anomalies if a.get("severity") == "warning"]),
                "info": len([a for a in anomalies if a.get("severity") == "info"])
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Anomaly detection failed: {str(e)}"
        )


@router.get("/metrics/aggregated")
async def get_aggregated_metrics(
    channel_ids: List[str] = Query(..., description="Channel IDs"),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    aggregation: str = Query("daily", description="hourly, daily, weekly, monthly"),
    metrics: List[str] = Query(default=[], description="Specific metrics"),
    current_user: User = Depends(get_current_verified_user)
):
    """
    Get aggregated analytics metrics with advanced processing
    """
    try:
        aggregated_data = await analytics_pipeline.metrics_aggregator.aggregate_metrics(
            channel_ids=channel_ids,
            start_date=start_date,
            end_date=end_date,
            aggregation_level=aggregation,
            metrics=metrics or ["views", "subscribers", "revenue", "engagement"],
            user_id=str(current_user.id)
        )
        
        return {
            "success": True,
            "date_range": f"{start_date} to {end_date}",
            "aggregation_level": aggregation,
            "channels_analyzed": len(channel_ids),
            "data_points": len(aggregated_data.get("time_series", [])),
            "aggregated_metrics": aggregated_data,
            "summary_statistics": {
                "total_views": sum(d.get("views", 0) for d in aggregated_data.get("time_series", [])),
                "average_engagement": aggregated_data.get("avg_engagement_rate", 0),
                "growth_rate": aggregated_data.get("growth_indicators", {}).get("overall_growth", 0)
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metrics aggregation failed: {str(e)}"
        )


@router.post("/gdpr/request")
async def handle_gdpr_request(
    request: GDPRDataRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_verified_user)
):
    """
    Handle GDPR data requests (export, delete, anonymize)
    """
    try:
        request_id = f"gdpr_{uuid.uuid4().hex[:8]}"
        
        # Process GDPR request
        result = await analytics_pipeline.gdpr_manager.process_data_request(
            request_id=request_id,
            request_type=request.request_type,
            user_email=request.user_email,
            data_categories=request.data_categories,
            reason=request.reason,
            requester_id=str(current_user.id)
        )
        
        return {
            "success": True,
            "request_id": request_id,
            "request_type": request.request_type,
            "status": "processing",
            "estimated_completion": datetime.utcnow() + timedelta(hours=24),
            "result": result,
            "compliance_notes": "Request processed according to GDPR requirements"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"GDPR request failed: {str(e)}"
        )


@router.get("/performance/stats")
async def get_analytics_performance_stats(
    current_user: User = Depends(get_current_verified_user)
):
    """
    Get analytics pipeline performance statistics
    """
    try:
        stats = analytics_pipeline.get_performance_stats()
        
        return {
            "success": True,
            "performance_stats": stats,
            "pipeline_health": {
                "real_time_streams": stats.get("active_streams", 0),
                "processing_queue": stats.get("queue_size", 0),
                "model_accuracy": stats.get("model_performance", {}),
                "data_throughput": stats.get("throughput_metrics", {})
            },
            "capabilities": {
                "real_time_streaming": "Real-time analytics with <1s latency",
                "predictive_models": "ML-based trend forecasting",
                "anomaly_detection": "Automated anomaly detection",
                "gdpr_compliance": "Full GDPR compliance with encryption",
                "data_aggregation": "Multi-level metric aggregation",
                "stream_processing": "High-throughput stream processing"
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance stats: {str(e)}"
        )


@router.get("/streams/active")
async def get_active_streams(
    current_user: User = Depends(get_current_verified_user)
):
    """
    Get list of active analytics streams for the user
    """
    try:
        active_streams = await analytics_pipeline.stream_processor.get_user_streams(
            user_id=str(current_user.id)
        )
        
        return {
            "success": True,
            "active_streams": len(active_streams),
            "streams": active_streams,
            "total_data_points": sum(s.get("data_points", 0) for s in active_streams)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get active streams: {str(e)}"
        )


@router.delete("/cache/clear")
async def clear_analytics_cache(
    cache_type: str = Query("all", description="all, metrics, predictions, aggregations"),
    current_user: User = Depends(get_current_verified_user)
):
    """
    Clear analytics cache for fresh data processing
    """
    try:
        cleared_items = await analytics_pipeline.clear_cache(
            cache_type=cache_type,
            user_id=str(current_user.id)
        )
        
        return {
            "success": True,
            "message": f"Cache cleared: {cache_type}",
            "cleared_items": cleared_items
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cache clear failed: {str(e)}"
        )