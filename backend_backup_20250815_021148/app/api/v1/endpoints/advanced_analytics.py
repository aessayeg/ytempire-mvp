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
    MetricType, AggregationLevel, DataSource, MetricPoint
)
from app.services.cost_tracking import cost_tracker
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


@router.get("/revenue/analytics")
async def get_revenue_analytics(
    channel_ids: List[str] = Query(..., description="Channel IDs to analyze"),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    breakdown_by: str = Query("channel", description="channel, video, source, geography"),
    include_forecasting: bool = Query(False, description="Include revenue forecasting"),
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get comprehensive revenue analytics with forecasting
    """
    try:
        revenue_data = await analytics_pipeline.revenue_analyzer.get_revenue_analytics(
            channel_ids=channel_ids,
            start_date=start_date,
            end_date=end_date,
            breakdown_by=breakdown_by,
            user_id=str(current_user.id)
        )
        
        # Add forecasting if requested
        forecast_data = None
        if include_forecasting:
            forecast_data = await analytics_pipeline.revenue_analyzer.forecast_revenue(
                channel_ids=channel_ids,
                historical_data=revenue_data,
                forecast_days=30
            )
        
        return {
            "success": True,
            "date_range": f"{start_date} to {end_date}",
            "breakdown_by": breakdown_by,
            "revenue_analytics": revenue_data,
            "forecast": forecast_data,
            "summary": {
                "total_revenue": revenue_data.get("total_revenue", 0),
                "average_daily_revenue": revenue_data.get("avg_daily_revenue", 0),
                "top_performing_channel": revenue_data.get("top_channel", {}),
                "revenue_growth_rate": revenue_data.get("growth_rate", 0),
                "projected_monthly_revenue": forecast_data.get("monthly_projection", 0) if forecast_data else None
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Revenue analytics failed: {str(e)}"
        )


@router.get("/channels/comparison")
async def compare_channel_performance(
    channel_ids: List[str] = Query(..., description="Channel IDs to compare (2-10 channels)"),
    comparison_period: str = Query("30d", description="7d, 30d, 90d, 1y"),
    metrics: List[str] = Query(default=["views", "subscribers", "revenue", "engagement"], description="Metrics to compare"),
    include_benchmarks: bool = Query(True, description="Include industry benchmarks"),
    current_user: User = Depends(get_current_verified_user)
):
    """
    Compare performance across multiple channels with benchmarking
    """
    try:
        if len(channel_ids) < 2 or len(channel_ids) > 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Please provide 2-10 channels for comparison"
            )
        
        comparison_data = await analytics_pipeline.channel_comparator.compare_channels(
            channel_ids=channel_ids,
            period=comparison_period,
            metrics=metrics,
            user_id=str(current_user.id)
        )
        
        # Add industry benchmarks if requested
        benchmark_data = None
        if include_benchmarks:
            benchmark_data = await analytics_pipeline.benchmark_service.get_industry_benchmarks(
                channel_ids=channel_ids,
                metrics=metrics
            )
        
        return {
            "success": True,
            "comparison_period": comparison_period,
            "channels_compared": len(channel_ids),
            "metrics_analyzed": metrics,
            "comparison_data": comparison_data,
            "industry_benchmarks": benchmark_data,
            "performance_ranking": comparison_data.get("channel_rankings", []),
            "insights": {
                "best_performer": comparison_data.get("top_performer", {}),
                "fastest_growing": comparison_data.get("fastest_growing", {}),
                "highest_engagement": comparison_data.get("highest_engagement", {}),
                "improvement_opportunities": comparison_data.get("improvement_suggestions", [])
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Channel comparison failed: {str(e)}"
        )


@router.get("/costs/analysis")
async def get_cost_analysis(
    time_period: str = Query("30d", description="7d, 30d, 90d, 1y"),
    breakdown_by: str = Query("service", description="service, channel, video, date"),
    include_optimization: bool = Query(True, description="Include cost optimization suggestions"),
    cost_threshold: float = Query(3.0, description="Cost per video threshold for alerts"),
    current_user: User = Depends(get_current_verified_user)
):
    """
    Get detailed cost analysis with optimization recommendations
    """
    try:
        cost_data = await analytics_pipeline.cost_analyzer.analyze_costs(
            user_id=str(current_user.id),
            time_period=time_period,
            breakdown_by=breakdown_by,
            threshold=cost_threshold
        )
        
        # Generate optimization suggestions if requested
        optimization_suggestions = None
        if include_optimization:
            optimization_suggestions = await analytics_pipeline.cost_optimizer.generate_suggestions(
                cost_data=cost_data,
                target_reduction=0.3,  # 30% cost reduction target
                user_id=str(current_user.id)
            )
        
        return {
            "success": True,
            "analysis_period": time_period,
            "breakdown_by": breakdown_by,
            "cost_analysis": cost_data,
            "optimization_suggestions": optimization_suggestions,
            "summary": {
                "total_costs": cost_data.get("total_costs", 0),
                "average_cost_per_video": cost_data.get("avg_cost_per_video", 0),
                "cost_trend": cost_data.get("trend", "stable"),
                "videos_over_threshold": cost_data.get("videos_over_threshold", 0),
                "potential_savings": optimization_suggestions.get("total_potential_savings", 0) if optimization_suggestions else 0
            },
            "alerts": {
                "high_cost_videos": cost_data.get("high_cost_alerts", []),
                "budget_warnings": cost_data.get("budget_warnings", []),
                "anomalies": cost_data.get("cost_anomalies", [])
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cost analysis failed: {str(e)}"
        )


@router.get("/predictive/revenue")
async def get_predictive_revenue_analytics(
    channel_ids: List[str] = Query(..., description="Channel IDs for prediction"),
    forecast_period: int = Query(30, ge=7, le=365, description="Days to forecast"),
    confidence_level: float = Query(0.95, ge=0.8, le=0.99, description="Prediction confidence"),
    include_scenarios: bool = Query(True, description="Include best/worst case scenarios"),
    external_factors: Optional[Dict[str, Any]] = None,
    current_user: User = Depends(get_current_verified_user)
):
    """
    Generate predictive revenue analytics with scenario modeling
    """
    try:
        predictions = await analytics_pipeline.revenue_predictor.generate_revenue_forecast(
            channel_ids=channel_ids,
            forecast_days=forecast_period,
            confidence_level=confidence_level,
            external_factors=external_factors or {},
            user_id=str(current_user.id)
        )
        
        # Generate scenarios if requested
        scenario_data = None
        if include_scenarios:
            scenario_data = await analytics_pipeline.scenario_modeler.generate_scenarios(
                base_prediction=predictions,
                scenario_types=["optimistic", "pessimistic", "realistic"]
            )
        
        return {
            "success": True,
            "forecast_period": f"{forecast_period} days",
            "confidence_level": confidence_level,
            "channels_analyzed": len(channel_ids),
            "revenue_predictions": predictions,
            "scenario_analysis": scenario_data,
            "key_insights": {
                "predicted_total_revenue": predictions.get("total_predicted_revenue", 0),
                "growth_trajectory": predictions.get("growth_trend", "stable"),
                "peak_revenue_period": predictions.get("peak_period", {}),
                "risk_factors": predictions.get("risk_indicators", []),
                "confidence_score": predictions.get("model_confidence", 0)
            },
            "recommendations": {
                "optimization_opportunities": predictions.get("optimization_suggestions", []),
                "investment_recommendations": predictions.get("investment_advice", []),
                "risk_mitigation": predictions.get("risk_mitigation_strategies", [])
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Predictive revenue analytics failed: {str(e)}"
        )


@router.get("/competitive/analysis")
async def get_competitive_analysis(
    channel_ids: List[str] = Query(..., description="Your channel IDs"),
    competitor_keywords: List[str] = Query(..., description="Keywords to find competitors"),
    analysis_depth: str = Query("standard", description="quick, standard, deep"),
    include_opportunities: bool = Query(True, description="Include growth opportunities"),
    current_user: User = Depends(get_current_verified_user)
):
    """
    Perform competitive analysis against similar channels
    """
    try:
        competitive_data = await analytics_pipeline.competitive_analyzer.analyze_competition(
            user_channel_ids=channel_ids,
            competitor_keywords=competitor_keywords,
            analysis_depth=analysis_depth,
            user_id=str(current_user.id)
        )
        
        # Generate growth opportunities if requested
        opportunities = None
        if include_opportunities:
            opportunities = await analytics_pipeline.opportunity_finder.find_growth_opportunities(
                user_channels=channel_ids,
                competitive_landscape=competitive_data,
                user_id=str(current_user.id)
            )
        
        return {
            "success": True,
            "analysis_depth": analysis_depth,
            "channels_analyzed": len(channel_ids),
            "competitors_found": len(competitive_data.get("competitors", [])),
            "competitive_analysis": competitive_data,
            "growth_opportunities": opportunities,
            "market_position": {
                "relative_performance": competitive_data.get("market_position", {}),
                "competitive_advantages": competitive_data.get("advantages", []),
                "areas_for_improvement": competitive_data.get("improvement_areas", []),
                "market_share_estimate": competitive_data.get("market_share", 0)
            },
            "strategic_insights": {
                "content_gaps": opportunities.get("content_opportunities", []) if opportunities else [],
                "untapped_keywords": opportunities.get("keyword_opportunities", []) if opportunities else [],
                "audience_insights": competitive_data.get("audience_analysis", {}),
                "recommended_actions": opportunities.get("action_plan", []) if opportunities else []
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Competitive analysis failed: {str(e)}"
        )