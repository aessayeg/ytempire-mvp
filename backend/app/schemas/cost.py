"""
Cost tracking schemas
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
from decimal import Decimal


class CostRecordCreate(BaseModel):
    """Schema for creating cost record"""
    service: str = Field(..., description="Service name (openai, elevenlabs, etc.)")
    operation: str = Field(..., description="Operation type (gpt-4_input, dall-e_hd, etc.)")
    units: float = Field(..., description="Number of units consumed")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class CostRecordResponse(BaseModel):
    """Schema for cost record response"""
    service: str
    operation: str
    units: float
    total_cost: float
    timestamp: datetime


class VideoCostResponse(BaseModel):
    """Schema for video cost response"""
    video_id: str
    total_cost: float
    breakdown: Dict[str, float]
    timestamp: str


class CostAggregationResponse(BaseModel):
    """Schema for cost aggregation response"""
    period: str
    service: str
    total_cost: float
    operation_count: int
    average_cost: float


class ThresholdCreate(BaseModel):
    """Schema for creating threshold"""
    threshold_type: str = Field(..., regex="^(daily|monthly|per_video|service)$")
    value: float = Field(..., gt=0, description="Threshold value in USD")
    service: Optional[str] = Field(None, description="Service name for service-specific thresholds")
    alert_email: Optional[str] = Field(None, description="Email for alerts")


class ThresholdResponse(BaseModel):
    """Schema for threshold response"""
    threshold_type: str
    value: float
    service: Optional[str]
    alert_email: Optional[str]
    is_active: bool


class CostMetricsResponse(BaseModel):
    """Schema for cost metrics response"""
    total_cost: float
    api_costs: Dict[str, float]
    infrastructure_costs: Dict[str, float]
    per_video_cost: Optional[float]
    daily_cost: float
    monthly_projection: float
    threshold_status: Dict[str, Any]


class DailySummaryResponse(BaseModel):
    """Schema for daily cost summary"""
    date: str
    total_cost: float
    cost_by_service: Dict[str, float]
    hourly_breakdown: List[Dict[str, Any]]
    per_video_cost: Optional[float]
    threshold_status: Dict[str, Any]
    projections: Dict[str, float]


class CostTrendResponse(BaseModel):
    """Schema for cost trend analysis"""
    period_days: int
    daily_costs: Dict[str, float]
    average_daily_cost: float
    trend: str
    change_percentage: float
    total_period_cost: float
    highest_day: Optional[tuple]
    lowest_day: Optional[tuple]