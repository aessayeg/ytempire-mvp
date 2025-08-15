"""
Data Quality Pydantic schemas
P2 Enhancement - Request/response models for data quality API
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class QualityLevelEnum(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

class ValidationSeverityEnum(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ValidationRuleResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    name: str
    description: str
    field: str
    rule_type: str
    severity: ValidationSeverityEnum
    enabled: bool
    parameters: Dict[str, Any]

class QualityIssueResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    rule_name: str
    field: str
    issue_type: str
    severity: ValidationSeverityEnum
    description: str
    affected_records: int
    recommendation: str
    sample_data: List[Dict[str, Any]] = Field(default_factory=list)

class QualityMetricsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    total_records: int
    valid_records: int
    invalid_records: int
    completeness_score: float = Field(ge=0, le=100)
    accuracy_score: float = Field(ge=0, le=100)
    consistency_score: float = Field(ge=0, le=100)
    freshness_score: float = Field(ge=0, le=100)
    issues_by_severity: Dict[str, int]

class QualityReportResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    success: bool
    dataset_name: str
    timestamp: datetime
    overall_score: float = Field(ge=0, le=100)
    quality_level: QualityLevelEnum
    metrics: QualityMetricsResponse
    issues: List[QualityIssueResponse] = Field(default_factory=list)
    trends: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list)

class BatchProcessingRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    datasets: List[str] = Field(description="List of datasets to process")
    options: Dict[str, Any] = Field(default_factory=dict, description="Processing options")

class BatchProcessingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    success: bool
    message: str
    datasets: List[str]
    estimated_duration: int = Field(description="Estimated duration in seconds")
    timestamp: datetime

class CleanupOperationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    issue: str
    action: str
    records: int

class BatchCleanupPreview(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    would_fix_issues: int
    affected_records: int
    operations: List[CleanupOperationResponse]

class BatchCleanupResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    success: bool
    dry_run: bool
    dataset: str
    preview: Optional[BatchCleanupPreview] = None
    message: str
    timestamp: Optional[datetime] = None

class QualityTrendDataPoint(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    date: str
    overall_score: float
    completeness_score: float
    accuracy_score: float
    consistency_score: float
    freshness_score: float

class QualityTrendSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    overall_trend: str = Field(description="Overall trend direction: improving, declining, stable")
    best_metric: str = Field(description="Best performing metric")
    worst_metric: str = Field(description="Worst performing metric")
    trend_analysis: str = Field(description="Human-readable trend analysis")

class QualityTrendResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    success: bool
    dataset_name: str
    period_days: int
    trend_data: List[QualityTrendDataPoint]
    summary: QualityTrendSummary
    timestamp: datetime

class QualityAlert(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: str
    dataset: str
    severity: ValidationSeverityEnum
    title: str
    description: str
    affected_records: int
    recommendation: str
    created_at: datetime
    status: str = "active"

class QualityAlertsSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    critical: int = 0
    error: int = 0
    warning: int = 0
    info: int = 0

class QualityAlertsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    success: bool
    alerts: List[QualityAlert]
    total_alerts: int
    summary: QualityAlertsSummary

class MonitoringScheduleConfig(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    daily_checks: List[str]
    weekly_checks: List[str]
    monthly_reports: List[str]
    alerts: Dict[str, float]

class MonitoringScheduleResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    success: bool
    message: str
    schedule: MonitoringScheduleConfig
    timestamp: datetime

class DatasetQualitySummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    overall_score: float
    quality_level: QualityLevelEnum
    total_records: int
    critical_issues: int
    error_issues: int
    last_updated: datetime

class QualityMetricsSummaryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    success: bool
    datasets: Dict[str, DatasetQualitySummary]
    timestamp: datetime

class ValidationRulesResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    success: bool
    rules: List[ValidationRuleResponse]
    total_rules: int
    enabled_rules: int

# Request models for updates
class ValidationRuleUpdateRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    enabled: Optional[bool] = None
    parameters: Optional[Dict[str, Any]] = None
    severity: Optional[ValidationSeverityEnum] = None

class QualityThresholdUpdateRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    excellent_threshold: Optional[float] = Field(None, ge=0, le=100)
    good_threshold: Optional[float] = Field(None, ge=0, le=100)
    fair_threshold: Optional[float] = Field(None, ge=0, le=100)
    poor_threshold: Optional[float] = Field(None, ge=0, le=100)

# WebSocket schemas for real-time updates
class RealTimeQualityUpdate(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    dataset: str
    metric_type: str  # "overall", "completeness", "accuracy", etc.
    current_value: float
    previous_value: float
    change_percent: float
    timestamp: datetime
    alert_level: Optional[ValidationSeverityEnum] = None

class QualityMonitoringStatus(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    active_checks: int
    pending_checks: int
    failed_checks: int
    last_successful_run: Optional[datetime] = None
    next_scheduled_run: Optional[datetime] = None
    status: str = Field(description="Status: healthy, degraded, failed")

# Dashboard aggregation schemas
class QualityDashboardData(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    overall_health_score: float
    datasets_monitored: int
    total_records_analyzed: int
    active_alerts: int
    trends: Dict[str, str]  # metric -> trend direction
    top_issues: List[str]
    recent_improvements: List[str]
    timestamp: datetime

class DataProfileSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    dataset: str
    row_count: int
    column_count: int
    numeric_columns: int
    text_columns: int
    datetime_columns: int
    null_percentage: float
    duplicate_percentage: float
    data_types: Dict[str, str]
    profiling_timestamp: datetime

# Export all schemas
__all__ = [
    'QualityReportResponse',
    'QualityMetricsResponse',
    'QualityIssueResponse',
    'ValidationRuleResponse',
    'BatchProcessingRequest',
    'BatchProcessingResponse',
    'QualityTrendResponse',
    'QualityAlertsResponse',
    'MonitoringScheduleResponse',
    'QualityMetricsSummaryResponse',
    'ValidationRulesResponse',
    'QualityDashboardData',
    'DataProfileSummary'
]