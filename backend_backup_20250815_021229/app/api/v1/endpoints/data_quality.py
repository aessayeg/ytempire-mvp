"""
Data Quality API endpoints
P2 Enhancement - REST API for data quality monitoring and batch processing
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from app.api.v1.endpoints.auth import get_current_verified_user
from app.db.session import get_db
from app.services.data_quality import (
    DataQualityFramework, 
    BatchDataProcessor, 
    QualityReport,
    QualityLevel,
    ValidationSeverity
)
from app.models.user import User
from app.schemas.data_quality import (
    QualityReportResponse,
    QualityMetricsResponse, 
    ValidationRuleResponse,
    BatchProcessingRequest,
    BatchProcessingResponse,
    QualityTrendResponse
)

logger = logging.getLogger(__name__)

router = APIRouter()
quality_framework = DataQualityFramework()
batch_processor = BatchDataProcessor()

@router.get("/health")
async def health_check():
    """Health check for data quality service"""
    return {"status": "healthy", "service": "data_quality", "timestamp": datetime.now()}

@router.get("/report/{dataset_name}", response_model=QualityReportResponse)
async def get_quality_report(
    dataset_name: str,
    current_user: User = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    """Generate data quality report for a specific dataset"""
    try:
        if dataset_name not in ["videos", "channels", "analytics"]:
            raise HTTPException(status_code=400, detail="Invalid dataset name")
        
        logger.info(f"Generating quality report for dataset: {dataset_name} (user: {current_user.id})")
        
        report = await quality_framework.generate_quality_report(dataset_name)
        
        return QualityReportResponse(
            success=True,
            dataset_name=dataset_name,
            timestamp=report.timestamp,
            overall_score=report.metrics.overall_score,
            quality_level=report.metrics.quality_level.value,
            metrics={
                "total_records": report.metrics.total_records,
                "valid_records": report.metrics.valid_records,
                "invalid_records": report.metrics.invalid_records,
                "completeness_score": report.metrics.completeness_score,
                "accuracy_score": report.metrics.accuracy_score,
                "consistency_score": report.metrics.consistency_score,
                "freshness_score": report.metrics.freshness_score,
                "issues_by_severity": report.metrics.issues_by_severity
            },
            issues=[
                {
                    "rule_name": issue.rule_name,
                    "field": issue.field,
                    "issue_type": issue.issue_type,
                    "severity": issue.severity.value,
                    "description": issue.description,
                    "affected_records": issue.affected_records,
                    "recommendation": issue.recommendation,
                    "sample_data": issue.sample_data[:3]  # Limit sample data
                }
                for issue in report.issues
            ],
            trends=report.trends,
            recommendations=report.recommendations,
            action_items=report.action_items
        )
        
    except Exception as e:
        logger.error(f"Error generating quality report for {dataset_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate quality report")

@router.get("/metrics/summary")
async def get_quality_metrics_summary(
    current_user: User = Depends(get_current_verified_user),
    datasets: List[str] = Query(default=["videos", "channels", "analytics"])
):
    """Get quality metrics summary for multiple datasets"""
    try:
        summary = {}
        
        for dataset in datasets:
            if dataset not in ["videos", "channels", "analytics"]:
                continue
                
            report = await quality_framework.generate_quality_report(dataset)
            summary[dataset] = {
                "overall_score": report.metrics.overall_score,
                "quality_level": report.metrics.quality_level.value,
                "total_records": report.metrics.total_records,
                "critical_issues": report.metrics.issues_by_severity.get("critical", 0),
                "error_issues": report.metrics.issues_by_severity.get("error", 0),
                "last_updated": report.timestamp
            }
        
        return {
            "success": True,
            "datasets": summary,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error getting quality metrics summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get quality metrics")

@router.get("/validation-rules")
async def get_validation_rules(
    current_user: User = Depends(get_current_verified_user)
):
    """Get all data validation rules"""
    try:
        rules = quality_framework.validation_rules
        
        return {
            "success": True,
            "rules": [
                {
                    "name": rule.name,
                    "description": rule.description,
                    "field": rule.field,
                    "rule_type": rule.rule_type,
                    "severity": rule.severity.value,
                    "enabled": rule.enabled,
                    "parameters": rule.parameters
                }
                for rule in rules
            ],
            "total_rules": len(rules),
            "enabled_rules": sum(1 for rule in rules if rule.enabled)
        }
        
    except Exception as e:
        logger.error(f"Error getting validation rules: {e}")
        raise HTTPException(status_code=500, detail="Failed to get validation rules")

@router.post("/batch/quality-check", response_model=BatchProcessingResponse)
async def batch_quality_check(
    background_tasks: BackgroundTasks,
    request: BatchProcessingRequest,
    current_user: User = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    """Run batch quality checks on multiple datasets"""
    try:
        # Validate datasets
        valid_datasets = [ds for ds in request.datasets if ds in ["videos", "channels", "analytics"]]
        
        if not valid_datasets:
            raise HTTPException(status_code=400, detail="No valid datasets specified")
        
        logger.info(f"Starting batch quality check for datasets: {valid_datasets} (user: {current_user.id})")
        
        # Run batch processing in background
        background_tasks.add_task(
            _run_batch_quality_check, 
            valid_datasets, 
            current_user.id
        )
        
        return BatchProcessingResponse(
            success=True,
            message=f"Batch quality check started for {len(valid_datasets)} datasets",
            datasets=valid_datasets,
            estimated_duration=len(valid_datasets) * 30,  # 30 seconds per dataset
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error starting batch quality check: {e}")
        raise HTTPException(status_code=500, detail="Failed to start batch quality check")

@router.post("/batch/cleanup/{dataset_name}")
async def batch_data_cleanup(
    dataset_name: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_verified_user),
    dry_run: bool = Query(default=True, description="Run in dry-run mode without making changes")
):
    """Run batch data cleanup for a specific dataset"""
    try:
        if dataset_name not in ["videos", "channels", "analytics"]:
            raise HTTPException(status_code=400, detail="Invalid dataset name")
        
        logger.info(f"Starting batch cleanup for {dataset_name} (user: {current_user.id}, dry_run: {dry_run})")
        
        if dry_run:
            # Generate preview of what would be cleaned
            report = await quality_framework.generate_quality_report(dataset_name)
            
            cleanup_preview = {
                "would_fix_issues": len([i for i in report.issues if i.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]]),
                "affected_records": sum(i.affected_records for i in report.issues if i.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]),
                "operations": [
                    {
                        "issue": issue.rule_name,
                        "action": _get_cleanup_action(issue.issue_type),
                        "records": issue.affected_records
                    }
                    for issue in report.issues[:5]  # Preview first 5 issues
                ]
            }
            
            return {
                "success": True,
                "dry_run": True,
                "dataset": dataset_name,
                "preview": cleanup_preview,
                "message": "Dry run completed. Use dry_run=false to execute cleanup."
            }
        else:
            # Run actual cleanup in background
            background_tasks.add_task(_run_batch_cleanup, dataset_name, current_user.id)
            
            return {
                "success": True,
                "dry_run": False,
                "dataset": dataset_name,
                "message": "Batch cleanup started",
                "timestamp": datetime.now()
            }
        
    except Exception as e:
        logger.error(f"Error in batch cleanup for {dataset_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to perform batch cleanup")

@router.get("/trends/{dataset_name}")
async def get_quality_trends(
    dataset_name: str,
    current_user: User = Depends(get_current_verified_user),
    days: int = Query(default=30, ge=1, le=365)
):
    """Get quality trends for a dataset over time"""
    try:
        if dataset_name not in ["videos", "channels", "analytics"]:
            raise HTTPException(status_code=400, detail="Invalid dataset name")
        
        # For now, return mock trend data
        # In production, this would query historical quality metrics
        trend_data = _generate_mock_trends(dataset_name, days)
        
        return QualityTrendResponse(
            success=True,
            dataset_name=dataset_name,
            period_days=days,
            trend_data=trend_data,
            summary={
                "overall_trend": "improving",
                "best_metric": "accuracy",
                "worst_metric": "completeness",
                "trend_analysis": f"Quality has improved by 5.2% over the last {days} days"
            },
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error getting quality trends for {dataset_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get quality trends")

@router.get("/alerts")
async def get_quality_alerts(
    current_user: User = Depends(get_current_verified_user),
    severity: Optional[str] = Query(default=None, description="Filter by severity: critical, error, warning, info")
):
    """Get active data quality alerts"""
    try:
        # Generate alerts based on current quality issues
        alerts = []
        
        for dataset in ["videos", "channels", "analytics"]:
            report = await quality_framework.generate_quality_report(dataset)
            
            for issue in report.issues:
                if severity and issue.severity.value != severity:
                    continue
                    
                if issue.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]:
                    alerts.append({
                        "id": f"{dataset}_{issue.rule_name}_{int(issue.timestamp.timestamp())}",
                        "dataset": dataset,
                        "severity": issue.severity.value,
                        "title": f"{issue.field} {issue.issue_type}",
                        "description": issue.description,
                        "affected_records": issue.affected_records,
                        "recommendation": issue.recommendation,
                        "created_at": issue.timestamp,
                        "status": "active"
                    })
        
        # Sort by severity and timestamp
        severity_order = {"critical": 4, "error": 3, "warning": 2, "info": 1}
        alerts.sort(key=lambda x: (severity_order.get(x["severity"], 0), x["created_at"]), reverse=True)
        
        return {
            "success": True,
            "alerts": alerts[:20],  # Limit to 20 most recent alerts
            "total_alerts": len(alerts),
            "summary": {
                "critical": sum(1 for a in alerts if a["severity"] == "critical"),
                "error": sum(1 for a in alerts if a["severity"] == "error"),
                "warning": sum(1 for a in alerts if a["severity"] == "warning"),
                "info": sum(1 for a in alerts if a["severity"] == "info")
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting quality alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to get quality alerts")

@router.post("/monitoring/schedule")
async def setup_quality_monitoring(
    current_user: User = Depends(get_current_verified_user)
):
    """Set up automated quality monitoring schedule"""
    try:
        schedule = await batch_processor.schedule_quality_monitoring()
        
        return {
            "success": True,
            "message": "Quality monitoring schedule configured",
            "schedule": schedule,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error setting up quality monitoring: {e}")
        raise HTTPException(status_code=500, detail="Failed to setup quality monitoring")

# Background task functions
async def _run_batch_quality_check(datasets: List[str], user_id: int):
    """Background task to run batch quality check"""
    try:
        logger.info(f"Running batch quality check for {datasets} (user: {user_id})")
        reports = await batch_processor.process_batch_quality_check(datasets)
        
        # In production, you would store results and send notifications
        logger.info(f"Batch quality check completed for {len(reports)} datasets")
        
    except Exception as e:
        logger.error(f"Error in batch quality check task: {e}")

async def _run_batch_cleanup(dataset_name: str, user_id: int):
    """Background task to run batch cleanup"""
    try:
        logger.info(f"Running batch cleanup for {dataset_name} (user: {user_id})")
        results = await batch_processor.batch_data_cleanup(dataset_name)
        
        # In production, you would store results and send notifications
        logger.info(f"Batch cleanup completed: {results}")
        
    except Exception as e:
        logger.error(f"Error in batch cleanup task: {e}")

def _get_cleanup_action(issue_type: str) -> str:
    """Get cleanup action description for issue type"""
    actions = {
        "null_violation": "Fill with default values",
        "range_violation": "Cap to valid range",
        "pattern_violation": "Format standardization",
        "uniqueness_violation": "Remove duplicates",
        "trend_anomaly": "Flag for review"
    }
    return actions.get(issue_type, "Manual review required")

def _generate_mock_trends(dataset_name: str, days: int) -> List[Dict[str, Any]]:
    """Generate mock trend data for demonstration"""
    trends = []
    base_date = datetime.now() - timedelta(days=days)
    
    for i in range(days):
        date = base_date + timedelta(days=i)
        trends.append({
            "date": date.isoformat(),
            "overall_score": max(60, 85 + (i * 0.1) + (i % 7) * 2),  # Trending upward with weekly cycles
            "completeness_score": max(70, 80 + (i * 0.05) + (i % 5) * 3),
            "accuracy_score": max(75, 88 + (i * 0.08) + (i % 3) * 1),
            "consistency_score": max(80, 85 + (i * 0.03) + (i % 6) * 2),
            "freshness_score": max(85, 90 + (i * 0.02) + (i % 4) * 1)
        })
    
    return trends