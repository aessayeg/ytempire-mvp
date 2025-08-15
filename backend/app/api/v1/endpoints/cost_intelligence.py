"""
Cost Intelligence System API Endpoints
ML-based cost prediction, optimization recommendations, and budget management
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from datetime import datetime, date
from enum import Enum

from app.db.session import get_db
from app.api.v1.endpoints.auth import get_current_verified_user
from app.models.user import User
from app.services.cost_optimizer import get_cost_optimizer
from app.services.cost_tracking import cost_tracker

import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class PredictionTimeframe(str, Enum):
    """Prediction timeframe options"""
    WEEK = "7d"
    MONTH = "30d"
    QUARTER = "90d"


class OptimizationLevel(str, Enum):
    """Optimization aggressiveness levels"""
    CONSERVATIVE = "conservative"  # 10-20% savings
    MODERATE = "moderate"          # 20-35% savings
    AGGRESSIVE = "aggressive"      # 35-50% savings


class BudgetPeriod(str, Enum):
    """Budget management periods"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


class CostPredictionRequest(BaseModel):
    """Request for cost prediction"""
    timeframe: PredictionTimeframe = PredictionTimeframe.MONTH
    include_confidence_intervals: bool = True
    include_scenario_analysis: bool = False


class OptimizationRequest(BaseModel):
    """Request for optimization recommendations"""
    target_reduction_percent: float = Field(30.0, ge=10.0, le=60.0)
    optimization_level: OptimizationLevel = OptimizationLevel.MODERATE
    maintain_quality_threshold: float = Field(85.0, ge=70.0, le=100.0)
    services_to_optimize: List[str] = Field(default=[], description="Specific services to optimize")


class BudgetManagementRequest(BaseModel):
    """Request for budget management setup"""
    budget_amount: float = Field(..., gt=0, description="Budget amount in USD")
    period: BudgetPeriod = BudgetPeriod.MONTHLY
    alert_thresholds: Dict[str, float] = Field(
        default={"warning": 0.75, "critical": 0.90, "emergency": 0.95},
        description="Alert thresholds as percentages"
    )
    auto_actions_enabled: bool = Field(True, description="Enable automatic cost control actions")


@router.get("/predict")
async def predict_costs(
    timeframe: PredictionTimeframe = Query(PredictionTimeframe.MONTH, description="Prediction timeframe"),
    include_confidence: bool = Query(True, description="Include confidence intervals"),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Generate ML-based cost predictions for future periods
    """
    try:
        cost_optimizer = get_cost_optimizer()
        
        # Convert timeframe to days
        days_map = {
            PredictionTimeframe.WEEK: 7,
            PredictionTimeframe.MONTH: 30,
            PredictionTimeframe.QUARTER: 90
        }
        days_ahead = days_map[timeframe]
        
        predictions = await cost_optimizer.predict_future_costs(
            days_ahead=days_ahead,
            include_confidence=include_confidence
        )
        
        if "error" in predictions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=predictions["error"]
            )
        
        return {
            "success": True,
            "predictions": predictions,
            "generated_at": datetime.utcnow().isoformat(),
            "user_id": str(current_user.id)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cost prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate cost predictions: {str(e)}"
        )


@router.post("/optimize/recommendations")
async def get_optimization_recommendations(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Generate ML-driven cost optimization recommendations
    """
    try:
        cost_optimizer = get_cost_optimizer()
        
        # Track cost for this analysis
        background_tasks.add_task(
            cost_tracker.track_cost,
            user_id=str(current_user.id),
            service="cost_intelligence",
            operation="optimization_analysis",
            cost=0.01,
            metadata={"target_reduction": request.target_reduction_percent}
        )
        
        # Get current costs for analysis
        current_costs = await cost_optimizer.get_cost_report(days=30)
        
        # Generate recommendations
        recommendations = await cost_optimizer.generate_optimization_recommendations(
            current_costs=current_costs["costs_by_service"],
            target_reduction=request.target_reduction_percent / 100
        )
        
        # Apply optimization level filters
        filtered_recommendations = _filter_recommendations_by_level(
            recommendations, 
            request.optimization_level,
            request.maintain_quality_threshold
        )
        
        return {
            "success": True,
            "optimization_analysis": {
                "target_reduction_percent": request.target_reduction_percent,
                "optimization_level": request.optimization_level.value,
                "quality_threshold": request.maintain_quality_threshold,
                "current_cost_analysis": current_costs,
                "recommendations": filtered_recommendations,
                "implementation_roadmap": _generate_implementation_roadmap(filtered_recommendations),
                "expected_outcomes": {
                    "estimated_monthly_savings": filtered_recommendations.get("estimated_total_savings", 0),
                    "payback_period_days": 30,  # Immediate for most optimizations
                    "quality_impact_assessment": "Minimal to no impact with current settings",
                    "implementation_effort": _assess_implementation_effort(filtered_recommendations)
                }
            },
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Optimization recommendations failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate optimization recommendations: {str(e)}"
        )


@router.post("/budget/manage")
async def setup_budget_management(
    request: BudgetManagementRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Set up intelligent budget management with ML-based monitoring
    """
    try:
        cost_optimizer = get_cost_optimizer()
        
        # Convert period to monthly budget for calculation
        monthly_budget = request.budget_amount
        if request.period == BudgetPeriod.WEEKLY:
            monthly_budget = request.budget_amount * 4.33  # Average weeks per month
        elif request.period == BudgetPeriod.DAILY:
            monthly_budget = request.budget_amount * 30
        elif request.period == BudgetPeriod.QUARTERLY:
            monthly_budget = request.budget_amount / 3
        
        # Track cost for this setup
        background_tasks.add_task(
            cost_tracker.track_cost,
            user_id=str(current_user.id),
            service="cost_intelligence",
            operation="budget_setup",
            cost=0.005,
            metadata={"budget_amount": request.budget_amount, "period": request.period.value}
        )
        
        # Implement budget management
        budget_analysis = await cost_optimizer.implement_budget_management(
            monthly_budget=monthly_budget,
            alert_thresholds=request.alert_thresholds
        )
        
        if "error" in budget_analysis:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=budget_analysis["error"]
            )
        
        # Generate budget insights
        insights = _generate_budget_insights(budget_analysis, request)
        
        return {
            "success": True,
            "budget_management": {
                "configuration": {
                    "budget_amount": request.budget_amount,
                    "period": request.period.value,
                    "monthly_equivalent": monthly_budget,
                    "alert_thresholds": request.alert_thresholds,
                    "auto_actions_enabled": request.auto_actions_enabled
                },
                "current_status": budget_analysis,
                "insights": insights,
                "monitoring_setup": {
                    "real_time_tracking": True,
                    "daily_reports": True,
                    "alert_channels": ["email", "dashboard", "api"],
                    "auto_optimization": request.auto_actions_enabled
                }
            },
            "setup_completed_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Budget management setup failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to setup budget management: {str(e)}"
        )


# Helper functions

def _filter_recommendations_by_level(
    recommendations: Dict[str, Any],
    level: OptimizationLevel,
    quality_threshold: float
) -> Dict[str, Any]:
    """Filter recommendations based on optimization level and quality threshold"""
    filtered = {
        "recommendations": [],
        "estimated_total_savings": 0
    }
    
    # Define aggressiveness thresholds
    thresholds = {
        OptimizationLevel.CONSERVATIVE: 0.2,  # 20% max reduction
        OptimizationLevel.MODERATE: 0.35,      # 35% max reduction
        OptimizationLevel.AGGRESSIVE: 0.5      # 50% max reduction
    }
    
    max_reduction = thresholds[level]
    
    # Filter recommendations
    if "recommendations" in recommendations:
        for rec in recommendations.get("recommendations", []):
            # Check if recommendation meets criteria
            if rec.get("quality_impact", 100) >= quality_threshold:
                if rec.get("savings_percent", 0) <= max_reduction * 100:
                    filtered["recommendations"].append(rec)
                    filtered["estimated_total_savings"] += rec.get("estimated_savings", 0)
    
    return filtered


def _generate_implementation_roadmap(recommendations: Dict[str, Any]) -> Dict[str, Any]:
    """Generate implementation roadmap for recommendations"""
    roadmap = {
        "phases": [],
        "total_duration_days": 0,
        "priority_order": []
    }
    
    # Phase 1: Quick wins (immediate)
    quick_wins = [r for r in recommendations.get("recommendations", []) 
                  if r.get("implementation_effort", "low") == "low"]
    if quick_wins:
        roadmap["phases"].append({
            "phase": 1,
            "name": "Quick Wins",
            "duration_days": 7,
            "recommendations": quick_wins,
            "expected_savings": sum(r.get("estimated_savings", 0) for r in quick_wins)
        })
        roadmap["total_duration_days"] += 7
    
    # Phase 2: Medium effort optimizations
    medium_effort = [r for r in recommendations.get("recommendations", [])
                     if r.get("implementation_effort", "low") == "medium"]
    if medium_effort:
        roadmap["phases"].append({
            "phase": 2,
            "name": "Core Optimizations",
            "duration_days": 14,
            "recommendations": medium_effort,
            "expected_savings": sum(r.get("estimated_savings", 0) for r in medium_effort)
        })
        roadmap["total_duration_days"] += 14
    
    # Phase 3: High effort transformations
    high_effort = [r for r in recommendations.get("recommendations", [])
                   if r.get("implementation_effort", "low") == "high"]
    if high_effort:
        roadmap["phases"].append({
            "phase": 3,
            "name": "Strategic Transformations",
            "duration_days": 30,
            "recommendations": high_effort,
            "expected_savings": sum(r.get("estimated_savings", 0) for r in high_effort)
        })
        roadmap["total_duration_days"] += 30
    
    # Priority order
    roadmap["priority_order"] = [
        r.get("service", "unknown") 
        for r in sorted(recommendations.get("recommendations", []), 
                       key=lambda x: x.get("estimated_savings", 0), 
                       reverse=True)
    ]
    
    return roadmap


def _assess_implementation_effort(recommendations: Dict[str, Any]) -> str:
    """Assess overall implementation effort"""
    effort_counts = {"low": 0, "medium": 0, "high": 0}
    
    for rec in recommendations.get("recommendations", []):
        effort = rec.get("implementation_effort", "medium")
        effort_counts[effort] = effort_counts.get(effort, 0) + 1
    
    total = sum(effort_counts.values())
    if total == 0:
        return "minimal"
    
    # Calculate weighted effort
    weighted_score = (
        effort_counts["low"] * 1 +
        effort_counts["medium"] * 2 +
        effort_counts["high"] * 3
    ) / total
    
    if weighted_score <= 1.5:
        return "Low - Can be implemented within a week"
    elif weighted_score <= 2.5:
        return "Medium - Requires 2-3 weeks of implementation"
    else:
        return "High - Strategic initiative requiring 1+ months"


def _generate_budget_insights(budget_analysis: Dict[str, Any], request: BudgetManagementRequest) -> Dict[str, Any]:
    """Generate insights for budget management"""
    insights = {
        "current_utilization": budget_analysis.get("current_utilization", 0),
        "projected_overrun_risk": "low",
        "optimization_opportunities": [],
        "recommendations": []
    }
    
    # Calculate utilization percentage
    if "current_spend" in budget_analysis and request.budget_amount > 0:
        utilization = (budget_analysis["current_spend"] / request.budget_amount) * 100
        insights["current_utilization"] = utilization
        
        # Assess overrun risk
        if utilization > 90:
            insights["projected_overrun_risk"] = "high"
            insights["recommendations"].append("Immediate cost reduction required")
        elif utilization > 75:
            insights["projected_overrun_risk"] = "medium"
            insights["recommendations"].append("Consider implementing cost optimization")
        else:
            insights["projected_overrun_risk"] = "low"
            insights["recommendations"].append("Budget is healthy")
    
    # Identify optimization opportunities
    if "top_services" in budget_analysis:
        for service in budget_analysis["top_services"][:3]:
            insights["optimization_opportunities"].append({
                "service": service["name"],
                "current_cost": service["cost"],
                "potential_savings": service["cost"] * 0.3  # 30% potential
            })
    
    return insights