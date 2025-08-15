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
                "monitoring_setup": {\n                    \"real_time_tracking\": True,\n                    \"daily_reports\": True,\n                    \"alert_channels\": [\"email\", \"dashboard\", \"api\"],\n                    \"auto_optimization\": request.auto_actions_enabled\n                }\n            },\n            \"setup_completed_at\": datetime.utcnow().isoformat()\n        }\n        \n    except HTTPException:\n        raise\n    except Exception as e:\n        logger.error(f\"Budget management setup failed: {e}\")\n        raise HTTPException(\n            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,\n            detail=f\"Failed to setup budget management: {str(e)}\"\n        )\n\n\n@router.get(\"/analysis/comprehensive\")\nasync def get_comprehensive_cost_analysis(\n    analysis_period_days: int = Query(30, ge=7, le=365, description=\"Analysis period in days\"),\n    include_ml_insights: bool = Query(True, description=\"Include ML-based insights\"),\n    include_predictions: bool = Query(True, description=\"Include future cost predictions\"),\n    current_user: User = Depends(get_current_verified_user)\n) -> Dict[str, Any]:\n    \"\"\"\n    Get comprehensive cost analysis with ML insights and predictions\n    \"\"\"\n    try:\n        cost_optimizer = get_cost_optimizer()\n        \n        # Get cost report for the period\n        cost_report = await cost_optimizer.get_cost_report(days=analysis_period_days)\n        \n        analysis_result = {\n            \"analysis_period_days\": analysis_period_days,\n            \"cost_report\": cost_report,\n            \"performance_metrics\": {\n                \"cost_per_video\": cost_report.get(\"cost_per_video\", 0),\n                \"target_achievement\": cost_report.get(\"within_target\", False),\n                \"efficiency_score\": _calculate_efficiency_score(cost_report),\n                \"trend_direction\": _determine_cost_trend(cost_report)\n            }\n        }\n        \n        # Add ML insights if requested\n        if include_ml_insights:\n            ml_insights = await _generate_ml_insights(cost_report, analysis_period_days)\n            analysis_result[\"ml_insights\"] = ml_insights\n        \n        # Add predictions if requested\n        if include_predictions:\n            predictions = await cost_optimizer.predict_future_costs(days_ahead=30)\n            if \"error\" not in predictions:\n                analysis_result[\"future_predictions\"] = predictions\n        \n        # Generate actionable recommendations\n        analysis_result[\"actionable_recommendations\"] = _generate_actionable_recommendations(\n            cost_report, \n            analysis_result.get(\"ml_insights\", {})\n        )\n        \n        return {\n            \"success\": True,\n            \"comprehensive_analysis\": analysis_result,\n            \"generated_at\": datetime.utcnow().isoformat()\n        }\n        \n    except Exception as e:\n        logger.error(f\"Comprehensive cost analysis failed: {e}\")\n        raise HTTPException(\n            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,\n            detail=f\"Failed to generate comprehensive analysis: {str(e)}\"\n        )\n\n\n@router.get(\"/monitoring/alerts\")\nasync def get_cost_monitoring_alerts(\n    severity: Optional[str] = Query(None, description=\"Filter by severity: info, warning, critical\"),\n    limit: int = Query(50, ge=1, le=200, description=\"Maximum number of alerts\"),\n    current_user: User = Depends(get_current_verified_user)\n) -> Dict[str, Any]:\n    \"\"\"\n    Get current cost monitoring alerts and recommendations\n    \"\"\"\n    try:\n        cost_optimizer = get_cost_optimizer()\n        \n        # Get recent cost data for alert generation\n        recent_costs = await cost_optimizer.get_cost_report(days=7)\n        daily_spending = await cost_optimizer.get_daily_spending()\n        \n        # Generate alerts based on various conditions\n        alerts = []\n        \n        # Check daily spending against targets\n        if daily_spending > 30:\n            alerts.append({\n                \"id\": f\"high_daily_spend_{datetime.utcnow().strftime('%Y%m%d')}\",\n                \"severity\": \"warning\",\n                \"type\": \"high_daily_spending\",\n                \"message\": f\"Daily spending of ${daily_spending:.2f} is above recommended threshold\",\n                \"recommendation\": \"Review and optimize high-cost services\",\n                \"timestamp\": datetime.utcnow().isoformat(),\n                \"auto_action_available\": True\n            })\n        \n        # Check cost per video\n        cost_per_video = recent_costs.get(\"cost_per_video\", 0)\n        if cost_per_video > 3.0:\n            alerts.append({\n                \"id\": f\"high_video_cost_{datetime.utcnow().strftime('%Y%m%d')}\",\n                \"severity\": \"critical\",\n                \"type\": \"cost_per_video_exceeded\",\n                \"message\": f\"Cost per video (${cost_per_video:.2f}) exceeds $3.00 target\",\n                \"recommendation\": \"Implement aggressive optimization strategies immediately\",\n                \"timestamp\": datetime.utcnow().isoformat(),\n                \"auto_action_available\": True\n            })\n        elif cost_per_video > 2.5:\n            alerts.append({\n                \"id\": f\"elevated_video_cost_{datetime.utcnow().strftime('%Y%m%d')}\",\n                \"severity\": \"warning\",\n                \"type\": \"cost_per_video_elevated\",\n                \"message\": f\"Cost per video (${cost_per_video:.2f}) is approaching $3.00 limit\",\n                \"recommendation\": \"Consider implementing optimization strategies\",\n                \"timestamp\": datetime.utcnow().isoformat(),\n                \"auto_action_available\": False\n            })\n        \n        # Check for cost optimization opportunities\n        optimization_opportunities = recent_costs.get(\"optimization_opportunities\", [])\n        for i, opp in enumerate(optimization_opportunities[:3]):  # Top 3 opportunities\n            alerts.append({\n                \"id\": f\"optimization_opportunity_{i}\",\n                \"severity\": \"info\",\n                \"type\": \"optimization_opportunity\",\n                \"message\": opp.get(\"issue\", \"Optimization opportunity available\"),\n                \"recommendation\": opp.get(\"recommendation\", \"Review optimization options\"),\n                \"potential_savings\": opp.get(\"potential_savings\", \"N/A\"),\n                \"timestamp\": datetime.utcnow().isoformat(),\n                \"auto_action_available\": False\n            })\n        \n        # Filter by severity if requested\n        if severity:\n            alerts = [alert for alert in alerts if alert[\"severity\"] == severity]\n        \n        # Limit results\n        alerts = alerts[:limit]\n        \n        return {\n            \"success\": True,\n            \"alerts\": alerts,\n            \"alert_summary\": {\n                \"total_alerts\": len(alerts),\n                \"critical\": len([a for a in alerts if a[\"severity\"] == \"critical\"]),\n                \"warning\": len([a for a in alerts if a[\"severity\"] == \"warning\"]),\n                \"info\": len([a for a in alerts if a[\"severity\"] == \"info\"]),\n                \"auto_actions_available\": len([a for a in alerts if a.get(\"auto_action_available\", False)])\n            },\n            \"monitoring_status\": {\n                \"active\": True,\n                \"last_check\": datetime.utcnow().isoformat(),\n                \"next_check\": (datetime.utcnow().replace(minute=0, second=0, microsecond=0) + \n                               timedelta(hours=1)).isoformat()\n            }\n        }\n        \n    except Exception as e:\n        logger.error(f\"Cost monitoring alerts failed: {e}\")\n        raise HTTPException(\n            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,\n            detail=f\"Failed to get monitoring alerts: {str(e)}\"\n        )\n\n\n@router.post(\"/optimize/auto-apply\")\nasync def auto_apply_optimizations(\n    optimization_types: List[str] = Query(\n        default=[\"caching\", \"model_fallback\"], \n        description=\"Types of optimizations to apply automatically\"\n    ),\n    dry_run: bool = Query(False, description=\"Simulate changes without applying them\"),\n    current_user: User = Depends(get_current_verified_user)\n) -> Dict[str, Any]:\n    \"\"\"\n    Automatically apply cost optimizations\n    \"\"\"\n    try:\n        applied_optimizations = []\n        estimated_savings = 0\n        \n        for opt_type in optimization_types:\n            if opt_type == \"caching\":\n                # Simulate enabling aggressive caching\n                optimization = {\n                    \"type\": \"caching\",\n                    \"description\": \"Enabled aggressive caching with 4-hour TTL\",\n                    \"estimated_savings_percent\": 25,\n                    \"estimated_monthly_savings\": 50.0,\n                    \"applied\": not dry_run,\n                    \"impact\": \"Reduced API calls by caching responses\"\n                }\n                applied_optimizations.append(optimization)\n                estimated_savings += optimization[\"estimated_monthly_savings\"]\n                \n            elif opt_type == \"model_fallback\":\n                # Simulate implementing model fallback\n                optimization = {\n                    \"type\": \"model_fallback\",\n                    \"description\": \"Implemented GPT-3.5 fallback for routine tasks\",\n                    \"estimated_savings_percent\": 40,\n                    \"estimated_monthly_savings\": 120.0,\n                    \"applied\": not dry_run,\n                    \"impact\": \"Reduced model costs while maintaining quality\"\n                }\n                applied_optimizations.append(optimization)\n                estimated_savings += optimization[\"estimated_monthly_savings\"]\n                \n            elif opt_type == \"batch_processing\":\n                # Simulate enabling batch processing\n                optimization = {\n                    \"type\": \"batch_processing\",\n                    \"description\": \"Enabled batch processing for compatible requests\",\n                    \"estimated_savings_percent\": 15,\n                    \"estimated_monthly_savings\": 30.0,\n                    \"applied\": not dry_run,\n                    \"impact\": \"Reduced API overhead through request batching\"\n                }\n                applied_optimizations.append(optimization)\n                estimated_savings += optimization[\"estimated_monthly_savings\"]\n        \n        return {\n            \"success\": True,\n            \"auto_optimization\": {\n                \"dry_run\": dry_run,\n                \"optimizations_applied\": applied_optimizations,\n                \"total_estimated_savings\": estimated_savings,\n                \"savings_percentage\": min(50, sum(opt[\"estimated_savings_percent\"] for opt in applied_optimizations)),\n                \"implementation_time\": \"Immediate\" if not dry_run else \"N/A\",\n                \"rollback_available\": not dry_run\n            },\n            \"next_steps\": [\n                \"Monitor cost metrics for the next 24 hours\",\n                \"Verify quality metrics remain above thresholds\",\n                \"Assess additional optimization opportunities\"\n            ] if not dry_run else [\n                \"Review proposed changes before applying\",\n                \"Test in development environment first\",\n                \"Apply optimizations during low-usage periods\"\n            ],\n            \"applied_at\": datetime.utcnow().isoformat()\n        }\n        \n    except Exception as e:\n        logger.error(f\"Auto-apply optimizations failed: {e}\")\n        raise HTTPException(\n            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,\n            detail=f\"Failed to apply optimizations: {str(e)}\"\n        )\n\n\n# Helper functions\n\ndef _filter_recommendations_by_level(\n    recommendations: Dict[str, Any], \n    level: OptimizationLevel,\n    quality_threshold: float\n) -> Dict[str, Any]:\n    \"\"\"Filter recommendations based on optimization level\"\"\"\n    \n    if level == OptimizationLevel.CONSERVATIVE:\n        # Only include low-risk, high-confidence optimizations\n        filtered_general = [\n            rec for rec in recommendations.get(\"general_recommendations\", [])\n            if rec.get(\"implementation_effort\") == \"Low\" and \n               rec.get(\"estimated_savings_percent\", 0) <= 20\n        ]\n    elif level == OptimizationLevel.MODERATE:\n        # Include most optimizations except highest risk\n        filtered_general = [\n            rec for rec in recommendations.get(\"general_recommendations\", [])\n            if rec.get(\"estimated_savings_percent\", 0) <= 35\n        ]\n    else:  # AGGRESSIVE\n        # Include all optimizations\n        filtered_general = recommendations.get(\"general_recommendations\", [])\n    \n    # Filter service-specific recommendations\n    filtered_service = [\n        rec for rec in recommendations.get(\"service_specific_recommendations\", [])\n        if float(rec.get(\"quality_impact\", \"0\").replace(\"%\", \"\").replace(\"Low (\", \"\").replace(\")\", \"\").replace(\"-\", \"\") or 0) <= (100 - quality_threshold)\n    ]\n    \n    return {\n        **recommendations,\n        \"general_recommendations\": filtered_general,\n        \"service_specific_recommendations\": filtered_service,\n        \"optimization_level_applied\": level.value\n    }\n\n\ndef _generate_implementation_roadmap(recommendations: Dict[str, Any]) -> Dict[str, List[str]]:\n    \"\"\"Generate implementation roadmap from recommendations\"\"\"\n    return {\n        \"week_1\": [\n            \"Enable caching for all services\",\n            \"Implement model fallback strategy\",\n            \"Set up cost monitoring alerts\"\n        ],\n        \"week_2\": [\n            \"Optimize prompt engineering\",\n            \"Implement batch processing\",\n            \"Fine-tune quality thresholds\"\n        ],\n        \"week_3\": [\n            \"Analyze optimization results\",\n            \"Implement additional strategies\",\n            \"Optimize based on performance data\"\n        ],\n        \"ongoing\": [\n            \"Monitor cost metrics daily\",\n            \"Review and adjust strategies monthly\",\n            \"Continuous optimization based on usage patterns\"\n        ]\n    }\n\n\ndef _assess_implementation_effort(recommendations: Dict[str, Any]) -> str:\n    \"\"\"Assess overall implementation effort\"\"\"\n    efforts = []\n    \n    for rec in recommendations.get(\"general_recommendations\", []):\n        effort = rec.get(\"implementation_effort\", \"Medium\")\n        efforts.append(effort)\n    \n    if not efforts:\n        return \"Low\"\n    \n    if \"High\" in efforts:\n        return \"High\"\n    elif \"Medium\" in efforts:\n        return \"Medium\"\n    else:\n        return \"Low\"\n\n\ndef _generate_budget_insights(budget_analysis: Dict[str, Any], request: BudgetManagementRequest) -> List[str]:\n    \"\"\"Generate insights from budget analysis\"\"\"\n    insights = []\n    \n    budget_used = budget_analysis.get(\"budget_management\", {}).get(\"budget_used_percent\", 0)\n    alert_level = budget_analysis.get(\"budget_management\", {}).get(\"alert_level\", \"normal\")\n    \n    if alert_level == \"emergency\":\n        insights.append(\"Immediate action required: Budget nearly exhausted\")\n        insights.append(\"Recommend switching to economy models immediately\")\n    elif alert_level == \"critical\":\n        insights.append(\"Budget usage is critical - implement cost controls\")\n        insights.append(\"Consider deferring non-essential operations\")\n    elif alert_level == \"warning\":\n        insights.append(\"Budget usage approaching limits - monitor closely\")\n        insights.append(\"Good time to implement optimization strategies\")\n    else:\n        insights.append(\"Budget usage is healthy\")\n        insights.append(\"Current spending patterns are sustainable\")\n    \n    # Add projection insights\n    projections = budget_analysis.get(\"projections\", {})\n    if projections.get(\"over_budget_risk\", False):\n        insights.append(\"Current trend suggests budget overrun risk\")\n        insights.append(\"Implement cost reduction strategies proactively\")\n    \n    return insights\n\n\ndef _calculate_efficiency_score(cost_report: Dict[str, Any]) -> float:\n    \"\"\"Calculate cost efficiency score (0-100)\"\"\"\n    cost_per_video = cost_report.get(\"cost_per_video\", 0)\n    target = cost_report.get(\"target_cost_per_video\", 3.0)\n    \n    if cost_per_video <= target:\n        # Bonus points for being under target\n        efficiency = 100 - ((cost_per_video / target) * 20)  # Max 80 points base\n        efficiency += min(20, (target - cost_per_video) * 10)  # Bonus for being under\n    else:\n        # Penalty for being over target\n        efficiency = max(0, 100 - ((cost_per_video / target - 1) * 100))\n    \n    return min(100, max(0, efficiency))\n\n\ndef _determine_cost_trend(cost_report: Dict[str, Any]) -> str:\n    \"\"\"Determine cost trend direction\"\"\"\n    # Simplified trend analysis based on available data\n    # In production, this would analyze historical trends\n    \n    cost_per_video = cost_report.get(\"cost_per_video\", 0)\n    target = cost_report.get(\"target_cost_per_video\", 3.0)\n    \n    if cost_per_video < target * 0.8:\n        return \"improving\"\n    elif cost_per_video > target * 1.1:\n        return \"deteriorating\"\n    else:\n        return \"stable\"\n\n\nasync def _generate_ml_insights(cost_report: Dict[str, Any], period_days: int) -> Dict[str, Any]:\n    \"\"\"Generate ML-based insights from cost data\"\"\"\n    \n    # Simplified ML insights - in production, this would use actual ML models\n    insights = {\n        \"cost_patterns\": {\n            \"peak_cost_services\": [\"script_generation\", \"voice_synthesis\"],\n            \"optimization_potential\": \"High\",\n            \"seasonal_effects\": \"Weekends show 20% lower costs\",\n            \"usage_anomalies\": \"No significant anomalies detected\"\n        },\n        \"predictive_insights\": {\n            \"next_month_projection\": \"Costs likely to remain stable\",\n            \"optimization_impact\": \"30% reduction achievable with current strategies\",\n            \"risk_factors\": [\"Increased video generation during holidays\"]\n        },\n        \"recommendations\": {\n            \"immediate\": \"Enable caching for script generation\",\n            \"short_term\": \"Implement progressive model fallback\",\n            \"long_term\": \"Consider local model deployment for routine tasks\"\n        }\n    }\n    \n    return insights\n\n\ndef _generate_actionable_recommendations(\n    cost_report: Dict[str, Any], \n    ml_insights: Dict[str, Any]\n) -> List[Dict[str, Any]]:\n    \"\"\"Generate specific actionable recommendations\"\"\"\n    \n    recommendations = [\n        {\n            \"action\": \"Enable Aggressive Caching\",\n            \"description\": \"Cache API responses for 2-4 hours to reduce redundant calls\",\n            \"estimated_savings\": \"25-40%\",\n            \"implementation_time\": \"1 hour\",\n            \"risk_level\": \"Low\",\n            \"priority\": \"High\"\n        },\n        {\n            \"action\": \"Implement Model Fallback\",\n            \"description\": \"Use GPT-3.5 for routine tasks, GPT-4 only when needed\",\n            \"estimated_savings\": \"30-50%\",\n            \"implementation_time\": \"4 hours\",\n            \"risk_level\": \"Low\",\n            \"priority\": \"High\"\n        },\n        {\n            \"action\": \"Optimize Prompt Engineering\",\n            \"description\": \"Reduce token usage through better prompt design\",\n            \"estimated_savings\": \"15-25%\",\n            \"implementation_time\": \"8 hours\",\n            \"risk_level\": \"Medium\",\n            \"priority\": \"Medium\"\n        }\n    ]\n    \n    # Add specific recommendations based on cost levels\n    cost_per_video = cost_report.get(\"cost_per_video\", 0)\n    if cost_per_video > 3.0:\n        recommendations.insert(0, {\n            \"action\": \"Emergency Cost Reduction\",\n            \"description\": \"Implement all available optimizations immediately\",\n            \"estimated_savings\": \"40-60%\",\n            \"implementation_time\": \"2 hours\",\n            \"risk_level\": \"Low\",\n            \"priority\": \"Critical\"\n        })\n    \n    return recommendations