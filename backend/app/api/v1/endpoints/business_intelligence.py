"""
Business Intelligence API endpoints
Executive dashboard, financial reports, and strategic insights
"""
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_

from app.db.session import get_db
from app.services.realtime_analytics_service import realtime_analytics_service
from app.services.beta_success_metrics import beta_success_metrics_service
from app.services.analytics_service import analytics_pipeline
from app.models.user import User
from app.models.video import Video
from app.models.analytics import Analytics
from app.api.v1.endpoints.auth import get_current_user, get_current_verified_user

router = APIRouter()


@router.get("/executive-metrics")
async def get_executive_metrics(
    period: str = Query("30d", description="Time period: 24h, 7d, 30d, 90d, 1y"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get executive-level business metrics"""
    try:
        # Calculate time range
        end_time = datetime.utcnow()
        if period == "24h":
            start_time = end_time - timedelta(hours=24)
        elif period == "7d":
            start_time = end_time - timedelta(days=7)
        elif period == "30d":
            start_time = end_time - timedelta(days=30)
        elif period == "90d":
            start_time = end_time - timedelta(days=90)
        elif period == "1y":
            start_time = end_time - timedelta(days=365)
        else:
            start_time = end_time - timedelta(days=30)

        # Previous period for comparison
        period_length = end_time - start_time
        previous_start = start_time - period_length
        previous_end = start_time

        # Get beta user metrics summary
        beta_summary = await beta_success_metrics_service.get_success_summary()

        # Get real-time metrics
        realtime_metrics = await realtime_analytics_service.get_realtime_metrics()

        # Calculate MRR (Monthly Recurring Revenue)
        # Mock calculation - in production this would come from billing system
        active_users = beta_summary.get("total_users", 0)
        mrr = active_users * 99  # $99/month subscription
        previous_mrr = mrr * 0.85  # Mock 15% growth

        # Calculate ARR (Annual Recurring Revenue)
        arr = mrr * 12
        previous_arr = previous_mrr * 12

        # Monthly Active Users
        mau = active_users
        previous_mau = int(mau * 0.9)  # Mock 10% growth

        # Videos generated
        videos_generated = realtime_metrics.get("views", {}).get("count", 0)
        previous_videos = int(videos_generated * 0.8)  # Mock 20% growth

        # Cost metrics
        total_cost = realtime_metrics.get("cost", {}).get("sum", 0)
        avg_cost_per_video = (
            total_cost / videos_generated if videos_generated > 0 else 0
        )
        previous_cost_per_video = avg_cost_per_video * 1.1  # Mock 10% improvement

        # Quality score
        quality_score = (
            beta_summary.get("average_metrics", {}).get("overall_score", 0) / 10
        )
        previous_quality = quality_score * 0.95  # Mock 5% improvement

        return {
            "mrr": mrr,
            "previous_mrr": previous_mrr,
            "mrr_change": mrr - previous_mrr,
            "mrr_change_percent": ((mrr - previous_mrr) / previous_mrr * 100)
            if previous_mrr > 0
            else 0,
            "arr": arr,
            "previous_arr": previous_arr,
            "arr_change": arr - previous_arr,
            "arr_change_percent": ((arr - previous_arr) / previous_arr * 100)
            if previous_arr > 0
            else 0,
            "mau": mau,
            "previous_mau": previous_mau,
            "mau_change": mau - previous_mau,
            "mau_change_percent": ((mau - previous_mau) / previous_mau * 100)
            if previous_mau > 0
            else 0,
            "videos_generated": videos_generated,
            "previous_videos": previous_videos,
            "videos_change": videos_generated - previous_videos,
            "videos_change_percent": (
                (videos_generated - previous_videos) / previous_videos * 100
            )
            if previous_videos > 0
            else 0,
            "avg_cost_per_video": round(avg_cost_per_video, 2),
            "previous_cost_per_video": round(previous_cost_per_video, 2),
            "cost_change": avg_cost_per_video - previous_cost_per_video,
            "cost_change_percent": (
                (avg_cost_per_video - previous_cost_per_video)
                / previous_cost_per_video
                * 100
            )
            if previous_cost_per_video > 0
            else 0,
            "quality_score": round(quality_score, 1),
            "previous_quality": round(previous_quality, 1),
            "quality_change": quality_score - previous_quality,
            "quality_change_percent": (
                (quality_score - previous_quality) / previous_quality * 100
            )
            if previous_quality > 0
            else 0,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch executive metrics: {str(e)}"
        )


@router.get("/business-kpis")
async def get_business_kpis(
    period: str = Query("30d", description="Time period: 24h, 7d, 30d, 90d, 1y"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get business KPIs with targets and benchmarks"""
    try:
        # Get beta user success KPIs
        kpis = await beta_success_metrics_service.get_kpis()

        # Convert to API format
        business_kpis = []
        for kpi_name, kpi_data in kpis.items():
            business_kpis.append(
                {
                    "id": kpi_name,
                    "name": kpi_data.name,
                    "current": kpi_data.current_value,
                    "target": kpi_data.target_value,
                    "benchmark": kpi_data.target_value * 0.8,  # Mock industry benchmark
                    "trend": [
                        kpi_data.current_value * (1 + np.random.normal(0, 0.1))
                        for _ in range(30)
                    ],  # Mock trend data
                    "status": kpi_data.status,
                    "category": kpi_data.category,
                    "unit": kpi_data.measurement_unit,
                }
            )

        return business_kpis

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch business KPIs: {str(e)}"
        )


@router.get("/financial-performance")
async def get_financial_performance(
    period: str = Query("30d", description="Time period: 24h, 7d, 30d, 90d, 1y"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get financial performance data over time"""
    try:
        # Get beta user metrics
        beta_summary = await beta_success_metrics_service.get_success_summary()

        # Mock financial data - in production this would come from billing system
        periods = []
        base_date = datetime.utcnow() - timedelta(days=30)

        for i in range(30):
            date = base_date + timedelta(days=i)

            # Mock progressive growth
            growth_factor = 1 + (i * 0.02)  # 2% daily growth
            base_users = beta_summary.get("total_users", 10)

            revenue = base_users * growth_factor * 3.30  # $3.30 daily ARPU
            costs = revenue * 0.6  # 60% cost ratio
            profit = revenue - costs
            margin = (profit / revenue * 100) if revenue > 0 else 0
            users = int(base_users * growth_factor)
            arpu = revenue / users if users > 0 else 0

            periods.append(
                {
                    "period": date.strftime("%Y-%m-%d"),
                    "revenue": round(revenue, 2),
                    "costs": round(costs, 2),
                    "profit": round(profit, 2),
                    "margin": round(margin, 1),
                    "users": users,
                    "arpu": round(arpu, 2),
                    "ltv": round(arpu * 12, 2),  # Mock 12-month LTV
                    "cac": round(costs * 0.3, 2),  # Mock 30% of costs as CAC
                }
            )

        return periods

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch financial performance: {str(e)}"
        )


@router.get("/user-segments")
async def get_user_segments(
    period: str = Query("30d", description="Time period: 24h, 7d, 30d, 90d, 1y"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get user segmentation analysis"""
    try:
        # Get beta user segments from success metrics
        success_summary = await beta_success_metrics_service.get_success_summary()

        # Mock user segments - in production this would come from detailed user analysis
        total_users = success_summary.get("total_users", 0)

        segments = [
            {
                "segment": "Power Users",
                "count": int(total_users * 0.2),
                "percentage": 20.0,
                "revenue": total_users * 0.2 * 199,  # $199/month
                "avgLifetime": 24.0,  # months
                "churnRate": 2.5,
                "growthRate": 15.2,
            },
            {
                "segment": "Regular Users",
                "count": int(total_users * 0.5),
                "percentage": 50.0,
                "revenue": total_users * 0.5 * 99,  # $99/month
                "avgLifetime": 12.0,
                "churnRate": 8.1,
                "growthRate": 12.5,
            },
            {
                "segment": "Casual Users",
                "count": int(total_users * 0.25),
                "percentage": 25.0,
                "revenue": total_users * 0.25 * 49,  # $49/month
                "avgLifetime": 6.0,
                "churnRate": 15.3,
                "growthRate": 8.7,
            },
            {
                "segment": "Trial Users",
                "count": int(total_users * 0.05),
                "percentage": 5.0,
                "revenue": 0,  # No revenue yet
                "avgLifetime": 0.5,
                "churnRate": 35.0,
                "growthRate": 25.0,
            },
        ]

        return segments

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch user segments: {str(e)}"
        )


@router.get("/competitive-analysis")
async def get_competitive_analysis(
    db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)
):
    """Get competitive analysis metrics"""
    try:
        # Mock competitive analysis - in production this would come from market research
        competitive_metrics = [
            {
                "metric": "Cost per Video",
                "ourValue": 1.85,
                "industry": 3.20,
                "leader": 1.50,
                "position": "Competitive",
            },
            {
                "metric": "Generation Speed (minutes)",
                "ourValue": 2.3,
                "industry": 5.8,
                "leader": 1.8,
                "position": "Leading",
            },
            {
                "metric": "Quality Score",
                "ourValue": 8.7,
                "industry": 7.2,
                "leader": 9.1,
                "position": "Competitive",
            },
            {
                "metric": "User Satisfaction",
                "ourValue": 92,
                "industry": 78,
                "leader": 94,
                "position": "Leading",
            },
            {
                "metric": "Feature Count",
                "ourValue": 24,
                "industry": 18,
                "leader": 31,
                "position": "Competitive",
            },
            {
                "metric": "API Response Time (ms)",
                "ourValue": 145,
                "industry": 280,
                "leader": 120,
                "position": "Leading",
            },
        ]

        return competitive_metrics

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch competitive analysis: {str(e)}"
        )


@router.get("/alerts")
async def get_business_alerts(
    db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)
):
    """Get current business alerts and notifications"""
    try:
        alerts = []

        # Get success metrics alerts
        success_summary = await beta_success_metrics_service.get_success_summary()

        # Check for high churn risk
        avg_churn_prob = success_summary.get("average_metrics", {}).get(
            "churn_probability", 0
        )
        if avg_churn_prob > 0.3:
            alerts.append(
                {
                    "severity": "warning",
                    "title": "Elevated Churn Risk",
                    "message": f"Average churn probability is {avg_churn_prob:.1%}. Consider user engagement initiatives.",
                }
            )

        # Check for low engagement
        avg_engagement = success_summary.get("average_metrics", {}).get(
            "overall_score", 0
        )
        if avg_engagement < 60:
            alerts.append(
                {
                    "severity": "warning",
                    "title": "Low User Engagement",
                    "message": f"Average engagement score is {avg_engagement:.1f}/100. Review onboarding process.",
                }
            )

        # Check for cost issues
        realtime_metrics = await realtime_analytics_service.get_realtime_metrics()
        if realtime_metrics.get("cost", {}).get("avg", 0) > 2.5:
            alerts.append(
                {
                    "severity": "error",
                    "title": "Cost Target Exceeded",
                    "message": "Average cost per video exceeds $2.50 target. Immediate optimization required.",
                }
            )

        # Performance alert
        if len(alerts) == 0:
            # All good
            pass

        return alerts

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch business alerts: {str(e)}"
        )


@router.get("/insights")
async def get_strategic_insights(
    period: str = Query("30d", description="Time period: 24h, 7d, 30d, 90d, 1y"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get strategic insights and recommendations"""
    try:
        # Get current metrics
        success_summary = await beta_success_metrics_service.get_success_summary()
        realtime_metrics = await realtime_analytics_service.get_realtime_metrics()

        insights = []

        # Analyze user engagement
        avg_engagement = success_summary.get("average_metrics", {}).get(
            "overall_score", 0
        )
        if avg_engagement > 75:
            insights.append(
                {
                    "title": "High User Engagement Opportunity",
                    "description": "Users are highly engaged. Consider introducing premium features or expanding feature set to capture more value.",
                    "impact": "High",
                    "effort": "Medium",
                    "tags": ["Revenue", "Product", "Growth"],
                }
            )
        elif avg_engagement < 50:
            insights.append(
                {
                    "title": "User Experience Improvement Needed",
                    "description": "Low engagement scores indicate UX issues. Focus on onboarding optimization and feature discoverability.",
                    "impact": "High",
                    "effort": "High",
                    "tags": ["UX", "Retention", "Product"],
                }
            )

        # Cost optimization insights
        avg_cost = realtime_metrics.get("cost", {}).get("avg", 0)
        if avg_cost > 2.0:
            insights.append(
                {
                    "title": "Cost Optimization Opportunity",
                    "description": "Current cost per video exceeds target. Implement model optimization and caching strategies.",
                    "impact": "High",
                    "effort": "Medium",
                    "tags": ["Cost", "Technical", "Efficiency"],
                }
            )

        # Growth insights
        total_users = success_summary.get("total_users", 0)
        if total_users < 50:
            insights.append(
                {
                    "title": "Accelerate Beta User Acquisition",
                    "description": "Current beta user count is below target. Increase marketing efforts and referral programs.",
                    "impact": "Medium",
                    "effort": "Medium",
                    "tags": ["Growth", "Marketing", "Beta"],
                }
            )

        # Quality insights
        insights.append(
            {
                "title": "Content Quality Differentiation",
                "description": "High content quality scores present competitive advantage. Consider quality as key marketing message.",
                "impact": "Medium",
                "effort": "Low",
                "tags": ["Marketing", "Competitive", "Quality"],
            }
        )

        # Feature adoption insights
        insights.append(
            {
                "title": "Feature Adoption Analysis",
                "description": "Track which features drive highest user satisfaction to prioritize development resources.",
                "impact": "Medium",
                "effort": "Low",
                "tags": ["Analytics", "Product", "Development"],
            }
        )

        # Market expansion insight
        insights.append(
            {
                "title": "Market Expansion Readiness",
                "description": "Strong beta metrics indicate readiness for broader market expansion. Plan public launch strategy.",
                "impact": "High",
                "effort": "High",
                "tags": ["Strategy", "Launch", "Growth"],
            }
        )

        return insights

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch strategic insights: {str(e)}"
        )


@router.get("/revenue-forecast")
async def get_revenue_forecast(
    months: int = Query(12, description="Number of months to forecast"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get revenue forecast based on current trends"""
    try:
        # Get current metrics
        success_summary = await beta_success_metrics_service.get_success_summary()
        current_users = success_summary.get("total_users", 0)

        # Mock revenue forecasting - in production this would use ML models
        forecast = []
        base_date = datetime.utcnow()

        # Assumptions
        monthly_growth_rate = 0.15  # 15% monthly growth
        arpu = 99  # $99/month average revenue per user
        churn_rate = 0.05  # 5% monthly churn

        users = current_users

        for month in range(months):
            date = base_date + timedelta(days=30 * month)

            # Calculate users with growth and churn
            new_users = users * monthly_growth_rate
            churned_users = users * churn_rate
            users = users + new_users - churned_users

            # Calculate revenue
            monthly_revenue = users * arpu

            forecast.append(
                {
                    "month": date.strftime("%Y-%m"),
                    "users": int(users),
                    "revenue": round(monthly_revenue, 2),
                    "new_users": int(new_users),
                    "churned_users": int(churned_users),
                    "growth_rate": monthly_growth_rate * 100,
                    "arpu": arpu,
                }
            )

        return {
            "forecast": forecast,
            "assumptions": {
                "monthly_growth_rate": monthly_growth_rate * 100,
                "average_arpu": arpu,
                "monthly_churn_rate": churn_rate * 100,
            },
            "projections": {
                "year_1_revenue": sum(f["revenue"] for f in forecast),
                "year_1_users": forecast[-1]["users"] if forecast else 0,
                "peak_monthly_revenue": max(f["revenue"] for f in forecast)
                if forecast
                else 0,
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate revenue forecast: {str(e)}"
        )


@router.get("/cohort-analysis")
async def get_cohort_analysis(
    db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)
):
    """Get user cohort analysis"""
    try:
        # Mock cohort analysis - in production this would analyze actual user cohorts
        cohorts = []
        base_date = datetime.utcnow() - timedelta(weeks=12)

        for week in range(12):
            cohort_date = base_date + timedelta(weeks=week)
            cohort_size = 10 + week * 2  # Growing cohort sizes

            # Mock retention rates
            retention = []
            for retention_week in range(min(12 - week, 12)):
                # Typical SaaS retention curve
                if retention_week == 0:
                    rate = 100.0
                else:
                    rate = max(20.0, 100.0 * (0.85**retention_week))

                retention.append(
                    {
                        "week": retention_week,
                        "users": int(cohort_size * rate / 100),
                        "rate": round(rate, 1),
                    }
                )

            cohorts.append(
                {
                    "cohort_date": cohort_date.strftime("%Y-%m-%d"),
                    "initial_size": cohort_size,
                    "retention": retention,
                }
            )

        return cohorts

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch cohort analysis: {str(e)}"
        )


@router.get("/conversion-funnels")
async def get_conversion_funnels(
    db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)
):
    """Get conversion funnel analysis"""
    try:
        # Mock conversion funnel data - in production this would come from event tracking
        funnels = {
            "signup_to_first_video": {
                "name": "Signup to First Video",
                "steps": [
                    {"step": "Signup", "users": 100, "conversion_rate": 100.0},
                    {
                        "step": "Email Verification",
                        "users": 85,
                        "conversion_rate": 85.0,
                    },
                    {"step": "Profile Setup", "users": 72, "conversion_rate": 84.7},
                    {
                        "step": "Channel Connection",
                        "users": 58,
                        "conversion_rate": 80.6,
                    },
                    {
                        "step": "First Video Generated",
                        "users": 45,
                        "conversion_rate": 77.6,
                    },
                ],
            },
            "trial_to_paid": {
                "name": "Trial to Paid Conversion",
                "steps": [
                    {"step": "Trial Start", "users": 100, "conversion_rate": 100.0},
                    {"step": "First Video", "users": 78, "conversion_rate": 78.0},
                    {"step": "5+ Videos", "users": 52, "conversion_rate": 66.7},
                    {"step": "Pricing Page", "users": 34, "conversion_rate": 65.4},
                    {"step": "Payment", "users": 23, "conversion_rate": 67.6},
                ],
            },
            "onboarding": {
                "name": "Onboarding Completion",
                "steps": [
                    {"step": "Welcome", "users": 100, "conversion_rate": 100.0},
                    {"step": "Tutorial", "users": 89, "conversion_rate": 89.0},
                    {"step": "First Action", "users": 76, "conversion_rate": 85.4},
                    {"step": "Feature Discovery", "users": 61, "conversion_rate": 80.3},
                    {
                        "step": "Onboarding Complete",
                        "users": 54,
                        "conversion_rate": 88.5,
                    },
                ],
            },
        }

        return funnels

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch conversion funnels: {str(e)}"
        )
