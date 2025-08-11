"""
Reporting Infrastructure
Business intelligence and reporting system
"""
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, date
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
import logging
from io import BytesIO
import json

logger = logging.getLogger(__name__)


class ReportType(Enum):
    """Types of reports"""
    DAILY_SUMMARY = "daily_summary"
    WEEKLY_PERFORMANCE = "weekly_performance"
    MONTHLY_ANALYTICS = "monthly_analytics"
    CHANNEL_REPORT = "channel_report"
    REVENUE_REPORT = "revenue_report"
    CONTENT_REPORT = "content_report"
    COST_ANALYSIS = "cost_analysis"
    TREND_REPORT = "trend_report"
    CUSTOM = "custom"


class ReportFormat(Enum):
    """Report output formats"""
    JSON = "json"
    PDF = "pdf"
    EXCEL = "excel"
    CSV = "csv"
    HTML = "html"


@dataclass
class ReportConfig:
    """Report configuration"""
    report_type: ReportType
    format: ReportFormat
    period_start: datetime
    period_end: datetime
    filters: Dict[str, Any]
    include_charts: bool = True
    include_recommendations: bool = True


@dataclass
class ReportSection:
    """Report section"""
    title: str
    data: Any
    chart_type: Optional[str] = None
    summary: Optional[str] = None


class ReportGenerator:
    """Main report generation system"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.report_cache = {}
    
    async def generate_report(
        self,
        config: ReportConfig,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate report based on configuration"""
        
        # Check cache
        cache_key = self._get_cache_key(config, user_id)
        if cache_key in self.report_cache:
            cached = self.report_cache[cache_key]
            if (datetime.utcnow() - cached['timestamp']).seconds < 3600:
                return cached['report']
        
        # Generate report based on type
        if config.report_type == ReportType.DAILY_SUMMARY:
            report = await self._generate_daily_summary(config, user_id)
        elif config.report_type == ReportType.WEEKLY_PERFORMANCE:
            report = await self._generate_weekly_performance(config, user_id)
        elif config.report_type == ReportType.MONTHLY_ANALYTICS:
            report = await self._generate_monthly_analytics(config, user_id)
        elif config.report_type == ReportType.REVENUE_REPORT:
            report = await self._generate_revenue_report(config, user_id)
        elif config.report_type == ReportType.CONTENT_REPORT:
            report = await self._generate_content_report(config, user_id)
        elif config.report_type == ReportType.COST_ANALYSIS:
            report = await self._generate_cost_analysis(config, user_id)
        else:
            report = await self._generate_custom_report(config, user_id)
        
        # Cache report
        self.report_cache[cache_key] = {
            'report': report,
            'timestamp': datetime.utcnow()
        }
        
        # Format report
        formatted = await self._format_report(report, config.format)
        
        return formatted
    
    async def _generate_daily_summary(
        self,
        config: ReportConfig,
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Generate daily summary report"""
        
        sections = []
        
        # Key metrics
        metrics = await self._get_key_metrics(
            config.period_start,
            config.period_end,
            user_id
        )
        sections.append(ReportSection(
            title="Key Metrics",
            data=metrics,
            chart_type="scorecard",
            summary=self._summarize_metrics(metrics)
        ))
        
        # Video performance
        video_stats = await self._get_video_statistics(
            config.period_start,
            config.period_end,
            user_id
        )
        sections.append(ReportSection(
            title="Video Performance",
            data=video_stats,
            chart_type="bar",
            summary=f"Published {video_stats['total_videos']} videos"
        ))
        
        # Channel health
        channel_health = await self._get_channel_health(user_id)
        sections.append(ReportSection(
            title="Channel Health",
            data=channel_health,
            chart_type="radar"
        ))
        
        # Recommendations
        if config.include_recommendations:
            recommendations = await self._generate_recommendations(
                metrics, video_stats, channel_health
            )
            sections.append(ReportSection(
                title="Recommendations",
                data=recommendations
            ))
        
        return {
            "title": "Daily Summary Report",
            "date": config.period_start.date().isoformat(),
            "sections": sections,
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "user_id": user_id
            }
        }
    
    async def _generate_weekly_performance(
        self,
        config: ReportConfig,
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Generate weekly performance report"""
        
        sections = []
        
        # Week over week comparison
        current_week = await self._get_week_metrics(
            config.period_start,
            config.period_end,
            user_id
        )
        
        previous_start = config.period_start - timedelta(days=7)
        previous_end = config.period_end - timedelta(days=7)
        previous_week = await self._get_week_metrics(
            previous_start,
            previous_end,
            user_id
        )
        
        comparison = self._compare_periods(current_week, previous_week)
        sections.append(ReportSection(
            title="Week over Week",
            data=comparison,
            chart_type="line"
        ))
        
        # Top performing content
        top_content = await self._get_top_content(
            config.period_start,
            config.period_end,
            user_id,
            limit=10
        )
        sections.append(ReportSection(
            title="Top Performing Content",
            data=top_content,
            chart_type="table"
        ))
        
        # Engagement trends
        engagement = await self._get_engagement_trends(
            config.period_start,
            config.period_end,
            user_id
        )
        sections.append(ReportSection(
            title="Engagement Trends",
            data=engagement,
            chart_type="area"
        ))
        
        return {
            "title": "Weekly Performance Report",
            "period": f"{config.period_start.date()} to {config.period_end.date()}",
            "sections": sections,
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "user_id": user_id
            }
        }
    
    async def _generate_revenue_report(
        self,
        config: ReportConfig,
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Generate revenue report"""
        
        sections = []
        
        # Revenue summary
        revenue_data = await self._get_revenue_data(
            config.period_start,
            config.period_end,
            user_id
        )
        sections.append(ReportSection(
            title="Revenue Summary",
            data=revenue_data,
            chart_type="line",
            summary=f"Total revenue: ${revenue_data['total']:,.2f}"
        ))
        
        # Revenue by source
        by_source = await self._get_revenue_by_source(
            config.period_start,
            config.period_end,
            user_id
        )
        sections.append(ReportSection(
            title="Revenue by Source",
            data=by_source,
            chart_type="pie"
        ))
        
        # Revenue by channel
        by_channel = await self._get_revenue_by_channel(
            config.period_start,
            config.period_end,
            user_id
        )
        sections.append(ReportSection(
            title="Revenue by Channel",
            data=by_channel,
            chart_type="bar"
        ))
        
        # Projections
        projections = self._calculate_revenue_projections(revenue_data)
        sections.append(ReportSection(
            title="Revenue Projections",
            data=projections,
            chart_type="line"
        ))
        
        return {
            "title": "Revenue Report",
            "period": f"{config.period_start.date()} to {config.period_end.date()}",
            "sections": sections,
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "user_id": user_id,
                "currency": "USD"
            }
        }
    
    async def _generate_cost_analysis(
        self,
        config: ReportConfig,
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Generate cost analysis report"""
        
        sections = []
        
        # Cost breakdown
        costs = await self._get_cost_breakdown(
            config.period_start,
            config.period_end,
            user_id
        )
        sections.append(ReportSection(
            title="Cost Breakdown",
            data=costs,
            chart_type="treemap",
            summary=f"Total costs: ${costs['total']:,.2f}"
        ))
        
        # Cost per video
        cost_per_video = await self._get_cost_per_video(
            config.period_start,
            config.period_end,
            user_id
        )
        sections.append(ReportSection(
            title="Cost per Video",
            data=cost_per_video,
            chart_type="scatter"
        ))
        
        # ROI analysis
        roi = await self._calculate_roi(
            config.period_start,
            config.period_end,
            user_id
        )
        sections.append(ReportSection(
            title="ROI Analysis",
            data=roi,
            chart_type="gauge"
        ))
        
        # Cost optimization opportunities
        optimizations = await self._identify_cost_optimizations(costs)
        sections.append(ReportSection(
            title="Optimization Opportunities",
            data=optimizations
        ))
        
        return {
            "title": "Cost Analysis Report",
            "period": f"{config.period_start.date()} to {config.period_end.date()}",
            "sections": sections,
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "user_id": user_id
            }
        }
    
    async def _get_key_metrics(
        self,
        start: datetime,
        end: datetime,
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Get key performance metrics"""
        # Simplified query - would be actual database queries
        return {
            "total_views": 150000,
            "total_revenue": 2500.00,
            "videos_published": 25,
            "subscriber_growth": 1200,
            "avg_watch_time": 480,  # seconds
            "engagement_rate": 0.065,
            "ctr": 0.042
        }
    
    async def _get_video_statistics(
        self,
        start: datetime,
        end: datetime,
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Get video statistics"""
        return {
            "total_videos": 25,
            "successful": 23,
            "failed": 2,
            "avg_generation_time": 420,  # seconds
            "avg_quality_score": 0.82,
            "by_category": {
                "tutorial": 10,
                "review": 8,
                "entertainment": 7
            }
        }
    
    async def _get_channel_health(self, user_id: Optional[str]) -> Dict[str, Any]:
        """Get channel health metrics"""
        return {
            "health_score": 0.85,
            "quota_usage": 0.65,
            "compliance_score": 1.0,
            "posting_consistency": 0.9,
            "audience_retention": 0.72
        }
    
    async def _get_revenue_data(
        self,
        start: datetime,
        end: datetime,
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Get revenue data"""
        days = (end - start).days
        daily_revenue = []
        
        for i in range(days):
            day = start + timedelta(days=i)
            daily_revenue.append({
                "date": day.date().isoformat(),
                "revenue": np.random.uniform(50, 150)  # Simulated
            })
        
        total = sum(d["revenue"] for d in daily_revenue)
        
        return {
            "total": total,
            "daily_average": total / days,
            "daily_data": daily_revenue,
            "growth_rate": 0.15  # 15% growth
        }
    
    async def _get_cost_breakdown(
        self,
        start: datetime,
        end: datetime,
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Get cost breakdown"""
        return {
            "total": 850.00,
            "by_service": {
                "openai": 400.00,
                "elevenlabs": 200.00,
                "dalle": 150.00,
                "storage": 50.00,
                "compute": 50.00
            },
            "by_type": {
                "ai_generation": 750.00,
                "infrastructure": 100.00
            }
        }
    
    async def _calculate_roi(
        self,
        start: datetime,
        end: datetime,
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Calculate ROI"""
        revenue = 2500.00
        costs = 850.00
        roi = ((revenue - costs) / costs) * 100
        
        return {
            "roi_percentage": roi,
            "revenue": revenue,
            "costs": costs,
            "profit": revenue - costs,
            "profit_margin": (revenue - costs) / revenue
        }
    
    def _compare_periods(
        self,
        current: Dict[str, Any],
        previous: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare two periods"""
        comparison = {}
        
        for key in current:
            if key in previous and isinstance(current[key], (int, float)):
                change = ((current[key] - previous[key]) / previous[key]) * 100
                comparison[key] = {
                    "current": current[key],
                    "previous": previous[key],
                    "change_percent": change
                }
        
        return comparison
    
    def _calculate_revenue_projections(
        self,
        historical_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate revenue projections"""
        growth_rate = historical_data.get("growth_rate", 0.15)
        current_revenue = historical_data.get("total", 0)
        
        projections = []
        for month in range(1, 4):  # Next 3 months
            projected = current_revenue * (1 + growth_rate) ** month
            projections.append({
                "month": month,
                "projected_revenue": projected
            })
        
        return {
            "projections": projections,
            "growth_rate_used": growth_rate,
            "confidence": 0.75
        }
    
    async def _generate_recommendations(
        self,
        metrics: Dict,
        video_stats: Dict,
        channel_health: Dict
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Based on metrics
        if metrics.get("ctr", 0) < 0.05:
            recommendations.append("Improve thumbnails to increase CTR")
        
        if metrics.get("engagement_rate", 0) < 0.05:
            recommendations.append("Focus on more engaging content formats")
        
        # Based on video stats
        if video_stats.get("avg_quality_score", 0) < 0.8:
            recommendations.append("Review content quality guidelines")
        
        # Based on channel health
        if channel_health.get("posting_consistency", 1) < 0.8:
            recommendations.append("Maintain consistent posting schedule")
        
        return recommendations
    
    async def _format_report(
        self,
        report: Dict[str, Any],
        format: ReportFormat
    ) -> Any:
        """Format report based on requested format"""
        
        if format == ReportFormat.JSON:
            return report
        
        elif format == ReportFormat.CSV:
            # Convert to CSV
            df = pd.DataFrame()
            for section in report.get("sections", []):
                if isinstance(section.data, dict):
                    section_df = pd.DataFrame([section.data])
                    section_df["section"] = section.title
                    df = pd.concat([df, section_df], ignore_index=True)
            
            csv_buffer = BytesIO()
            df.to_csv(csv_buffer, index=False)
            return csv_buffer.getvalue()
        
        elif format == ReportFormat.HTML:
            # Generate HTML report
            html = self._generate_html_report(report)
            return html
        
        # For PDF and Excel, would need additional libraries
        return report
    
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML report"""
        html = f"""
        <html>
        <head>
            <title>{report.get('title', 'Report')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                .section {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>{report.get('title', 'Report')}</h1>
            <p>Generated: {report.get('metadata', {}).get('generated_at', '')}</p>
        """
        
        for section in report.get('sections', []):
            html += f"""
            <div class="section">
                <h2>{section.title}</h2>
                {self._format_section_html(section)}
            </div>
            """
        
        html += "</body></html>"
        return html
    
    def _format_section_html(self, section: ReportSection) -> str:
        """Format section as HTML"""
        if isinstance(section.data, dict):
            html = "<table>"
            for key, value in section.data.items():
                html += f"<tr><th>{key}</th><td>{value}</td></tr>"
            html += "</table>"
        elif isinstance(section.data, list):
            html = "<ul>"
            for item in section.data:
                html += f"<li>{item}</li>"
            html += "</ul>"
        else:
            html = f"<p>{section.data}</p>"
        
        if section.summary:
            html += f"<p><strong>{section.summary}</strong></p>"
        
        return html
    
    def _get_cache_key(
        self,
        config: ReportConfig,
        user_id: Optional[str]
    ) -> str:
        """Generate cache key for report"""
        key_parts = [
            config.report_type.value,
            config.period_start.date().isoformat(),
            config.period_end.date().isoformat(),
            user_id or "global"
        ]
        return "_".join(key_parts)
    
    async def _get_week_metrics(
        self,
        start: datetime,
        end: datetime,
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Get metrics for a week"""
        return await self._get_key_metrics(start, end, user_id)
    
    async def _get_top_content(
        self,
        start: datetime,
        end: datetime,
        user_id: Optional[str],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top performing content"""
        # Simulated data
        return [
            {"title": f"Video {i}", "views": 10000 - i * 500, "revenue": 100 - i * 5}
            for i in range(limit)
        ]
    
    async def _get_engagement_trends(
        self,
        start: datetime,
        end: datetime,
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Get engagement trends"""
        days = (end - start).days
        trend_data = []
        
        for i in range(days):
            day = start + timedelta(days=i)
            trend_data.append({
                "date": day.date().isoformat(),
                "likes": np.random.randint(100, 500),
                "comments": np.random.randint(20, 100),
                "shares": np.random.randint(10, 50)
            })
        
        return {"daily_engagement": trend_data}
    
    async def _get_revenue_by_source(
        self,
        start: datetime,
        end: datetime,
        user_id: Optional[str]
    ) -> Dict[str, float]:
        """Get revenue breakdown by source"""
        return {
            "ads": 1500.00,
            "sponsorships": 800.00,
            "affiliates": 200.00
        }
    
    async def _get_revenue_by_channel(
        self,
        start: datetime,
        end: datetime,
        user_id: Optional[str]
    ) -> Dict[str, float]:
        """Get revenue breakdown by channel"""
        return {
            "Channel 1": 1000.00,
            "Channel 2": 800.00,
            "Channel 3": 700.00
        }
    
    async def _get_cost_per_video(
        self,
        start: datetime,
        end: datetime,
        user_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Get cost per video"""
        return [
            {"video_id": f"vid_{i}", "cost": np.random.uniform(20, 50)}
            for i in range(25)
        ]
    
    async def _identify_cost_optimizations(
        self,
        costs: Dict[str, Any]
    ) -> List[str]:
        """Identify cost optimization opportunities"""
        optimizations = []
        
        if costs.get("by_service", {}).get("openai", 0) > 300:
            optimizations.append("Consider using GPT-3.5 for simpler tasks")
        
        if costs.get("by_service", {}).get("dalle", 0) > 100:
            optimizations.append("Optimize image generation prompts to reduce iterations")
        
        return optimizations
    
    def _summarize_metrics(self, metrics: Dict[str, Any]) -> str:
        """Generate summary from metrics"""
        return f"Generated {metrics.get('videos_published', 0)} videos with " \
               f"{metrics.get('total_views', 0):,} views and " \
               f"${metrics.get('total_revenue', 0):,.2f} in revenue"
    
    async def _generate_custom_report(
        self,
        config: ReportConfig,
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Generate custom report based on filters"""
        return {
            "title": "Custom Report",
            "sections": [],
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "filters": config.filters
            }
        }