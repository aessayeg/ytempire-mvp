"""
Automated Reporting System for YTEmpire
Generates and distributes quality reports automatically
"""

import asyncio
import json
import smtplib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import io
import base64

import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Template
from weasyprint import HTML
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from app.services.quality_metrics import QualityMetricsCollector, QualityDashboard
from app.services.defect_tracking import DefectTracker

logger = logging.getLogger(__name__)

# ============================================================================
# Report Configuration
# ============================================================================

class ReportType(str, Enum):
    """Types of automated reports"""
    DAILY_SUMMARY = "daily_summary"
    WEEKLY_QUALITY = "weekly_quality"
    MONTHLY_EXECUTIVE = "monthly_executive"
    INCIDENT_REPORT = "incident_report"
    SLA_COMPLIANCE = "sla_compliance"
    PERFORMANCE_TREND = "performance_trend"
    DEFECT_ANALYSIS = "defect_analysis"
    USER_FEEDBACK = "user_feedback"

class ReportFormat(str, Enum):
    """Report output formats"""
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"

class DeliveryMethod(str, Enum):
    """Report delivery methods"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    S3 = "s3"
    LOCAL_FILE = "local_file"

@dataclass
class ReportSchedule:
    """Report scheduling configuration"""
    name: str
    report_type: ReportType
    format: ReportFormat
    delivery_method: DeliveryMethod
    recipients: List[str]
    schedule_cron: str  # Cron expression
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None

@dataclass
class ReportTemplate:
    """Report template configuration"""
    name: str
    report_type: ReportType
    html_template: str
    css_styles: str
    include_charts: bool = True
    include_tables: bool = True
    custom_sections: List[str] = None

# ============================================================================
# Report Generator
# ============================================================================

class ReportGenerator:
    """Generates various types of reports"""
    
    def __init__(
        self,
        metrics_collector: QualityMetricsCollector,
        quality_dashboard: QualityDashboard,
        defect_tracker: DefectTracker
    ):
        self.metrics_collector = metrics_collector
        self.quality_dashboard = quality_dashboard
        self.defect_tracker = defect_tracker
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[ReportType, ReportTemplate]:
        """Load report templates"""
        return {
            ReportType.DAILY_SUMMARY: ReportTemplate(
                name="Daily Summary",
                report_type=ReportType.DAILY_SUMMARY,
                html_template=self._get_daily_summary_template(),
                css_styles=self._get_base_css()
            ),
            ReportType.WEEKLY_QUALITY: ReportTemplate(
                name="Weekly Quality Report",
                report_type=ReportType.WEEKLY_QUALITY,
                html_template=self._get_weekly_quality_template(),
                css_styles=self._get_base_css()
            ),
            ReportType.MONTHLY_EXECUTIVE: ReportTemplate(
                name="Monthly Executive Report",
                report_type=ReportType.MONTHLY_EXECUTIVE,
                html_template=self._get_executive_template(),
                css_styles=self._get_executive_css()
            )
        }
    
    async def generate_daily_summary(self, date: datetime) -> Dict[str, Any]:
        """Generate daily summary report"""
        start_time = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = start_time + timedelta(days=1)
        
        # Get latest metrics
        latest_metrics = await self.metrics_collector.get_latest_metrics()
        
        # Get defects created today
        defects_today = await self.defect_tracker.get_defects(
            filters={"created_after": start_time, "created_before": end_time}
        )
        
        # Calculate daily statistics
        daily_stats = {
            "date": date.strftime("%Y-%m-%d"),
            "system_status": "healthy",
            "total_requests": 0,
            "error_count": 0,
            "avg_response_time": 0,
            "videos_generated": 0,
            "generation_success_rate": 0,
            "new_defects": len(defects_today),
            "critical_issues": len([d for d in defects_today if d.severity.value == "critical"]),
            "uptime_percentage": 0
        }
        
        # Process metrics
        for metric_id, metric_value in latest_metrics.items():
            if metric_id == "api_response_time_p95":
                daily_stats["avg_response_time"] = metric_value.value
            elif metric_id == "video_generation_success_rate":
                daily_stats["generation_success_rate"] = metric_value.value
            elif metric_id == "system_uptime":
                daily_stats["uptime_percentage"] = metric_value.value
                if metric_value.value < 99:
                    daily_stats["system_status"] = "degraded"
            elif metric_id == "error_rate":
                daily_stats["error_count"] = metric_value.value
        
        # Determine overall system status
        critical_defects = len([d for d in defects_today if d.severity.value == "critical"])
        if critical_defects > 0:
            daily_stats["system_status"] = "critical"
        elif daily_stats["uptime_percentage"] < 99.5:
            daily_stats["system_status"] = "warning"
        
        return {
            "report_type": "daily_summary",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data": daily_stats,
            "metrics": {k: v.value for k, v in latest_metrics.items()},
            "defects": [
                {
                    "id": d.id,
                    "title": d.title,
                    "severity": d.severity.value,
                    "status": d.status.value
                } for d in defects_today
            ]
        }
    
    async def generate_weekly_quality_report(self, week_start: datetime) -> Dict[str, Any]:
        """Generate weekly quality report"""
        week_end = week_start + timedelta(days=7)
        
        # Get quality overview
        quality_overview = await self.quality_dashboard.generate_quality_overview()
        
        # Get trend analysis
        trends = await self.quality_dashboard.generate_trend_analysis(days=7)
        
        # Get week's defects
        week_defects = await self.defect_tracker.get_defects(
            filters={"created_after": week_start, "created_before": week_end}
        )
        
        # Get defect statistics
        defect_stats = await self.defect_tracker.get_defect_statistics()
        
        # Calculate weekly metrics
        weekly_data = {
            "week_start": week_start.strftime("%Y-%m-%d"),
            "week_end": week_end.strftime("%Y-%m-%d"),
            "overall_quality_score": quality_overview.get("quality_score", 0),
            "system_status": quality_overview.get("overall_status", "unknown"),
            "total_defects_created": len(week_defects),
            "defects_resolved": len([d for d in week_defects if d.status.value in ["resolved", "closed"]]),
            "critical_incidents": len([d for d in week_defects if d.severity.value == "critical"]),
            "avg_resolution_time": defect_stats.get("avg_resolution_time_hours", 0),
            "sla_compliance": defect_stats.get("sla_compliance", {}),
            "top_issues": self._get_top_issues(week_defects),
            "performance_trends": trends.get("metrics", {})
        }
        
        return {
            "report_type": "weekly_quality",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data": weekly_data,
            "quality_overview": quality_overview,
            "trends": trends,
            "charts": await self._generate_weekly_charts(weekly_data)
        }
    
    async def generate_monthly_executive_report(self, month_start: datetime) -> Dict[str, Any]:
        """Generate monthly executive report"""
        month_end = (month_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        
        # Get monthly metrics
        monthly_trends = await self.quality_dashboard.generate_trend_analysis(days=30)
        
        # Get monthly defects
        monthly_defects = await self.defect_tracker.get_defects(
            filters={"created_after": month_start, "created_before": month_end}
        )
        
        # Calculate business metrics
        business_metrics = {
            "total_videos_generated": 0,
            "total_cost": 0,
            "avg_cost_per_video": 0,
            "user_satisfaction": 0,
            "system_uptime": 0,
            "revenue_impact": 0
        }
        
        # Process trends for business insights
        for metric_id, trend in monthly_trends.get("metrics", {}).items():
            if metric_id == "cost_per_video":
                business_metrics["avg_cost_per_video"] = trend.get("average", 0)
            elif metric_id == "user_satisfaction_score":
                business_metrics["user_satisfaction"] = trend.get("average", 0)
            elif metric_id == "system_uptime":
                business_metrics["system_uptime"] = trend.get("average", 0)
        
        executive_summary = {
            "month": month_start.strftime("%B %Y"),
            "executive_summary": self._generate_executive_summary(business_metrics, monthly_defects),
            "key_achievements": self._identify_key_achievements(monthly_trends),
            "areas_for_improvement": self._identify_improvement_areas(monthly_defects, monthly_trends),
            "business_impact": business_metrics,
            "quality_metrics": monthly_trends,
            "risk_assessment": self._assess_risks(monthly_defects),
            "recommendations": self._generate_recommendations(monthly_defects, monthly_trends)
        }
        
        return {
            "report_type": "monthly_executive",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data": executive_summary,
            "charts": await self._generate_executive_charts(executive_summary)
        }
    
    def _get_top_issues(self, defects: List) -> List[Dict]:
        """Get top issues from defects"""
        # Group by component and severity
        component_issues = {}
        for defect in defects:
            component = defect.component or "Unknown"
            if component not in component_issues:
                component_issues[component] = {"total": 0, "critical": 0, "high": 0}
            
            component_issues[component]["total"] += 1
            if defect.severity.value == "critical":
                component_issues[component]["critical"] += 1
            elif defect.severity.value == "high":
                component_issues[component]["high"] += 1
        
        # Sort by severity and count
        sorted_issues = sorted(
            component_issues.items(),
            key=lambda x: (x[1]["critical"], x[1]["high"], x[1]["total"]),
            reverse=True
        )
        
        return [
            {
                "component": component,
                "total_issues": data["total"],
                "critical_issues": data["critical"],
                "high_issues": data["high"]
            }
            for component, data in sorted_issues[:5]
        ]
    
    def _generate_executive_summary(self, business_metrics: Dict, defects: List) -> str:
        """Generate executive summary text"""
        summary_parts = []
        
        # System performance
        uptime = business_metrics.get("system_uptime", 0)
        if uptime >= 99.9:
            summary_parts.append("System demonstrated excellent reliability with >99.9% uptime.")
        elif uptime >= 99:
            summary_parts.append("System maintained good reliability with >99% uptime.")
        else:
            summary_parts.append(f"System uptime of {uptime:.2f}% requires attention.")
        
        # Cost efficiency
        cost_per_video = business_metrics.get("avg_cost_per_video", 0)
        if cost_per_video < 2.0:
            summary_parts.append(f"Video generation costs optimized to ${cost_per_video:.2f} per video.")
        elif cost_per_video < 3.0:
            summary_parts.append(f"Video generation costs at ${cost_per_video:.2f} per video within target.")
        else:
            summary_parts.append(f"Video generation costs at ${cost_per_video:.2f} per video exceed target.")
        
        # Quality issues
        critical_defects = len([d for d in defects if d.severity.value == "critical"])
        if critical_defects == 0:
            summary_parts.append("No critical issues reported this month.")
        else:
            summary_parts.append(f"{critical_defects} critical issues requiring immediate attention.")
        
        return " ".join(summary_parts)
    
    def _identify_key_achievements(self, trends: Dict) -> List[str]:
        """Identify key achievements from trends"""
        achievements = []
        
        for metric_id, trend in trends.get("metrics", {}).items():
            if trend.get("trend_direction") == "improving" and trend.get("change_percent", 0) > 10:
                achievements.append(f"{metric_id.replace('_', ' ').title()} improved by {trend['change_percent']:.1f}%")
        
        if not achievements:
            achievements.append("System maintained stable performance metrics")
        
        return achievements
    
    def _identify_improvement_areas(self, defects: List, trends: Dict) -> List[str]:
        """Identify areas needing improvement"""
        areas = []
        
        # Check for declining trends
        for metric_id, trend in trends.get("metrics", {}).items():
            if trend.get("trend_direction") == "declining" and trend.get("change_percent", 0) < -5:
                areas.append(f"{metric_id.replace('_', ' ').title()} declined by {abs(trend['change_percent']):.1f}%")
        
        # Check for high defect counts in specific components
        component_counts = {}
        for defect in defects:
            component = defect.component or "Unknown"
            component_counts[component] = component_counts.get(component, 0) + 1
        
        for component, count in component_counts.items():
            if count >= 5:
                areas.append(f"{component} component has {count} reported issues")
        
        return areas
    
    def _assess_risks(self, defects: List) -> Dict[str, Any]:
        """Assess risks based on defects and trends"""
        critical_count = len([d for d in defects if d.severity.value == "critical"])
        high_count = len([d for d in defects if d.severity.value == "high"])
        
        risk_level = "LOW"
        if critical_count > 0:
            risk_level = "HIGH"
        elif high_count >= 5:
            risk_level = "MEDIUM"
        
        return {
            "overall_risk_level": risk_level,
            "critical_issues": critical_count,
            "high_priority_issues": high_count,
            "risk_factors": [
                f"{critical_count} critical issues" if critical_count > 0 else None,
                f"{high_count} high priority issues" if high_count >= 3 else None
            ]
        }
    
    def _generate_recommendations(self, defects: List, trends: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Defect-based recommendations
        critical_defects = [d for d in defects if d.severity.value == "critical"]
        if critical_defects:
            recommendations.append("Immediate focus on resolving critical defects to prevent system degradation")
        
        # Trend-based recommendations
        for metric_id, trend in trends.get("metrics", {}).items():
            if trend.get("trend_direction") == "declining":
                if "response_time" in metric_id:
                    recommendations.append("Implement performance optimization to improve response times")
                elif "success_rate" in metric_id:
                    recommendations.append("Investigate and fix reliability issues affecting success rates")
        
        if not recommendations:
            recommendations.append("Continue monitoring current performance levels and maintain system health")
        
        return recommendations
    
    async def _generate_weekly_charts(self, data: Dict) -> Dict[str, str]:
        """Generate charts for weekly report"""
        charts = {}
        
        # Quality score chart
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=data["overall_quality_score"],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Quality Score"},
            delta={'reference': 90},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 70], 'color': "lightgray"},
                    {'range': [70, 85], 'color': "gray"},
                    {'range': [85, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        charts["quality_score"] = fig.to_html(include_plotlyjs=False)
        
        return charts
    
    async def _generate_executive_charts(self, data: Dict) -> Dict[str, str]:
        """Generate charts for executive report"""
        charts = {}
        
        # Business metrics chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('System Uptime', 'Cost per Video', 'User Satisfaction', 'Quality Score')
        )
        
        business_metrics = data["business_impact"]
        
        # Add gauges for each metric
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=business_metrics.get("system_uptime", 0),
            domain={'row': 0, 'column': 0},
            gauge={'axis': {'range': [0, 100]}}
        ), row=1, col=1)
        
        charts["business_overview"] = fig.to_html(include_plotlyjs=False)
        
        return charts
    
    def _get_daily_summary_template(self) -> str:
        """Get daily summary HTML template"""
        return """
        <html>
        <head>
            <title>Daily Summary - {{ date }}</title>
            <style>{{ css_styles }}</style>
        </head>
        <body>
            <div class="header">
                <h1>YTEmpire Daily Summary</h1>
                <p class="date">{{ date }}</p>
            </div>
            
            <div class="status-card {{ system_status }}">
                <h2>System Status: {{ system_status|title }}</h2>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Uptime</h3>
                    <span class="metric-value">{{ uptime_percentage }}%</span>
                </div>
                <div class="metric-card">
                    <h3>Response Time</h3>
                    <span class="metric-value">{{ avg_response_time }}ms</span>
                </div>
                <div class="metric-card">
                    <h3>Success Rate</h3>
                    <span class="metric-value">{{ generation_success_rate }}%</span>
                </div>
                <div class="metric-card">
                    <h3>New Issues</h3>
                    <span class="metric-value">{{ new_defects }}</span>
                </div>
            </div>
            
            {% if critical_issues > 0 %}
            <div class="alert-section">
                <h2>⚠️ Critical Issues</h2>
                <p>{{ critical_issues }} critical issues require immediate attention</p>
            </div>
            {% endif %}
        </body>
        </html>
        """
    
    def _get_weekly_quality_template(self) -> str:
        """Get weekly quality report HTML template"""
        return """
        <html>
        <head>
            <title>Weekly Quality Report</title>
            <style>{{ css_styles }}</style>
        </head>
        <body>
            <div class="header">
                <h1>Weekly Quality Report</h1>
                <p class="date-range">{{ week_start }} - {{ week_end }}</p>
            </div>
            
            <div class="quality-score">
                <h2>Overall Quality Score: {{ overall_quality_score }}/100</h2>
                {{ charts.quality_score|safe }}
            </div>
            
            <div class="metrics-section">
                <h2>Key Metrics</h2>
                <table class="metrics-table">
                    <tr>
                        <td>Total Defects Created</td>
                        <td>{{ total_defects_created }}</td>
                    </tr>
                    <tr>
                        <td>Defects Resolved</td>
                        <td>{{ defects_resolved }}</td>
                    </tr>
                    <tr>
                        <td>Average Resolution Time</td>
                        <td>{{ avg_resolution_time }} hours</td>
                    </tr>
                </table>
            </div>
        </body>
        </html>
        """
    
    def _get_executive_template(self) -> str:
        """Get executive report HTML template"""
        return """
        <html>
        <head>
            <title>Monthly Executive Report</title>
            <style>{{ css_styles }}</style>
        </head>
        <body>
            <div class="executive-header">
                <h1>Monthly Executive Report</h1>
                <p class="month">{{ month }}</p>
            </div>
            
            <div class="executive-summary">
                <h2>Executive Summary</h2>
                <p>{{ executive_summary }}</p>
            </div>
            
            <div class="achievements">
                <h2>Key Achievements</h2>
                <ul>
                {% for achievement in key_achievements %}
                    <li>{{ achievement }}</li>
                {% endfor %}
                </ul>
            </div>
            
            <div class="recommendations">
                <h2>Recommendations</h2>
                <ul>
                {% for recommendation in recommendations %}
                    <li>{{ recommendation }}</li>
                {% endfor %}
                </ul>
            </div>
            
            {{ charts.business_overview|safe }}
        </body>
        </html>
        """
    
    def _get_base_css(self) -> str:
        """Get base CSS styles"""
        return """
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .status-card { padding: 20px; border-radius: 8px; margin: 20px 0; }
        .status-card.healthy { background-color: #d4edda; border: 1px solid #c3e6cb; }
        .status-card.warning { background-color: #fff3cd; border: 1px solid #ffeaa7; }
        .status-card.critical { background-color: #f8d7da; border: 1px solid #f5c6cb; }
        .metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; }
        .metric-card { background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }
        .metric-value { font-size: 2em; font-weight: bold; color: #007bff; }
        .alert-section { background: #f8d7da; padding: 20px; border-radius: 8px; margin: 20px 0; }
        """
    
    def _get_executive_css(self) -> str:
        """Get executive report CSS styles"""
        return self._get_base_css() + """
        .executive-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; border-radius: 12px; }
        .executive-summary { background: #e3f2fd; padding: 30px; border-radius: 8px; margin: 30px 0; }
        .achievements, .recommendations { margin: 30px 0; }
        .achievements ul, .recommendations ul { list-style-type: none; padding: 0; }
        .achievements li, .recommendations li { background: #f1f8e9; margin: 10px 0; padding: 15px; border-left: 4px solid #4caf50; }
        """


# ============================================================================
# Report Delivery System
# ============================================================================

class ReportDelivery:
    """Handles report delivery via various methods"""
    
    def __init__(self, smtp_config: Optional[Dict] = None):
        self.smtp_config = smtp_config or {}
    
    async def deliver_report(
        self,
        report_data: Dict,
        delivery_method: DeliveryMethod,
        recipients: List[str],
        format: ReportFormat = ReportFormat.HTML
    ) -> bool:
        """Deliver report via specified method"""
        
        try:
            if delivery_method == DeliveryMethod.EMAIL:
                return await self._deliver_via_email(report_data, recipients, format)
            elif delivery_method == DeliveryMethod.LOCAL_FILE:
                return await self._deliver_to_file(report_data, format)
            # Add other delivery methods as needed
            
        except Exception as e:
            logger.error(f"Failed to deliver report: {e}")
            return False
    
    async def _deliver_via_email(
        self,
        report_data: Dict,
        recipients: List[str],
        format: ReportFormat
    ) -> bool:
        """Deliver report via email"""
        
        if not self.smtp_config:
            logger.warning("SMTP not configured, cannot send email")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_config.get('from_email', 'reports@ytempire.com')
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"YTEmpire {report_data['report_type'].replace('_', ' ').title()} Report"
            
            # Add HTML body
            if 'html_content' in report_data:
                msg.attach(MIMEText(report_data['html_content'], 'html'))
            else:
                msg.attach(MIMEText(json.dumps(report_data, indent=2), 'plain'))
            
            # Send email
            with smtplib.SMTP(self.smtp_config['host'], self.smtp_config['port']) as server:
                if self.smtp_config.get('use_tls'):
                    server.starttls()
                if self.smtp_config.get('username'):
                    server.login(self.smtp_config['username'], self.smtp_config['password'])
                server.send_message(msg)
            
            logger.info(f"Report emailed to {len(recipients)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    async def _deliver_to_file(self, report_data: Dict, format: ReportFormat) -> bool:
        """Save report to local file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reports/{report_data['report_type']}_{timestamp}"
            
            if format == ReportFormat.JSON:
                filename += ".json"
                with open(filename, 'w') as f:
                    json.dump(report_data, f, indent=2, default=str)
            elif format == ReportFormat.HTML:
                filename += ".html"
                with open(filename, 'w') as f:
                    f.write(report_data.get('html_content', ''))
            
            logger.info(f"Report saved to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            return False


# ============================================================================
# Automated Reporting Scheduler
# ============================================================================

class AutomatedReporting:
    """Main automated reporting system"""
    
    def __init__(
        self,
        generator: ReportGenerator,
        delivery: ReportDelivery
    ):
        self.generator = generator
        self.delivery = delivery
        self.schedules = []
        self.running_tasks = {}
    
    def add_schedule(self, schedule: ReportSchedule):
        """Add a report schedule"""
        self.schedules.append(schedule)
        logger.info(f"Added report schedule: {schedule.name}")
    
    async def start_scheduler(self):
        """Start the automated reporting scheduler"""
        logger.info("Starting automated reporting scheduler")
        
        for schedule in self.schedules:
            if schedule.enabled:
                task = asyncio.create_task(self._schedule_reports(schedule))
                self.running_tasks[schedule.name] = task
    
    async def stop_scheduler(self):
        """Stop the automated reporting scheduler"""
        logger.info("Stopping automated reporting scheduler")
        
        for task in self.running_tasks.values():
            task.cancel()
        
        self.running_tasks.clear()
    
    async def _schedule_reports(self, schedule: ReportSchedule):
        """Schedule and run reports based on schedule"""
        while True:
            try:
                # Simple daily scheduling (would use proper cron parser in production)
                now = datetime.now()
                next_run = now.replace(hour=8, minute=0, second=0, microsecond=0)
                
                if next_run <= now:
                    next_run += timedelta(days=1)
                
                wait_seconds = (next_run - now).total_seconds()
                await asyncio.sleep(wait_seconds)
                
                # Generate and deliver report
                await self._execute_scheduled_report(schedule)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduled report {schedule.name}: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retry
    
    async def _execute_scheduled_report(self, schedule: ReportSchedule):
        """Execute a scheduled report"""
        try:
            logger.info(f"Executing scheduled report: {schedule.name}")
            
            # Generate report based on type
            if schedule.report_type == ReportType.DAILY_SUMMARY:
                report_data = await self.generator.generate_daily_summary(datetime.now())
            elif schedule.report_type == ReportType.WEEKLY_QUALITY:
                week_start = datetime.now() - timedelta(days=7)
                report_data = await self.generator.generate_weekly_quality_report(week_start)
            elif schedule.report_type == ReportType.MONTHLY_EXECUTIVE:
                month_start = datetime.now().replace(day=1)
                report_data = await self.generator.generate_monthly_executive_report(month_start)
            else:
                logger.warning(f"Unknown report type: {schedule.report_type}")
                return
            
            # Render HTML if needed
            if schedule.format == ReportFormat.HTML:
                template = self.generator.templates.get(schedule.report_type)
                if template:
                    jinja_template = Template(template.html_template)
                    html_content = jinja_template.render(
                        css_styles=template.css_styles,
                        **report_data.get('data', {})
                    )
                    report_data['html_content'] = html_content
            
            # Deliver report
            success = await self.delivery.deliver_report(
                report_data,
                schedule.delivery_method,
                schedule.recipients,
                schedule.format
            )
            
            if success:
                schedule.last_run = datetime.now()
                logger.info(f"Successfully delivered report: {schedule.name}")
            else:
                logger.error(f"Failed to deliver report: {schedule.name}")
                
        except Exception as e:
            logger.error(f"Failed to execute scheduled report {schedule.name}: {e}")


# ============================================================================
# Global Instances and Setup
# ============================================================================

def setup_automated_reporting(
    metrics_collector: QualityMetricsCollector,
    quality_dashboard: QualityDashboard,
    defect_tracker: DefectTracker,
    smtp_config: Optional[Dict] = None
) -> AutomatedReporting:
    """Set up automated reporting system"""
    
    generator = ReportGenerator(metrics_collector, quality_dashboard, defect_tracker)
    delivery = ReportDelivery(smtp_config)
    reporting = AutomatedReporting(generator, delivery)
    
    # Add default schedules
    reporting.add_schedule(ReportSchedule(
        name="Daily Summary",
        report_type=ReportType.DAILY_SUMMARY,
        format=ReportFormat.HTML,
        delivery_method=DeliveryMethod.EMAIL,
        recipients=["ops@ytempire.com"],
        schedule_cron="0 8 * * *"  # Daily at 8 AM
    ))
    
    reporting.add_schedule(ReportSchedule(
        name="Weekly Quality Report",
        report_type=ReportType.WEEKLY_QUALITY,
        format=ReportFormat.HTML,
        delivery_method=DeliveryMethod.EMAIL,
        recipients=["management@ytempire.com", "ops@ytempire.com"],
        schedule_cron="0 9 * * 1"  # Weekly on Monday at 9 AM
    ))
    
    reporting.add_schedule(ReportSchedule(
        name="Monthly Executive Report",
        report_type=ReportType.MONTHLY_EXECUTIVE,
        format=ReportFormat.PDF,
        delivery_method=DeliveryMethod.EMAIL,
        recipients=["executives@ytempire.com"],
        schedule_cron="0 10 1 * *"  # Monthly on 1st at 10 AM
    ))
    
    return reporting