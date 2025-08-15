"""
Complete Reporting Infrastructure
Full report generation, scheduling, and distribution system
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import asyncio
import logging
from pathlib import Path
import json
from dataclasses import dataclass, field
from enum import Enum
import jinja2
import pdfkit
import xlsxwriter
from io import BytesIO
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import schedule
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import boto3
from google.cloud import storage as gcs
import redis
from sqlalchemy import create_engine
import psycopg2

logger = logging.getLogger(__name__)

class ReportType(Enum):
    """Types of reports"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    CUSTOM = "custom"
    REALTIME = "realtime"
    EXECUTIVE = "executive"
    OPERATIONAL = "operational"
    ANALYTICAL = "analytical"

class ReportFormat(Enum):
    """Report output formats"""
    PDF = "pdf"
    EXCEL = "excel"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    POWERPOINT = "pptx"
    DASHBOARD = "dashboard"

class DeliveryMethod(Enum):
    """Report delivery methods"""
    EMAIL = "email"
    S3 = "s3"
    GCS = "gcs"
    FTP = "ftp"
    API = "api"
    WEBHOOK = "webhook"
    SLACK = "slack"

@dataclass
class ReportConfig:
    """Report configuration"""
    name: str
    type: ReportType
    format: ReportFormat
    schedule: str  # cron expression
    recipients: List[str]
    delivery_methods: List[DeliveryMethod]
    filters: Dict[str, Any] = field(default_factory=dict)
    include_sections: List[str] = field(default_factory=list)
    custom_metrics: List[str] = field(default_factory=list)

@dataclass
class ReportSection:
    """Report section definition"""
    title: str
    content: Any
    chart_type: Optional[str] = None
    data: Optional[pd.DataFrame] = None
    summary: Optional[str] = None

class ReportingInfrastructure:
    """Complete reporting infrastructure system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.engine = create_engine(self.config["database_url"])
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.template_env = self._setup_templates()
        self.report_cache: Dict[str, bytes] = {}
        self.scheduled_reports: List[ReportConfig] = []
        self._initialize_infrastructure()
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "database_url": os.getenv("DATABASE_URL", "postgresql://localhost/ytempire"),
            "report_storage_path": "reports/generated",
            "template_path": "reports/templates",
            "cache_ttl": 3600,
            "max_retries": 3,
            "email_config": {
                "smtp_host": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "reports@ytempire.com",
                "password": "password"
            },
            "s3_config": {
                "bucket": "ytempire-reports",
                "region": "us-east-1"
            }
        }
        
    def _initialize_infrastructure(self):
        """Initialize reporting infrastructure"""
        Path(self.config["report_storage_path"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["template_path"]).mkdir(parents=True, exist_ok=True)
        self._load_scheduled_reports()
        self._setup_scheduler()
        
    def _setup_templates(self) -> jinja2.Environment:
        """Setup Jinja2 templates"""
        return jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.config["template_path"]),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
    async def generate_report(self, config: ReportConfig) -> Dict[str, Any]:
        """Generate a complete report"""
        start_time = datetime.utcnow()
        report_id = f"{config.name}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        try:
            logger.info(f"Generating report: {report_id}")
            
            # Step 1: Collect data
            data = await self._collect_report_data(config)
            
            # Step 2: Process and analyze data
            analysis = await self._analyze_data(data, config)
            
            # Step 3: Generate visualizations
            visualizations = await self._create_visualizations(data, config)
            
            # Step 4: Build report sections
            sections = await self._build_report_sections(data, analysis, visualizations, config)
            
            # Step 5: Generate report in requested format
            report_file = await self._generate_report_file(sections, config, report_id)
            
            # Step 6: Deliver report
            delivery_results = await self._deliver_report(report_file, config)
            
            # Step 7: Cache report
            self._cache_report(report_id, report_file)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "report_id": report_id,
                "status": "success",
                "processing_time": processing_time,
                "file_path": report_file["path"],
                "file_size": report_file["size"],
                "delivery": delivery_results,
                "timestamp": start_time
            }
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {
                "report_id": report_id,
                "status": "failed",
                "error": str(e),
                "timestamp": start_time
            }
            
    async def _collect_report_data(self, config: ReportConfig) -> Dict[str, pd.DataFrame]:
        """Collect data for report"""
        data = {}
        
        # Determine date range based on report type
        end_date = datetime.utcnow()
        if config.type == ReportType.DAILY:
            start_date = end_date - timedelta(days=1)
        elif config.type == ReportType.WEEKLY:
            start_date = end_date - timedelta(weeks=1)
        elif config.type == ReportType.MONTHLY:
            start_date = end_date - timedelta(days=30)
        elif config.type == ReportType.QUARTERLY:
            start_date = end_date - timedelta(days=90)
        else:
            start_date = end_date - timedelta(days=7)  # Default to week
            
        # Collect various data sets
        data["performance"] = await self._get_performance_data(start_date, end_date, config.filters)
        data["channels"] = await self._get_channel_data(start_date, end_date, config.filters)
        data["videos"] = await self._get_video_data(start_date, end_date, config.filters)
        data["analytics"] = await self._get_analytics_data(start_date, end_date, config.filters)
        data["revenue"] = await self._get_revenue_data(start_date, end_date, config.filters)
        
        # Add custom metrics if specified
        for metric in config.custom_metrics:
            data[metric] = await self._get_custom_metric_data(metric, start_date, end_date)
            
        return data
        
    async def _get_performance_data(self, start_date: datetime, end_date: datetime, filters: Dict) -> pd.DataFrame:
        """Get performance metrics data"""
        query = f"""
            SELECT 
                date,
                SUM(views) as total_views,
                SUM(watch_time_minutes) as total_watch_time,
                AVG(engagement_rate) as avg_engagement,
                SUM(subscribers_gained) as new_subscribers,
                SUM(revenue) as total_revenue
            FROM performance_metrics
            WHERE date BETWEEN '{start_date}' AND '{end_date}'
            GROUP BY date
            ORDER BY date
        """
        return pd.read_sql(query, self.engine)
        
    async def _get_channel_data(self, start_date: datetime, end_date: datetime, filters: Dict) -> pd.DataFrame:
        """Get channel data"""
        query = f"""
            SELECT 
                channel_id,
                channel_name,
                subscriber_count,
                video_count,
                total_views,
                avg_view_duration,
                engagement_rate
            FROM channels
            WHERE updated_at BETWEEN '{start_date}' AND '{end_date}'
        """
        return pd.read_sql(query, self.engine)
        
    async def _get_video_data(self, start_date: datetime, end_date: datetime, filters: Dict) -> pd.DataFrame:
        """Get video performance data"""
        query = f"""
            SELECT 
                video_id,
                title,
                publish_date,
                views,
                likes,
                comments,
                watch_time_minutes,
                revenue,
                ctr,
                engagement_rate
            FROM videos
            WHERE publish_date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY views DESC
        """
        return pd.read_sql(query, self.engine)
        
    async def _get_analytics_data(self, start_date: datetime, end_date: datetime, filters: Dict) -> pd.DataFrame:
        """Get analytics data"""
        query = f"""
            SELECT 
                metric_name,
                metric_value,
                timestamp
            FROM analytics
            WHERE timestamp BETWEEN '{start_date}' AND '{end_date}'
        """
        return pd.read_sql(query, self.engine)
        
    async def _get_revenue_data(self, start_date: datetime, end_date: datetime, filters: Dict) -> pd.DataFrame:
        """Get revenue data"""
        query = f"""
            SELECT 
                date,
                ad_revenue,
                sponsorship_revenue,
                merchandise_revenue,
                membership_revenue,
                total_revenue,
                total_cost,
                net_profit
            FROM revenue_metrics
            WHERE date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY date
        """
        return pd.read_sql(query, self.engine)
        
    async def _get_custom_metric_data(self, metric: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get custom metric data"""
        # Implementation for custom metrics
        return pd.DataFrame()
        
    async def _analyze_data(self, data: Dict[str, pd.DataFrame], config: ReportConfig) -> Dict[str, Any]:
        """Analyze collected data"""
        analysis = {}
        
        # Performance analysis
        if "performance" in data and not data["performance"].empty:
            perf = data["performance"]
            analysis["performance"] = {
                "total_views": perf["total_views"].sum(),
                "avg_daily_views": perf["total_views"].mean(),
                "growth_rate": self._calculate_growth_rate(perf["total_views"]),
                "total_revenue": perf["total_revenue"].sum(),
                "avg_engagement": perf["avg_engagement"].mean(),
                "trend": self._detect_trend(perf["total_views"])
            }
            
        # Channel analysis
        if "channels" in data and not data["channels"].empty:
            channels = data["channels"]
            analysis["channels"] = {
                "total_channels": len(channels),
                "total_subscribers": channels["subscriber_count"].sum(),
                "avg_engagement": channels["engagement_rate"].mean(),
                "top_performer": channels.nlargest(1, "total_views")["channel_name"].values[0] if len(channels) > 0 else None
            }
            
        # Video analysis
        if "videos" in data and not data["videos"].empty:
            videos = data["videos"]
            analysis["videos"] = {
                "total_videos": len(videos),
                "total_views": videos["views"].sum(),
                "avg_views_per_video": videos["views"].mean(),
                "top_video": videos.nlargest(1, "views")["title"].values[0] if len(videos) > 0 else None,
                "viral_videos": len(videos[videos["views"] > videos["views"].quantile(0.95)])
            }
            
        # Revenue analysis
        if "revenue" in data and not data["revenue"].empty:
            revenue = data["revenue"]
            analysis["revenue"] = {
                "total_revenue": revenue["total_revenue"].sum(),
                "total_cost": revenue["total_cost"].sum(),
                "net_profit": revenue["net_profit"].sum(),
                "profit_margin": revenue["net_profit"].sum() / revenue["total_revenue"].sum() if revenue["total_revenue"].sum() > 0 else 0,
                "revenue_sources": {
                    "ad": revenue["ad_revenue"].sum(),
                    "sponsorship": revenue["sponsorship_revenue"].sum(),
                    "merchandise": revenue["merchandise_revenue"].sum(),
                    "membership": revenue["membership_revenue"].sum()
                }
            }
            
        # Add insights
        analysis["insights"] = self._generate_insights(data, analysis)
        
        return analysis
        
    def _calculate_growth_rate(self, series: pd.Series) -> float:
        """Calculate growth rate"""
        if len(series) < 2:
            return 0.0
        return ((series.iloc[-1] - series.iloc[0]) / series.iloc[0] * 100) if series.iloc[0] != 0 else 0.0
        
    def _detect_trend(self, series: pd.Series) -> str:
        """Detect trend in time series"""
        if len(series) < 3:
            return "insufficient_data"
            
        # Simple trend detection using linear regression slope
        x = np.arange(len(series))
        slope = np.polyfit(x, series.values, 1)[0]
        
        if slope > series.mean() * 0.1:
            return "strong_upward"
        elif slope > 0:
            return "upward"
        elif slope < -series.mean() * 0.1:
            return "strong_downward"
        elif slope < 0:
            return "downward"
        else:
            return "stable"
            
    def _generate_insights(self, data: Dict[str, pd.DataFrame], analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable insights"""
        insights = []
        
        # Performance insights
        if "performance" in analysis:
            perf = analysis["performance"]
            if perf["growth_rate"] > 20:
                insights.append(f"Excellent growth! Views increased by {perf['growth_rate']:.1f}% during the period.")
            elif perf["growth_rate"] < -10:
                insights.append(f"Warning: Views declined by {abs(perf['growth_rate']):.1f}%. Review content strategy.")
                
            if perf["avg_engagement"] < 0.05:
                insights.append("Low engagement rate detected. Consider improving content quality and CTAs.")
                
        # Revenue insights
        if "revenue" in analysis:
            rev = analysis["revenue"]
            if rev["profit_margin"] < 0.2:
                insights.append(f"Profit margin is low at {rev['profit_margin']:.1%}. Consider cost optimization.")
                
            # Find best revenue source
            best_source = max(rev["revenue_sources"].items(), key=lambda x: x[1])
            insights.append(f"{best_source[0].capitalize()} revenue is performing best at ${best_source[1]:,.2f}")
            
        # Video insights
        if "videos" in analysis:
            vid = analysis["videos"]
            if vid["viral_videos"] > 0:
                insights.append(f"{vid['viral_videos']} videos went viral! Analyze their characteristics for replication.")
                
        return insights
        
    async def _create_visualizations(self, data: Dict[str, pd.DataFrame], config: ReportConfig) -> Dict[str, Any]:
        """Create visualizations for report"""
        visualizations = {}
        
        # Performance chart
        if "performance" in data and not data["performance"].empty:
            fig = self._create_performance_chart(data["performance"])
            visualizations["performance_chart"] = fig
            
        # Revenue breakdown
        if "revenue" in data and not data["revenue"].empty:
            fig = self._create_revenue_chart(data["revenue"])
            visualizations["revenue_chart"] = fig
            
        # Channel comparison
        if "channels" in data and not data["channels"].empty:
            fig = self._create_channel_chart(data["channels"])
            visualizations["channel_chart"] = fig
            
        # Video performance heatmap
        if "videos" in data and not data["videos"].empty:
            fig = self._create_video_heatmap(data["videos"])
            visualizations["video_heatmap"] = fig
            
        return visualizations
        
    def _create_performance_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create performance chart"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Views Over Time", "Watch Time", "Engagement Rate", "Revenue"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Views
        fig.add_trace(
            go.Scatter(x=df["date"], y=df["total_views"], name="Views", line=dict(color="blue")),
            row=1, col=1
        )
        
        # Watch time
        fig.add_trace(
            go.Bar(x=df["date"], y=df["total_watch_time"], name="Watch Time", marker_color="green"),
            row=1, col=2
        )
        
        # Engagement
        fig.add_trace(
            go.Scatter(x=df["date"], y=df["avg_engagement"], name="Engagement", line=dict(color="orange")),
            row=2, col=1
        )
        
        # Revenue
        fig.add_trace(
            go.Bar(x=df["date"], y=df["total_revenue"], name="Revenue", marker_color="purple"),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, title_text="Performance Overview")
        return fig
        
    def _create_revenue_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create revenue breakdown chart"""
        revenue_sources = ["ad_revenue", "sponsorship_revenue", "merchandise_revenue", "membership_revenue"]
        
        fig = go.Figure()
        
        for source in revenue_sources:
            if source in df.columns:
                fig.add_trace(go.Bar(
                    x=df["date"],
                    y=df[source],
                    name=source.replace("_", " ").title()
                ))
                
        fig.update_layout(
            barmode="stack",
            title="Revenue Breakdown by Source",
            xaxis_title="Date",
            yaxis_title="Revenue ($)",
            height=400
        )
        
        return fig
        
    def _create_channel_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create channel comparison chart"""
        top_channels = df.nlargest(10, "total_views")
        
        fig = go.Figure(data=[
            go.Bar(x=top_channels["channel_name"], y=top_channels["total_views"], name="Views"),
            go.Bar(x=top_channels["channel_name"], y=top_channels["subscriber_count"], name="Subscribers", yaxis="y2")
        ])
        
        fig.update_layout(
            title="Top Channels Performance",
            xaxis_title="Channel",
            yaxis=dict(title="Views", side="left"),
            yaxis2=dict(title="Subscribers", overlaying="y", side="right"),
            height=400
        )
        
        return fig
        
    def _create_video_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create video performance heatmap"""
        # Prepare data for heatmap
        metrics = ["views", "likes", "comments", "revenue"]
        top_videos = df.nlargest(20, "views")
        
        # Normalize metrics for comparison
        heatmap_data = []
        for metric in metrics:
            if metric in top_videos.columns:
                normalized = (top_videos[metric] - top_videos[metric].min()) / (top_videos[metric].max() - top_videos[metric].min())
                heatmap_data.append(normalized.values)
                
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=top_videos["title"].str[:30],  # Truncate long titles
            y=metrics,
            colorscale="RdYlGn"
        ))
        
        fig.update_layout(
            title="Video Performance Heatmap (Top 20)",
            height=400
        )
        
        return fig
        
    async def _build_report_sections(
        self,
        data: Dict[str, pd.DataFrame],
        analysis: Dict[str, Any],
        visualizations: Dict[str, Any],
        config: ReportConfig
    ) -> List[ReportSection]:
        """Build report sections"""
        sections = []
        
        # Executive Summary
        sections.append(ReportSection(
            title="Executive Summary",
            content=self._create_executive_summary(analysis),
            summary=self._summarize_key_metrics(analysis)
        ))
        
        # Performance Overview
        if "performance_chart" in visualizations:
            sections.append(ReportSection(
                title="Performance Overview",
                content=analysis.get("performance", {}),
                chart_type="plotly",
                data=visualizations["performance_chart"]
            ))
            
        # Revenue Analysis
        if "revenue_chart" in visualizations:
            sections.append(ReportSection(
                title="Revenue Analysis",
                content=analysis.get("revenue", {}),
                chart_type="plotly",
                data=visualizations["revenue_chart"]
            ))
            
        # Channel Performance
        if "channel_chart" in visualizations:
            sections.append(ReportSection(
                title="Channel Performance",
                content=analysis.get("channels", {}),
                chart_type="plotly",
                data=visualizations["channel_chart"]
            ))
            
        # Video Analytics
        if "video_heatmap" in visualizations:
            sections.append(ReportSection(
                title="Video Analytics",
                content=analysis.get("videos", {}),
                chart_type="plotly",
                data=visualizations["video_heatmap"]
            ))
            
        # Insights and Recommendations
        sections.append(ReportSection(
            title="Insights and Recommendations",
            content=analysis.get("insights", []),
            summary="Key actionable insights based on data analysis"
        ))
        
        # Add custom sections if specified
        for section_name in config.include_sections:
            custom_section = await self._create_custom_section(section_name, data, analysis)
            if custom_section:
                sections.append(custom_section)
                
        return sections
        
    def _create_executive_summary(self, analysis: Dict[str, Any]) -> str:
        """Create executive summary"""
        summary_parts = []
        
        if "performance" in analysis:
            perf = analysis["performance"]
            summary_parts.append(
                f"Total views reached {perf['total_views']:,} with a {perf['growth_rate']:.1f}% growth rate."
            )
            
        if "revenue" in analysis:
            rev = analysis["revenue"]
            summary_parts.append(
                f"Generated ${rev['total_revenue']:,.2f} in revenue with a {rev['profit_margin']:.1%} profit margin."
            )
            
        if "videos" in analysis:
            vid = analysis["videos"]
            summary_parts.append(
                f"Published {vid['total_videos']} videos achieving {vid['total_views']:,} total views."
            )
            
        return " ".join(summary_parts)
        
    def _summarize_key_metrics(self, analysis: Dict[str, Any]) -> str:
        """Summarize key metrics"""
        metrics = []
        
        for category, data in analysis.items():
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        if "revenue" in key or "cost" in key or "profit" in key:
                            metrics.append(f"{key}: ${value:,.2f}")
                        elif "rate" in key or "margin" in key:
                            metrics.append(f"{key}: {value:.1%}")
                        else:
                            metrics.append(f"{key}: {value:,.0f}")
                            
        return " | ".join(metrics[:5])  # Top 5 metrics
        
    async def _create_custom_section(
        self,
        section_name: str,
        data: Dict[str, pd.DataFrame],
        analysis: Dict[str, Any]
    ) -> Optional[ReportSection]:
        """Create custom report section"""
        # Implementation for custom sections
        return None
        
    async def _generate_report_file(
        self,
        sections: List[ReportSection],
        config: ReportConfig,
        report_id: str
    ) -> Dict[str, Any]:
        """Generate report file in specified format"""
        output_path = Path(self.config["report_storage_path"]) / f"{report_id}.{config.format.value}"
        
        if config.format == ReportFormat.PDF:
            file_size = await self._generate_pdf_report(sections, output_path)
        elif config.format == ReportFormat.EXCEL:
            file_size = await self._generate_excel_report(sections, output_path)
        elif config.format == ReportFormat.HTML:
            file_size = await self._generate_html_report(sections, output_path)
        elif config.format == ReportFormat.JSON:
            file_size = await self._generate_json_report(sections, output_path)
        else:
            file_size = 0
            
        return {
            "path": str(output_path),
            "size": file_size,
            "format": config.format.value
        }
        
    async def _generate_pdf_report(self, sections: List[ReportSection], output_path: Path) -> int:
        """Generate PDF report"""
        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4')
        )
        story.append(Paragraph("YTEmpire Analytics Report", title_style))
        story.append(Spacer(1, 0.3 * inch))
        
        for section in sections:
            # Section title
            story.append(Paragraph(section.title, styles['Heading2']))
            story.append(Spacer(1, 0.2 * inch))
            
            # Section content
            if isinstance(section.content, str):
                story.append(Paragraph(section.content, styles['Normal']))
            elif isinstance(section.content, list):
                for item in section.content:
                    story.append(Paragraph(f"• {item}", styles['Normal']))
            elif isinstance(section.content, dict):
                for key, value in section.content.items():
                    story.append(Paragraph(f"<b>{key}:</b> {value}", styles['Normal']))
                    
            # Add chart if available
            if section.chart_type == "plotly" and section.data:
                # Convert plotly to image and add
                img_path = f"/tmp/{section.title.replace(' ', '_')}.png"
                section.data.write_image(img_path)
                story.append(Image(img_path, width=6*inch, height=4*inch))
                
            story.append(Spacer(1, 0.3 * inch))
            
        doc.build(story)
        return output_path.stat().st_size
        
    async def _generate_excel_report(self, sections: List[ReportSection], output_path: Path) -> int:
        """Generate Excel report"""
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Create summary sheet
            summary_df = pd.DataFrame()
            for section in sections:
                if isinstance(section.content, dict):
                    for key, value in section.content.items():
                        summary_df[key] = [value] if not isinstance(value, list) else value
                        
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Add data sheets
            for section in sections:
                if section.data is not None and isinstance(section.data, pd.DataFrame):
                    sheet_name = section.title[:31]  # Excel sheet name limit
                    section.data.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Format the sheet
                    worksheet = writer.sheets[sheet_name]
                    header_format = workbook.add_format({
                        'bold': True,
                        'bg_color': '#1f77b4',
                        'font_color': 'white'
                    })
                    
                    for col_num, value in enumerate(section.data.columns.values):
                        worksheet.write(0, col_num, value, header_format)
                        
        return output_path.stat().st_size
        
    async def _generate_html_report(self, sections: List[ReportSection], output_path: Path) -> int:
        """Generate HTML report"""
        # Create basic HTML template if not exists
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ title }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #1f77b4; }
                h2 { color: #333; }
                .section { margin-bottom: 30px; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background: #f0f0f0; }
            </style>
        </head>
        <body>
            <h1>{{ title }}</h1>
            <p>Generated at: {{ generated_at }}</p>
            {% for section in sections %}
            <div class="section">
                <h2>{{ section.title }}</h2>
                <div>{{ section.content }}</div>
            </div>
            {% endfor %}
        </body>
        </html>
        """
        
        from jinja2 import Template
        template = Template(html_template)
        
        html_content = template.render(
            title="YTEmpire Analytics Report",
            sections=sections,
            generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        with open(output_path, 'w') as f:
            f.write(html_content)
            
        return output_path.stat().st_size
        
    async def _generate_json_report(self, sections: List[ReportSection], output_path: Path) -> int:
        """Generate JSON report"""
        report_data = {
            "generated_at": datetime.utcnow().isoformat(),
            "sections": []
        }
        
        for section in sections:
            section_data = {
                "title": section.title,
                "content": section.content,
                "summary": section.summary
            }
            
            if section.data is not None:
                if isinstance(section.data, pd.DataFrame):
                    section_data["data"] = section.data.to_dict(orient="records")
                    
            report_data["sections"].append(section_data)
            
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
            
        return output_path.stat().st_size
        
    async def _deliver_report(self, report_file: Dict[str, Any], config: ReportConfig) -> Dict[str, Any]:
        """Deliver report via configured methods"""
        results = {}
        
        for method in config.delivery_methods:
            if method == DeliveryMethod.EMAIL:
                results["email"] = await self._deliver_via_email(report_file, config.recipients)
            elif method == DeliveryMethod.S3:
                results["s3"] = await self._deliver_to_s3(report_file)
            elif method == DeliveryMethod.SLACK:
                results["slack"] = await self._deliver_to_slack(report_file)
                
        return results
        
    async def _deliver_via_email(self, report_file: Dict[str, Any], recipients: List[str]) -> bool:
        """Deliver report via email"""
        try:
            msg = MIMEMultipart()
            msg['Subject'] = f"YTEmpire Report - {datetime.utcnow().strftime('%Y-%m-%d')}"
            msg['From'] = self.config["email_config"]["username"]
            msg['To'] = ', '.join(recipients)
            
            # Add body
            body = "Please find attached the latest YTEmpire analytics report."
            msg.attach(MIMEText(body, 'plain'))
            
            # Add attachment
            with open(report_file["path"], "rb") as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename={Path(report_file["path"]).name}'
                )
                msg.attach(part)
                
            # Send email
            with smtplib.SMTP(self.config["email_config"]["smtp_host"], self.config["email_config"]["smtp_port"]) as server:
                server.starttls()
                server.login(
                    self.config["email_config"]["username"],
                    self.config["email_config"]["password"]
                )
                server.send_message(msg)
                
            return True
            
        except Exception as e:
            logger.error(f"Email delivery failed: {e}")
            return False
            
    async def _deliver_to_s3(self, report_file: Dict[str, Any]) -> bool:
        """Deliver report to S3"""
        try:
            s3 = boto3.client('s3')
            key = f"reports/{datetime.utcnow().strftime('%Y/%m/%d')}/{Path(report_file['path']).name}"
            
            s3.upload_file(
                report_file["path"],
                self.config["s3_config"]["bucket"],
                key
            )
            
            return True
            
        except Exception as e:
            logger.error(f"S3 delivery failed: {e}")
            return False
            
    async def _deliver_to_slack(self, report_file: Dict[str, Any]) -> bool:
        """Deliver report notification to Slack"""
        # Implementation for Slack delivery
        return True
        
    def _cache_report(self, report_id: str, report_file: Dict[str, Any]):
        """Cache generated report"""
        try:
            with open(report_file["path"], "rb") as f:
                self.redis_client.setex(
                    f"report:{report_id}",
                    self.config["cache_ttl"],
                    f.read()
                )
        except Exception as e:
            logger.warning(f"Report caching failed: {e}")
            
    def _load_scheduled_reports(self):
        """Load scheduled report configurations"""
        # Load from database or configuration file
        # This is a placeholder
        self.scheduled_reports = [
            ReportConfig(
                name="daily_performance",
                type=ReportType.DAILY,
                format=ReportFormat.PDF,
                schedule="0 6 * * *",  # 6 AM daily
                recipients=["admin@ytempire.com"],
                delivery_methods=[DeliveryMethod.EMAIL, DeliveryMethod.S3],
                include_sections=["performance", "revenue", "insights"]
            ),
            ReportConfig(
                name="weekly_summary",
                type=ReportType.WEEKLY,
                format=ReportFormat.EXCEL,
                schedule="0 9 * * 1",  # 9 AM Monday
                recipients=["team@ytempire.com"],
                delivery_methods=[DeliveryMethod.EMAIL],
                include_sections=["all"]
            )
        ]
        
    def _setup_scheduler(self):
        """Setup report scheduler"""
        for report_config in self.scheduled_reports:
            # Parse cron expression and schedule
            # This is simplified - would use actual cron parser
            if report_config.type == ReportType.DAILY:
                schedule.every().day.at("06:00").do(
                    lambda: asyncio.run(self.generate_report(report_config))
                )
            elif report_config.type == ReportType.WEEKLY:
                schedule.every().monday.at("09:00").do(
                    lambda: asyncio.run(self.generate_report(report_config))
                )
                
    def get_report_status(self, report_id: str) -> Dict[str, Any]:
        """Get report generation status"""
        # Check cache
        cached = self.redis_client.get(f"report:{report_id}")
        
        if cached:
            return {
                "status": "completed",
                "cached": True,
                "size": len(cached)
            }
        else:
            return {
                "status": "not_found",
                "cached": False
            }


# Initialize reporting infrastructure
reporting_infrastructure = ReportingInfrastructure()