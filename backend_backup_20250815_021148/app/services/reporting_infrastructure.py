"""
Reporting Infrastructure for YTEmpire
Generates comprehensive reports and analytics
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import json
from jinja2 import Template
from io import BytesIO
import xlsxwriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

from app.models.video import Video
from app.models.channel import Channel
from app.models.analytics import Analytics
from app.db.session import AsyncSessionLocal


class ReportGenerator:
    """Generates various types of reports"""
    
    async def generate_performance_report(
        self,
        user_id: str,
        start_date: datetime,
        end_date: datetime,
        format: str = "pdf"
    ) -> bytes:
        """Generate comprehensive performance report"""
        
        # Gather data
        data = await self._gather_report_data(user_id, start_date, end_date)
        
        if format == "pdf":
            return self._generate_pdf_report(data)
        elif format == "excel":
            return self._generate_excel_report(data)
        elif format == "json":
            return json.dumps(data, default=str).encode()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    async def _gather_report_data(
        self,
        user_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Gather all data needed for report"""
        
        async with AsyncSessionLocal() as db:
            # Get user channels
            channels = await db.execute(
                "SELECT * FROM channels WHERE user_id = :user_id",
                {"user_id": user_id}
            )
            
            # Get videos in period
            videos = await db.execute(
                """
                SELECT v.* FROM videos v
                JOIN channels c ON v.channel_id = c.id
                WHERE c.user_id = :user_id
                AND v.created_at BETWEEN :start_date AND :end_date
                """,
                {"user_id": user_id, "start_date": start_date, "end_date": end_date}
            )
            
            # Aggregate metrics
            metrics = {
                "total_videos": len(videos),
                "total_views": sum(v.view_count for v in videos),
                "total_revenue": sum(v.estimated_revenue for v in videos),
                "total_cost": sum(v.total_cost for v in videos),
                "channels": channels,
                "videos": videos,
                "period": {
                    "start": start_date,
                    "end": end_date
                }
            }
            
            return metrics
    
    def _generate_pdf_report(self, data: Dict[str, Any]) -> bytes:
        """Generate PDF report"""
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        story.append(Paragraph("YTEmpire Performance Report", styles['Title']))
        story.append(Spacer(1, 12))
        
        # Period
        period_text = f"Period: {data['period']['start']} to {data['period']['end']}"
        story.append(Paragraph(period_text, styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Summary metrics
        summary_data = [
            ['Metric', 'Value'],
            ['Total Videos', str(data['total_videos'])],
            ['Total Views', f"{data['total_views']:,}"],
            ['Total Revenue', f"${data['total_revenue']:.2f}"],
            ['Total Cost', f"${data['total_cost']:.2f}"],
            ['Profit', f"${data['total_revenue'] - data['total_cost']:.2f}"],
            ['ROI', f"{((data['total_revenue'] - data['total_cost']) / data['total_cost'] * 100):.1f}%"]
        ]
        
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.read()
    
    def _generate_excel_report(self, data: Dict[str, Any]) -> bytes:
        """Generate Excel report"""
        buffer = BytesIO()
        
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # Summary sheet
            summary_df = pd.DataFrame({
                'Metric': ['Total Videos', 'Total Views', 'Total Revenue', 'Total Cost', 'Profit', 'ROI'],
                'Value': [
                    data['total_videos'],
                    data['total_views'],
                    f"${data['total_revenue']:.2f}",
                    f"${data['total_cost']:.2f}",
                    f"${data['total_revenue'] - data['total_cost']:.2f}",
                    f"{((data['total_revenue'] - data['total_cost']) / data['total_cost'] * 100):.1f}%"
                ]
            })
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Videos sheet
            if data['videos']:
                videos_df = pd.DataFrame(data['videos'])
                videos_df.to_excel(writer, sheet_name='Videos', index=False)
            
            # Channels sheet
            if data['channels']:
                channels_df = pd.DataFrame(data['channels'])
                channels_df.to_excel(writer, sheet_name='Channels', index=False)
        
        buffer.seek(0)
        return buffer.read()
    
    async def generate_scheduled_reports(self):
        """Generate and send scheduled reports"""
        # Daily reports
        # Weekly reports
        # Monthly reports
        pass


# Global report generator instance
report_generator = ReportGenerator()