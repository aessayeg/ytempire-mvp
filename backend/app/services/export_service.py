"""
Data Export Service
Comprehensive data export functionality with multiple formats and streaming support
"""
import asyncio
import csv
import json
import logging
import io
import zipfile
from typing import Dict, List, Optional, Any, AsyncIterator, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
import pandas as pd
import xlsxwriter
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, text
import aiofiles
import redis.asyncio as redis

from app.core.config import settings
from app.core.cache import cache_service
from app.models.video import Video
from app.models.channel import Channel
from app.models.cost import Cost
from app.models.user import User

logger = logging.getLogger(__name__)


class ExportFormat(str, Enum):
    """Supported export formats"""
    CSV = "csv"
    JSON = "json"
    EXCEL = "excel"
    PARQUET = "parquet"
    TSV = "tsv"
    XML = "xml"
    HTML = "html"
    MARKDOWN = "markdown"


class ExportStatus(str, Enum):
    """Export job status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class DataType(str, Enum):
    """Types of data that can be exported"""
    ANALYTICS = "analytics"
    VIDEOS = "videos"
    CHANNELS = "channels"
    COSTS = "costs"
    PERFORMANCE = "performance"
    AB_TESTS = "ab_tests"
    USER_BEHAVIOR = "user_behavior"
    REVENUE = "revenue"
    CUSTOM_QUERY = "custom_query"


@dataclass
class ExportJob:
    """Export job metadata"""
    job_id: str
    user_id: str
    data_type: DataType
    format: ExportFormat
    status: ExportStatus
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    file_path: Optional[str]
    file_size: Optional[int]
    row_count: Optional[int]
    parameters: Dict[str, Any]
    error_message: Optional[str]
    download_url: Optional[str]
    expires_at: Optional[datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExportSchedule:
    """Scheduled export configuration"""
    schedule_id: str
    user_id: str
    name: str
    data_type: DataType
    format: ExportFormat
    frequency: str  # cron expression
    parameters: Dict[str, Any]
    recipients: List[str]  # email addresses
    is_active: bool
    last_run: Optional[datetime]
    next_run: datetime
    created_at: datetime


class DataExportService:
    """Service for exporting data in various formats"""
    
    def __init__(self):
        self.export_base = Path(settings.DATA_STORAGE_PATH) / "exports"
        self.export_base.mkdir(parents=True, exist_ok=True)
        self.redis_client: Optional[redis.Redis] = None
        self.max_rows_in_memory = 100000  # Stream larger datasets
        self.export_ttl = 86400 * 7  # 7 days retention
        self.chunk_size = 10000
        
    async def initialize(self):
        """Initialize service connections"""
        self.redis_client = await redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
        logger.info("Export service initialized")
        
    async def create_export(
        self,
        db: AsyncSession,
        user_id: str,
        data_type: DataType,
        format: ExportFormat,
        parameters: Optional[Dict[str, Any]] = None
    ) -> ExportJob:
        """Create a new export job"""
        job_id = self._generate_job_id(user_id, data_type)
        
        job = ExportJob(
            job_id=job_id,
            user_id=user_id,
            data_type=data_type,
            format=format,
            status=ExportStatus.PENDING,
            created_at=datetime.utcnow(),
            started_at=None,
            completed_at=None,
            file_path=None,
            file_size=None,
            row_count=None,
            parameters=parameters or {},
            error_message=None,
            download_url=None,
            expires_at=datetime.utcnow() + timedelta(days=7),
            metadata={}
        )
        
        # Store job metadata
        await self._store_job(job)
        
        # Queue job for processing
        await self._queue_job(job_id)
        
        logger.info(f"Created export job {job_id} for user {user_id}")
        return job
        
    async def process_export(self, job_id: str):
        """Process an export job"""
        job = await self.get_job(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return
            
        try:
            # Update job status
            job.status = ExportStatus.PROCESSING
            job.started_at = datetime.utcnow()
            await self._store_job(job)
            
            # Get database session
            async with AsyncSession() as db:
                # Fetch data based on type
                data = await self._fetch_data(db, job.data_type, job.parameters)
                
                # Export data to file
                file_path, file_size, row_count = await self._export_data(
                    data, 
                    job.format, 
                    job.job_id
                )
                
                # Update job with results
                job.status = ExportStatus.COMPLETED
                job.completed_at = datetime.utcnow()
                job.file_path = str(file_path)
                job.file_size = file_size
                job.row_count = row_count
                job.download_url = self._generate_download_url(job.job_id)
                
                await self._store_job(job)
                
                logger.info(f"Completed export job {job_id}: {row_count} rows, {file_size} bytes")
                
        except Exception as e:
            logger.error(f"Failed to process export job {job_id}: {e}")
            job.status = ExportStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            await self._store_job(job)
            
    async def stream_export(
        self,
        db: AsyncSession,
        data_type: DataType,
        format: ExportFormat,
        parameters: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[bytes]:
        """Stream export data without creating a file"""
        # Create in-memory buffer
        buffer = io.BytesIO()
        
        # Fetch data in chunks
        async for chunk in self._fetch_data_stream(db, data_type, parameters):
            # Convert chunk to requested format
            chunk_data = await self._format_chunk(chunk, format)
            yield chunk_data
            
    async def _fetch_data(
        self,
        db: AsyncSession,
        data_type: DataType,
        parameters: Dict[str, Any]
    ) -> pd.DataFrame:
        """Fetch data based on type"""
        start_date = parameters.get("start_date")
        end_date = parameters.get("end_date")
        channel_id = parameters.get("channel_id")
        user_id = parameters.get("user_id")
        
        if data_type == DataType.ANALYTICS:
            return await self._fetch_analytics_data(db, start_date, end_date, channel_id)
        elif data_type == DataType.VIDEOS:
            return await self._fetch_videos_data(db, start_date, end_date, channel_id)
        elif data_type == DataType.CHANNELS:
            return await self._fetch_channels_data(db, user_id)
        elif data_type == DataType.COSTS:
            return await self._fetch_costs_data(db, start_date, end_date)
        elif data_type == DataType.PERFORMANCE:
            return await self._fetch_performance_data(db, start_date, end_date, channel_id)
        elif data_type == DataType.REVENUE:
            return await self._fetch_revenue_data(db, start_date, end_date)
        elif data_type == DataType.CUSTOM_QUERY:
            query = parameters.get("query")
            if query:
                return await self._execute_custom_query(db, query)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
            
    async def _fetch_analytics_data(
        self,
        db: AsyncSession,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        channel_id: Optional[str]
    ) -> pd.DataFrame:
        """Fetch analytics data"""
        query = select(Video).where(Video.status == "published")
        
        if start_date:
            query = query.where(Video.created_at >= start_date)
        if end_date:
            query = query.where(Video.created_at <= end_date)
        if channel_id:
            query = query.where(Video.channel_id == channel_id)
            
        result = await db.execute(query)
        videos = result.scalars().all()
        
        # Convert to DataFrame
        data = []
        for video in videos:
            data.append({
                "video_id": video.id,
                "title": video.title,
                "channel_id": video.channel_id,
                "views": video.view_count,
                "likes": video.like_count,
                "comments": video.comment_count,
                "engagement_rate": video.engagement_rate,
                "watch_time": video.watch_time_minutes,
                "revenue": video.estimated_revenue,
                "created_at": video.created_at,
                "published_at": video.published_at
            })
            
        return pd.DataFrame(data)
        
    async def _fetch_videos_data(
        self,
        db: AsyncSession,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        channel_id: Optional[str]
    ) -> pd.DataFrame:
        """Fetch videos data"""
        query = select(Video)
        
        if start_date:
            query = query.where(Video.created_at >= start_date)
        if end_date:
            query = query.where(Video.created_at <= end_date)
        if channel_id:
            query = query.where(Video.channel_id == channel_id)
            
        result = await db.execute(query)
        videos = result.scalars().all()
        
        # Convert to DataFrame with all fields
        data = []
        for video in videos:
            data.append({
                "id": video.id,
                "title": video.title,
                "description": video.description,
                "channel_id": video.channel_id,
                "youtube_video_id": video.youtube_video_id,
                "status": video.status,
                "script": video.script,
                "voice_url": video.voice_url,
                "thumbnail_url": video.thumbnail_url,
                "video_url": video.video_url,
                "tags": json.dumps(video.tags) if video.tags else None,
                "quality_score": video.quality_score,
                "view_count": video.view_count,
                "like_count": video.like_count,
                "comment_count": video.comment_count,
                "engagement_rate": video.engagement_rate,
                "watch_time_minutes": video.watch_time_minutes,
                "estimated_revenue": video.estimated_revenue,
                "created_at": video.created_at,
                "published_at": video.published_at,
                "updated_at": video.updated_at
            })
            
        return pd.DataFrame(data)
        
    async def _fetch_channels_data(
        self,
        db: AsyncSession,
        user_id: Optional[str]
    ) -> pd.DataFrame:
        """Fetch channels data"""
        query = select(Channel)
        
        if user_id:
            query = query.where(Channel.user_id == user_id)
            
        result = await db.execute(query)
        channels = result.scalars().all()
        
        data = []
        for channel in channels:
            data.append({
                "id": channel.id,
                "name": channel.name,
                "youtube_channel_id": channel.youtube_channel_id,
                "user_id": channel.user_id,
                "description": channel.description,
                "subscriber_count": channel.subscriber_count,
                "video_count": channel.video_count,
                "total_views": channel.total_views,
                "status": channel.status,
                "created_at": channel.created_at,
                "updated_at": channel.updated_at
            })
            
        return pd.DataFrame(data)
        
    async def _fetch_costs_data(
        self,
        db: AsyncSession,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> pd.DataFrame:
        """Fetch costs data"""
        query = select(Cost)
        
        if start_date:
            query = query.where(Cost.created_at >= start_date)
        if end_date:
            query = query.where(Cost.created_at <= end_date)
            
        result = await db.execute(query)
        costs = result.scalars().all()
        
        data = []
        for cost in costs:
            data.append({
                "id": cost.id,
                "service": cost.service,
                "amount": cost.amount,
                "video_id": cost.video_id,
                "user_id": cost.user_id,
                "description": cost.description,
                "created_at": cost.created_at
            })
            
        return pd.DataFrame(data)
        
    async def _fetch_performance_data(
        self,
        db: AsyncSession,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        channel_id: Optional[str]
    ) -> pd.DataFrame:
        """Fetch performance metrics data"""
        # Aggregate performance metrics
        query = """
        SELECT 
            DATE(created_at) as date,
            channel_id,
            COUNT(*) as videos_count,
            SUM(view_count) as total_views,
            SUM(like_count) as total_likes,
            SUM(comment_count) as total_comments,
            AVG(engagement_rate) as avg_engagement_rate,
            SUM(estimated_revenue) as total_revenue,
            AVG(quality_score) as avg_quality_score
        FROM videos
        WHERE status = 'published'
        """
        
        conditions = []
        if start_date:
            conditions.append(f"created_at >= '{start_date}'")
        if end_date:
            conditions.append(f"created_at <= '{end_date}'")
        if channel_id:
            conditions.append(f"channel_id = '{channel_id}'")
            
        if conditions:
            query += " AND " + " AND ".join(conditions)
            
        query += " GROUP BY DATE(created_at), channel_id ORDER BY date DESC"
        
        result = await db.execute(text(query))
        rows = result.fetchall()
        
        data = []
        for row in rows:
            data.append({
                "date": row.date,
                "channel_id": row.channel_id,
                "videos_count": row.videos_count,
                "total_views": row.total_views,
                "total_likes": row.total_likes,
                "total_comments": row.total_comments,
                "avg_engagement_rate": row.avg_engagement_rate,
                "total_revenue": row.total_revenue,
                "avg_quality_score": row.avg_quality_score
            })
            
        return pd.DataFrame(data)
        
    async def _fetch_revenue_data(
        self,
        db: AsyncSession,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> pd.DataFrame:
        """Fetch revenue data"""
        query = """
        SELECT 
            DATE(v.created_at) as date,
            v.channel_id,
            c.name as channel_name,
            COUNT(v.id) as videos_count,
            SUM(v.estimated_revenue) as revenue,
            SUM(co.amount) as costs,
            SUM(v.estimated_revenue) - COALESCE(SUM(co.amount), 0) as profit,
            CASE 
                WHEN SUM(co.amount) > 0 
                THEN ((SUM(v.estimated_revenue) - SUM(co.amount)) / SUM(co.amount)) * 100
                ELSE 0 
            END as roi_percentage
        FROM videos v
        LEFT JOIN channels c ON v.channel_id = c.id
        LEFT JOIN costs co ON v.id = co.video_id
        WHERE v.status = 'published'
        """
        
        conditions = []
        if start_date:
            conditions.append(f"v.created_at >= '{start_date}'")
        if end_date:
            conditions.append(f"v.created_at <= '{end_date}'")
            
        if conditions:
            query += " AND " + " AND ".join(conditions)
            
        query += " GROUP BY DATE(v.created_at), v.channel_id, c.name ORDER BY date DESC"
        
        result = await db.execute(text(query))
        rows = result.fetchall()
        
        data = []
        for row in rows:
            data.append({
                "date": row.date,
                "channel_id": row.channel_id,
                "channel_name": row.channel_name,
                "videos_count": row.videos_count,
                "revenue": row.revenue,
                "costs": row.costs,
                "profit": row.profit,
                "roi_percentage": row.roi_percentage
            })
            
        return pd.DataFrame(data)
        
    async def _execute_custom_query(
        self,
        db: AsyncSession,
        query: str
    ) -> pd.DataFrame:
        """Execute custom SQL query (with safety checks)"""
        # Basic safety checks (in production, use proper SQL sanitization)
        forbidden_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE"]
        query_upper = query.upper()
        
        for keyword in forbidden_keywords:
            if keyword in query_upper:
                raise ValueError(f"Query contains forbidden keyword: {keyword}")
                
        # Execute query
        result = await db.execute(text(query))
        rows = result.fetchall()
        
        # Convert to DataFrame
        if rows:
            columns = result.keys()
            data = [dict(zip(columns, row)) for row in rows]
            return pd.DataFrame(data)
        else:
            return pd.DataFrame()
            
    async def _export_data(
        self,
        data: pd.DataFrame,
        format: ExportFormat,
        job_id: str
    ) -> Tuple[Path, int, int]:
        """Export data to file"""
        file_name = f"{job_id}.{format.value}"
        file_path = self.export_base / file_name
        
        if format == ExportFormat.CSV:
            data.to_csv(file_path, index=False)
        elif format == ExportFormat.JSON:
            data.to_json(file_path, orient='records', date_format='iso')
        elif format == ExportFormat.EXCEL:
            with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
                data.to_excel(writer, sheet_name='Export', index=False)
                
                # Add formatting
                workbook = writer.book
                worksheet = writer.sheets['Export']
                
                # Format header
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#4CAF50',
                    'font_color': 'white'
                })
                
                for col_num, value in enumerate(data.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                    
        elif format == ExportFormat.PARQUET:
            data.to_parquet(file_path, index=False)
        elif format == ExportFormat.TSV:
            data.to_csv(file_path, sep='\t', index=False)
        elif format == ExportFormat.XML:
            data.to_xml(file_path, index=False)
        elif format == ExportFormat.HTML:
            html = data.to_html(index=False, classes='table table-striped')
            file_path.write_text(html)
        elif format == ExportFormat.MARKDOWN:
            markdown = data.to_markdown(index=False)
            file_path.write_text(markdown)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        file_size = file_path.stat().st_size
        row_count = len(data)
        
        return file_path, file_size, row_count
        
    async def _fetch_data_stream(
        self,
        db: AsyncSession,
        data_type: DataType,
        parameters: Dict[str, Any]
    ) -> AsyncIterator[pd.DataFrame]:
        """Fetch data in chunks for streaming"""
        offset = 0
        
        while True:
            # Fetch chunk
            chunk = await self._fetch_data_chunk(
                db, 
                data_type, 
                parameters, 
                offset, 
                self.chunk_size
            )
            
            if chunk.empty:
                break
                
            yield chunk
            offset += self.chunk_size
            
    async def _fetch_data_chunk(
        self,
        db: AsyncSession,
        data_type: DataType,
        parameters: Dict[str, Any],
        offset: int,
        limit: int
    ) -> pd.DataFrame:
        """Fetch a chunk of data"""
        # Add pagination to the query
        query_params = parameters.copy()
        query_params['offset'] = offset
        query_params['limit'] = limit
        
        # Fetch data with pagination
        return await self._fetch_data(db, data_type, query_params)
        
    async def _format_chunk(
        self,
        chunk: pd.DataFrame,
        format: ExportFormat
    ) -> bytes:
        """Format a data chunk"""
        buffer = io.BytesIO()
        
        if format == ExportFormat.CSV:
            chunk.to_csv(buffer, index=False)
        elif format == ExportFormat.JSON:
            chunk.to_json(buffer, orient='records', date_format='iso')
        elif format == ExportFormat.TSV:
            chunk.to_csv(buffer, sep='\t', index=False)
        else:
            # For other formats, convert to CSV as fallback
            chunk.to_csv(buffer, index=False)
            
        buffer.seek(0)
        return buffer.read()
        
    async def create_scheduled_export(
        self,
        user_id: str,
        name: str,
        data_type: DataType,
        format: ExportFormat,
        frequency: str,  # cron expression
        parameters: Dict[str, Any],
        recipients: List[str]
    ) -> ExportSchedule:
        """Create a scheduled export"""
        schedule_id = self._generate_id("schedule")
        
        schedule = ExportSchedule(
            schedule_id=schedule_id,
            user_id=user_id,
            name=name,
            data_type=data_type,
            format=format,
            frequency=frequency,
            parameters=parameters,
            recipients=recipients,
            is_active=True,
            last_run=None,
            next_run=self._calculate_next_run(frequency),
            created_at=datetime.utcnow()
        )
        
        # Store schedule
        await self._store_schedule(schedule)
        
        logger.info(f"Created scheduled export {schedule_id}")
        return schedule
        
    async def get_job(self, job_id: str) -> Optional[ExportJob]:
        """Get export job by ID"""
        key = f"export_job:{job_id}"
        data = await self.redis_client.get(key)
        
        if data:
            job_dict = json.loads(data)
            # Convert datetime strings
            for field in ['created_at', 'started_at', 'completed_at', 'expires_at']:
                if job_dict.get(field):
                    job_dict[field] = datetime.fromisoformat(job_dict[field])
            return ExportJob(**job_dict)
            
        return None
        
    async def list_user_exports(
        self,
        user_id: str,
        status: Optional[ExportStatus] = None
    ) -> List[ExportJob]:
        """List user's export jobs"""
        pattern = f"export_job:*"
        jobs = []
        
        cursor = 0
        while True:
            cursor, keys = await self.redis_client.scan(
                cursor=cursor,
                match=pattern,
                count=100
            )
            
            for key in keys:
                job_data = await self.redis_client.get(key)
                if job_data:
                    job_dict = json.loads(job_data)
                    
                    # Filter by user
                    if job_dict.get('user_id') != user_id:
                        continue
                        
                    # Filter by status if specified
                    if status and job_dict.get('status') != status.value:
                        continue
                        
                    # Convert datetime strings
                    for field in ['created_at', 'started_at', 'completed_at', 'expires_at']:
                        if job_dict.get(field):
                            job_dict[field] = datetime.fromisoformat(job_dict[field])
                            
                    jobs.append(ExportJob(**job_dict))
                    
            if cursor == 0:
                break
                
        # Sort by created_at descending
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs
        
    async def cancel_export(self, job_id: str) -> bool:
        """Cancel an export job"""
        job = await self.get_job(job_id)
        if not job:
            return False
            
        if job.status in [ExportStatus.COMPLETED, ExportStatus.FAILED]:
            return False
            
        job.status = ExportStatus.CANCELLED
        job.completed_at = datetime.utcnow()
        await self._store_job(job)
        
        # Remove from queue
        await self.redis_client.lrem("export_queue", 0, job_id)
        
        logger.info(f"Cancelled export job {job_id}")
        return True
        
    async def cleanup_expired_exports(self):
        """Clean up expired export files"""
        pattern = f"export_job:*"
        
        cursor = 0
        while True:
            cursor, keys = await self.redis_client.scan(
                cursor=cursor,
                match=pattern,
                count=100
            )
            
            for key in keys:
                job_data = await self.redis_client.get(key)
                if job_data:
                    job_dict = json.loads(job_data)
                    expires_at = datetime.fromisoformat(job_dict.get('expires_at'))
                    
                    if expires_at < datetime.utcnow():
                        # Delete file
                        if job_dict.get('file_path'):
                            file_path = Path(job_dict['file_path'])
                            if file_path.exists():
                                file_path.unlink()
                                
                        # Update status
                        job_dict['status'] = ExportStatus.EXPIRED.value
                        await self.redis_client.setex(key, 86400, json.dumps(job_dict))
                        
                        logger.info(f"Cleaned up expired export {job_dict['job_id']}")
                        
            if cursor == 0:
                break
                
    async def _store_job(self, job: ExportJob):
        """Store job metadata"""
        key = f"export_job:{job.job_id}"
        await self.redis_client.setex(
            key,
            self.export_ttl,
            json.dumps(asdict(job), default=str)
        )
        
    async def _store_schedule(self, schedule: ExportSchedule):
        """Store schedule metadata"""
        key = f"export_schedule:{schedule.schedule_id}"
        await self.redis_client.set(
            key,
            json.dumps(asdict(schedule), default=str)
        )
        
    async def _queue_job(self, job_id: str):
        """Queue job for processing"""
        await self.redis_client.lpush("export_queue", job_id)
        
    def _generate_job_id(self, user_id: str, data_type: DataType) -> str:
        """Generate unique job ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        return f"{data_type.value}_{user_id[-8:]}_{timestamp}"
        
    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        return f"{prefix}_{timestamp}"
        
    def _generate_download_url(self, job_id: str) -> str:
        """Generate download URL for export"""
        return f"{settings.API_URL}/exports/download/{job_id}"
        
    def _calculate_next_run(self, cron_expression: str) -> datetime:
        """Calculate next run time from cron expression"""
        # Simplified - in production use croniter or similar
        # For now, assume daily at midnight
        tomorrow = datetime.utcnow() + timedelta(days=1)
        return tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)


# Singleton instance
export_service = DataExportService()