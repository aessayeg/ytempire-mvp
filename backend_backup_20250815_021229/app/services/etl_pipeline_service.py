"""
ETL Pipeline Service
Integrates Advanced ETL Pipeline with backend services
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import sys
import os

# Add ETL pipeline path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data'))

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text

from app.core.config import settings
from app.models.video import Video
from app.models.channel import Channel
# from app.models.analytics import VideoAnalytics  # Use the actual analytics model if available

# Import ETL Pipeline
try:
    from etl.advanced_etl_pipeline import (
        AdvancedETLPipeline,
        ETLConfig,
        ETLResult,
        ETLJobStatus
    )
    ETL_PIPELINE_AVAILABLE = True
except ImportError:
    ETL_PIPELINE_AVAILABLE = False
    logging.warning("Advanced ETL Pipeline not available")

logger = logging.getLogger(__name__)


class ETLPipelineService:
    """Service for managing ETL pipelines"""
    
    def __init__(self):
        self.pipeline = None
        self.is_initialized = False
        
        if ETL_PIPELINE_AVAILABLE:
            self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the ETL pipeline"""
        try:
            self.pipeline = AdvancedETLPipeline(
                database_url=settings.DATABASE_URL,
                redis_url=getattr(settings, 'REDIS_URL', None),
                storage_path=getattr(settings, 'ETL_STORAGE_PATH', 'data/etl')
            )
            logger.info("Advanced ETL Pipeline initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ETL Pipeline: {e}")
            self.pipeline = None
    
    async def initialize(self):
        """Initialize async components and create dimension tables"""
        if self.pipeline and not self.is_initialized:
            try:
                await self.pipeline.initialize()
                self.is_initialized = True
                logger.info("ETL Pipeline async components and dimension tables initialized")
            except Exception as e:
                logger.error(f"Failed to initialize ETL async components: {e}")
    
    async def run_video_performance_etl(
        self,
        db: AsyncSession,
        incremental: bool = True
    ) -> Dict[str, Any]:
        """
        Run ETL pipeline for video performance data
        """
        if not self.pipeline:
            return {
                "status": "error",
                "message": "ETL Pipeline not available"
            }
        
        await self.initialize()
        
        try:
            # Configure ETL for video performance
            config = ETLConfig(
                pipeline_name="video_performance_etl",
                source_config={
                    "type": "database",
                    "query": """
                        SELECT 
                            v.id as video_id,
                            v.title,
                            v.description,
                            v.duration,
                            v.published_at,
                            v.channel_id,
                            c.name as channel_name,
                            c.subscriber_count,
                            va.views,
                            va.likes,
                            va.comments,
                            va.engagement_rate,
                            va.updated_at as recorded_at
                        FROM videos v
                        JOIN channels c ON v.channel_id = c.id
                        LEFT JOIN video_analytics va ON v.id = va.video_id
                    """,
                    "incremental": incremental,
                    "watermark_column": "recorded_at"
                },
                target_config={
                    "database": "analytics"
                },
                dimensions={
                    "channel": {
                        "table": "dim_channel",
                        "columns": ["channel_id", "channel_name", "subscriber_count"],
                        "key_column": "channel_id",
                        "scd_type": 2  # Track history
                    },
                    "video": {
                        "table": "dim_video",
                        "columns": ["video_id", "title", "description", "duration"],
                        "key_column": "video_id",
                        "scd_type": 1  # Overwrite
                    }
                },
                fact_tables={
                    "video_performance": {
                        "table": "fact_video_performance",
                        "dimension_mappings": [
                            {
                                "table": "dim_channel",
                                "source_column": "channel_id",
                                "target_column": "channel_key",
                                "lookup_column": "channel_id"
                            },
                            {
                                "table": "dim_video",
                                "source_column": "video_id",
                                "target_column": "video_key",
                                "lookup_column": "video_id"
                            }
                        ],
                        "add_date_dimension": True,
                        "date_column": "published_at",
                        "add_time_dimension": True,
                        "time_column": "published_at"
                    }
                },
                transformations=[
                    {
                        "type": "clean",
                        "remove_duplicates": True,
                        "fill_missing": True,
                        "fill_strategy": "mean"
                    },
                    {
                        "type": "derive",
                        "columns": {
                            "title_length": "data['title'].str.len()",
                            "description_length": "data['description'].str.len()",
                            "engagement_score": "(data['likes'] + data['comments']) / data['views']"
                        }
                    },
                    {
                        "type": "filter",
                        "condition": "views > 0"
                    }
                ],
                quality_checks=[
                    {
                        "type": "completeness",
                        "columns": ["video_id", "channel_id", "views"],
                        "threshold": 0.95
                    },
                    {
                        "type": "uniqueness",
                        "columns": ["video_id"]
                    },
                    {
                        "type": "validity",
                        "column": "views",
                        "min_value": 0
                    },
                    {
                        "type": "validity",
                        "column": "engagement_rate",
                        "min_value": 0,
                        "max_value": 1
                    }
                ],
                batch_size=10000,
                incremental=incremental
            )
            
            # Run ETL pipeline
            result: ETLResult = await self.pipeline.run_pipeline(config)
            
            return {
                "status": result.status.value,
                "job_id": result.job_id,
                "records_processed": result.records_transformed,
                "dimensions_updated": result.dimensions_updated,
                "facts_inserted": result.facts_inserted,
                "quality_score": result.quality_score,
                "quality_issues": result.quality_issues,
                "duration": (result.end_time - result.start_time).total_seconds() if result.end_time else None
            }
            
        except Exception as e:
            logger.error(f"Failed to run video performance ETL: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def run_generation_metrics_etl(
        self,
        db: AsyncSession,
        incremental: bool = True
    ) -> Dict[str, Any]:
        """
        Run ETL pipeline for generation metrics
        """
        if not self.pipeline:
            return {
                "status": "error",
                "message": "ETL Pipeline not available"
            }
        
        await self.initialize()
        
        try:
            # Configure ETL for generation metrics
            config = ETLConfig(
                pipeline_name="generation_metrics_etl",
                source_config={
                    "type": "database",
                    "query": """
                        SELECT 
                            vg.id as generation_id,
                            vg.video_id,
                            vg.user_id,
                            vg.created_at,
                            vg.generation_time,
                            vg.script_generation_time,
                            vg.voice_synthesis_time,
                            vg.video_assembly_time,
                            vg.total_cost,
                            vg.ai_cost,
                            vg.voice_cost,
                            vg.quality_score,
                            vg.status
                        FROM video_generations vg
                    """,
                    "incremental": incremental,
                    "watermark_column": "created_at"
                },
                target_config={
                    "database": "analytics"
                },
                dimensions={
                    "user": {
                        "table": "dim_user",
                        "columns": ["user_id"],
                        "key_column": "user_id",
                        "scd_type": 1
                    }
                },
                fact_tables={
                    "generation_metrics": {
                        "table": "fact_generation_metrics",
                        "dimension_mappings": [
                            {
                                "table": "dim_video",
                                "source_column": "video_id",
                                "target_column": "video_key",
                                "lookup_column": "video_id"
                            },
                            {
                                "table": "dim_user",
                                "source_column": "user_id",
                                "target_column": "user_key",
                                "lookup_column": "user_id"
                            }
                        ],
                        "add_date_dimension": True,
                        "date_column": "created_at"
                    }
                },
                transformations=[
                    {
                        "type": "clean",
                        "remove_duplicates": True
                    },
                    {
                        "type": "derive",
                        "columns": {
                            "cost_per_second": "data['total_cost'] / data['generation_time']",
                            "success": "data['status'] == 'completed'"
                        }
                    }
                ],
                quality_checks=[
                    {
                        "type": "completeness",
                        "columns": ["generation_id", "video_id", "user_id"],
                        "threshold": 1.0
                    },
                    {
                        "type": "validity",
                        "column": "total_cost",
                        "min_value": 0,
                        "max_value": 100
                    },
                    {
                        "type": "validity",
                        "column": "quality_score",
                        "min_value": 0,
                        "max_value": 1
                    }
                ],
                batch_size=5000,
                incremental=incremental
            )
            
            # Run ETL pipeline
            result: ETLResult = await self.pipeline.run_pipeline(config)
            
            return {
                "status": result.status.value,
                "job_id": result.job_id,
                "records_processed": result.records_transformed,
                "dimensions_updated": result.dimensions_updated,
                "facts_inserted": result.facts_inserted,
                "quality_score": result.quality_score
            }
            
        except Exception as e:
            logger.error(f"Failed to run generation metrics ETL: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def run_channel_analytics_etl(
        self,
        db: AsyncSession,
        channel_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run ETL pipeline for channel analytics
        """
        if not self.pipeline:
            return {
                "status": "error",
                "message": "ETL Pipeline not available"
            }
        
        await self.initialize()
        
        try:
            # Build query based on channel_id
            query = """
                SELECT 
                    c.id as channel_id,
                    c.name as channel_name,
                    c.description,
                    c.subscriber_count,
                    c.video_count,
                    c.created_at,
                    COUNT(v.id) as total_videos,
                    SUM(va.views) as total_views,
                    AVG(va.engagement_rate) as avg_engagement
                FROM channels c
                LEFT JOIN videos v ON c.id = v.channel_id
                LEFT JOIN video_analytics va ON v.id = va.video_id
            """
            
            if channel_id:
                query += f" WHERE c.id = '{channel_id}'"
            
            query += " GROUP BY c.id, c.name, c.description, c.subscriber_count, c.video_count, c.created_at"
            
            config = ETLConfig(
                pipeline_name="channel_analytics_etl",
                source_config={
                    "type": "database",
                    "query": query
                },
                target_config={
                    "database": "analytics"
                },
                dimensions={
                    "channel": {
                        "table": "dim_channel",
                        "columns": [
                            "channel_id", "channel_name", "description",
                            "subscriber_count", "video_count"
                        ],
                        "key_column": "channel_id",
                        "scd_type": 2
                    }
                },
                transformations=[
                    {
                        "type": "derive",
                        "columns": {
                            "avg_views_per_video": "data['total_views'] / data['total_videos']",
                            "channel_age_days": "(pd.Timestamp.now() - pd.to_datetime(data['created_at'])).dt.days"
                        }
                    }
                ],
                quality_checks=[
                    {
                        "type": "completeness",
                        "columns": ["channel_id", "channel_name"],
                        "threshold": 1.0
                    }
                ]
            )
            
            result = await self.pipeline.run_pipeline(config)
            
            return {
                "status": result.status.value,
                "job_id": result.job_id,
                "records_processed": result.records_transformed,
                "dimensions_updated": result.dimensions_updated
            }
            
        except Exception as e:
            logger.error(f"Failed to run channel analytics ETL: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def schedule_etl_pipeline(
        self,
        pipeline_name: str,
        cron_expression: str
    ) -> Dict[str, Any]:
        """
        Schedule periodic ETL pipeline execution
        """
        if not self.pipeline:
            return {
                "status": "error",
                "message": "ETL Pipeline not available"
            }
        
        await self.initialize()
        
        try:
            # Get configuration based on pipeline name
            if pipeline_name == "video_performance_etl":
                config = self._get_video_performance_config()
            elif pipeline_name == "generation_metrics_etl":
                config = self._get_generation_metrics_config()
            elif pipeline_name == "channel_analytics_etl":
                config = self._get_channel_analytics_config()
            else:
                return {
                    "status": "error",
                    "message": f"Unknown pipeline: {pipeline_name}"
                }
            
            # Schedule pipeline
            await self.pipeline.schedule_pipeline(config, cron_expression)
            
            return {
                "status": "scheduled",
                "pipeline_name": pipeline_name,
                "schedule": cron_expression
            }
            
        except Exception as e:
            logger.error(f"Failed to schedule ETL pipeline: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get ETL job status
        """
        if not self.pipeline:
            return None
        
        await self.initialize()
        
        try:
            result = await self.pipeline.get_job_status(job_id)
            
            if result:
                return {
                    "job_id": result.job_id,
                    "pipeline_name": result.pipeline_name,
                    "status": result.status.value,
                    "start_time": result.start_time.isoformat() if result.start_time else None,
                    "end_time": result.end_time.isoformat() if result.end_time else None,
                    "records_processed": result.records_transformed,
                    "quality_score": result.quality_score,
                    "errors": result.errors
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            return None
    
    async def get_data_quality_report(
        self,
        dimension: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get data quality report for dimensions
        """
        if not self.pipeline:
            return {}
        
        await self.initialize()
        
        try:
            # Query dimension tables for quality metrics
            async with AsyncSession(self.pipeline.db_engine) as session:
                # Get completeness for each dimension
                dimensions_quality = {}
                
                dimensions_to_check = ["channel", "video", "user"] if not dimension else [dimension]
                
                for dim in dimensions_to_check:
                    table_name = f"dim_{dim}"
                    
                    # Count total records
                    total_query = f"SELECT COUNT(*) FROM {table_name}"
                    total_result = await session.execute(text(total_query))
                    total_count = total_result.scalar()
                    
                    # Count records with nulls in key columns
                    null_query = f"SELECT COUNT(*) FROM {table_name} WHERE {dim}_id IS NULL"
                    null_result = await session.execute(text(null_query))
                    null_count = null_result.scalar()
                    
                    completeness = (total_count - null_count) / total_count if total_count > 0 else 0
                    
                    dimensions_quality[dim] = {
                        "total_records": total_count,
                        "null_records": null_count,
                        "completeness": completeness
                    }
                
                return {
                    "status": "success",
                    "dimensions": dimensions_quality,
                    "overall_quality": sum(d["completeness"] for d in dimensions_quality.values()) / len(dimensions_quality) if dimensions_quality else 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get data quality report: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _get_video_performance_config(self) -> ETLConfig:
        """Get configuration for video performance ETL"""
        return ETLConfig(
            pipeline_name="video_performance_etl",
            source_config={
                "type": "database",
                "query": "SELECT * FROM video_performance_view",
                "incremental": True,
                "watermark_column": "updated_at"
            },
            target_config={"database": "analytics"},
            dimensions={
                "channel": {
                    "table": "dim_channel",
                    "columns": ["channel_id", "channel_name"],
                    "key_column": "channel_id",
                    "scd_type": 2
                },
                "video": {
                    "table": "dim_video",
                    "columns": ["video_id", "title"],
                    "key_column": "video_id",
                    "scd_type": 1
                }
            },
            fact_tables={
                "video_performance": {
                    "table": "fact_video_performance"
                }
            },
            incremental=True
        )
    
    def _get_generation_metrics_config(self) -> ETLConfig:
        """Get configuration for generation metrics ETL"""
        return ETLConfig(
            pipeline_name="generation_metrics_etl",
            source_config={
                "type": "database",
                "query": "SELECT * FROM video_generations",
                "incremental": True,
                "watermark_column": "created_at"
            },
            target_config={"database": "analytics"},
            fact_tables={
                "generation_metrics": {
                    "table": "fact_generation_metrics"
                }
            },
            incremental=True
        )
    
    def _get_channel_analytics_config(self) -> ETLConfig:
        """Get configuration for channel analytics ETL"""
        return ETLConfig(
            pipeline_name="channel_analytics_etl",
            source_config={
                "type": "database",
                "query": "SELECT * FROM channel_analytics_view"
            },
            target_config={"database": "analytics"},
            dimensions={
                "channel": {
                    "table": "dim_channel",
                    "columns": ["channel_id", "channel_name", "subscriber_count"],
                    "key_column": "channel_id",
                    "scd_type": 2
                }
            }
        )


# Singleton instance
etl_service = ETLPipelineService()