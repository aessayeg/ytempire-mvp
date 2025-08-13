"""
Advanced ETL Pipeline for YTEmpire
Implements dimension tables, fact tables, and comprehensive data transformations
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Index
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge
import yaml

logger = logging.getLogger(__name__)

# Metrics
etl_jobs_started = Counter('etl_jobs_started', 'ETL jobs started', ['pipeline_name'])
etl_jobs_completed = Counter('etl_jobs_completed', 'ETL jobs completed', ['pipeline_name', 'status'])
etl_duration = Histogram('etl_duration_seconds', 'ETL duration', ['pipeline_name'])
etl_records_processed = Counter('etl_records_processed', 'Records processed', ['table_name'])
data_quality_score = Gauge('data_quality_score', 'Data quality score', ['dimension'])


class ETLJobStatus(Enum):
    """ETL Job Status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class ETLConfig:
    """ETL Pipeline Configuration"""
    pipeline_name: str
    source_config: Dict[str, Any]
    target_config: Dict[str, Any]
    
    # Dimension configurations
    dimensions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Fact table configuration
    fact_tables: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Transformation rules
    transformations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Data quality checks
    quality_checks: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance settings
    batch_size: int = 10000
    parallel_workers: int = 4
    
    # Scheduling
    schedule: Optional[str] = None  # Cron expression
    
    # Incremental loading
    incremental: bool = True
    watermark_column: Optional[str] = None
    
    # Error handling
    max_retries: int = 3
    retry_delay: int = 60  # seconds


@dataclass
class ETLResult:
    """ETL Pipeline Result"""
    job_id: str
    pipeline_name: str
    status: ETLJobStatus
    start_time: datetime
    end_time: Optional[datetime]
    
    # Record counts
    records_extracted: int = 0
    records_transformed: int = 0
    records_loaded: int = 0
    records_failed: int = 0
    
    # Dimension updates
    dimensions_updated: Dict[str, int] = field(default_factory=dict)
    
    # Fact table updates
    facts_inserted: Dict[str, int] = field(default_factory=dict)
    
    # Data quality
    quality_score: float = 0.0
    quality_issues: List[Dict[str, Any]] = field(default_factory=list)
    
    # Errors
    errors: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedETLPipeline:
    """
    Advanced ETL Pipeline with dimension tables and comprehensive transformations
    """
    
    def __init__(
        self,
        database_url: str,
        redis_url: Optional[str] = None,
        storage_path: str = "data/etl"
    ):
        self.database_url = database_url
        self.redis_url = redis_url
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize connections
        self.db_engine = None
        self.redis_client = None
        
        # Cache for dimension lookups
        self.dimension_cache = {}
        
        # Active jobs
        self.active_jobs = {}
    
    async def initialize(self):
        """Initialize async components"""
        self.db_engine = create_async_engine(self.database_url)
        
        if self.redis_url:
            self.redis_client = await redis.from_url(self.redis_url)
        
        # Create dimension tables if not exists
        await self._create_dimension_tables()
    
    async def _create_dimension_tables(self):
        """Create dimension tables in the database"""
        async with self.db_engine.begin() as conn:
            # Create dimension tables
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS dim_channel (
                    channel_key SERIAL PRIMARY KEY,
                    channel_id VARCHAR(255) UNIQUE NOT NULL,
                    channel_name VARCHAR(255),
                    channel_type VARCHAR(100),
                    channel_category VARCHAR(100),
                    subscriber_count INTEGER,
                    video_count INTEGER,
                    created_date DATE,
                    last_active_date DATE,
                    is_active BOOLEAN DEFAULT true,
                    effective_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expiry_date TIMESTAMP,
                    is_current BOOLEAN DEFAULT true
                )
            """))
            
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS dim_video (
                    video_key SERIAL PRIMARY KEY,
                    video_id VARCHAR(255) UNIQUE NOT NULL,
                    title VARCHAR(500),
                    description TEXT,
                    duration INTEGER,
                    category VARCHAR(100),
                    tags TEXT,
                    thumbnail_url TEXT,
                    published_date TIMESTAMP,
                    channel_key INTEGER REFERENCES dim_channel(channel_key),
                    effective_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_current BOOLEAN DEFAULT true
                )
            """))
            
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS dim_date (
                    date_key INTEGER PRIMARY KEY,
                    full_date DATE UNIQUE NOT NULL,
                    day_of_week INTEGER,
                    day_of_month INTEGER,
                    day_of_year INTEGER,
                    week_of_year INTEGER,
                    month INTEGER,
                    month_name VARCHAR(20),
                    quarter INTEGER,
                    year INTEGER,
                    is_weekend BOOLEAN,
                    is_holiday BOOLEAN,
                    holiday_name VARCHAR(100)
                )
            """))
            
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS dim_time (
                    time_key INTEGER PRIMARY KEY,
                    hour INTEGER,
                    minute INTEGER,
                    second INTEGER,
                    time_of_day VARCHAR(20),
                    am_pm VARCHAR(2)
                )
            """))
            
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS dim_user (
                    user_key SERIAL PRIMARY KEY,
                    user_id VARCHAR(255) UNIQUE NOT NULL,
                    username VARCHAR(255),
                    email VARCHAR(255),
                    user_type VARCHAR(50),
                    subscription_tier VARCHAR(50),
                    registration_date DATE,
                    last_login_date DATE,
                    is_active BOOLEAN DEFAULT true,
                    effective_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_current BOOLEAN DEFAULT true
                )
            """))
            
            # Create fact tables
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS fact_video_performance (
                    performance_key SERIAL PRIMARY KEY,
                    video_key INTEGER REFERENCES dim_video(video_key),
                    channel_key INTEGER REFERENCES dim_channel(channel_key),
                    date_key INTEGER REFERENCES dim_date(date_key),
                    time_key INTEGER REFERENCES dim_time(time_key),
                    views INTEGER,
                    likes INTEGER,
                    dislikes INTEGER,
                    comments INTEGER,
                    shares INTEGER,
                    watch_time_minutes FLOAT,
                    engagement_rate FLOAT,
                    click_through_rate FLOAT,
                    revenue_usd DECIMAL(10,2),
                    cost_usd DECIMAL(10,2),
                    profit_usd DECIMAL(10,2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS fact_generation_metrics (
                    generation_key SERIAL PRIMARY KEY,
                    video_key INTEGER REFERENCES dim_video(video_key),
                    user_key INTEGER REFERENCES dim_user(user_key),
                    date_key INTEGER REFERENCES dim_date(date_key),
                    generation_time_seconds FLOAT,
                    script_generation_time FLOAT,
                    voice_synthesis_time FLOAT,
                    video_assembly_time FLOAT,
                    total_cost DECIMAL(10,4),
                    ai_cost DECIMAL(10,4),
                    voice_cost DECIMAL(10,4),
                    storage_cost DECIMAL(10,4),
                    quality_score FLOAT,
                    success BOOLEAN,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Create indexes for performance
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fact_perf_date ON fact_video_performance(date_key)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fact_perf_channel ON fact_video_performance(channel_key)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fact_perf_video ON fact_video_performance(video_key)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fact_gen_date ON fact_generation_metrics(date_key)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fact_gen_user ON fact_generation_metrics(user_key)"))
    
    async def run_pipeline(self, config: ETLConfig) -> ETLResult:
        """
        Execute ETL pipeline
        """
        job_id = self._generate_job_id(config.pipeline_name)
        result = ETLResult(
            job_id=job_id,
            pipeline_name=config.pipeline_name,
            status=ETLJobStatus.RUNNING,
            start_time=datetime.now(),
            end_time=None
        )
        
        etl_jobs_started.labels(pipeline_name=config.pipeline_name).inc()
        self.active_jobs[job_id] = result
        
        try:
            # Step 1: Extract data
            logger.info(f"Extracting data for pipeline {config.pipeline_name}")
            extracted_data = await self._extract_data(config)
            result.records_extracted = len(extracted_data)
            
            # Step 2: Transform data
            logger.info(f"Transforming {result.records_extracted} records")
            transformed_data = await self._transform_data(extracted_data, config)
            result.records_transformed = len(transformed_data)
            
            # Step 3: Data quality checks
            logger.info("Running data quality checks")
            quality_result = await self._check_data_quality(transformed_data, config)
            result.quality_score = quality_result["score"]
            result.quality_issues = quality_result["issues"]
            
            # Step 4: Load dimensions
            logger.info("Loading dimension tables")
            dimension_results = await self._load_dimensions(transformed_data, config)
            result.dimensions_updated = dimension_results
            
            # Step 5: Load fact tables
            logger.info("Loading fact tables")
            fact_results = await self._load_facts(transformed_data, config)
            result.facts_inserted = fact_results
            
            # Step 6: Update watermarks for incremental loading
            if config.incremental and config.watermark_column:
                await self._update_watermark(config, transformed_data)
            
            result.status = ETLJobStatus.COMPLETED
            result.end_time = datetime.now()
            
            # Record metrics
            duration = (result.end_time - result.start_time).total_seconds()
            etl_duration.labels(pipeline_name=config.pipeline_name).observe(duration)
            etl_jobs_completed.labels(
                pipeline_name=config.pipeline_name,
                status="success"
            ).inc()
            
            logger.info(f"Pipeline {config.pipeline_name} completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline {config.pipeline_name} failed: {e}")
            result.status = ETLJobStatus.FAILED
            result.end_time = datetime.now()
            result.errors.append(str(e))
            
            etl_jobs_completed.labels(
                pipeline_name=config.pipeline_name,
                status="failed"
            ).inc()
            
            # Retry logic
            if len(result.errors) < config.max_retries:
                result.status = ETLJobStatus.RETRYING
                await asyncio.sleep(config.retry_delay)
                return await self.run_pipeline(config)
        
        finally:
            # Clean up
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
            
            # Cache result
            if self.redis_client:
                await self._cache_result(result)
        
        return result
    
    async def _extract_data(self, config: ETLConfig) -> pd.DataFrame:
        """Extract data from source"""
        source_type = config.source_config.get("type")
        
        if source_type == "database":
            return await self._extract_from_database(config.source_config)
        elif source_type == "api":
            return await self._extract_from_api(config.source_config)
        elif source_type == "file":
            return self._extract_from_file(config.source_config)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
    
    async def _extract_from_database(self, source_config: Dict) -> pd.DataFrame:
        """Extract data from database"""
        query = source_config.get("query")
        
        # Handle incremental loading
        if source_config.get("incremental"):
            watermark = await self._get_watermark(source_config.get("table"))
            if watermark:
                query += f" WHERE {source_config.get('watermark_column')} > '{watermark}'"
        
        async with AsyncSession(self.db_engine) as session:
            result = await session.execute(text(query))
            rows = result.fetchall()
            columns = result.keys()
        
        return pd.DataFrame(rows, columns=columns)
    
    async def _extract_from_api(self, source_config: Dict) -> pd.DataFrame:
        """Extract data from API"""
        # Implement API extraction logic
        # This would typically involve making HTTP requests
        # and parsing the response
        return pd.DataFrame()
    
    def _extract_from_file(self, source_config: Dict) -> pd.DataFrame:
        """Extract data from file"""
        file_path = source_config.get("path")
        file_type = source_config.get("format", "csv")
        
        if file_type == "csv":
            return pd.read_csv(file_path)
        elif file_type == "parquet":
            return pd.read_parquet(file_path)
        elif file_type == "json":
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_type}")
    
    async def _transform_data(
        self,
        data: pd.DataFrame,
        config: ETLConfig
    ) -> pd.DataFrame:
        """Apply transformations to data"""
        
        for transformation in config.transformations:
            transform_type = transformation.get("type")
            
            if transform_type == "filter":
                data = self._apply_filter(data, transformation)
            elif transform_type == "aggregate":
                data = self._apply_aggregation(data, transformation)
            elif transform_type == "join":
                data = await self._apply_join(data, transformation)
            elif transform_type == "derive":
                data = self._apply_derivation(data, transformation)
            elif transform_type == "clean":
                data = self._apply_cleaning(data, transformation)
            elif transform_type == "normalize":
                data = self._apply_normalization(data, transformation)
        
        return data
    
    def _apply_filter(self, data: pd.DataFrame, transformation: Dict) -> pd.DataFrame:
        """Apply filter transformation"""
        condition = transformation.get("condition")
        return data.query(condition) if condition else data
    
    def _apply_aggregation(self, data: pd.DataFrame, transformation: Dict) -> pd.DataFrame:
        """Apply aggregation transformation"""
        group_by = transformation.get("group_by", [])
        aggregations = transformation.get("aggregations", {})
        
        if group_by and aggregations:
            return data.groupby(group_by).agg(aggregations).reset_index()
        return data
    
    async def _apply_join(self, data: pd.DataFrame, transformation: Dict) -> pd.DataFrame:
        """Apply join transformation"""
        join_table = transformation.get("table")
        join_keys = transformation.get("keys")
        join_type = transformation.get("type", "inner")
        
        # Load join data
        query = f"SELECT * FROM {join_table}"
        async with AsyncSession(self.db_engine) as session:
            result = await session.execute(text(query))
            join_data = pd.DataFrame(result.fetchall(), columns=result.keys())
        
        return pd.merge(data, join_data, on=join_keys, how=join_type)
    
    def _apply_derivation(self, data: pd.DataFrame, transformation: Dict) -> pd.DataFrame:
        """Apply derivation transformation"""
        derivations = transformation.get("columns", {})
        
        for column_name, expression in derivations.items():
            data[column_name] = eval(expression, {"data": data, "pd": pd, "np": np})
        
        return data
    
    def _apply_cleaning(self, data: pd.DataFrame, transformation: Dict) -> pd.DataFrame:
        """Apply data cleaning transformation"""
        # Remove duplicates
        if transformation.get("remove_duplicates"):
            data = data.drop_duplicates()
        
        # Handle missing values
        if transformation.get("fill_missing"):
            fill_strategy = transformation.get("fill_strategy", "mean")
            if fill_strategy == "mean":
                data = data.fillna(data.mean())
            elif fill_strategy == "median":
                data = data.fillna(data.median())
            elif fill_strategy == "mode":
                data = data.fillna(data.mode().iloc[0])
            elif fill_strategy == "forward":
                data = data.fillna(method="ffill")
            elif fill_strategy == "backward":
                data = data.fillna(method="bfill")
        
        # Remove outliers
        if transformation.get("remove_outliers"):
            z_threshold = transformation.get("z_threshold", 3)
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            z_scores = np.abs((data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std())
            data = data[(z_scores < z_threshold).all(axis=1)]
        
        return data
    
    def _apply_normalization(self, data: pd.DataFrame, transformation: Dict) -> pd.DataFrame:
        """Apply normalization transformation"""
        columns = transformation.get("columns", [])
        method = transformation.get("method", "standard")
        
        for column in columns:
            if column in data.columns:
                if method == "standard":
                    data[column] = (data[column] - data[column].mean()) / data[column].std()
                elif method == "minmax":
                    min_val = data[column].min()
                    max_val = data[column].max()
                    data[column] = (data[column] - min_val) / (max_val - min_val)
        
        return data
    
    async def _check_data_quality(
        self,
        data: pd.DataFrame,
        config: ETLConfig
    ) -> Dict[str, Any]:
        """Check data quality"""
        issues = []
        total_checks = len(config.quality_checks)
        passed_checks = 0
        
        for check in config.quality_checks:
            check_type = check.get("type")
            
            if check_type == "completeness":
                result = self._check_completeness(data, check)
            elif check_type == "uniqueness":
                result = self._check_uniqueness(data, check)
            elif check_type == "validity":
                result = self._check_validity(data, check)
            elif check_type == "consistency":
                result = self._check_consistency(data, check)
            else:
                result = {"passed": False, "message": f"Unknown check type: {check_type}"}
            
            if result["passed"]:
                passed_checks += 1
            else:
                issues.append(result)
        
        score = (passed_checks / total_checks) if total_checks > 0 else 1.0
        data_quality_score.labels(dimension=config.pipeline_name).set(score)
        
        return {
            "score": score,
            "issues": issues,
            "total_checks": total_checks,
            "passed_checks": passed_checks
        }
    
    def _check_completeness(self, data: pd.DataFrame, check: Dict) -> Dict[str, Any]:
        """Check data completeness"""
        columns = check.get("columns", [])
        threshold = check.get("threshold", 0.95)
        
        for column in columns:
            if column in data.columns:
                completeness = 1 - (data[column].isna().sum() / len(data))
                if completeness < threshold:
                    return {
                        "passed": False,
                        "type": "completeness",
                        "column": column,
                        "message": f"Column {column} has completeness {completeness:.2%}, below threshold {threshold:.2%}"
                    }
        
        return {"passed": True}
    
    def _check_uniqueness(self, data: pd.DataFrame, check: Dict) -> Dict[str, Any]:
        """Check data uniqueness"""
        columns = check.get("columns", [])
        
        for column in columns:
            if column in data.columns:
                duplicates = data[column].duplicated().sum()
                if duplicates > 0:
                    return {
                        "passed": False,
                        "type": "uniqueness",
                        "column": column,
                        "message": f"Column {column} has {duplicates} duplicate values"
                    }
        
        return {"passed": True}
    
    def _check_validity(self, data: pd.DataFrame, check: Dict) -> Dict[str, Any]:
        """Check data validity"""
        column = check.get("column")
        valid_values = check.get("valid_values")
        min_value = check.get("min_value")
        max_value = check.get("max_value")
        pattern = check.get("pattern")
        
        if column not in data.columns:
            return {"passed": True}
        
        if valid_values:
            invalid = ~data[column].isin(valid_values)
            if invalid.any():
                return {
                    "passed": False,
                    "type": "validity",
                    "column": column,
                    "message": f"Column {column} has {invalid.sum()} invalid values"
                }
        
        if min_value is not None:
            below_min = data[column] < min_value
            if below_min.any():
                return {
                    "passed": False,
                    "type": "validity",
                    "column": column,
                    "message": f"Column {column} has {below_min.sum()} values below minimum {min_value}"
                }
        
        if max_value is not None:
            above_max = data[column] > max_value
            if above_max.any():
                return {
                    "passed": False,
                    "type": "validity",
                    "column": column,
                    "message": f"Column {column} has {above_max.sum()} values above maximum {max_value}"
                }
        
        return {"passed": True}
    
    def _check_consistency(self, data: pd.DataFrame, check: Dict) -> Dict[str, Any]:
        """Check data consistency"""
        # Implement consistency checks between columns
        # For example, check if start_date < end_date
        return {"passed": True}
    
    async def _load_dimensions(
        self,
        data: pd.DataFrame,
        config: ETLConfig
    ) -> Dict[str, int]:
        """Load dimension tables"""
        results = {}
        
        for dim_name, dim_config in config.dimensions.items():
            logger.info(f"Loading dimension: {dim_name}")
            
            # Extract dimension data
            dim_columns = dim_config.get("columns", [])
            dim_data = data[dim_columns].drop_duplicates() if dim_columns else data
            
            # Load to dimension table
            table_name = dim_config.get("table", f"dim_{dim_name}")
            records_loaded = await self._upsert_dimension(
                dim_data,
                table_name,
                dim_config
            )
            
            results[dim_name] = records_loaded
            etl_records_processed.labels(table_name=table_name).inc(records_loaded)
        
        return results
    
    async def _upsert_dimension(
        self,
        data: pd.DataFrame,
        table_name: str,
        config: Dict
    ) -> int:
        """Upsert records to dimension table"""
        key_column = config.get("key_column")
        scd_type = config.get("scd_type", 1)  # Slowly Changing Dimension type
        
        records_loaded = 0
        
        async with AsyncSession(self.db_engine) as session:
            for _, row in data.iterrows():
                row_dict = row.to_dict()
                
                if scd_type == 1:
                    # Type 1: Overwrite
                    await self._upsert_type1(session, table_name, key_column, row_dict)
                elif scd_type == 2:
                    # Type 2: Historical tracking
                    await self._upsert_type2(session, table_name, key_column, row_dict)
                
                records_loaded += 1
            
            await session.commit()
        
        return records_loaded
    
    async def _upsert_type1(
        self,
        session: AsyncSession,
        table_name: str,
        key_column: str,
        row_dict: Dict
    ):
        """Type 1 SCD: Overwrite existing records"""
        # Check if record exists
        check_query = f"SELECT 1 FROM {table_name} WHERE {key_column} = :key_value"
        result = await session.execute(
            text(check_query),
            {"key_value": row_dict[key_column]}
        )
        exists = result.scalar()
        
        if exists:
            # Update existing record
            set_clause = ", ".join([f"{k} = :{k}" for k in row_dict.keys() if k != key_column])
            update_query = f"UPDATE {table_name} SET {set_clause} WHERE {key_column} = :key_value"
            row_dict["key_value"] = row_dict[key_column]
            await session.execute(text(update_query), row_dict)
        else:
            # Insert new record
            columns = ", ".join(row_dict.keys())
            values = ", ".join([f":{k}" for k in row_dict.keys()])
            insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({values})"
            await session.execute(text(insert_query), row_dict)
    
    async def _upsert_type2(
        self,
        session: AsyncSession,
        table_name: str,
        key_column: str,
        row_dict: Dict
    ):
        """Type 2 SCD: Historical tracking with effective dates"""
        # Check if record exists and is different
        check_query = f"""
            SELECT * FROM {table_name} 
            WHERE {key_column} = :key_value AND is_current = true
        """
        result = await session.execute(
            text(check_query),
            {"key_value": row_dict[key_column]}
        )
        existing = result.fetchone()
        
        if existing:
            # Check if data has changed
            has_changed = any(existing[k] != row_dict.get(k) for k in row_dict.keys())
            
            if has_changed:
                # Expire current record
                expire_query = f"""
                    UPDATE {table_name} 
                    SET is_current = false, expiry_date = CURRENT_TIMESTAMP 
                    WHERE {key_column} = :key_value AND is_current = true
                """
                await session.execute(
                    text(expire_query),
                    {"key_value": row_dict[key_column]}
                )
                
                # Insert new version
                row_dict["effective_date"] = datetime.now()
                row_dict["is_current"] = True
                columns = ", ".join(row_dict.keys())
                values = ", ".join([f":{k}" for k in row_dict.keys()])
                insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({values})"
                await session.execute(text(insert_query), row_dict)
        else:
            # Insert new record
            row_dict["effective_date"] = datetime.now()
            row_dict["is_current"] = True
            columns = ", ".join(row_dict.keys())
            values = ", ".join([f":{k}" for k in row_dict.keys()])
            insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({values})"
            await session.execute(text(insert_query), row_dict)
    
    async def _load_facts(
        self,
        data: pd.DataFrame,
        config: ETLConfig
    ) -> Dict[str, int]:
        """Load fact tables"""
        results = {}
        
        for fact_name, fact_config in config.fact_tables.items():
            logger.info(f"Loading fact table: {fact_name}")
            
            # Prepare fact data
            fact_data = await self._prepare_fact_data(data, fact_config)
            
            # Load to fact table
            table_name = fact_config.get("table", f"fact_{fact_name}")
            records_loaded = await self._insert_facts(fact_data, table_name)
            
            results[fact_name] = records_loaded
            etl_records_processed.labels(table_name=table_name).inc(records_loaded)
        
        return results
    
    async def _prepare_fact_data(
        self,
        data: pd.DataFrame,
        config: Dict
    ) -> pd.DataFrame:
        """Prepare data for fact table loading"""
        # Map to dimension keys
        for dim_mapping in config.get("dimension_mappings", []):
            dim_table = dim_mapping.get("table")
            source_column = dim_mapping.get("source_column")
            target_column = dim_mapping.get("target_column")
            lookup_column = dim_mapping.get("lookup_column")
            
            # Lookup dimension keys
            data[target_column] = await self._lookup_dimension_keys(
                data[source_column],
                dim_table,
                lookup_column
            )
        
        # Add date and time dimensions
        if config.get("add_date_dimension"):
            data["date_key"] = data[config.get("date_column")].apply(
                lambda x: int(pd.to_datetime(x).strftime("%Y%m%d"))
            )
        
        if config.get("add_time_dimension"):
            data["time_key"] = data[config.get("time_column")].apply(
                lambda x: int(pd.to_datetime(x).strftime("%H%M%S"))
            )
        
        return data
    
    async def _lookup_dimension_keys(
        self,
        values: pd.Series,
        dim_table: str,
        lookup_column: str
    ) -> pd.Series:
        """Lookup dimension keys for natural keys"""
        # Check cache first
        cache_key = f"{dim_table}:{lookup_column}"
        if cache_key not in self.dimension_cache:
            # Load dimension mapping
            query = f"SELECT {lookup_column}, {lookup_column}_key FROM {dim_table}"
            async with AsyncSession(self.db_engine) as session:
                result = await session.execute(text(query))
                mapping = dict(result.fetchall())
            self.dimension_cache[cache_key] = mapping
        
        mapping = self.dimension_cache[cache_key]
        return values.map(mapping)
    
    async def _insert_facts(
        self,
        data: pd.DataFrame,
        table_name: str
    ) -> int:
        """Insert records into fact table"""
        records_loaded = 0
        
        # Convert DataFrame to records
        records = data.to_dict("records")
        
        # Batch insert for performance
        batch_size = 1000
        async with AsyncSession(self.db_engine) as session:
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                
                if batch:
                    columns = ", ".join(batch[0].keys())
                    values_template = ", ".join([f":{k}" for k in batch[0].keys()])
                    insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({values_template})"
                    
                    for record in batch:
                        await session.execute(text(insert_query), record)
                        records_loaded += 1
            
            await session.commit()
        
        return records_loaded
    
    async def _update_watermark(self, config: ETLConfig, data: pd.DataFrame):
        """Update watermark for incremental loading"""
        if not data.empty and config.watermark_column in data.columns:
            max_value = data[config.watermark_column].max()
            
            if self.redis_client:
                watermark_key = f"etl_watermark:{config.pipeline_name}"
                await self.redis_client.set(watermark_key, str(max_value))
    
    async def _get_watermark(self, pipeline_name: str) -> Optional[str]:
        """Get watermark for incremental loading"""
        if self.redis_client:
            watermark_key = f"etl_watermark:{pipeline_name}"
            watermark = await self.redis_client.get(watermark_key)
            return watermark.decode() if watermark else None
        return None
    
    def _generate_job_id(self, pipeline_name: str) -> str:
        """Generate unique job ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        hash_input = f"{pipeline_name}_{timestamp}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    async def _cache_result(self, result: ETLResult):
        """Cache ETL result"""
        if self.redis_client:
            key = f"etl_result:{result.job_id}"
            value = json.dumps(asdict(result), default=str)
            await self.redis_client.set(key, value, ex=86400 * 7)  # 7 days TTL
    
    async def get_job_status(self, job_id: str) -> Optional[ETLResult]:
        """Get job status"""
        # Check active jobs
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        
        # Check cache
        if self.redis_client:
            key = f"etl_result:{job_id}"
            value = await self.redis_client.get(key)
            if value:
                data = json.loads(value)
                return ETLResult(**data)
        
        return None
    
    async def schedule_pipeline(
        self,
        config: ETLConfig,
        cron_expression: str
    ):
        """Schedule periodic ETL pipeline execution"""
        if self.redis_client:
            schedule_data = {
                "config": asdict(config),
                "cron": cron_expression,
                "next_run": datetime.now().isoformat(),
                "created_at": datetime.now().isoformat()
            }
            
            await self.redis_client.set(
                f"etl_schedule:{config.pipeline_name}",
                json.dumps(schedule_data),
                ex=86400 * 30  # 30 days TTL
            )
            
            logger.info(f"Scheduled ETL pipeline {config.pipeline_name}: {cron_expression}")


# Example usage
async def main():
    # Initialize pipeline
    pipeline = AdvancedETLPipeline(
        database_url="postgresql+asyncpg://user:pass@localhost/ytempire",
        redis_url="redis://localhost:6379"
    )
    
    await pipeline.initialize()
    
    # Configure ETL pipeline
    config = ETLConfig(
        pipeline_name="video_performance_etl",
        source_config={
            "type": "database",
            "query": """
                SELECT 
                    v.video_id, v.title, v.channel_id,
                    vp.views, vp.likes, vp.comments,
                    vp.recorded_at
                FROM videos v
                JOIN video_performance vp ON v.id = vp.video_id
                WHERE vp.recorded_at >= NOW() - INTERVAL '24 hours'
            """,
            "incremental": True,
            "watermark_column": "recorded_at"
        },
        target_config={
            "database": "analytics"
        },
        dimensions={
            "channel": {
                "table": "dim_channel",
                "columns": ["channel_id", "channel_name", "channel_type"],
                "key_column": "channel_id",
                "scd_type": 2
            },
            "video": {
                "table": "dim_video",
                "columns": ["video_id", "title", "duration", "category"],
                "key_column": "video_id",
                "scd_type": 1
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
                "date_column": "recorded_at",
                "add_time_dimension": True,
                "time_column": "recorded_at"
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
                    "engagement_rate": "data['likes'] / data['views']",
                    "comment_rate": "data['comments'] / data['views']"
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
        ]
    )
    
    # Run ETL pipeline
    result = await pipeline.run_pipeline(config)
    
    print(f"Pipeline completed: {result.status.value}")
    print(f"Records processed: {result.records_transformed}")
    print(f"Quality score: {result.quality_score:.2%}")
    print(f"Dimensions updated: {result.dimensions_updated}")
    print(f"Facts inserted: {result.facts_inserted}")
    
    # Schedule for periodic execution
    await pipeline.schedule_pipeline(config, "0 */6 * * *")  # Every 6 hours


if __name__ == "__main__":
    asyncio.run(main())