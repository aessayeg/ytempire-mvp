"""
Data Lake Architecture Implementation
Manages raw data storage, processing, and versioning
"""
import os
import json
import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import hashlib
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum
import asyncio
import aiofiles
from minio import Minio
from minio.error import S3Error
import duckdb
from delta import DeltaTable, configure_spark_with_delta_pip
from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)

class DataFormat(Enum):
    """Supported data formats in the lake"""
    PARQUET = "parquet"
    JSON = "json"
    CSV = "csv"
    AVRO = "avro"
    DELTA = "delta"

class DataZone(Enum):
    """Data lake zones for different processing stages"""
    LANDING = "landing"      # Raw data as received
    RAW = "raw"              # Validated raw data
    BRONZE = "bronze"        # Cleansed data
    SILVER = "silver"        # Transformed data
    GOLD = "gold"           # Business-ready data

@dataclass
class DataLakeConfig:
    """Configuration for data lake"""
    storage_type: str = "minio"  # minio, s3, local
    endpoint: str = "localhost:9000"
    access_key: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    secret_key: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    bucket_prefix: str = "ytempire"
    use_ssl: bool = False
    region: str = "us-east-1"
    local_cache_dir: str = "/data/lake/cache"
    metadata_db: str = "duckdb:///:memory:"
    enable_versioning: bool = True
    retention_days: int = 90
    compression: str = "snappy"

class DataLakeService:
    """Main data lake service for managing data lifecycle"""
    
    def __init__(self, config: DataLakeConfig = None):
        self.config = config or DataLakeConfig()
        self._init_storage()
        self._init_metadata_store()
        self._init_spark()
        
    def _init_storage(self):
        """Initialize storage backend"""
        if self.config.storage_type == "minio":
            self.storage = Minio(
                self.config.endpoint,
                access_key=self.config.access_key,
                secret_key=self.config.secret_key,
                secure=self.config.use_ssl
            )
            self._ensure_buckets()
        elif self.config.storage_type == "s3":
            self.storage = boto3.client(
                's3',
                aws_access_key_id=self.config.access_key,
                aws_secret_access_key=self.config.secret_key,
                region_name=self.config.region
            )
        else:  # local
            self.storage = LocalStorage(self.config.local_cache_dir)
            
    def _init_metadata_store(self):
        """Initialize metadata storage using DuckDB"""
        self.metadata_conn = duckdb.connect(self.config.metadata_db)
        
        # Create metadata tables
        self.metadata_conn.execute("""
            CREATE TABLE IF NOT EXISTS data_catalog (
                dataset_id VARCHAR PRIMARY KEY,
                dataset_name VARCHAR,
                zone VARCHAR,
                format VARCHAR,
                path VARCHAR,
                size_bytes BIGINT,
                row_count BIGINT,
                columns JSON,
                schema_version INTEGER,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                tags JSON,
                lineage JSON
            )
        """)
        
        self.metadata_conn.execute("""
            CREATE TABLE IF NOT EXISTS data_versions (
                version_id VARCHAR PRIMARY KEY,
                dataset_id VARCHAR,
                version_number INTEGER,
                path VARCHAR,
                checksum VARCHAR,
                created_at TIMESTAMP,
                created_by VARCHAR,
                is_current BOOLEAN,
                FOREIGN KEY (dataset_id) REFERENCES data_catalog(dataset_id)
            )
        """)
        
        self.metadata_conn.execute("""
            CREATE TABLE IF NOT EXISTS data_quality (
                check_id VARCHAR PRIMARY KEY,
                dataset_id VARCHAR,
                check_type VARCHAR,
                check_result JSON,
                passed BOOLEAN,
                executed_at TIMESTAMP,
                FOREIGN KEY (dataset_id) REFERENCES data_catalog(dataset_id)
            )
        """)
        
    def _init_spark(self):
        """Initialize Spark session for large-scale processing"""
        builder = SparkSession.builder \
            .appName("YTEmpireDataLake") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            
        if self.config.storage_type == "minio":
            builder = builder \
                .config("spark.hadoop.fs.s3a.endpoint", f"http://{self.config.endpoint}") \
                .config("spark.hadoop.fs.s3a.access.key", self.config.access_key) \
                .config("spark.hadoop.fs.s3a.secret.key", self.config.secret_key) \
                .config("spark.hadoop.fs.s3a.path.style.access", "true") \
                .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
                
        self.spark = configure_spark_with_delta_pip(builder).getOrCreate()
        
    def _ensure_buckets(self):
        """Ensure all required buckets exist"""
        for zone in DataZone:
            bucket_name = f"{self.config.bucket_prefix}-{zone.value}"
            try:
                if not self.storage.bucket_exists(bucket_name):
                    self.storage.make_bucket(bucket_name)
                    logger.info(f"Created bucket: {bucket_name}")
                    
                    if self.config.enable_versioning:
                        # Enable versioning on bucket
                        self.storage.set_bucket_versioning(bucket_name, True)
            except S3Error as e:
                logger.error(f"Error creating bucket {bucket_name}: {e}")
                
    async def ingest_data(
        self,
        data: Union[pd.DataFrame, Dict, List, str],
        dataset_name: str,
        zone: DataZone = DataZone.LANDING,
        format: DataFormat = DataFormat.PARQUET,
        partition_by: Optional[List[str]] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Ingest data into the data lake"""
        try:
            # Generate dataset ID
            dataset_id = self._generate_dataset_id(dataset_name)
            
            # Convert data to DataFrame if needed
            if isinstance(data, dict) or isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, str):
                # Assume it's a file path
                df = await self._read_file(data)
            else:
                df = data
                
            # Validate data
            validation_result = await self._validate_data(df)
            if not validation_result["passed"]:
                logger.warning(f"Data validation failed: {validation_result['errors']}")
                
            # Generate path
            path = self._generate_path(dataset_name, zone, format, partition_by)
            
            # Write data based on format
            if format == DataFormat.PARQUET:
                await self._write_parquet(df, path, partition_by)
            elif format == DataFormat.DELTA:
                await self._write_delta(df, path, partition_by)
            elif format == DataFormat.JSON:
                await self._write_json(df, path)
            elif format == DataFormat.CSV:
                await self._write_csv(df, path)
                
            # Calculate metadata
            metadata = {
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "zone": zone.value,
                "format": format.value,
                "path": path,
                "size_bytes": df.memory_usage(deep=True).sum(),
                "row_count": len(df),
                "columns": df.columns.tolist(),
                "schema": self._extract_schema(df),
                "created_at": datetime.utcnow(),
                "tags": tags or {}
            }
            
            # Store metadata
            await self._store_metadata(metadata)
            
            # Trigger quality checks
            asyncio.create_task(self._run_quality_checks(dataset_id, df))
            
            logger.info(f"Successfully ingested {dataset_name} to {zone.value} zone")
            return dataset_id
            
        except Exception as e:
            logger.error(f"Error ingesting data: {e}")
            raise
            
    async def read_data(
        self,
        dataset_id: str,
        version: Optional[int] = None,
        columns: Optional[List[str]] = None,
        filters: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Read data from the data lake"""
        try:
            # Get metadata
            metadata = self._get_metadata(dataset_id)
            if not metadata:
                raise ValueError(f"Dataset {dataset_id} not found")
                
            path = metadata["path"]
            format = DataFormat(metadata["format"])
            
            # Read data based on format
            if format == DataFormat.PARQUET:
                df = await self._read_parquet(path, columns, filters)
            elif format == DataFormat.DELTA:
                df = await self._read_delta(path, version, columns, filters)
            elif format == DataFormat.JSON:
                df = await self._read_json(path)
            elif format == DataFormat.CSV:
                df = await self._read_csv(path)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            return df
            
        except Exception as e:
            logger.error(f"Error reading data: {e}")
            raise
            
    async def transform_data(
        self,
        source_dataset_id: str,
        target_dataset_name: str,
        transformation_func: callable,
        target_zone: DataZone = DataZone.SILVER,
        target_format: DataFormat = DataFormat.PARQUET
    ) -> str:
        """Transform data and store in a higher zone"""
        try:
            # Read source data
            source_df = await self.read_data(source_dataset_id)
            
            # Apply transformation
            transformed_df = transformation_func(source_df)
            
            # Ingest transformed data
            target_dataset_id = await self.ingest_data(
                transformed_df,
                target_dataset_name,
                target_zone,
                target_format
            )
            
            # Track lineage
            await self._track_lineage(source_dataset_id, target_dataset_id, "transformation")
            
            return target_dataset_id
            
        except Exception as e:
            logger.error(f"Error transforming data: {e}")
            raise
            
    async def _write_parquet(
        self,
        df: pd.DataFrame,
        path: str,
        partition_by: Optional[List[str]] = None
    ):
        """Write DataFrame to Parquet format"""
        table = pa.Table.from_pandas(df)
        
        if partition_by:
            pq.write_to_dataset(
                table,
                root_path=path,
                partition_cols=partition_by,
                compression=self.config.compression
            )
        else:
            pq.write_table(
                table,
                path,
                compression=self.config.compression
            )
            
    async def _write_delta(
        self,
        df: pd.DataFrame,
        path: str,
        partition_by: Optional[List[str]] = None
    ):
        """Write DataFrame to Delta Lake format"""
        spark_df = self.spark.createDataFrame(df)
        
        writer = spark_df.write.format("delta").mode("overwrite")
        
        if partition_by:
            writer = writer.partitionBy(*partition_by)
            
        writer.save(path)
        
    async def _read_parquet(
        self,
        path: str,
        columns: Optional[List[str]] = None,
        filters: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Read Parquet file(s)"""
        return pd.read_parquet(path, columns=columns, filters=filters)
        
    async def _read_delta(
        self,
        path: str,
        version: Optional[int] = None,
        columns: Optional[List[str]] = None,
        filters: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Read Delta Lake table"""
        if version:
            df = self.spark.read.format("delta").option("versionAsOf", version).load(path)
        else:
            df = self.spark.read.format("delta").load(path)
            
        if columns:
            df = df.select(*columns)
            
        if filters:
            for col, value in filters.items():
                df = df.filter(df[col] == value)
                
        return df.toPandas()
        
    async def _validate_data(self, df: pd.DataFrame) -> Dict:
        """Validate data quality"""
        errors = []
        
        # Check for null values
        null_counts = df.isnull().sum()
        if null_counts.any():
            errors.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
            
        # Check for duplicates
        if df.duplicated().any():
            errors.append(f"Duplicate rows found: {df.duplicated().sum()}")
            
        # Check data types
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for mixed types in object columns
                try:
                    pd.to_numeric(df[col])
                except:
                    pass  # Expected for string columns
                    
        return {
            "passed": len(errors) == 0,
            "errors": errors
        }
        
    async def _run_quality_checks(self, dataset_id: str, df: pd.DataFrame):
        """Run data quality checks"""
        checks = []
        
        # Completeness check
        completeness = (1 - df.isnull().sum() / len(df)).to_dict()
        checks.append({
            "type": "completeness",
            "result": completeness,
            "passed": all(v > 0.95 for v in completeness.values())
        })
        
        # Uniqueness check
        for col in df.columns:
            uniqueness = df[col].nunique() / len(df)
            checks.append({
                "type": f"uniqueness_{col}",
                "result": {"ratio": uniqueness},
                "passed": True  # Define threshold based on column
            })
            
        # Store quality results
        for check in checks:
            check_id = f"{dataset_id}_{check['type']}_{datetime.utcnow().timestamp()}"
            self.metadata_conn.execute("""
                INSERT INTO data_quality (check_id, dataset_id, check_type, check_result, passed, executed_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (check_id, dataset_id, check["type"], json.dumps(check["result"]), 
                  check["passed"], datetime.utcnow()))
                  
    def _generate_dataset_id(self, dataset_name: str) -> str:
        """Generate unique dataset ID"""
        timestamp = datetime.utcnow().isoformat()
        return hashlib.md5(f"{dataset_name}_{timestamp}".encode()).hexdigest()[:16]
        
    def _generate_path(
        self,
        dataset_name: str,
        zone: DataZone,
        format: DataFormat,
        partition_by: Optional[List[str]] = None
    ) -> str:
        """Generate storage path"""
        date_str = datetime.utcnow().strftime("%Y/%m/%d")
        
        if self.config.storage_type in ["minio", "s3"]:
            bucket = f"{self.config.bucket_prefix}-{zone.value}"
            path = f"s3a://{bucket}/{dataset_name}/{date_str}"
        else:
            path = f"{self.config.local_cache_dir}/{zone.value}/{dataset_name}/{date_str}"
            
        if format != DataFormat.DELTA:
            path += f".{format.value}"
            
        return path
        
    def _extract_schema(self, df: pd.DataFrame) -> Dict:
        """Extract schema from DataFrame"""
        schema = {}
        for col in df.columns:
            schema[col] = {
                "type": str(df[col].dtype),
                "nullable": df[col].isnull().any(),
                "unique": df[col].nunique() == len(df)
            }
        return schema
        
    async def _store_metadata(self, metadata: Dict):
        """Store metadata in catalog"""
        self.metadata_conn.execute("""
            INSERT OR REPLACE INTO data_catalog 
            (dataset_id, dataset_name, zone, format, path, size_bytes, row_count, 
             columns, created_at, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metadata["dataset_id"],
            metadata["dataset_name"],
            metadata["zone"],
            metadata["format"],
            metadata["path"],
            metadata["size_bytes"],
            metadata["row_count"],
            json.dumps(metadata["columns"]),
            metadata["created_at"],
            json.dumps(metadata["tags"])
        ))
        
    def _get_metadata(self, dataset_id: str) -> Optional[Dict]:
        """Get metadata for dataset"""
        result = self.metadata_conn.execute("""
            SELECT * FROM data_catalog WHERE dataset_id = ?
        """, (dataset_id,)).fetchone()
        
        if result:
            return dict(result)
        return None
        
    async def _track_lineage(
        self,
        source_id: str,
        target_id: str,
        operation: str
    ):
        """Track data lineage"""
        # Update source lineage
        source_meta = self._get_metadata(source_id)
        if source_meta:
            lineage = json.loads(source_meta.get("lineage", "{}"))
            if "downstream" not in lineage:
                lineage["downstream"] = []
            lineage["downstream"].append({
                "dataset_id": target_id,
                "operation": operation,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            self.metadata_conn.execute("""
                UPDATE data_catalog SET lineage = ? WHERE dataset_id = ?
            """, (json.dumps(lineage), source_id))
            
        # Update target lineage
        target_meta = self._get_metadata(target_id)
        if target_meta:
            lineage = json.loads(target_meta.get("lineage", "{}"))
            lineage["upstream"] = {
                "dataset_id": source_id,
                "operation": operation,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.metadata_conn.execute("""
                UPDATE data_catalog SET lineage = ? WHERE dataset_id = ?
            """, (json.dumps(lineage), target_id))
            
    async def cleanup_old_data(self):
        """Clean up data older than retention period"""
        cutoff_date = datetime.utcnow() - timedelta(days=self.config.retention_days)
        
        # Find datasets to delete
        old_datasets = self.metadata_conn.execute("""
            SELECT dataset_id, path FROM data_catalog 
            WHERE created_at < ? AND zone != ?
        """, (cutoff_date, DataZone.GOLD.value)).fetchall()
        
        for dataset in old_datasets:
            dataset_id, path = dataset
            
            # Delete physical data
            if self.config.storage_type == "minio":
                bucket, key = path.replace("s3a://", "").split("/", 1)
                self.storage.remove_object(bucket, key)
            elif self.config.storage_type == "local":
                if os.path.exists(path):
                    os.remove(path)
                    
            # Delete metadata
            self.metadata_conn.execute("""
                DELETE FROM data_catalog WHERE dataset_id = ?
            """, (dataset_id,))
            
            logger.info(f"Cleaned up dataset: {dataset_id}")
            
class LocalStorage:
    """Local file system storage backend"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def bucket_exists(self, bucket: str) -> bool:
        return (self.base_path / bucket).exists()
        
    def make_bucket(self, bucket: str):
        (self.base_path / bucket).mkdir(parents=True, exist_ok=True)