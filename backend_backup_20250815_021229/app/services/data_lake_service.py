"""
Analytics Data Lake Service
S3-compatible object storage with partitioning, cataloging, and governance
"""
import asyncio
import json
import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple, AsyncIterator
from datetime import datetime, timedelta, date
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import aioboto3
from botocore.exceptions import ClientError
import redis.asyncio as redis

from app.core.config import settings

logger = logging.getLogger(__name__)


class StorageTier(str, Enum):
    """Data storage tiers"""
    HOT = "hot"  # Frequent access (SSD)
    WARM = "warm"  # Occasional access (HDD)
    COLD = "cold"  # Rare access (Archive)
    GLACIER = "glacier"  # Long-term archive


class DataFormat(str, Enum):
    """Supported data formats"""
    PARQUET = "parquet"
    ORC = "orc"
    AVRO = "avro"
    JSON = "json"
    CSV = "csv"
    DELTA = "delta"


class PartitionStrategy(str, Enum):
    """Data partitioning strategies"""
    DATE = "date"
    HOUR = "hour"
    CHANNEL = "channel"
    USER = "user"
    COMPOSITE = "composite"


@dataclass
class DataLakeObject:
    """Data lake object metadata"""
    object_id: str
    bucket: str
    key: str
    size_bytes: int
    format: DataFormat
    partition_keys: Dict[str, str]
    schema_version: str
    created_at: datetime
    modified_at: datetime
    storage_tier: StorageTier
    compression: Optional[str]
    encryption: Optional[str]
    checksum: str
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataCatalogEntry:
    """Data catalog entry for discovery"""
    catalog_id: str
    dataset_name: str
    table_name: str
    description: str
    location: str  # S3 path
    format: DataFormat
    partition_columns: List[str]
    schema: Dict[str, str]  # column_name: data_type
    statistics: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    owner: str
    access_level: str
    tags: List[str] = field(default_factory=list)


@dataclass
class DataLifecyclePolicy:
    """Data lifecycle management policy"""
    policy_id: str
    name: str
    description: str
    rules: List[Dict[str, Any]]
    created_at: datetime
    is_active: bool


class DataLakeService:
    """Service for managing analytics data lake"""
    
    def __init__(self):
        self.s3_client = None
        self.redis_client: Optional[redis.Redis] = None
        self.bucket_name = settings.DATA_LAKE_BUCKET or "ytempire-data-lake"
        self.catalog_db = "ytempire_catalog"
        
        # Storage configuration
        self.storage_tiers = {
            StorageTier.HOT: {"retention_days": 7, "storage_class": "STANDARD"},
            StorageTier.WARM: {"retention_days": 30, "storage_class": "STANDARD_IA"},
            StorageTier.COLD: {"retention_days": 90, "storage_class": "GLACIER"},
            StorageTier.GLACIER: {"retention_days": 365, "storage_class": "DEEP_ARCHIVE"}
        }
        
        # Partitioning configuration
        self.partition_configs = {
            "analytics": {
                "strategy": PartitionStrategy.DATE,
                "columns": ["year", "month", "day"]
            },
            "videos": {
                "strategy": PartitionStrategy.COMPOSITE,
                "columns": ["channel_id", "year", "month"]
            },
            "costs": {
                "strategy": PartitionStrategy.DATE,
                "columns": ["year", "month", "day"]
            },
            "user_behavior": {
                "strategy": PartitionStrategy.USER,
                "columns": ["user_id", "year", "month"]
            }
        }
        
    async def initialize(self):
        """Initialize data lake connections"""
        try:
            # Initialize S3 client (MinIO or AWS S3)
            session = aioboto3.Session(
                aws_access_key_id=settings.S3_ACCESS_KEY or "minioadmin",
                aws_secret_access_key=settings.S3_SECRET_KEY or "minioadmin",
                region_name=settings.S3_REGION or "us-east-1"
            )
            
            self.s3_client = session.client(
                's3',
                endpoint_url=settings.S3_ENDPOINT_URL or "http://localhost:9000"
            )
            
            # Initialize Redis for metadata
            self.redis_client = await redis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
            
            # Create bucket if not exists
            await self._ensure_bucket_exists()
            
            # Initialize catalog
            await self._initialize_catalog()
            
            logger.info("Data lake service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize data lake: {e}")
            raise
            
    async def ingest_data(
        self,
        dataset_name: str,
        data: pd.DataFrame,
        partition_by: Optional[List[str]] = None,
        format: DataFormat = DataFormat.PARQUET,
        storage_tier: StorageTier = StorageTier.HOT
    ) -> DataLakeObject:
        """Ingest data into the data lake"""
        # Generate object metadata
        object_id = self._generate_object_id(dataset_name)
        
        # Apply partitioning
        if partition_by:
            partition_keys = self._extract_partition_keys(data, partition_by)
            key = self._generate_partitioned_key(dataset_name, partition_keys, object_id, format)
        else:
            key = f"{dataset_name}/{datetime.utcnow().strftime('%Y/%m/%d')}/{object_id}.{format.value}"
            partition_keys = {}
            
        # Convert data to specified format
        data_bytes = await self._convert_to_format(data, format)
        
        # Calculate checksum
        checksum = hashlib.sha256(data_bytes).hexdigest()
        
        # Upload to S3
        async with self.s3_client as s3:
            await s3.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=data_bytes,
                StorageClass=self.storage_tiers[storage_tier]["storage_class"],
                Metadata={
                    "dataset": dataset_name,
                    "format": format.value,
                    "checksum": checksum,
                    "created_at": datetime.utcnow().isoformat()
                }
            )
            
        # Create object metadata
        obj = DataLakeObject(
            object_id=object_id,
            bucket=self.bucket_name,
            key=key,
            size_bytes=len(data_bytes),
            format=format,
            partition_keys=partition_keys,
            schema_version="1.0",
            created_at=datetime.utcnow(),
            modified_at=datetime.utcnow(),
            storage_tier=storage_tier,
            compression="snappy" if format == DataFormat.PARQUET else None,
            encryption="AES256",
            checksum=checksum
        )
        
        # Store metadata
        await self._store_object_metadata(obj)
        
        # Update catalog
        await self._update_catalog(dataset_name, data, key, format, partition_by)
        
        # Apply lifecycle policy
        await self._apply_lifecycle_policy(obj)
        
        logger.info(f"Ingested {len(data)} rows to {key}")
        return obj
        
    async def query_data(
        self,
        dataset_name: str,
        filters: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
        partition_filters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Query data from the data lake"""
        # Get catalog entry
        catalog_entry = await self._get_catalog_entry(dataset_name)
        if not catalog_entry:
            raise ValueError(f"Dataset {dataset_name} not found in catalog")
            
        # Build S3 prefix based on partition filters
        prefix = self._build_query_prefix(dataset_name, partition_filters)
        
        # List objects matching prefix
        objects = await self._list_objects(prefix)
        
        # Read and combine data
        dataframes = []
        for obj_key in objects[:limit] if limit else objects:
            df = await self._read_object(obj_key, catalog_entry.format)
            
            # Apply column selection
            if columns:
                df = df[columns]
                
            # Apply filters
            if filters:
                for col, value in filters.items():
                    if col in df.columns:
                        df = df[df[col] == value]
                        
            dataframes.append(df)
            
        # Combine results
        if dataframes:
            result = pd.concat(dataframes, ignore_index=True)
            return result[:limit] if limit else result
        else:
            return pd.DataFrame()
            
    async def create_external_table(
        self,
        table_name: str,
        dataset_name: str,
        schema: Dict[str, str],
        partition_columns: Optional[List[str]] = None,
        format: DataFormat = DataFormat.PARQUET
    ) -> DataCatalogEntry:
        """Create external table for SQL queries"""
        location = f"s3://{self.bucket_name}/{dataset_name}/"
        
        catalog_entry = DataCatalogEntry(
            catalog_id=self._generate_id("catalog"),
            dataset_name=dataset_name,
            table_name=table_name,
            description=f"External table for {dataset_name}",
            location=location,
            format=format,
            partition_columns=partition_columns or [],
            schema=schema,
            statistics={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            owner="system",
            access_level="read"
        )
        
        # Store catalog entry
        await self._store_catalog_entry(catalog_entry)
        
        # Create Hive-compatible metadata
        await self._create_hive_metadata(catalog_entry)
        
        logger.info(f"Created external table {table_name}")
        return catalog_entry
        
    async def optimize_storage(self, dataset_name: str) -> Dict[str, Any]:
        """Optimize storage for a dataset (compaction, format conversion)"""
        results = {
            "dataset": dataset_name,
            "files_before": 0,
            "files_after": 0,
            "size_before": 0,
            "size_after": 0,
            "optimization_time": 0
        }
        
        start_time = datetime.utcnow()
        
        # List all objects for dataset
        prefix = f"{dataset_name}/"
        objects = await self._list_objects_with_metadata(prefix)
        
        results["files_before"] = len(objects)
        results["size_before"] = sum(obj["Size"] for obj in objects)
        
        # Group small files by partition
        partitions = {}
        for obj in objects:
            partition_key = self._extract_partition_from_key(obj["Key"])
            if partition_key not in partitions:
                partitions[partition_key] = []
            partitions[partition_key].append(obj)
            
        # Compact partitions with many small files
        for partition_key, partition_objects in partitions.items():
            if len(partition_objects) > 10:  # Threshold for compaction
                await self._compact_partition(dataset_name, partition_key, partition_objects)
                
        # Re-count after optimization
        objects_after = await self._list_objects_with_metadata(prefix)
        results["files_after"] = len(objects_after)
        results["size_after"] = sum(obj["Size"] for obj in objects_after)
        results["optimization_time"] = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(f"Optimized {dataset_name}: {results['files_before']} -> {results['files_after']} files")
        return results
        
    async def apply_retention_policy(
        self,
        dataset_name: str,
        retention_days: int,
        archive_tier: StorageTier = StorageTier.COLD
    ) -> int:
        """Apply retention policy to dataset"""
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        archived_count = 0
        
        # List objects older than retention period
        prefix = f"{dataset_name}/"
        objects = await self._list_objects_with_metadata(prefix)
        
        for obj in objects:
            last_modified = obj["LastModified"]
            if last_modified < cutoff_date:
                # Move to archive tier
                await self._transition_storage_tier(
                    obj["Key"],
                    archive_tier
                )
                archived_count += 1
                
        logger.info(f"Archived {archived_count} objects from {dataset_name}")
        return archived_count
        
    async def create_data_pipeline(
        self,
        pipeline_name: str,
        source_dataset: str,
        target_dataset: str,
        transformations: List[Dict[str, Any]],
        schedule: Optional[str] = None  # cron expression
    ) -> Dict[str, Any]:
        """Create automated data pipeline"""
        pipeline_id = self._generate_id("pipeline")
        
        pipeline_config = {
            "pipeline_id": pipeline_id,
            "name": pipeline_name,
            "source": source_dataset,
            "target": target_dataset,
            "transformations": transformations,
            "schedule": schedule,
            "created_at": datetime.utcnow().isoformat(),
            "is_active": True
        }
        
        # Store pipeline configuration
        await self.redis_client.setex(
            f"pipeline:{pipeline_id}",
            86400 * 30,  # 30 days retention
            json.dumps(pipeline_config)
        )
        
        logger.info(f"Created data pipeline {pipeline_name}")
        return pipeline_config
        
    async def get_storage_metrics(self) -> Dict[str, Any]:
        """Get storage metrics for the data lake"""
        metrics = {
            "total_objects": 0,
            "total_size_bytes": 0,
            "by_tier": {},
            "by_format": {},
            "by_dataset": {},
            "oldest_object": None,
            "newest_object": None
        }
        
        # List all objects
        async with self.s3_client as s3:
            paginator = s3.get_paginator('list_objects_v2')
            
            async for page in paginator.paginate(Bucket=self.bucket_name):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        metrics["total_objects"] += 1
                        metrics["total_size_bytes"] += obj["Size"]
                        
                        # Parse metadata
                        key_parts = obj["Key"].split("/")
                        if key_parts:
                            dataset = key_parts[0]
                            if dataset not in metrics["by_dataset"]:
                                metrics["by_dataset"][dataset] = {
                                    "count": 0,
                                    "size": 0
                                }
                            metrics["by_dataset"][dataset]["count"] += 1
                            metrics["by_dataset"][dataset]["size"] += obj["Size"]
                            
                        # Track oldest/newest
                        if not metrics["oldest_object"] or obj["LastModified"] < metrics["oldest_object"]:
                            metrics["oldest_object"] = obj["LastModified"]
                        if not metrics["newest_object"] or obj["LastModified"] > metrics["newest_object"]:
                            metrics["newest_object"] = obj["LastModified"]
                            
        # Get tier distribution from metadata
        pattern = "object:*"
        cursor = 0
        while True:
            cursor, keys = await self.redis_client.scan(
                cursor=cursor,
                match=pattern,
                count=100
            )
            
            for key in keys:
                obj_data = await self.redis_client.get(key)
                if obj_data:
                    obj = json.loads(obj_data)
                    tier = obj.get("storage_tier", "hot")
                    format_type = obj.get("format", "unknown")
                    
                    if tier not in metrics["by_tier"]:
                        metrics["by_tier"][tier] = {"count": 0, "size": 0}
                    metrics["by_tier"][tier]["count"] += 1
                    metrics["by_tier"][tier]["size"] += obj.get("size_bytes", 0)
                    
                    if format_type not in metrics["by_format"]:
                        metrics["by_format"][format_type] = {"count": 0, "size": 0}
                    metrics["by_format"][format_type]["count"] += 1
                    metrics["by_format"][format_type]["size"] += obj.get("size_bytes", 0)
                    
            if cursor == 0:
                break
                
        return metrics
        
    async def _ensure_bucket_exists(self):
        """Ensure S3 bucket exists"""
        async with self.s3_client as s3:
            try:
                await s3.head_bucket(Bucket=self.bucket_name)
            except ClientError:
                # Create bucket
                await s3.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': settings.S3_REGION or 'us-east-1'}
                )
                
                # Enable versioning
                await s3.put_bucket_versioning(
                    Bucket=self.bucket_name,
                    VersioningConfiguration={'Status': 'Enabled'}
                )
                
                # Set lifecycle configuration
                await self._setup_lifecycle_rules()
                
                logger.info(f"Created bucket {self.bucket_name}")
                
    async def _setup_lifecycle_rules(self):
        """Setup S3 lifecycle rules"""
        rules = []
        
        for tier, config in self.storage_tiers.items():
            if tier != StorageTier.HOT:
                rules.append({
                    'ID': f'transition-to-{tier.value}',
                    'Status': 'Enabled',
                    'Transitions': [{
                        'Days': config['retention_days'],
                        'StorageClass': config['storage_class']
                    }]
                })
                
        if rules:
            async with self.s3_client as s3:
                await s3.put_bucket_lifecycle_configuration(
                    Bucket=self.bucket_name,
                    LifecycleConfiguration={'Rules': rules}
                )
                
    async def _initialize_catalog(self):
        """Initialize data catalog"""
        # Create catalog database in Redis
        catalog_key = f"catalog:database:{self.catalog_db}"
        catalog_metadata = {
            "name": self.catalog_db,
            "created_at": datetime.utcnow().isoformat(),
            "tables": []
        }
        
        await self.redis_client.set(
            catalog_key,
            json.dumps(catalog_metadata)
        )
        
    async def _convert_to_format(
        self,
        data: pd.DataFrame,
        format: DataFormat
    ) -> bytes:
        """Convert DataFrame to specified format"""
        buffer = io.BytesIO()
        
        if format == DataFormat.PARQUET:
            table = pa.Table.from_pandas(data)
            pq.write_table(table, buffer, compression='snappy')
        elif format == DataFormat.JSON:
            data.to_json(buffer, orient='records', date_format='iso')
        elif format == DataFormat.CSV:
            data.to_csv(buffer, index=False)
        else:
            # Default to Parquet
            table = pa.Table.from_pandas(data)
            pq.write_table(table, buffer)
            
        buffer.seek(0)
        return buffer.read()
        
    async def _read_object(
        self,
        key: str,
        format: DataFormat
    ) -> pd.DataFrame:
        """Read object from S3"""
        async with self.s3_client as s3:
            response = await s3.get_object(Bucket=self.bucket_name, Key=key)
            data = await response['Body'].read()
            
        if format == DataFormat.PARQUET:
            return pd.read_parquet(io.BytesIO(data))
        elif format == DataFormat.JSON:
            return pd.read_json(io.BytesIO(data), orient='records')
        elif format == DataFormat.CSV:
            return pd.read_csv(io.BytesIO(data))
        else:
            return pd.DataFrame()
            
    async def _list_objects(self, prefix: str) -> List[str]:
        """List objects with prefix"""
        objects = []
        
        async with self.s3_client as s3:
            paginator = s3.get_paginator('list_objects_v2')
            
            async for page in paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=prefix
            ):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        objects.append(obj['Key'])
                        
        return objects
        
    async def _list_objects_with_metadata(
        self,
        prefix: str
    ) -> List[Dict[str, Any]]:
        """List objects with full metadata"""
        objects = []
        
        async with self.s3_client as s3:
            paginator = s3.get_paginator('list_objects_v2')
            
            async for page in paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=prefix
            ):
                if 'Contents' in page:
                    objects.extend(page['Contents'])
                    
        return objects
        
    async def _compact_partition(
        self,
        dataset_name: str,
        partition_key: str,
        objects: List[Dict[str, Any]]
    ):
        """Compact small files in a partition"""
        # Read all objects
        dataframes = []
        for obj in objects:
            df = await self._read_object(obj['Key'], DataFormat.PARQUET)
            dataframes.append(df)
            
        # Combine data
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Write combined file
        new_key = f"{dataset_name}/{partition_key}/compacted_{self._generate_id('file')}.parquet"
        data_bytes = await self._convert_to_format(combined_df, DataFormat.PARQUET)
        
        async with self.s3_client as s3:
            await s3.put_object(
                Bucket=self.bucket_name,
                Key=new_key,
                Body=data_bytes
            )
            
            # Delete original files
            for obj in objects:
                await s3.delete_object(
                    Bucket=self.bucket_name,
                    Key=obj['Key']
                )
                
    async def _transition_storage_tier(
        self,
        key: str,
        new_tier: StorageTier
    ):
        """Transition object to different storage tier"""
        async with self.s3_client as s3:
            # Copy object with new storage class
            await s3.copy_object(
                Bucket=self.bucket_name,
                CopySource={'Bucket': self.bucket_name, 'Key': key},
                Key=key,
                StorageClass=self.storage_tiers[new_tier]['storage_class']
            )
            
    def _extract_partition_keys(
        self,
        data: pd.DataFrame,
        partition_by: List[str]
    ) -> Dict[str, str]:
        """Extract partition key values from data"""
        partition_keys = {}
        
        for col in partition_by:
            if col in data.columns:
                # Use first value as partition key
                value = data[col].iloc[0] if len(data) > 0 else "unknown"
                partition_keys[col] = str(value)
            elif col in ["year", "month", "day", "hour"]:
                # Extract from timestamp if present
                if "timestamp" in data.columns or "created_at" in data.columns:
                    time_col = "timestamp" if "timestamp" in data.columns else "created_at"
                    timestamp = pd.to_datetime(data[time_col].iloc[0])
                    
                    if col == "year":
                        partition_keys[col] = str(timestamp.year)
                    elif col == "month":
                        partition_keys[col] = f"{timestamp.month:02d}"
                    elif col == "day":
                        partition_keys[col] = f"{timestamp.day:02d}"
                    elif col == "hour":
                        partition_keys[col] = f"{timestamp.hour:02d}"
                        
        return partition_keys
        
    def _generate_partitioned_key(
        self,
        dataset_name: str,
        partition_keys: Dict[str, str],
        object_id: str,
        format: DataFormat
    ) -> str:
        """Generate S3 key with partition structure"""
        # Build Hive-style partition path
        partition_path = "/".join(
            f"{k}={v}" for k, v in partition_keys.items()
        )
        
        return f"{dataset_name}/{partition_path}/{object_id}.{format.value}"
        
    def _build_query_prefix(
        self,
        dataset_name: str,
        partition_filters: Optional[Dict[str, Any]]
    ) -> str:
        """Build S3 prefix for querying"""
        prefix = f"{dataset_name}/"
        
        if partition_filters:
            partition_path = "/".join(
                f"{k}={v}" for k, v in partition_filters.items()
            )
            prefix += partition_path + "/"
            
        return prefix
        
    def _extract_partition_from_key(self, key: str) -> str:
        """Extract partition identifier from S3 key"""
        parts = key.split("/")
        # Remove dataset name and filename
        if len(parts) > 2:
            return "/".join(parts[1:-1])
        return ""
        
    async def _store_object_metadata(self, obj: DataLakeObject):
        """Store object metadata in Redis"""
        key = f"object:{obj.object_id}"
        await self.redis_client.setex(
            key,
            86400 * 90,  # 90 days retention
            json.dumps(asdict(obj), default=str)
        )
        
    async def _store_catalog_entry(self, entry: DataCatalogEntry):
        """Store catalog entry"""
        key = f"catalog:table:{entry.table_name}"
        await self.redis_client.set(
            key,
            json.dumps(asdict(entry), default=str)
        )
        
    async def _get_catalog_entry(
        self,
        dataset_name: str
    ) -> Optional[DataCatalogEntry]:
        """Get catalog entry for dataset"""
        key = f"catalog:table:{dataset_name}"
        data = await self.redis_client.get(key)
        
        if data:
            entry_dict = json.loads(data)
            # Convert timestamps
            entry_dict['created_at'] = datetime.fromisoformat(entry_dict['created_at'])
            entry_dict['updated_at'] = datetime.fromisoformat(entry_dict['updated_at'])
            return DataCatalogEntry(**entry_dict)
            
        return None
        
    async def _update_catalog(
        self,
        dataset_name: str,
        data: pd.DataFrame,
        location: str,
        format: DataFormat,
        partition_columns: Optional[List[str]]
    ):
        """Update catalog with new data"""
        # Get or create catalog entry
        entry = await self._get_catalog_entry(dataset_name)
        
        if not entry:
            # Create new entry
            schema = {col: str(dtype) for col, dtype in data.dtypes.items()}
            
            entry = DataCatalogEntry(
                catalog_id=self._generate_id("catalog"),
                dataset_name=dataset_name,
                table_name=dataset_name,
                description=f"Data lake table for {dataset_name}",
                location=f"s3://{self.bucket_name}/{dataset_name}/",
                format=format,
                partition_columns=partition_columns or [],
                schema=schema,
                statistics={
                    "row_count": len(data),
                    "size_bytes": data.memory_usage(deep=True).sum()
                },
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                owner="system",
                access_level="read-write"
            )
        else:
            # Update statistics
            entry.statistics["row_count"] = entry.statistics.get("row_count", 0) + len(data)
            entry.statistics["size_bytes"] = entry.statistics.get("size_bytes", 0) + data.memory_usage(deep=True).sum()
            entry.updated_at = datetime.utcnow()
            
        await self._store_catalog_entry(entry)
        
    async def _create_hive_metadata(self, entry: DataCatalogEntry):
        """Create Hive-compatible metadata for external table"""
        # This would integrate with Hive Metastore or AWS Glue
        # For now, store in Redis
        hive_metadata = {
            "tableName": entry.table_name,
            "location": entry.location,
            "inputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
            "outputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
            "serdeInfo": {
                "serializationLib": "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
            },
            "columns": [
                {"name": col, "type": dtype}
                for col, dtype in entry.schema.items()
            ],
            "partitionKeys": entry.partition_columns
        }
        
        key = f"hive:table:{entry.table_name}"
        await self.redis_client.set(
            key,
            json.dumps(hive_metadata)
        )
        
    async def _apply_lifecycle_policy(self, obj: DataLakeObject):
        """Apply lifecycle policy to object"""
        # Schedule transition based on storage tier
        if obj.storage_tier == StorageTier.HOT:
            # Schedule transition to WARM after 7 days
            transition_date = obj.created_at + timedelta(days=7)
            
            task = {
                "object_id": obj.object_id,
                "action": "transition",
                "target_tier": StorageTier.WARM.value,
                "scheduled_at": transition_date.isoformat()
            }
            
            await self.redis_client.zadd(
                "lifecycle:tasks",
                {json.dumps(task): transition_date.timestamp()}
            )
            
    def _generate_object_id(self, dataset_name: str) -> str:
        """Generate unique object ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        return f"{dataset_name}_{timestamp}"
        
    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        return f"{prefix}_{timestamp}"


# Singleton instance
data_lake_service = DataLakeService()