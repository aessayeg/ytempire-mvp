"""
Training Data Management Service
Comprehensive system for managing ML training datasets with versioning and lineage tracking
"""
import asyncio
import json
import hashlib
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import gzip
import shutil
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
import redis.asyncio as redis

from app.core.config import settings
from app.core.cache import cache_service

logger = logging.getLogger(__name__)


class DatasetType(str, Enum):
    """Types of training datasets"""

    CONTENT_GENERATION = "content_generation"
    TREND_PREDICTION = "trend_prediction"
    QUALITY_SCORING = "quality_scoring"
    THUMBNAIL_OPTIMIZATION = "thumbnail_optimization"
    ENGAGEMENT_PREDICTION = "engagement_prediction"
    REVENUE_FORECAST = "revenue_forecast"
    USER_BEHAVIOR = "user_behavior"
    AB_TEST_RESULTS = "ab_test_results"


class DatasetStatus(str, Enum):
    """Dataset lifecycle status"""

    COLLECTING = "collecting"
    PROCESSING = "processing"
    VALIDATED = "validated"
    READY = "ready"
    TRAINING = "training"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class ValidationStatus(str, Enum):
    """Data validation status"""

    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class DatasetVersion:
    """Dataset version information"""

    version_id: str
    dataset_id: str
    version_number: str  # e.g., "1.0.0"
    created_at: datetime
    created_by: str
    parent_version: Optional[str]
    changes_description: str
    size_bytes: int
    row_count: int
    column_count: int
    checksum: str
    storage_path: str
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataLineage:
    """Data lineage tracking"""

    lineage_id: str
    dataset_id: str
    source_datasets: List[str]
    transformations: List[Dict[str, Any]]
    created_at: datetime
    processing_time_seconds: float
    quality_metrics: Dict[str, float]
    dependencies: List[str]


@dataclass
class DataValidation:
    """Data validation results"""

    validation_id: str
    dataset_id: str
    version_id: str
    timestamp: datetime
    status: ValidationStatus
    completeness_score: float  # % of non-null values
    consistency_score: float  # % of consistent records
    accuracy_score: float  # % of accurate data points
    uniqueness_score: float  # % of unique records
    timeliness_score: float  # Data freshness score
    overall_score: float
    issues: List[Dict[str, Any]]
    recommendations: List[str]


@dataclass
class DatasetStatistics:
    """Dataset statistical summary"""

    dataset_id: str
    version_id: str
    numeric_columns: Dict[str, Dict[str, float]]  # mean, std, min, max, etc.
    categorical_columns: Dict[str, Dict[str, Any]]  # unique values, mode, etc.
    missing_values: Dict[str, float]  # % missing per column
    correlations: Dict[str, float]
    outliers: Dict[str, List[Any]]
    distribution_tests: Dict[str, Dict[str, float]]


class TrainingDataService:
    """Service for managing training data with versioning and lineage"""

    def __init__(self):
        # Use ML_MODELS_PATH or UPLOAD_DIR as storage base
        self.storage_base = (
            Path(getattr(settings, "DATA_STORAGE_PATH", settings.ML_MODELS_PATH))
            / "training_data"
        )
        self.storage_base.mkdir(parents=True, exist_ok=True)
        self.redis_client: Optional[redis.Redis] = None
        self.version_prefix = "v"
        self.cache_ttl = 3600  # 1 hour

    async def initialize(self):
        """Initialize service connections"""
        self.redis_client = await redis.from_url(
            settings.REDIS_URL, encoding="utf-8", decode_responses=True
        )
        logger.info("Training data service initialized")

    async def create_dataset(
        self,
        db: AsyncSession,
        name: str,
        dataset_type: DatasetType,
        description: str,
        source_data: pd.DataFrame,
        created_by: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DatasetVersion:
        """Create a new training dataset with initial version"""
        dataset_id = self._generate_dataset_id(name, dataset_type)
        version_id = self._generate_version_id(dataset_id, "1.0.0")

        # Validate data
        validation = await self.validate_data(source_data, dataset_type)
        if validation.status == ValidationStatus.FAILED:
            raise ValueError(f"Data validation failed: {validation.issues}")

        # Calculate statistics
        statistics = await self.calculate_statistics(source_data)

        # Store dataset
        storage_path = await self._store_dataset(dataset_id, version_id, source_data)

        # Calculate checksum
        checksum = self._calculate_checksum(source_data)

        # Create version record
        version = DatasetVersion(
            version_id=version_id,
            dataset_id=dataset_id,
            version_number="1.0.0",
            created_at=datetime.utcnow(),
            created_by=created_by,
            parent_version=None,
            changes_description="Initial version",
            size_bytes=source_data.memory_usage(deep=True).sum(),
            row_count=len(source_data),
            column_count=len(source_data.columns),
            checksum=checksum,
            storage_path=str(storage_path),
            is_active=True,
            metadata=metadata or {},
        )

        # Store metadata in Redis
        await self._store_version_metadata(version)

        # Track lineage
        await self.track_lineage(
            dataset_id=dataset_id,
            source_datasets=[],
            transformations=[
                {"type": "initial_creation", "timestamp": datetime.utcnow().isoformat()}
            ],
        )

        # Store statistics
        await self._store_statistics(dataset_id, version_id, statistics)

        logger.info(f"Created dataset {dataset_id} version {version.version_number}")
        return version

    async def create_version(
        self,
        db: AsyncSession,
        dataset_id: str,
        parent_version_id: str,
        changes: pd.DataFrame,
        changes_description: str,
        created_by: str,
        transformations: Optional[List[Dict[str, Any]]] = None,
    ) -> DatasetVersion:
        """Create a new version of an existing dataset"""
        # Get parent version
        parent_version = await self.get_version(parent_version_id)
        if not parent_version:
            raise ValueError(f"Parent version {parent_version_id} not found")

        # Load parent data
        parent_data = await self.load_dataset(parent_version_id)

        # Apply changes (merge, append, or replace based on transformation type)
        new_data = await self._apply_changes(parent_data, changes, transformations)

        # Validate new data
        validation = await self.validate_data(
            new_data, DatasetType(parent_version.metadata.get("type"))
        )
        if validation.status == ValidationStatus.FAILED:
            raise ValueError(f"Data validation failed: {validation.issues}")

        # Generate new version number
        new_version_number = self._increment_version(parent_version.version_number)
        version_id = self._generate_version_id(dataset_id, new_version_number)

        # Store new version
        storage_path = await self._store_dataset(dataset_id, version_id, new_data)

        # Create version record
        version = DatasetVersion(
            version_id=version_id,
            dataset_id=dataset_id,
            version_number=new_version_number,
            created_at=datetime.utcnow(),
            created_by=created_by,
            parent_version=parent_version_id,
            changes_description=changes_description,
            size_bytes=new_data.memory_usage(deep=True).sum(),
            row_count=len(new_data),
            column_count=len(new_data.columns),
            checksum=self._calculate_checksum(new_data),
            storage_path=str(storage_path),
            is_active=True,
            metadata=parent_version.metadata.copy(),
        )

        # Store metadata
        await self._store_version_metadata(version)

        # Track lineage
        await self.track_lineage(
            dataset_id=dataset_id,
            source_datasets=[parent_version_id],
            transformations=transformations or [],
        )

        # Calculate and store statistics
        statistics = await self.calculate_statistics(new_data)
        await self._store_statistics(dataset_id, version_id, statistics)

        logger.info(f"Created dataset {dataset_id} version {version.version_number}")
        return version

    async def validate_data(
        self, data: pd.DataFrame, dataset_type: DatasetType
    ) -> DataValidation:
        """Validate training data quality"""
        validation_id = self._generate_id("validation")
        issues = []
        recommendations = []

        # Completeness check
        completeness_score = 1 - (
            data.isnull().sum().sum() / (len(data) * len(data.columns))
        )
        if completeness_score < 0.95:
            issues.append(
                {
                    "type": "completeness",
                    "severity": "medium",
                    "message": f"Data completeness is {completeness_score:.2%}",
                    "columns": data.columns[data.isnull().any()].tolist(),
                }
            )
            recommendations.append(
                "Consider imputing missing values or removing incomplete records"
            )

        # Consistency check (check for duplicates)
        duplicates = data.duplicated().sum()
        consistency_score = 1 - (duplicates / len(data))
        if duplicates > 0:
            issues.append(
                {
                    "type": "consistency",
                    "severity": "low",
                    "message": f"Found {duplicates} duplicate records",
                }
            )
            recommendations.append("Remove duplicate records to improve data quality")

        # Accuracy check (dataset-specific validation)
        accuracy_score = await self._validate_accuracy(data, dataset_type)
        if accuracy_score < 0.95:
            issues.append(
                {
                    "type": "accuracy",
                    "severity": "high",
                    "message": f"Data accuracy is {accuracy_score:.2%}",
                }
            )

        # Uniqueness check
        uniqueness_scores = []
        for col in data.select_dtypes(include=["object"]).columns:
            unique_ratio = data[col].nunique() / len(data)
            uniqueness_scores.append(unique_ratio)
        uniqueness_score = np.mean(uniqueness_scores) if uniqueness_scores else 1.0

        # Timeliness check (data freshness)
        timeliness_score = 1.0  # Default, can be customized based on timestamps
        if "timestamp" in data.columns or "created_at" in data.columns:
            time_col = "timestamp" if "timestamp" in data.columns else "created_at"
            latest_data = pd.to_datetime(data[time_col]).max()
            days_old = (datetime.utcnow() - latest_data).days
            timeliness_score = max(0, 1 - (days_old / 30))  # Decay over 30 days

        # Calculate overall score
        overall_score = np.mean(
            [
                completeness_score,
                consistency_score,
                accuracy_score,
                uniqueness_score,
                timeliness_score,
            ]
        )

        # Determine status
        if overall_score >= 0.95:
            status = ValidationStatus.PASSED
        elif overall_score >= 0.80:
            status = ValidationStatus.PARTIAL
        else:
            status = ValidationStatus.FAILED

        validation = DataValidation(
            validation_id=validation_id,
            dataset_id="",  # Will be set by caller
            version_id="",  # Will be set by caller
            timestamp=datetime.utcnow(),
            status=status,
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            accuracy_score=accuracy_score,
            uniqueness_score=uniqueness_score,
            timeliness_score=timeliness_score,
            overall_score=overall_score,
            issues=issues,
            recommendations=recommendations,
        )

        return validation

    async def calculate_statistics(self, data: pd.DataFrame) -> DatasetStatistics:
        """Calculate comprehensive dataset statistics"""
        statistics = DatasetStatistics(
            dataset_id="",  # Will be set by caller
            version_id="",  # Will be set by caller
            numeric_columns={},
            categorical_columns={},
            missing_values={},
            correlations={},
            outliers={},
            distribution_tests={},
        )

        # Numeric column statistics
        for col in data.select_dtypes(include=[np.number]).columns:
            statistics.numeric_columns[col] = {
                "mean": float(data[col].mean()),
                "std": float(data[col].std()),
                "min": float(data[col].min()),
                "max": float(data[col].max()),
                "median": float(data[col].median()),
                "q25": float(data[col].quantile(0.25)),
                "q75": float(data[col].quantile(0.75)),
                "skewness": float(data[col].skew()),
                "kurtosis": float(data[col].kurtosis()),
            }

            # Detect outliers using IQR method
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = data[
                (data[col] < Q1 - 1.5 * IQR) | (data[col] > Q3 + 1.5 * IQR)
            ][col].tolist()
            if outliers:
                statistics.outliers[col] = outliers[:10]  # Store max 10 outliers

        # Categorical column statistics
        for col in data.select_dtypes(include=["object", "category"]).columns:
            value_counts = data[col].value_counts()
            statistics.categorical_columns[col] = {
                "unique_values": int(data[col].nunique()),
                "mode": str(data[col].mode()[0])
                if not data[col].mode().empty
                else None,
                "top_values": value_counts.head(5).to_dict(),
                "entropy": float(
                    -sum(
                        (value_counts / len(data))
                        * np.log2(value_counts / len(data) + 1e-10)
                    )
                ),
            }

        # Missing value statistics
        for col in data.columns:
            missing_pct = (data[col].isnull().sum() / len(data)) * 100
            statistics.missing_values[col] = round(missing_pct, 2)

        # Correlation matrix for numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            corr_matrix = numeric_data.corr()
            # Store only significant correlations (>0.5 or <-0.5)
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.5:
                        key = f"{corr_matrix.columns[i]}_{corr_matrix.columns[j]}"
                        statistics.correlations[key] = round(float(corr_value), 3)

        return statistics

    async def track_lineage(
        self,
        dataset_id: str,
        source_datasets: List[str],
        transformations: List[Dict[str, Any]],
        processing_time: Optional[float] = None,
    ) -> DataLineage:
        """Track data lineage and transformations"""
        lineage_id = self._generate_id("lineage")

        lineage = DataLineage(
            lineage_id=lineage_id,
            dataset_id=dataset_id,
            source_datasets=source_datasets,
            transformations=transformations,
            created_at=datetime.utcnow(),
            processing_time_seconds=processing_time or 0,
            quality_metrics={},
            dependencies=[],
        )

        # Store lineage in Redis
        key = f"lineage:{dataset_id}:{lineage_id}"
        await self.redis_client.setex(
            key,
            86400 * 30,  # 30 days retention
            json.dumps(asdict(lineage), default=str),
        )

        # Update lineage graph
        await self._update_lineage_graph(dataset_id, source_datasets)

        return lineage

    async def load_dataset(
        self, version_id: str, sample_size: Optional[int] = None
    ) -> pd.DataFrame:
        """Load a specific version of a dataset"""
        # Get version metadata
        version = await self.get_version(version_id)
        if not version:
            raise ValueError(f"Version {version_id} not found")

        # Load from storage
        storage_path = Path(version.storage_path)
        if not storage_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {storage_path}")

        # Load compressed pickle file
        with gzip.open(storage_path, "rb") as f:
            data = pickle.load(f)

        # Apply sampling if requested
        if sample_size and sample_size < len(data):
            data = data.sample(n=sample_size, random_state=42)

        return data

    async def get_version(self, version_id: str) -> Optional[DatasetVersion]:
        """Get version metadata"""
        key = f"version:{version_id}"
        data = await self.redis_client.get(key)

        if data:
            version_dict = json.loads(data)
            # Convert datetime strings back to datetime objects
            version_dict["created_at"] = datetime.fromisoformat(
                version_dict["created_at"]
            )
            return DatasetVersion(**version_dict)

        return None

    async def list_versions(
        self, dataset_id: str, include_inactive: bool = False
    ) -> List[DatasetVersion]:
        """List all versions of a dataset"""
        pattern = f"version:{dataset_id}:*"
        versions = []

        cursor = 0
        while True:
            cursor, keys = await self.redis_client.scan(
                cursor=cursor, match=pattern, count=100
            )

            for key in keys:
                version_data = await self.redis_client.get(key)
                if version_data:
                    version_dict = json.loads(version_data)
                    version_dict["created_at"] = datetime.fromisoformat(
                        version_dict["created_at"]
                    )
                    version = DatasetVersion(**version_dict)

                    if include_inactive or version.is_active:
                        versions.append(version)

            if cursor == 0:
                break

        # Sort by version number
        versions.sort(key=lambda v: v.version_number, reverse=True)
        return versions

    async def get_lineage_graph(
        self, dataset_id: str, depth: int = 3
    ) -> Dict[str, Any]:
        """Get lineage graph for a dataset"""
        graph = {"nodes": [], "edges": [], "metadata": {}}

        visited = set()
        queue = [(dataset_id, 0)]

        while queue and depth > 0:
            current_id, current_depth = queue.pop(0)

            if current_id in visited or current_depth >= depth:
                continue

            visited.add(current_id)

            # Add node
            graph["nodes"].append(
                {"id": current_id, "depth": current_depth, "type": "dataset"}
            )

            # Get lineage data
            pattern = f"lineage:{current_id}:*"
            cursor = 0
            while True:
                cursor, keys = await self.redis_client.scan(
                    cursor=cursor, match=pattern, count=10
                )

                for key in keys:
                    lineage_data = await self.redis_client.get(key)
                    if lineage_data:
                        lineage = json.loads(lineage_data)

                        # Add edges
                        for source in lineage.get("source_datasets", []):
                            graph["edges"].append(
                                {
                                    "from": source,
                                    "to": current_id,
                                    "transformations": lineage.get(
                                        "transformations", []
                                    ),
                                }
                            )

                            # Add to queue for traversal
                            if source not in visited:
                                queue.append((source, current_depth + 1))

                if cursor == 0:
                    break

        return graph

    async def archive_version(self, version_id: str, reason: str) -> bool:
        """Archive a dataset version"""
        version = await self.get_version(version_id)
        if not version:
            return False

        version.is_active = False
        version.metadata["archived_at"] = datetime.utcnow().isoformat()
        version.metadata["archive_reason"] = reason

        await self._store_version_metadata(version)

        # Move physical file to archive storage
        storage_path = Path(version.storage_path)
        archive_path = self.storage_base / "archive" / storage_path.name
        archive_path.parent.mkdir(parents=True, exist_ok=True)

        if storage_path.exists():
            shutil.move(str(storage_path), str(archive_path))
            version.storage_path = str(archive_path)
            await self._store_version_metadata(version)

        logger.info(f"Archived version {version_id}")
        return True

    async def _store_dataset(
        self, dataset_id: str, version_id: str, data: pd.DataFrame
    ) -> Path:
        """Store dataset to disk"""
        # Create storage directory
        dataset_dir = self.storage_base / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Save as compressed pickle
        file_path = dataset_dir / f"{version_id}.pkl.gz"
        with gzip.open(file_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        return file_path

    async def _store_version_metadata(self, version: DatasetVersion):
        """Store version metadata in Redis"""
        key = f"version:{version.version_id}"
        await self.redis_client.setex(
            key,
            86400 * 90,  # 90 days retention
            json.dumps(asdict(version), default=str),
        )

        # Also store in dataset index
        dataset_key = f"dataset:{version.dataset_id}:versions"
        await self.redis_client.sadd(dataset_key, version.version_id)

    async def _store_statistics(
        self, dataset_id: str, version_id: str, statistics: DatasetStatistics
    ):
        """Store dataset statistics"""
        statistics.dataset_id = dataset_id
        statistics.version_id = version_id

        key = f"statistics:{version_id}"
        await self.redis_client.setex(
            key,
            86400 * 30,  # 30 days retention
            json.dumps(asdict(statistics), default=str),
        )

    async def _apply_changes(
        self,
        parent_data: pd.DataFrame,
        changes: pd.DataFrame,
        transformations: Optional[List[Dict[str, Any]]],
    ) -> pd.DataFrame:
        """Apply changes to create new version"""
        if not transformations:
            # Default: append changes to parent data
            return pd.concat([parent_data, changes], ignore_index=True)

        new_data = parent_data.copy()

        for transformation in transformations:
            transform_type = transformation.get("type")

            if transform_type == "append":
                new_data = pd.concat([new_data, changes], ignore_index=True)
            elif transform_type == "merge":
                merge_on = transformation.get("merge_on", [])
                new_data = pd.merge(
                    new_data,
                    changes,
                    on=merge_on,
                    how=transformation.get("how", "inner"),
                )
            elif transform_type == "replace":
                new_data = changes
            elif transform_type == "filter":
                condition = transformation.get("condition")
                if condition:
                    new_data = new_data.query(condition)
            elif transform_type == "aggregate":
                group_by = transformation.get("group_by", [])
                agg_func = transformation.get("agg_func", {})
                new_data = new_data.groupby(group_by).agg(agg_func).reset_index()

        return new_data

    async def _validate_accuracy(
        self, data: pd.DataFrame, dataset_type: DatasetType
    ) -> float:
        """Validate data accuracy based on dataset type"""
        accuracy_score = 1.0

        if dataset_type == DatasetType.CONTENT_GENERATION:
            # Check for required columns
            required_cols = ["prompt", "response", "quality_score"]
            missing_cols = set(required_cols) - set(data.columns)
            if missing_cols:
                accuracy_score -= 0.2 * len(missing_cols)

        elif dataset_type == DatasetType.TREND_PREDICTION:
            # Check for time series data
            if "timestamp" in data.columns:
                # Check for temporal ordering
                timestamps = pd.to_datetime(data["timestamp"])
                if not timestamps.is_monotonic_increasing:
                    accuracy_score -= 0.1

        elif dataset_type == DatasetType.QUALITY_SCORING:
            # Check score ranges
            if "score" in data.columns:
                invalid_scores = ((data["score"] < 0) | (data["score"] > 100)).sum()
                accuracy_score -= (invalid_scores / len(data)) * 0.5

        return max(0, accuracy_score)

    async def _update_lineage_graph(self, dataset_id: str, source_datasets: List[str]):
        """Update the lineage graph connections"""
        graph_key = f"lineage_graph:{dataset_id}"

        # Store forward connections (sources -> dataset)
        for source in source_datasets:
            await self.redis_client.sadd(f"lineage_forward:{source}", dataset_id)

        # Store backward connections (dataset -> sources)
        await self.redis_client.sadd(graph_key, *source_datasets)

    def _generate_dataset_id(self, name: str, dataset_type: DatasetType) -> str:
        """Generate unique dataset ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        name_slug = name.lower().replace(" ", "_")[:20]
        return f"{dataset_type.value}_{name_slug}_{timestamp}"

    def _generate_version_id(self, dataset_id: str, version_number: str) -> str:
        """Generate unique version ID"""
        return f"{dataset_id}:{self.version_prefix}{version_number.replace('.', '_')}"

    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID with prefix"""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        return f"{prefix}_{timestamp}"

    def _increment_version(self, version_number: str) -> str:
        """Increment version number (semantic versioning)"""
        parts = version_number.split(".")
        if len(parts) == 3:
            # Increment patch version
            parts[2] = str(int(parts[2]) + 1)
        elif len(parts) == 2:
            parts.append("1")
        else:
            parts = ["1", "0", "0"]

        return ".".join(parts)

    def _calculate_checksum(self, data: pd.DataFrame) -> str:
        """Calculate checksum for data integrity"""
        # Convert DataFrame to bytes and calculate SHA256
        data_bytes = pickle.dumps(data)
        return hashlib.sha256(data_bytes).hexdigest()


# Singleton instance
training_data_service = TrainingDataService()
