"""
Automated Feature Engineering Pipeline
Complete automation for feature extraction, transformation, and serving
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
import asyncio
import logging
from pathlib import Path
import json
import pickle
from dataclasses import dataclass, field
from enum import Enum
import schedule
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import ray
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import feast
from feast import FeatureStore, Entity, FeatureView, Field
from feast.types import Float32, Int64, String
import redis
import psycopg2
from sqlalchemy import create_engine
import pyarrow.parquet as pq
import dask.dataframe as dd

logger = logging.getLogger(__name__)

class FeatureType(Enum):
    """Types of features"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    TEMPORAL = "temporal"
    EMBEDDING = "embedding"
    GRAPH = "graph"

class TransformationType(Enum):
    """Types of transformations"""
    SCALING = "scaling"
    ENCODING = "encoding"
    AGGREGATION = "aggregation"
    WINDOW = "window"
    INTERACTION = "interaction"
    POLYNOMIAL = "polynomial"

@dataclass
class FeatureDefinition:
    """Feature definition with metadata"""
    name: str
    type: FeatureType
    source: str
    transformation: Optional[TransformationType] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    refresh_frequency: str = "daily"
    description: str = ""

@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    name: str
    features: List[FeatureDefinition]
    source_connections: Dict[str, str]
    output_store: str
    batch_size: int = 10000
    parallel_workers: int = 4
    enable_caching: bool = True
    enable_monitoring: bool = True

class AutomatedFeaturePipeline:
    """Fully automated feature engineering pipeline"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.feature_store = FeatureStore(repo_path=".")
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.transformers: Dict[str, Any] = {}
        self.feature_cache: Dict[str, pd.DataFrame] = {}
        self.monitoring_metrics: Dict[str, List] = {}
        self._initialize_pipeline()
        
    def _initialize_pipeline(self):
        """Initialize pipeline components"""
        # Initialize Ray for distributed processing
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # Setup transformers
        self._setup_transformers()
        
        # Setup monitoring
        self._setup_monitoring()
        
        # Load existing feature definitions
        self._load_feature_definitions()
        
    def _setup_transformers(self):
        """Setup feature transformers"""
        self.transformers = {
            "scaler": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
            "tfidf": TfidfVectorizer(max_features=1000),
            "count": CountVectorizer(max_features=1000),
            "pca": PCA(n_components=50),
            "svd": TruncatedSVD(n_components=50)
        }
        
    def _setup_monitoring(self):
        """Setup pipeline monitoring"""
        self.monitoring_metrics = {
            "processing_time": [],
            "features_generated": [],
            "errors": [],
            "cache_hits": [],
            "data_quality": []
        }
        
    async def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete feature pipeline"""
        start_time = time.time()
        results = {}
        
        try:
            # Step 1: Extract raw data
            logger.info("Extracting raw data...")
            raw_data = await self._extract_data()
            
            # Step 2: Validate data quality
            logger.info("Validating data quality...")
            validation_results = await self._validate_data(raw_data)
            
            if not validation_results["passed"]:
                raise ValueError(f"Data validation failed: {validation_results['errors']}")
            
            # Step 3: Generate features in parallel
            logger.info("Generating features...")
            features = await self._generate_features_parallel(raw_data)
            
            # Step 4: Apply transformations
            logger.info("Applying transformations...")
            transformed_features = await self._apply_transformations(features)
            
            # Step 5: Feature selection
            logger.info("Selecting features...")
            selected_features = await self._select_features(transformed_features)
            
            # Step 6: Store features
            logger.info("Storing features...")
            await self._store_features(selected_features)
            
            # Step 7: Update monitoring metrics
            processing_time = time.time() - start_time
            self._update_monitoring(processing_time, len(selected_features.columns))
            
            results = {
                "status": "success",
                "features_generated": len(selected_features.columns),
                "rows_processed": len(selected_features),
                "processing_time": processing_time,
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow()
            }
            self.monitoring_metrics["errors"].append({
                "error": str(e),
                "timestamp": datetime.utcnow()
            })
            
        return results
        
    async def _extract_data(self) -> pd.DataFrame:
        """Extract data from multiple sources"""
        data_frames = []
        
        for source_name, connection_string in self.config.source_connections.items():
            if source_name == "postgres":
                df = await self._extract_from_postgres(connection_string)
            elif source_name == "parquet":
                df = await self._extract_from_parquet(connection_string)
            elif source_name == "api":
                df = await self._extract_from_api(connection_string)
            else:
                logger.warning(f"Unknown source: {source_name}")
                continue
                
            data_frames.append(df)
            
        # Merge all data sources
        if len(data_frames) > 1:
            merged_data = pd.concat(data_frames, axis=1)
        else:
            merged_data = data_frames[0] if data_frames else pd.DataFrame()
            
        return merged_data
        
    async def _extract_from_postgres(self, connection_string: str) -> pd.DataFrame:
        """Extract data from PostgreSQL"""
        engine = create_engine(connection_string)
        query = """
            SELECT * FROM video_metrics 
            WHERE created_at >= NOW() - INTERVAL '7 days'
        """
        return pd.read_sql(query, engine)
        
    async def _extract_from_parquet(self, file_path: str) -> pd.DataFrame:
        """Extract data from Parquet files"""
        return pq.read_table(file_path).to_pandas()
        
    async def _extract_from_api(self, api_url: str) -> pd.DataFrame:
        """Extract data from API"""
        # Implementation for API extraction
        pass
        
    async def _validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality"""
        validation_results = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "stats": {}
        }
        
        # Check for missing values
        missing_ratio = data.isnull().sum() / len(data)
        high_missing = missing_ratio[missing_ratio > 0.5]
        
        if not high_missing.empty:
            validation_results["warnings"].append(
                f"High missing values in columns: {high_missing.index.tolist()}"
            )
            
        # Check for duplicates
        duplicates = data.duplicated().sum()
        if duplicates > 0:
            validation_results["warnings"].append(
                f"Found {duplicates} duplicate rows"
            )
            
        # Check data types
        for col in data.columns:
            if data[col].dtype == 'object':
                try:
                    pd.to_numeric(data[col])
                    validation_results["warnings"].append(
                        f"Column {col} might be numeric but stored as object"
                    )
                except:
                    pass
                    
        # Calculate statistics
        validation_results["stats"] = {
            "rows": len(data),
            "columns": len(data.columns),
            "memory_usage": data.memory_usage(deep=True).sum() / 1024**2,  # MB
            "missing_ratio": missing_ratio.mean()
        }
        
        return validation_results
        
    async def _generate_features_parallel(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate features in parallel using Ray"""
        
        @ray.remote
        def generate_feature_batch(feature_def: FeatureDefinition, data_chunk: pd.DataFrame):
            """Generate features for a batch of data"""
            if feature_def.type == FeatureType.NUMERIC:
                return generate_numeric_features(data_chunk, feature_def)
            elif feature_def.type == FeatureType.CATEGORICAL:
                return generate_categorical_features(data_chunk, feature_def)
            elif feature_def.type == FeatureType.TEXT:
                return generate_text_features(data_chunk, feature_def)
            elif feature_def.type == FeatureType.TEMPORAL:
                return generate_temporal_features(data_chunk, feature_def)
            else:
                return pd.DataFrame()
                
        # Split data into chunks for parallel processing
        chunk_size = self.config.batch_size
        chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
        
        # Process features in parallel
        all_features = []
        
        for feature_def in self.config.features:
            # Check cache first
            cache_key = f"feature:{feature_def.name}"
            cached_feature = self._get_cached_feature(cache_key)
            
            if cached_feature is not None and self.config.enable_caching:
                all_features.append(cached_feature)
                self.monitoring_metrics["cache_hits"].append(feature_def.name)
                continue
                
            # Generate feature in parallel
            feature_futures = [
                generate_feature_batch.remote(feature_def, chunk)
                for chunk in chunks
            ]
            
            feature_results = ray.get(feature_futures)
            feature_df = pd.concat(feature_results, ignore_index=True)
            
            # Cache the feature
            if self.config.enable_caching:
                self._cache_feature(cache_key, feature_df)
                
            all_features.append(feature_df)
            
        # Combine all features
        return pd.concat(all_features, axis=1)
        
    async def _apply_transformations(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply transformations to features"""
        transformed = features.copy()
        
        for feature_def in self.config.features:
            if feature_def.transformation == TransformationType.SCALING:
                scaler_type = feature_def.parameters.get("scaler", "standard")
                scaler = self.transformers.get(f"{scaler_type}_scaler", self.transformers["scaler"])
                
                if feature_def.name in transformed.columns:
                    transformed[f"{feature_def.name}_scaled"] = scaler.fit_transform(
                        transformed[[feature_def.name]]
                    )
                    
            elif feature_def.transformation == TransformationType.ENCODING:
                if feature_def.name in transformed.columns:
                    encoded = pd.get_dummies(transformed[feature_def.name], prefix=feature_def.name)
                    transformed = pd.concat([transformed, encoded], axis=1)
                    
            elif feature_def.transformation == TransformationType.AGGREGATION:
                window = feature_def.parameters.get("window", 7)
                agg_funcs = feature_def.parameters.get("functions", ["mean", "std"])
                
                for func in agg_funcs:
                    if feature_def.name in transformed.columns:
                        transformed[f"{feature_def.name}_{func}_{window}d"] = (
                            transformed[feature_def.name].rolling(window).agg(func)
                        )
                        
        return transformed
        
    async def _select_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Select most important features"""
        # Separate numeric and categorical columns
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) <= 50:
            return features  # Keep all if not too many
            
        # Use mutual information for feature selection
        selector = SelectKBest(score_func=mutual_info_regression, k=50)
        
        # Create a dummy target (you would use actual target in production)
        target = np.random.randn(len(features))
        
        selected_features = selector.fit_transform(features[numeric_cols], target)
        selected_columns = numeric_cols[selector.get_support()]
        
        # Keep non-numeric columns and selected numeric columns
        final_columns = list(features.select_dtypes(exclude=[np.number]).columns) + list(selected_columns)
        
        return features[final_columns]
        
    async def _store_features(self, features: pd.DataFrame):
        """Store features in feature store"""
        timestamp = datetime.utcnow()
        
        # Store in Feast feature store
        feature_df = features.copy()
        feature_df["event_timestamp"] = timestamp
        feature_df["created"] = timestamp
        
        # Write to offline store (Parquet)
        output_path = f"data/features/{timestamp.strftime('%Y%m%d_%H%M%S')}.parquet"
        feature_df.to_parquet(output_path)
        
        # Write to online store (Redis)
        for idx, row in feature_df.iterrows():
            key = f"features:{idx}:{timestamp.timestamp()}"
            self.redis_client.hset(key, mapping=row.to_dict())
            self.redis_client.expire(key, 86400)  # 24 hour TTL
            
        logger.info(f"Stored {len(features)} features to {output_path}")
        
    def _get_cached_feature(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get cached feature if available"""
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return pickle.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        return None
        
    def _cache_feature(self, cache_key: str, feature_df: pd.DataFrame):
        """Cache feature for reuse"""
        try:
            serialized = pickle.dumps(feature_df)
            self.redis_client.setex(cache_key, 3600, serialized)  # 1 hour cache
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
            
    def _update_monitoring(self, processing_time: float, features_count: int):
        """Update monitoring metrics"""
        self.monitoring_metrics["processing_time"].append(processing_time)
        self.monitoring_metrics["features_generated"].append(features_count)
        
        # Keep only recent metrics
        for key in self.monitoring_metrics:
            if len(self.monitoring_metrics[key]) > 1000:
                self.monitoring_metrics[key] = self.monitoring_metrics[key][-1000:]
                
    def schedule_pipeline(self):
        """Schedule pipeline to run automatically"""
        
        def run_async_pipeline():
            """Wrapper to run async pipeline in sync context"""
            asyncio.run(self.run_pipeline())
            
        # Schedule based on feature refresh frequencies
        schedule.every().day.at("00:00").do(run_async_pipeline)
        schedule.every().hour.do(self._refresh_realtime_features)
        schedule.every(5).minutes.do(self._update_cache)
        
        logger.info("Pipeline scheduled successfully")
        
        # Run scheduler in background thread
        def scheduler_loop():
            while True:
                schedule.run_pending()
                time.sleep(60)
                
        import threading
        scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        scheduler_thread.start()
        
    def _refresh_realtime_features(self):
        """Refresh real-time features"""
        # Implementation for real-time feature refresh
        pass
        
    def _update_cache(self):
        """Update feature cache"""
        # Implementation for cache update
        pass
        
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get monitoring dashboard data"""
        return {
            "avg_processing_time": np.mean(self.monitoring_metrics["processing_time"]) if self.monitoring_metrics["processing_time"] else 0,
            "total_features_generated": sum(self.monitoring_metrics["features_generated"]),
            "error_count": len(self.monitoring_metrics["errors"]),
            "cache_hit_rate": len(self.monitoring_metrics["cache_hits"]) / max(len(self.monitoring_metrics["features_generated"]), 1),
            "recent_errors": self.monitoring_metrics["errors"][-5:],
            "health_status": "healthy" if len(self.monitoring_metrics["errors"]) == 0 else "degraded"
        }


def generate_numeric_features(data: pd.DataFrame, feature_def: FeatureDefinition) -> pd.DataFrame:
    """Generate numeric features"""
    result = pd.DataFrame()
    col = feature_def.source
    
    if col in data.columns:
        # Basic statistics
        result[f"{col}_log"] = np.log1p(data[col].clip(lower=0))
        result[f"{col}_sqrt"] = np.sqrt(data[col].clip(lower=0))
        result[f"{col}_square"] = data[col] ** 2
        
        # Binning
        result[f"{col}_bin"] = pd.qcut(data[col], q=5, labels=False, duplicates='drop')
        
        # Outlier indicators
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        result[f"{col}_is_outlier"] = ((data[col] < q1 - 1.5 * iqr) | (data[col] > q3 + 1.5 * iqr)).astype(int)
        
    return result


def generate_categorical_features(data: pd.DataFrame, feature_def: FeatureDefinition) -> pd.DataFrame:
    """Generate categorical features"""
    result = pd.DataFrame()
    col = feature_def.source
    
    if col in data.columns:
        # Frequency encoding
        freq_map = data[col].value_counts().to_dict()
        result[f"{col}_frequency"] = data[col].map(freq_map)
        
        # Target encoding (would need actual target in production)
        # This is a placeholder
        result[f"{col}_target_enc"] = data[col].factorize()[0]
        
    return result


def generate_text_features(data: pd.DataFrame, feature_def: FeatureDefinition) -> pd.DataFrame:
    """Generate text features"""
    result = pd.DataFrame()
    col = feature_def.source
    
    if col in data.columns:
        # Text length
        result[f"{col}_length"] = data[col].str.len()
        
        # Word count
        result[f"{col}_word_count"] = data[col].str.split().str.len()
        
        # Special character count
        result[f"{col}_special_chars"] = data[col].str.count('[^a-zA-Z0-9 ]')
        
        # Sentiment (placeholder - would use actual sentiment analysis)
        result[f"{col}_sentiment"] = np.random.uniform(-1, 1, len(data))
        
    return result


def generate_temporal_features(data: pd.DataFrame, feature_def: FeatureDefinition) -> pd.DataFrame:
    """Generate temporal features"""
    result = pd.DataFrame()
    col = feature_def.source
    
    if col in data.columns:
        # Convert to datetime if needed
        dt = pd.to_datetime(data[col], errors='coerce')
        
        # Extract components
        result[f"{col}_year"] = dt.dt.year
        result[f"{col}_month"] = dt.dt.month
        result[f"{col}_day"] = dt.dt.day
        result[f"{col}_dayofweek"] = dt.dt.dayofweek
        result[f"{col}_hour"] = dt.dt.hour
        result[f"{col}_quarter"] = dt.dt.quarter
        
        # Cyclical encoding
        result[f"{col}_month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
        result[f"{col}_month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
        
        # Is weekend
        result[f"{col}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        
    return result


# Initialize pipeline
if __name__ == "__main__":
    # Example configuration
    config = PipelineConfig(
        name="ytempire_features",
        features=[
            FeatureDefinition(
                name="views",
                type=FeatureType.NUMERIC,
                source="view_count",
                transformation=TransformationType.SCALING
            ),
            FeatureDefinition(
                name="category",
                type=FeatureType.CATEGORICAL,
                source="video_category",
                transformation=TransformationType.ENCODING
            ),
            FeatureDefinition(
                name="title",
                type=FeatureType.TEXT,
                source="video_title"
            ),
            FeatureDefinition(
                name="publish_date",
                type=FeatureType.TEMPORAL,
                source="published_at"
            )
        ],
        source_connections={
            "postgres": "postgresql://user:pass@localhost/ytempire",
            "parquet": "data/raw/*.parquet"
        },
        output_store="feast",
        batch_size=10000,
        parallel_workers=4
    )
    
    pipeline = AutomatedFeaturePipeline(config)
    
    # Schedule automatic runs
    pipeline.schedule_pipeline()
    
    # Run once immediately
    asyncio.run(pipeline.run_pipeline())