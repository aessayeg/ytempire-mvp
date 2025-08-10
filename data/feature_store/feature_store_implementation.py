"""
Feature Store Implementation
Centralized feature management for ML pipelines
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import aiohttp
import redis.asyncio as redis
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, JSON, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pyarrow as pa
import pyarrow.parquet as pq
import feast
from feast import Entity, Feature, FeatureView, FileSource, ValueType
from feast.data_source import DataSource
import logging
import json
import hashlib
from prometheus_client import Histogram, Counter, Gauge

# Metrics
feature_computation_time = Histogram('feature_computation_duration', 'Feature computation time', ['feature_name'])
feature_cache_hits = Counter('feature_cache_hits', 'Feature cache hits', ['feature_name'])
feature_cache_misses = Counter('feature_cache_misses', 'Feature cache misses', ['feature_name'])
feature_store_size = Gauge('feature_store_size_gb', 'Feature store size in GB')

logger = logging.getLogger(__name__)

Base = declarative_base()

@dataclass
class FeatureDefinition:
    """Feature definition"""
    name: str
    description: str
    dtype: str
    computation: Optional[callable] = None
    dependencies: List[str] = field(default_factory=list)
    ttl_seconds: int = 3600
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)

@dataclass
class FeatureSet:
    """Collection of related features"""
    name: str
    features: List[FeatureDefinition]
    entity: str
    description: str
    version: str = "1.0"

class FeatureMetadata(Base):
    """Feature metadata table"""
    __tablename__ = 'feature_metadata'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, index=True)
    description = Column(String(1000))
    dtype = Column(String(50))
    feature_set = Column(String(255), index=True)
    version = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    schema = Column(JSON)
    statistics = Column(JSON)
    tags = Column(JSON)
    
    __table_args__ = (
        Index('idx_feature_set_version', 'feature_set', 'version'),
    )

class FeatureStore:
    """Centralized feature store for ML pipelines"""
    
    def __init__(self,
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 db_url: str = None,
                 storage_path: str = '/data/feature_store'):
        
        self.redis_client = None
        self.redis_host = redis_host
        self.redis_port = redis_port
        
        # Database for metadata
        if db_url:
            self.engine = create_engine(db_url)
            Base.metadata.create_all(self.engine)
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
        else:
            self.engine = None
            self.session = None
        
        self.storage_path = storage_path
        
        # Feature definitions
        self.feature_sets = {}
        self._register_feature_sets()
        
        # Feature computation cache
        self.computation_cache = {}
    
    async def initialize(self):
        """Initialize async components"""
        self.redis_client = await redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            decode_responses=False  # For binary data
        )
    
    def _register_feature_sets(self):
        """Register all feature sets"""
        
        # User features
        user_features = FeatureSet(
            name="user_features",
            entity="user_id",
            description="User-level features for personalization",
            features=[
                FeatureDefinition(
                    name="user_watch_time_7d",
                    description="Total watch time in last 7 days",
                    dtype="float",
                    computation=self._compute_user_watch_time,
                    ttl_seconds=3600
                ),
                FeatureDefinition(
                    name="user_video_count_30d",
                    description="Videos created in last 30 days",
                    dtype="int",
                    computation=self._compute_user_video_count,
                    ttl_seconds=3600
                ),
                FeatureDefinition(
                    name="user_engagement_score",
                    description="User engagement score",
                    dtype="float",
                    computation=self._compute_user_engagement,
                    dependencies=["user_watch_time_7d", "user_video_count_30d"],
                    ttl_seconds=1800
                ),
                FeatureDefinition(
                    name="user_revenue_ltv",
                    description="User lifetime value",
                    dtype="float",
                    computation=self._compute_user_ltv,
                    ttl_seconds=86400
                ),
                FeatureDefinition(
                    name="user_churn_probability",
                    description="Probability of user churn",
                    dtype="float",
                    computation=self._compute_churn_probability,
                    ttl_seconds=3600
                )
            ]
        )
        self.feature_sets["user_features"] = user_features
        
        # Video features
        video_features = FeatureSet(
            name="video_features",
            entity="video_id",
            description="Video-level features for content optimization",
            features=[
                FeatureDefinition(
                    name="video_view_count",
                    description="Total video views",
                    dtype="int",
                    computation=self._compute_video_views,
                    ttl_seconds=300
                ),
                FeatureDefinition(
                    name="video_avg_watch_duration",
                    description="Average watch duration",
                    dtype="float",
                    computation=self._compute_avg_watch_duration,
                    ttl_seconds=600
                ),
                FeatureDefinition(
                    name="video_engagement_rate",
                    description="Video engagement rate",
                    dtype="float",
                    computation=self._compute_video_engagement,
                    ttl_seconds=600
                ),
                FeatureDefinition(
                    name="video_quality_score",
                    description="Content quality score",
                    dtype="float",
                    computation=self._compute_quality_score,
                    ttl_seconds=3600
                ),
                FeatureDefinition(
                    name="video_virality_score",
                    description="Virality potential score",
                    dtype="float",
                    computation=self._compute_virality_score,
                    dependencies=["video_view_count", "video_engagement_rate"],
                    ttl_seconds=1800
                )
            ]
        )
        self.feature_sets["video_features"] = video_features
        
        # Channel features
        channel_features = FeatureSet(
            name="channel_features",
            entity="channel_id",
            description="Channel-level features for channel optimization",
            features=[
                FeatureDefinition(
                    name="channel_subscriber_count",
                    description="Channel subscriber count",
                    dtype="int",
                    computation=self._compute_subscriber_count,
                    ttl_seconds=3600
                ),
                FeatureDefinition(
                    name="channel_growth_rate",
                    description="Channel growth rate",
                    dtype="float",
                    computation=self._compute_growth_rate,
                    ttl_seconds=3600
                ),
                FeatureDefinition(
                    name="channel_avg_views",
                    description="Average views per video",
                    dtype="float",
                    computation=self._compute_channel_avg_views,
                    ttl_seconds=3600
                ),
                FeatureDefinition(
                    name="channel_consistency_score",
                    description="Upload consistency score",
                    dtype="float",
                    computation=self._compute_consistency_score,
                    ttl_seconds=86400
                )
            ]
        )
        self.feature_sets["channel_features"] = channel_features
        
        # Real-time features
        realtime_features = FeatureSet(
            name="realtime_features",
            entity="session_id",
            description="Real-time features for online serving",
            features=[
                FeatureDefinition(
                    name="current_trending_score",
                    description="Current trending score",
                    dtype="float",
                    computation=self._compute_trending_score,
                    ttl_seconds=60
                ),
                FeatureDefinition(
                    name="current_platform_load",
                    description="Current platform load",
                    dtype="float",
                    computation=self._compute_platform_load,
                    ttl_seconds=30
                ),
                FeatureDefinition(
                    name="realtime_recommendation_context",
                    description="Real-time recommendation context",
                    dtype="json",
                    computation=self._compute_recommendation_context,
                    ttl_seconds=60
                )
            ]
        )
        self.feature_sets["realtime_features"] = realtime_features
    
    async def get_features(self,
                          entity_id: str,
                          feature_names: List[str],
                          feature_set: str = None) -> Dict[str, Any]:
        """
        Get features for an entity
        
        Args:
            entity_id: Entity identifier
            feature_names: List of feature names to retrieve
            feature_set: Optional feature set name
            
        Returns:
            Dictionary of feature values
        """
        features = {}
        
        for feature_name in feature_names:
            # Check cache first
            cached_value = await self._get_cached_feature(entity_id, feature_name)
            if cached_value is not None:
                features[feature_name] = cached_value
                feature_cache_hits.labels(feature_name=feature_name).inc()
                continue
            
            feature_cache_misses.labels(feature_name=feature_name).inc()
            
            # Find and compute feature
            feature_def = self._find_feature_definition(feature_name, feature_set)
            if feature_def:
                # Check dependencies
                if feature_def.dependencies:
                    dep_features = await self.get_features(
                        entity_id,
                        feature_def.dependencies,
                        feature_set
                    )
                    features.update(dep_features)
                
                # Compute feature
                value = await self._compute_feature(entity_id, feature_def, features)
                features[feature_name] = value
                
                # Cache feature
                await self._cache_feature(entity_id, feature_name, value, feature_def.ttl_seconds)
            else:
                logger.warning(f"Feature {feature_name} not found")
                features[feature_name] = None
        
        return features
    
    async def get_feature_vector(self,
                                entity_id: str,
                                feature_set: str) -> np.ndarray:
        """
        Get feature vector for an entity
        
        Args:
            entity_id: Entity identifier
            feature_set: Feature set name
            
        Returns:
            Feature vector as numpy array
        """
        if feature_set not in self.feature_sets:
            raise ValueError(f"Feature set {feature_set} not found")
        
        fs = self.feature_sets[feature_set]
        feature_names = [f.name for f in fs.features]
        
        features = await self.get_features(entity_id, feature_names, feature_set)
        
        # Convert to vector
        vector = []
        for feature_name in feature_names:
            value = features.get(feature_name, 0)
            if isinstance(value, (list, dict)):
                # Flatten complex features
                value = hashlib.md5(json.dumps(value).encode()).hexdigest()[:8]
                value = int(value, 16) / (16**8)  # Normalize to [0, 1]
            vector.append(float(value))
        
        return np.array(vector)
    
    async def get_training_dataset(self,
                                  entity_ids: List[str],
                                  feature_set: str,
                                  labels: Optional[List[Any]] = None) -> pd.DataFrame:
        """
        Get training dataset for multiple entities
        
        Args:
            entity_ids: List of entity identifiers
            feature_set: Feature set name
            labels: Optional labels for supervised learning
            
        Returns:
            DataFrame with features and optional labels
        """
        if feature_set not in self.feature_sets:
            raise ValueError(f"Feature set {feature_set} not found")
        
        fs = self.feature_sets[feature_set]
        feature_names = [f.name for f in fs.features]
        
        # Collect features for all entities
        data = []
        for i, entity_id in enumerate(entity_ids):
            features = await self.get_features(entity_id, feature_names, feature_set)
            features[fs.entity] = entity_id
            
            if labels:
                features['label'] = labels[i]
            
            data.append(features)
        
        return pd.DataFrame(data)
    
    async def store_features(self,
                            entity_id: str,
                            features: Dict[str, Any],
                            feature_set: str = None):
        """
        Store computed features
        
        Args:
            entity_id: Entity identifier
            features: Dictionary of feature values
            feature_set: Optional feature set name
        """
        for feature_name, value in features.items():
            # Find feature definition
            feature_def = self._find_feature_definition(feature_name, feature_set)
            if feature_def:
                # Cache feature
                await self._cache_feature(
                    entity_id,
                    feature_name,
                    value,
                    feature_def.ttl_seconds
                )
                
                # Store in persistent storage
                await self._persist_feature(entity_id, feature_name, value, feature_set)
                
                # Update metadata
                await self._update_feature_metadata(feature_name, value, feature_set)
    
    async def _compute_feature(self,
                              entity_id: str,
                              feature_def: FeatureDefinition,
                              context: Dict = None) -> Any:
        """Compute feature value"""
        start_time = datetime.utcnow()
        
        try:
            if feature_def.computation:
                # Call computation function
                if asyncio.iscoroutinefunction(feature_def.computation):
                    value = await feature_def.computation(entity_id, context)
                else:
                    value = feature_def.computation(entity_id, context)
            else:
                # Fetch from storage
                value = await self._fetch_from_storage(entity_id, feature_def.name)
            
            # Record computation time
            computation_time = (datetime.utcnow() - start_time).total_seconds()
            feature_computation_time.labels(feature_name=feature_def.name).observe(computation_time)
            
            return value
            
        except Exception as e:
            logger.error(f"Error computing feature {feature_def.name}: {e}")
            return None
    
    def _find_feature_definition(self,
                                feature_name: str,
                                feature_set: str = None) -> Optional[FeatureDefinition]:
        """Find feature definition"""
        if feature_set:
            fs = self.feature_sets.get(feature_set)
            if fs:
                for feature in fs.features:
                    if feature.name == feature_name:
                        return feature
        else:
            # Search all feature sets
            for fs in self.feature_sets.values():
                for feature in fs.features:
                    if feature.name == feature_name:
                        return feature
        return None
    
    async def _get_cached_feature(self,
                                 entity_id: str,
                                 feature_name: str) -> Optional[Any]:
        """Get cached feature value"""
        if not self.redis_client:
            return None
        
        key = f"feature:{feature_name}:{entity_id}"
        
        try:
            cached = await self.redis_client.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.error(f"Error getting cached feature: {e}")
        
        return None
    
    async def _cache_feature(self,
                           entity_id: str,
                           feature_name: str,
                           value: Any,
                           ttl: int):
        """Cache feature value"""
        if not self.redis_client:
            return
        
        key = f"feature:{feature_name}:{entity_id}"
        
        try:
            serialized = json.dumps(value)
            await self.redis_client.setex(key, ttl, serialized)
        except Exception as e:
            logger.error(f"Error caching feature: {e}")
    
    async def _persist_feature(self,
                              entity_id: str,
                              feature_name: str,
                              value: Any,
                              feature_set: str):
        """Persist feature to storage"""
        # Save to parquet file
        timestamp = datetime.utcnow()
        data = {
            'entity_id': entity_id,
            'feature_name': feature_name,
            'value': value,
            'feature_set': feature_set,
            'timestamp': timestamp
        }
        
        df = pd.DataFrame([data])
        
        # Partition by date and feature set
        date_str = timestamp.strftime('%Y%m%d')
        file_path = f"{self.storage_path}/{feature_set}/{date_str}/{feature_name}.parquet"
        
        # Append to existing file or create new
        import os
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if os.path.exists(file_path):
            existing_df = pd.read_parquet(file_path)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        df.to_parquet(file_path, engine='pyarrow', compression='snappy')
    
    async def _fetch_from_storage(self,
                                 entity_id: str,
                                 feature_name: str) -> Optional[Any]:
        """Fetch feature from storage"""
        # Look for most recent value in parquet files
        import os
        import glob
        
        pattern = f"{self.storage_path}/*/*/*/{feature_name}.parquet"
        files = glob.glob(pattern)
        
        if not files:
            return None
        
        # Read most recent file
        files.sort(reverse=True)
        df = pd.read_parquet(files[0])
        
        # Filter for entity
        entity_df = df[df['entity_id'] == entity_id]
        
        if entity_df.empty:
            return None
        
        # Return most recent value
        return entity_df.iloc[-1]['value']
    
    async def _update_feature_metadata(self,
                                      feature_name: str,
                                      value: Any,
                                      feature_set: str):
        """Update feature metadata"""
        if not self.session:
            return
        
        try:
            # Find or create metadata record
            metadata = self.session.query(FeatureMetadata).filter_by(name=feature_name).first()
            
            if not metadata:
                metadata = FeatureMetadata(
                    name=feature_name,
                    feature_set=feature_set,
                    dtype=type(value).__name__
                )
                self.session.add(metadata)
            
            # Update statistics
            if metadata.statistics is None:
                metadata.statistics = {}
            
            stats = metadata.statistics
            stats['last_updated'] = datetime.utcnow().isoformat()
            stats['sample_value'] = str(value)[:100]
            
            metadata.statistics = stats
            metadata.updated_at = datetime.utcnow()
            
            self.session.commit()
            
        except Exception as e:
            logger.error(f"Error updating feature metadata: {e}")
            self.session.rollback()
    
    # Feature computation functions
    def _compute_user_watch_time(self, user_id: str, context: Dict = None) -> float:
        """Compute user watch time in last 7 days"""
        # Placeholder - would query actual data
        return np.random.gamma(2, 100)
    
    def _compute_user_video_count(self, user_id: str, context: Dict = None) -> int:
        """Compute user video count in last 30 days"""
        # Placeholder
        return np.random.poisson(10)
    
    def _compute_user_engagement(self, user_id: str, context: Dict = None) -> float:
        """Compute user engagement score"""
        watch_time = context.get('user_watch_time_7d', 100)
        video_count = context.get('user_video_count_30d', 5)
        
        # Simple engagement formula
        engagement = (watch_time / 100) * 0.6 + (video_count / 10) * 0.4
        return min(1.0, engagement)
    
    def _compute_user_ltv(self, user_id: str, context: Dict = None) -> float:
        """Compute user lifetime value"""
        # Placeholder
        return np.random.gamma(3, 100)
    
    def _compute_churn_probability(self, user_id: str, context: Dict = None) -> float:
        """Compute churn probability"""
        # Placeholder
        return np.random.beta(2, 8)
    
    def _compute_video_views(self, video_id: str, context: Dict = None) -> int:
        """Compute video view count"""
        # Placeholder
        return np.random.lognormal(8, 2)
    
    def _compute_avg_watch_duration(self, video_id: str, context: Dict = None) -> float:
        """Compute average watch duration"""
        # Placeholder
        return np.random.gamma(2, 30)
    
    def _compute_video_engagement(self, video_id: str, context: Dict = None) -> float:
        """Compute video engagement rate"""
        # Placeholder
        return np.random.beta(2, 5)
    
    def _compute_quality_score(self, video_id: str, context: Dict = None) -> float:
        """Compute video quality score"""
        # Placeholder
        return np.random.beta(8, 2)
    
    def _compute_virality_score(self, video_id: str, context: Dict = None) -> float:
        """Compute virality score"""
        views = context.get('video_view_count', 1000)
        engagement = context.get('video_engagement_rate', 0.1)
        
        # Virality formula
        virality = np.log1p(views) * engagement * 0.1
        return min(1.0, virality)
    
    def _compute_subscriber_count(self, channel_id: str, context: Dict = None) -> int:
        """Compute channel subscriber count"""
        # Placeholder
        return np.random.lognormal(10, 2)
    
    def _compute_growth_rate(self, channel_id: str, context: Dict = None) -> float:
        """Compute channel growth rate"""
        # Placeholder
        return np.random.normal(0.1, 0.05)
    
    def _compute_channel_avg_views(self, channel_id: str, context: Dict = None) -> float:
        """Compute channel average views"""
        # Placeholder
        return np.random.lognormal(9, 1.5)
    
    def _compute_consistency_score(self, channel_id: str, context: Dict = None) -> float:
        """Compute upload consistency score"""
        # Placeholder
        return np.random.beta(5, 2)
    
    def _compute_trending_score(self, session_id: str, context: Dict = None) -> float:
        """Compute current trending score"""
        # Placeholder
        return np.random.beta(2, 3)
    
    def _compute_platform_load(self, session_id: str, context: Dict = None) -> float:
        """Compute current platform load"""
        # Placeholder
        return np.random.beta(3, 7)
    
    def _compute_recommendation_context(self, session_id: str, context: Dict = None) -> Dict:
        """Compute recommendation context"""
        return {
            'time_of_day': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
            'trending_topics': ['tech', 'gaming', 'music'],
            'user_mood': np.random.choice(['happy', 'neutral', 'excited'])
        }

# Feature serving layer
class FeatureServer:
    """Online feature serving for real-time inference"""
    
    def __init__(self, feature_store: FeatureStore):
        self.feature_store = feature_store
        self.cache = {}
    
    async def get_online_features(self,
                                 entity_ids: List[str],
                                 feature_sets: List[str]) -> pd.DataFrame:
        """
        Get online features for real-time serving
        
        Args:
            entity_ids: List of entity identifiers
            feature_sets: List of feature set names
            
        Returns:
            DataFrame with features
        """
        all_features = []
        
        for entity_id in entity_ids:
            entity_features = {'entity_id': entity_id}
            
            for feature_set in feature_sets:
                if feature_set in self.feature_store.feature_sets:
                    fs = self.feature_store.feature_sets[feature_set]
                    feature_names = [f.name for f in fs.features]
                    
                    features = await self.feature_store.get_features(
                        entity_id,
                        feature_names,
                        feature_set
                    )
                    
                    entity_features.update(features)
            
            all_features.append(entity_features)
        
        return pd.DataFrame(all_features)
    
    async def get_batch_features(self,
                                entity_ids: List[str],
                                feature_sets: List[str],
                                point_in_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get batch features for training
        
        Args:
            entity_ids: List of entity identifiers
            feature_sets: List of feature set names
            point_in_time: Optional point-in-time for historical features
            
        Returns:
            DataFrame with features
        """
        # For batch processing, we can parallelize feature computation
        tasks = []
        
        for entity_id in entity_ids:
            for feature_set in feature_sets:
                task = self._get_entity_features_async(entity_id, feature_set)
                tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Combine results
        features_by_entity = {}
        for result in results:
            if result:
                entity_id = result['entity_id']
                if entity_id not in features_by_entity:
                    features_by_entity[entity_id] = {}
                features_by_entity[entity_id].update(result)
        
        # Convert to DataFrame
        data = []
        for entity_id, features in features_by_entity.items():
            features['entity_id'] = entity_id
            data.append(features)
        
        return pd.DataFrame(data)
    
    async def _get_entity_features_async(self,
                                        entity_id: str,
                                        feature_set: str) -> Dict:
        """Get features for single entity asynchronously"""
        try:
            if feature_set in self.feature_store.feature_sets:
                fs = self.feature_store.feature_sets[feature_set]
                feature_names = [f.name for f in fs.features]
                
                features = await self.feature_store.get_features(
                    entity_id,
                    feature_names,
                    feature_set
                )
                
                features['entity_id'] = entity_id
                return features
        except Exception as e:
            logger.error(f"Error getting features for {entity_id}: {e}")
            return None

# Example usage
async def main():
    # Initialize feature store
    feature_store = FeatureStore(
        redis_host='localhost',
        redis_port=6379,
        db_url='postgresql://user:pass@localhost/features',
        storage_path='/data/feature_store'
    )
    
    await feature_store.initialize()
    
    # Get features for a user
    user_features = await feature_store.get_features(
        entity_id='user_123',
        feature_names=['user_engagement_score', 'user_churn_probability'],
        feature_set='user_features'
    )
    
    print(f"User features: {user_features}")
    
    # Get feature vector
    feature_vector = await feature_store.get_feature_vector(
        entity_id='user_123',
        feature_set='user_features'
    )
    
    print(f"Feature vector shape: {feature_vector.shape}")
    
    # Get training dataset
    training_data = await feature_store.get_training_dataset(
        entity_ids=['user_1', 'user_2', 'user_3'],
        feature_set='user_features',
        labels=[0, 1, 0]
    )
    
    print(f"Training data shape: {training_data.shape}")
    
    # Initialize feature server
    feature_server = FeatureServer(feature_store)
    
    # Get online features
    online_features = await feature_server.get_online_features(
        entity_ids=['user_123', 'user_456'],
        feature_sets=['user_features', 'realtime_features']
    )
    
    print(f"Online features: {online_features}")

if __name__ == "__main__":
    asyncio.run(main())