"""
Feature Store Implementation
P1 Task: [DATA] Centralized feature management for ML pipelines
"""
import asyncio
import json
import hashlib
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from enum import Enum
import logging
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update

logger = logging.getLogger(__name__)

class FeatureType(Enum):
    """Types of features"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    EMBEDDING = "embedding"
    TIME_SERIES = "time_series"
    IMAGE = "image"

class FeatureSource(Enum):
    """Feature data sources"""
    DATABASE = "database"
    API = "api"
    STREAM = "stream"
    BATCH = "batch"
    COMPUTED = "computed"

@dataclass
class FeatureDefinition:
    """Feature definition and metadata"""
    name: str
    type: FeatureType
    source: FeatureSource
    description: str
    version: str
    schema: Dict[str, Any]
    transformation: Optional[str] = None
    dependencies: List[str] = None
    ttl_seconds: int = 3600
    is_online: bool = True
    is_offline: bool = True
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()

@dataclass
class FeatureVector:
    """Feature vector for ML models"""
    entity_id: str
    features: Dict[str, Any]
    timestamp: datetime
    version: str
    metadata: Dict[str, Any] = None

class FeatureStore:
    """
    Centralized feature store for ML pipelines
    Provides online and offline feature serving
    """
    
    def __init__(self, redis_client: redis.Redis = None, db_session: AsyncSession = None):
        self.redis_client = redis_client  # Online store
        self.db_session = db_session      # Offline store
        self.features: Dict[str, FeatureDefinition] = {}
        self.feature_cache: Dict[str, Any] = {}
        self.transformations: Dict[str, callable] = {}
        self._lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize feature store"""
        try:
            # Load feature definitions
            await self._load_feature_definitions()
            
            # Register default transformations
            self._register_default_transformations()
            
            logger.info(f"Feature store initialized with {len(self.features)} features")
            
        except Exception as e:
            logger.error(f"Failed to initialize feature store: {e}")
            raise
    
    async def register_feature(self, feature: FeatureDefinition) -> bool:
        """Register a new feature definition"""
        async with self._lock:
            try:
                # Validate feature
                if not self._validate_feature(feature):
                    return False
                
                # Store feature definition
                self.features[feature.name] = feature
                
                # Persist to database
                if self.db_session:
                    await self._persist_feature_definition(feature)
                
                logger.info(f"Feature '{feature.name}' registered successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to register feature: {e}")
                return False
    
    async def get_online_features(
        self,
        entity_ids: List[str],
        feature_names: List[str],
        version: Optional[str] = None
    ) -> List[FeatureVector]:
        """
        Get features from online store (low latency)
        Used for real-time inference
        """
        feature_vectors = []
        
        for entity_id in entity_ids:
            features = {}
            
            for feature_name in feature_names:
                if feature_name not in self.features:
                    logger.warning(f"Feature '{feature_name}' not found")
                    continue
                
                # Get from cache first
                cache_key = self._get_cache_key(entity_id, feature_name, version)
                
                if self.redis_client:
                    value = await self.redis_client.get(cache_key)
                    if value:
                        features[feature_name] = json.loads(value)
                    else:
                        # Compute feature if not in cache
                        value = await self._compute_feature(entity_id, feature_name)
                        if value is not None:
                            features[feature_name] = value
                            # Cache the computed value
                            await self._cache_feature(entity_id, feature_name, value, version)
            
            feature_vectors.append(FeatureVector(
                entity_id=entity_id,
                features=features,
                timestamp=datetime.utcnow(),
                version=version or "latest"
            ))
        
        return feature_vectors
    
    async def get_offline_features(
        self,
        entity_ids: List[str],
        feature_names: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Get features from offline store (batch processing)
        Used for training and batch inference
        """
        if not self.db_session:
            raise ValueError("Database session required for offline features")
        
        # Build SQL query for feature retrieval
        query = f"""
        SELECT 
            entity_id,
            {', '.join(feature_names)},
            timestamp
        FROM features
        WHERE entity_id IN :entity_ids
            AND timestamp BETWEEN :start_date AND :end_date
        ORDER BY entity_id, timestamp
        """
        
        result = await self.db_session.execute(
            query,
            {
                "entity_ids": entity_ids,
                "start_date": start_date,
                "end_date": end_date
            }
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(result.fetchall())
        
        # Apply transformations if needed
        for feature_name in feature_names:
            if feature_name in self.features:
                feature_def = self.features[feature_name]
                if feature_def.transformation:
                    df[feature_name] = await self._apply_transformation(
                        df[feature_name],
                        feature_def.transformation
                    )
        
        return df
    
    async def write_features(
        self,
        entity_id: str,
        features: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ):
        """Write features to both online and offline stores"""
        timestamp = timestamp or datetime.utcnow()
        
        async with self._lock:
            # Write to online store (Redis)
            if self.redis_client:
                for feature_name, value in features.items():
                    if feature_name in self.features:
                        feature_def = self.features[feature_name]
                        if feature_def.is_online:
                            cache_key = self._get_cache_key(entity_id, feature_name)
                            await self.redis_client.setex(
                                cache_key,
                                feature_def.ttl_seconds,
                                json.dumps(value, default=str)
                            )
            
            # Write to offline store (Database)
            if self.db_session:
                await self._write_to_offline_store(entity_id, features, timestamp)
    
    async def _compute_feature(self, entity_id: str, feature_name: str) -> Any:
        """Compute feature value dynamically"""
        if feature_name not in self.features:
            return None
        
        feature_def = self.features[feature_name]
        
        # Check dependencies
        if feature_def.dependencies:
            dep_values = {}
            for dep in feature_def.dependencies:
                dep_value = await self.get_online_features(
                    [entity_id],
                    [dep]
                )
                if dep_value and dep_value[0].features:
                    dep_values[dep] = dep_value[0].features.get(dep)
        
        # Apply computation based on source
        if feature_def.source == FeatureSource.COMPUTED:
            # Use transformation function
            if feature_def.transformation in self.transformations:
                transform_func = self.transformations[feature_def.transformation]
                return transform_func(entity_id, dep_values if feature_def.dependencies else None)
        
        elif feature_def.source == FeatureSource.DATABASE:
            # Fetch from database
            return await self._fetch_from_database(entity_id, feature_name)
        
        elif feature_def.source == FeatureSource.API:
            # Fetch from external API
            return await self._fetch_from_api(entity_id, feature_name)
        
        return None
    
    async def _fetch_from_database(self, entity_id: str, feature_name: str) -> Any:
        """Fetch feature from database"""
        if not self.db_session:
            return None
        
        query = f"SELECT {feature_name} FROM features WHERE entity_id = :entity_id ORDER BY timestamp DESC LIMIT 1"
        result = await self.db_session.execute(query, {"entity_id": entity_id})
        row = result.fetchone()
        
        return row[0] if row else None
    
    async def _fetch_from_api(self, entity_id: str, feature_name: str) -> Any:
        """Fetch feature from external API"""
        # Placeholder for API integration
        # In production, this would call actual APIs
        return None
    
    async def _cache_feature(
        self,
        entity_id: str,
        feature_name: str,
        value: Any,
        version: Optional[str] = None
    ):
        """Cache feature value"""
        if not self.redis_client or feature_name not in self.features:
            return
        
        feature_def = self.features[feature_name]
        cache_key = self._get_cache_key(entity_id, feature_name, version)
        
        await self.redis_client.setex(
            cache_key,
            feature_def.ttl_seconds,
            json.dumps(value, default=str)
        )
    
    def _get_cache_key(
        self,
        entity_id: str,
        feature_name: str,
        version: Optional[str] = None
    ) -> str:
        """Generate cache key for feature"""
        version = version or "latest"
        return f"feature:{feature_name}:{version}:{entity_id}"
    
    def _validate_feature(self, feature: FeatureDefinition) -> bool:
        """Validate feature definition"""
        # Check for circular dependencies
        if feature.dependencies:
            visited = set()
            
            def has_cycle(feat_name: str, path: set) -> bool:
                if feat_name in path:
                    return True
                if feat_name in visited:
                    return False
                
                visited.add(feat_name)
                path.add(feat_name)
                
                if feat_name in self.features:
                    for dep in self.features[feat_name].dependencies or []:
                        if has_cycle(dep, path.copy()):
                            return True
                
                return False
            
            if has_cycle(feature.name, set()):
                logger.error(f"Circular dependency detected for feature '{feature.name}'")
                return False
        
        return True
    
    async def _write_to_offline_store(
        self,
        entity_id: str,
        features: Dict[str, Any],
        timestamp: datetime
    ):
        """Write features to offline store (database)"""
        if not self.db_session:
            return
        
        # Prepare data for insertion
        data = {
            "entity_id": entity_id,
            "timestamp": timestamp,
            **features
        }
        
        # Insert or update
        query = """
        INSERT INTO features (entity_id, timestamp, features_json)
        VALUES (:entity_id, :timestamp, :features_json)
        ON CONFLICT (entity_id, timestamp) 
        DO UPDATE SET features_json = :features_json
        """
        
        await self.db_session.execute(
            query,
            {
                "entity_id": entity_id,
                "timestamp": timestamp,
                "features_json": json.dumps(features, default=str)
            }
        )
        await self.db_session.commit()
    
    def _register_default_transformations(self):
        """Register default feature transformations"""
        
        # Normalization
        self.transformations["normalize"] = lambda x, _: (x - np.mean(x)) / np.std(x)
        
        # One-hot encoding
        self.transformations["one_hot"] = lambda x, _: pd.get_dummies(x)
        
        # Log transformation
        self.transformations["log"] = lambda x, _: np.log1p(x)
        
        # Binning
        self.transformations["bin"] = lambda x, bins: pd.cut(x, bins=bins or 10)
        
        # Moving average
        self.transformations["moving_avg"] = lambda x, window: pd.Series(x).rolling(window=window or 7).mean()
    
    async def _apply_transformation(self, data: Any, transformation: str) -> Any:
        """Apply transformation to data"""
        if transformation in self.transformations:
            return self.transformations[transformation](data, None)
        return data
    
    async def _load_feature_definitions(self):
        """Load feature definitions from storage"""
        # In production, this would load from database or config file
        # For now, define some default features
        
        default_features = [
            FeatureDefinition(
                name="video_count",
                type=FeatureType.NUMERIC,
                source=FeatureSource.DATABASE,
                description="Total number of videos for channel",
                version="1.0",
                schema={"type": "integer", "min": 0}
            ),
            FeatureDefinition(
                name="avg_view_count",
                type=FeatureType.NUMERIC,
                source=FeatureSource.COMPUTED,
                description="Average view count per video",
                version="1.0",
                schema={"type": "float", "min": 0},
                dependencies=["total_views", "video_count"]
            ),
            FeatureDefinition(
                name="engagement_rate",
                type=FeatureType.NUMERIC,
                source=FeatureSource.COMPUTED,
                description="Engagement rate for channel",
                version="1.0",
                schema={"type": "float", "min": 0, "max": 100}
            ),
            FeatureDefinition(
                name="channel_category",
                type=FeatureType.CATEGORICAL,
                source=FeatureSource.DATABASE,
                description="Channel category/niche",
                version="1.0",
                schema={"type": "string", "enum": ["tech", "gaming", "education", "entertainment"]}
            ),
            FeatureDefinition(
                name="trending_score",
                type=FeatureType.NUMERIC,
                source=FeatureSource.COMPUTED,
                description="Current trending score",
                version="1.0",
                schema={"type": "float", "min": 0, "max": 1},
                ttl_seconds=1800  # 30 minutes cache
            )
        ]
        
        for feature in default_features:
            self.features[feature.name] = feature
    
    async def _persist_feature_definition(self, feature: FeatureDefinition):
        """Persist feature definition to database"""
        if not self.db_session:
            return
        
        query = """
        INSERT INTO feature_definitions (name, definition_json)
        VALUES (:name, :definition_json)
        ON CONFLICT (name) 
        DO UPDATE SET definition_json = :definition_json, updated_at = NOW()
        """
        
        await self.db_session.execute(
            query,
            {
                "name": feature.name,
                "definition_json": json.dumps(asdict(feature), default=str)
            }
        )
        await self.db_session.commit()
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """Get feature store statistics"""
        return {
            "total_features": len(self.features),
            "online_features": sum(1 for f in self.features.values() if f.is_online),
            "offline_features": sum(1 for f in self.features.values() if f.is_offline),
            "computed_features": sum(1 for f in self.features.values() if f.source == FeatureSource.COMPUTED),
            "features_by_type": {
                ft.value: sum(1 for f in self.features.values() if f.type == ft)
                for ft in FeatureType
            },
            "cache_size": len(self.feature_cache)
        }

# Global feature store instance
feature_store = FeatureStore()