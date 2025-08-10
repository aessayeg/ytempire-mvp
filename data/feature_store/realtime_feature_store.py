"""
Real-time Feature Store Implementation
Manages features for ML models with real-time serving
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
import asyncio
import redis.asyncio as redis
from feast import FeatureStore, Entity, Feature, FeatureView, FileSource, Field
from feast.types import Float32, Float64, Int64, String, Bool
from feast.data_source import DataSource
from feast.infra.offline_stores.file_source import FileSource
from feast.infra.online_stores.redis import RedisOnlineStore
import pyarrow.parquet as pq
import hashlib
from prometheus_client import Counter, Histogram, Gauge
import pickle

logger = logging.getLogger(__name__)

@dataclass
class FeatureStoreConfig:
    """Configuration for feature store"""
    project_name: str = "ytempire"
    provider: str = "local"
    registry_path: str = "/data/feature_store/registry.db"
    online_store_type: str = "redis"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 2
    offline_store_path: str = "/data/feature_store/offline"
    feature_ttl_seconds: int = 86400  # 24 hours
    batch_size: int = 1000
    enable_monitoring: bool = True

@dataclass
class FeatureDefinition:
    """Definition of a feature"""
    name: str
    dtype: str
    description: str
    tags: Dict[str, str] = field(default_factory=dict)
    compute_function: Optional[callable] = None
    dependencies: List[str] = field(default_factory=list)

class RealtimeFeatureStore:
    """Real-time feature store for ML features"""
    
    def __init__(self, config: FeatureStoreConfig = None):
        self.config = config or FeatureStoreConfig()
        self._init_feast()
        self._init_redis()
        self._init_monitoring()
        self._init_feature_definitions()
        
    def _init_feast(self):
        """Initialize Feast feature store"""
        # Create Feast configuration
        feast_config = {
            "project": self.config.project_name,
            "provider": self.config.provider,
            "registry": self.config.registry_path,
            "online_store": {
                "type": "redis",
                "connection_string": f"redis://{self.config.redis_host}:{self.config.redis_port}/{self.config.redis_db}"
            },
            "offline_store": {
                "type": "file"
            }
        }
        
        # Save Feast config
        config_path = "/tmp/feast_config.yaml"
        with open(config_path, "w") as f:
            import yaml
            yaml.dump(feast_config, f)
            
        # Initialize Feast store
        self.feast_store = FeatureStore(repo_path=config_path)
        
    async def _init_redis(self):
        """Initialize Redis connection for real-time features"""
        self.redis_client = await redis.Redis(
            host=self.config.redis_host,
            port=self.config.redis_port,
            db=self.config.redis_db,
            decode_responses=False
        )
        
    def _init_monitoring(self):
        """Initialize monitoring metrics"""
        if self.config.enable_monitoring:
            self.feature_compute_counter = Counter(
                'feature_store_compute_total',
                'Total feature computations',
                ['feature_name']
            )
            self.feature_latency = Histogram(
                'feature_store_latency_seconds',
                'Feature computation latency',
                ['feature_name']
            )
            self.feature_cache_hits = Counter(
                'feature_store_cache_hits_total',
                'Cache hits for features',
                ['feature_name']
            )
            self.feature_cache_misses = Counter(
                'feature_store_cache_misses_total',
                'Cache misses for features',
                ['feature_name']
            )
            
    def _init_feature_definitions(self):
        """Initialize feature definitions"""
        self.features = {
            # Video features
            "video_view_velocity": FeatureDefinition(
                name="video_view_velocity",
                dtype="float32",
                description="View growth rate per hour",
                compute_function=self._compute_view_velocity
            ),
            "video_engagement_rate": FeatureDefinition(
                name="video_engagement_rate",
                dtype="float32",
                description="Engagement rate (likes + comments) / views",
                compute_function=self._compute_engagement_rate
            ),
            "video_virality_score": FeatureDefinition(
                name="video_virality_score",
                dtype="float32",
                description="Virality potential score",
                compute_function=self._compute_virality_score,
                dependencies=["video_view_velocity", "video_engagement_rate"]
            ),
            
            # Channel features
            "channel_avg_views": FeatureDefinition(
                name="channel_avg_views",
                dtype="float32",
                description="Average views per video",
                compute_function=self._compute_channel_avg_views
            ),
            "channel_upload_frequency": FeatureDefinition(
                name="channel_upload_frequency",
                dtype="float32",
                description="Videos uploaded per week",
                compute_function=self._compute_upload_frequency
            ),
            "channel_growth_rate": FeatureDefinition(
                name="channel_growth_rate",
                dtype="float32",
                description="Subscriber growth rate",
                compute_function=self._compute_channel_growth
            ),
            
            # Trend features
            "topic_trending_score": FeatureDefinition(
                name="topic_trending_score",
                dtype="float32",
                description="Trending score for topic",
                compute_function=self._compute_trending_score
            ),
            "topic_saturation": FeatureDefinition(
                name="topic_saturation",
                dtype="float32",
                description="Market saturation for topic",
                compute_function=self._compute_topic_saturation
            ),
            
            # Cost features
            "video_roi": FeatureDefinition(
                name="video_roi",
                dtype="float32",
                description="Return on investment per video",
                compute_function=self._compute_video_roi
            ),
            "channel_profitability": FeatureDefinition(
                name="channel_profitability",
                dtype="float32",
                description="Channel profitability score",
                compute_function=self._compute_channel_profitability
            )
        }
        
    async def compute_features(
        self,
        entity_id: str,
        feature_names: List[str],
        input_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Compute requested features for an entity"""
        results = {}
        
        for feature_name in feature_names:
            if feature_name not in self.features:
                logger.warning(f"Unknown feature: {feature_name}")
                continue
                
            # Check cache first
            cached_value = await self._get_cached_feature(entity_id, feature_name)
            if cached_value is not None:
                results[feature_name] = cached_value
                if self.config.enable_monitoring:
                    self.feature_cache_hits.labels(feature_name=feature_name).inc()
                continue
                
            if self.config.enable_monitoring:
                self.feature_cache_misses.labels(feature_name=feature_name).inc()
                
            # Compute feature
            feature_def = self.features[feature_name]
            
            # Check dependencies
            if feature_def.dependencies:
                dep_features = await self.compute_features(
                    entity_id,
                    feature_def.dependencies,
                    input_data
                )
                input_data = {**(input_data or {}), **dep_features}
                
            # Compute feature value
            if feature_def.compute_function:
                import time
                start_time = time.time()
                
                value = await feature_def.compute_function(entity_id, input_data)
                
                if self.config.enable_monitoring:
                    self.feature_latency.labels(feature_name=feature_name).observe(
                        time.time() - start_time
                    )
                    self.feature_compute_counter.labels(feature_name=feature_name).inc()
                    
                # Cache the result
                await self._cache_feature(entity_id, feature_name, value)
                results[feature_name] = value
            else:
                # Fetch from offline store
                value = await self._fetch_offline_feature(entity_id, feature_name)
                results[feature_name] = value
                
        return results
        
    async def get_feature_vector(
        self,
        entity_id: str,
        feature_set: str = "default"
    ) -> np.ndarray:
        """Get feature vector for ML model input"""
        # Define feature sets
        feature_sets = {
            "default": [
                "video_view_velocity",
                "video_engagement_rate",
                "channel_avg_views",
                "topic_trending_score"
            ],
            "advanced": [
                "video_view_velocity",
                "video_engagement_rate",
                "video_virality_score",
                "channel_avg_views",
                "channel_upload_frequency",
                "channel_growth_rate",
                "topic_trending_score",
                "topic_saturation"
            ],
            "cost": [
                "video_roi",
                "channel_profitability"
            ]
        }
        
        feature_names = feature_sets.get(feature_set, feature_sets["default"])
        features = await self.compute_features(entity_id, feature_names)
        
        # Convert to numpy array
        vector = np.array([features.get(f, 0.0) for f in feature_names])
        return vector
        
    async def store_features(
        self,
        entity_id: str,
        features: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ):
        """Store computed features"""
        timestamp = timestamp or datetime.utcnow()
        
        # Store in Redis for online serving
        for feature_name, value in features.items():
            key = f"feature:{entity_id}:{feature_name}"
            
            # Serialize value
            serialized = pickle.dumps({
                "value": value,
                "timestamp": timestamp.isoformat()
            })
            
            await self.redis_client.setex(
                key,
                self.config.feature_ttl_seconds,
                serialized
            )
            
        # Store in offline store (Parquet)
        await self._store_offline_features(entity_id, features, timestamp)
        
    async def _get_cached_feature(
        self,
        entity_id: str,
        feature_name: str
    ) -> Optional[Any]:
        """Get cached feature value"""
        key = f"feature:{entity_id}:{feature_name}"
        cached = await self.redis_client.get(key)
        
        if cached:
            data = pickle.loads(cached)
            return data["value"]
        return None
        
    async def _cache_feature(
        self,
        entity_id: str,
        feature_name: str,
        value: Any
    ):
        """Cache computed feature"""
        key = f"feature:{entity_id}:{feature_name}"
        data = {
            "value": value,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.redis_client.setex(
            key,
            self.config.feature_ttl_seconds,
            pickle.dumps(data)
        )
        
    async def _store_offline_features(
        self,
        entity_id: str,
        features: Dict[str, Any],
        timestamp: datetime
    ):
        """Store features in offline store"""
        # Prepare data
        data = {
            "entity_id": entity_id,
            "timestamp": timestamp,
            **features
        }
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Determine partition
        date_str = timestamp.strftime("%Y%m%d")
        path = f"{self.config.offline_store_path}/{date_str}/features.parquet"
        
        # Append to parquet file
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if os.path.exists(path):
            existing_df = pd.read_parquet(path)
            df = pd.concat([existing_df, df], ignore_index=True)
            
        df.to_parquet(path, index=False)
        
    async def _fetch_offline_feature(
        self,
        entity_id: str,
        feature_name: str
    ) -> Optional[Any]:
        """Fetch feature from offline store"""
        # Look for most recent feature value
        today = datetime.utcnow()
        for days_back in range(7):  # Look back up to 7 days
            date = today - timedelta(days=days_back)
            date_str = date.strftime("%Y%m%d")
            path = f"{self.config.offline_store_path}/{date_str}/features.parquet"
            
            if os.path.exists(path):
                df = pd.read_parquet(path)
                entity_data = df[df["entity_id"] == entity_id]
                
                if not entity_data.empty and feature_name in entity_data.columns:
                    return entity_data[feature_name].iloc[-1]  # Return most recent
                    
        return None
        
    # Feature computation functions
    async def _compute_view_velocity(
        self,
        entity_id: str,
        input_data: Optional[Dict] = None
    ) -> float:
        """Compute view velocity for a video"""
        # This would fetch actual data from database
        # Simplified calculation for demonstration
        if input_data and "views" in input_data and "hours_since_upload" in input_data:
            views = input_data["views"]
            hours = max(input_data["hours_since_upload"], 1)
            return views / hours
        return 0.0
        
    async def _compute_engagement_rate(
        self,
        entity_id: str,
        input_data: Optional[Dict] = None
    ) -> float:
        """Compute engagement rate"""
        if input_data:
            views = input_data.get("views", 1)
            likes = input_data.get("likes", 0)
            comments = input_data.get("comments", 0)
            return (likes + comments) / max(views, 1)
        return 0.0
        
    async def _compute_virality_score(
        self,
        entity_id: str,
        input_data: Optional[Dict] = None
    ) -> float:
        """Compute virality score"""
        if input_data:
            velocity = input_data.get("video_view_velocity", 0)
            engagement = input_data.get("video_engagement_rate", 0)
            
            # Weighted combination
            score = (velocity * 0.6) + (engagement * 10000 * 0.4)
            return min(score / 100, 1.0)  # Normalize to 0-1
        return 0.0
        
    async def _compute_channel_avg_views(
        self,
        entity_id: str,
        input_data: Optional[Dict] = None
    ) -> float:
        """Compute average views per video for channel"""
        # Would fetch from database
        return input_data.get("channel_avg_views", 0.0) if input_data else 0.0
        
    async def _compute_upload_frequency(
        self,
        entity_id: str,
        input_data: Optional[Dict] = None
    ) -> float:
        """Compute upload frequency"""
        # Videos per week
        return input_data.get("videos_per_week", 0.0) if input_data else 0.0
        
    async def _compute_channel_growth(
        self,
        entity_id: str,
        input_data: Optional[Dict] = None
    ) -> float:
        """Compute channel growth rate"""
        if input_data:
            current_subs = input_data.get("current_subscribers", 0)
            prev_subs = input_data.get("prev_week_subscribers", 0)
            
            if prev_subs > 0:
                growth = (current_subs - prev_subs) / prev_subs
                return growth
        return 0.0
        
    async def _compute_trending_score(
        self,
        entity_id: str,
        input_data: Optional[Dict] = None
    ) -> float:
        """Compute trending score for topic"""
        # Would use search volume, growth rate, etc.
        return input_data.get("search_volume_growth", 0.0) if input_data else 0.0
        
    async def _compute_topic_saturation(
        self,
        entity_id: str,
        input_data: Optional[Dict] = None
    ) -> float:
        """Compute topic saturation"""
        # Competition level for topic
        return input_data.get("competition_level", 0.5) if input_data else 0.5
        
    async def _compute_video_roi(
        self,
        entity_id: str,
        input_data: Optional[Dict] = None
    ) -> float:
        """Compute video ROI"""
        if input_data:
            revenue = input_data.get("revenue", 0)
            cost = input_data.get("cost", 1)
            return (revenue - cost) / max(cost, 1)
        return 0.0
        
    async def _compute_channel_profitability(
        self,
        entity_id: str,
        input_data: Optional[Dict] = None
    ) -> float:
        """Compute channel profitability"""
        if input_data:
            total_revenue = input_data.get("total_revenue", 0)
            total_cost = input_data.get("total_cost", 1)
            return (total_revenue - total_cost) / max(total_cost, 1)
        return 0.0
        
    async def get_feature_importance(
        self,
        model_name: str = "default"
    ) -> Dict[str, float]:
        """Get feature importance scores"""
        # This would be computed from model analysis
        importance = {
            "video_view_velocity": 0.25,
            "video_engagement_rate": 0.20,
            "video_virality_score": 0.15,
            "channel_avg_views": 0.10,
            "channel_upload_frequency": 0.08,
            "channel_growth_rate": 0.07,
            "topic_trending_score": 0.10,
            "topic_saturation": 0.05
        }
        return importance
        
    async def monitor_feature_drift(
        self,
        feature_name: str,
        reference_window_days: int = 30,
        comparison_window_days: int = 7
    ) -> Dict[str, Any]:
        """Monitor feature drift"""
        # Load historical data
        reference_data = await self._load_historical_features(
            feature_name,
            reference_window_days
        )
        recent_data = await self._load_historical_features(
            feature_name,
            comparison_window_days
        )
        
        if reference_data and recent_data:
            # Calculate statistics
            ref_mean = np.mean(reference_data)
            ref_std = np.std(reference_data)
            recent_mean = np.mean(recent_data)
            recent_std = np.std(recent_data)
            
            # Calculate drift metrics
            mean_drift = abs(recent_mean - ref_mean) / max(ref_mean, 1e-6)
            std_drift = abs(recent_std - ref_std) / max(ref_std, 1e-6)
            
            return {
                "feature": feature_name,
                "reference_mean": ref_mean,
                "reference_std": ref_std,
                "recent_mean": recent_mean,
                "recent_std": recent_std,
                "mean_drift_pct": mean_drift * 100,
                "std_drift_pct": std_drift * 100,
                "is_drifting": mean_drift > 0.2 or std_drift > 0.3
            }
            
        return {"feature": feature_name, "is_drifting": False}
        
    async def _load_historical_features(
        self,
        feature_name: str,
        days: int
    ) -> List[float]:
        """Load historical feature values"""
        values = []
        today = datetime.utcnow()
        
        for days_back in range(days):
            date = today - timedelta(days=days_back)
            date_str = date.strftime("%Y%m%d")
            path = f"{self.config.offline_store_path}/{date_str}/features.parquet"
            
            if os.path.exists(path):
                df = pd.read_parquet(path)
                if feature_name in df.columns:
                    values.extend(df[feature_name].dropna().tolist())
                    
        return values