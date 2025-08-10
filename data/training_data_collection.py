"""
Training Data Collection System
Collects and processes data for ML model training
"""
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from pathlib import Path
import aiohttp
import aiofiles
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
import redis.asyncio as redis
from sklearn.preprocessing import StandardScaler
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class DataSource:
    """Configuration for a data source"""
    name: str
    type: str  # youtube, trending, analytics, user_feedback
    endpoint: Optional[str] = None
    frequency: str = "daily"  # hourly, daily, weekly
    enabled: bool = True
    last_collected: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingDataset:
    """Represents a training dataset"""
    dataset_id: str
    name: str
    version: str
    created_at: datetime
    features: List[str]
    target: str
    size: int
    data_path: str
    metadata: Dict[str, Any]
    validation_metrics: Optional[Dict[str, float]] = None


class TrainingDataCollector:
    """
    Collects and processes training data from multiple sources
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        data_dir: str = "data/training"
    ):
        self.db = db_session
        self.redis_client = None
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Data sources configuration
        self.data_sources = [
            DataSource(
                name="youtube_trending",
                type="youtube",
                endpoint="/api/v1/youtube/trending",
                frequency="hourly"
            ),
            DataSource(
                name="channel_analytics",
                type="analytics",
                endpoint="/api/v1/analytics/channels",
                frequency="daily"
            ),
            DataSource(
                name="video_performance",
                type="analytics",
                endpoint="/api/v1/analytics/videos",
                frequency="daily"
            ),
            DataSource(
                name="user_interactions",
                type="user_feedback",
                endpoint="/api/v1/analytics/interactions",
                frequency="hourly"
            )
        ]
        
        # Feature extractors
        self.feature_extractors = {
            "youtube": self._extract_youtube_features,
            "analytics": self._extract_analytics_features,
            "user_feedback": self._extract_user_features
        }
        
    async def initialize(self):
        """Initialize the data collector"""
        try:
            self.redis_client = await redis.from_url(
                f"redis://localhost:{6379}",
                encoding="utf-8",
                decode_responses=True
            )
            logger.info("Training data collector initialized")
        except Exception as e:
            logger.error(f"Failed to initialize data collector: {e}")
            raise
    
    async def collect_all_sources(self) -> Dict[str, Any]:
        """
        Collect data from all configured sources
        
        Returns:
            Dictionary with collected data from each source
        """
        collected_data = {}
        tasks = []
        
        for source in self.data_sources:
            if source.enabled and self._should_collect(source):
                tasks.append(self._collect_from_source(source))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for source, result in zip(self.data_sources, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to collect from {source.name}: {result}")
                collected_data[source.name] = None
            else:
                collected_data[source.name] = result
                source.last_collected = datetime.now()
        
        return collected_data
    
    def _should_collect(self, source: DataSource) -> bool:
        """Check if data should be collected from source"""
        if not source.last_collected:
            return True
        
        frequency_hours = {
            "hourly": 1,
            "daily": 24,
            "weekly": 168
        }
        
        hours_since_last = (datetime.now() - source.last_collected).total_seconds() / 3600
        return hours_since_last >= frequency_hours.get(source.frequency, 24)
    
    async def _collect_from_source(self, source: DataSource) -> Dict[str, Any]:
        """Collect data from a specific source"""
        try:
            if source.type == "youtube":
                return await self._collect_youtube_data(source)
            elif source.type == "analytics":
                return await self._collect_analytics_data(source)
            elif source.type == "user_feedback":
                return await self._collect_user_data(source)
            else:
                logger.warning(f"Unknown source type: {source.type}")
                return {}
        except Exception as e:
            logger.error(f"Error collecting from {source.name}: {e}")
            raise
    
    async def _collect_youtube_data(self, source: DataSource) -> Dict[str, Any]:
        """Collect YouTube trending and video data"""
        # Simulate YouTube data collection
        # In production, this would call actual YouTube API
        
        data = {
            "trending_videos": [],
            "categories": {},
            "collected_at": datetime.now().isoformat()
        }
        
        # Mock trending videos data
        for i in range(50):
            video = {
                "video_id": f"video_{i}",
                "title": f"Trending Video {i}",
                "view_count": np.random.randint(10000, 1000000),
                "like_count": np.random.randint(100, 50000),
                "comment_count": np.random.randint(10, 5000),
                "category_id": np.random.choice(["1", "2", "10", "15", "17", "19", "20", "22", "23", "24"]),
                "duration": np.random.randint(60, 1200),
                "published_at": (datetime.now() - timedelta(days=np.random.randint(0, 30))).isoformat(),
                "tags": [f"tag_{j}" for j in range(np.random.randint(3, 10))],
                "engagement_rate": np.random.random() * 0.1
            }
            data["trending_videos"].append(video)
        
        # Category performance
        categories = ["Gaming", "Music", "Education", "Entertainment", "Tech", "Sports"]
        for cat in categories:
            data["categories"][cat] = {
                "avg_views": np.random.randint(50000, 500000),
                "avg_engagement": np.random.random() * 0.1,
                "trending_count": np.random.randint(1, 10)
            }
        
        return data
    
    async def _collect_analytics_data(self, source: DataSource) -> Dict[str, Any]:
        """Collect analytics data from database"""
        # This would query actual analytics tables
        # For now, return mock data
        
        data = {
            "channel_metrics": [],
            "video_metrics": [],
            "collected_at": datetime.now().isoformat()
        }
        
        # Mock channel metrics
        for i in range(10):
            channel = {
                "channel_id": f"channel_{i}",
                "subscriber_count": np.random.randint(1000, 100000),
                "total_views": np.random.randint(100000, 10000000),
                "avg_view_duration": np.random.randint(60, 600),
                "videos_published": np.random.randint(10, 500),
                "growth_rate": np.random.random() * 0.2 - 0.1
            }
            data["channel_metrics"].append(channel)
        
        # Mock video metrics
        for i in range(100):
            video = {
                "video_id": f"video_{i}",
                "views": np.random.randint(1000, 500000),
                "watch_time_minutes": np.random.randint(100, 50000),
                "ctr": np.random.random() * 0.1,
                "avg_view_duration": np.random.randint(30, 600),
                "retention_rate": np.random.random() * 0.6 + 0.2
            }
            data["video_metrics"].append(video)
        
        return data
    
    async def _collect_user_data(self, source: DataSource) -> Dict[str, Any]:
        """Collect user interaction and feedback data"""
        # Mock user interaction data
        
        data = {
            "user_interactions": [],
            "feedback": [],
            "collected_at": datetime.now().isoformat()
        }
        
        # Mock interactions
        for i in range(200):
            interaction = {
                "user_id": f"user_{np.random.randint(1, 50)}",
                "video_id": f"video_{np.random.randint(1, 100)}",
                "action": np.random.choice(["view", "like", "comment", "share", "subscribe"]),
                "timestamp": (datetime.now() - timedelta(hours=np.random.randint(0, 168))).isoformat(),
                "duration": np.random.randint(10, 600) if np.random.random() > 0.5 else None
            }
            data["user_interactions"].append(interaction)
        
        # Mock feedback
        for i in range(50):
            feedback = {
                "feedback_id": f"feedback_{i}",
                "video_id": f"video_{np.random.randint(1, 100)}",
                "rating": np.random.randint(1, 6),
                "sentiment": np.random.choice(["positive", "neutral", "negative"]),
                "timestamp": (datetime.now() - timedelta(hours=np.random.randint(0, 72))).isoformat()
            }
            data["feedback"].append(feedback)
        
        return data
    
    async def process_and_store(
        self,
        raw_data: Dict[str, Any],
        dataset_name: str
    ) -> TrainingDataset:
        """
        Process raw data and create training dataset
        
        Args:
            raw_data: Raw collected data
            dataset_name: Name for the dataset
        
        Returns:
            TrainingDataset object
        """
        # Extract features from all sources
        all_features = []
        all_targets = []
        
        for source_name, source_data in raw_data.items():
            if source_data is None:
                continue
            
            # Determine source type
            source_type = next(
                (s.type for s in self.data_sources if s.name == source_name),
                None
            )
            
            if source_type and source_type in self.feature_extractors:
                features, targets = self.feature_extractors[source_type](source_data)
                all_features.extend(features)
                all_targets.extend(targets)
        
        # Create DataFrame
        if all_features:
            df = pd.DataFrame(all_features)
            
            # Add targets if available
            if all_targets:
                df['target'] = all_targets
            
            # Generate dataset ID
            dataset_id = hashlib.md5(
                f"{dataset_name}_{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12]
            
            # Save dataset
            dataset_path = self.data_dir / f"{dataset_id}.parquet"
            df.to_parquet(dataset_path, compression='snappy')
            
            # Create dataset object
            dataset = TrainingDataset(
                dataset_id=dataset_id,
                name=dataset_name,
                version="1.0",
                created_at=datetime.now(),
                features=list(df.columns),
                target="target" if "target" in df.columns else None,
                size=len(df),
                data_path=str(dataset_path),
                metadata={
                    "sources": list(raw_data.keys()),
                    "processing_date": datetime.now().isoformat()
                }
            )
            
            # Store metadata in Redis
            await self._store_dataset_metadata(dataset)
            
            return dataset
        
        raise ValueError("No features extracted from raw data")
    
    def _extract_youtube_features(self, data: Dict[str, Any]) -> tuple:
        """Extract features from YouTube data"""
        features = []
        targets = []
        
        for video in data.get("trending_videos", []):
            feature = {
                "view_count": video["view_count"],
                "like_count": video["like_count"],
                "comment_count": video["comment_count"],
                "duration_seconds": video["duration"],
                "category_id": int(video["category_id"]),
                "num_tags": len(video.get("tags", [])),
                "title_length": len(video["title"]),
                "days_since_published": (
                    datetime.now() - datetime.fromisoformat(video["published_at"])
                ).days,
                "engagement_rate": video.get("engagement_rate", 0)
            }
            features.append(feature)
            
            # Target could be engagement rate or view count growth
            targets.append(video.get("engagement_rate", 0))
        
        return features, targets
    
    def _extract_analytics_features(self, data: Dict[str, Any]) -> tuple:
        """Extract features from analytics data"""
        features = []
        targets = []
        
        for video in data.get("video_metrics", []):
            feature = {
                "views": video["views"],
                "watch_time_minutes": video["watch_time_minutes"],
                "ctr": video["ctr"],
                "avg_view_duration": video["avg_view_duration"],
                "retention_rate": video["retention_rate"]
            }
            features.append(feature)
            
            # Target could be video success metric
            targets.append(video["views"] > 100000)  # Binary classification
        
        return features, targets
    
    def _extract_user_features(self, data: Dict[str, Any]) -> tuple:
        """Extract features from user interaction data"""
        features = []
        targets = []
        
        # Aggregate user interactions by video
        video_interactions = {}
        
        for interaction in data.get("user_interactions", []):
            video_id = interaction["video_id"]
            if video_id not in video_interactions:
                video_interactions[video_id] = {
                    "view_count": 0,
                    "like_count": 0,
                    "comment_count": 0,
                    "share_count": 0,
                    "subscribe_count": 0,
                    "avg_duration": []
                }
            
            action = interaction["action"]
            if action == "view":
                video_interactions[video_id]["view_count"] += 1
                if interaction.get("duration"):
                    video_interactions[video_id]["avg_duration"].append(interaction["duration"])
            elif action == "like":
                video_interactions[video_id]["like_count"] += 1
            elif action == "comment":
                video_interactions[video_id]["comment_count"] += 1
            elif action == "share":
                video_interactions[video_id]["share_count"] += 1
            elif action == "subscribe":
                video_interactions[video_id]["subscribe_count"] += 1
        
        # Convert to features
        for video_id, interactions in video_interactions.items():
            avg_duration = np.mean(interactions["avg_duration"]) if interactions["avg_duration"] else 0
            
            feature = {
                "user_views": interactions["view_count"],
                "user_likes": interactions["like_count"],
                "user_comments": interactions["comment_count"],
                "user_shares": interactions["share_count"],
                "user_subscribes": interactions["subscribe_count"],
                "avg_watch_duration": avg_duration,
                "engagement_score": (
                    interactions["like_count"] * 1 +
                    interactions["comment_count"] * 2 +
                    interactions["share_count"] * 3 +
                    interactions["subscribe_count"] * 5
                ) / max(interactions["view_count"], 1)
            }
            features.append(feature)
            
            # Target is high engagement
            targets.append(feature["engagement_score"] > 0.1)
        
        return features, targets
    
    async def _store_dataset_metadata(self, dataset: TrainingDataset):
        """Store dataset metadata in Redis"""
        if self.redis_client:
            metadata = {
                "dataset_id": dataset.dataset_id,
                "name": dataset.name,
                "version": dataset.version,
                "created_at": dataset.created_at.isoformat(),
                "features": dataset.features,
                "target": dataset.target,
                "size": dataset.size,
                "data_path": dataset.data_path,
                "metadata": dataset.metadata
            }
            
            await self.redis_client.setex(
                f"training_dataset:{dataset.dataset_id}",
                86400 * 30,  # 30 days TTL
                json.dumps(metadata)
            )
            
            # Add to dataset index
            await self.redis_client.sadd("training_datasets", dataset.dataset_id)
    
    async def get_latest_dataset(self, dataset_name: str = None) -> Optional[TrainingDataset]:
        """Get the latest training dataset"""
        if not self.redis_client:
            return None
        
        # Get all dataset IDs
        dataset_ids = await self.redis_client.smembers("training_datasets")
        
        latest_dataset = None
        latest_time = None
        
        for dataset_id in dataset_ids:
            metadata_str = await self.redis_client.get(f"training_dataset:{dataset_id}")
            if metadata_str:
                metadata = json.loads(metadata_str)
                
                # Filter by name if specified
                if dataset_name and metadata["name"] != dataset_name:
                    continue
                
                created_at = datetime.fromisoformat(metadata["created_at"])
                if latest_time is None or created_at > latest_time:
                    latest_time = created_at
                    latest_dataset = TrainingDataset(
                        dataset_id=metadata["dataset_id"],
                        name=metadata["name"],
                        version=metadata["version"],
                        created_at=created_at,
                        features=metadata["features"],
                        target=metadata["target"],
                        size=metadata["size"],
                        data_path=metadata["data_path"],
                        metadata=metadata["metadata"]
                    )
        
        return latest_dataset
    
    async def validate_dataset(self, dataset: TrainingDataset) -> Dict[str, Any]:
        """Validate a training dataset"""
        try:
            # Load dataset
            df = pd.read_parquet(dataset.data_path)
            
            validation_results = {
                "is_valid": True,
                "issues": [],
                "statistics": {}
            }
            
            # Check for missing values
            missing_counts = df.isnull().sum()
            if missing_counts.any():
                validation_results["issues"].append({
                    "type": "missing_values",
                    "columns": missing_counts[missing_counts > 0].to_dict()
                })
            
            # Check for duplicates
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                validation_results["issues"].append({
                    "type": "duplicates",
                    "count": int(duplicate_count)
                })
            
            # Calculate statistics
            validation_results["statistics"] = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
                "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=['object']).columns.tolist()
            }
            
            # Check data quality
            if len(df) < 100:
                validation_results["issues"].append({
                    "type": "insufficient_data",
                    "message": f"Dataset has only {len(df)} samples"
                })
                validation_results["is_valid"] = False
            
            # Update dataset validation metrics
            dataset.validation_metrics = validation_results
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Dataset validation failed: {e}")
            return {
                "is_valid": False,
                "issues": [{"type": "error", "message": str(e)}],
                "statistics": {}
            }


# API Integration
class TrainingDataAPI:
    """API endpoints for training data collection"""
    
    def __init__(self, db_session: AsyncSession):
        self.collector = TrainingDataCollector(db_session)
    
    async def initialize(self):
        """Initialize the API"""
        await self.collector.initialize()
    
    async def trigger_collection(self) -> Dict[str, Any]:
        """Trigger data collection from all sources"""
        raw_data = await self.collector.collect_all_sources()
        
        # Process and store
        dataset = await self.collector.process_and_store(
            raw_data,
            f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Validate dataset
        validation_results = await self.collector.validate_dataset(dataset)
        
        return {
            "dataset_id": dataset.dataset_id,
            "size": dataset.size,
            "features": len(dataset.features),
            "validation": validation_results,
            "data_path": dataset.data_path
        }
    
    async def get_datasets(self) -> List[Dict[str, Any]]:
        """Get list of available datasets"""
        # Implementation would query stored datasets
        return []
    
    async def get_dataset_info(self, dataset_id: str) -> Dict[str, Any]:
        """Get information about a specific dataset"""
        # Implementation would retrieve dataset metadata
        return {}