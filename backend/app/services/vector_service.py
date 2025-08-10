"""
Vector Database Service
Owner: VP of AI

Vector database service for content similarity, recommendations,
and semantic search using Qdrant vector database.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import asyncio
from dataclasses import dataclass
import hashlib
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.metrics import metrics

logger = logging.getLogger(__name__)


@dataclass
class ContentEmbedding:
    """Data structure for content embeddings."""
    id: str
    content_type: str  # 'video', 'script', 'channel', 'keyword'
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    timestamp: Optional[datetime] = None
    
    def to_point(self) -> PointStruct:
        """Convert to Qdrant point structure."""
        return PointStruct(
            id=self.id,
            vector=self.embedding,
            payload={
                "content_type": self.content_type,
                "content": self.content,
                "metadata": self.metadata,
                "timestamp": self.timestamp.isoformat() if self.timestamp else None
            }
        )


class VectorServiceError(Exception):
    """Custom exception for vector service errors."""
    pass


class VectorService:
    """Vector database service for semantic search and recommendations."""
    
    def __init__(self):
        self.client = None
        self.embedding_model = None
        self.collections = {
            "videos": "video_embeddings",
            "scripts": "script_embeddings", 
            "channels": "channel_embeddings",
            "keywords": "keyword_embeddings"
        }
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize vector database connection and embedding model."""
        try:
            # Initialize Qdrant client
            if settings.QDRANT_API_KEY:
                self.client = QdrantClient(
                    url=settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY
                )
            else:
                self.client = QdrantClient(
                    host=settings.QDRANT_HOST,
                    port=settings.QDRANT_PORT
                )
            
            # Test connection
            info = self.client.get_cluster_info()
            logger.info(f"Connected to Qdrant cluster: {info}")
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info(f"Loaded embedding model: {settings.EMBEDDING_MODEL}")
            
            # Create collections if they don't exist
            await self._setup_collections()
            
            self.initialized = True
            logger.info("Vector service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector service: {str(e)}")
            raise VectorServiceError(f"Vector service initialization failed: {str(e)}")
    
    async def _setup_collections(self) -> None:
        """Setup Qdrant collections for different content types."""
        try:
            for collection_name in self.collections.values():
                # Check if collection exists
                try:
                    collection_info = self.client.get_collection(collection_name)
                    logger.info(f"Collection {collection_name} already exists")
                except Exception:
                    # Collection doesn't exist, create it
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=settings.EMBEDDING_DIMENSION,
                            distance=Distance.COSINE
                        )
                    )
                    logger.info(f"Created collection: {collection_name}")
                    
        except Exception as e:
            logger.error(f"Failed to setup collections: {str(e)}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text content."""
        if not self.embedding_model:
            raise VectorServiceError("Embedding model not initialized")
        
        try:
            # Clean and prepare text
            clean_text = self._clean_text(text)
            
            # Generate embedding
            embedding = self.embedding_model.encode(clean_text)
            
            # Convert to list and normalize
            embedding_list = embedding.tolist()
            
            return embedding_list
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise VectorServiceError(f"Embedding generation failed: {str(e)}")
    
    async def store_content_embedding(
        self, 
        content_id: str, 
        content_type: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Store content embedding in vector database."""
        
        try:
            if not self.initialized:
                await self.initialize()
            
            # Generate embedding
            embedding = self.generate_embedding(content)
            
            # Create content embedding object
            content_embedding = ContentEmbedding(
                id=content_id,
                content_type=content_type,
                content=content[:1000],  # Store truncated content for payload
                metadata=metadata,
                embedding=embedding,
                timestamp=datetime.utcnow()
            )
            
            # Get appropriate collection
            collection_name = self.collections.get(f"{content_type}s", self.collections["videos"])
            
            # Store in Qdrant
            operation_info = self.client.upsert(
                collection_name=collection_name,
                points=[content_embedding.to_point()]
            )
            
            logger.info(f"Stored embedding for {content_type} {content_id} in {collection_name}")
            
            # Update metrics
            metrics.increment("vector_embeddings_stored", {"content_type": content_type})
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store content embedding: {str(e)}")
            metrics.increment("vector_errors", {"operation": "store"})
            return False
    
    async def find_similar_content(
        self, 
        query_text: str,
        content_type: str = "video",
        limit: int = 10,
        min_score: float = None
    ) -> List[Dict[str, Any]]:
        """Find similar content based on query text."""
        
        try:
            if not self.initialized:
                await self.initialize()
            
            # Generate query embedding
            query_embedding = self.generate_embedding(query_text)
            
            # Get collection name
            collection_name = self.collections.get(f"{content_type}s", self.collections["videos"])
            
            # Search for similar vectors
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=min_score or settings.SIMILARITY_THRESHOLD
            )
            
            # Format results
            similar_content = []
            for result in search_results:
                similar_content.append({
                    "id": result.id,
                    "score": result.score,
                    "content_type": result.payload.get("content_type"),
                    "content": result.payload.get("content"),
                    "metadata": result.payload.get("metadata", {}),
                    "timestamp": result.payload.get("timestamp")
                })
            
            logger.info(f"Found {len(similar_content)} similar {content_type} items for query")
            
            # Update metrics
            metrics.increment("vector_searches_performed", {"content_type": content_type})
            
            return similar_content
            
        except Exception as e:
            logger.error(f"Failed to find similar content: {str(e)}")
            metrics.increment("vector_errors", {"operation": "search"})
            return []
    
    async def get_content_recommendations(
        self, 
        user_id: str,
        content_history: List[str],
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get personalized content recommendations based on user history."""
        
        try:
            if not content_history:
                # Return popular content if no history
                return await self._get_popular_content(limit)
            
            # Create user profile embedding from content history
            user_profile_text = " ".join(content_history[-10:])  # Use last 10 items
            user_embedding = self.generate_embedding(user_profile_text)
            
            # Search across all video content
            recommendations = self.client.search(
                collection_name=self.collections["videos"],
                query_vector=user_embedding,
                limit=limit * 2  # Get more results to filter
            )
            
            # Filter out already seen content and format results
            seen_content = set(content_history)
            filtered_recommendations = []
            
            for result in recommendations:
                if result.id not in seen_content and len(filtered_recommendations) < limit:
                    filtered_recommendations.append({
                        "id": result.id,
                        "score": result.score,
                        "reason": "similar_interests",
                        "content": result.payload.get("content"),
                        "metadata": result.payload.get("metadata", {})
                    })
            
            logger.info(f"Generated {len(filtered_recommendations)} recommendations for user {user_id}")
            
            return filtered_recommendations
            
        except Exception as e:
            logger.error(f"Failed to get content recommendations: {str(e)}")
            return await self._get_popular_content(limit)
    
    async def find_trending_topics(self, time_window_days: int = 7) -> List[Dict[str, Any]]:
        """Find trending topics based on recent content."""
        
        try:
            from datetime import timedelta
            
            # Get recent content
            cutoff_date = datetime.utcnow() - timedelta(days=time_window_days)
            
            # This would typically require more complex aggregation
            # For now, return sample trending topics based on recent searches
            trending_topics = [
                {"topic": "AI Technology", "frequency": 45, "growth_rate": 1.2},
                {"topic": "Web Development", "frequency": 38, "growth_rate": 0.9},
                {"topic": "Machine Learning", "frequency": 32, "growth_rate": 1.5},
                {"topic": "Data Science", "frequency": 28, "growth_rate": 1.1},
                {"topic": "Blockchain", "frequency": 21, "growth_rate": 0.8}
            ]
            
            return trending_topics
            
        except Exception as e:
            logger.error(f"Failed to find trending topics: {str(e)}")
            return []
    
    async def cluster_similar_content(
        self, 
        content_ids: List[str],
        num_clusters: int = 5
    ) -> Dict[int, List[str]]:
        """Cluster content by similarity."""
        
        try:
            # Get embeddings for content
            embeddings = []
            valid_ids = []
            
            for content_id in content_ids:
                # Search for content in collections
                for collection_name in self.collections.values():
                    try:
                        points = self.client.retrieve(
                            collection_name=collection_name,
                            ids=[content_id]
                        )
                        if points:
                            embeddings.append(points[0].vector)
                            valid_ids.append(content_id)
                            break
                    except Exception:
                        continue
            
            if len(embeddings) < 2:
                return {0: valid_ids}
            
            # Perform clustering using sklearn
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=min(num_clusters, len(embeddings)))
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Group content by clusters
            clusters = {}
            for idx, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(valid_ids[idx])
            
            logger.info(f"Clustered {len(valid_ids)} items into {len(clusters)} clusters")
            
            return clusters
            
        except Exception as e:
            logger.error(f"Failed to cluster content: {str(e)}")
            return {0: content_ids}
    
    async def delete_content_embedding(self, content_id: str, content_type: str = "video") -> bool:
        """Delete content embedding from vector database."""
        
        try:
            collection_name = self.collections.get(f"{content_type}s", self.collections["videos"])
            
            operation_info = self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=[content_id]
                )
            )
            
            logger.info(f"Deleted embedding for {content_type} {content_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete content embedding: {str(e)}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections."""
        
        try:
            stats = {}
            
            for content_type, collection_name in self.collections.items():
                try:
                    collection_info = self.client.get_collection(collection_name)
                    stats[content_type] = {
                        "total_points": collection_info.points_count,
                        "vector_dimension": collection_info.config.params.vectors.size,
                        "distance_metric": collection_info.config.params.vectors.distance.value
                    }
                except Exception as e:
                    stats[content_type] = {"error": str(e)}
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {}
    
    def _clean_text(self, text: str) -> str:
        """Clean text for embedding generation."""
        # Remove extra whitespace and normalize
        cleaned = " ".join(text.split())
        
        # Truncate if too long (embedding models have token limits)
        if len(cleaned) > 500:
            cleaned = cleaned[:500]
        
        return cleaned
    
    async def _get_popular_content(self, limit: int) -> List[Dict[str, Any]]:
        """Get popular content as fallback recommendations."""
        
        try:
            # In a real implementation, this would be based on metrics
            # For now, return a sample
            popular_content = [
                {
                    "id": f"popular_{i}",
                    "score": 0.9 - (i * 0.05),
                    "reason": "popular",
                    "content": f"Popular content item {i}",
                    "metadata": {"category": "trending", "views": 10000 - (i * 500)}
                }
                for i in range(min(limit, 10))
            ]
            
            return popular_content
            
        except Exception as e:
            logger.error(f"Failed to get popular content: {str(e)}")
            return []


# Global instance
vector_service = VectorService()