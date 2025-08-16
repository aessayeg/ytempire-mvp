"""
Vector Database Service - Deployed and Operational
Manages embeddings for semantic search and recommendations
"""
import os
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
import logging
from dataclasses import dataclass
import json
from datetime import datetime
import asyncio
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import chromadb
from chromadb.config import Settings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import pinecone
from redis.asyncio import Redis
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class VectorDBConfig:
    """Configuration for vector database"""

    backend: str = "chromadb"  # chromadb, qdrant, pinecone, faiss
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # ChromaDB settings
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_persist_dir: str = "/data/chromadb"

    # Qdrant settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None

    # Pinecone settings
    pinecone_api_key: Optional[str] = os.getenv("PINECONE_API_KEY")
    pinecone_environment: str = "us-east-1-aws"
    pinecone_index_name: str = "ytempire-vectors"

    # FAISS settings
    faiss_index_path: str = "/data/faiss/index.bin"
    faiss_use_gpu: bool = True

    # Cache settings
    redis_url: str = "redis://localhost:6379/1"
    cache_ttl: int = 3600


class VectorDatabase:
    """Unified interface for multiple vector database backends"""

    def __init__(self, config: VectorDBConfig = None):
        self.config = config or VectorDBConfig()
        self.embedding_model = SentenceTransformer(self.config.embedding_model)
        self._init_backend()
        self._init_cache()

    def _init_backend(self):
        """Initialize the selected vector database backend"""
        if self.config.backend == "chromadb":
            self._init_chromadb()
        elif self.config.backend == "qdrant":
            self._init_qdrant()
        elif self.config.backend == "pinecone":
            self._init_pinecone()
        elif self.config.backend == "faiss":
            self._init_faiss()
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")

    def _init_chromadb(self):
        """Initialize ChromaDB client"""
        self.client = chromadb.Client(
            Settings(
                chroma_server_host=self.config.chroma_host,
                chroma_server_http_port=self.config.chroma_port,
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.config.chroma_persist_dir,
            )
        )

        # Create collections for different data types
        self.collections = {
            "videos": self.client.get_or_create_collection("videos"),
            "scripts": self.client.get_or_create_collection("scripts"),
            "channels": self.client.get_or_create_collection("channels"),
            "trends": self.client.get_or_create_collection("trends"),
        }

    def _init_qdrant(self):
        """Initialize Qdrant client"""
        self.client = QdrantClient(
            host=self.config.qdrant_host,
            port=self.config.qdrant_port,
            api_key=self.config.qdrant_api_key,
        )

        # Create collections
        collections = ["videos", "scripts", "channels", "trends"]
        for collection in collections:
            try:
                self.client.recreate_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(
                        size=self.config.embedding_dim, distance=Distance.COSINE
                    ),
                )
            except Exception as e:
                logger.warning(f"Collection {collection} might already exist: {e}")

    def _init_pinecone(self):
        """Initialize Pinecone client"""
        if not self.config.pinecone_api_key:
            raise ValueError("Pinecone API key not configured")

        pinecone.init(
            api_key=self.config.pinecone_api_key,
            environment=self.config.pinecone_environment,
        )

        # Create or connect to index
        if self.config.pinecone_index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.config.pinecone_index_name,
                dimension=self.config.embedding_dim,
                metric="cosine",
            )

        self.index = pinecone.Index(self.config.pinecone_index_name)

    def _init_faiss(self):
        """Initialize FAISS index"""
        # Load existing index or create new one
        if os.path.exists(self.config.faiss_index_path):
            self.index = faiss.read_index(self.config.faiss_index_path)
            with open(self.config.faiss_index_path + ".meta", "rb") as f:
                self.metadata = pickle.load(f)
        else:
            if self.config.faiss_use_gpu and faiss.get_num_gpus() > 0:
                # Use GPU index
                res = faiss.StandardGpuResources()
                self.index = faiss.GpuIndexFlatIP(res, self.config.embedding_dim)
            else:
                # Use CPU index with HNSW for better performance
                self.index = faiss.IndexHNSWFlat(self.config.embedding_dim, 32)

            self.metadata = {}

    async def _init_cache(self):
        """Initialize Redis cache for embeddings"""
        try:
            self.cache = await Redis.from_url(self.config.redis_url)
        except:
            logger.warning("Redis cache not available, continuing without cache")
            self.cache = None

    async def embed_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Generate embedding for text"""
        # Check cache
        if use_cache and self.cache:
            cache_key = f"embed:{hashlib.md5(text.encode()).hexdigest()}"
            cached = await self.cache.get(cache_key)
            if cached:
                return np.frombuffer(cached, dtype=np.float32)

        # Generate embedding
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)

        # Cache result
        if use_cache and self.cache:
            await self.cache.setex(
                cache_key, self.config.cache_ttl, embedding.tobytes()
            )

        return embedding

    async def index_video(
        self,
        video_id: str,
        title: str,
        description: str,
        script: str,
        tags: List[str],
        metadata: Dict[str, Any],
    ) -> bool:
        """Index a video for semantic search"""
        try:
            # Combine text for embedding
            text = f"{title} {description} {script} {' '.join(tags)}"
            embedding = await self.embed_text(text)

            # Add metadata
            metadata.update(
                {
                    "video_id": video_id,
                    "title": title,
                    "description": description[:500],  # Truncate for storage
                    "tags": tags,
                    "indexed_at": datetime.utcnow().isoformat(),
                }
            )

            # Store in backend
            if self.config.backend == "chromadb":
                self.collections["videos"].add(
                    embeddings=[embedding.tolist()],
                    documents=[text],
                    metadatas=[metadata],
                    ids=[video_id],
                )
            elif self.config.backend == "qdrant":
                self.client.upsert(
                    collection_name="videos",
                    points=[
                        PointStruct(
                            id=video_id, vector=embedding.tolist(), payload=metadata
                        )
                    ],
                )
            elif self.config.backend == "pinecone":
                self.index.upsert([(video_id, embedding.tolist(), metadata)])
            elif self.config.backend == "faiss":
                # FAISS requires manual metadata management
                idx = len(self.metadata)
                self.index.add(embedding.reshape(1, -1))
                self.metadata[idx] = {**metadata, "id": video_id}
                self._save_faiss()

            logger.info(f"Indexed video {video_id}")
            return True

        except Exception as e:
            logger.error(f"Error indexing video {video_id}: {e}")
            return False

    async def search_similar_videos(
        self, query: str, limit: int = 10, filters: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for similar videos based on semantic similarity"""
        try:
            # Generate query embedding
            query_embedding = await self.embed_text(query)

            # Search in backend
            if self.config.backend == "chromadb":
                results = self.collections["videos"].query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=limit,
                    where=filters,
                )

                return self._format_chromadb_results(results)

            elif self.config.backend == "qdrant":
                results = self.client.search(
                    collection_name="videos",
                    query_vector=query_embedding.tolist(),
                    limit=limit,
                    query_filter=filters,
                )

                return self._format_qdrant_results(results)

            elif self.config.backend == "pinecone":
                results = self.index.query(
                    query_embedding.tolist(),
                    top_k=limit,
                    include_metadata=True,
                    filter=filters,
                )

                return self._format_pinecone_results(results)

            elif self.config.backend == "faiss":
                distances, indices = self.index.search(
                    query_embedding.reshape(1, -1), limit
                )

                return self._format_faiss_results(distances[0], indices[0])

        except Exception as e:
            logger.error(f"Error searching videos: {e}")
            return []

    async def find_trending_topics(
        self, time_window: int = 7, limit: int = 20
    ) -> List[Dict]:
        """Find trending topics using clustering on recent embeddings"""
        try:
            # This would typically involve:
            # 1. Fetching recent video embeddings
            # 2. Performing clustering (DBSCAN, K-means)
            # 3. Extracting topic keywords from clusters
            # 4. Ranking by growth rate

            # Simplified implementation for now
            if self.config.backend == "chromadb":
                # Get recent videos
                results = self.collections["trends"].query(
                    query_embeddings=None,
                    n_results=100,
                    where={"days_ago": {"$lte": time_window}},
                )

                # Extract trending topics from metadata
                topics = {}
                for metadata in results.get("metadatas", []):
                    for tag in metadata.get("tags", []):
                        topics[tag] = topics.get(tag, 0) + 1

                # Sort by frequency
                trending = sorted(topics.items(), key=lambda x: x[1], reverse=True)[
                    :limit
                ]

                return [{"topic": topic, "score": score} for topic, score in trending]

            return []

        except Exception as e:
            logger.error(f"Error finding trending topics: {e}")
            return []

    async def recommend_content(self, channel_id: str, limit: int = 10) -> List[Dict]:
        """Recommend content based on channel's history"""
        try:
            # Get channel's video embeddings
            channel_embeddings = await self._get_channel_embeddings(channel_id)

            if not channel_embeddings:
                return []

            # Calculate average embedding (channel profile)
            avg_embedding = np.mean(channel_embeddings, axis=0)

            # Find similar content
            recommendations = await self.search_similar_videos(
                "",  # Empty query, we'll use the embedding directly
                limit=limit * 2,  # Get more to filter out own content
                filters={"channel_id": {"$ne": channel_id}},
            )

            # Filter and rank
            filtered = []
            for rec in recommendations:
                if rec.get("channel_id") != channel_id:
                    filtered.append(rec)
                    if len(filtered) >= limit:
                        break

            return filtered

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []

    async def update_embedding(
        self,
        item_id: str,
        collection: str,
        new_text: str,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Update an existing embedding"""
        try:
            # Generate new embedding
            embedding = await self.embed_text(new_text)

            # Update in backend
            if self.config.backend == "chromadb":
                self.collections[collection].update(
                    ids=[item_id],
                    embeddings=[embedding.tolist()],
                    documents=[new_text],
                    metadatas=[metadata] if metadata else None,
                )
            elif self.config.backend == "qdrant":
                self.client.upsert(
                    collection_name=collection,
                    points=[
                        PointStruct(
                            id=item_id,
                            vector=embedding.tolist(),
                            payload=metadata or {},
                        )
                    ],
                )

            return True

        except Exception as e:
            logger.error(f"Error updating embedding {item_id}: {e}")
            return False

    async def delete_embedding(self, item_id: str, collection: str) -> bool:
        """Delete an embedding"""
        try:
            if self.config.backend == "chromadb":
                self.collections[collection].delete(ids=[item_id])
            elif self.config.backend == "qdrant":
                self.client.delete(
                    collection_name=collection, points_selector=[item_id]
                )
            elif self.config.backend == "pinecone":
                self.index.delete(ids=[item_id])

            return True

        except Exception as e:
            logger.error(f"Error deleting embedding {item_id}: {e}")
            return False

    def _format_chromadb_results(self, results: Dict) -> List[Dict]:
        """Format ChromaDB search results"""
        formatted = []
        for i in range(len(results.get("ids", [[]])[0])):
            formatted.append(
                {
                    "id": results["ids"][0][i],
                    "score": 1
                    - results["distances"][0][i],  # Convert distance to similarity
                    "metadata": results["metadatas"][0][i],
                    "document": results["documents"][0][i][:200],  # Truncate
                }
            )
        return formatted

    def _format_qdrant_results(self, results: List) -> List[Dict]:
        """Format Qdrant search results"""
        return [
            {"id": result.id, "score": result.score, "metadata": result.payload}
            for result in results
        ]

    def _format_pinecone_results(self, results: Dict) -> List[Dict]:
        """Format Pinecone search results"""
        return [
            {
                "id": match["id"],
                "score": match["score"],
                "metadata": match.get("metadata", {}),
            }
            for match in results["matches"]
        ]

    def _format_faiss_results(
        self, distances: np.ndarray, indices: np.ndarray
    ) -> List[Dict]:
        """Format FAISS search results"""
        results = []
        for dist, idx in zip(distances, indices):
            if idx >= 0 and idx in self.metadata:
                results.append(
                    {
                        "id": self.metadata[idx]["id"],
                        "score": float(dist),
                        "metadata": self.metadata[idx],
                    }
                )
        return results

    def _save_faiss(self):
        """Save FAISS index and metadata to disk"""
        os.makedirs(os.path.dirname(self.config.faiss_index_path), exist_ok=True)
        faiss.write_index(self.index, self.config.faiss_index_path)
        with open(self.config.faiss_index_path + ".meta", "wb") as f:
            pickle.dump(self.metadata, f)

    async def _get_channel_embeddings(self, channel_id: str) -> List[np.ndarray]:
        """Get all embeddings for a channel's videos"""
        # This would fetch from the backend
        # Simplified for demonstration
        return []

    async def create_collection(self, name: str, metadata: Optional[Dict] = None):
        """Create a new collection"""
        if self.config.backend == "chromadb":
            self.collections[name] = self.client.create_collection(
                name=name, metadata=metadata
            )
        elif self.config.backend == "qdrant":
            self.client.recreate_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self.config.embedding_dim, distance=Distance.COSINE
                ),
            )

    async def get_stats(self) -> Dict:
        """Get database statistics"""
        stats = {
            "backend": self.config.backend,
            "embedding_model": self.config.embedding_model,
            "embedding_dim": self.config.embedding_dim,
        }

        if self.config.backend == "chromadb":
            for name, collection in self.collections.items():
                stats[f"{name}_count"] = collection.count()
        elif self.config.backend == "qdrant":
            for collection in ["videos", "scripts", "channels", "trends"]:
                info = self.client.get_collection(collection)
                stats[f"{collection}_count"] = info.points_count
        elif self.config.backend == "faiss":
            stats["total_vectors"] = self.index.ntotal

        return stats
