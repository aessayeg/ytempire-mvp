"""
Vector Database Service for YTEmpire
Handles semantic search and content similarity using ChromaDB/Pinecone
"""
import os
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import logging
from dataclasses import dataclass
import asyncio

# Vector database imports
import chromadb
from chromadb.config import Settings
import pinecone
from sentence_transformers import SentenceTransformer
import openai
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VectorDocument:
    """Container for vector database documents"""

    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    score: Optional[float] = None


class VectorDatabaseService:
    """
    Unified vector database service supporting multiple backends
    """

    def __init__(
        self,
        backend: str = "chroma",
        embedding_model: str = "all-MiniLM-L6-v2",
        api_key: Optional[str] = None,
    ):
        """
        Initialize vector database service

        Args:
            backend: Vector DB backend ('chroma', 'pinecone')
            embedding_model: Model for generating embeddings
            api_key: API key for cloud services
        """
        self.backend = backend
        self.embedding_model_name = embedding_model

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dimension = (
            self.embedding_model.get_sentence_embedding_dimension()
        )

        # Initialize backend
        if backend == "chroma":
            self._init_chroma()
        elif backend == "pinecone":
            self._init_pinecone(api_key)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        logger.info(f"Vector database initialized with {backend} backend")

    def _init_chroma(self):
        """Initialize ChromaDB backend"""
        self.chroma_client = chromadb.Client(
            Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db")
        )

        # Create collections
        self.collections = {}

        # Video scripts collection
        try:
            self.collections["scripts"] = self.chroma_client.get_collection(
                "video_scripts"
            )
        except:
            self.collections["scripts"] = self.chroma_client.create_collection(
                name="video_scripts", metadata={"hnsw:space": "cosine"}
            )

        # Video metadata collection
        try:
            self.collections["metadata"] = self.chroma_client.get_collection(
                "video_metadata"
            )
        except:
            self.collections["metadata"] = self.chroma_client.create_collection(
                name="video_metadata", metadata={"hnsw:space": "cosine"}
            )

        # Channel profiles collection
        try:
            self.collections["channels"] = self.chroma_client.get_collection("channels")
        except:
            self.collections["channels"] = self.chroma_client.create_collection(
                name="channels", metadata={"hnsw:space": "cosine"}
            )

        # Trending topics collection
        try:
            self.collections["trends"] = self.chroma_client.get_collection(
                "trending_topics"
            )
        except:
            self.collections["trends"] = self.chroma_client.create_collection(
                name="trending_topics", metadata={"hnsw:space": "cosine"}
            )

    def _init_pinecone(self, api_key: str):
        """Initialize Pinecone backend"""
        if not api_key:
            raise ValueError("Pinecone API key required")

        pinecone.init(api_key=api_key, environment="us-west1-gcp")

        # Create or connect to indexes
        self.indexes = {}

        index_names = ["scripts", "metadata", "channels", "trends"]
        for name in index_names:
            index_name = f"ytempire-{name}"

            if index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    index_name, dimension=self.embedding_dimension, metric="cosine"
                )

            self.indexes[name] = pinecone.Index(index_name)

    async def add_document(
        self,
        collection: str,
        content: str,
        metadata: Dict[str, Any],
        document_id: Optional[str] = None,
    ) -> str:
        """
        Add document to vector database

        Args:
            collection: Collection name
            content: Document content
            metadata: Document metadata
            document_id: Optional document ID

        Returns:
            Document ID
        """
        # Generate document ID if not provided
        if not document_id:
            document_id = self._generate_id(content)

        # Generate embedding
        embedding = self._generate_embedding(content)

        # Add timestamp to metadata
        metadata["indexed_at"] = datetime.utcnow().isoformat()
        metadata["content_length"] = len(content)

        if self.backend == "chroma":
            self.collections[collection].add(
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata],
                ids=[document_id],
            )
        elif self.backend == "pinecone":
            self.indexes[collection].upsert(
                vectors=[(document_id, embedding, metadata)]
            )

        logger.info(f"Added document {document_id} to {collection}")
        return document_id

    async def search(
        self,
        collection: str,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorDocument]:
        """
        Search for similar documents

        Args:
            collection: Collection to search
            query: Search query
            top_k: Number of results
            filters: Optional metadata filters

        Returns:
            List of similar documents
        """
        # Generate query embedding
        query_embedding = self._generate_embedding(query)

        results = []

        if self.backend == "chroma":
            # ChromaDB search
            search_results = self.collections[collection].query(
                query_embeddings=[query_embedding], n_results=top_k, where=filters
            )

            for i in range(len(search_results["ids"][0])):
                results.append(
                    VectorDocument(
                        id=search_results["ids"][0][i],
                        content=search_results["documents"][0][i],
                        metadata=search_results["metadatas"][0][i],
                        score=1
                        - search_results["distances"][0][
                            i
                        ],  # Convert distance to similarity
                    )
                )

        elif self.backend == "pinecone":
            # Pinecone search
            search_results = self.indexes[collection].query(
                query_embedding, top_k=top_k, include_metadata=True, filter=filters
            )

            for match in search_results["matches"]:
                results.append(
                    VectorDocument(
                        id=match["id"],
                        content=match["metadata"].get("content", ""),
                        metadata=match["metadata"],
                        score=match["score"],
                    )
                )

        return results

    async def find_similar_videos(
        self, video_id: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find videos similar to a given video

        Args:
            video_id: Reference video ID
            top_k: Number of similar videos

        Returns:
            List of similar videos with scores
        """
        # Get reference video embedding
        if self.backend == "chroma":
            ref_result = self.collections["scripts"].get(
                ids=[video_id], include=["embeddings", "metadatas"]
            )

            if not ref_result["ids"]:
                return []

            ref_embedding = ref_result["embeddings"][0]
            ref_metadata = ref_result["metadatas"][0]
        else:
            ref_result = self.indexes["scripts"].fetch([video_id])
            if video_id not in ref_result["vectors"]:
                return []

            ref_embedding = ref_result["vectors"][video_id]["values"]
            ref_metadata = ref_result["vectors"][video_id]["metadata"]

        # Search for similar videos
        similar = await self.search(
            "scripts",
            "",  # Empty query, we'll use the embedding directly
            top_k=top_k + 1,  # +1 to account for self
        )

        # Filter out the reference video
        similar = [v for v in similar if v.id != video_id][:top_k]

        # Format results
        results = []
        for video in similar:
            results.append(
                {
                    "video_id": video.id,
                    "similarity_score": video.score,
                    "title": video.metadata.get("title", "Unknown"),
                    "channel": video.metadata.get("channel", "Unknown"),
                    "views": video.metadata.get("views", 0),
                    "published_at": video.metadata.get("published_at"),
                }
            )

        return results

    async def recommend_content(
        self, user_history: List[str], exclude_ids: List[str] = None, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Recommend content based on user history

        Args:
            user_history: List of video IDs user has watched
            exclude_ids: IDs to exclude from recommendations
            top_k: Number of recommendations

        Returns:
            List of recommended videos
        """
        if not user_history:
            return []

        exclude_ids = exclude_ids or []
        exclude_ids.extend(user_history)

        # Get embeddings for user history
        history_embeddings = []

        for video_id in user_history[-10:]:  # Use last 10 videos
            if self.backend == "chroma":
                result = self.collections["scripts"].get(
                    ids=[video_id], include=["embeddings"]
                )
                if result["ids"]:
                    history_embeddings.append(result["embeddings"][0])
            else:
                result = self.indexes["scripts"].fetch([video_id])
                if video_id in result["vectors"]:
                    history_embeddings.append(result["vectors"][video_id]["values"])

        if not history_embeddings:
            return []

        # Calculate average embedding (user preference vector)
        user_preference = np.mean(history_embeddings, axis=0)

        # Search for similar content
        if self.backend == "chroma":
            results = self.collections["scripts"].query(
                query_embeddings=[user_preference.tolist()],
                n_results=top_k + len(exclude_ids),
            )

            recommendations = []
            for i in range(len(results["ids"][0])):
                if results["ids"][0][i] not in exclude_ids:
                    recommendations.append(
                        {
                            "video_id": results["ids"][0][i],
                            "score": 1 - results["distances"][0][i],
                            "title": results["metadatas"][0][i].get("title", "Unknown"),
                            "channel": results["metadatas"][0][i].get(
                                "channel", "Unknown"
                            ),
                            "reason": "Based on your viewing history",
                        }
                    )

                if len(recommendations) >= top_k:
                    break
        else:
            results = self.indexes["scripts"].query(
                user_preference.tolist(),
                top_k=top_k + len(exclude_ids),
                include_metadata=True,
            )

            recommendations = []
            for match in results["matches"]:
                if match["id"] not in exclude_ids:
                    recommendations.append(
                        {
                            "video_id": match["id"],
                            "score": match["score"],
                            "title": match["metadata"].get("title", "Unknown"),
                            "channel": match["metadata"].get("channel", "Unknown"),
                            "reason": "Based on your viewing history",
                        }
                    )

                if len(recommendations) >= top_k:
                    break

        return recommendations

    async def cluster_topics(
        self, collection: str = "trends", n_clusters: int = 10
    ) -> Dict[str, List[str]]:
        """
        Cluster topics to identify content themes

        Args:
            collection: Collection to cluster
            n_clusters: Number of clusters

        Returns:
            Dictionary of cluster_id to topic list
        """
        from sklearn.cluster import KMeans

        # Get all documents from collection
        if self.backend == "chroma":
            all_docs = self.collections[collection].get(
                include=["embeddings", "documents"]
            )

            if not all_docs["ids"]:
                return {}

            embeddings = np.array(all_docs["embeddings"])
            documents = all_docs["documents"]
        else:
            # For Pinecone, we'd need to implement pagination
            # This is a simplified version
            return {}

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Group topics by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            cluster_id = f"cluster_{label}"
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(documents[i])

        return clusters

    async def deduplicate_content(
        self, collection: str, similarity_threshold: float = 0.95
    ) -> List[Tuple[str, str, float]]:
        """
        Find duplicate or near-duplicate content

        Args:
            collection: Collection to check
            similarity_threshold: Similarity threshold for duplicates

        Returns:
            List of duplicate pairs (id1, id2, similarity)
        """
        duplicates = []

        if self.backend == "chroma":
            # Get all documents
            all_docs = self.collections[collection].get(include=["embeddings", "ids"])

            if len(all_docs["ids"]) < 2:
                return []

            embeddings = np.array(all_docs["embeddings"])
            ids = all_docs["ids"]

            # Calculate pairwise similarities
            similarities = cosine_similarity(embeddings)

            # Find duplicates
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    if similarities[i][j] >= similarity_threshold:
                        duplicates.append((ids[i], ids[j], float(similarities[i][j])))

        return duplicates

    async def update_document(
        self,
        collection: str,
        document_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update document in vector database

        Args:
            collection: Collection name
            document_id: Document ID
            content: New content (optional)
            metadata: New metadata (optional)

        Returns:
            Success status
        """
        try:
            if self.backend == "chroma":
                # Get existing document
                existing = self.collections[collection].get(
                    ids=[document_id], include=["documents", "metadatas"]
                )

                if not existing["ids"]:
                    return False

                # Update content and/or metadata
                new_content = content or existing["documents"][0]
                new_metadata = metadata or existing["metadatas"][0]
                new_metadata["updated_at"] = datetime.utcnow().isoformat()

                # Generate new embedding if content changed
                if content:
                    new_embedding = self._generate_embedding(new_content)
                    self.collections[collection].update(
                        ids=[document_id],
                        embeddings=[new_embedding],
                        documents=[new_content],
                        metadatas=[new_metadata],
                    )
                else:
                    self.collections[collection].update(
                        ids=[document_id], metadatas=[new_metadata]
                    )

            elif self.backend == "pinecone":
                if content:
                    new_embedding = self._generate_embedding(content)
                    metadata = metadata or {}
                    metadata["content"] = content
                    metadata["updated_at"] = datetime.utcnow().isoformat()

                    self.indexes[collection].upsert(
                        vectors=[(document_id, new_embedding, metadata)]
                    )
                elif metadata:
                    # Pinecone doesn't support metadata-only updates
                    # Need to fetch existing and re-upsert
                    existing = self.indexes[collection].fetch([document_id])
                    if document_id in existing["vectors"]:
                        existing_vec = existing["vectors"][document_id]
                        existing_vec["metadata"].update(metadata)
                        existing_vec["metadata"][
                            "updated_at"
                        ] = datetime.utcnow().isoformat()

                        self.indexes[collection].upsert(
                            vectors=[
                                (
                                    document_id,
                                    existing_vec["values"],
                                    existing_vec["metadata"],
                                )
                            ]
                        )

            logger.info(f"Updated document {document_id} in {collection}")
            return True

        except Exception as e:
            logger.error(f"Failed to update document: {e}")
            return False

    async def delete_document(self, collection: str, document_id: str) -> bool:
        """
        Delete document from vector database

        Args:
            collection: Collection name
            document_id: Document ID

        Returns:
            Success status
        """
        try:
            if self.backend == "chroma":
                self.collections[collection].delete(ids=[document_id])
            elif self.backend == "pinecone":
                self.indexes[collection].delete(ids=[document_id])

            logger.info(f"Deleted document {document_id} from {collection}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()

    def _generate_id(self, content: str) -> str:
        """Generate unique ID for content"""
        return hashlib.md5(content.encode()).hexdigest()

    async def get_collection_stats(self, collection: str) -> Dict[str, Any]:
        """
        Get statistics for a collection

        Args:
            collection: Collection name

        Returns:
            Collection statistics
        """
        stats = {"collection": collection, "backend": self.backend}

        if self.backend == "chroma":
            coll = self.collections[collection]
            count = coll.count()
            stats["document_count"] = count
            stats["embedding_dimension"] = self.embedding_dimension

        elif self.backend == "pinecone":
            index_stats = self.indexes[collection].describe_index_stats()
            stats["document_count"] = index_stats["total_vector_count"]
            stats["embedding_dimension"] = index_stats["dimension"]
            stats["index_fullness"] = index_stats["index_fullness"]

        return stats


# Example usage
async def main():
    """Example usage of vector database service"""

    # Initialize service
    vector_db = VectorDatabaseService(backend="chroma")

    # Add some documents
    await vector_db.add_document(
        collection="scripts",
        content="This is a tutorial about Python programming and machine learning basics",
        metadata={
            "title": "Python ML Tutorial",
            "channel": "TechChannel",
            "views": 10000,
            "duration": 600,
        },
    )

    await vector_db.add_document(
        collection="scripts",
        content="Learn how to build web applications with React and Node.js",
        metadata={
            "title": "Full Stack Web Development",
            "channel": "WebDev Pro",
            "views": 15000,
            "duration": 900,
        },
    )

    # Search for similar content
    results = await vector_db.search(
        collection="scripts", query="programming tutorials for beginners", top_k=5
    )

    print("Search Results:")
    for doc in results:
        print(f"  - {doc.metadata.get('title')}: Score {doc.score:.3f}")

    # Get collection stats
    stats = await vector_db.get_collection_stats("scripts")
    print(f"\nCollection Stats: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
