"""
Vector Database Setup and Management
Owner: Data Engineer

Vector database implementation for semantic search and content similarity.
Supports both Pinecone and Weaviate as vector database backends.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime
import json
import os
from dataclasses import dataclass
from enum import Enum

# Vector database imports
try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    import weaviate
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False

from sentence_transformers import SentenceTransformer
from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorDBType(Enum):
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    CHROMA = "chroma"


@dataclass
class VectorDocument:
    """Represents a document in the vector database."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class SearchResult:
    """Represents a search result from vector database."""
    document_id: str
    score: float
    content: str
    metadata: Dict[str, Any]


class VectorDatabase:
    """Abstract base class for vector database operations."""
    
    def __init__(self, index_name: str, dimension: int = 384):
        self.index_name = index_name
        self.dimension = dimension
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    async def create_index(self, **kwargs) -> bool:
        """Create vector index."""
        raise NotImplementedError
    
    async def insert_documents(self, documents: List[VectorDocument]) -> bool:
        """Insert documents into the vector database."""
        raise NotImplementedError
    
    async def search(self, query: str, top_k: int = 10, filter_dict: Optional[Dict] = None) -> List[SearchResult]:
        """Search for similar documents."""
        raise NotImplementedError
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents by IDs."""
        raise NotImplementedError
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        raise NotImplementedError
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        return self.embedding_model.encode(text).tolist()


class PineconeVectorDB(VectorDatabase):
    """Pinecone vector database implementation."""
    
    def __init__(self, index_name: str, dimension: int = 384):
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone library not installed")
        
        super().__init__(index_name, dimension)
        
        # Initialize Pinecone
        pinecone.init(
            api_key=settings.PINECONE_API_KEY,
            environment=settings.PINECONE_ENVIRONMENT
        )
        
        self.index = None
        
    async def create_index(self, metric: str = "cosine", **kwargs) -> bool:
        """Create Pinecone index."""
        try:
            # Check if index already exists
            if self.index_name in pinecone.list_indexes():
                logger.info(f"Index {self.index_name} already exists")
                self.index = pinecone.Index(self.index_name)
                return True
            
            # Create new index
            pinecone.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=metric,
                pod_type=kwargs.get('pod_type', 'p1.x1'),
                replicas=kwargs.get('replicas', 1)
            )
            
            self.index = pinecone.Index(self.index_name)
            logger.info(f"Created Pinecone index: {self.index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create Pinecone index: {str(e)}")
            return False
    
    async def insert_documents(self, documents: List[VectorDocument]) -> bool:
        """Insert documents into Pinecone."""
        try:
            if not self.index:
                await self.create_index()
            
            vectors = []
            for doc in documents:
                if not doc.embedding:
                    doc.embedding = self.generate_embedding(doc.content)
                
                vectors.append((
                    doc.id,
                    doc.embedding,
                    {
                        'content': doc.content,
                        **doc.metadata
                    }
                ))
            
            # Batch insert
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            logger.info(f"Inserted {len(documents)} documents into Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert documents into Pinecone: {str(e)}")
            return False
    
    async def search(self, query: str, top_k: int = 10, filter_dict: Optional[Dict] = None) -> List[SearchResult]:
        """Search Pinecone for similar documents."""
        try:
            if not self.index:
                await self.create_index()
            
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            
            # Search
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict or {}
            )
            
            results = []
            for match in search_results['matches']:
                results.append(SearchResult(
                    document_id=match['id'],
                    score=match['score'],
                    content=match['metadata'].get('content', ''),
                    metadata={k: v for k, v in match['metadata'].items() if k != 'content'}
                ))
            
            logger.info(f"Found {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Pinecone search failed: {str(e)}")
            return []
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from Pinecone."""
        try:
            if not self.index:
                return False
            
            self.index.delete(ids=document_ids)
            logger.info(f"Deleted {len(document_ids)} documents from Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents from Pinecone: {str(e)}")
            return False
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics."""
        try:
            if not self.index:
                return {}
            
            stats = self.index.describe_index_stats()
            return {
                'total_vector_count': stats['total_vector_count'],
                'dimension': stats['dimension'],
                'index_fullness': stats['index_fullness'],
                'namespaces': stats.get('namespaces', {})
            }
            
        except Exception as e:
            logger.error(f"Failed to get Pinecone stats: {str(e)}")
            return {}


class WeaviateVectorDB(VectorDatabase):
    """Weaviate vector database implementation."""
    
    def __init__(self, index_name: str, dimension: int = 384, url: str = "http://localhost:8080"):
        if not WEAVIATE_AVAILABLE:
            raise ImportError("Weaviate library not installed")
        
        super().__init__(index_name, dimension)
        
        # Initialize Weaviate client
        self.client = weaviate.Client(url=url)
        self.class_name = index_name.capitalize()
        
    async def create_index(self, **kwargs) -> bool:
        """Create Weaviate class (schema)."""
        try:
            # Check if class already exists
            if self.client.schema.exists(self.class_name):
                logger.info(f"Weaviate class {self.class_name} already exists")
                return True
            
            # Define class schema
            class_schema = {
                "class": self.class_name,
                "description": f"YTEmpire {self.index_name} documents",
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "Document content"
                    },
                    {
                        "name": "document_type",
                        "dataType": ["string"],
                        "description": "Type of document"
                    },
                    {
                        "name": "created_at",
                        "dataType": ["date"],
                        "description": "Creation timestamp"
                    },
                    {
                        "name": "metadata",
                        "dataType": ["object"],
                        "description": "Additional metadata"
                    }
                ],
                "vectorizer": "none",  # We'll provide our own vectors
            }
            
            self.client.schema.create_class(class_schema)
            logger.info(f"Created Weaviate class: {self.class_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create Weaviate class: {str(e)}")
            return False
    
    async def insert_documents(self, documents: List[VectorDocument]) -> bool:
        """Insert documents into Weaviate."""
        try:
            if not self.client.schema.exists(self.class_name):
                await self.create_index()
            
            # Batch insert
            with self.client.batch as batch:
                batch.batch_size = 100
                
                for doc in documents:
                    if not doc.embedding:
                        doc.embedding = self.generate_embedding(doc.content)
                    
                    properties = {
                        "content": doc.content,
                        "document_type": doc.metadata.get('document_type', 'unknown'),
                        "created_at": doc.metadata.get('created_at', datetime.now().isoformat()),
                        "metadata": doc.metadata
                    }
                    
                    batch.add_data_object(
                        data_object=properties,
                        class_name=self.class_name,
                        uuid=doc.id,
                        vector=doc.embedding
                    )
            
            logger.info(f"Inserted {len(documents)} documents into Weaviate")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert documents into Weaviate: {str(e)}")
            return False
    
    async def search(self, query: str, top_k: int = 10, filter_dict: Optional[Dict] = None) -> List[SearchResult]:
        """Search Weaviate for similar documents."""
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            
            # Build GraphQL query
            near_vector = {
                "vector": query_embedding,
                "certainty": 0.7  # Minimum similarity threshold
            }
            
            query_builder = (
                self.client.query
                .get(self.class_name, ["content", "metadata"])
                .with_near_vector(near_vector)
                .with_limit(top_k)
                .with_additional(["certainty", "id"])
            )
            
            # Add filters if provided
            if filter_dict:
                where_filter = {
                    "operator": "And",
                    "operands": []
                }
                
                for key, value in filter_dict.items():
                    where_filter["operands"].append({
                        "path": [key],
                        "operator": "Equal",
                        "valueText": str(value)
                    })
                
                query_builder = query_builder.with_where(where_filter)
            
            # Execute query
            response = query_builder.do()
            
            results = []
            if "data" in response and "Get" in response["data"]:
                for item in response["data"]["Get"][self.class_name]:
                    results.append(SearchResult(
                        document_id=item["_additional"]["id"],
                        score=item["_additional"]["certainty"],
                        content=item["content"],
                        metadata=item.get("metadata", {})
                    ))
            
            logger.info(f"Found {len(results)} results in Weaviate for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Weaviate search failed: {str(e)}")
            return []
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from Weaviate."""
        try:
            for doc_id in document_ids:
                self.client.data_object.delete(
                    uuid=doc_id,
                    class_name=self.class_name
                )
            
            logger.info(f"Deleted {len(document_ids)} documents from Weaviate")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents from Weaviate: {str(e)}")
            return False
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get Weaviate class statistics."""
        try:
            # Get object count
            response = (
                self.client.query
                .aggregate(self.class_name)
                .with_meta_count()
                .do()
            )
            
            count = 0
            if "data" in response and "Aggregate" in response["data"]:
                count = response["data"]["Aggregate"][self.class_name][0]["meta"]["count"]
            
            return {
                'total_objects': count,
                'class_name': self.class_name,
                'dimension': self.dimension
            }
            
        except Exception as e:
            logger.error(f"Failed to get Weaviate stats: {str(e)}")
            return {}


class VectorDBManager:
    """Manager class for vector database operations."""
    
    def __init__(self, db_type: VectorDBType = VectorDBType.PINECONE):
        self.db_type = db_type
        self.databases: Dict[str, VectorDatabase] = {}
        
    def get_database(self, index_name: str, dimension: int = 384) -> VectorDatabase:
        """Get or create vector database instance."""
        key = f"{self.db_type.value}_{index_name}"
        
        if key not in self.databases:
            if self.db_type == VectorDBType.PINECONE:
                self.databases[key] = PineconeVectorDB(index_name, dimension)
            elif self.db_type == VectorDBType.WEAVIATE:
                self.databases[key] = WeaviateVectorDB(index_name, dimension)
            else:
                raise ValueError(f"Unsupported vector DB type: {self.db_type}")
        
        return self.databases[key]
    
    async def setup_ytempire_indexes(self) -> Dict[str, bool]:
        """Set up all YTEmpire vector indexes."""
        indexes = {
            'video_content': 384,  # For video scripts and descriptions
            'channel_profiles': 384,  # For channel information
            'trending_topics': 384,  # For trend analysis
            'user_preferences': 384   # For personalization
        }
        
        results = {}
        
        for index_name, dimension in indexes.items():
            try:
                db = self.get_database(index_name, dimension)
                success = await db.create_index()
                results[index_name] = success
                
                if success:
                    logger.info(f"Successfully set up {index_name} index")
                else:
                    logger.error(f"Failed to set up {index_name} index")
                    
            except Exception as e:
                logger.error(f"Error setting up {index_name}: {str(e)}")
                results[index_name] = False
        
        return results
    
    async def index_video_content(self, video_data: List[Dict[str, Any]]) -> bool:
        """Index video content for semantic search."""
        try:
            db = self.get_database('video_content')
            
            documents = []
            for video in video_data:
                # Combine title, description, and script for comprehensive content search
                content_parts = []
                if video.get('title'):
                    content_parts.append(f"Title: {video['title']}")
                if video.get('description'):
                    content_parts.append(f"Description: {video['description']}")
                if video.get('script'):
                    content_parts.append(f"Script: {video['script']}")
                
                content = "\n".join(content_parts)
                
                doc = VectorDocument(
                    id=video['id'],
                    content=content,
                    metadata={
                        'video_id': video['id'],
                        'channel_id': video.get('channel_id'),
                        'title': video.get('title', ''),
                        'category': video.get('category', ''),
                        'created_at': video.get('created_at', datetime.now().isoformat()),
                        'status': video.get('status', 'unknown'),
                        'document_type': 'video_content'
                    }
                )
                documents.append(doc)
            
            success = await db.insert_documents(documents)
            
            if success:
                logger.info(f"Successfully indexed {len(documents)} videos")
            else:
                logger.error("Failed to index video content")
            
            return success
            
        except Exception as e:
            logger.error(f"Video content indexing failed: {str(e)}")
            return False
    
    async def search_similar_videos(self, query: str, top_k: int = 10, filters: Optional[Dict] = None) -> List[SearchResult]:
        """Search for similar videos based on content."""
        try:
            db = self.get_database('video_content')
            results = await db.search(query, top_k, filters)
            
            logger.info(f"Found {len(results)} similar videos for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Similar video search failed: {str(e)}")
            return []
    
    async def get_content_recommendations(self, video_id: str, top_k: int = 5) -> List[SearchResult]:
        """Get content recommendations based on a video."""
        try:
            # First, get the video content
            db = self.get_database('video_content')
            
            # This would typically fetch the video content from database
            # For now, we'll use a placeholder
            video_content = f"Get recommendations for video {video_id}"
            
            # Search for similar content
            results = await db.search(video_content, top_k + 1)  # +1 to exclude self
            
            # Filter out the original video
            filtered_results = [r for r in results if r.document_id != video_id][:top_k]
            
            logger.info(f"Generated {len(filtered_results)} recommendations for video {video_id}")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Content recommendations failed: {str(e)}")
            return []


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_vector_db():
        """Test vector database functionality."""
        
        # Initialize manager
        manager = VectorDBManager(VectorDBType.WEAVIATE)  # or PINECONE
        
        # Set up indexes
        setup_results = await manager.setup_ytempire_indexes()
        print("Setup results:", setup_results)
        
        # Sample video data
        sample_videos = [
            {
                'id': 'video_001',
                'title': 'Introduction to AI and Machine Learning',
                'description': 'Learn the basics of artificial intelligence and machine learning',
                'script': 'Today we will explore the fundamental concepts of AI...',
                'channel_id': 'channel_tech',
                'category': 'education'
            },
            {
                'id': 'video_002', 
                'title': 'Advanced Python Programming',
                'description': 'Master advanced Python concepts and techniques',
                'script': 'In this tutorial, we will dive deep into Python...',
                'channel_id': 'channel_tech',
                'category': 'programming'
            }
        ]
        
        # Index video content
        index_success = await manager.index_video_content(sample_videos)
        print(f"Indexing success: {index_success}")
        
        if index_success:
            # Search for similar videos
            search_results = await manager.search_similar_videos("python programming tutorial")
            print(f"Search results: {len(search_results)}")
            
            for result in search_results:
                print(f"  - {result.document_id}: {result.score:.3f}")
            
            # Get recommendations
            recommendations = await manager.get_content_recommendations('video_001')
            print(f"Recommendations: {len(recommendations)}")
    
    # Run test
    asyncio.run(test_vector_db())