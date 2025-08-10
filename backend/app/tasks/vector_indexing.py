"""
Vector Indexing Tasks
Owner: VP of AI

Celery tasks for automatic content indexing in vector database.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from app.core.celery_app import celery_app
from app.core.database import AsyncSessionLocal
from app.models.video import Video, VideoStatus
from app.services.vector_service import vector_service

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 60})
def index_video_content(self, video_id: str) -> Dict[str, Any]:
    """
    Index video content for semantic search.
    
    Args:
        video_id: Video identifier
        
    Returns:
        Indexing result
    """
    try:
        logger.info(f"Starting content indexing for video: {video_id}")
        
        # Get video from database
        async def get_video_data():
            async with AsyncSessionLocal() as db:
                video = await db.get(Video, video_id)
                if not video:
                    raise ValueError(f"Video not found: {video_id}")
                
                return {
                    "id": video.id,
                    "title": video.title or "",
                    "description": video.description or "",
                    "channel_id": video.channel_id,
                    "status": video.status.value,
                    "created_at": video.created_at.isoformat() if video.created_at else None,
                    "content_settings": video.content_settings or {}
                }
        
        # Run async function
        import asyncio
        video_data = asyncio.run(get_video_data())
        
        # Prepare content for indexing
        content_text = f"{video_data['title']} {video_data['description']}"
        
        # Add context from content settings
        if video_data.get("content_settings"):
            topic = video_data["content_settings"].get("topic", "")
            keywords = video_data["content_settings"].get("keywords", [])
            style = video_data["content_settings"].get("style", "")
            
            content_text += f" {topic} {' '.join(keywords)} {style}"
        
        metadata = {
            "video_id": video_data["id"],
            "title": video_data["title"],
            "channel_id": video_data["channel_id"],
            "status": video_data["status"],
            "created_at": video_data["created_at"],
            "indexed_at": datetime.utcnow().isoformat()
        }
        
        # Store in vector database
        async def store_embedding():
            return await vector_service.store_content_embedding(
                content_id=video_id,
                content_type="video",
                content=content_text,
                metadata=metadata
            )
        
        success = asyncio.run(store_embedding())
        
        result = {
            "video_id": video_id,
            "indexed": success,
            "content_length": len(content_text),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if success:
            logger.info(f"Successfully indexed video content: {video_id}")
        else:
            logger.error(f"Failed to index video content: {video_id}")
        
        return result
        
    except Exception as e:
        logger.error(f"Video content indexing failed for {video_id}: {str(e)}")
        # Retry with exponential backoff
        raise self.retry(countdown=60 * (self.request.retries + 1))


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 2, 'countdown': 120})
def bulk_index_videos(self, video_ids: list) -> Dict[str, Any]:
    """
    Index multiple videos in bulk.
    
    Args:
        video_ids: List of video identifiers
        
    Returns:
        Bulk indexing results
    """
    try:
        logger.info(f"Starting bulk indexing for {len(video_ids)} videos")
        
        results = {
            "total_videos": len(video_ids),
            "successful": 0,
            "failed": 0,
            "errors": []
        }
        
        for video_id in video_ids:
            try:
                result = index_video_content.delay(video_id)
                # Wait for result with timeout
                indexing_result = result.get(timeout=300)
                
                if indexing_result.get("indexed"):
                    results["successful"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append(f"Failed to index {video_id}")
                    
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Error indexing {video_id}: {str(e)}")
                logger.error(f"Bulk indexing error for {video_id}: {str(e)}")
        
        logger.info(f"Bulk indexing completed: {results['successful']} successful, {results['failed']} failed")
        return results
        
    except Exception as e:
        logger.error(f"Bulk video indexing failed: {str(e)}")
        raise


@celery_app.task(bind=True)
def remove_video_from_index(self, video_id: str) -> Dict[str, Any]:
    """
    Remove video content from vector index.
    
    Args:
        video_id: Video identifier
        
    Returns:
        Removal result
    """
    try:
        logger.info(f"Removing video from index: {video_id}")
        
        async def remove_embedding():
            return await vector_service.delete_content_embedding(
                content_id=video_id,
                content_type="video"
            )
        
        import asyncio
        success = asyncio.run(remove_embedding())
        
        result = {
            "video_id": video_id,
            "removed": success,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if success:
            logger.info(f"Successfully removed video from index: {video_id}")
        else:
            logger.warning(f"Video may not have been in index: {video_id}")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to remove video from index {video_id}: {str(e)}")
        return {
            "video_id": video_id,
            "removed": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@celery_app.task(bind=True)
def reindex_all_videos(self, user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Reindex all videos or videos for a specific user.
    
    Args:
        user_id: Optional user ID to limit reindexing
        
    Returns:
        Reindexing results
    """
    try:
        logger.info(f"Starting full reindex for user: {user_id or 'all users'}")
        
        async def get_video_ids():
            async with AsyncSessionLocal() as db:
                query = db.query(Video.id).filter(Video.status == VideoStatus.COMPLETED)
                if user_id:
                    query = query.filter(Video.user_id == user_id)
                
                results = await query.all()
                return [result.id for result in results]
        
        import asyncio
        video_ids = asyncio.run(get_video_ids())
        
        if not video_ids:
            return {
                "message": "No completed videos found for reindexing",
                "total_videos": 0,
                "user_id": user_id
            }
        
        # Start bulk indexing
        bulk_result = bulk_index_videos.delay(video_ids)
        
        return {
            "message": "Reindexing started",
            "total_videos": len(video_ids),
            "task_id": bulk_result.id,
            "user_id": user_id,
            "started_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to start reindexing: {str(e)}")
        return {
            "message": "Reindexing failed to start",
            "error": str(e),
            "user_id": user_id
        }


@celery_app.task(bind=True)
def update_video_embeddings_on_completion(self, video_id: str) -> Dict[str, Any]:
    """
    Update video embeddings when video generation is completed.
    This is triggered by the video pipeline completion.
    
    Args:
        video_id: Video identifier
        
    Returns:
        Update result
    """
    try:
        logger.info(f"Updating embeddings for completed video: {video_id}")
        
        # Add delay to ensure video content is fully available
        import time
        time.sleep(30)  # Wait 30 seconds
        
        # Index the completed video
        result = index_video_content.delay(video_id)
        indexing_result = result.get(timeout=300)
        
        return {
            "video_id": video_id,
            "embeddings_updated": indexing_result.get("indexed", False),
            "trigger": "video_completion",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to update embeddings for completed video {video_id}: {str(e)}")
        return {
            "video_id": video_id,
            "embeddings_updated": False,
            "error": str(e),
            "trigger": "video_completion"
        }