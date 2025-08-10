"""
Search and Recommendations Endpoints
Owner: VP of AI

Semantic search, content recommendations, and trending topics endpoints.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.core.database import get_db
from app.api.v1.endpoints.auth import get_current_user
from app.models.user import User
from app.services.vector_service import vector_service
from app.repositories.video_repository import VideoRepository

router = APIRouter()


class SearchRequest(BaseModel):
    """Search request model."""
    query: str
    content_type: str = "video"
    limit: int = 10
    min_score: Optional[float] = None


class RecommendationResponse(BaseModel):
    """Recommendation response model."""
    id: str
    score: float
    reason: str
    title: str
    description: str
    metadata: Dict[str, Any]


class TrendingTopicResponse(BaseModel):
    """Trending topic response model."""
    topic: str
    frequency: int
    growth_rate: float


class SearchResponse(BaseModel):
    """Search response model."""
    results: List[Dict[str, Any]]
    total_count: int
    query: str
    search_time_ms: int


def get_video_repo(db: AsyncSession = Depends(get_db)) -> VideoRepository:
    """Get video repository instance."""
    return VideoRepository(db)


@router.post("/semantic-search", response_model=SearchResponse)
async def semantic_search(
    search_request: SearchRequest,
    current_user: User = Depends(get_current_user),
    video_repo: VideoRepository = Depends(get_video_repo)
):
    """
    Perform semantic search across content using vector embeddings.
    """
    try:
        import time
        start_time = time.time()
        
        # Perform vector search
        similar_content = await vector_service.find_similar_content(
            query_text=search_request.query,
            content_type=search_request.content_type,
            limit=search_request.limit,
            min_score=search_request.min_score
        )
        
        # Enrich results with additional metadata from database
        enriched_results = []
        for item in similar_content:
            # Get full video details if it's a video search
            if search_request.content_type == "video" and item.get("metadata", {}).get("video_id"):
                video_id = item["metadata"]["video_id"]
                video = await video_repo.get_by_id(video_id)
                
                if video and await video_repo.check_ownership(video_id, current_user.id):
                    enriched_results.append({
                        **item,
                        "title": video.title,
                        "description": video.description,
                        "status": video.status.value,
                        "created_at": video.created_at.isoformat() if video.created_at else None
                    })
            else:
                enriched_results.append(item)
        
        search_time_ms = int((time.time() - start_time) * 1000)
        
        return SearchResponse(
            results=enriched_results,
            total_count=len(enriched_results),
            query=search_request.query,
            search_time_ms=search_time_ms
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/recommendations", response_model=List[RecommendationResponse])
async def get_personalized_recommendations(
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    video_repo: VideoRepository = Depends(get_video_repo)
):
    """
    Get personalized content recommendations based on user history.
    """
    try:
        # Get user's video history
        user_videos = await video_repo.get_by_user_id(current_user.id, limit=50)
        
        # Extract content for user profiling
        content_history = []
        for video in user_videos:
            if video.title:
                content_history.append(video.title)
            if video.description:
                content_history.append(video.description[:200])
        
        # Get recommendations
        recommendations = await vector_service.get_content_recommendations(
            user_id=current_user.id,
            content_history=content_history,
            limit=limit
        )
        
        # Format recommendations
        formatted_recommendations = []
        for rec in recommendations:
            formatted_recommendations.append(RecommendationResponse(
                id=rec["id"],
                score=rec["score"],
                reason=rec["reason"],
                title=rec.get("metadata", {}).get("title", "Recommended Content"),
                description=rec.get("content", "")[:200],
                metadata=rec.get("metadata", {})
            ))
        
        return formatted_recommendations
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get recommendations: {str(e)}"
        )


@router.get("/trending-topics", response_model=List[TrendingTopicResponse])
async def get_trending_topics(
    time_window_days: int = Query(7, ge=1, le=30),
    limit: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_current_user)
):
    """
    Get trending topics based on recent content activity.
    """
    try:
        trending_topics = await vector_service.find_trending_topics(
            time_window_days=time_window_days
        )
        
        # Format and limit results
        formatted_topics = []
        for topic in trending_topics[:limit]:
            formatted_topics.append(TrendingTopicResponse(
                topic=topic["topic"],
                frequency=topic["frequency"],
                growth_rate=topic["growth_rate"]
            ))
        
        return formatted_topics
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get trending topics: {str(e)}"
        )


@router.get("/similar-videos/{video_id}")
async def get_similar_videos(
    video_id: str,
    limit: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_current_user),
    video_repo: VideoRepository = Depends(get_video_repo)
):
    """
    Find videos similar to the specified video.
    """
    try:
        # Verify video ownership
        if not await video_repo.check_ownership(video_id, current_user.id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found or access denied"
            )
        
        # Get video details
        video = await video_repo.get_by_id(video_id)
        if not video:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found"
            )
        
        # Create search query from video content
        search_text = f"{video.title or ''} {video.description or ''}"
        
        # Find similar content
        similar_videos = await vector_service.find_similar_content(
            query_text=search_text,
            content_type="video",
            limit=limit + 1  # +1 to exclude the original video
        )
        
        # Filter out the original video and enrich results
        enriched_results = []
        for item in similar_videos:
            if item["id"] != video_id:
                # Get full video details
                similar_video_id = item.get("metadata", {}).get("video_id", item["id"])
                similar_video = await video_repo.get_by_id(similar_video_id)
                
                if similar_video and await video_repo.check_ownership(similar_video_id, current_user.id):
                    enriched_results.append({
                        **item,
                        "title": similar_video.title,
                        "description": similar_video.description,
                        "status": similar_video.status.value,
                        "created_at": similar_video.created_at.isoformat() if similar_video.created_at else None
                    })
                
                if len(enriched_results) >= limit:
                    break
        
        return {
            "original_video_id": video_id,
            "similar_videos": enriched_results,
            "total_found": len(enriched_results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to find similar videos: {str(e)}"
        )


@router.post("/index-content/{content_id}")
async def index_content(
    content_id: str,
    content_type: str = Query("video", regex="^(video|script|channel|keyword)$"),
    current_user: User = Depends(get_current_user),
    video_repo: VideoRepository = Depends(get_video_repo)
):
    """
    Index content for semantic search (admin/system endpoint).
    """
    try:
        # Get content based on type
        content_text = ""
        metadata = {}
        
        if content_type == "video":
            # Verify ownership
            if not await video_repo.check_ownership(content_id, current_user.id):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Video not found or access denied"
                )
            
            video = await video_repo.get_by_id(content_id)
            if video:
                content_text = f"{video.title or ''} {video.description or ''}"
                metadata = {
                    "video_id": video.id,
                    "title": video.title,
                    "channel_id": video.channel_id,
                    "status": video.status.value,
                    "created_at": video.created_at.isoformat() if video.created_at else None
                }
        
        if not content_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No content found to index"
            )
        
        # Store in vector database
        success = await vector_service.store_content_embedding(
            content_id=content_id,
            content_type=content_type,
            content=content_text,
            metadata=metadata
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to index content"
            )
        
        return {
            "content_id": content_id,
            "content_type": content_type,
            "indexed": True,
            "message": f"{content_type.title()} indexed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index content: {str(e)}"
        )


@router.get("/vector-stats")
async def get_vector_database_stats(
    current_user: User = Depends(get_current_user)
):
    """
    Get vector database statistics (admin endpoint).
    """
    try:
        stats = await vector_service.get_collection_stats()
        
        return {
            "collections": stats,
            "total_embeddings": sum(
                collection.get("total_points", 0) 
                for collection in stats.values() 
                if isinstance(collection, dict) and "total_points" in collection
            ),
            "embedding_dimension": next(
                (collection.get("vector_dimension") 
                 for collection in stats.values() 
                 if isinstance(collection, dict) and "vector_dimension" in collection),
                None
            ),
            "status": "healthy" if stats else "error"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get vector database stats: {str(e)}"
        )


@router.delete("/content/{content_id}")
async def remove_content_from_index(
    content_id: str,
    content_type: str = Query("video", regex="^(video|script|channel|keyword)$"),
    current_user: User = Depends(get_current_user),
    video_repo: VideoRepository = Depends(get_video_repo)
):
    """
    Remove content from vector search index.
    """
    try:
        # Verify ownership for videos
        if content_type == "video":
            if not await video_repo.check_ownership(content_id, current_user.id):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Video not found or access denied"
                )
        
        # Remove from vector database
        success = await vector_service.delete_content_embedding(
            content_id=content_id,
            content_type=content_type
        )
        
        return {
            "content_id": content_id,
            "content_type": content_type,
            "removed": success,
            "message": f"{content_type.title()} removed from index" if success else "Failed to remove from index"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove content from index: {str(e)}"
        )