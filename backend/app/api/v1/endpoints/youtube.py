"""
YouTube API Endpoints
Owner: Integration Specialist
"""

from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Query, Body
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, List, Optional, Any
import tempfile
import os

from app.core.database import get_db
from app.models.user import User
from app.api.v1.endpoints.auth import get_current_user
from app.services.youtube_service import get_youtube_service, YouTubeService
from app.schemas.youtube import (
    VideoUploadRequest,
    VideoUploadResponse,
    VideoStatsResponse,
    ChannelInfoResponse,
    VideoUpdateRequest,
    YouTubeOAuthResponse
)

router = APIRouter()


@router.get("/oauth/start", response_model=YouTubeOAuthResponse)
async def start_youtube_oauth(
    current_user: User = Depends(get_current_user)
):
    """Start YouTube OAuth authorization flow."""
    try:
        from app.services.youtube_service import YouTubeService
        service = YouTubeService()
        auth_url = await service.start_oauth_flow(current_user.id)
        
        if not auth_url:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to start OAuth flow"
            )
        
        return YouTubeOAuthResponse(
            authorization_url=auth_url,
            state=current_user.id
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OAuth initialization failed: {str(e)}"
        )


@router.get("/oauth/callback")
async def youtube_oauth_callback(
    code: str = Query(...),
    state: str = Query(...),  # This contains user_id
    db: AsyncSession = Depends(get_db)
):
    """Handle YouTube OAuth callback."""
    try:
        from app.services.youtube_service import YouTubeService
        service = YouTubeService()
        
        success = await service.handle_oauth_callback(state, code)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="OAuth authorization failed"
            )
        
        return {
            "message": "YouTube authorization successful",
            "user_id": state,
            "status": "authorized"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OAuth callback failed: {str(e)}"
        )


@router.post("/upload", response_model=VideoUploadResponse)
async def upload_video(
    video_file: UploadFile = File(...),
    title: str = Body(...),
    description: str = Body(...),
    tags: str = Body(default=""),
    category_id: str = Body(default="22"),
    privacy_status: str = Body(default="private"),
    thumbnail_file: Optional[UploadFile] = File(default=None),
    current_user: User = Depends(get_current_user)
):
    """Upload video to YouTube."""
    
    # Get YouTube service
    try:
        youtube_service = await get_youtube_service(current_user.id)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="YouTube not authorized. Please authorize first."
        )
    
    # Validate file
    if not video_file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid video file format"
        )
    
    # Save files temporarily
    video_temp_path = None
    thumbnail_temp_path = None
    
    try:
        # Save video file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            content = await video_file.read()
            temp_video.write(content)
            video_temp_path = temp_video.name
        
        # Save thumbnail if provided
        if thumbnail_file:
            if not thumbnail_file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid thumbnail format. Use JPG or PNG."
                )
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_thumb:
                thumb_content = await thumbnail_file.read()
                temp_thumb.write(thumb_content)
                thumbnail_temp_path = temp_thumb.name
        
        # Parse tags
        tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()] if tags else []
        
        # Upload to YouTube
        result = await youtube_service.upload_video(
            video_file_path=video_temp_path,
            title=title,
            description=description,
            tags=tag_list,
            category_id=category_id,
            privacy_status=privacy_status,
            thumbnail_path=thumbnail_temp_path
        )
        
        return VideoUploadResponse(
            video_id=result['video_id'],
            url=result['url'],
            status=result['status'],
            title=title,
            privacy_status=result['privacy_status']
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )
    
    finally:
        # Cleanup temporary files
        if video_temp_path and os.path.exists(video_temp_path):
            os.unlink(video_temp_path)
        if thumbnail_temp_path and os.path.exists(thumbnail_temp_path):
            os.unlink(thumbnail_temp_path)


@router.get("/video/{video_id}/stats", response_model=VideoStatsResponse)
async def get_video_stats(
    video_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get video statistics from YouTube."""
    try:
        youtube_service = await get_youtube_service(current_user.id)
        stats = await youtube_service.get_video_stats(video_id)
        
        if not stats:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found or not accessible"
            )
        
        return VideoStatsResponse(**stats)
        
    except Exception as e:
        if "not authorized" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="YouTube not authorized"
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get video stats: {str(e)}"
        )


@router.get("/channel/info", response_model=ChannelInfoResponse)
async def get_channel_info(
    current_user: User = Depends(get_current_user)
):
    """Get YouTube channel information."""
    try:
        youtube_service = await get_youtube_service(current_user.id)
        info = await youtube_service.get_channel_info()
        
        if not info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Channel information not available"
            )
        
        return ChannelInfoResponse(**info)
        
    except Exception as e:
        if "not authorized" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="YouTube not authorized"
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get channel info: {str(e)}"
        )


@router.put("/video/{video_id}")
async def update_video(
    video_id: str,
    update_data: VideoUpdateRequest,
    current_user: User = Depends(get_current_user)
):
    """Update video metadata on YouTube."""
    try:
        youtube_service = await get_youtube_service(current_user.id)
        
        success = await youtube_service.update_video(
            video_id=video_id,
            title=update_data.title,
            description=update_data.description,
            tags=update_data.tags,
            privacy_status=update_data.privacy_status
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to update video"
            )
        
        return {"message": "Video updated successfully", "video_id": video_id}
        
    except Exception as e:
        if "not authorized" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="YouTube not authorized"
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update video: {str(e)}"
        )


@router.delete("/video/{video_id}")
async def delete_video(
    video_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete video from YouTube."""
    try:
        youtube_service = await get_youtube_service(current_user.id)
        
        success = await youtube_service.delete_video(video_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to delete video"
            )
        
        return {"message": "Video deleted successfully", "video_id": video_id}
        
    except Exception as e:
        if "not authorized" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="YouTube not authorized"
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete video: {str(e)}"
        )


@router.get("/search")
async def search_videos(
    q: str = Query(..., description="Search query"),
    max_results: int = Query(default=10, le=50, description="Maximum results"),
    current_user: User = Depends(get_current_user)
):
    """Search YouTube videos."""
    try:
        youtube_service = await get_youtube_service(current_user.id)
        results = await youtube_service.search_videos(q, max_results)
        
        return {
            "query": q,
            "results": results,
            "total_results": len(results)
        }
        
    except Exception as e:
        if "not authorized" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="YouTube not authorized"
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/quota/usage")
async def get_quota_usage(
    current_user: User = Depends(get_current_user)
):
    """Get YouTube API quota usage."""
    try:
        youtube_service = await get_youtube_service(current_user.id)
        usage = await youtube_service.get_quota_usage()
        
        return usage
        
    except Exception as e:
        if "not authorized" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="YouTube not authorized"
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get quota usage: {str(e)}"
        )


@router.post("/webhook/upload-complete")
async def upload_complete_webhook(
    data: Dict[str, Any] = Body(...),
    # Add webhook authentication here in production
):
    """Handle YouTube upload complete webhook."""
    try:
        # Process upload completion
        video_id = data.get('video_id')
        status = data.get('status')
        
        if not video_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing video_id"
            )
        
        # Update video status in database
        # This would typically trigger further processing
        
        return {
            "message": "Webhook processed successfully",
            "video_id": video_id,
            "status": status
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Webhook processing failed: {str(e)}"
        )