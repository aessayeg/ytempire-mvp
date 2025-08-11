"""
YouTube Multi-Account Management API Endpoints
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from datetime import datetime

from app.db.session import get_db
from app.api.v1.endpoints.auth import get_current_verified_user
from app.models.user import User
from app.services.youtube_multi_account import (
    MultiAccountYouTubeService,
    YouTubeServiceWrapper,
    AccountStatus
)
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Global service instance
youtube_wrapper = YouTubeServiceWrapper()


class YouTubeAccountInfo(BaseModel):
    """YouTube account information"""
    account_id: str
    email: str
    status: str
    channel_id: Optional[str] = None
    quota_used: int
    quota_limit: int
    quota_percentage: float
    health_score: float
    error_count: int
    last_used: Optional[datetime] = None


class YouTubeAccountStats(BaseModel):
    """YouTube accounts statistics"""
    total_accounts: int
    active_accounts: int
    total_quota_used: int
    total_quota_limit: int
    quota_usage_percentage: float
    accounts: List[Dict[str, Any]]


class YouTubeAuthRequest(BaseModel):
    """Request to authenticate a YouTube account"""
    account_id: str
    client_secrets_path: Optional[str] = None


class YouTubeSearchRequest(BaseModel):
    """YouTube search request"""
    query: str
    max_results: int = Field(default=25, le=50)
    channel_id: Optional[str] = None
    order: str = Field(default="relevance")
    published_after: Optional[datetime] = None


class YouTubeUploadRequest(BaseModel):
    """YouTube video upload request"""
    video_file_path: str
    title: str
    description: str
    tags: List[str] = []
    category_id: str = "22"
    privacy_status: str = "private"
    thumbnail_path: Optional[str] = None


@router.on_event("startup")
async def startup_event():
    """Initialize YouTube multi-account service on startup"""
    try:
        await youtube_wrapper.initialize()
        logger.info("YouTube multi-account service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize YouTube service: {e}")


@router.get("/accounts", response_model=YouTubeAccountStats)
async def get_youtube_accounts(
    current_user: User = Depends(get_current_verified_user)
) -> YouTubeAccountStats:
    """
    Get all YouTube accounts and their status
    
    Returns statistics and status for all 15 configured YouTube accounts
    """
    try:
        stats = await youtube_wrapper.get_statistics()
        return YouTubeAccountStats(**stats)
    except Exception as e:
        logger.error(f"Failed to get YouTube accounts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/accounts/{account_id}", response_model=YouTubeAccountInfo)
async def get_youtube_account(
    account_id: str,
    current_user: User = Depends(get_current_verified_user)
) -> YouTubeAccountInfo:
    """
    Get specific YouTube account information
    """
    try:
        account = youtube_wrapper.multi_account_service.accounts.get(account_id)
        if not account:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Account {account_id} not found"
            )
            
        return YouTubeAccountInfo(
            account_id=account.account_id,
            email=account.email,
            status=account.status.value,
            channel_id=account.channel_id,
            quota_used=account.quota_used,
            quota_limit=account.quota_limit,
            quota_percentage=(account.quota_used / account.quota_limit * 100) if account.quota_limit > 0 else 0,
            health_score=account.health_score,
            error_count=account.error_count,
            last_used=account.last_used
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get account {account_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/accounts/{account_id}/authenticate")
async def authenticate_youtube_account(
    account_id: str,
    request: YouTubeAuthRequest,
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Authenticate a YouTube account with OAuth
    
    This endpoint initiates the OAuth flow for a specific account
    """
    try:
        success = await youtube_wrapper.multi_account_service.authenticate_account(
            account_id,
            request.client_secrets_path
        )
        
        if success:
            return {
                "status": "success",
                "message": f"Account {account_id} authenticated successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to authenticate account {account_id}"
            )
    except Exception as e:
        logger.error(f"Authentication error for {account_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/accounts/health-check")
async def health_check_accounts(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Perform health check on all YouTube accounts
    
    Tests API connectivity and updates health scores
    """
    try:
        results = await youtube_wrapper.health_check()
        return results
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/accounts/reset-quotas")
async def reset_account_quotas(
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Reset daily quotas for all accounts
    
    This should be called at the start of each day or when needed
    """
    try:
        await youtube_wrapper.multi_account_service.reset_daily_quotas()
        return {
            "status": "success",
            "message": "Daily quotas reset for all accounts"
        }
    except Exception as e:
        logger.error(f"Failed to reset quotas: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/search")
async def search_videos_multi_account(
    request: YouTubeSearchRequest,
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Search YouTube videos using multi-account rotation
    
    Automatically selects the best available account based on health and quota
    """
    try:
        results = await youtube_wrapper.search_videos(
            query=request.query,
            max_results=request.max_results,
            channel_id=request.channel_id,
            order=request.order,
            published_after=request.published_after
        )
        
        return {
            "status": "success",
            "results": results,
            "count": len(results.get("items", []))
        }
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/upload")
async def upload_video_multi_account(
    request: YouTubeUploadRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Upload video to YouTube using multi-account rotation
    
    Automatically selects the best available authenticated account
    """
    try:
        result = await youtube_wrapper.upload_video(
            video_file_path=request.video_file_path,
            title=request.title,
            description=request.description,
            tags=request.tags,
            category_id=request.category_id,
            privacy_status=request.privacy_status,
            thumbnail_path=request.thumbnail_path
        )
        
        return {
            "status": "success",
            "video_id": result.get("id"),
            "title": result.get("snippet", {}).get("title"),
            "upload_status": result.get("status", {}).get("uploadStatus")
        }
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/quota-status")
async def get_quota_status(
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Get current quota status across all accounts
    
    Shows aggregate quota usage and availability
    """
    try:
        stats = await youtube_wrapper.get_statistics()
        
        return {
            "total_quota_available": stats["total_quota_limit"] - stats["total_quota_used"],
            "total_quota_used": stats["total_quota_used"],
            "total_quota_limit": stats["total_quota_limit"],
            "usage_percentage": stats["quota_usage_percentage"],
            "active_accounts": stats["active_accounts"],
            "accounts_near_limit": [
                acc for acc in stats["accounts"]
                if acc["quota_percentage"] > 80
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get quota status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )