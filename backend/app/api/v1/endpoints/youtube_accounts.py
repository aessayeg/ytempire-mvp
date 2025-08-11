"""
YouTube Multi-Account Management API Endpoints
"""
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.db.session import get_async_db
from app.core.auth import get_current_user
from app.models.user import User
from app.services.youtube_multi_account import get_youtube_manager, OperationType

router = APIRouter()


class YouTubeAccountStatus(BaseModel):
    """YouTube account status response"""
    account_id: str
    email: str
    channel_id: str
    channel_name: str
    status: str
    health_score: float
    quota_used: int
    quota_limit: int
    last_used: str
    error_count: int
    total_uploads: int


class YouTubeAccountStats(BaseModel):
    """YouTube accounts statistics"""
    total_accounts: int
    active_accounts: int
    total_quota_used: int
    total_quota_available: int
    average_health_score: float
    accounts: List[Dict[str, Any]]


class OAuthSetupRequest(BaseModel):
    """OAuth setup request"""
    account_index: int = Field(..., ge=0, lt=15, description="Account index (0-14)")


class OAuthCompleteRequest(BaseModel):
    """OAuth completion request"""
    account_index: int = Field(..., ge=0, lt=15, description="Account index (0-14)")
    auth_code: str = Field(..., description="Authorization code from OAuth flow")


class VideoUploadRequest(BaseModel):
    """Video upload request with multi-account rotation"""
    video_path: str
    title: str
    description: str
    tags: List[str]
    category_id: str = "22"  # People & Blogs default
    privacy_status: str = "private"  # private, unlisted, public
    thumbnail_path: Optional[str] = None


@router.get("/stats", response_model=YouTubeAccountStats)
async def get_accounts_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get statistics for all YouTube accounts"""
    try:
        manager = get_youtube_manager()
        stats = manager.get_account_stats()
        return YouTubeAccountStats(**stats)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get account stats: {str(e)}"
        )


@router.get("/accounts", response_model=List[YouTubeAccountStatus])
async def list_youtube_accounts(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """List all YouTube accounts with their status"""
    try:
        manager = get_youtube_manager()
        accounts = []
        
        for account in manager.accounts:
            accounts.append(YouTubeAccountStatus(
                account_id=account.account_id,
                email=account.email,
                channel_id=account.channel_id,
                channel_name=account.channel_name,
                status=account.status.value,
                health_score=account.health_score,
                quota_used=account.quota_used,
                quota_limit=manager.DAILY_QUOTA_LIMIT,
                last_used=account.last_used.isoformat(),
                error_count=account.error_count,
                total_uploads=account.total_uploads
            ))
            
        return accounts
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list accounts: {str(e)}"
        )


@router.get("/accounts/{account_id}", response_model=YouTubeAccountStatus)
async def get_youtube_account(
    account_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get specific YouTube account details"""
    try:
        manager = get_youtube_manager()
        
        for account in manager.accounts:
            if account.account_id == account_id:
                return YouTubeAccountStatus(
                    account_id=account.account_id,
                    email=account.email,
                    channel_id=account.channel_id,
                    channel_name=account.channel_name,
                    status=account.status.value,
                    health_score=account.health_score,
                    quota_used=account.quota_used,
                    quota_limit=manager.DAILY_QUOTA_LIMIT,
                    last_used=account.last_used.isoformat(),
                    error_count=account.error_count,
                    total_uploads=account.total_uploads
                )
                
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Account {account_id} not found"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get account: {str(e)}"
        )


@router.post("/oauth/setup")
async def setup_oauth(
    request: OAuthSetupRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Setup OAuth2 flow for a YouTube account"""
    try:
        # Check admin permission
        if not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
            
        manager = get_youtube_manager()
        auth_url = manager.setup_oauth_for_account(request.account_index)
        
        if not auth_url:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate OAuth URL"
            )
            
        return {
            "auth_url": auth_url,
            "instructions": "Visit the URL, authorize the account, and use the auth code with /oauth/complete endpoint"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OAuth setup failed: {str(e)}"
        )


@router.post("/oauth/complete")
async def complete_oauth(
    request: OAuthCompleteRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Complete OAuth2 flow with authorization code"""
    try:
        # Check admin permission
        if not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
            
        manager = get_youtube_manager()
        success = manager.complete_oauth_for_account(
            request.account_index,
            request.auth_code
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to complete OAuth flow"
            )
            
        return {
            "status": "success",
            "message": f"OAuth completed for account index {request.account_index}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OAuth completion failed: {str(e)}"
        )


@router.post("/upload")
async def upload_video_with_rotation(
    request: VideoUploadRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Upload video using multi-account rotation"""
    try:
        manager = get_youtube_manager()
        
        # Prepare metadata
        metadata = {
            "title": request.title,
            "description": request.description,
            "tags": request.tags,
            "category_id": request.category_id,
            "privacy_status": request.privacy_status,
            "thumbnail_path": request.thumbnail_path
        }
        
        # Upload with rotation
        result = await manager.upload_video_with_rotation(
            request.video_path,
            metadata
        )
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="All YouTube accounts unavailable or upload failed"
            )
            
        return {
            "status": "success",
            "video_id": result["video_id"],
            "channel_id": result["channel_id"],
            "account_used": result["account_used"],
            "upload_time": result["upload_time"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


@router.post("/accounts/{account_id}/reset-quota")
async def reset_account_quota(
    account_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Manually reset quota for a YouTube account (admin only)"""
    try:
        # Check admin permission
        if not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
            
        manager = get_youtube_manager()
        
        for account in manager.accounts:
            if account.account_id == account_id:
                manager._reset_account_quota(account)
                return {
                    "status": "success",
                    "message": f"Quota reset for account {account_id}"
                }
                
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Account {account_id} not found"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset quota: {str(e)}"
        )


@router.get("/best-account")
async def get_best_available_account(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get the best available YouTube account for upload"""
    try:
        manager = get_youtube_manager()
        account = manager.get_best_account()
        
        if not account:
            return {
                "status": "unavailable",
                "message": "No YouTube accounts available",
                "account": None
            }
            
        return {
            "status": "available",
            "account": {
                "account_id": account.account_id,
                "email": account.email,
                "channel_id": account.channel_id,
                "health_score": account.health_score,
                "quota_available": manager.DAILY_QUOTA_LIMIT - account.quota_used
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get best account: {str(e)}"
        )