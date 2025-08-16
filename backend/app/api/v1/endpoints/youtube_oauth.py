"""
YouTube OAuth Endpoints
Handles OAuth flow for multiple YouTube accounts
"""
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import RedirectResponse
from typing import Optional
import logging

from app.services.youtube_oauth_service import youtube_oauth_service
from app.core.security import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/oauth/authorize/{account_id}")
async def authorize_youtube_account(
    account_id: str, current_user=Depends(get_current_user)
):
    """
    Start OAuth flow for a YouTube account
    Returns authorization URL for user to grant permissions
    """
    try:
        # Only allow admins to authorize accounts
        if not current_user.is_superuser:
            raise HTTPException(status_code=403, detail="Admin access required")

        auth_url = youtube_oauth_service.get_auth_url(account_id)

        return {
            "auth_url": auth_url,
            "account_id": account_id,
            "message": "Visit the auth_url to authorize the YouTube account",
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"OAuth authorization error: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to generate authorization URL"
        )


@router.get("/oauth/callback")
async def oauth_callback(
    code: str = Query(..., description="Authorization code from Google"),
    state: str = Query(..., description="State parameter with account_id"),
):
    """
    Handle OAuth callback from Google
    Stores credentials for the YouTube account
    """
    try:
        # Extract account_id from state
        account_id = state.split(":")[0] if ":" in state else state

        # Handle the callback and store credentials
        account_info = youtube_oauth_service.handle_callback(account_id, code)

        # Redirect to success page
        return RedirectResponse(
            url=f"/admin/youtube-accounts?authorized={account_id}&status=success"
        )

    except Exception as e:
        logger.error(f"OAuth callback error: {e}")
        return RedirectResponse(
            url=f"/admin/youtube-accounts?error=authorization_failed"
        )


@router.get("/accounts/authorized")
async def list_authorized_accounts(current_user=Depends(get_current_user)):
    """
    List all authorized YouTube accounts
    """
    try:
        if not current_user.is_superuser:
            raise HTTPException(status_code=403, detail="Admin access required")

        accounts = youtube_oauth_service.list_authorized_accounts()

        return {"total": len(accounts), "accounts": accounts}

    except Exception as e:
        logger.error(f"Failed to list authorized accounts: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve account list")


@router.post("/accounts/{account_id}/health")
async def check_account_health(account_id: str, current_user=Depends(get_current_user)):
    """
    Check health status of a YouTube account
    """
    try:
        if not current_user.is_superuser:
            raise HTTPException(status_code=403, detail="Admin access required")

        health_status = youtube_oauth_service.check_account_health(account_id)

        return health_status

    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail="Failed to check account health")


@router.post("/accounts/authorize-all")
async def generate_all_auth_urls(current_user=Depends(get_current_user)):
    """
    Generate authorization URLs for all unconfigured accounts
    """
    try:
        if not current_user.is_superuser:
            raise HTTPException(status_code=403, detail="Admin access required")

        auth_urls = []
        for account_id in youtube_oauth_service.accounts.keys():
            account = youtube_oauth_service.accounts[account_id]
            if not account.get("refresh_token"):
                try:
                    auth_url = youtube_oauth_service.get_auth_url(account_id)
                    auth_urls.append(
                        {
                            "account_id": account_id,
                            "email": account.get("email"),
                            "auth_url": auth_url,
                        }
                    )
                except Exception as e:
                    logger.error(f"Failed to generate auth URL for {account_id}: {e}")

        return {"total_unauthorized": len(auth_urls), "accounts": auth_urls}

    except Exception as e:
        logger.error(f"Failed to generate auth URLs: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to generate authorization URLs"
        )
