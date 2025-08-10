"""
Webhook Endpoints for N8N Integration
Owner: Integration Specialist
"""

from fastapi import APIRouter, HTTPException, Header, Request, Depends, status
from typing import Dict, Any, Optional
import json
import logging
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.n8n_service import N8NService
from app.core.config import settings
from app.core.metrics import metrics

router = APIRouter()
logger = logging.getLogger(__name__)


def get_n8n_service(db: AsyncSession = Depends(get_db)) -> N8NService:
    """Get N8N service instance."""
    return N8NService(db)


@router.post("/n8n/video-complete")
async def n8n_video_complete(
    request: Request,
    payload: Dict[str, Any],
    n8n_service: N8NService = Depends(get_n8n_service),
    x_webhook_signature: Optional[str] = Header(None)
):
    """
    Webhook for N8N when video generation is complete
    """
    try:
        # Process the webhook through N8N service
        result = await n8n_service.process_video_completion_webhook(
            payload, 
            x_webhook_signature
        )
        
        # Record webhook received
        metrics.record_celery_task("webhook_video_complete", "received")
        
        if result.get("status") == "success":
            logger.info(f"Video completion webhook processed successfully: {result.get('video_id')}")
            return {
                "received": True,
                "status": "success",
                "message": result.get("message"),
                "video_id": result.get("video_id")
            }
        else:
            logger.error(f"Video completion webhook failed: {result.get('message')}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "Webhook processing failed")
            )
            
    except Exception as e:
        logger.error(f"Error processing video completion webhook: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error processing webhook"
        )


@router.post("/n8n/trigger-upload")
async def trigger_youtube_upload(
    payload: Dict[str, Any],
    n8n_service: N8NService = Depends(get_n8n_service)
):
    """
    Webhook to trigger YouTube upload via N8N
    """
    try:
        video_id = payload.get("video_id")
        channel_id = payload.get("channel_id")
        upload_config = payload.get("upload_config", {})
        
        if not video_id or not channel_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="video_id and channel_id are required"
            )
        
        # Trigger N8N upload workflow
        success = await n8n_service.trigger_youtube_upload_workflow(
            video_id, channel_id, upload_config
        )
        
        if success:
            logger.info(f"YouTube upload workflow triggered for video {video_id}")
            return {
                "received": True,
                "action": "upload_workflow_triggered",
                "video_id": video_id,
                "channel_id": channel_id
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to trigger upload workflow"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering YouTube upload: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/n8n/cost-alert")
async def n8n_cost_alert(
    payload: Dict[str, Any],
    n8n_service: N8NService = Depends(get_n8n_service)
):
    """
    Webhook for cost threshold alerts from N8N
    """
    try:
        user_id = payload.get("user_id")
        current_costs = payload.get("current_costs", {})
        threshold_breached = payload.get("threshold_breached", {})
        alert_type = payload.get("alert_type", "warning")  # warning, critical
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="user_id is required"
            )
        
        # Log the cost alert
        logger.warning(
            f"Cost alert for user {user_id}: {alert_type} - "
            f"Current: {current_costs}, Threshold breached: {threshold_breached}"
        )
        
        # Record cost alert metric
        metrics.record_celery_task("cost_alert_webhook", "received")
        
        # TODO: Implement cost alert actions:
        # - Send email notification to user
        # - Temporarily pause video generation if critical
        # - Update user cost limits
        # - Trigger cost optimization workflow
        
        response_action = "alert_logged"
        
        if alert_type == "critical":
            # In critical cases, we might want to pause operations
            response_action = "operations_paused"
            logger.critical(f"Critical cost alert for user {user_id} - operations may be paused")
        
        return {
            "received": True,
            "action": response_action,
            "user_id": user_id,
            "alert_type": alert_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing cost alert webhook: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/youtube/callback")
async def youtube_callback(
    request: Request,
    payload: Dict[str, Any],
    n8n_service: N8NService = Depends(get_n8n_service)
):
    """
    Webhook for YouTube API callbacks
    """
    try:
        # Process the YouTube callback through N8N service
        result = await n8n_service.process_youtube_callback_webhook(payload)
        
        # Record webhook received
        metrics.record_celery_task("webhook_youtube_callback", "received")
        
        if result.get("status") == "success":
            logger.info(f"YouTube callback webhook processed: {result.get('message')}")
            return {
                "received": True,
                "status": "success",
                "message": result.get("message"),
                "video_id": result.get("video_id"),
                "youtube_video_id": result.get("youtube_video_id")
            }
        else:
            logger.error(f"YouTube callback webhook failed: {result.get('message')}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "Webhook processing failed")
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing YouTube callback webhook: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error processing webhook"
        )


@router.post("/n8n/analytics-complete")
async def n8n_analytics_complete(
    payload: Dict[str, Any],
    n8n_service: N8NService = Depends(get_n8n_service)
):
    """
    Webhook for N8N analytics sync completion
    """
    try:
        video_id = payload.get("video_id")
        channel_id = payload.get("channel_id")
        analytics_data = payload.get("analytics_data", {})
        sync_status = payload.get("status", "success")
        
        if not video_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="video_id is required"
            )
        
        # Record analytics sync completion
        metrics.record_celery_task("n8n_analytics_sync", sync_status)
        
        logger.info(f"Analytics sync completed for video {video_id}: {sync_status}")
        
        # TODO: Store analytics data in database
        
        return {
            "received": True,
            "status": sync_status,
            "video_id": video_id,
            "channel_id": channel_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing analytics completion webhook: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/n8n/optimization-complete")
async def n8n_optimization_complete(
    payload: Dict[str, Any],
    n8n_service: N8NService = Depends(get_n8n_service)
):
    """
    Webhook for N8N content optimization completion
    """
    try:
        video_id = payload.get("video_id")
        optimization_type = payload.get("optimization_type")
        suggestions = payload.get("suggestions", {})
        optimization_status = payload.get("status", "success")
        
        if not video_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="video_id is required"
            )
        
        # Record optimization completion
        metrics.record_celery_task("n8n_content_optimization", optimization_status)
        
        logger.info(f"Content optimization completed for video {video_id}: {optimization_type}")
        
        # TODO: Apply optimization suggestions if approved by user
        
        return {
            "received": True,
            "status": optimization_status,
            "video_id": video_id,
            "optimization_type": optimization_type,
            "suggestions_count": len(suggestions)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing optimization completion webhook: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/n8n/health")
async def n8n_health_check():
    """
    Health check endpoint for N8N integration
    """
    try:
        # Test N8N connectivity
        import httpx
        
        n8n_url = settings.N8N_BASE_URL or "http://n8n:5678"
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{n8n_url}/healthz")
            n8n_healthy = response.status_code == 200
        
        return {
            "status": "healthy" if n8n_healthy else "unhealthy",
            "n8n_connection": "connected" if n8n_healthy else "disconnected",
            "webhook_endpoints": [
                "/api/v1/webhooks/n8n/video-complete",
                "/api/v1/webhooks/n8n/trigger-upload",
                "/api/v1/webhooks/n8n/cost-alert",
                "/api/v1/webhooks/youtube/callback",
                "/api/v1/webhooks/n8n/analytics-complete",
                "/api/v1/webhooks/n8n/optimization-complete"
            ]
        }
        
    except Exception as e:
        logger.error(f"N8N health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "n8n_connection": "disconnected",
            "error": str(e)
        }


@router.post("/stripe/payment")
async def stripe_payment_webhook(
    request: Request,
    payload: Dict[str, Any],
    stripe_signature: Optional[str] = Header(None)
):
    """
    Webhook for Stripe payment events (future implementation)
    """
    event_type = payload.get("type")
    
    if event_type == "payment_intent.succeeded":
        # Handle successful payment
        user_id = payload.get("metadata", {}).get("user_id")
        amount = payload.get("amount")
        print(f"Payment received from user {user_id}: ${amount/100}")
    
    return {"received": True}