"""
N8N Integration Service
Owner: Integration Specialist
"""

import logging
import json
import hmac
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime
import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.metrics import metrics
from app.repositories.video_repository import VideoRepository
from app.repositories.user_repository import UserRepository
from app.repositories.channel_repository import ChannelRepository
from app.schemas.video import VideoStatus
from app.models.video import Video

logger = logging.getLogger(__name__)


class N8NService:
    """Service for integrating with N8N workflow automation."""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.video_repo = VideoRepository(db_session)
        self.user_repo = UserRepository(db_session)
        self.channel_repo = ChannelRepository(db_session)
        
        # N8N Configuration
        self.n8n_base_url = settings.N8N_BASE_URL or "http://n8n:5678"
        self.webhook_secret = settings.N8N_WEBHOOK_SECRET or "ytempire-n8n-secret"
        self.api_key = settings.N8N_API_KEY or None
        
        # Workflow URLs
        self.workflows = {
            "video_generation": f"{self.n8n_base_url}/webhook/video-generation",
            "youtube_upload": f"{self.n8n_base_url}/webhook/youtube-upload", 
            "cost_monitoring": f"{self.n8n_base_url}/webhook/cost-monitoring",
            "analytics_sync": f"{self.n8n_base_url}/webhook/analytics-sync",
            "content_optimization": f"{self.n8n_base_url}/webhook/content-optimization"
        }
    
    async def trigger_video_generation_workflow(
        self, 
        video_id: str, 
        user_id: str,
        channel_id: str,
        video_config: Dict[str, Any]
    ) -> bool:
        """Trigger N8N workflow for video generation."""
        
        try:
            # Prepare workflow payload
            payload = {
                "workflow_type": "video_generation",
                "video_id": video_id,
                "user_id": user_id,
                "channel_id": channel_id,
                "timestamp": datetime.utcnow().isoformat(),
                "config": {
                    "title": video_config.get("title"),
                    "description": video_config.get("description"),
                    "keywords": video_config.get("keywords", []),
                    "duration": video_config.get("duration", 60),
                    "style": video_config.get("style", "standard"),
                    "voice_id": video_config.get("voice_id"),
                    "background_music": video_config.get("background_music", True),
                    "thumbnail_style": video_config.get("thumbnail_style", "auto")
                },
                "ai_services": {
                    "script_generation": True,
                    "voice_synthesis": True,
                    "image_generation": True,
                    "video_editing": True
                },
                "callback_url": f"{settings.BACKEND_URL}/api/v1/webhooks/n8n/video-complete"
            }
            
            # Make request to N8N workflow webhook
            success = await self._send_webhook_request(
                self.workflows["video_generation"],
                payload
            )
            
            if success:
                # Update video status to processing
                await self.video_repo.update_status(video_id, VideoStatus.PROCESSING)
                
                # Record metrics
                metrics.record_celery_task("n8n_video_generation", "started")
                
                logger.info(f"N8N video generation workflow triggered for video {video_id}")
                return True
            else:
                logger.error(f"Failed to trigger N8N video generation for video {video_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error triggering N8N video generation workflow: {str(e)}")
            metrics.record_celery_task("n8n_video_generation", "failed")
            return False
    
    async def trigger_youtube_upload_workflow(
        self,
        video_id: str,
        channel_id: str,
        upload_config: Dict[str, Any]
    ) -> bool:
        """Trigger N8N workflow for YouTube upload."""
        
        try:
            # Get video and channel details
            video = await self.video_repo.get_by_id(video_id)
            if not video:
                raise ValueError(f"Video {video_id} not found")
            
            channel = await self.channel_repo.get_by_id(channel_id)
            if not channel:
                raise ValueError(f"Channel {channel_id} not found")
            
            payload = {
                "workflow_type": "youtube_upload",
                "video_id": video_id,
                "channel_id": channel_id,
                "user_id": video.user_id,
                "timestamp": datetime.utcnow().isoformat(),
                "video_data": {
                    "title": video.title,
                    "description": video.description,
                    "tags": video.keywords,
                    "category_id": upload_config.get("category_id", "22"),  # Entertainment
                    "privacy_status": upload_config.get("privacy_status", "private"),
                    "video_file_path": video.video_file_path,
                    "thumbnail_file_path": video.thumbnail_file_path
                },
                "channel_data": {
                    "youtube_channel_id": channel.youtube_channel_id,
                    "oauth_credentials": channel.oauth_tokens
                },
                "upload_config": upload_config,
                "callback_url": f"{settings.BACKEND_URL}/api/v1/webhooks/youtube/callback"
            }
            
            success = await self._send_webhook_request(
                self.workflows["youtube_upload"],
                payload
            )
            
            if success:
                await self.video_repo.update_status(video_id, VideoStatus.UPLOADING)
                metrics.record_celery_task("n8n_youtube_upload", "started")
                logger.info(f"N8N YouTube upload workflow triggered for video {video_id}")
                return True
            else:
                logger.error(f"Failed to trigger N8N YouTube upload for video {video_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error triggering N8N YouTube upload workflow: {str(e)}")
            metrics.record_celery_task("n8n_youtube_upload", "failed")
            return False
    
    async def trigger_cost_monitoring_workflow(
        self,
        user_id: str,
        current_costs: Dict[str, float],
        thresholds: Dict[str, float]
    ) -> bool:
        """Trigger N8N workflow for cost monitoring and alerts."""
        
        try:
            payload = {
                "workflow_type": "cost_monitoring",
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat(),
                "costs": current_costs,
                "thresholds": thresholds,
                "alert_config": {
                    "email_enabled": True,
                    "slack_enabled": False,
                    "webhook_enabled": True
                },
                "callback_url": f"{settings.BACKEND_URL}/api/v1/webhooks/n8n/cost-alert"
            }
            
            success = await self._send_webhook_request(
                self.workflows["cost_monitoring"],
                payload
            )
            
            if success:
                metrics.record_celery_task("n8n_cost_monitoring", "started")
                logger.info(f"N8N cost monitoring workflow triggered for user {user_id}")
                return True
            else:
                logger.error(f"Failed to trigger N8N cost monitoring for user {user_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error triggering N8N cost monitoring workflow: {str(e)}")
            return False
    
    async def trigger_analytics_sync_workflow(
        self,
        video_id: str,
        channel_id: str,
        sync_config: Dict[str, Any]
    ) -> bool:
        """Trigger N8N workflow for YouTube analytics synchronization."""
        
        try:
            payload = {
                "workflow_type": "analytics_sync",
                "video_id": video_id,
                "channel_id": channel_id,
                "timestamp": datetime.utcnow().isoformat(),
                "sync_config": sync_config,
                "metrics_to_sync": [
                    "views", "likes", "dislikes", "comments", 
                    "shares", "subscribers_gained", "watch_time",
                    "impressions", "ctr", "revenue"
                ],
                "callback_url": f"{settings.BACKEND_URL}/api/v1/webhooks/n8n/analytics-complete"
            }
            
            success = await self._send_webhook_request(
                self.workflows["analytics_sync"],
                payload
            )
            
            if success:
                metrics.record_celery_task("n8n_analytics_sync", "started")
                logger.info(f"N8N analytics sync workflow triggered for video {video_id}")
                return True
            else:
                logger.error(f"Failed to trigger N8N analytics sync for video {video_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error triggering N8N analytics sync workflow: {str(e)}")
            return False
    
    async def trigger_content_optimization_workflow(
        self,
        video_id: str,
        optimization_type: str,
        performance_data: Dict[str, Any]
    ) -> bool:
        """Trigger N8N workflow for content optimization based on performance data."""
        
        try:
            payload = {
                "workflow_type": "content_optimization",
                "video_id": video_id,
                "optimization_type": optimization_type,  # title, description, tags, thumbnail
                "timestamp": datetime.utcnow().isoformat(),
                "performance_data": performance_data,
                "optimization_config": {
                    "ai_suggestions": True,
                    "a_b_testing": True,
                    "trend_analysis": True,
                    "competitor_analysis": False
                },
                "callback_url": f"{settings.BACKEND_URL}/api/v1/webhooks/n8n/optimization-complete"
            }
            
            success = await self._send_webhook_request(
                self.workflows["content_optimization"],
                payload
            )
            
            if success:
                metrics.record_celery_task("n8n_content_optimization", "started")
                logger.info(f"N8N content optimization workflow triggered for video {video_id}")
                return True
            else:
                logger.error(f"Failed to trigger N8N content optimization for video {video_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error triggering N8N content optimization workflow: {str(e)}")
            return False
    
    async def process_video_completion_webhook(
        self,
        payload: Dict[str, Any],
        signature: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process video completion webhook from N8N."""
        
        try:
            # Verify webhook signature
            if signature and not self._verify_signature(json.dumps(payload), signature):
                raise ValueError("Invalid webhook signature")
            
            video_id = payload.get("video_id")
            status = payload.get("status")
            file_paths = payload.get("file_paths", {})
            generation_costs = payload.get("costs", {})
            error_details = payload.get("error_details")
            
            if not video_id:
                raise ValueError("video_id is required")
            
            # Update video in database
            video = await self.video_repo.get_by_id(video_id)
            if not video:
                raise ValueError(f"Video {video_id} not found")
            
            if status == "completed":
                # Update video with generated files
                update_data = {
                    "video_file_path": file_paths.get("video_file"),
                    "thumbnail_file_path": file_paths.get("thumbnail_file"),
                    "script_file_path": file_paths.get("script_file"),
                    "status": VideoStatus.COMPLETED
                }
                
                await self.video_repo.update(video_id, update_data)
                
                # Record costs
                if generation_costs:
                    for service, cost in generation_costs.items():
                        metrics.record_api_cost(service, video.user_id, cost)
                
                # Record successful completion
                metrics.record_video_generated(video.user_id, video.channel_id)
                metrics.record_celery_task("n8n_video_generation", "completed")
                
                logger.info(f"Video {video_id} generation completed successfully")
                
                return {
                    "status": "success",
                    "message": "Video generation completed",
                    "video_id": video_id
                }
                
            elif status == "failed":
                # Update video status to failed
                await self.video_repo.update_status(video_id, VideoStatus.FAILED)
                
                # Log error details
                if error_details:
                    logger.error(f"Video {video_id} generation failed: {error_details}")
                
                metrics.record_celery_task("n8n_video_generation", "failed")
                
                return {
                    "status": "error",
                    "message": "Video generation failed",
                    "video_id": video_id,
                    "error": error_details
                }
            
            else:
                raise ValueError(f"Unknown status: {status}")
                
        except Exception as e:
            logger.error(f"Error processing video completion webhook: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def process_youtube_callback_webhook(
        self,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process YouTube callback webhook."""
        
        try:
            event_type = payload.get("event_type")
            video_id = payload.get("video_id")
            youtube_video_id = payload.get("youtube_video_id")
            error_details = payload.get("error_details")
            
            if not video_id:
                raise ValueError("video_id is required")
            
            video = await self.video_repo.get_by_id(video_id)
            if not video:
                raise ValueError(f"Video {video_id} not found")
            
            if event_type == "upload_complete":
                # Update video with YouTube video ID
                await self.video_repo.update(video_id, {
                    "youtube_video_id": youtube_video_id,
                    "status": VideoStatus.PUBLISHED,
                    "published_at": datetime.utcnow()
                })
                
                # Record successful upload
                metrics.record_video_uploaded(video.user_id, video.channel_id)
                metrics.record_celery_task("n8n_youtube_upload", "completed")
                
                logger.info(f"Video {video_id} uploaded to YouTube: {youtube_video_id}")
                
                return {
                    "status": "success",
                    "message": "Video uploaded to YouTube",
                    "video_id": video_id,
                    "youtube_video_id": youtube_video_id
                }
                
            elif event_type == "upload_failed":
                # Update video status to failed
                await self.video_repo.update_status(video_id, VideoStatus.UPLOAD_FAILED)
                
                logger.error(f"YouTube upload failed for video {video_id}: {error_details}")
                metrics.record_celery_task("n8n_youtube_upload", "failed")
                
                return {
                    "status": "error",
                    "message": "YouTube upload failed",
                    "video_id": video_id,
                    "error": error_details
                }
            
            else:
                # Handle other events (processing_complete, etc.)
                logger.info(f"YouTube event {event_type} for video {video_id}")
                return {
                    "status": "success",
                    "message": f"Event {event_type} processed",
                    "video_id": video_id
                }
                
        except Exception as e:
            logger.error(f"Error processing YouTube callback webhook: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def _send_webhook_request(
        self,
        webhook_url: str,
        payload: Dict[str, Any]
    ) -> bool:
        """Send webhook request to N8N."""
        
        try:
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "YTEmpire-Backend/1.0"
            }
            
            # Add API key if available
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            
            # Add webhook signature
            payload_json = json.dumps(payload, sort_keys=True)
            signature = self._generate_signature(payload_json)
            headers["X-Webhook-Signature"] = signature
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    webhook_url,
                    json=payload,
                    headers=headers
                )
                
                if response.status_code == 200:
                    logger.debug(f"Webhook sent successfully to {webhook_url}")
                    return True
                else:
                    logger.error(f"Webhook failed: {response.status_code} - {response.text}")
                    return False
                    
        except httpx.TimeoutException:
            logger.error(f"Webhook timeout for {webhook_url}")
            return False
        except Exception as e:
            logger.error(f"Error sending webhook to {webhook_url}: {str(e)}")
            return False
    
    def _generate_signature(self, payload: str) -> str:
        """Generate HMAC signature for webhook payload."""
        return hmac.new(
            self.webhook_secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def _verify_signature(self, payload: str, signature: str) -> bool:
        """Verify webhook signature."""
        expected = self._generate_signature(payload)
        return hmac.compare_digest(expected, signature)
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific N8N workflow execution."""
        
        try:
            if not self.api_key:
                logger.warning("No N8N API key configured, cannot check workflow status")
                return None
            
            headers = {
                "X-N8N-API-KEY": self.api_key,
                "Accept": "application/json"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.n8n_base_url}/api/v1/executions/{workflow_id}",
                    headers=headers
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to get workflow status: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting workflow status: {str(e)}")
            return None