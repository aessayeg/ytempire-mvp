"""
Video generation Celery tasks
"""
from typing import Dict, Any
import asyncio
from celery import Task
from app.core.celery_app import celery_app
from app.services.ai_service import AIService
from app.services.video_service import VideoService
from app.services.cost_service import CostService
import logging

logger = logging.getLogger(__name__)


class VideoGenerationTask(Task):
    """Base task with error handling"""
    autoretry_for = (Exception,)
    retry_kwargs = {'max_retries': 3}
    retry_backoff = True


@celery_app.task(bind=True, base=VideoGenerationTask)
def generate_video(self, video_id: str, user_id: str, config: Dict[str, Any]):
    """
    Main video generation task
    
    Pipeline:
    1. Generate script
    2. Generate voice
    3. Generate visuals
    4. Assemble video
    5. Generate thumbnail
    6. Upload to YouTube
    """
    try:
        logger.info(f"Starting video generation for video_id: {video_id}")
        
        # Update status
        self.update_state(state="PROGRESS", meta={"step": "script_generation"})
        
        # Step 1: Generate script
        script_result = generate_script.apply_async(
            args=[video_id, config.get("topic"), config.get("style")]
        ).get()
        
        # Step 2: Generate voice
        self.update_state(state="PROGRESS", meta={"step": "voice_synthesis"})
        voice_result = generate_voice.apply_async(
            args=[video_id, script_result["script"], config.get("voice_id")]
        ).get()
        
        # Step 3: Generate visuals
        self.update_state(state="PROGRESS", meta={"step": "visual_generation"})
        visual_result = generate_visuals.apply_async(
            args=[video_id, script_result["prompts"]]
        ).get()
        
        # Step 4: Assemble video
        self.update_state(state="PROGRESS", meta={"step": "video_assembly"})
        video_result = assemble_video.apply_async(
            args=[video_id, voice_result["audio_path"], visual_result["visuals"]]
        ).get()
        
        # Step 5: Generate thumbnail
        self.update_state(state="PROGRESS", meta={"step": "thumbnail_generation"})
        thumbnail_result = generate_thumbnail.apply_async(
            args=[video_id, config.get("title")]
        ).get()
        
        # Step 6: Calculate total cost
        total_cost = calculate_video_cost.apply_async(
            args=[video_id, user_id]
        ).get()
        
        logger.info(f"Video generation completed for video_id: {video_id}, cost: ${total_cost}")
        
        return {
            "video_id": video_id,
            "status": "completed",
            "video_path": video_result["video_path"],
            "thumbnail_path": thumbnail_result["thumbnail_path"],
            "total_cost": total_cost
        }
        
    except Exception as e:
        logger.error(f"Video generation failed for video_id {video_id}: {str(e)}")
        raise


@celery_app.task
def generate_script(video_id: str, topic: str, style: str) -> Dict[str, Any]:
    """Generate video script using AI"""
    try:
        ai_service = AIService()
        
        # Generate script
        script = ai_service.generate_script(topic, style)
        
        # Generate visual prompts
        prompts = ai_service.generate_visual_prompts(script)
        
        # Track cost
        cost = ai_service.calculate_cost("script_generation", script)
        
        return {
            "script": script,
            "prompts": prompts,
            "cost": cost
        }
    except Exception as e:
        logger.error(f"Script generation failed: {str(e)}")
        raise


@celery_app.task
def generate_voice(video_id: str, script: str, voice_id: str) -> Dict[str, Any]:
    """Generate voice from script"""
    try:
        ai_service = AIService()
        
        # Generate voice
        audio_path = ai_service.generate_voice(script, voice_id)
        
        # Track cost
        cost = ai_service.calculate_cost("voice_synthesis", script)
        
        return {
            "audio_path": audio_path,
            "duration": ai_service.get_audio_duration(audio_path),
            "cost": cost
        }
    except Exception as e:
        logger.error(f"Voice generation failed: {str(e)}")
        raise


@celery_app.task
def generate_visuals(video_id: str, prompts: list) -> Dict[str, Any]:
    """Generate visuals for video"""
    try:
        video_service = VideoService()
        
        # Generate images/video clips
        visuals = video_service.generate_visuals(prompts)
        
        # Track cost
        cost = video_service.calculate_visual_cost(len(prompts))
        
        return {
            "visuals": visuals,
            "count": len(visuals),
            "cost": cost
        }
    except Exception as e:
        logger.error(f"Visual generation failed: {str(e)}")
        raise


@celery_app.task
def assemble_video(video_id: str, audio_path: str, visuals: list) -> Dict[str, Any]:
    """Assemble final video"""
    try:
        video_service = VideoService()
        
        # Assemble video
        video_path = video_service.assemble_video(audio_path, visuals)
        
        # Get video metadata
        metadata = video_service.get_video_metadata(video_path)
        
        return {
            "video_path": video_path,
            "duration": metadata["duration"],
            "resolution": metadata["resolution"],
            "file_size": metadata["file_size"]
        }
    except Exception as e:
        logger.error(f"Video assembly failed: {str(e)}")
        raise


@celery_app.task
def generate_thumbnail(video_id: str, title: str) -> Dict[str, Any]:
    """Generate video thumbnail"""
    try:
        ai_service = AIService()
        
        # Generate thumbnail
        thumbnail_path = ai_service.generate_thumbnail(title)
        
        # Track cost
        cost = ai_service.calculate_cost("thumbnail_generation", title)
        
        return {
            "thumbnail_path": thumbnail_path,
            "cost": cost
        }
    except Exception as e:
        logger.error(f"Thumbnail generation failed: {str(e)}")
        raise


@celery_app.task
def calculate_video_cost(video_id: str, user_id: str) -> float:
    """Calculate total cost for video generation"""
    try:
        cost_service = CostService()
        total_cost = cost_service.calculate_video_cost(video_id, user_id)
        
        # Check if within budget
        if total_cost > 3.0:
            logger.warning(f"Video {video_id} exceeded $3 target: ${total_cost}")
        
        return total_cost
    except Exception as e:
        logger.error(f"Cost calculation failed: {str(e)}")
        raise


@celery_app.task
def check_pending_videos():
    """Check and process pending videos"""
    try:
        # This would check database for pending videos
        # and trigger generation for queued videos
        logger.info("Checking for pending videos...")
        # Implementation would go here
        return {"checked": True}
    except Exception as e:
        logger.error(f"Pending video check failed: {str(e)}")
        raise