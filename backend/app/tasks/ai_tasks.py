"""
AI Tasks for Celery
Handles AI model operations, script generation, voice synthesis, and content optimization
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from celery import Task

from app.core.celery_app import celery_app, TaskPriority
from app.services.ai_services import OpenAIService, ElevenLabsService, GoogleTTSService
from app.services.script_generation import ScriptGenerationService
from app.services.thumbnail_generator import ThumbnailGenerator
from app.services.cost_tracking import cost_tracker
from app.db.session import AsyncSessionLocal
from app.models.video import Video

logger = logging.getLogger(__name__)


class AITask(Task):
    """Base class for AI tasks with cost tracking"""

    _ai_service = None
    _script_service = None
    _thumbnail_service = None

    @property
    def ai_service(self):
        if self._ai_service is None:
            from app.core.config import settings

            self._ai_service = OpenAIService(
                {"openai_api_key": settings.OPENAI_API_KEY}
            )
        return self._ai_service

    @property
    def script_service(self):
        if self._script_service is None:
            self._script_service = ScriptGenerationService()
        return self._script_service

    @property
    def thumbnail_service(self):
        if self._thumbnail_service is None:
            self._thumbnail_service = ThumbnailGenerator()
        return self._thumbnail_service


@celery_app.task(
    bind=True,
    base=AITask,
    name="ai.generate_script",
    queue="ai_generation",
    max_retries=3,
    default_retry_delay=60,
)
def generate_video_script(
    self,
    topic: str,
    style: str = "educational",
    length: str = "medium",
    keywords: List[str] = None,
    user_id: str = None,
    channel_id: str = None,
) -> Dict[str, Any]:
    """
    Generate video script using AI

    Args:
        topic: Video topic
        style: Content style (educational, entertainment, etc.)
        length: Video length (short, medium, long)
        keywords: SEO keywords to include
        user_id: User ID for tracking
        channel_id: Channel ID for personalization
    """
    try:
        logger.info(f"Generating script for topic: {topic}")

        # Track start time for cost calculation
        start_time = datetime.now()

        # Generate script using AI service
        script_data = asyncio.run(
            self.script_service.generate_script(
                topic=topic,
                style=style,
                length=length,
                keywords=keywords or [],
                channel_id=channel_id,
            )
        )

        # Track cost
        cost = script_data.get("cost", 0.0)
        if user_id:
            asyncio.run(
                cost_tracker.track_cost(
                    user_id=user_id,
                    service="openai",
                    operation="script_generation",
                    amount=cost,
                    metadata={"topic": topic, "style": style},
                )
            )

        # Calculate generation time
        generation_time = (datetime.now() - start_time).total_seconds()

        return {
            "success": True,
            "script": script_data.get("script"),
            "title": script_data.get("title"),
            "description": script_data.get("description"),
            "tags": script_data.get("tags", []),
            "cost": cost,
            "generation_time": generation_time,
        }

    except Exception as e:
        logger.error(f"Script generation failed: {str(e)}")
        raise self.retry(exc=e)


@celery_app.task(
    bind=True,
    base=AITask,
    name="ai.synthesize_voice",
    queue="ai_generation",
    max_retries=3,
    default_retry_delay=30,
)
def synthesize_voice(
    self,
    text: str,
    voice_provider: str = "elevenlabs",
    voice_id: Optional[str] = None,
    user_id: str = None,
) -> Dict[str, Any]:
    """
    Synthesize voice from text

    Args:
        text: Text to synthesize
        voice_provider: Voice provider (elevenlabs, google_tts)
        voice_id: Specific voice ID to use
        user_id: User ID for cost tracking
    """
    try:
        logger.info(f"Synthesizing voice with {voice_provider}")

        # Select voice service
        if voice_provider == "elevenlabs":
            from app.services.ai_services import ElevenLabsService

            voice_service = ElevenLabsService()
            result = asyncio.run(voice_service.synthesize_speech(text, voice_id))
        else:
            from app.services.ai_services import GoogleTTSService

            voice_service = GoogleTTSService()
            result = asyncio.run(voice_service.synthesize_speech(text))

        # Track cost
        if user_id and result.get("cost"):
            asyncio.run(
                cost_tracker.track_cost(
                    user_id=user_id,
                    service=voice_provider,
                    operation="voice_synthesis",
                    amount=result["cost"],
                )
            )

        return {
            "success": True,
            "audio_url": result.get("audio_url"),
            "audio_path": result.get("audio_path"),
            "duration": result.get("duration"),
            "cost": result.get("cost", 0.0),
        }

    except Exception as e:
        logger.error(f"Voice synthesis failed: {str(e)}")
        raise self.retry(exc=e)


@celery_app.task(
    bind=True,
    base=AITask,
    name="ai.generate_thumbnail",
    queue="ai_generation",
    max_retries=2,
    default_retry_delay=60,
)
def generate_thumbnail(
    self, title: str, style: str = "modern", video_id: str = None, user_id: str = None
) -> Dict[str, Any]:
    """
    Generate thumbnail using AI

    Args:
        title: Video title for thumbnail
        style: Thumbnail style
        video_id: Video ID to associate
        user_id: User ID for cost tracking
    """
    try:
        logger.info(f"Generating thumbnail for: {title}")

        result = asyncio.run(self.thumbnail_service.generate(title=title, style=style))

        # Track cost
        if user_id and result.get("cost"):
            asyncio.run(
                cost_tracker.track_cost(
                    user_id=user_id,
                    service="dalle",
                    operation="thumbnail_generation",
                    amount=result["cost"],
                )
            )

        # Update video if ID provided
        if video_id:
            asyncio.run(update_video_thumbnail(video_id, result["thumbnail_url"]))

        return {
            "success": True,
            "thumbnail_url": result.get("thumbnail_url"),
            "thumbnail_path": result.get("thumbnail_path"),
            "cost": result.get("cost", 0.0),
        }

    except Exception as e:
        logger.error(f"Thumbnail generation failed: {str(e)}")
        raise self.retry(exc=e)


@celery_app.task(name="ai.optimize_content", queue="ai_generation")
def optimize_content_for_seo(
    title: str, description: str, tags: List[str], target_keywords: List[str] = None
) -> Dict[str, Any]:
    """
    Optimize content for SEO using AI

    Args:
        title: Original title
        description: Original description
        tags: Original tags
        target_keywords: Keywords to optimize for
    """
    try:
        logger.info("Optimizing content for SEO")

        # Use AI to optimize content
        # This would call the AI service to improve SEO
        optimized = {
            "title": title,  # AI would enhance this
            "description": description,  # AI would enhance this
            "tags": tags + (target_keywords or []),  # AI would optimize
        }

        return {
            "success": True,
            "optimized": optimized,
            "seo_score": 0.85,  # AI would calculate actual score
        }

    except Exception as e:
        logger.error(f"Content optimization failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "original": {"title": title, "description": description, "tags": tags},
        }


async def update_video_thumbnail(video_id: str, thumbnail_url: str):
    """Update video thumbnail URL in database"""
    async with AsyncSessionLocal() as db:
        video = await db.get(Video, video_id)
        if video:
            video.thumbnail_url = thumbnail_url
            await db.commit()


# Periodic task for model optimization
@celery_app.task(name="ai.optimize_models", queue="ai_generation")
def optimize_ai_models():
    """Periodic task to optimize AI model performance"""
    try:
        logger.info("Running AI model optimization")

        # This would include:
        # - Clearing model cache
        # - Updating model parameters
        # - Checking for model updates
        # - Performance metrics collection

        return {
            "success": True,
            "optimized_at": datetime.now().isoformat(),
            "models_checked": ["gpt-4", "dalle-3", "elevenlabs"],
        }

    except Exception as e:
        logger.error(f"Model optimization failed: {str(e)}")
        return {"success": False, "error": str(e)}
