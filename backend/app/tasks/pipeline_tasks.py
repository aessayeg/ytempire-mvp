"""
Video Processing Pipeline Tasks
Comprehensive Celery task definitions for video automation
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from celery import chain, group, chord, signature
from celery.result import AsyncResult

from app.core.celery_app import celery_app, TaskPriority
from app.db.session import get_db
from app.models.video import Video, VideoStatus
from app.models.channel import Channel
from app.services.ai_services import AIServiceManager
from app.services.video_processor import VideoProcessor
from app.services.youtube_multi_account import get_youtube_manager
from app.services.cost_optimizer import get_cost_optimizer, ServiceType
from app.services.n8n_integration import get_n8n_integration, WorkflowType
from app.websocket.manager import websocket_manager

logger = logging.getLogger(__name__)


# Pipeline Stage Tasks

@celery_app.task(name="pipeline.analyze_trends", priority=TaskPriority.HIGH)
def analyze_trends(channel_id: int, categories: List[str] = None) -> Dict[str, Any]:
    """Analyze current trends for content selection"""
    try:
        logger.info(f"Analyzing trends for channel {channel_id}")
        
        # Trigger N8N workflow for trend analysis
        n8n = get_n8n_integration()
        result = n8n.trigger_trend_analysis(
            categories=categories,
            max_results=20
        )
        
        # Store trends in cache
        import redis
        r = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"))
        r.setex(
            f"trends:{channel_id}",
            3600,  # 1 hour cache
            json.dumps(result)
        )
        
        return {
            "status": "success",
            "trends": result.get("trends", []),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Trend analysis failed: {e}")
        raise


@celery_app.task(name="pipeline.select_topic", priority=TaskPriority.HIGH)
def select_topic(trends: Dict[str, Any], channel_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
    """Select best topic from trends based on channel preferences"""
    try:
        logger.info("Selecting optimal topic from trends")
        
        trend_list = trends.get("trends", [])
        if not trend_list:
            raise ValueError("No trends available")
            
        # Score topics based on various factors
        scored_topics = []
        for trend in trend_list:
            score = 0
            
            # Trend velocity score
            score += trend.get("velocity", 0) * 10
            
            # Search volume score
            score += min(trend.get("search_volume", 0) / 1000, 100)
            
            # Competition score (lower is better)
            competition = trend.get("competition", 100)
            score += max(0, 100 - competition)
            
            # Channel preference matching
            if channel_preferences:
                for pref in channel_preferences.get("preferred_topics", []):
                    if pref.lower() in trend.get("title", "").lower():
                        score += 50
                        
            scored_topics.append({
                "topic": trend,
                "score": score
            })
            
        # Sort by score and select top
        scored_topics.sort(key=lambda x: x["score"], reverse=True)
        selected = scored_topics[0]
        
        return {
            "status": "success",
            "selected_topic": selected["topic"],
            "score": selected["score"],
            "alternatives": [t["topic"] for t in scored_topics[1:4]]
        }
        
    except Exception as e:
        logger.error(f"Topic selection failed: {e}")
        raise


@celery_app.task(name="pipeline.generate_script", priority=TaskPriority.HIGH)
def generate_script_task(topic: Dict[str, Any], style: str = "educational") -> Dict[str, Any]:
    """Generate video script with cost optimization"""
    try:
        logger.info(f"Generating script for topic: {topic.get('title')}")
        
        # Get optimal model for script generation
        optimizer = get_cost_optimizer()
        model, tier = optimizer.get_optimal_model(
            ServiceType.SCRIPT_GENERATION,
            quality_required=85.0
        )
        
        # Check cache first
        cached = optimizer.get_cached_response(
            ServiceType.SCRIPT_GENERATION,
            topic.get("title"),
            {"style": style}
        )
        
        if cached:
            logger.info("Using cached script")
            return cached
            
        # Generate new script
        ai_manager = AIServiceManager()
        script = ai_manager.generate_script(
            topic.get("title"),
            style,
            model=model
        )
        
        # Cache the response
        optimizer.cache_response(
            ServiceType.SCRIPT_GENERATION,
            topic.get("title"),
            script,
            {"style": style}
        )
        
        return {
            "status": "success",
            "script": script,
            "model_used": model,
            "tier": tier.value
        }
        
    except Exception as e:
        logger.error(f"Script generation failed: {e}")
        raise


@celery_app.task(name="pipeline.generate_voice", priority=TaskPriority.HIGH)
def generate_voice_task(script: Dict[str, Any]) -> Dict[str, Any]:
    """Generate voice from script"""
    try:
        logger.info("Generating voice from script")
        
        # Check cost budget
        optimizer = get_cost_optimizer()
        if not optimizer.check_budget_available(
            ServiceType.VOICE_SYNTHESIS,
            estimated_cost=0.5
        ):
            raise ValueError("Voice synthesis budget exceeded")
            
        ai_manager = AIServiceManager()
        voice_result = ai_manager.generate_voice(
            script.get("script", {}).get("text", "")
        )
        
        return {
            "status": "success",
            "audio_path": voice_result.get("path"),
            "duration": voice_result.get("duration"),
            "cost": voice_result.get("cost", 0.5)
        }
        
    except Exception as e:
        logger.error(f"Voice generation failed: {e}")
        raise


@celery_app.task(name="pipeline.generate_visuals", priority=TaskPriority.NORMAL)
def generate_visuals_task(script: Dict[str, Any], topic: Dict[str, Any]) -> Dict[str, Any]:
    """Generate or fetch visuals for video"""
    try:
        logger.info("Generating visuals for video")
        
        ai_manager = AIServiceManager()
        scenes = script.get("script", {}).get("scenes", [])
        
        visuals = ai_manager.generate_visuals(
            scenes,
            topic.get("title")
        )
        
        return {
            "status": "success",
            "visuals": visuals.get("clips", []),
            "thumbnail": visuals.get("thumbnail"),
            "cost": visuals.get("cost", 0.0)
        }
        
    except Exception as e:
        logger.error(f"Visual generation failed: {e}")
        raise


@celery_app.task(name="pipeline.create_video", priority=TaskPriority.HIGH)
def create_video_task(
    audio: Dict[str, Any],
    visuals: Dict[str, Any],
    script: Dict[str, Any],
    video_id: int
) -> Dict[str, Any]:
    """Assemble final video from components"""
    try:
        logger.info(f"Creating video {video_id}")
        
        video_processor = VideoProcessor()
        
        result = video_processor.create_video_with_audio(
            audio_path=audio.get("audio_path"),
            visuals=visuals.get("visuals", []),
            output_path=f"generated_videos/video_{video_id}.mp4",
            title=script.get("script", {}).get("title", ""),
            subtitles=script.get("script", {}).get("subtitles", "")
        )
        
        # Update video record
        with next(get_db()) as db:
            video = db.query(Video).filter(Video.id == video_id).first()
            if video:
                video.file_path = result["path"]
                video.duration = result.get("duration", 0)
                video.status = VideoStatus.READY
                db.commit()
                
        return {
            "status": "success",
            "video_path": result["path"],
            "duration": result.get("duration"),
            "size_mb": result.get("size_mb")
        }
        
    except Exception as e:
        logger.error(f"Video creation failed: {e}")
        raise


@celery_app.task(name="pipeline.quality_check", priority=TaskPriority.NORMAL)
def quality_check_task(video_path: str, video_id: int) -> Dict[str, Any]:
    """Perform quality checks on generated video"""
    try:
        logger.info(f"Running quality check for video {video_id}")
        
        # Trigger N8N quality check workflow
        n8n = get_n8n_integration()
        result = n8n.trigger_quality_check(
            video_id=video_id,
            video_path=video_path,
            strict_mode=True
        )
        
        quality_score = result.get("overall_score", 0)
        
        # Update video with quality score
        with next(get_db()) as db:
            video = db.query(Video).filter(Video.id == video_id).first()
            if video:
                video.quality_score = quality_score
                db.commit()
                
        return {
            "status": "success",
            "quality_score": quality_score,
            "checks": result.get("checks", {}),
            "passed": quality_score >= 75
        }
        
    except Exception as e:
        logger.error(f"Quality check failed: {e}")
        raise


@celery_app.task(name="pipeline.upload_youtube", priority=TaskPriority.HIGH)
def upload_youtube_task(
    video_path: str,
    video_id: int,
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Upload video to YouTube with multi-account rotation"""
    try:
        logger.info(f"Uploading video {video_id} to YouTube")
        
        youtube_manager = get_youtube_manager()
        
        result = youtube_manager.upload_video_with_rotation(
            video_path=video_path,
            metadata=metadata
        )
        
        if not result:
            raise ValueError("YouTube upload failed - no accounts available")
            
        # Update video record
        with next(get_db()) as db:
            video = db.query(Video).filter(Video.id == video_id).first()
            if video:
                video.youtube_id = result["video_id"]
                video.youtube_url = f"https://youtube.com/watch?v={result['video_id']}"
                video.status = VideoStatus.PUBLISHED
                video.published_at = datetime.utcnow()
                db.commit()
                
        return {
            "status": "success",
            "youtube_id": result["video_id"],
            "youtube_url": result.get("youtube_url"),
            "account_used": result.get("account_used")
        }
        
    except Exception as e:
        logger.error(f"YouTube upload failed: {e}")
        raise


# Pipeline Orchestration

@celery_app.task(name="pipeline.orchestrate_full", priority=TaskPriority.HIGH)
def orchestrate_full_pipeline(
    channel_id: int,
    topic: str = None,
    style: str = "educational",
    auto_upload: bool = True
) -> Dict[str, Any]:
    """Orchestrate complete video generation pipeline"""
    try:
        logger.info(f"Starting full pipeline for channel {channel_id}")
        
        # Create video record
        with next(get_db()) as db:
            video = Video(
                channel_id=channel_id,
                title=topic or "Auto-generated",
                status=VideoStatus.PROCESSING,
                started_at=datetime.utcnow()
            )
            db.add(video)
            db.commit()
            db.refresh(video)
            video_id = video.id
            
        # Build pipeline chain
        pipeline = chain(
            # Stage 1: Trend Analysis & Topic Selection
            analyze_trends.s(channel_id),
            select_topic.s(),
            
            # Stage 2: Content Generation (parallel)
            group(
                generate_script_task.s(style=style),
                generate_voice_task.s(),
                generate_visuals_task.s()
            ),
            
            # Stage 3: Video Assembly
            create_video_task.s(video_id=video_id),
            
            # Stage 4: Quality & Upload
            quality_check_task.s(video_id=video_id)
        )
        
        if auto_upload:
            pipeline |= upload_youtube_task.s(video_id=video_id)
            
        # Execute pipeline
        result = pipeline.apply_async()
        
        return {
            "status": "started",
            "video_id": video_id,
            "pipeline_id": result.id,
            "stages": [
                "trend_analysis",
                "topic_selection",
                "content_generation",
                "video_assembly",
                "quality_check",
                "youtube_upload" if auto_upload else None
            ]
        }
        
    except Exception as e:
        logger.error(f"Pipeline orchestration failed: {e}")
        raise


@celery_app.task(name="pipeline.batch_generate", priority=TaskPriority.NORMAL)
def batch_generate_videos(
    channel_id: int,
    count: int = 5,
    style: str = "educational",
    schedule_hours: int = 24
) -> Dict[str, Any]:
    """Generate multiple videos in batch with scheduling"""
    try:
        logger.info(f"Batch generating {count} videos for channel {channel_id}")
        
        tasks = []
        schedule_interval = schedule_hours / count
        
        for i in range(count):
            # Schedule each video generation
            eta = datetime.utcnow() + timedelta(hours=i * schedule_interval)
            
            task = orchestrate_full_pipeline.apply_async(
                args=[channel_id],
                kwargs={"style": style, "auto_upload": True},
                eta=eta,
                priority=TaskPriority.NORMAL
            )
            
            tasks.append({
                "task_id": task.id,
                "scheduled_time": eta.isoformat(),
                "position": i + 1
            })
            
        return {
            "status": "scheduled",
            "channel_id": channel_id,
            "total_videos": count,
            "tasks": tasks,
            "completion_time": (datetime.utcnow() + timedelta(hours=schedule_hours)).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        raise


@celery_app.task(name="pipeline.get_status", priority=TaskPriority.LOW)
def get_pipeline_status(pipeline_id: str) -> Dict[str, Any]:
    """Get status of a pipeline execution"""
    try:
        result = AsyncResult(pipeline_id, app=celery_app)
        
        status = {
            "pipeline_id": pipeline_id,
            "state": result.state,
            "ready": result.ready(),
            "successful": result.successful() if result.ready() else None,
            "result": result.result if result.ready() and result.successful() else None,
            "error": str(result.info) if result.failed() else None
        }
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get pipeline status: {e}")
        return {
            "pipeline_id": pipeline_id,
            "state": "UNKNOWN",
            "error": str(e)
        }