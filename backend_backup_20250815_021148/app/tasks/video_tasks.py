"""
Video Processing Tasks for YTEmpire
Handles all asynchronous video generation and processing
"""
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
from sqlalchemy.orm import Session

from app.core.celery_app import celery_app, TaskPriority
from app.db.session import get_db
from app.models.video import Video, VideoStatus
from app.models.cost import Cost
from app.services.ai_services import AIServiceManager
from app.services.video_processor import VideoProcessor
from app.services.youtube_service import YouTubeService
from app.services.cost_tracker import CostTracker
from app.websocket.manager import websocket_manager

logger = logging.getLogger(__name__)


class VideoTaskBase(Task):
    """Base class for video tasks with error handling"""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        logger.error(f"Task {task_id} failed: {exc}")
        
        # Update video status in database
        video_id = kwargs.get('video_id')
        if video_id:
            with next(get_db()) as db:
                video = db.query(Video).filter(Video.id == video_id).first()
                if video:
                    video.status = VideoStatus.FAILED
                    video.error_message = str(exc)
                    db.commit()
                    
                    # Send WebSocket notification
                    websocket_manager.send_video_update(video_id, {
                        'status': 'failed',
                        'error': str(exc)
                    })
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry"""
        logger.warning(f"Task {task_id} retrying: {exc}")
        
        video_id = kwargs.get('video_id')
        if video_id:
            websocket_manager.send_video_update(video_id, {
                'status': 'retrying',
                'attempt': self.request.retries + 1
            })


@celery_app.task(
    base=VideoTaskBase,
    bind=True,
    name="app.tasks.video_tasks.generate_video",
    max_retries=3,
    default_retry_delay=60,
    priority=TaskPriority.HIGH
)
def generate_video(self, video_id: int, user_id: int, channel_id: int, 
                  topic: str, style: str = "educational") -> Dict[str, Any]:
    """
    Main video generation task
    Orchestrates the entire video creation pipeline
    """
    try:
        logger.info(f"Starting video generation for video_id: {video_id}")
        
        with next(get_db()) as db:
            # Update video status
            video = db.query(Video).filter(Video.id == video_id).first()
            if not video:
                raise ValueError(f"Video {video_id} not found")
            
            video.status = VideoStatus.PROCESSING
            video.started_at = datetime.utcnow()
            db.commit()
            
            # Send WebSocket update
            websocket_manager.send_video_update(video_id, {
                'status': 'processing',
                'step': 'initializing'
            })
            
            # Initialize services
            ai_manager = AIServiceManager()
            video_processor = VideoProcessor()
            cost_tracker = CostTracker(db)
            
            # Step 1: Generate script
            logger.info("Generating script...")
            websocket_manager.send_video_update(video_id, {
                'status': 'processing',
                'step': 'generating_script',
                'progress': 10
            })
            
            script = ai_manager.generate_script(topic, style)
            cost_tracker.track_cost(
                video_id=video_id,
                service="openai",
                operation="script_generation",
                cost=script.get('cost', 0.10)
            )
            
            # Step 2: Generate voice
            logger.info("Generating voice...")
            websocket_manager.send_video_update(video_id, {
                'status': 'processing',
                'step': 'generating_voice',
                'progress': 30
            })
            
            audio_path = ai_manager.generate_voice(script['text'])
            cost_tracker.track_cost(
                video_id=video_id,
                service="elevenlabs",
                operation="voice_synthesis",
                cost=audio_path.get('cost', 0.05)
            )
            
            # Step 3: Generate visuals
            logger.info("Generating visuals...")
            websocket_manager.send_video_update(video_id, {
                'status': 'processing',
                'step': 'generating_visuals',
                'progress': 50
            })
            
            visuals = ai_manager.generate_visuals(script['scenes'])
            cost_tracker.track_cost(
                video_id=video_id,
                service="pexels",
                operation="visual_generation",
                cost=visuals.get('cost', 0.00)
            )
            
            # Step 4: Create video
            logger.info("Creating video...")
            websocket_manager.send_video_update(video_id, {
                'status': 'processing',
                'step': 'creating_video',
                'progress': 70
            })
            
            video_result = video_processor.create_video_with_audio(
                audio_path=audio_path['path'],
                visuals=visuals['clips'],
                output_path=f"generated_videos/video_{video_id}.mp4",
                title=topic,
                subtitles=script.get('subtitles', '')
            )
            
            # Step 5: Quality check
            logger.info("Quality check...")
            websocket_manager.send_video_update(video_id, {
                'status': 'processing',
                'step': 'quality_check',
                'progress': 85
            })
            
            quality_score = ai_manager.check_quality(video_result['path'])
            
            # Update video record
            video.file_path = video_result['path']
            video.duration = video_result.get('duration', 0)
            video.quality_score = quality_score
            video.status = VideoStatus.READY
            video.completed_at = datetime.utcnow()
            
            # Calculate total cost
            total_cost = cost_tracker.get_video_cost(video_id)
            video.generation_cost = total_cost
            
            db.commit()
            
            # Send final update
            websocket_manager.send_video_update(video_id, {
                'status': 'completed',
                'progress': 100,
                'video_url': video_result['path'],
                'cost': total_cost,
                'quality_score': quality_score
            })
            
            # Trigger upload task if auto-upload is enabled
            if video.auto_upload:
                upload_to_youtube.apply_async(
                    kwargs={'video_id': video_id},
                    priority=TaskPriority.NORMAL
                )
            
            logger.info(f"Video generation completed for video_id: {video_id}")
            
            return {
                'video_id': video_id,
                'status': 'completed',
                'path': video_result['path'],
                'cost': total_cost,
                'quality_score': quality_score,
                'duration': video_result.get('duration', 0)
            }
            
    except SoftTimeLimitExceeded:
        logger.error(f"Video generation timed out for video_id: {video_id}")
        raise self.retry(exc=Exception("Task timed out"))
    
    except Exception as e:
        logger.error(f"Video generation failed: {str(e)}")
        raise self.retry(exc=e)


@celery_app.task(
    base=VideoTaskBase,
    bind=True,
    name="app.tasks.video_tasks.upload_to_youtube",
    max_retries=5,
    default_retry_delay=120,
    priority=TaskPriority.NORMAL
)
def upload_to_youtube(self, video_id: int) -> Dict[str, Any]:
    """Upload video to YouTube"""
    try:
        logger.info(f"Starting YouTube upload for video_id: {video_id}")
        
        with next(get_db()) as db:
            video = db.query(Video).filter(Video.id == video_id).first()
            if not video:
                raise ValueError(f"Video {video_id} not found")
            
            if video.status != VideoStatus.READY:
                raise ValueError(f"Video {video_id} is not ready for upload")
            
            # Update status
            video.status = VideoStatus.UPLOADING
            db.commit()
            
            # Send WebSocket update
            websocket_manager.send_video_update(video_id, {
                'status': 'uploading',
                'platform': 'youtube'
            })
            
            # Initialize YouTube service
            youtube_service = YouTubeService()
            
            # Upload video
            result = youtube_service.upload_video(
                video_path=video.file_path,
                title=video.title,
                description=video.description,
                tags=video.tags,
                channel_id=video.channel_id
            )
            
            # Update video record
            video.youtube_id = result['video_id']
            video.youtube_url = result['url']
            video.status = VideoStatus.PUBLISHED
            video.published_at = datetime.utcnow()
            db.commit()
            
            # Send completion update
            websocket_manager.send_video_update(video_id, {
                'status': 'published',
                'youtube_url': result['url'],
                'youtube_id': result['video_id']
            })
            
            logger.info(f"YouTube upload completed for video_id: {video_id}")
            
            return {
                'video_id': video_id,
                'youtube_id': result['video_id'],
                'youtube_url': result['url'],
                'status': 'published'
            }
            
    except Exception as e:
        logger.error(f"YouTube upload failed: {str(e)}")
        raise self.retry(exc=e)


@celery_app.task(
    name="app.tasks.video_tasks.cleanup_old_videos",
    priority=TaskPriority.LOW
)
def cleanup_old_videos():
    """Periodic task to clean up old video files"""
    try:
        logger.info("Starting old video cleanup...")
        
        with next(get_db()) as db:
            # Find videos older than 30 days
            cutoff_date = datetime.utcnow() - timedelta(days=30)
            old_videos = db.query(Video).filter(
                Video.created_at < cutoff_date,
                Video.status.in_([VideoStatus.PUBLISHED, VideoStatus.FAILED])
            ).all()
            
            cleaned_count = 0
            for video in old_videos:
                if video.file_path and os.path.exists(video.file_path):
                    try:
                        os.remove(video.file_path)
                        video.file_path = None
                        cleaned_count += 1
                    except Exception as e:
                        logger.error(f"Failed to delete file {video.file_path}: {e}")
            
            db.commit()
            logger.info(f"Cleaned up {cleaned_count} old video files")
            
            return {'cleaned': cleaned_count}
            
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        raise


@celery_app.task(
    name="app.tasks.video_tasks.check_stuck_videos",
    priority=TaskPriority.LOW
)
def check_stuck_videos():
    """Check for videos stuck in processing state"""
    try:
        logger.info("Checking for stuck videos...")
        
        with next(get_db()) as db:
            # Find videos processing for more than 30 minutes
            timeout = datetime.utcnow() - timedelta(minutes=30)
            stuck_videos = db.query(Video).filter(
                Video.status == VideoStatus.PROCESSING,
                Video.started_at < timeout
            ).all()
            
            for video in stuck_videos:
                logger.warning(f"Video {video.id} appears stuck, marking as failed")
                video.status = VideoStatus.FAILED
                video.error_message = "Processing timeout"
                
                # Send notification
                websocket_manager.send_video_update(video.id, {
                    'status': 'failed',
                    'error': 'Processing timeout'
                })
            
            db.commit()
            
            return {'stuck_videos': len(stuck_videos)}
            
    except Exception as e:
        logger.error(f"Stuck video check failed: {str(e)}")
        raise


@celery_app.task(
    bind=True,
    name="app.tasks.video_tasks.batch_generate_videos",
    priority=TaskPriority.NORMAL
)
def batch_generate_videos(self, channel_id: int, topics: list, style: str = "educational"):
    """Generate multiple videos in batch"""
    try:
        logger.info(f"Starting batch generation for {len(topics)} videos")
        
        results = []
        for topic in topics:
            # Create video record
            with next(get_db()) as db:
                video = Video(
                    channel_id=channel_id,
                    title=topic,
                    status=VideoStatus.QUEUED
                )
                db.add(video)
                db.commit()
                db.refresh(video)
                
                # Queue generation task
                task = generate_video.apply_async(
                    kwargs={
                        'video_id': video.id,
                        'user_id': video.channel.user_id,
                        'channel_id': channel_id,
                        'topic': topic,
                        'style': style
                    },
                    priority=TaskPriority.NORMAL
                )
                
                results.append({
                    'video_id': video.id,
                    'task_id': task.id,
                    'topic': topic
                })
        
        return {'batch_id': self.request.id, 'videos': results}
        
    except Exception as e:
        logger.error(f"Batch generation failed: {str(e)}")
        raise