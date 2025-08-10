"""
Video Processing Pipeline
Owner: Data Pipeline Engineer #1

Complete Celery task chain for video generation and processing.
Handles the entire lifecycle from content generation to YouTube upload.
"""

from celery import chain, group, chord
from celery.exceptions import Retry
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta
import json

from app.core.celery_app import celery_app
from app.core.database import get_db
from app.models.video import Video, VideoStatus
from app.models.channel import Channel
from app.tasks.content_generation import generate_content_task
from app.tasks.audio_synthesis import synthesize_audio_task
from app.tasks.video_compilation import compile_video_task
from app.tasks.youtube_upload import upload_to_youtube_task
from app.core.config import settings

logger = logging.getLogger(__name__)


class VideoPipelineError(Exception):
    """Custom exception for video pipeline errors."""
    pass


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 60})
def start_video_generation(self, video_request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize video generation pipeline.
    
    Args:
        video_request: Dictionary containing video generation parameters
        
    Returns:
        Dictionary with pipeline initialization data
    """
    try:
        logger.info(f"Starting video generation pipeline for request: {video_request.get('id')}")
        
        # Validate request
        required_fields = ['id', 'channel_id', 'topic', 'user_id']
        for field in required_fields:
            if field not in video_request:
                raise VideoPipelineError(f"Missing required field: {field}")
        
        # Initialize video record in database
        db = next(get_db())
        video = Video(
            id=video_request['id'],
            channel_id=video_request['channel_id'],
            user_id=video_request['user_id'],
            topic=video_request['topic'],
            status=VideoStatus.PROCESSING,
            created_at=datetime.utcnow(),
            pipeline_started_at=datetime.utcnow()
        )
        db.add(video)
        db.commit()
        
        # Update request with pipeline metadata
        pipeline_data = {
            **video_request,
            'pipeline_id': self.request.id,
            'started_at': datetime.utcnow().isoformat(),
            'status': 'processing',
            'stage': 'initialized',
            'costs': {
                'estimated_total': 0.0,
                'actual_total': 0.0,
                'budget_limit': settings.MAX_COST_PER_VIDEO
            }
        }
        
        logger.info(f"Video generation pipeline initialized: {pipeline_data['id']}")
        return pipeline_data
        
    except Exception as e:
        logger.error(f"Failed to initialize video pipeline: {str(e)}")
        self.retry(countdown=60, max_retries=3)


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 2, 'countdown': 120})
def update_video_status(self, video_id: str, status: str, stage: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Update video status in database and notify frontend.
    
    Args:
        video_id: Video identifier
        status: Current processing status
        stage: Current processing stage
        metadata: Additional metadata to store
        
    Returns:
        Updated video status
    """
    try:
        db = next(get_db())
        video = db.query(Video).filter(Video.id == video_id).first()
        
        if not video:
            raise VideoPipelineError(f"Video not found: {video_id}")
        
        # Update status
        video.status = VideoStatus(status) if isinstance(status, str) else status
        video.current_stage = stage
        video.updated_at = datetime.utcnow()
        
        if metadata:
            video.metadata = {**(video.metadata or {}), **metadata}
        
        db.commit()
        
        # Send real-time notification (WebSocket)
        notification_data = {
            'video_id': video_id,
            'status': status,
            'stage': stage,
            'timestamp': datetime.utcnow().isoformat(),
            'metadata': metadata
        }
        
        # TODO: Send WebSocket notification
        # websocket_manager.broadcast(f"video_{video_id}", notification_data)
        
        logger.info(f"Updated video {video_id} status: {status} - {stage}")
        return notification_data
        
    except Exception as e:
        logger.error(f"Failed to update video status: {str(e)}")
        raise


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 300})
def execute_video_pipeline(self, pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the complete video generation pipeline using Celery chains.
    
    Args:
        pipeline_data: Pipeline configuration and video data
        
    Returns:
        Final pipeline result
    """
    try:
        video_id = pipeline_data['id']
        logger.info(f"Executing video pipeline for: {video_id}")
        
        # Update status to processing
        update_video_status.delay(video_id, 'processing', 'content_generation')
        
        # Create task chain for video pipeline
        pipeline_chain = chain(
            # Stage 1: Content Generation
            generate_content_task.s(pipeline_data),
            
            # Stage 2: Audio and Visual Generation (Parallel)
            chord(
                group([
                    synthesize_audio_task.s(),
                    generate_visuals_task.s()
                ]),
                # Stage 3: Video Compilation (after audio and visuals complete)
                compile_video_task.s()
            ),
            
            # Stage 4: YouTube Upload
            upload_to_youtube_task.s(),
            
            # Stage 5: Finalization
            finalize_video_pipeline.s()
        )
        
        # Execute the pipeline
        result = pipeline_chain.apply_async()
        
        # Store pipeline task ID for tracking
        update_video_status.delay(
            video_id, 
            'processing', 
            'pipeline_executing',
            {'pipeline_task_id': result.id}
        )
        
        return {
            'video_id': video_id,
            'pipeline_task_id': result.id,
            'status': 'pipeline_executing',
            'started_at': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to execute video pipeline: {str(e)}")
        # Update video status to failed
        update_video_status.delay(
            pipeline_data['id'], 
            'failed', 
            'pipeline_error',
            {'error': str(e), 'failed_at': datetime.utcnow().isoformat()}
        )
        raise


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 2, 'countdown': 180})
def generate_visuals_task(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate visual content (images, animations) for video.
    
    Args:
        content_data: Content generation results
        
    Returns:
        Visual generation results
    """
    try:
        video_id = content_data['video_id']
        logger.info(f"Generating visuals for video: {video_id}")
        
        # Update status
        update_video_status.delay(video_id, 'processing', 'visual_generation')
        
        # Extract visual prompts from content
        script = content_data.get('script', '')
        visual_prompts = content_data.get('visual_prompts', [])
        
        # Generate images using AI services
        from app.services.ai_service import generate_images
        
        visual_results = generate_images(
            prompts=visual_prompts,
            video_id=video_id,
            style=content_data.get('visual_style', 'realistic'),
            quality=content_data.get('visual_quality', 'standard')
        )
        
        # Calculate costs
        visual_cost = sum(img.get('cost', 0) for img in visual_results.get('images', []))
        
        result = {
            **content_data,
            'visuals': visual_results,
            'visual_cost': visual_cost,
            'visual_generation_completed': True
        }
        
        logger.info(f"Visual generation completed for video: {video_id}, cost: ${visual_cost:.2f}")
        return result
        
    except Exception as e:
        logger.error(f"Visual generation failed: {str(e)}")
        raise


@celery_app.task(bind=True)
def finalize_video_pipeline(self, upload_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Finalize video pipeline after successful completion.
    
    Args:
        upload_result: Results from YouTube upload
        
    Returns:
        Final pipeline results
    """
    try:
        video_id = upload_result['video_id']
        logger.info(f"Finalizing video pipeline for: {video_id}")
        
        # Calculate total costs
        total_cost = (
            upload_result.get('content_cost', 0) +
            upload_result.get('audio_cost', 0) +
            upload_result.get('visual_cost', 0) +
            upload_result.get('compilation_cost', 0)
        )
        
        # Update video status to completed
        db = next(get_db())
        video = db.query(Video).filter(Video.id == video_id).first()
        
        if video:
            video.status = VideoStatus.COMPLETED
            video.youtube_video_id = upload_result.get('youtube_video_id')
            video.total_cost = total_cost
            video.completed_at = datetime.utcnow()
            video.metadata = {
                **(video.metadata or {}),
                'final_costs': upload_result.get('costs', {}),
                'youtube_url': upload_result.get('youtube_url'),
                'processing_duration': (
                    datetime.utcnow() - video.pipeline_started_at
                ).total_seconds() if video.pipeline_started_at else None
            }
            db.commit()
        
        # Send completion notification
        update_video_status.delay(
            video_id, 
            'completed', 
            'pipeline_complete',
            {
                'youtube_video_id': upload_result.get('youtube_video_id'),
                'youtube_url': upload_result.get('youtube_url'),
                'total_cost': total_cost,
                'completed_at': datetime.utcnow().isoformat()
            }
        )
        
        # Check if cost exceeded budget
        if total_cost > settings.MAX_COST_PER_VIDEO:
            logger.warning(f"Video {video_id} exceeded budget: ${total_cost:.2f} > ${settings.MAX_COST_PER_VIDEO}")
            # TODO: Send budget alert
        
        final_result = {
            'video_id': video_id,
            'status': 'completed',
            'youtube_video_id': upload_result.get('youtube_video_id'),
            'youtube_url': upload_result.get('youtube_url'),
            'total_cost': total_cost,
            'processing_time': upload_result.get('processing_time'),
            'completed_at': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Video pipeline completed successfully: {video_id}")
        return final_result
        
    except Exception as e:
        logger.error(f"Failed to finalize video pipeline: {str(e)}")
        raise


@celery_app.task(bind=True)
def handle_pipeline_failure(self, video_id: str, error: str, stage: str) -> Dict[str, Any]:
    """
    Handle pipeline failures and cleanup.
    
    Args:
        video_id: Video identifier
        error: Error message
        stage: Stage where failure occurred
        
    Returns:
        Failure handling result
    """
    try:
        logger.error(f"Video pipeline failed for {video_id} at stage {stage}: {error}")
        
        # Update video status
        db = next(get_db())
        video = db.query(Video).filter(Video.id == video_id).first()
        
        if video:
            video.status = VideoStatus.FAILED
            video.error_message = error
            video.failed_at = datetime.utcnow()
            video.metadata = {
                **(video.metadata or {}),
                'failure_stage': stage,
                'failure_reason': error,
                'failed_at': datetime.utcnow().isoformat()
            }
            db.commit()
        
        # Send failure notification
        update_video_status.delay(
            video_id,
            'failed',
            f'failed_at_{stage}',
            {
                'error': error,
                'failed_at': datetime.utcnow().isoformat(),
                'stage': stage
            }
        )
        
        # TODO: Cleanup temporary files
        # cleanup_video_files.delay(video_id)
        
        return {
            'video_id': video_id,
            'status': 'failed',
            'error': error,
            'stage': stage,
            'handled_at': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to handle pipeline failure: {str(e)}")
        raise


# Pipeline helper functions
def create_video_pipeline(video_request: Dict[str, Any]) -> str:
    """
    Create and start a complete video generation pipeline.
    
    Args:
        video_request: Video generation request data
        
    Returns:
        Pipeline task ID
    """
    # Chain the entire pipeline
    pipeline = chain(
        start_video_generation.s(video_request),
        execute_video_pipeline.s()
    )
    
    result = pipeline.apply_async()
    return result.id


def get_pipeline_status(pipeline_task_id: str) -> Dict[str, Any]:
    """
    Get current status of video pipeline.
    
    Args:
        pipeline_task_id: Celery task ID
        
    Returns:
        Pipeline status information
    """
    task_result = celery_app.AsyncResult(pipeline_task_id)
    
    return {
        'task_id': pipeline_task_id,
        'status': task_result.status,
        'result': task_result.result if task_result.successful() else None,
        'error': str(task_result.info) if task_result.failed() else None,
        'progress': task_result.info if task_result.state == 'PROGRESS' else None
    }


def cancel_pipeline(pipeline_task_id: str) -> bool:
    """
    Cancel a running video pipeline.
    
    Args:
        pipeline_task_id: Celery task ID
        
    Returns:
        True if successfully cancelled
    """
    try:
        celery_app.control.revoke(pipeline_task_id, terminate=True)
        return True
    except Exception as e:
        logger.error(f"Failed to cancel pipeline {pipeline_task_id}: {str(e)}")
        return False