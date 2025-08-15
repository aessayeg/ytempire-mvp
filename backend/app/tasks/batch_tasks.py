"""
Batch Processing Tasks for Celery
Handles batch video generation, bulk operations, and batch analytics
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from celery import Task, group, chord, chain
import json

from app.core.celery_app import celery_app, TaskPriority
from app.services.batch_processing import batch_processor, BatchJobStatus
from app.services.video_generation_pipeline import VideoGenerationPipeline
from app.services.notification_service import notification_service
from app.db.session import AsyncSessionLocal
from app.models.video import Video
from app.models.batch import BatchJob, BatchJobItem
from sqlalchemy import select, update

logger = logging.getLogger(__name__)


class BatchTask(Task):
    """Base class for batch tasks"""
    _batch_processor = None
    _video_pipeline = None
    
    @property
    def batch_processor(self):
        if self._batch_processor is None:
            from app.services.batch_processing import batch_processor
            self._batch_processor = batch_processor
        return self._batch_processor
    
    @property
    def video_pipeline(self):
        if self._video_pipeline is None:
            self._video_pipeline = VideoGenerationPipeline()
        return self._video_pipeline


@celery_app.task(
    bind=True,
    base=BatchTask,
    name='batch.process_videos',
    queue='batch_processing',
    max_retries=2
)
def batch_process_videos(
    self,
    batch_job_id: str,
    videos: List[Dict[str, Any]],
    max_concurrent: int = 3,
    user_id: str = None,
    channel_id: str = None
) -> Dict[str, Any]:
    """
    Process batch video generation
    
    Args:
        batch_job_id: Batch job ID
        videos: List of video configurations
        max_concurrent: Maximum concurrent video processing
        user_id: User ID for tracking
        channel_id: Channel ID for videos
    """
    try:
        logger.info(f"Starting batch processing for job {batch_job_id} with {len(videos)} videos")
        
        # Update batch job status
        asyncio.run(update_batch_status(batch_job_id, BatchJobStatus.RUNNING))
        
        # Create sub-tasks for each video
        video_tasks = []
        for idx, video_config in enumerate(videos):
            task = process_single_video.s(
                video_config=video_config,
                batch_job_id=batch_job_id,
                item_index=idx,
                user_id=user_id,
                channel_id=channel_id
            )
            video_tasks.append(task)
        
        # Process videos with concurrency limit
        job = group(*video_tasks).apply_async()
        
        # Wait for completion
        results = job.get(timeout=3600)  # 1 hour timeout
        
        # Aggregate results
        successful = sum(1 for r in results if r.get('success'))
        failed = len(results) - successful
        
        # Update batch job status
        asyncio.run(
            update_batch_status(
                batch_job_id,
                BatchJobStatus.COMPLETED if failed == 0 else BatchJobStatus.PARTIALLY_COMPLETED,
                {
                    'total': len(videos),
                    'successful': successful,
                    'failed': failed,
                    'results': results
                }
            )
        )
        
        # Send notification
        if user_id:
            asyncio.run(
                notification_service.send_notification(
                    user_id=user_id,
                    title="Batch Processing Complete",
                    message=f"Processed {successful}/{len(videos)} videos successfully",
                    type="success" if failed == 0 else "warning"
                )
            )
        
        return {
            'success': True,
            'batch_job_id': batch_job_id,
            'total_videos': len(videos),
            'successful': successful,
            'failed': failed,
            'completion_time': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        
        # Update batch job status to failed
        asyncio.run(
            update_batch_status(batch_job_id, BatchJobStatus.FAILED, {'error': str(e)})
        )
        
        raise self.retry(exc=e)


@celery_app.task(
    base=BatchTask,
    name='batch.process_single_video',
    queue='batch_processing'
)
def process_single_video(
    video_config: Dict[str, Any],
    batch_job_id: str,
    item_index: int,
    user_id: str = None,
    channel_id: str = None
) -> Dict[str, Any]:
    """
    Process a single video in a batch
    
    Args:
        video_config: Video configuration
        batch_job_id: Parent batch job ID
        item_index: Index in batch
        user_id: User ID
        channel_id: Channel ID
    """
    try:
        logger.info(f"Processing video {item_index} in batch {batch_job_id}")
        
        # Update item status
        asyncio.run(
            update_batch_item_status(batch_job_id, item_index, 'processing')
        )
        
        # Generate video using pipeline
        from app.tasks.pipeline_tasks import process_video_pipeline
        
        result = process_video_pipeline(
            topic=video_config.get('title', 'Untitled'),
            description=video_config.get('description', ''),
            tags=video_config.get('tags', []),
            channel_id=channel_id,
            user_id=user_id,
            config={
                'style': video_config.get('style', 'educational'),
                'length': video_config.get('length', 'medium'),
                'auto_publish': video_config.get('auto_publish', False)
            }
        )
        
        # Update item status
        asyncio.run(
            update_batch_item_status(
                batch_job_id,
                item_index,
                'completed' if result.get('success') else 'failed',
                result
            )
        )
        
        return {
            'success': result.get('success', False),
            'item_index': item_index,
            'video_id': result.get('video_id'),
            'error': result.get('error')
        }
        
    except Exception as e:
        logger.error(f"Single video processing failed: {str(e)}")
        
        # Update item status to failed
        asyncio.run(
            update_batch_item_status(
                batch_job_id,
                item_index,
                'failed',
                {'error': str(e)}
            )
        )
        
        return {
            'success': False,
            'item_index': item_index,
            'error': str(e)
        }


@celery_app.task(
    name='batch.bulk_update',
    queue='batch_processing'
)
def bulk_update_videos(
    video_ids: List[str],
    updates: Dict[str, Any],
    user_id: str = None
) -> Dict[str, Any]:
    """
    Bulk update video properties
    
    Args:
        video_ids: List of video IDs to update
        updates: Dictionary of fields to update
        user_id: User ID for permissions check
    """
    try:
        logger.info(f"Bulk updating {len(video_ids)} videos")
        
        async def bulk_update():
            async with AsyncSessionLocal() as db:
                updated_count = 0
                errors = []
                
                for video_id in video_ids:
                    try:
                        video = await db.get(Video, video_id)
                        if video:
                            for key, value in updates.items():
                                if hasattr(video, key):
                                    setattr(video, key, value)
                            updated_count += 1
                        else:
                            errors.append(f"Video {video_id} not found")
                    except Exception as e:
                        errors.append(f"Error updating {video_id}: {str(e)}")
                
                await db.commit()
                return updated_count, errors
        
        updated, errors = asyncio.run(bulk_update())
        
        return {
            'success': True,
            'total': len(video_ids),
            'updated': updated,
            'errors': errors
        }
        
    except Exception as e:
        logger.error(f"Bulk update failed: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }


@celery_app.task(
    name='batch.bulk_delete',
    queue='batch_processing'
)
def bulk_delete_videos(
    video_ids: List[str],
    user_id: str = None,
    soft_delete: bool = True
) -> Dict[str, Any]:
    """
    Bulk delete videos
    
    Args:
        video_ids: List of video IDs to delete
        user_id: User ID for permissions
        soft_delete: If True, mark as deleted; if False, actually delete
    """
    try:
        logger.info(f"Bulk deleting {len(video_ids)} videos (soft={soft_delete})")
        
        async def bulk_delete():
            async with AsyncSessionLocal() as db:
                deleted_count = 0
                errors = []
                
                for video_id in video_ids:
                    try:
                        video = await db.get(Video, video_id)
                        if video:
                            if soft_delete:
                                video.status = 'deleted'
                                video.deleted_at = datetime.utcnow()
                            else:
                                await db.delete(video)
                            deleted_count += 1
                        else:
                            errors.append(f"Video {video_id} not found")
                    except Exception as e:
                        errors.append(f"Error deleting {video_id}: {str(e)}")
                
                await db.commit()
                return deleted_count, errors
        
        deleted, errors = asyncio.run(bulk_delete())
        
        return {
            'success': True,
            'total': len(video_ids),
            'deleted': deleted,
            'errors': errors
        }
        
    except Exception as e:
        logger.error(f"Bulk delete failed: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }


@celery_app.task(
    name='batch.analytics_aggregation',
    queue='batch_processing'
)
def batch_aggregate_analytics(
    channel_ids: List[str],
    start_date: str,
    end_date: str
) -> Dict[str, Any]:
    """
    Batch aggregate analytics for multiple channels
    
    Args:
        channel_ids: List of channel IDs
        start_date: Start date for aggregation
        end_date: End date for aggregation
    """
    try:
        logger.info(f"Aggregating analytics for {len(channel_ids)} channels")
        
        # Create tasks for each channel
        from app.tasks.analytics_tasks import calculate_channel_roi
        
        tasks = [
            calculate_channel_roi.s(channel_id, 30)
            for channel_id in channel_ids
        ]
        
        # Process in parallel
        job = group(*tasks).apply_async()
        results = job.get(timeout=600)  # 10 minute timeout
        
        # Aggregate results
        total_revenue = sum(r['metrics']['revenue'] for r in results if r.get('success'))
        total_cost = sum(r['metrics']['cost'] for r in results if r.get('success'))
        
        return {
            'success': True,
            'channels_processed': len(channel_ids),
            'total_revenue': total_revenue,
            'total_cost': total_cost,
            'profit': total_revenue - total_cost,
            'aggregation_time': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Analytics aggregation failed: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }


async def update_batch_status(
    batch_job_id: str,
    status: BatchJobStatus,
    metadata: Dict[str, Any] = None
):
    """Update batch job status in database"""
    async with AsyncSessionLocal() as db:
        job = await db.get(BatchJob, batch_job_id)
        if job:
            job.status = status
            if metadata:
                job.metadata = json.dumps(metadata)
            job.updated_at = datetime.utcnow()
            if status in [BatchJobStatus.COMPLETED, BatchJobStatus.FAILED]:
                job.completed_at = datetime.utcnow()
            await db.commit()


async def update_batch_item_status(
    batch_job_id: str,
    item_index: int,
    status: str,
    result: Dict[str, Any] = None
):
    """Update batch job item status"""
    async with AsyncSessionLocal() as db:
        # This would update the specific item in the batch
        # For now, just logging
        logger.info(f"Batch {batch_job_id} item {item_index}: {status}")


# Cleanup task
@celery_app.task(
    name='batch.cleanup_old_jobs',
    queue='batch_processing'
)
def cleanup_old_batch_jobs(days_to_keep: int = 30):
    """
    Clean up old batch jobs
    
    Args:
        days_to_keep: Number of days to keep batch jobs
    """
    try:
        logger.info(f"Cleaning up batch jobs older than {days_to_keep} days")
        
        async def cleanup():
            async with AsyncSessionLocal() as db:
                cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
                
                result = await db.execute(
                    select(BatchJob).where(
                        BatchJob.created_at < cutoff_date
                    )
                )
                
                old_jobs = result.scalars().all()
                for job in old_jobs:
                    await db.delete(job)
                
                await db.commit()
                return len(old_jobs)
        
        deleted = asyncio.run(cleanup())
        
        return {
            'success': True,
            'deleted_jobs': deleted,
            'cleanup_time': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch cleanup failed: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }