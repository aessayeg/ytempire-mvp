"""
Batch Processing Framework for YTEmpire
Handles batch operations for video generation, data processing, and system maintenance
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager

from app.core.config import settings
from app.db.session import get_db
from app.services.cost_tracking import cost_tracker
from app.services.notification_service import notification_service, NotificationType
from app.services.websocket_manager import ConnectionManager

logger = logging.getLogger(__name__)


class BatchJobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class BatchJobType(str, Enum):
    VIDEO_GENERATION = "video_generation"
    DATA_PROCESSING = "data_processing"
    ANALYTICS_AGGREGATION = "analytics_aggregation"
    COST_AGGREGATION = "cost_aggregation"
    SYSTEM_MAINTENANCE = "system_maintenance"
    NOTIFICATION_BATCH = "notification_batch"
    BACKUP_OPERATION = "backup_operation"
    THUMBNAIL_GENERATION = "thumbnail_generation"
    CHANNEL_SYNC = "channel_sync"
    CONTENT_OPTIMIZATION = "content_optimization"
    REPORT_GENERATION = "report_generation"


@dataclass
class BatchJobItem:
    id: str
    data: Dict[str, Any]
    status: BatchJobStatus = BatchJobStatus.PENDING
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class BatchJobConfig:
    job_id: str
    job_type: BatchJobType
    items: List[BatchJobItem]
    max_concurrent: int = 5
    max_retries: int = 3
    retry_delay: int = 60  # seconds
    timeout_per_item: int = 300  # seconds
    callback_url: Optional[str] = None
    notification_user_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    priority: int = 5  # 1-10, higher is more priority
    scheduled_at: Optional[datetime] = None
    dependencies: List[str] = None  # List of job IDs that must complete first
    checkpoint_interval: int = 10  # Save progress every N items
    resume_on_failure: bool = True


class BatchProcessor:
    def __init__(self):
        self.running_jobs: Dict[str, BatchJobConfig] = {}
        self.job_history: List[Dict[str, Any]] = []
        self.websocket_manager = ConnectionManager()
        self.job_queue: List[BatchJobConfig] = []  # Priority queue for scheduled jobs
        self.job_checkpoints: Dict[str, Dict[str, Any]] = {}  # Checkpoint storage
        self.resource_limits = {
            "max_concurrent_jobs": 10,
            "max_items_per_job": 1000,
            "max_memory_mb": 4096,
        }
        self.metrics = {
            "total_jobs_processed": 0,
            "total_items_processed": 0,
            "average_processing_time": 0,
            "failure_rate": 0,
        }

    async def submit_batch_job(
        self, job_config: BatchJobConfig, processor_function: Callable
    ) -> str:
        """Submit a batch job for processing with enhanced scheduling"""
        try:
            # Validate resource limits
            if len(job_config.items) > self.resource_limits["max_items_per_job"]:
                raise ValueError(
                    f"Job exceeds maximum items limit of {self.resource_limits['max_items_per_job']}"
                )

            if len(self.running_jobs) >= self.resource_limits["max_concurrent_jobs"]:
                # Queue the job for later processing
                self.job_queue.append(job_config)
                self.job_queue.sort(
                    key=lambda x: (x.priority, x.scheduled_at or datetime.min),
                    reverse=True,
                )
                logger.info(f"Job {job_config.job_id} queued due to resource limits")
                return job_config.job_id

            job_config.metadata = job_config.metadata or {}
            job_config.metadata.update(
                {
                    "created_at": datetime.utcnow().isoformat(),
                    "total_items": len(job_config.items),
                    "priority": job_config.priority,
                }
            )

            # Check dependencies
            if job_config.dependencies:
                for dep_job_id in job_config.dependencies:
                    dep_status = await self.get_job_status(dep_job_id)
                    if (
                        not dep_status
                        or dep_status.get("status") != BatchJobStatus.COMPLETED
                    ):
                        # Queue job until dependencies are met
                        self.job_queue.append(job_config)
                        logger.info(
                            f"Job {job_config.job_id} waiting for dependencies: {job_config.dependencies}"
                        )
                        return job_config.job_id

            # Check if job should be scheduled for later
            if job_config.scheduled_at and job_config.scheduled_at > datetime.utcnow():
                self.job_queue.append(job_config)
                logger.info(
                    f"Job {job_config.job_id} scheduled for {job_config.scheduled_at}"
                )
                return job_config.job_id

            self.running_jobs[job_config.job_id] = job_config

            # Start processing in background
            asyncio.create_task(self._process_batch_job(job_config, processor_function))

            logger.info(
                f"Submitted batch job {job_config.job_id} with {len(job_config.items)} items, priority {job_config.priority}"
            )

            return job_config.job_id

        except Exception as e:
            logger.error(f"Failed to submit batch job: {str(e)}")
            raise

    async def _process_batch_job(
        self, job_config: BatchJobConfig, processor_function: Callable
    ):
        """Process a batch job with enhanced features"""
        job_id = job_config.job_id
        start_time = datetime.utcnow()

        try:
            logger.info(
                f"Starting batch job {job_id} with priority {job_config.priority}"
            )

            # Update job status
            await self._update_job_status(
                job_id, {"status": "running", "started_at": start_time}
            )

            # Check for checkpoint to resume from
            checkpoint = self.job_checkpoints.get(job_id, {})
            start_index = (
                checkpoint.get("last_processed_index", 0)
                if job_config.resume_on_failure
                else 0
            )

            # Process items with concurrency control
            semaphore = asyncio.Semaphore(job_config.max_concurrent)

            tasks = []
            processed_count = 0

            for i, item in enumerate(job_config.items[start_index:], start=start_index):
                # Skip already completed items from checkpoint
                if checkpoint and item.id in checkpoint.get("completed_items", []):
                    continue

                task = self._process_batch_item(
                    item, processor_function, semaphore, job_config
                )
                tasks.append(task)

                # Save checkpoint at intervals
                processed_count += 1
                if processed_count % job_config.checkpoint_interval == 0:
                    await self._save_checkpoint(job_id, i, job_config)

            # Wait for all items to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Calculate final statistics
            completed_items = sum(1 for r in results if not isinstance(r, Exception))
            failed_items = len(results) - completed_items

            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()

            # Update metrics
            self.metrics["total_jobs_processed"] += 1
            self.metrics["total_items_processed"] += completed_items
            self.metrics["average_processing_time"] = (
                self.metrics["average_processing_time"]
                * (self.metrics["total_jobs_processed"] - 1)
                + duration
            ) / self.metrics["total_jobs_processed"]
            self.metrics["failure_rate"] = (
                self.metrics["failure_rate"]
                * (self.metrics["total_jobs_processed"] - 1)
                + (failed_items / len(job_config.items))
            ) / self.metrics["total_jobs_processed"]

            final_status = {
                "status": BatchJobStatus.COMPLETED
                if failed_items == 0
                else BatchJobStatus.FAILED,
                "completed_at": end_time,
                "duration_seconds": duration,
                "total_items": len(job_config.items),
                "completed_items": completed_items,
                "failed_items": failed_items,
                "success_rate": completed_items / len(job_config.items) * 100,
                "items_per_second": completed_items / duration if duration > 0 else 0,
            }

            await self._update_job_status(job_id, final_status)

            # Call webhook if configured
            if job_config.callback_url:
                await self._call_webhook(job_config.callback_url, final_status)

            # Send completion notification
            if job_config.notification_user_id:
                await self._send_job_completion_notification(job_config, final_status)

            # Clean up checkpoint
            if job_id in self.job_checkpoints:
                del self.job_checkpoints[job_id]

            # Move to history and remove from running jobs
            self.job_history.append(
                {
                    "job_id": job_id,
                    "job_type": job_config.job_type.value,
                    **final_status,
                    "metadata": job_config.metadata,
                }
            )

            if job_id in self.running_jobs:
                del self.running_jobs[job_id]

            # Process queued jobs if any
            await self._process_queued_jobs()

            logger.info(
                f"Completed batch job {job_id}: {completed_items}/{len(job_config.items)} successful"
            )

        except Exception as e:
            logger.error(f"Batch job {job_id} failed: {str(e)}")
            await self._update_job_status(
                job_id,
                {
                    "status": BatchJobStatus.FAILED,
                    "error": str(e),
                    "completed_at": datetime.utcnow(),
                },
            )

    async def _process_batch_item(
        self,
        item: BatchJobItem,
        processor_function: Callable,
        semaphore: asyncio.Semaphore,
        job_config: BatchJobConfig,
    ):
        """Process a single batch item with retry logic"""
        async with semaphore:
            retries = 0

            while retries <= job_config.max_retries:
                try:
                    item.started_at = datetime.utcnow()
                    item.status = BatchJobStatus.RUNNING

                    # Apply timeout
                    result = await asyncio.wait_for(
                        processor_function(item.data),
                        timeout=job_config.timeout_per_item,
                    )

                    item.result = result
                    item.status = BatchJobStatus.COMPLETED
                    item.completed_at = datetime.utcnow()

                    logger.debug(f"Completed batch item {item.id}")
                    return result

                except asyncio.TimeoutError:
                    error_msg = (
                        f"Item {item.id} timed out after {job_config.timeout_per_item}s"
                    )
                    logger.warning(error_msg)
                    item.error = error_msg

                except Exception as e:
                    error_msg = f"Item {item.id} failed: {str(e)}"
                    logger.warning(error_msg)
                    item.error = error_msg

                retries += 1

                if retries <= job_config.max_retries:
                    logger.info(
                        f"Retrying item {item.id}, attempt {retries}/{job_config.max_retries}"
                    )
                    await asyncio.sleep(job_config.retry_delay)
                else:
                    item.status = BatchJobStatus.FAILED
                    item.completed_at = datetime.utcnow()
                    logger.error(
                        f"Item {item.id} failed after {job_config.max_retries} retries"
                    )
                    return None

    async def _update_job_status(self, job_id: str, status_update: Dict[str, Any]):
        """Update job status and notify via WebSocket"""
        try:
            if job_id in self.running_jobs:
                job_config = self.running_jobs[job_id]
                job_config.metadata.update(status_update)

                # Send real-time update via WebSocket
                if job_config.notification_user_id:
                    await self.websocket_manager.send_personal_message(
                        json.dumps(
                            {
                                "type": "batch_job_update",
                                "job_id": job_id,
                                "job_type": job_config.job_type.value,
                                **status_update,
                            }
                        ),
                        job_config.notification_user_id,
                    )

        except Exception as e:
            logger.error(f"Failed to update job status for {job_id}: {str(e)}")

    async def _send_job_completion_notification(
        self, job_config: BatchJobConfig, final_status: Dict[str, Any]
    ):
        """Send job completion notification"""
        try:
            success_rate = final_status.get("success_rate", 0)
            template_id = (
                "batch_job_complete_success"
                if success_rate >= 90
                else "batch_job_complete_partial"
            )

            await notification_service.send_template_notification(
                user_id=job_config.notification_user_id,
                template_id=template_id,
                variables={
                    "job_type": job_config.job_type.value,
                    "total_items": str(final_status["total_items"]),
                    "completed_items": str(final_status["completed_items"]),
                    "failed_items": str(final_status["failed_items"]),
                    "success_rate": f"{success_rate:.1f}",
                    "duration": f"{final_status['duration_seconds']:.1f}",
                },
                notification_types=[NotificationType.IN_APP],
            )

        except Exception as e:
            logger.error(f"Failed to send job completion notification: {str(e)}")

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a batch job"""
        if job_id in self.running_jobs:
            job_config = self.running_jobs[job_id]
            return {
                "job_id": job_id,
                "job_type": job_config.job_type.value,
                "status": job_config.metadata.get("status", "pending"),
                "total_items": len(job_config.items),
                "completed_items": sum(
                    1
                    for item in job_config.items
                    if item.status == BatchJobStatus.COMPLETED
                ),
                "failed_items": sum(
                    1
                    for item in job_config.items
                    if item.status == BatchJobStatus.FAILED
                ),
                "metadata": job_config.metadata,
            }

        # Check history
        for job in self.job_history:
            if job["job_id"] == job_id:
                return job

        return None

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running batch job"""
        if job_id in self.running_jobs:
            job_config = self.running_jobs[job_id]
            job_config.metadata["status"] = BatchJobStatus.CANCELLED
            job_config.metadata["cancelled_at"] = datetime.utcnow().isoformat()

            # Mark pending items as cancelled
            for item in job_config.items:
                if item.status == BatchJobStatus.PENDING:
                    item.status = BatchJobStatus.CANCELLED

            logger.info(f"Cancelled batch job {job_id}")
            return True

        return False

    async def get_all_jobs(self) -> Dict[str, Any]:
        """Get status of all jobs with enhanced details"""
        running = []
        for job_id, job_config in self.running_jobs.items():
            status = await self.get_job_status(job_id)
            if status:
                running.append(status)

        queued = []
        for job_config in self.job_queue:
            queued.append(
                {
                    "job_id": job_config.job_id,
                    "job_type": job_config.job_type.value,
                    "priority": job_config.priority,
                    "scheduled_at": job_config.scheduled_at.isoformat()
                    if job_config.scheduled_at
                    else None,
                    "dependencies": job_config.dependencies,
                    "total_items": len(job_config.items),
                }
            )

        return {
            "running_jobs": running,
            "queued_jobs": queued,
            "completed_jobs": self.job_history[-50:],  # Last 50 completed jobs
            "total_running": len(running),
            "total_queued": len(queued),
            "total_history": len(self.job_history),
            "metrics": self.metrics,
            "resource_limits": self.resource_limits,
        }

    async def _save_checkpoint(
        self, job_id: str, last_index: int, job_config: BatchJobConfig
    ):
        """Save checkpoint for job recovery"""
        try:
            completed_items = [
                item.id
                for item in job_config.items
                if item.status == BatchJobStatus.COMPLETED
            ]
            self.job_checkpoints[job_id] = {
                "last_processed_index": last_index,
                "completed_items": completed_items,
                "saved_at": datetime.utcnow().isoformat(),
            }
            logger.debug(f"Saved checkpoint for job {job_id} at index {last_index}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint for {job_id}: {str(e)}")

    async def _process_queued_jobs(self):
        """Process jobs from the queue when resources are available"""
        try:
            while (
                self.job_queue
                and len(self.running_jobs) < self.resource_limits["max_concurrent_jobs"]
            ):
                # Get next job from queue
                next_job = None
                for i, job in enumerate(self.job_queue):
                    # Check if scheduled time has arrived
                    if job.scheduled_at and job.scheduled_at > datetime.utcnow():
                        continue

                    # Check if dependencies are met
                    if job.dependencies:
                        deps_met = True
                        for dep_id in job.dependencies:
                            dep_status = await self.get_job_status(dep_id)
                            if (
                                not dep_status
                                or dep_status.get("status") != BatchJobStatus.COMPLETED
                            ):
                                deps_met = False
                                break
                        if not deps_met:
                            continue

                    # This job can be processed
                    next_job = self.job_queue.pop(i)
                    break

                if next_job:
                    # Process the job
                    self.running_jobs[next_job.job_id] = next_job
                    # Note: processor_function needs to be stored with the job
                    logger.info(f"Processing queued job {next_job.job_id}")
                else:
                    break

        except Exception as e:
            logger.error(f"Failed to process queued jobs: {str(e)}")

    async def _call_webhook(self, webhook_url: str, data: Dict[str, Any]):
        """Call webhook with job completion data"""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=data, timeout=30) as response:
                    if response.status == 200:
                        logger.info(f"Webhook called successfully: {webhook_url}")
                    else:
                        logger.warning(
                            f"Webhook returned status {response.status}: {webhook_url}"
                        )
        except Exception as e:
            logger.error(f"Failed to call webhook {webhook_url}: {str(e)}")

    async def pause_job(self, job_id: str) -> bool:
        """Pause a running batch job"""
        if job_id in self.running_jobs:
            job_config = self.running_jobs[job_id]
            job_config.metadata["status"] = BatchJobStatus.PAUSED
            job_config.metadata["paused_at"] = datetime.utcnow().isoformat()

            # Save current state as checkpoint
            await self._save_checkpoint(job_id, 0, job_config)

            logger.info(f"Paused batch job {job_id}")
            return True
        return False

    async def resume_job(self, job_id: str) -> bool:
        """Resume a paused batch job"""
        if job_id in self.running_jobs:
            job_config = self.running_jobs[job_id]
            if job_config.metadata.get("status") == BatchJobStatus.PAUSED:
                job_config.metadata["status"] = BatchJobStatus.RUNNING
                job_config.metadata["resumed_at"] = datetime.utcnow().isoformat()

                # Continue processing from checkpoint
                # Note: This would need the processor_function stored with the job
                logger.info(f"Resumed batch job {job_id}")
                return True
        return False

    async def get_job_metrics(self) -> Dict[str, Any]:
        """Get overall batch processing metrics"""
        return {
            "system_metrics": self.metrics,
            "active_jobs": len(self.running_jobs),
            "queued_jobs": len(self.job_queue),
            "total_processed": len(self.job_history),
            "resource_usage": {
                "jobs_capacity": f"{len(self.running_jobs)}/{self.resource_limits['max_concurrent_jobs']}",
                "queue_depth": len(self.job_queue),
            },
        }


# Specific batch processors for common operations


class VideoBatchProcessor:
    def __init__(self, batch_processor: BatchProcessor):
        self.batch_processor = batch_processor

    async def batch_generate_videos(
        self,
        video_requests: List[Dict[str, Any]],
        user_id: str,
        max_concurrent: int = 3,
    ) -> str:
        """Batch generate multiple videos"""

        job_id = f"video_batch_{uuid.uuid4().hex[:8]}"

        # Create batch items
        items = []
        for i, request in enumerate(video_requests):
            item = BatchJobItem(id=f"{job_id}_item_{i}", data=request)
            items.append(item)

        job_config = BatchJobConfig(
            job_id=job_id,
            job_type=BatchJobType.VIDEO_GENERATION,
            items=items,
            max_concurrent=max_concurrent,
            max_retries=2,
            timeout_per_item=600,  # 10 minutes per video
            notification_user_id=user_id,
        )

        return await self.batch_processor.submit_batch_job(
            job_config, self._process_video_generation
        )

    async def _process_video_generation(
        self, video_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a single video generation"""
        # TODO: Integrate with actual video generation service
        await asyncio.sleep(2)  # Simulate processing time

        return {
            "video_id": f"video_{uuid.uuid4().hex[:8]}",
            "title": video_data.get("title", "Generated Video"),
            "status": "completed",
            "cost": 2.45,
            "duration": 180,
        }


class DataBatchProcessor:
    def __init__(self, batch_processor: BatchProcessor):
        self.batch_processor = batch_processor

    async def batch_process_analytics(
        self, channel_ids: List[str], user_id: str, date_range: Dict[str, str]
    ) -> str:
        """Batch process analytics for multiple channels"""

        job_id = f"analytics_batch_{uuid.uuid4().hex[:8]}"

        # Create batch items
        items = []
        for channel_id in channel_ids:
            item = BatchJobItem(
                id=f"{job_id}_{channel_id}",
                data={"channel_id": channel_id, "date_range": date_range},
            )
            items.append(item)

        job_config = BatchJobConfig(
            job_id=job_id,
            job_type=BatchJobType.ANALYTICS_AGGREGATION,
            items=items,
            max_concurrent=5,
            max_retries=3,
            timeout_per_item=120,
            notification_user_id=user_id,
        )

        return await self.batch_processor.submit_batch_job(
            job_config, self._process_analytics_data
        )

    async def _process_analytics_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process analytics data for a single channel"""
        # TODO: Integrate with actual analytics processing
        await asyncio.sleep(1)  # Simulate processing time

        return {
            "channel_id": data["channel_id"],
            "metrics_processed": ["views", "subscribers", "revenue"],
            "date_range": data["date_range"],
            "records_processed": 1000,
        }

    async def process_batch(
        self,
        items: List[Dict[str, Any]],
        processor_func: Optional[Callable] = None,
        batch_type: BatchJobType = BatchJobType.VIDEO_GENERATION,
        max_concurrent: int = 10,
        user_id: Optional[str] = None,
    ) -> BatchJobResult:
        """
        Process a batch of items with configurable concurrency.
        Designed to handle 50-100 videos/day capacity.

        Args:
            items: List of items to process
            processor_func: Function to process each item
            batch_type: Type of batch job
            max_concurrent: Maximum concurrent processing (default: 10)
            user_id: User ID for notifications

        Returns:
            BatchJobResult with processing results
        """
        job_id = str(uuid.uuid4())

        # Create batch job items
        batch_items = []
        for idx, item in enumerate(items):
            batch_item = BatchJobItem(id=f"{job_id}_{idx}", data=item)
            batch_items.append(batch_item)

        # Configure batch job
        job_config = BatchJobConfig(
            job_id=job_id,
            job_type=batch_type,
            items=batch_items,
            max_concurrent=max_concurrent,
            max_retries=3,
            timeout_per_item=600,  # 10 minutes per video
            notification_user_id=user_id,
        )

        # Use default processor if none provided
        if processor_func is None:
            processor_func = self._default_processor

        # Submit and process batch
        return await self.submit_batch_job(job_config, processor_func)

    async def _default_processor(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Default processor for batch items"""
        logger.info(f"Processing item: {data}")
        await asyncio.sleep(0.1)  # Minimal processing simulation
        return {"status": "processed", "data": data}


# Global batch processor instance
batch_processor = BatchProcessor()
video_batch_processor = VideoBatchProcessor(batch_processor)
data_batch_processor = DataBatchProcessor(batch_processor)
