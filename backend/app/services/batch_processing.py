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

class BatchProcessor:
    def __init__(self):
        self.running_jobs: Dict[str, BatchJobConfig] = {}
        self.job_history: List[Dict[str, Any]] = []
        self.websocket_manager = ConnectionManager()
        
    async def submit_batch_job(
        self, 
        job_config: BatchJobConfig,
        processor_function: Callable
    ) -> str:
        """Submit a batch job for processing"""
        try:
            job_config.metadata = job_config.metadata or {}
            job_config.metadata.update({
                "created_at": datetime.utcnow().isoformat(),
                "total_items": len(job_config.items)
            })
            
            self.running_jobs[job_config.job_id] = job_config
            
            # Start processing in background
            asyncio.create_task(
                self._process_batch_job(job_config, processor_function)
            )
            
            logger.info(f"Submitted batch job {job_config.job_id} with {len(job_config.items)} items")
            
            return job_config.job_id
            
        except Exception as e:
            logger.error(f"Failed to submit batch job: {str(e)}")
            raise

    async def _process_batch_job(
        self,
        job_config: BatchJobConfig,
        processor_function: Callable
    ):
        """Process a batch job with concurrency control"""
        job_id = job_config.job_id
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Starting batch job {job_id}")
            
            # Update job status
            await self._update_job_status(job_id, {"status": "running", "started_at": start_time})
            
            # Process items with concurrency control
            semaphore = asyncio.Semaphore(job_config.max_concurrent)
            
            tasks = []
            for item in job_config.items:
                task = self._process_batch_item(
                    item, processor_function, semaphore, job_config
                )
                tasks.append(task)
            
            # Wait for all items to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Calculate final statistics
            completed_items = sum(1 for r in results if not isinstance(r, Exception))
            failed_items = len(results) - completed_items
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            final_status = {
                "status": BatchJobStatus.COMPLETED if failed_items == 0 else BatchJobStatus.FAILED,
                "completed_at": end_time,
                "duration_seconds": duration,
                "total_items": len(job_config.items),
                "completed_items": completed_items,
                "failed_items": failed_items,
                "success_rate": completed_items / len(job_config.items) * 100
            }
            
            await self._update_job_status(job_id, final_status)
            
            # Send completion notification
            if job_config.notification_user_id:
                await self._send_job_completion_notification(job_config, final_status)
            
            # Move to history and remove from running jobs
            self.job_history.append({
                "job_id": job_id,
                "job_type": job_config.job_type.value,
                **final_status,
                "metadata": job_config.metadata
            })
            
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]
            
            logger.info(f"Completed batch job {job_id}: {completed_items}/{len(job_config.items)} successful")
            
        except Exception as e:
            logger.error(f"Batch job {job_id} failed: {str(e)}")
            await self._update_job_status(job_id, {
                "status": BatchJobStatus.FAILED,
                "error": str(e),
                "completed_at": datetime.utcnow()
            })

    async def _process_batch_item(
        self,
        item: BatchJobItem,
        processor_function: Callable,
        semaphore: asyncio.Semaphore,
        job_config: BatchJobConfig
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
                        timeout=job_config.timeout_per_item
                    )
                    
                    item.result = result
                    item.status = BatchJobStatus.COMPLETED
                    item.completed_at = datetime.utcnow()
                    
                    logger.debug(f"Completed batch item {item.id}")
                    return result
                    
                except asyncio.TimeoutError:
                    error_msg = f"Item {item.id} timed out after {job_config.timeout_per_item}s"
                    logger.warning(error_msg)
                    item.error = error_msg
                    
                except Exception as e:
                    error_msg = f"Item {item.id} failed: {str(e)}"
                    logger.warning(error_msg)
                    item.error = error_msg
                
                retries += 1
                
                if retries <= job_config.max_retries:
                    logger.info(f"Retrying item {item.id}, attempt {retries}/{job_config.max_retries}")
                    await asyncio.sleep(job_config.retry_delay)
                else:
                    item.status = BatchJobStatus.FAILED
                    item.completed_at = datetime.utcnow()
                    logger.error(f"Item {item.id} failed after {job_config.max_retries} retries")
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
                        json.dumps({
                            "type": "batch_job_update",
                            "job_id": job_id,
                            "job_type": job_config.job_type.value,
                            **status_update
                        }),
                        job_config.notification_user_id
                    )
                    
        except Exception as e:
            logger.error(f"Failed to update job status for {job_id}: {str(e)}")

    async def _send_job_completion_notification(
        self, 
        job_config: BatchJobConfig, 
        final_status: Dict[str, Any]
    ):
        """Send job completion notification"""
        try:
            success_rate = final_status.get("success_rate", 0)
            template_id = "batch_job_complete_success" if success_rate >= 90 else "batch_job_complete_partial"
            
            await notification_service.send_template_notification(
                user_id=job_config.notification_user_id,
                template_id=template_id,
                variables={
                    "job_type": job_config.job_type.value,
                    "total_items": str(final_status["total_items"]),
                    "completed_items": str(final_status["completed_items"]),
                    "failed_items": str(final_status["failed_items"]),
                    "success_rate": f"{success_rate:.1f}",
                    "duration": f"{final_status['duration_seconds']:.1f}"
                },
                notification_types=[NotificationType.IN_APP]
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
                "completed_items": sum(1 for item in job_config.items if item.status == BatchJobStatus.COMPLETED),
                "failed_items": sum(1 for item in job_config.items if item.status == BatchJobStatus.FAILED),
                "metadata": job_config.metadata
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
        """Get status of all jobs (running and completed)"""
        running = []
        for job_id, job_config in self.running_jobs.items():
            status = await self.get_job_status(job_id)
            if status:
                running.append(status)
        
        return {
            "running_jobs": running,
            "completed_jobs": self.job_history[-50:],  # Last 50 completed jobs
            "total_running": len(running),
            "total_history": len(self.job_history)
        }

# Specific batch processors for common operations

class VideoBatchProcessor:
    def __init__(self, batch_processor: BatchProcessor):
        self.batch_processor = batch_processor

    async def batch_generate_videos(
        self,
        video_requests: List[Dict[str, Any]],
        user_id: str,
        max_concurrent: int = 3
    ) -> str:
        """Batch generate multiple videos"""
        
        job_id = f"video_batch_{uuid.uuid4().hex[:8]}"
        
        # Create batch items
        items = []
        for i, request in enumerate(video_requests):
            item = BatchJobItem(
                id=f"{job_id}_item_{i}",
                data=request
            )
            items.append(item)
        
        job_config = BatchJobConfig(
            job_id=job_id,
            job_type=BatchJobType.VIDEO_GENERATION,
            items=items,
            max_concurrent=max_concurrent,
            max_retries=2,
            timeout_per_item=600,  # 10 minutes per video
            notification_user_id=user_id
        )
        
        return await self.batch_processor.submit_batch_job(
            job_config, self._process_video_generation
        )

    async def _process_video_generation(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single video generation"""
        # TODO: Integrate with actual video generation service
        await asyncio.sleep(2)  # Simulate processing time
        
        return {
            "video_id": f"video_{uuid.uuid4().hex[:8]}",
            "title": video_data.get("title", "Generated Video"),
            "status": "completed",
            "cost": 2.45,
            "duration": 180
        }

class DataBatchProcessor:
    def __init__(self, batch_processor: BatchProcessor):
        self.batch_processor = batch_processor

    async def batch_process_analytics(
        self,
        channel_ids: List[str],
        user_id: str,
        date_range: Dict[str, str]
    ) -> str:
        """Batch process analytics for multiple channels"""
        
        job_id = f"analytics_batch_{uuid.uuid4().hex[:8]}"
        
        # Create batch items
        items = []
        for channel_id in channel_ids:
            item = BatchJobItem(
                id=f"{job_id}_{channel_id}",
                data={
                    "channel_id": channel_id,
                    "date_range": date_range
                }
            )
            items.append(item)
        
        job_config = BatchJobConfig(
            job_id=job_id,
            job_type=BatchJobType.ANALYTICS_AGGREGATION,
            items=items,
            max_concurrent=5,
            max_retries=3,
            timeout_per_item=120,
            notification_user_id=user_id
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
            "records_processed": 1000
        }

# Global batch processor instance
batch_processor = BatchProcessor()
video_batch_processor = VideoBatchProcessor(batch_processor)
data_batch_processor = DataBatchProcessor(batch_processor)