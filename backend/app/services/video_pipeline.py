"""
Enhanced Video Processing Pipeline for YTEmpire
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
import logging
from enum import Enum
from dataclasses import dataclass
import json

from app.core.celery_app import celery_app
from app.models.video import Video
from app.models.cost import Cost
from app.db.session import AsyncSessionLocal

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Video pipeline stages"""
    QUEUED = "queued"
    SCRIPT_GENERATION = "script_generation"
    VOICE_SYNTHESIS = "voice_synthesis"
    VISUAL_GENERATION = "visual_generation"
    VIDEO_ASSEMBLY = "video_assembly"
    THUMBNAIL_GENERATION = "thumbnail_generation"
    QUALITY_CHECK = "quality_check"
    PUBLISHING = "publishing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineStatus:
    """Pipeline execution status"""
    video_id: str
    current_stage: PipelineStage
    progress_percentage: int
    started_at: datetime
    estimated_completion: datetime
    errors: List[str]
    metadata: Dict[str, Any]


class VideoPipelineOrchestrator:
    """Orchestrate video generation pipeline with error handling and status tracking"""
    
    def __init__(self):
        self.active_pipelines = {}
        self.stage_weights = {
            PipelineStage.SCRIPT_GENERATION: 15,
            PipelineStage.VOICE_SYNTHESIS: 20,
            PipelineStage.VISUAL_GENERATION: 30,
            PipelineStage.VIDEO_ASSEMBLY: 20,
            PipelineStage.THUMBNAIL_GENERATION: 10,
            PipelineStage.QUALITY_CHECK: 5,
        }
        
    async def start_video_generation(self, 
                                    video_id: str,
                                    channel_id: str,
                                    config: Dict[str, Any]) -> str:
        """Start video generation pipeline"""
        try:
            # Initialize pipeline status
            pipeline_status = PipelineStatus(
                video_id=video_id,
                current_stage=PipelineStage.QUEUED,
                progress_percentage=0,
                started_at=datetime.utcnow(),
                estimated_completion=self._estimate_completion_time(),
                errors=[],
                metadata={"channel_id": channel_id, "config": config}
            )
            
            self.active_pipelines[video_id] = pipeline_status
            
            # Start async pipeline execution
            task_id = celery_app.send_task(
                'app.tasks.video_generation.generate_video',
                args=[video_id, channel_id, config],
                queue='video_generation'
            )
            
            # Update video status in database
            await self._update_video_status(video_id, "processing", {
                "task_id": str(task_id),
                "pipeline_started": datetime.utcnow().isoformat()
            })
            
            logger.info(f"Started video generation pipeline for {video_id}")
            return str(task_id)
            
        except Exception as e:
            logger.error(f"Failed to start video generation: {e}")
            await self._update_video_status(video_id, "failed", {"error": str(e)})
            raise
            
    async def update_stage(self, 
                          video_id: str, 
                          stage: PipelineStage,
                          metadata: Optional[Dict[str, Any]] = None):
        """Update pipeline stage and calculate progress"""
        if video_id not in self.active_pipelines:
            logger.warning(f"Video {video_id} not in active pipelines")
            return
            
        pipeline_status = self.active_pipelines[video_id]
        pipeline_status.current_stage = stage
        
        # Calculate progress
        progress = self._calculate_progress(stage)
        pipeline_status.progress_percentage = progress
        
        # Update metadata
        if metadata:
            pipeline_status.metadata.update(metadata)
            
        # Persist status update
        await self._update_video_status(video_id, stage.value, {
            "progress": progress,
            "metadata": metadata
        })
        
        logger.info(f"Video {video_id} progressed to {stage.value} ({progress}%)")
        
    async def handle_stage_error(self,
                                video_id: str,
                                stage: PipelineStage,
                                error: str,
                                retry: bool = True) -> bool:
        """Handle errors in pipeline stages"""
        if video_id not in self.active_pipelines:
            return False
            
        pipeline_status = self.active_pipelines[video_id]
        pipeline_status.errors.append({
            "stage": stage.value,
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        if retry and len(pipeline_status.errors) < 3:
            # Retry the stage
            logger.warning(f"Retrying {stage.value} for video {video_id}")
            await asyncio.sleep(5)  # Wait before retry
            return True
        else:
            # Mark as failed
            pipeline_status.current_stage = PipelineStage.FAILED
            await self._update_video_status(video_id, "failed", {
                "errors": pipeline_status.errors
            })
            logger.error(f"Video {video_id} failed at {stage.value}: {error}")
            return False
            
    async def get_pipeline_status(self, video_id: str) -> Optional[PipelineStatus]:
        """Get current pipeline status"""
        return self.active_pipelines.get(video_id)
        
    async def complete_pipeline(self, 
                               video_id: str,
                               result: Dict[str, Any]):
        """Mark pipeline as completed"""
        if video_id not in self.active_pipelines:
            return
            
        pipeline_status = self.active_pipelines[video_id]
        pipeline_status.current_stage = PipelineStage.COMPLETED
        pipeline_status.progress_percentage = 100
        
        # Calculate total time
        total_time = (datetime.utcnow() - pipeline_status.started_at).total_seconds()
        
        # Update final status
        await self._update_video_status(video_id, "completed", {
            "result": result,
            "total_time_seconds": total_time,
            "completed_at": datetime.utcnow().isoformat()
        })
        
        # Clean up
        del self.active_pipelines[video_id]
        
        logger.info(f"Video {video_id} pipeline completed in {total_time}s")
        
    def _calculate_progress(self, stage: PipelineStage) -> int:
        """Calculate overall progress percentage"""
        completed_weight = 0
        total_weight = sum(self.stage_weights.values())
        
        # Add weights of completed stages
        for s in PipelineStage:
            if s == stage:
                break
            if s in self.stage_weights:
                completed_weight += self.stage_weights[s]
                
        return int((completed_weight / total_weight) * 100)
        
    def _estimate_completion_time(self) -> datetime:
        """Estimate pipeline completion time"""
        # Average times per stage (in seconds)
        stage_times = {
            PipelineStage.SCRIPT_GENERATION: 30,
            PipelineStage.VOICE_SYNTHESIS: 60,
            PipelineStage.VISUAL_GENERATION: 180,
            PipelineStage.VIDEO_ASSEMBLY: 120,
            PipelineStage.THUMBNAIL_GENERATION: 20,
            PipelineStage.QUALITY_CHECK: 30,
        }
        
        total_seconds = sum(stage_times.values())
        return datetime.utcnow() + timedelta(seconds=total_seconds)
        
    async def _update_video_status(self, 
                                  video_id: str,
                                  status: str,
                                  metadata: Dict[str, Any]):
        """Update video status in database"""
        async with AsyncSessionLocal() as db:
            video = await db.get(Video, video_id)
            if video:
                video.generation_status = status
                video.updated_at = datetime.utcnow()
                
                # Store metadata as JSON
                if hasattr(video, 'metadata'):
                    video.metadata = json.dumps(metadata)
                    
                await db.commit()


class PipelineMonitor:
    """Monitor and report on pipeline health"""
    
    def __init__(self, orchestrator: VideoPipelineOrchestrator):
        self.orchestrator = orchestrator
        self.metrics = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "average_time": 0,
            "stage_times": {}
        }
        
    async def monitor_pipelines(self):
        """Monitor active pipelines"""
        while True:
            active_count = len(self.orchestrator.active_pipelines)
            
            if active_count > 0:
                logger.info(f"Active pipelines: {active_count}")
                
                # Check for stuck pipelines
                for video_id, status in self.orchestrator.active_pipelines.items():
                    elapsed = (datetime.utcnow() - status.started_at).total_seconds()
                    
                    # Alert if pipeline is taking too long
                    if elapsed > 1200:  # 20 minutes
                        logger.warning(f"Pipeline {video_id} is taking longer than expected: {elapsed}s")
                        
                    # Check for stalled pipelines
                    if elapsed > 1800:  # 30 minutes
                        logger.error(f"Pipeline {video_id} appears to be stalled")
                        await self.orchestrator.handle_stage_error(
                            video_id,
                            status.current_stage,
                            "Pipeline timeout",
                            retry=False
                        )
                        
            await asyncio.sleep(30)  # Check every 30 seconds
            
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics"""
        active_pipelines = []
        
        for video_id, status in self.orchestrator.active_pipelines.items():
            active_pipelines.append({
                "video_id": video_id,
                "stage": status.current_stage.value,
                "progress": status.progress_percentage,
                "errors": len(status.errors),
                "elapsed_time": (datetime.utcnow() - status.started_at).total_seconds()
            })
            
        return {
            "active_count": len(active_pipelines),
            "active_pipelines": active_pipelines,
            "metrics": self.metrics
        }


# Global orchestrator instance
pipeline_orchestrator = VideoPipelineOrchestrator()
pipeline_monitor = PipelineMonitor(pipeline_orchestrator)