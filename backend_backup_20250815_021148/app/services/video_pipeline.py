"""
Video Pipeline Module
Provides video pipeline orchestration functionality
"""

import logging
from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline stage enumeration"""
    SCRIPT_GENERATION = "script_generation"
    VOICE_SYNTHESIS = "voice_synthesis"
    VIDEO_ASSEMBLY = "video_assembly"
    THUMBNAIL_GENERATION = "thumbnail_generation"
    UPLOAD_PREPARATION = "upload_preparation"
    YOUTUBE_UPLOAD = "youtube_upload"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineTask:
    """Pipeline task data"""
    task_id: str
    stage: PipelineStage
    video_id: str
    channel_id: str
    status: str
    progress: float
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]


class VideoPipelineOrchestrator:
    """
    Video pipeline orchestrator for coordinating video generation stages
    """
    
    def __init__(self):
        self.active_pipelines: Dict[str, PipelineTask] = {}
        self.completed_pipelines: List[PipelineTask] = []
        
    async def start_pipeline(self, video_id: str, channel_id: str, metadata: Dict[str, Any] = None) -> str:
        """Start a new video pipeline"""
        import uuid
        task_id = str(uuid.uuid4())
        
        task = PipelineTask(
            task_id=task_id,
            stage=PipelineStage.SCRIPT_GENERATION,
            video_id=video_id,
            channel_id=channel_id,
            status="started",
            progress=0.0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        self.active_pipelines[task_id] = task
        logger.info(f"Started video pipeline {task_id} for video {video_id}")
        
        return task_id
    
    async def update_pipeline_stage(self, task_id: str, stage: PipelineStage, progress: float = None):
        """Update pipeline stage"""
        if task_id not in self.active_pipelines:
            raise ValueError(f"Pipeline {task_id} not found")
        
        task = self.active_pipelines[task_id]
        task.stage = stage
        task.updated_at = datetime.utcnow()
        
        if progress is not None:
            task.progress = progress
        
        logger.info(f"Pipeline {task_id} updated to stage {stage.value}")
    
    async def complete_pipeline(self, task_id: str, success: bool = True):
        """Complete a pipeline"""
        if task_id not in self.active_pipelines:
            raise ValueError(f"Pipeline {task_id} not found")
        
        task = self.active_pipelines[task_id]
        task.stage = PipelineStage.COMPLETED if success else PipelineStage.FAILED
        task.status = "completed" if success else "failed"
        task.progress = 100.0 if success else task.progress
        task.updated_at = datetime.utcnow()
        
        # Move to completed
        self.completed_pipelines.append(task)
        del self.active_pipelines[task_id]
        
        logger.info(f"Pipeline {task_id} {'completed' if success else 'failed'}")
    
    def get_pipeline_status(self, task_id: str) -> Optional[PipelineTask]:
        """Get pipeline status"""
        return self.active_pipelines.get(task_id)
    
    def get_active_pipelines(self) -> List[PipelineTask]:
        """Get all active pipelines"""
        return list(self.active_pipelines.values())
    
    def get_completed_pipelines(self, limit: int = 100) -> List[PipelineTask]:
        """Get completed pipelines"""
        return self.completed_pipelines[-limit:]


# Global pipeline orchestrator instance
video_pipeline_orchestrator = VideoPipelineOrchestrator()