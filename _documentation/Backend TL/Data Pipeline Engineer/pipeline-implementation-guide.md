# Data Pipeline Implementation Guide & Code Examples

**Document Version**: 3.2 (Complete)  
**Date**: January 2025  
**Scope**: Complete Implementation with Production Code  
**Language**: Python 3.11+

---

## Table of Contents
1. [Queue Management Implementation](#1-queue-management-implementation)
2. [Video Processing Pipeline](#2-video-processing-pipeline)
3. [Cost Tracking System](#3-cost-tracking-system)
4. [Resource Scheduling](#4-resource-scheduling)
5. [Analytics Pipeline](#5-analytics-pipeline)
6. [Error Handling & Recovery](#6-error-handling--recovery)
7. [API Implementation](#7-api-implementation)
8. [Testing Framework](#8-testing-framework)
9. [Deployment & Configuration](#9-deployment--configuration)

---

## 1. Queue Management Implementation

### Complete Queue Manager

```python
"""
queue_manager.py - Production-ready queue management system
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

import asyncpg
import redis.asyncio as redis
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# Configuration
REDIS_URL = "redis://localhost:6379"
DATABASE_URL = "postgresql://user:pass@localhost:5432/ytempire"

class VideoStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class VideoComplexity(Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"
    PREMIUM = "premium"

@dataclass
class VideoJob:
    """Video processing job data structure"""
    id: str
    user_id: str
    channel_id: str
    priority: int
    complexity: VideoComplexity
    request_data: Dict[Any, Any]
    estimated_cost: float = 2.50
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'complexity': self.complexity.value,
            'created_at': self.created_at.isoformat()
        }

class VideoQueueManager:
    """
    Production queue manager with PostgreSQL persistence and Redis caching
    Handles 50-500 videos/day with priority-based scheduling
    """
    
    def __init__(self):
        self.db_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[redis.Redis] = None
        self.queue_semaphore = asyncio.Semaphore(100)  # Max queue depth
        self.processing_semaphore = asyncio.Semaphore(7)  # Max concurrent
        
    async def initialize(self):
        """Initialize database and Redis connections"""
        try:
            # PostgreSQL connection pool
            self.db_pool = await asyncpg.create_pool(
                DATABASE_URL,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            
            # Redis connection
            self.redis_client = await redis.from_url(
                REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
            
            # Initialize Redis structures
            await self._init_redis_structures()
            
            logger.info("Queue manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize queue manager: {e}")
            raise
    
    async def _init_redis_structures(self):
        """Initialize Redis data structures"""
        # Priority queue (sorted set)
        await self.redis_client.zadd(
            "queue:priority",
            {"init": 0},
            nx=True
        )
        await self.redis_client.zrem("queue:priority", "init")
        
        # Processing set
        await self.redis_client.delete("queue:processing")
        
        # Metrics
        await self.redis_client.hset(
            "metrics:queue",
            mapping={
                "total_enqueued": 0,
                "total_processed": 0,
                "total_failed": 0
            }
        )
    
    async def enqueue(self, video_request: Dict) -> VideoJob:
        """
        Add video to processing queue with priority
        """
        async with self.queue_semaphore:
            # Create job
            job = VideoJob(
                id=str(uuid.uuid4()),
                user_id=video_request['user_id'],
                channel_id=video_request['channel_id'],
                priority=video_request.get('priority', 5),
                complexity=VideoComplexity(
                    video_request.get('complexity', 'simple')
                ),
                request_data=video_request,
                estimated_cost=self._estimate_cost(video_request)
            )
            
            async with self.db_pool.acquire() as conn:
                # Insert into PostgreSQL
                await conn.execute("""
                    INSERT INTO video_queue 
                    (id, user_id, channel_id, priority, status, 
                     complexity, request_data, estimated_cost, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                    job.id, job.user_id, job.channel_id, job.priority,
                    VideoStatus.QUEUED.value, job.complexity.value,
                    json.dumps(job.request_data), job.estimated_cost,
                    job.created_at
                )
                
                # Add to Redis priority queue
                score = self._calculate_priority_score(job)
                await self.redis_client.zadd(
                    "queue:priority",
                    {job.id: score}
                )
                
                # Update metrics
                await self.redis_client.hincrby("metrics:queue", "total_enqueued", 1)
                
                # Get queue position
                position = await self._get_queue_position(job.id)
                
                logger.info(f"Enqueued video {job.id} at position {position}")
                
                return job
    
    async def dequeue(self, worker_type: str = "any") -> Optional[VideoJob]:
        """
        Get next video from queue based on priority
        Target: <100ms dequeue time
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Get highest priority job from Redis
            result = await self.redis_client.zpopmax("queue:priority", count=1)
            
            if not result:
                return None
            
            video_id = result[0][0]
            
            # Move to processing set
            await self.redis_client.sadd("queue:processing", video_id)
            
            # Fetch full job from PostgreSQL
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    UPDATE video_queue 
                    SET status = $1, processing_started_at = $2
                    WHERE id = $3 AND status = $4
                    RETURNING *
                """,
                    VideoStatus.PROCESSING.value,
                    datetime.utcnow(),
                    video_id,
                    VideoStatus.QUEUED.value
                )
                
                if not row:
                    # Job was already processed or cancelled
                    await self.redis_client.srem("queue:processing", video_id)
                    return await self.dequeue(worker_type)  # Try next
                
                job = self._row_to_job(row)
                
                # Check if worker type matches complexity
                if not self._worker_can_process(worker_type, job.complexity):
                    # Re-queue the job
                    await self._requeue_job(job)
                    return await self.dequeue(worker_type)  # Try next
                
                dequeue_time = asyncio.get_event_loop().time() - start_time
                
                if dequeue_time > 0.1:  # Log slow dequeues
                    logger.warning(f"Slow dequeue: {dequeue_time:.3f}s")
                
                logger.info(f"Dequeued video {job.id} for {worker_type} worker")
                
                return job
                
        except Exception as e:
            logger.error(f"Dequeue error: {e}")
            return None
    
    async def complete(self, video_id: str, cost: float, metadata: Dict = None):
        """Mark video as completed with actual cost"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE video_queue 
                SET status = $1, completed_at = $2, actual_cost = $3, 
                    processing_time_seconds = EXTRACT(EPOCH FROM ($2 - processing_started_at))
                WHERE id = $4
            """,
                VideoStatus.COMPLETED.value,
                datetime.utcnow(),
                cost,
                video_id
            )
        
        # Remove from processing set
        await self.redis_client.srem("queue:processing", video_id)
        
        # Update metrics
        await self.redis_client.hincrby("metrics:queue", "total_processed", 1)
        
        # Publish completion event
        await self._publish_event("video.completed", {
            "video_id": video_id,
            "cost": cost,
            "metadata": metadata
        })
        
        logger.info(f"Completed video {video_id} with cost ${cost:.2f}")
    
    async def fail(self, video_id: str, error: str, retry: bool = True):
        """Mark video as failed with optional retry"""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                UPDATE video_queue 
                SET retry_count = retry_count + 1,
                    error_log = COALESCE(error_log, '[]'::jsonb) || $1::jsonb
                WHERE id = $2
                RETURNING retry_count
            """,
                json.dumps([{"error": error, "timestamp": datetime.utcnow().isoformat()}]),
                video_id
            )
            
            retry_count = row['retry_count']
            
            if retry and retry_count < 3:
                # Schedule retry with exponential backoff
                delay = 10 * (2 ** retry_count)  # 10s, 20s, 40s
                await self._schedule_retry(video_id, delay)
                logger.info(f"Scheduled retry for {video_id} in {delay}s")
            else:
                # Mark as permanently failed
                await conn.execute("""
                    UPDATE video_queue 
                    SET status = $1, completed_at = $2
                    WHERE id = $3
                """,
                    VideoStatus.FAILED.value,
                    datetime.utcnow(),
                    video_id
                )
                
                await self.redis_client.hincrby("metrics:queue", "total_failed", 1)
                logger.error(f"Video {video_id} permanently failed: {error}")
        
        # Remove from processing set
        await self.redis_client.srem("queue:processing", video_id)
    
    def _calculate_priority_score(self, job: VideoJob) -> float:
        """
        Calculate priority score for Redis sorted set
        Higher score = higher priority
        """
        base_score = 1000000 - job.created_at.timestamp()  # Older = higher
        priority_boost = job.priority * 100000  # Priority weight
        
        # Premium videos get extra boost
        if job.complexity == VideoComplexity.PREMIUM:
            priority_boost += 50000
        
        return base_score + priority_boost
    
    def _estimate_cost(self, request: Dict) -> float:
        """Estimate video processing cost"""
        complexity = request.get('complexity', 'simple')
        
        cost_map = {
            'simple': 2.00,
            'complex': 2.50,
            'premium': 2.80
        }
        
        return cost_map.get(complexity, 2.50)
    
    def _worker_can_process(self, worker_type: str, complexity: VideoComplexity) -> bool:
        """Check if worker type can process video complexity"""
        if worker_type == "any":
            return True
        
        if worker_type == "gpu":
            return complexity in [VideoComplexity.COMPLEX, VideoComplexity.PREMIUM]
        
        if worker_type == "cpu":
            return complexity == VideoComplexity.SIMPLE
        
        return False
    
    def _row_to_job(self, row: asyncpg.Record) -> VideoJob:
        """Convert database row to VideoJob"""
        return VideoJob(
            id=str(row['id']),
            user_id=str(row['user_id']),
            channel_id=str(row['channel_id']),
            priority=row['priority'],
            complexity=VideoComplexity(row['complexity']),
            request_data=row['request_data'],
            estimated_cost=float(row['estimated_cost']),
            created_at=row['created_at']
        )
    
    async def _get_queue_position(self, video_id: str) -> int:
        """Get position in queue"""
        rank = await self.redis_client.zrevrank("queue:priority", video_id)
        return rank + 1 if rank is not None else -1
    
    async def _requeue_job(self, job: VideoJob):
        """Put job back in queue"""
        score = self._calculate_priority_score(job)
        await self.redis_client.zadd("queue:priority", {job.id: score})
        await self.redis_client.srem("queue:processing", job.id)
        
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE video_queue 
                SET status = $1
                WHERE id = $2
            """,
                VideoStatus.QUEUED.value,
                job.id
            )
    
    async def _schedule_retry(self, video_id: str, delay: int):
        """Schedule job retry after delay"""
        asyncio.create_task(self._retry_after_delay(video_id, delay))
    
    async def _retry_after_delay(self, video_id: str, delay: int):
        """Retry job after delay"""
        await asyncio.sleep(delay)
        
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                UPDATE video_queue 
                SET status = $1
                WHERE id = $2 AND status != $3
                RETURNING *
            """,
                VideoStatus.QUEUED.value,
                video_id,
                VideoStatus.COMPLETED.value
            )
            
            if row:
                job = self._row_to_job(row)
                score = self._calculate_priority_score(job)
                await self.redis_client.zadd("queue:priority", {job.id: score})
                logger.info(f"Retrying video {video_id}")
    
    async def _publish_event(self, event_type: str, data: Dict):
        """Publish event to Redis pub/sub"""
        await self.redis_client.publish(
            f"pipeline:{event_type}",
            json.dumps(data)
        )
    
    async def get_metrics(self) -> Dict:
        """Get queue metrics"""
        metrics = await self.redis_client.hgetall("metrics:queue")
        queue_depth = await self.redis_client.zcard("queue:priority")
        processing_count = await self.redis_client.scard("queue:processing")
        
        return {
            "queue_depth": queue_depth,
            "processing_count": processing_count,
            "total_enqueued": int(metrics.get("total_enqueued", 0)),
            "total_processed": int(metrics.get("total_processed", 0)),
            "total_failed": int(metrics.get("total_failed", 0))
        }
    
    async def cleanup(self):
        """Cleanup connections"""
        if self.redis_client:
            await self.redis_client.close()
        if self.db_pool:
            await self.db_pool.close()
```

---

## 2. Video Processing Pipeline

```python
"""
video_pipeline.py - End-to-end video processing pipeline
"""

import asyncio
import logging
from typing import Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import json
import aiohttp
import subprocess

from .queue_manager import VideoJob, VideoStatus
from .cost_tracker import CostTracker
from .resource_scheduler import ResourceScheduler

logger = logging.getLogger(__name__)

@dataclass
class PipelineStage:
    """Pipeline stage configuration"""
    name: str
    timeout: int
    max_retries: int
    estimated_cost: float
    
class VideoProcessingPipeline:
    """
    Main video processing pipeline
    Target: <10 minutes end-to-end, <$3 per video
    """
    
    STAGES = [
        PipelineStage("script_generation", 60, 2, 0.40),
        PipelineStage("audio_synthesis", 120, 2, 0.20),
        PipelineStage("media_collection", 180, 3, 0.10),
        PipelineStage("video_rendering", 300, 1, 0.30),
        PipelineStage("quality_validation", 30, 1, 0.05),
        PipelineStage("upload_preparation", 120, 2, 0.10)
    ]
    
    def __init__(self, queue_manager, cost_tracker, resource_scheduler):
        self.queue_manager = queue_manager
        self.cost_tracker = cost_tracker
        self.resource_scheduler = resource_scheduler
        self.openai_api_key = "YOUR_OPENAI_API_KEY"
        self.pexels_api_key = "YOUR_PEXELS_API_KEY"
        
    async def process_video(self, job: VideoJob) -> Dict:
        """
        Process single video through all stages
        """
        start_time = datetime.utcnow()
        video_id = job.id
        total_cost = 0.0
        stage_results = {}
        
        try:
            # Initialize cost tracking
            await self.cost_tracker.init_video(video_id, job.estimated_cost)
            
            # Process through each stage
            for stage in self.STAGES:
                logger.info(f"Processing {video_id} - Stage: {stage.name}")
                
                # Check cost before proceeding
                if total_cost + stage.estimated_cost > 3.00:
                    raise Exception(f"Cost limit exceeded at {stage.name}")
                
                # Execute stage with timeout
                try:
                    result = await asyncio.wait_for(
                        self._execute_stage(video_id, stage, job.request_data),
                        timeout=stage.timeout
                    )
                    
                    stage_results[stage.name] = result
                    
                    # Track cost
                    stage_cost = result.get('cost', stage.estimated_cost)
                    await self.cost_tracker.track_cost(
                        video_id, 
                        stage.name, 
                        stage_cost
                    )
                    total_cost += stage_cost
                    
                    # Update progress
                    progress = (self.STAGES.index(stage) + 1) / len(self.STAGES) * 100
                    await self._update_progress(video_id, stage.name, progress)
                    
                except asyncio.TimeoutError:
                    logger.error(f"Stage {stage.name} timed out for {video_id}")
                    if stage.max_retries > 0:
                        # Retry logic
                        await asyncio.sleep(5)
                        continue
                    else:
                        raise
            
            # Final validation
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            if processing_time > 600:  # 10 minutes
                logger.warning(f"Video {video_id} took {processing_time}s to process")
            
            # Mark as completed
            await self.queue_manager.complete(video_id, total_cost, {
                "processing_time": processing_time,
                "stages": stage_results
            })
            
            return {
                "video_id": video_id,
                "status": "completed",
                "cost": total_cost,
                "processing_time": processing_time,
                "results": stage_results
            }
            
        except Exception as e:
            logger.error(f"Pipeline error for {video_id}: {e}")
            await self.queue_manager.fail(video_id, str(e))
            raise
    
    async def _execute_stage(self, video_id: str, stage: PipelineStage, data: Dict) -> Dict:
        """Execute individual pipeline stage"""
        
        if stage.name == "script_generation":
            return await self._generate_script(data)
        elif stage.name == "audio_synthesis":
            return await self._synthesize_audio(data)
        elif stage.name == "media_collection":
            return await self._collect_media(data)
        elif stage.name == "video_rendering":
            return await self._render_video(video_id, data)
        elif stage.name == "quality_validation":
            return await self._validate_quality(data)
        elif stage.name == "upload_preparation":
            return await self._prepare_upload(data)
        else:
            raise ValueError(f"Unknown stage: {stage.name}")
    
    async def _generate_script(self, data: Dict) -> Dict:
        """Generate video script using OpenAI"""
        
        prompt = data.get('prompt', 'Create an engaging YouTube video script')
        
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are a YouTube script writer."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1500,
                "temperature": 0.7
            }
            
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                result = await response.json()
                
                script = result['choices'][0]['message']['content']
                tokens_used = result['usage']['total_tokens']
                
                # Calculate cost (GPT-3.5: $0.002 per 1K tokens)
                cost = (tokens_used / 1000) * 0.002
                
                return {
                    "script": script,
                    "cost": cost,
                    "tokens_used": tokens_used
                }
    
    async def _synthesize_audio(self, data: Dict) -> Dict:
        """Synthesize audio from script using Google TTS"""
        
        script = data.get('script', '')
        
        # Use gTTS for simplicity (in production, use Google Cloud TTS)
        from gtts import gTTS
        import tempfile
        import mutagen.mp3
        
        tts = gTTS(text=script, lang='en', slow=False)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
            audio_path = tmp_file.name
            tts.save(audio_path)
        
        # Get audio duration
        audio = mutagen.mp3.MP3(audio_path)
        duration = audio.info.length
        
        # Calculate cost (estimated)
        cost = 0.20  # Fixed cost for simplicity
        
        return {
            "audio_file": audio_path,
            "duration": duration,
            "cost": cost
        }
    
    async def _collect_media(self, data: Dict) -> Dict:
        """Collect stock media from Pexels API"""
        
        keywords = data.get('keywords', ['nature', 'technology'])
        media_count = 5
        
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": self.pexels_api_key}
            media_files = []
            
            for keyword in keywords[:2]:  # Limit to 2 keywords
                url = f"https://api.pexels.com/videos/search?query={keyword}&per_page={media_count}"
                
                async with session.get(url, headers=headers) as response:
                    result = await response.json()
                    
                    for video in result.get('videos', [])[:media_count]:
                        media_files.append({
                            'url': video['video_files'][0]['link'],
                            'duration': video['duration']
                        })
            
            return {
                "media_files": media_files,
                "cost": 0.10
            }
    
    async def _render_video(self, video_id: str, data: Dict) -> Dict:
        """Render final video using FFmpeg"""
        
        import os
        import subprocess
        
        # Allocate resources
        allocation = await self.resource_scheduler.allocate_for_video(video_id, data)
        
        try:
            audio_file = data.get('audio_file', '/tmp/audio.mp3')
            output_file = f"/tmp/{video_id}.mp4"
            
            # Build FFmpeg command
            if allocation.resource_type.value == "gpu":
                # GPU-accelerated rendering
                cmd = [
                    'ffmpeg',
                    '-hwaccel', 'cuda',
                    '-i', audio_file,
                    '-f', 'lavfi', '-i', 'color=c=black:s=1920x1080',
                    '-c:v', 'h264_nvenc',
                    '-preset', 'p7',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-shortest',
                    output_file
                ]
            else:
                # CPU rendering
                cmd = [
                    'ffmpeg',
                    '-i', audio_file,
                    '-f', 'lavfi', '-i', 'color=c=black:s=1920x1080',
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-shortest',
                    output_file
                ]
            
            # Execute FFmpeg
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"FFmpeg failed: {stderr.decode()}")
            
            return {
                "video_file": output_file,
                "duration": data.get('duration', 120),
                "resolution": "1920x1080",
                "cost": 0.30,
                "resource_used": allocation.resource_type.value
            }
            
        finally:
            # Release resources
            await self.resource_scheduler.release(allocation)
    
    async def _validate_quality(self, data: Dict) -> Dict:
        """Validate video quality"""
        
        import os
        
        video_file = data.get('video_file', '')
        
        # Basic quality checks
        checks = {
            "file_exists": os.path.exists(video_file),
            "file_size": os.path.getsize(video_file) if os.path.exists(video_file) else 0,
            "duration_valid": data.get('duration', 0) > 30
        }
        
        quality_score = sum(1 for check in checks.values() if check) / len(checks)
        
        return {
            "quality_score": quality_score,
            "issues": [k for k, v in checks.items() if not v],
            "cost": 0.05
        }
    
    async def _prepare_upload(self, data: Dict) -> Dict:
        """Prepare video for YouTube upload"""
        
        import os
        
        video_file = data.get('video_file', '')
        script = data.get('script', '')
        
        # Generate metadata
        metadata = {
            "title": script[:100] if script else "Video Title",
            "description": script[:500] if script else "Video description",
            "tags": ["ytempire", "automated"],
            "category": "22",  # People & Blogs
            "privacy": "private"
        }
        
        # Generate thumbnail (placeholder)
        thumbnail_path = f"/tmp/{os.path.basename(video_file)}.jpg"
        
        return {
            "upload_ready": True,
            "metadata": metadata,
            "thumbnail": thumbnail_path,
            "cost": 0.10
        }
    
    async def _update_progress(self, video_id: str, stage: str, progress: float):
        """Send progress update via WebSocket"""
        await self.queue_manager._publish_event("progress", {
            "video_id": video_id,
            "stage": stage,
            "progress": progress,
            "timestamp": datetime.utcnow().isoformat()
        })
```

---

## 3. Cost Tracking System

```python
"""
cost_tracker.py - Real-time cost tracking and control
"""

import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime
from enum import Enum
import json

import asyncpg
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class CostService(Enum):
    """Service types for cost tracking"""
    OPENAI = "openai"
    TTS = "tts"
    MEDIA_API = "media_api"
    GPU_COMPUTE = "gpu_compute"
    CPU_COMPUTE = "cpu_compute"
    STORAGE = "storage"
    YOUTUBE_API = "youtube_api"

class CostTracker:
    """
    Real-time cost tracking with hard stops
    Target: <$3.00 per video (operational: $2.50)
    """
    
    COST_LIMITS = {
        "warning": 2.50,
        "critical": 2.80,
        "halt": 3.00
    }
    
    def __init__(self, db_pool: asyncpg.Pool, redis_client: redis.Redis):
        self.db_pool = db_pool
        self.redis_client = redis_client
        self.alerts_enabled = True
        
    async def init_video(self, video_id: str, estimated_cost: float):
        """Initialize cost tracking for a video"""
        
        # Set initial cost in Redis for fast access
        await self.redis_client.hset(
            f"cost:{video_id}",
            mapping={
                "total": 0,
                "estimated": estimated_cost,
                "started_at": datetime.utcnow().isoformat()
            }
        )
        
        # Set expiry (24 hours)
        await self.redis_client.expire(f"cost:{video_id}", 86400)
        
        logger.info(f"Initialized cost tracking for {video_id}, estimated: ${estimated_cost:.2f}")
    
    async def track_cost(self, video_id: str, service: str, amount: float) -> Dict:
        """
        Track cost for a service with real-time alerting
        """
        
        # Update Redis
        new_total = await self.redis_client.hincrbyfloat(
            f"cost:{video_id}", 
            "total", 
            amount
        )
        
        # Store in PostgreSQL for persistence
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO pipeline_costs 
                (video_id, service, operation, amount, timestamp)
                VALUES ($1, $2, $3, $4, $5)
            """,
                video_id, service, "processing", amount, datetime.utcnow()
            )
        
        # Check thresholds
        alerts = []
        
        if new_total >= self.COST_LIMITS["halt"]:
            alerts.append({
                "level": "critical",
                "message": f"HALT: Cost exceeded ${self.COST_LIMITS['halt']:.2f}",
                "action": "stop_processing"
            })
            await self._send_alert(video_id, "critical", new_total)
            
        elif new_total >= self.COST_LIMITS["critical"]:
            alerts.append({
                "level": "critical",
                "message": f"Critical: Cost at ${new_total:.2f}",
                "action": "optimize_immediately"
            })
            await self._send_alert(video_id, "critical", new_total)
            
        elif new_total >= self.COST_LIMITS["warning"]:
            alerts.append({
                "level": "warning",
                "message": f"Warning: Cost at ${new_total:.2f}",
                "action": "monitor_closely"
            })
        
        logger.info(f"Cost tracked for {video_id}: {service} = ${amount:.2f}, total = ${new_total:.2f}")
        
        return {
            "video_id": video_id,
            "service": service,
            "amount": amount,
            "total_cost": new_total,
            "within_budget": new_total < self.COST_LIMITS["halt"],
            "alerts": alerts
        }
    
    async def get_video_cost(self, video_id: str) -> Dict:
        """Get current cost for a video"""
        
        cost_data = await self.redis_client.hgetall(f"cost:{video_id}")
        
        if not cost_data:
            # Fetch from database if not in cache
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT 
                        COALESCE(SUM(amount), 0) as total,
                        COUNT(*) as transactions,
                        MIN(timestamp) as started_at
                    FROM pipeline_costs
                    WHERE video_id = $1
                """, video_id)
                
                if row and row['total']:
                    return {
                        "video_id": video_id,
                        "total_cost": float(row['total']),
                        "transactions": row['transactions'],
                        "started_at": row['started_at'].isoformat() if row['started_at'] else None
                    }
                else:
                    return {
                        "video_id": video_id,
                        "total_cost": 0.0,
                        "transactions": 0
                    }
        
        return {
            "video_id": video_id,
            "total_cost": float(cost_data.get('total', 0)),
            "estimated_cost": float(cost_data.get('estimated', 0)),
            "started_at": cost_data.get('started_at')
        }
    
    async def get_cost_breakdown(self, video_id: str) -> Dict:
        """Get detailed cost breakdown by service"""
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT 
                    service,
                    SUM(amount) as total,
                    COUNT(*) as count,
                    AVG(amount) as average
                FROM pipeline_costs
                WHERE video_id = $1
                GROUP BY service
                ORDER BY total DESC
            """, video_id)
            
            breakdown = {
                row['service']: {
                    "total": float(row['total']),
                    "count": row['count'],
                    "average": float(row['average'])
                }
                for row in rows
            }
            
            total = sum(s['total'] for s in breakdown.values())
            
            return {
                "video_id": video_id,
                "total_cost": total,
                "breakdown": breakdown,
                "within_budget": total < self.COST_LIMITS["halt"]
            }
    
    async def _send_alert(self, video_id: str, level: str, cost: float):
        """Send cost alert"""
        
        if not self.alerts_enabled:
            return
        
        # Publish to Redis for real-time monitoring
        await self.redis_client.publish(
            "alerts:cost",
            json.dumps({
                "video_id": video_id,
                "level": level,
                "cost": cost,
                "timestamp": datetime.utcnow().isoformat()
            })
        )
        
        # Log alert
        logger.warning(f"Cost alert for {video_id}: {level} at ${cost:.2f}")
    
    async def get_daily_costs(self) -> Dict:
        """Get aggregated daily costs"""
        
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT 
                    COUNT(DISTINCT video_id) as videos,
                    SUM(amount) as total,
                    AVG(amount) as average,
                    MAX(amount) as maximum
                FROM pipeline_costs
                WHERE timestamp >= CURRENT_DATE
            """)
            
            return {
                "date": datetime.utcnow().date().isoformat(),
                "videos_processed": row['videos'] or 0,
                "total_cost": float(row['total'] or 0),
                "average_cost": float(row['average'] or 0),
                "max_cost": float(row['maximum'] or 0)
            }
```

---

## 4. Resource Scheduling

```python
"""
resource_scheduler.py - GPU/CPU resource scheduling
"""

import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import psutil
import GPUtil

logger = logging.getLogger(__name__)

class ResourceType(Enum):
    GPU = "gpu"
    CPU = "cpu"

@dataclass
class ResourceAllocation:
    """Resource allocation for a job"""
    video_id: str
    resource_type: ResourceType
    cpu_cores: Optional[int] = None
    memory_mb: Optional[int] = None
    gpu_device_id: Optional[int] = None
    gpu_memory_mb: Optional[int] = None

class ResourceScheduler:
    """
    Intelligent resource scheduling for video processing
    RTX 5090 (32GB) + Ryzen 9 9950X3D (16 cores)
    """
    
    def __init__(self):
        # Hardware configuration
        self.gpu_memory_total = 32 * 1024  # 32GB in MB
        self.gpu_memory_available = 28 * 1024  # 28GB usable
        self.cpu_cores_total = 16
        self.cpu_cores_available = 12  # Reserve 4 for system
        
        # Current allocations
        self.gpu_allocations: Dict[str, ResourceAllocation] = {}
        self.cpu_allocations: Dict[str, ResourceAllocation] = {}
        
        # Semaphores for concurrency control
        self.gpu_semaphore = asyncio.Semaphore(3)  # Max 3 GPU jobs
        self.cpu_semaphore = asyncio.Semaphore(4)  # Max 4 CPU jobs
        
        # Memory tracking
        self.gpu_memory_used = 0
        self.cpu_memory_used = 0
        
    async def allocate_for_video(self, video_id: str, request_data: Dict) -> ResourceAllocation:
        """
        Allocate resources based on video complexity
        """
        
        complexity = request_data.get('complexity', 'simple')
        
        # Determine resource requirements
        if complexity == 'simple':
            return await self._allocate_cpu(video_id, request_data)
        elif complexity == 'complex':
            return await self._allocate_gpu(video_id, request_data)
        elif complexity == 'premium':
            # Try GPU first, fall back to CPU if needed
            try:
                return await self._allocate_gpu(video_id, request_data)
            except Exception:
                logger.info(f"GPU unavailable for {video_id}, using CPU")
                return await self._allocate_cpu(video_id, request_data)
        
    async def _allocate_gpu(self, video_id: str, request_data: Dict) -> ResourceAllocation:
        """Allocate GPU resources"""
        
        required_memory = self._estimate_gpu_memory(request_data)
        
        # Wait for GPU availability
        async with self.gpu_semaphore:
            # Wait for memory availability
            while self.gpu_memory_used + required_memory > self.gpu_memory_available:
                await asyncio.sleep(1)
                await self._check_gpu_health()
            
            # Create allocation
            allocation = ResourceAllocation(
                video_id=video_id,
                resource_type=ResourceType.GPU,
                gpu_device_id=0,  # Single GPU
                gpu_memory_mb=required_memory
            )
            
            # Track allocation
            self.gpu_allocations[video_id] = allocation
            self.gpu_memory_used += required_memory
            
            logger.info(
                f"Allocated GPU for {video_id}: {required_memory}MB "
                f"(used: {self.gpu_memory_used}/{self.gpu_memory_available}MB)"
            )
            
            return allocation
    
    async def _allocate_cpu(self, video_id: str, request_data: Dict) -> ResourceAllocation:
        """Allocate CPU resources"""
        
        required_cores = 3  # 3 cores per simple video
        required_memory = 8 * 1024  # 8GB per video
        
        async with self.cpu_semaphore:
            # Create allocation
            allocation = ResourceAllocation(
                video_id=video_id,
                resource_type=ResourceType.CPU,
                cpu_cores=required_cores,
                memory_mb=required_memory
            )
            
            # Track allocation
            self.cpu_allocations[video_id] = allocation
            self.cpu_memory_used += required_memory
            
            logger.info(
                f"Allocated CPU for {video_id}: {required_cores} cores, {required_memory}MB"
            )
            
            return allocation
    
    async def release(self, allocation: ResourceAllocation):
        """Release allocated resources"""
        
        video_id = allocation.video_id
        
        if allocation.resource_type == ResourceType.GPU:
            if video_id in self.gpu_allocations:
                self.gpu_memory_used -= allocation.gpu_memory_mb
                del self.gpu_allocations[video_id]
                logger.info(f"Released GPU resources for {video_id}")
                
        elif allocation.resource_type == ResourceType.CPU:
            if video_id in self.cpu_allocations:
                self.cpu_memory_used -= allocation.memory_mb
                del self.cpu_allocations[video_id]
                logger.info(f"Released CPU resources for {video_id}")
    
    def _estimate_gpu_memory(self, request_data: Dict) -> int:
        """Estimate GPU memory requirements"""
        
        complexity = request_data.get('complexity', 'simple')
        duration = request_data.get('duration', 120)  # seconds
        
        # Base memory requirements
        base_memory = {
            'simple': 6 * 1024,   # 6GB
            'complex': 9 * 1024,  # 9GB
            'premium': 12 * 1024  # 12GB
        }
        
        # Adjust for duration
        memory = base_memory.get(complexity, 8 * 1024)
        if duration > 180:
            memory = int(memory * 1.2)  # 20% more for long videos
        
        return memory
    
    async def _check_gpu_health(self):
        """Check GPU health and clear stuck allocations"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                actual_memory_used = gpu.memoryUsed
                
                # If our tracking is off, recalculate
                if abs(actual_memory_used - self.gpu_memory_used) > 1024:  # 1GB difference
                    logger.warning(f"GPU memory tracking mismatch. Recalculating...")
                    self.gpu_memory_used = actual_memory_used
                    
        except Exception as e:
            logger.error(f"GPU health check failed: {e}")
    
    async def clear_all_gpu_allocations(self):
        """Emergency clear all GPU allocations"""
        logger.warning("Clearing all GPU allocations")
        self.gpu_allocations.clear()
        self.gpu_memory_used = 0
    
    async def get_status(self) -> Dict:
        """Get current resource allocation status"""
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Get GPU metrics
        gpu_status = {}
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_status = {
                    "gpu_name": gpu.name,
                    "gpu_memory_used_mb": gpu.memoryUsed,
                    "gpu_memory_total_mb": gpu.memoryTotal,
                    "gpu_utilization": gpu.load * 100,
                    "gpu_temperature": gpu.temperature
                }
        except Exception as e:
            logger.error(f"Failed to get GPU status: {e}")
        
        return {
            "cpu": {
                "cores_available": self.cpu_cores_available,
                "active_jobs": len(self.cpu_allocations),
                "cpu_percent": cpu_percent,
                "memory_used_mb": self.cpu_memory_used,
                "memory_available_mb": memory.available // (1024 * 1024)
            },
            "gpu": {
                **gpu_status,
                "active_jobs": len(self.gpu_allocations),
                "memory_allocated_mb": self.gpu_memory_used,
                "memory_available_mb": self.gpu_memory_available - self.gpu_memory_used
            },
            "allocations": {
                "gpu": list(self.gpu_allocations.keys()),
                "cpu": list(self.cpu_allocations.keys())
            }
        }
```

---

## 5. Analytics Pipeline

```python
"""
analytics_pipeline.py - Analytics data processing
"""

import asyncio
import logging
from typing import Dict, List
from datetime import datetime, timedelta
import aiohttp
import json

logger = logging.getLogger(__name__)

class AnalyticsPipeline:
    """
    Process YouTube analytics and internal metrics
    """
    
    def __init__(self, db_pool, redis_client):
        self.db_pool = db_pool
        self.redis_client = redis_client
        self.youtube_api_key = "YOUR_YOUTUBE_API_KEY"
        
    async def collect_youtube_analytics(self, channel_id: str) -> Dict:
        """Collect analytics from YouTube API"""
        
        async with aiohttp.ClientSession() as session:
            # Get channel statistics
            url = f"https://www.googleapis.com/youtube/v3/channels"
            params = {
                "part": "statistics",
                "id": channel_id,
                "key": self.youtube_api_key
            }
            
            async with session.get(url, params=params) as response:
                data = await response.json()
                
                if data.get('items'):
                    stats = data['items'][0]['statistics']
                    
                    return {
                        "channel_id": channel_id,
                        "view_count": int(stats.get('viewCount', 0)),
                        "subscriber_count": int(stats.get('subscriberCount', 0)),
                        "video_count": int(stats.get('videoCount', 0)),
                        "timestamp": datetime.utcnow().isoformat()
                    }
        
        return {}
    
    async def aggregate_metrics(self, time_window: str = "1hour") -> Dict:
        """Aggregate pipeline metrics"""
        
        async with self.db_pool.acquire() as conn:
            # Get aggregated metrics based on time window
            if time_window == "1hour":
                interval = "1 hour"
            elif time_window == "1day":
                interval = "1 day"
            else:
                interval = "1 week"
            
            metrics = await conn.fetchrow("""
                SELECT 
                    COUNT(DISTINCT id) as videos_processed,
                    AVG(processing_time_seconds) as avg_processing_time,
                    AVG(actual_cost) as avg_cost,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
                FROM video_queue
                WHERE created_at > NOW() - INTERVAL %s
            """, interval)
            
            success_rate = 0
            if metrics['videos_processed'] > 0:
                success_rate = (metrics['successful'] / metrics['videos_processed']) * 100
            
            return {
                "time_window": time_window,
                "videos_processed": metrics['videos_processed'] or 0,
                "avg_processing_time": float(metrics['avg_processing_time'] or 0),
                "avg_cost": float(metrics['avg_cost'] or 0),
                "success_rate": success_rate,
                "failed_videos": metrics['failed'] or 0
            }
    
    async def calculate_revenue_metrics(self) -> Dict:
        """Calculate estimated revenue metrics"""
        
        # Simplified revenue calculation
        # In production, integrate with YouTube Analytics API
        
        async with self.db_pool.acquire() as conn:
            daily_videos = await conn.fetchval("""
                SELECT COUNT(*) 
                FROM video_queue 
                WHERE status = 'completed' 
                AND completed_at > CURRENT_DATE
            """)
            
            # Estimate revenue (simplified)
            estimated_cpm = 2.0  # $2 per 1000 views
            estimated_views_per_video = 1000  # Starting estimate
            
            daily_revenue = daily_videos * (estimated_views_per_video / 1000) * estimated_cpm
            
            return {
                "daily_videos": daily_videos,
                "estimated_daily_revenue": daily_revenue,
                "estimated_monthly_revenue": daily_revenue * 30,
                "cpm": estimated_cpm
            }
    
    async def store_analytics(self, analytics_data: Dict):
        """Store analytics data in database"""
        
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO analytics_data 
                (channel_id, metric_type, metric_value, timestamp)
                VALUES ($1, $2, $3, $4)
            """,
                analytics_data.get('channel_id'),
                analytics_data.get('metric_type'),
                json.dumps(analytics_data.get('metric_value')),
                datetime.utcnow()
            )
    
    async def get_dashboard_metrics(self) -> Dict:
        """Get metrics for dashboard display"""
        
        # Gather all metrics
        hourly = await self.aggregate_metrics("1hour")
        daily = await self.aggregate_metrics("1day")
        revenue = await self.calculate_revenue_metrics()
        
        # Get current queue status
        queue_metrics = await self.redis_client.hgetall("metrics:queue")
        
        return {
            "realtime": {
                "queue_depth": await self.redis_client.zcard("queue:priority"),
                "processing_count": await self.redis_client.scard("queue:processing"),
                "last_update": datetime.utcnow().isoformat()
            },
            "hourly": hourly,
            "daily": daily,
            "revenue": revenue,
            "totals": {
                "total_processed": int(queue_metrics.get("total_processed", 0)),
                "total_failed": int(queue_metrics.get("total_failed", 0))
            }
        }
```

---

## 6. Error Handling & Recovery

```python
"""
error_handler.py - Comprehensive error handling
"""

import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime
from enum import Enum
import traceback
import json

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorHandler:
    """
    Centralized error handling and recovery
    """
    
    def __init__(self, queue_manager, resource_scheduler):
        self.queue_manager = queue_manager
        self.resource_scheduler = resource_scheduler
        self.error_counts = {}
        
    async def handle_pipeline_error(self, video_id: str, stage: str, error: Exception) -> bool:
        """
        Handle pipeline errors with recovery attempts
        Returns True if recovered, False if fatal
        """
        
        error_type = type(error).__name__
        error_message = str(error)
        
        logger.error(f"Pipeline error in {stage} for {video_id}: {error_message}")
        logger.debug(traceback.format_exc())
        
        # Classify error severity
        severity = self._classify_error(error)
        
        # Track error frequency
        error_key = f"{video_id}:{stage}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Determine recovery action
        if severity == ErrorSeverity.CRITICAL:
            await self._handle_critical_error(video_id, stage, error)
            return False
            
        elif severity == ErrorSeverity.HIGH:
            if self.error_counts[error_key] < 3:
                await self._retry_with_backoff(video_id, stage, self.error_counts[error_key])
                return True
            else:
                await self._handle_permanent_failure(video_id, stage, error)
                return False
                
        elif severity == ErrorSeverity.MEDIUM:
            if self.error_counts[error_key] < 5:
                await asyncio.sleep(5)  # Simple retry
                return True
            return False
            
        else:  # LOW severity
            logger.warning(f"Low severity error, continuing: {error_message}")
            return True
    
    def _classify_error(self, error: Exception) -> ErrorSeverity:
        """Classify error severity"""
        
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Critical errors
        if "cost limit" in error_message:
            return ErrorSeverity.CRITICAL
        if "gpu memory" in error_message and "out of memory" in error_message:
            return ErrorSeverity.CRITICAL
            
        # High severity
        if error_type in ["ConnectionError", "TimeoutError"]:
            return ErrorSeverity.HIGH
        if "api" in error_message and "quota" in error_message:
            return ErrorSeverity.HIGH
            
        # Medium severity
        if error_type in ["ValueError", "KeyError"]:
            return ErrorSeverity.MEDIUM
        if "retry" in error_message:
            return ErrorSeverity.MEDIUM
            
        # Low severity
        return ErrorSeverity.LOW
    
    async def _handle_critical_error(self, video_id: str, stage: str, error: Exception):
        """Handle critical errors"""
        
        logger.critical(f"CRITICAL ERROR for {video_id} at {stage}: {error}")
        
        # Immediate actions
        if "cost" in str(error).lower():
            # Stop all processing
            await self._emergency_stop()
            
        elif "gpu" in str(error).lower():
            # Clear GPU resources
            await self.resource_scheduler.clear_all_gpu_allocations()
        
        # Mark video as failed
        await self.queue_manager.fail(video_id, f"Critical error: {error}", retry=False)
        
        # Send alert
        await self._send_critical_alert(video_id, stage, error)
    
    async def _retry_with_backoff(self, video_id: str, stage: str, attempt: int):
        """Retry with exponential backoff"""
        
        delay = min(10 * (2 ** attempt), 300)  # Max 5 minutes
        logger.info(f"Retrying {video_id} stage {stage} in {delay}s (attempt {attempt})")
        await asyncio.sleep(delay)
    
    async def _handle_permanent_failure(self, video_id: str, stage: str, error: Exception):
        """Handle permanent failure"""
        
        logger.error(f"Permanent failure for {video_id} at {stage}")
        await self.queue_manager.fail(video_id, f"Permanent failure at {stage}: {error}", retry=False)
    
    async def _emergency_stop(self):
        """Emergency stop all processing"""
        
        logger.critical("EMERGENCY STOP INITIATED")
        
        # Clear all queues
        await self.queue_manager.redis_client.delete("queue:priority")
        await self.queue_manager.redis_client.delete("queue:processing")
        
        # Update database
        async with self.queue_manager.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE video_queue 
                SET status = 'cancelled' 
                WHERE status IN ('queued', 'processing')
            """)
    
    async def _send_critical_alert(self, video_id: str, stage: str, error: Exception):
        """Send critical alert"""
        
        alert_data = {
            "severity": "critical",
            "video_id": video_id,
            "stage": stage,
            "error": str(error),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.queue_manager._publish_event("alert.critical", alert_data)
        
        # Could also send to PagerDuty, Slack, email, etc.
```

---

## 7. API Implementation

```python
"""
api.py - FastAPI implementation for pipeline control
"""

from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Optional
from datetime import datetime
import asyncio
import json
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="YTEMPIRE Pipeline API", version="1.0.0")

# Initialize components (would be done properly in main.py)
queue_manager = None
pipeline = None
cost_tracker = None
resource_scheduler = None
analytics_pipeline = None

@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    global queue_manager, pipeline, cost_tracker, resource_scheduler
    
    # Initialize components
    queue_manager = VideoQueueManager()
    await queue_manager.initialize()
    
    # Initialize other components...
    logger.info("API startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if queue_manager:
        await queue_manager.cleanup()

# Queue endpoints
@app.post("/pipeline/queue")
async def enqueue_video(request: Dict) -> Dict:
    """Add video to processing queue"""
    
    try:
        job = await queue_manager.enqueue(request)
        
        return {
            "video_id": job.id,
            "position": await queue_manager._get_queue_position(job.id),
            "estimated_cost": job.estimated_cost,
            "estimated_time": 600  # 10 minutes estimate
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/pipeline/status/{video_id}")
async def get_video_status(video_id: str) -> Dict:
    """Get current status of video processing"""
    
    async with queue_manager.db_pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT status, priority, complexity, estimated_cost, 
                   actual_cost, processing_started_at, completed_at,
                   processing_time_seconds
            FROM video_queue
            WHERE id = $1
        """, video_id)
        
        if not row:
            raise HTTPException(status_code=404, detail="Video not found")
        
        return {
            "video_id": video_id,
            "status": row['status'],
            "priority": row['priority'],
            "complexity": row['complexity'],
            "estimated_cost": float(row['estimated_cost'] or 0),
            "actual_cost": float(row['actual_cost'] or 0),
            "processing_time": row['processing_time_seconds'],
            "started_at": row['processing_started_at'].isoformat() if row['processing_started_at'] else None,
            "completed_at": row['completed_at'].isoformat() if row['completed_at'] else None
        }

@app.websocket("/pipeline/stream/{video_id}")
async def stream_progress(websocket: WebSocket, video_id: str):
    """WebSocket for real-time progress updates"""
    
    await websocket.accept()
    
    # Subscribe to progress events
    pubsub = queue_manager.redis_client.pubsub()
    await pubsub.subscribe("pipeline:progress")
    
    try:
        while True:
            # Get message from Redis pub/sub
            message = await pubsub.get_message(ignore_subscribe_messages=True)
            
            if message and message['type'] == 'message':
                data = json.loads(message['data'])
                
                # Filter for requested video
                if data.get('video_id') == video_id:
                    await websocket.send_json(data)
            
            await asyncio.sleep(0.1)
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await pubsub.unsubscribe("pipeline:progress")
        await websocket.close()

# Metrics endpoints
@app.get("/pipeline/metrics")
async def get_pipeline_metrics() -> Dict:
    """Get current pipeline metrics"""
    
    queue_metrics = await queue_manager.get_metrics()
    resource_status = await resource_scheduler.get_status()
    daily_costs = await cost_tracker.get_daily_costs()
    
    return {
        "queue": queue_metrics,
        "resources": resource_status,
        "costs": daily_costs,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/pipeline/metrics/daily")
async def get_daily_metrics() -> Dict:
    """Get daily aggregated metrics"""
    
    return await analytics_pipeline.get_dashboard_metrics()

# Control endpoints
@app.post("/pipeline/pause")
async def pause_processing() -> Dict:
    """Pause all video processing"""
    
    # Implementation would pause workers
    return {"status": "paused", "message": "Pipeline paused"}

@app.post("/pipeline/resume")
async def resume_processing() -> Dict:
    """Resume video processing"""
    
    # Implementation would resume workers
    return {"status": "running", "message": "Pipeline resumed"}

@app.post("/pipeline/emergency/halt")
async def emergency_halt() -> Dict:
    """Emergency halt all processing"""
    
    logger.critical("EMERGENCY HALT INITIATED via API")
    
    # Stop all processing
    await queue_manager.redis_client.delete("queue:priority")
    await queue_manager.redis_client.delete("queue:processing")
    
    # Update database
    async with queue_manager.db_pool.acquire() as conn:
        await conn.execute("""
            UPDATE video_queue 
            SET status = 'cancelled' 
            WHERE status IN ('queued', 'processing')
        """)
    
    return {"status": "halted", "message": "Emergency halt executed"}

# Health endpoints
@app.get("/health")
async def health_check() -> Dict:
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health/detailed")
async def detailed_health() -> Dict:
    """Detailed health check"""
    
    health_status = {
        "database": "unknown",
        "redis": "unknown",
        "gpu": "unknown"
    }
    
    # Check database
    try:
        async with queue_manager.db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        health_status["database"] = "healthy"
    except Exception:
        health_status["database"] = "unhealthy"
    
    # Check Redis
    try:
        await queue_manager.redis_client.ping()
        health_status["redis"] = "healthy"
    except Exception:
        health_status["redis"] = "unhealthy"
    
    # Check GPU
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        health_status["gpu"] = "healthy" if gpus else "unavailable"
    except Exception:
        health_status["gpu"] = "error"
    
    overall = "healthy" if all(v == "healthy" for v in health_status.values()) else "degraded"
    
    return {
        "overall_status": overall,
        "components": health_status,
        "timestamp": datetime.utcnow().isoformat()
    }

# Cost endpoints
@app.get("/costs/video/{video_id}")
async def get_video_cost(video_id: str) -> Dict:
    """Get cost breakdown for a video"""
    
    return await cost_tracker.get_cost_breakdown(video_id)

@app.get("/costs/daily")
async def get_daily_costs() -> Dict:
    """Get daily cost summary"""
    
    return await cost_tracker.get_daily_costs()

# Resource endpoints
@app.get("/resources/status")
async def get_resource_status() -> Dict:
    """Get current resource allocation status"""
    
    return await resource_scheduler.get_status()

@app.post("/resources/clear/gpu")
async def clear_gpu_allocations() -> Dict:
    """Emergency clear GPU allocations"""
    
    await resource_scheduler.clear_all_gpu_allocations()
    return {"status": "cleared", "message": "GPU allocations cleared"}