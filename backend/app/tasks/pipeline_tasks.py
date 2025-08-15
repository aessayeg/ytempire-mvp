"""
Enhanced Video Processing Pipeline Tasks for 100+ Videos/Day
Week 2 Implementation: Distributed processing, batch operations, and auto-scaling
Comprehensive Celery task definitions for video automation
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from celery import chain, group, chord, signature
from celery.result import AsyncResult
from celery.exceptions import SoftTimeLimitExceeded
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import and_, or_
import redis

from app.core.celery_app import celery_app, TaskPriority
from app.db.session import get_db
from app.models.video import Video, VideoStatus
from app.models.channel import Channel
from app.models.cost import Cost
from app.services.ai_services import AIServiceManager
from app.services.video_generation_pipeline import VideoProcessor
from app.services.youtube_multi_account import get_youtube_manager
from app.services.cost_optimizer import get_cost_optimizer, ServiceType
from app.services.n8n_integration import get_n8n_integration, WorkflowType
from app.websocket.manager import websocket_manager

logger = logging.getLogger(__name__)

# Initialize Redis for distributed locking and caching
redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))

class PipelineMetrics:
    """Track pipeline performance metrics for optimization"""
    
    @staticmethod
    def record_metric(metric_name: str, value: float, tags: Dict[str, str] = None):
        """Record a metric for monitoring"""
        metric_data = {
            "metric": metric_name,
            "value": value,
            "timestamp": datetime.utcnow().isoformat(),
            "tags": tags or {}
        }
        redis_client.lpush("metrics:pipeline", json.dumps(metric_data))
        redis_client.ltrim("metrics:pipeline", 0, 10000)  # Keep last 10k metrics
    
    @staticmethod
    def get_throughput() -> float:
        """Calculate current pipeline throughput (videos/hour)"""
        hour_ago = datetime.utcnow() - timedelta(hours=1)
        with next(get_db()) as db:
            count = db.query(Video).filter(
                Video.completed_at >= hour_ago,
                Video.status == VideoStatus.PUBLISHED
            ).count()
        return count


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


# Enhanced Batch Processing for 100+ Videos/Day

@celery_app.task(name="pipeline.batch_process_optimized", priority=TaskPriority.HIGH)
def batch_process_optimized(
    channel_id: int,
    video_count: int = 10,
    parallel_limit: int = 5,
    style: str = "educational"
) -> Dict[str, Any]:
    """
    Optimized batch processing for high throughput
    Supports 100+ videos/day with intelligent resource management
    """
    start_time = datetime.utcnow()
    batch_id = f"batch_{channel_id}_{start_time.timestamp()}"
    
    try:
        logger.info(f"Starting optimized batch processing: {batch_id}")
        
        # Check current system load
        current_throughput = PipelineMetrics.get_throughput()
        queue_depth = redis_client.llen("celery:queue:video_processing")
        
        # Adjust parallelism based on system load
        if queue_depth > 50:
            parallel_limit = max(2, parallel_limit // 2)
            logger.info(f"Reducing parallelism to {parallel_limit} due to high queue depth")
        
        # Create video records in database
        video_ids = []
        with next(get_db()) as db:
            for i in range(video_count):
                video = Video(
                    channel_id=channel_id,
                    title=f"Batch Video {i+1}",
                    status=VideoStatus.PENDING,
                    created_at=datetime.utcnow()
                )
                db.add(video)
            db.commit()
            
            # Get IDs
            videos = db.query(Video).filter(
                Video.channel_id == channel_id,
                Video.status == VideoStatus.PENDING
            ).order_by(Video.created_at.desc()).limit(video_count).all()
            video_ids = [v.id for v in videos]
        
        # Process in parallel batches
        batches = [video_ids[i:i + parallel_limit] for i in range(0, len(video_ids), parallel_limit)]
        results = {"successful": [], "failed": [], "batch_id": batch_id}
        
        for batch_idx, batch in enumerate(batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(batches)}")
            
            # Create parallel task group
            batch_group = group(
                process_video_with_retry.s(vid, style)
                for vid in batch
            )
            
            # Execute batch with monitoring
            batch_result = batch_group.apply_async()
            
            # Monitor batch completion
            for idx, task_result in enumerate(batch_result):
                try:
                    result = task_result.get(timeout=1800)  # 30 min timeout
                    results["successful"].append(result)
                    
                    # Update metrics
                    PipelineMetrics.record_metric(
                        "batch.video.completed",
                        1,
                        {"batch_id": batch_id, "video_id": str(batch[idx])}
                    )
                    
                except Exception as e:
                    logger.error(f"Video {batch[idx]} failed: {str(e)}")
                    results["failed"].append({
                        "video_id": batch[idx],
                        "error": str(e)
                    })
            
            # Throttle between batches to prevent overload
            if batch_idx < len(batches) - 1:
                import time
                time.sleep(10)  # 10 second delay
        
        # Record overall metrics
        duration = (datetime.utcnow() - start_time).total_seconds()
        success_rate = len(results["successful"]) / video_count if video_count > 0 else 0
        
        PipelineMetrics.record_metric(
            "batch.completed",
            video_count,
            {
                "batch_id": batch_id,
                "duration": str(duration),
                "success_rate": str(success_rate)
            }
        )
        
        return {
            "batch_id": batch_id,
            "total_videos": video_count,
            "successful": len(results["successful"]),
            "failed": len(results["failed"]),
            "duration_seconds": duration,
            "throughput_per_hour": (video_count / duration) * 3600 if duration > 0 else 0,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        raise


@celery_app.task(
    name="pipeline.process_video_with_retry",
    bind=True,
    max_retries=3,
    default_retry_delay=60
)
def process_video_with_retry(self, video_id: int, style: str = "educational") -> Dict[str, Any]:
    """Process single video with retry logic and optimization"""
    try:
        # Check if video is already being processed (distributed lock)
        lock_key = f"video:processing:{video_id}"
        if not redis_client.set(lock_key, "1", nx=True, ex=3600):
            logger.warning(f"Video {video_id} already being processed")
            return {"status": "duplicate", "video_id": video_id}
        
        # Execute optimized pipeline
        result = orchestrate_full_pipeline(
            channel_id=None,  # Will be fetched from video record
            video_id=video_id,
            style=style,
            auto_upload=True
        )
        
        # Release lock
        redis_client.delete(lock_key)
        
        return result
        
    except SoftTimeLimitExceeded:
        logger.error(f"Video {video_id} processing timed out")
        self.retry(countdown=120)
    except Exception as e:
        logger.error(f"Video {video_id} processing failed: {str(e)}")
        if self.request.retries < self.max_retries:
            self.retry(exc=e, countdown=60 * (self.request.retries + 1))
        raise


@celery_app.task(name="pipeline.monitor_throughput")
def monitor_and_optimize_throughput() -> Dict[str, Any]:
    """Monitor pipeline throughput and suggest optimizations"""
    current_throughput = PipelineMetrics.get_throughput()
    target_throughput = 100 / 24  # 100 videos per day
    
    recommendations = []
    metrics = {}
    
    # Check queue depths
    for queue_name in ["video_processing", "ai_generation", "youtube_upload"]:
        depth = redis_client.llen(f"celery:queue:{queue_name}")
        metrics[f"queue_{queue_name}_depth"] = depth
        
        if depth > 20:
            recommendations.append(f"Scale up {queue_name} workers (depth: {depth})")
    
    # Check processing times
    recent_videos = []
    with next(get_db()) as db:
        recent_videos = db.query(Video).filter(
            Video.completed_at.isnot(None),
            Video.created_at.isnot(None)
        ).order_by(Video.completed_at.desc()).limit(10).all()
    
    if recent_videos:
        avg_processing_time = sum([
            (v.completed_at - v.created_at).total_seconds()
            for v in recent_videos
        ]) / len(recent_videos)
        
        metrics["avg_processing_time_seconds"] = avg_processing_time
        
        if avg_processing_time > 600:  # 10 minutes
            recommendations.append(f"Optimize pipeline stages (avg time: {avg_processing_time:.0f}s)")
    
    # Check cost efficiency
    with next(get_db()) as db:
        recent_costs = db.query(Cost).filter(
            Cost.created_at >= datetime.utcnow() - timedelta(hours=1)
        ).all()
        
        if recent_costs:
            avg_cost = sum([c.amount for c in recent_costs]) / len(recent_costs)
            metrics["avg_cost_per_video"] = avg_cost
            
            if avg_cost > 2.50:
                recommendations.append(f"Optimize costs (avg: ${avg_cost:.2f}/video)")
    
    # Performance assessment
    if current_throughput < target_throughput:
        recommendations.append(f"Increase throughput (current: {current_throughput:.1f}, target: {target_throughput:.1f} videos/hour)")
    
    return {
        "current_throughput": current_throughput,
        "target_throughput": target_throughput,
        "metrics": metrics,
        "recommendations": recommendations,
        "timestamp": datetime.utcnow().isoformat()
    }


# Auto-scaling helper tasks

@celery_app.task(name="pipeline.scale_workers")
def auto_scale_workers() -> Dict[str, Any]:
    """Auto-scale workers based on queue depth and throughput"""
    from app.core.celery_app import AutoScalingConfig
    
    scaled_queues = []
    
    for queue_name, config in AutoScalingConfig.QUEUE_SCALING.items():
        queue_key = f"celery:queue:{queue_name}"
        depth = redis_client.llen(queue_key)
        
        if depth > config["threshold"]:
            # Need to scale up
            scaled_queues.append({
                "queue": queue_name,
                "action": "scale_up",
                "depth": depth,
                "new_workers": config["max"]
            })
            
            # Record scaling event
            PipelineMetrics.record_metric(
                "autoscale.triggered",
                1,
                {"queue": queue_name, "action": "scale_up", "depth": str(depth)}
            )
    
    return {
        "scaled_queues": scaled_queues,
        "timestamp": datetime.utcnow().isoformat()
    }


# Schedule periodic optimization tasks
celery_app.conf.beat_schedule.update({
    "monitor-throughput": {
        "task": "pipeline.monitor_throughput",
        "schedule": 300.0,  # Every 5 minutes
    },
    "auto-scale-workers": {
        "task": "pipeline.scale_workers",
        "schedule": 60.0,  # Every minute
    },
    "optimize-batch-processing": {
        "task": "pipeline.monitor_and_optimize_throughput",
        "schedule": 600.0,  # Every 10 minutes
    }
})