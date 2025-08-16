"""
Master Video Generation Orchestrator
Coordinates all services for end-to-end video generation
Target: <10 min generation, <$3 per video
"""
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from enum import Enum
import logging
import json
from dataclasses import dataclass, asdict
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import WebSocket

from app.services.youtube_multi_account import get_youtube_manager

youtube_account_manager = get_youtube_manager()
from app.services.video_generation_pipeline import VideoGenerationPipeline

# Create compatibility for old names
VideoPipelineOrchestrator = VideoGenerationPipeline
PipelineStage = None  # Will be defined if needed
from app.services.cost_tracking import cost_tracker
from app.services.websocket_manager import ConnectionManager
from app.models.video import Video, VideoStatus
from app.models.channel import Channel
from app.models.cost import Cost as CostRecord
from app.db.session import get_db
from app.core.config import settings

# Import ML services
from app.services.ml_integration_service import (
    ml_service,
    get_personalized_video_content,
    predict_video_performance,
    update_channel_ml_profile,
)

# Import existing ML pipeline services (if available)
try:
    from ml_pipeline.src.trend_detection_model import TrendDetector
    from ml_pipeline.src.script_generation import ScriptGenerator
    from ml_pipeline.src.voice_synthesis import VoiceSynthesizer
    from ml_pipeline.src.thumbnail_generation import ThumbnailGenerator
    from ml_pipeline.src.content_quality_scorer import QualityScorer

    ML_PIPELINE_AVAILABLE = True
except ImportError:
    ML_PIPELINE_AVAILABLE = False

    # Fallback classes
    class TrendDetector:
        async def detect_trends(self, *args, **kwargs):
            return {"topics": ["AI", "Technology"], "scores": [0.9, 0.85]}

    class ScriptGenerator:
        async def generate_script(self, *args, **kwargs):
            return {"script": "Sample script content", "title": "Sample Title"}

    class VoiceSynthesizer:
        async def synthesize(self, *args, **kwargs):
            return {"audio_path": "/tmp/audio.mp3"}

    class ThumbnailGenerator:
        async def generate(self, *args, **kwargs):
            return {"thumbnail_path": "/tmp/thumbnail.jpg"}

    class QualityScorer:
        async def score(self, *args, **kwargs):
            return {"score": 85.0}


logger = logging.getLogger(__name__)


class GenerationPhase(Enum):
    """Video generation phases"""

    INITIALIZATION = "initialization"
    TREND_ANALYSIS = "trend_analysis"
    SCRIPT_GENERATION = "script_generation"
    VOICE_SYNTHESIS = "voice_synthesis"
    VISUAL_GENERATION = "visual_generation"
    VIDEO_ASSEMBLY = "video_assembly"
    QUALITY_CHECK = "quality_check"
    PUBLISHING = "publishing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GenerationMetrics:
    """Metrics for video generation"""

    video_id: str
    channel_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration_seconds: Optional[float] = None
    phase_durations: Dict[str, float] = None
    total_cost: Decimal = Decimal("0.00")
    cost_breakdown: Dict[str, Decimal] = None
    quality_score: Optional[float] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.phase_durations is None:
            self.phase_durations = {}
        if self.cost_breakdown is None:
            self.cost_breakdown = {}
        if self.errors is None:
            self.errors = []


class VideoGenerationOrchestrator:
    """
    Master orchestrator for end-to-end video generation
    Coordinates all services and tracks performance metrics
    """

    def __init__(self):
        self.pipeline = VideoPipelineOrchestrator()
        self.cost_tracker = cost_tracker
        self.ws_manager = ConnectionManager()
        self.trend_detector = TrendDetector()
        self.script_generator = ScriptGenerator()
        self.voice_synthesizer = VoiceSynthesizer()
        self.thumbnail_generator = ThumbnailGenerator()
        self.quality_scorer = QualityScorer()
        self.active_generations = {}
        self.target_duration = 600  # 10 minutes in seconds
        self.target_cost = Decimal("3.00")  # $3 per video

    async def generate_video(
        self,
        channel_id: str,
        topic: Optional[str] = None,
        db: AsyncSession = None,
        websocket: Optional[WebSocket] = None,
    ) -> Dict[str, Any]:
        """
        Generate a complete video from start to finish
        Returns generation metrics and status
        """
        video_id = f"video_{channel_id}_{int(time.time())}"
        metrics = GenerationMetrics(
            video_id=video_id, channel_id=channel_id, start_time=datetime.utcnow()
        )

        try:
            # Phase 1: Initialization
            await self._update_phase(
                video_id, GenerationPhase.INITIALIZATION, websocket
            )
            phase_start = time.time()

            # Get YouTube account for this channel
            youtube_account = await youtube_account_manager.get_best_account(
                quota_needed=2000
            )
            if not youtube_account:
                raise Exception("No YouTube account available")

            # Create video record in database
            video = await self._create_video_record(video_id, channel_id, db)

            metrics.phase_durations["initialization"] = time.time() - phase_start

            # Phase 2: Trend Analysis
            await self._update_phase(
                video_id, GenerationPhase.TREND_ANALYSIS, websocket
            )
            phase_start = time.time()

            if not topic:
                # Detect trending topic
                trending_topics = await self.trend_detector.get_trending_topics(
                    category="technology", limit=5  # Can be dynamic based on channel
                )
                topic = trending_topics[0] if trending_topics else "Latest Tech Trends"

            # Track cost for trend analysis
            trend_cost = Decimal("0.05")  # API calls for trend detection
            await self.cost_tracker.track_cost(
                service="trend_detection",
                operation="analyze",
                amount=trend_cost,
                video_id=video_id,
            )
            metrics.cost_breakdown["trend_analysis"] = trend_cost

            metrics.phase_durations["trend_analysis"] = time.time() - phase_start

            # Phase 3: Script Generation
            await self._update_phase(
                video_id, GenerationPhase.SCRIPT_GENERATION, websocket
            )
            phase_start = time.time()

            script_data = await self.script_generator.generate_script(
                topic=topic,
                duration_minutes=8,  # Target 8-minute videos
                style="engaging_educational",
                model="gpt-3.5-turbo",  # Start with cheaper model
            )

            # Track script generation cost
            script_cost = Decimal(str(script_data.get("cost", 0.10)))
            await self.cost_tracker.track_cost(
                service="openai",
                operation="script_generation",
                amount=script_cost,
                video_id=video_id,
            )
            metrics.cost_breakdown["script_generation"] = script_cost

            metrics.phase_durations["script_generation"] = time.time() - phase_start

            # Phase 4: Voice Synthesis
            await self._update_phase(
                video_id, GenerationPhase.VOICE_SYNTHESIS, websocket
            )
            phase_start = time.time()

            voice_data = await self.voice_synthesizer.synthesize_voice(
                text=script_data["script"],
                voice_id="adam",  # Default voice
                service="google_tts",  # Cheaper than ElevenLabs
            )

            # Track voice synthesis cost
            voice_cost = Decimal(str(voice_data.get("cost", 0.20)))
            await self.cost_tracker.track_cost(
                service="google_tts",
                operation="voice_synthesis",
                amount=voice_cost,
                video_id=video_id,
            )
            metrics.cost_breakdown["voice_synthesis"] = voice_cost

            metrics.phase_durations["voice_synthesis"] = time.time() - phase_start

            # Phase 5: Visual Generation (Thumbnail + Video frames)
            await self._update_phase(
                video_id, GenerationPhase.VISUAL_GENERATION, websocket
            )
            phase_start = time.time()

            # Generate thumbnail
            thumbnail_data = await self.thumbnail_generator.generate_thumbnail(
                title=script_data["title"], style="youtube_optimized", model="dall-e-3"
            )

            # Track thumbnail cost
            thumbnail_cost = Decimal(str(thumbnail_data.get("cost", 0.04)))
            await self.cost_tracker.track_cost(
                service="openai",
                operation="thumbnail_generation",
                amount=thumbnail_cost,
                video_id=video_id,
            )
            metrics.cost_breakdown["thumbnail_generation"] = thumbnail_cost

            metrics.phase_durations["visual_generation"] = time.time() - phase_start

            # Phase 6: Video Assembly
            await self._update_phase(
                video_id, GenerationPhase.VIDEO_ASSEMBLY, websocket
            )
            phase_start = time.time()

            # Simulate video assembly (would use FFmpeg in production)
            await asyncio.sleep(2)  # Simulate processing
            video_file_path = f"/tmp/videos/{video_id}.mp4"

            # Track assembly cost (compute resources)
            assembly_cost = Decimal("0.10")
            await self.cost_tracker.track_cost(
                service="compute",
                operation="video_assembly",
                amount=assembly_cost,
                video_id=video_id,
            )
            metrics.cost_breakdown["video_assembly"] = assembly_cost

            metrics.phase_durations["video_assembly"] = time.time() - phase_start

            # Phase 7: Quality Check
            await self._update_phase(video_id, GenerationPhase.QUALITY_CHECK, websocket)
            phase_start = time.time()

            quality_score = await self.quality_scorer.score_content(
                script=script_data["script"],
                title=script_data["title"],
                description=script_data.get("description", ""),
                tags=script_data.get("tags", []),
            )

            metrics.quality_score = quality_score["overall_score"]

            if quality_score["overall_score"] < 70:
                logger.warning(
                    f"Video {video_id} quality score below threshold: {quality_score['overall_score']}"
                )

            metrics.phase_durations["quality_check"] = time.time() - phase_start

            # Phase 8: Publishing
            await self._update_phase(video_id, GenerationPhase.PUBLISHING, websocket)
            phase_start = time.time()

            # Upload to YouTube (simulated for now)
            publish_result = await self._publish_to_youtube(
                video_file_path,
                script_data,
                thumbnail_data["file_path"],
                youtube_account,
                channel_id,
            )

            metrics.phase_durations["publishing"] = time.time() - phase_start

            # Complete
            await self._update_phase(video_id, GenerationPhase.COMPLETED, websocket)

            # Calculate final metrics
            metrics.end_time = datetime.utcnow()
            metrics.total_duration_seconds = (
                metrics.end_time - metrics.start_time
            ).total_seconds()
            metrics.total_cost = sum(metrics.cost_breakdown.values())

            # Update video record
            await self._update_video_record(video, metrics, db)

            # Report success to account manager
            await youtube_account_manager.report_success(
                youtube_account.id, quota_used=1600  # Upload quota
            )

            # Send final metrics via WebSocket
            if websocket:
                await websocket.send_json(
                    {
                        "type": "generation_complete",
                        "video_id": video_id,
                        "metrics": {
                            "duration_seconds": metrics.total_duration_seconds,
                            "total_cost": str(metrics.total_cost),
                            "quality_score": metrics.quality_score,
                            "success": True,
                        },
                    }
                )

            logger.info(
                f"Video {video_id} generated successfully in {metrics.total_duration_seconds:.1f}s "
                f"for ${metrics.total_cost:.2f} (quality: {metrics.quality_score:.1f})"
            )

            return {
                "success": True,
                "video_id": video_id,
                "youtube_id": publish_result.get("video_id"),
                "metrics": asdict(metrics),
                "warnings": metrics.errors if metrics.errors else None,
            }

        except Exception as e:
            logger.error(f"Video generation failed for {video_id}: {str(e)}")
            metrics.errors.append(str(e))
            metrics.end_time = datetime.utcnow()

            if websocket:
                await websocket.send_json(
                    {"type": "generation_failed", "video_id": video_id, "error": str(e)}
                )

            return {
                "success": False,
                "video_id": video_id,
                "error": str(e),
                "metrics": asdict(metrics),
            }

    async def _update_phase(
        self, video_id: str, phase: GenerationPhase, websocket: Optional[WebSocket]
    ):
        """Update generation phase and notify via WebSocket"""
        self.active_generations[video_id] = phase

        if websocket:
            progress = self._calculate_progress(phase)
            await websocket.send_json(
                {
                    "type": "phase_update",
                    "video_id": video_id,
                    "phase": phase.value,
                    "progress": progress,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

    def _calculate_progress(self, phase: GenerationPhase) -> int:
        """Calculate progress percentage based on phase"""
        progress_map = {
            GenerationPhase.INITIALIZATION: 5,
            GenerationPhase.TREND_ANALYSIS: 15,
            GenerationPhase.SCRIPT_GENERATION: 30,
            GenerationPhase.VOICE_SYNTHESIS: 45,
            GenerationPhase.VISUAL_GENERATION: 60,
            GenerationPhase.VIDEO_ASSEMBLY: 75,
            GenerationPhase.QUALITY_CHECK: 85,
            GenerationPhase.PUBLISHING: 95,
            GenerationPhase.COMPLETED: 100,
            GenerationPhase.FAILED: -1,
        }
        return progress_map.get(phase, 0)

    async def _create_video_record(
        self, video_id: str, channel_id: str, db: AsyncSession
    ) -> Video:
        """Create initial video record in database"""
        video = Video(
            id=video_id,
            channel_id=channel_id,
            status=VideoStatus.PROCESSING,
            created_at=datetime.utcnow(),
        )
        db.add(video)
        await db.commit()
        return video

    async def _update_video_record(
        self, video: Video, metrics: GenerationMetrics, db: AsyncSession
    ):
        """Update video record with final metrics"""
        video.status = (
            VideoStatus.PUBLISHED if not metrics.errors else VideoStatus.FAILED
        )
        video.generation_time_seconds = metrics.total_duration_seconds
        video.total_cost = float(metrics.total_cost)
        video.quality_score = metrics.quality_score
        video.metadata = asdict(metrics)
        video.published_at = (
            datetime.utcnow() if video.status == VideoStatus.PUBLISHED else None
        )

        await db.commit()

    async def _publish_to_youtube(
        self,
        video_path: str,
        script_data: Dict,
        thumbnail_path: str,
        youtube_account: Any,
        channel_id: str,
    ) -> Dict[str, Any]:
        """Publish video to YouTube (placeholder for actual implementation)"""
        # In production, this would use the YouTube API to upload
        # For now, return simulated success
        return {
            "video_id": f"yt_{int(time.time())}",
            "url": f"https://youtube.com/watch?v=demo_{channel_id}",
            "published_at": datetime.utcnow().isoformat(),
        }

    async def batch_generate(
        self, channel_ids: List[str], max_concurrent: int = 3, db: AsyncSession = None
    ) -> List[Dict[str, Any]]:
        """Generate multiple videos concurrently with rate limiting"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_with_limit(channel_id):
            async with semaphore:
                return await self.generate_video(channel_id, db=db)

        tasks = [generate_with_limit(channel_id) for channel_id in channel_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [
            result
            if not isinstance(result, Exception)
            else {"success": False, "error": str(result), "channel_id": channel_ids[i]}
            for i, result in enumerate(results)
        ]

    def get_active_generations(self) -> Dict[str, str]:
        """Get status of all active video generations"""
        return {
            video_id: phase.value for video_id, phase in self.active_generations.items()
        }


# Global instance
video_orchestrator = VideoGenerationOrchestrator()
