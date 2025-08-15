"""
Video Generation Pipeline
Orchestrates the complete video generation process from script to upload
"""
import os
import asyncio
import tempfile
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import aiofiles
from dataclasses import dataclass, asdict

from app.services.ai_services import OpenAIService, ElevenLabsService, GoogleTTSService, AIServiceConfig
from app.services.youtube_service import YouTubeService, YouTubeConfig
from app.services.video_processor import VideoProcessor
from app.services.thumbnail_generator import ThumbnailGenerator
from app.services.cost_tracking import cost_tracker
from app.db.session import AsyncSessionLocal
from app.models.video import Video, VideoStatus
from app.models.cost import Cost
from sqlalchemy import select

logger = logging.getLogger(__name__)

@dataclass
class VideoGenerationConfig:
    """Configuration for video generation pipeline"""
    channel_id: str
    user_id: str
    topic: str
    style: str = "educational"
    length: str = "medium"  # short (1-3min), medium (3-7min), long (7-15min)
    target_audience: str = "general"
    keywords: List[str] = None
    voice_provider: str = "elevenlabs"  # elevenlabs or google_tts
    voice_id: Optional[str] = None
    thumbnail_style: str = "modern"
    auto_publish: bool = False
    optimize_for_cost: bool = True
    max_cost: float = 3.00  # Maximum cost per video
    quality_threshold: float = 0.7  # Minimum quality score (0-1)


class VideoGenerationPipeline:
    """Main video generation orchestrator"""
    
    def __init__(self):
        # Initialize AI services
        ai_config = AIServiceConfig(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY", "")
        )
        
        self.openai_service = OpenAIService(ai_config) if ai_config.openai_api_key else None
        self.elevenlabs_service = ElevenLabsService(ai_config) if ai_config.elevenlabs_api_key else None
        self.google_tts_service = GoogleTTSService(ai_config)
        
        # Initialize other services
        self.youtube_service = YouTubeService()
        self.video_processor = VideoProcessor()
        self.thumbnail_generator = ThumbnailGenerator()
        
        # Temporary directory for processing
        self.temp_dir = Path(tempfile.gettempdir()) / "ytempire_pipeline"
        self.temp_dir.mkdir(exist_ok=True)
        
        self.total_cost = 0.0
        
    async def generate_video(self, config: VideoGenerationConfig) -> Dict[str, Any]:
        """
        Main pipeline execution
        Returns video metadata and generation results
        """
        logger.info(f"Starting video generation for topic: {config.topic}")
        
        # Create video record in database
        video_id = await self._create_video_record(config)
        
        try:
            # Track pipeline progress
            pipeline_result = {
                "video_id": video_id,
                "status": "in_progress",
                "stages": {},
                "total_cost": 0.0,
                "errors": []
            }
            
            # Stage 1: Generate Script
            await self._update_video_status(video_id, "generating_script")
            script_result = await self._generate_script(config)
            pipeline_result["stages"]["script"] = script_result
            pipeline_result["total_cost"] += script_result.get("cost", 0)
            
            # Check cost limit
            if pipeline_result["total_cost"] >= config.max_cost:
                raise Exception(f"Cost limit exceeded: ${pipeline_result['total_cost']:.2f}")
            
            # Stage 2: Generate Title and Description
            await self._update_video_status(video_id, "generating_metadata")
            metadata_result = await self._generate_metadata(
                config.topic,
                script_result["script"],
                config.keywords
            )
            pipeline_result["stages"]["metadata"] = metadata_result
            pipeline_result["total_cost"] += metadata_result.get("cost", 0)
            
            # Stage 3: Generate Voice
            await self._update_video_status(video_id, "generating_voice")
            voice_result = await self._generate_voice(
                script_result["script"]["narration"],
                config
            )
            pipeline_result["stages"]["voice"] = voice_result
            pipeline_result["total_cost"] += voice_result.get("cost", 0)
            
            # Stage 4: Generate Thumbnail
            await self._update_video_status(video_id, "generating_thumbnail")
            thumbnail_result = await self._generate_thumbnail(
                metadata_result["title"],
                config.topic,
                config.thumbnail_style
            )
            pipeline_result["stages"]["thumbnail"] = thumbnail_result
            pipeline_result["total_cost"] += thumbnail_result.get("cost", 0)
            
            # Stage 5: Generate Background Video/Images
            await self._update_video_status(video_id, "generating_visuals")
            visuals_result = await self._generate_visuals(
                script_result["script"],
                config
            )
            pipeline_result["stages"]["visuals"] = visuals_result
            pipeline_result["total_cost"] += visuals_result.get("cost", 0)
            
            # Stage 6: Assemble Video
            await self._update_video_status(video_id, "assembling_video")
            video_result = await self._assemble_video(
                voice_result["audio_path"],
                visuals_result["visuals"],
                script_result["script"],
                metadata_result
            )
            pipeline_result["stages"]["assembly"] = video_result
            
            # Stage 7: Upload to YouTube (if auto_publish)
            if config.auto_publish:
                await self._update_video_status(video_id, "uploading")
                upload_result = await self._upload_to_youtube(
                    video_result["video_path"],
                    thumbnail_result["thumbnail_path"],
                    metadata_result,
                    config.channel_id
                )
                pipeline_result["stages"]["upload"] = upload_result
                
            # Update final video record
            await self._finalize_video_record(
                video_id,
                pipeline_result,
                metadata_result,
                video_result["video_path"]
            )
            
            pipeline_result["status"] = "completed"
            logger.info(f"Video generation completed. Total cost: ${pipeline_result['total_cost']:.2f}")
            
            # Track total cost
            await self._track_cost(config.user_id, video_id, pipeline_result["total_cost"])
            
            return pipeline_result
            
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            await self._update_video_status(video_id, "failed", str(e))
            raise
            
    async def _generate_script(self, config: VideoGenerationConfig) -> Dict[str, Any]:
        """Generate video script using AI"""
        if not self.openai_service:
            # Return mock data for testing
            return {
                "script": {
                    "title": f"Amazing Facts About {config.topic}",
                    "hook": f"Did you know these incredible things about {config.topic}?",
                    "introduction": f"Welcome to our channel! Today we're exploring {config.topic}.",
                    "main_points": [
                        {"point": "First amazing fact", "explanation": "Detailed explanation here"},
                        {"point": "Second amazing fact", "explanation": "More details here"},
                        {"point": "Third amazing fact", "explanation": "Even more details"}
                    ],
                    "conclusion": "Thanks for watching! Don't forget to like and subscribe!",
                    "narration": f"Welcome to our channel! Today we're exploring {config.topic}. " * 10
                },
                "cost": 0.02,
                "tokens_used": 500
            }
            
        # Real implementation
        result = await self.openai_service.generate_script(
            topic=config.topic,
            style=config.style,
            length=config.length,
            target_audience=config.target_audience,
            keywords=config.keywords
        )
        
        return result
        
    async def _generate_metadata(self, topic: str, script: Dict, keywords: List[str]) -> Dict[str, Any]:
        """Generate title, description, and tags"""
        if not self.openai_service:
            # Return mock data for testing
            return {
                "title": f"Amazing Facts About {topic} | You Won't Believe #3!",
                "description": f"Discover incredible facts about {topic}. In this video, we explore...",
                "tags": keywords or ["education", "facts", "amazing", topic.lower()],
                "cost": 0.01
            }
            
        # Generate title
        title_result = await self.openai_service.generate_title(topic, keywords)
        
        # Generate description
        desc_result = await self.openai_service.generate_description(
            title_result["titles"][0],
            json.dumps(script),
            keywords
        )
        
        # Generate tags (not implemented yet in OpenAIService)
        # tags_result = await self.openai_service.generate_tags(topic, keywords)
        
        return {
            "title": title_result["titles"][0],
            "description": desc_result["description"],
            "tags": keywords or ["education", "tutorial"],  # Use keywords or defaults
            "cost": title_result["cost"] + desc_result["cost"]  # + tags_result["cost"]
        }
        
    async def _generate_voice(self, narration_text: str, config: VideoGenerationConfig) -> Dict[str, Any]:
        """Generate voice narration"""
        audio_path = self.temp_dir / f"narration_{datetime.now().timestamp()}.mp3"
        
        try:
            if config.voice_provider == "elevenlabs" and self.elevenlabs_service:
                result = await self.elevenlabs_service.text_to_speech(
                    text=narration_text,
                    voice_id=config.voice_id,
                    output_path=str(audio_path)
                )
            else:
                # Use Google TTS as fallback or default
                result = await self.google_tts_service.text_to_speech(
                    text=narration_text,
                    output_path=str(audio_path)
                )
        except Exception as e:
            logger.warning(f"Voice synthesis failed: {e}, using mock audio")
            # Use mock audio generator as fallback
            try:
                from app.services.mock_video_generator import mock_generator
                result = await mock_generator.create_mock_audio(
                    text=narration_text,
                    output_path=str(audio_path)
                )
            except:
                # Create a simple placeholder audio file
                with open(audio_path, 'wb') as f:
                    f.write(b'MOCK_AUDIO_FILE')
                result = {
                    "audio_path": str(audio_path),
                    "duration": 60,
                    "cost": 0,
                    "is_mock": True
                }
            
        return {
            "audio_path": str(audio_path),
            "duration": result.get("duration", 60),
            "cost": result.get("cost", 0.05)
        }
        
    async def _generate_thumbnail(self, title: str, topic: str, style: str) -> Dict[str, Any]:
        """Generate video thumbnail"""
        thumbnail_path = await self.thumbnail_generator.generate(
            title=title,
            topic=topic,
            style=style
        )
        
        return {
            "thumbnail_path": thumbnail_path,
            "cost": 0.02  # DALL-E cost estimate
        }
        
    async def _generate_visuals(self, script: Dict, config: VideoGenerationConfig) -> Dict[str, Any]:
        """Generate background visuals for the video"""
        try:
            # Import stock footage service
            from app.services.stock_footage import stock_footage_service
            
            # Determine content type and footage ratio
            footage_ratio = stock_footage_service.get_footage_ratio(config.style)
            
            # Prepare script segments for visual generation
            script_segments = []
            
            # Add introduction segment
            if script.get("introduction"):
                script_segments.append({
                    "id": "intro",
                    "text": script["introduction"],
                    "duration": 5,
                    "point": "Introduction"
                })
            
            # Add main points as segments
            for i, point in enumerate(script.get("main_points", [])):
                script_segments.append({
                    "id": f"point_{i}",
                    "text": point.get("explanation", point.get("point", "")),
                    "duration": 10,
                    "point": point.get("point", "")
                })
            
            # Add conclusion segment
            if script.get("conclusion"):
                script_segments.append({
                    "id": "conclusion",
                    "text": script["conclusion"],
                    "duration": 5,
                    "point": "Conclusion"
                })
            
            # Generate visual sequence using stock footage service
            visual_timeline = await stock_footage_service.generate_visual_sequence(
                script_segments,
                content_type=config.style
            )
            
            # Convert to flat list of visuals for video processor
            visuals = []
            for segment in visual_timeline:
                for clip in segment.get("clips", []):
                    visual = {
                        "type": clip.get("type", "image"),
                        "path": clip.get("path", clip.get("url", f"stock_{len(visuals)}.jpg")),
                        "duration": clip.get("duration", 5),
                        "text_overlay": segment.get("text_overlay", ""),
                        "effect": clip.get("effect", "none"),
                        "effect_params": clip.get("effect_params", {})
                    }
                    visuals.append(visual)
            
            # If no visuals generated, create defaults
            if not visuals:
                for i, point in enumerate(script.get("main_points", [])):
                    visual = {
                        "type": "image",
                        "path": f"stock_image_{i}.jpg",
                        "duration": 10,
                        "text_overlay": point.get("point", "")
                    }
                    visuals.append(visual)
            
            logger.info(f"Generated {len(visuals)} visuals using {footage_ratio['method']} method")
            
            return {
                "visuals": visuals,
                "cost": 0.0,  # Using free stock footage APIs
                "method": footage_ratio['method'],
                "footage_ratio": footage_ratio
            }
            
        except Exception as e:
            logger.warning(f"Stock footage generation failed: {e}, using fallback")
            # Fallback to simple visuals
            visuals = []
            for i, point in enumerate(script.get("main_points", [])):
                visual = {
                    "type": "image",
                    "path": f"stock_image_{i}.jpg",
                    "duration": 10,
                    "text_overlay": point.get("point", "")
                }
                visuals.append(visual)
            
            return {
                "visuals": visuals,
                "cost": 0.0
            }
        
    async def _assemble_video(
        self,
        audio_path: str,
        visuals: List[Dict],
        script: Dict,
        metadata: Dict
    ) -> Dict[str, Any]:
        """Assemble final video from components"""
        output_path = self.temp_dir / f"video_{datetime.now().timestamp()}.mp4"
        
        # Use video processor to create video
        result = await self.video_processor.create_video_with_audio(
            audio_path=audio_path,
            visuals=visuals,
            output_path=str(output_path),
            title=metadata["title"],
            subtitles=script.get("narration", "")
        )
        
        return {
            "video_path": str(output_path),
            "duration": result.get("duration", 60),
            "file_size": result.get("file_size", 0)
        }
        
    async def _upload_to_youtube(
        self,
        video_path: str,
        thumbnail_path: str,
        metadata: Dict,
        channel_id: str
    ) -> Dict[str, Any]:
        """Upload video to YouTube"""
        # Initialize YouTube service with OAuth
        self.youtube_service.authenticate_oauth()
        
        # Upload video
        result = await self.youtube_service.upload_video(
            file_path=video_path,
            title=metadata["title"],
            description=metadata["description"],
            tags=metadata["tags"],
            category_id="22",  # People & Blogs
            privacy_status="private",  # Start as private for review
            thumbnail_path=thumbnail_path
        )
        
        return {
            "youtube_video_id": result["id"],
            "youtube_url": f"https://youtube.com/watch?v={result['id']}",
            "status": result["status"]
        }
        
    async def _create_video_record(self, config: VideoGenerationConfig) -> str:
        """Create initial video record in database"""
        async with AsyncSessionLocal() as session:
            video = Video(
                channel_id=config.channel_id,
                title=f"Processing: {config.topic}",
                description="Video generation in progress...",
                generation_status=VideoStatus.PROCESSING
            )
            session.add(video)
            await session.commit()
            return video.id
            
    async def _update_video_status(self, video_id: str, status: str, error: str = None):
        """Update video status in database"""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Video).where(Video.id == video_id)
            )
            video = result.scalar_one_or_none()
            if video:
                video.generation_status = status
                # Note: Video model doesn't have error_message field
                video.updated_at = datetime.utcnow()
                await session.commit()
                
    async def _finalize_video_record(
        self,
        video_id: str,
        pipeline_result: Dict,
        metadata: Dict,
        video_path: str
    ):
        """Update final video record with all metadata"""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Video).where(Video.id == video_id)
            )
            video = result.scalar_one_or_none()
            if video:
                video.title = metadata["title"]
                video.description = metadata["description"]
                video.tags = metadata["tags"]
                video.generation_status = VideoStatus.COMPLETED
                video.video_file_path = video_path
                video.total_cost = pipeline_result["total_cost"]
                video.youtube_video_id = pipeline_result.get("stages", {}).get("upload", {}).get("youtube_video_id")
                video.published_at = datetime.utcnow() if video.youtube_video_id else None
                await session.commit()
                
    async def _track_cost(self, user_id: str, video_id: str, total_cost: float):
        """Track cost in database"""
        async with AsyncSessionLocal() as session:
            cost_record = Cost(
                user_id=user_id,
                video_id=video_id,
                service_type="video_generation",
                operation="complete_pipeline",
                amount=total_cost,
                units=1,
                unit_cost=total_cost
            )
            session.add(cost_record)
            await session.commit()
            
        # Also track with cost tracker service
        cost_tracker.track_cost(
            service="video_generation",
            operation="complete_pipeline",
            amount=total_cost,
            metadata={"video_id": video_id, "user_id": user_id}
        )