"""
Enhanced Video Generation Service with ML Integration
Integrates AutoML and Personalization for improved video generation
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from decimal import Decimal
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import WebSocket

from app.services.video_generation_orchestrator import (
    VideoGenerationOrchestrator,
    GenerationPhase,
    GenerationMetrics
)
from app.services.ml_integration_service import (
    ml_service,
    get_personalized_video_content,
    predict_video_performance,
    update_channel_ml_profile
)
from app.models.video import Video, VideoStatus
from app.models.channel import Channel
from app.services.cost_tracking import cost_tracker

logger = logging.getLogger(__name__)


class EnhancedVideoGenerationOrchestrator(VideoGenerationOrchestrator):
    """
    Enhanced video generation with ML-powered personalization and performance prediction
    """
    
    async def generate_video_with_ml(
        self,
        channel_id: str,
        topic: Optional[str] = None,
        db: AsyncSession = None,
        websocket: Optional[WebSocket] = None,
        use_personalization: bool = True,
        use_performance_prediction: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a video with ML-powered enhancements
        """
        video_id = f"video_{channel_id}_{int(time.time())}"
        metrics = GenerationMetrics(
            video_id=video_id,
            channel_id=channel_id,
            start_time=datetime.utcnow()
        )
        
        try:
            # Get channel information
            channel = await self._get_channel_info(channel_id, db)
            
            # Phase 1: ML-Powered Content Planning
            await self._update_phase(video_id, GenerationPhase.INITIALIZATION, websocket)
            phase_start = time.time()
            
            personalized_content = None
            if use_personalization:
                # Get historical videos for personalization
                historical_videos = await self._get_historical_videos(channel_id, db)
                
                # Get personalized content recommendation
                personalized_content = await get_personalized_video_content(
                    channel_id=channel_id,
                    channel_data={
                        'name': channel.name if channel else f'Channel_{channel_id}',
                        'description': channel.description if channel else '',
                        'niche': channel.niche if channel else 'general',
                        'target_audience': channel.metadata.get('target_audience', {}) if channel else {}
                    },
                    historical_videos=historical_videos,
                    trending_topics=[topic] if topic else None,
                    db=db
                )
                
                logger.info(f"Generated personalized content for channel {channel_id}")
                logger.info(f"Title: {personalized_content['title']}")
                logger.info(f"Confidence: {personalized_content['confidence_score']:.2%}")
            
            metrics.phase_durations["personalization"] = time.time() - phase_start
            
            # Phase 2: Performance Prediction
            predicted_performance = None
            if use_performance_prediction and personalized_content:
                phase_start = time.time()
                
                # Prepare features for prediction
                video_features = {
                    'title_length': len(personalized_content['title'].split()),
                    'description_length': 100,  # Default
                    'keyword_count': len(personalized_content['keywords']),
                    'trending_score': 0.7 if topic else 0.3,
                    'channel_subscriber_count': channel.subscriber_count if channel else 1000,
                    'channel_video_count': len(historical_videos),
                    'posting_hour': datetime.now().hour,
                    'posting_day': datetime.now().weekday(),
                    'video_duration': 600  # 10 minutes default
                }
                
                predicted_performance = await predict_video_performance(video_features, db)
                
                logger.info(f"Predicted performance for video:")
                logger.info(f"Views: {predicted_performance['predicted_views']:,}")
                logger.info(f"Engagement: {predicted_performance['predicted_engagement_rate']:.2%}")
                
                metrics.phase_durations["performance_prediction"] = time.time() - phase_start
            
            # Phase 3: Content Generation with Personalization
            await self._update_phase(video_id, GenerationPhase.SCRIPT_GENERATION, websocket)
            phase_start = time.time()
            
            # Use personalized content or fallback to standard generation
            if personalized_content:
                script_data = {
                    "title": personalized_content['title'],
                    "script": personalized_content['script_template'],
                    "keywords": personalized_content['keywords'],
                    "tone": personalized_content['tone'],
                    "style": personalized_content['style']
                }
            else:
                # Fallback to original script generation
                script_data = await self.script_generator.generate_script(
                    topic=topic or "Technology Trends",
                    duration_minutes=8,
                    style="engaging_educational"
                )
            
            # Track cost
            script_cost = Decimal("0.10")
            await cost_tracker.track_cost(
                service="openai",
                operation="script_generation",
                cost=script_cost,
                metadata={"video_id": video_id, "personalized": bool(personalized_content)}
            )
            metrics.cost_breakdown["script_generation"] = script_cost
            metrics.phase_durations["script_generation"] = time.time() - phase_start
            
            # Phase 4: Voice Synthesis with Personalized Settings
            await self._update_phase(video_id, GenerationPhase.VOICE_SYNTHESIS, websocket)
            phase_start = time.time()
            
            # Use voice preferences from personalization if available
            voice_settings = {}
            if personalized_content:
                # Could extract voice preferences from channel profile
                voice_settings = {
                    "voice_id": "adam",  # Could be personalized
                    "speed": 1.0,
                    "pitch": "medium"
                }
            
            voice_data = await self.voice_synthesizer.synthesize(
                text=script_data["script"],
                **voice_settings
            )
            
            voice_cost = Decimal("0.20")
            await cost_tracker.track_cost(
                service="elevenlabs",
                operation="voice_synthesis",
                cost=voice_cost,
                metadata={"video_id": video_id}
            )
            metrics.cost_breakdown["voice_synthesis"] = voice_cost
            metrics.phase_durations["voice_synthesis"] = time.time() - phase_start
            
            # Phase 5: Thumbnail Generation
            await self._update_phase(video_id, GenerationPhase.VISUAL_GENERATION, websocket)
            phase_start = time.time()
            
            thumbnail_data = await self.thumbnail_generator.generate(
                title=script_data["title"],
                keywords=script_data.get("keywords", [])
            )
            
            thumbnail_cost = Decimal("0.04")
            await cost_tracker.track_cost(
                service="openai",
                operation="thumbnail_generation",
                cost=thumbnail_cost,
                metadata={"video_id": video_id}
            )
            metrics.cost_breakdown["thumbnail_generation"] = thumbnail_cost
            metrics.phase_durations["visual_generation"] = time.time() - phase_start
            
            # Phase 6: Quality Scoring
            await self._update_phase(video_id, GenerationPhase.QUALITY_CHECK, websocket)
            phase_start = time.time()
            
            quality_score = await self.quality_scorer.score(
                script=script_data["script"],
                title=script_data["title"]
            )
            
            metrics.quality_score = quality_score.get("score", 85.0)
            metrics.phase_durations["quality_check"] = time.time() - phase_start
            
            # Calculate totals
            metrics.end_time = datetime.utcnow()
            metrics.total_duration_seconds = (
                metrics.end_time - metrics.start_time
            ).total_seconds()
            metrics.total_cost = sum(metrics.cost_breakdown.values())
            
            # Update ML profile with initial data
            if use_personalization:
                await update_channel_ml_profile(
                    channel_id=channel_id,
                    video_id=video_id,
                    performance_data={
                        'title': script_data['title'],
                        'keywords': script_data.get('keywords', []),
                        'duration': 600,
                        'quality_score': metrics.quality_score,
                        'generation_cost': float(metrics.total_cost),
                        'predicted_views': predicted_performance['predicted_views'] if predicted_performance else 0,
                        'predicted_engagement': predicted_performance['predicted_engagement_rate'] if predicted_performance else 0
                    },
                    db=db
                )
            
            # Prepare response
            result = {
                "video_id": video_id,
                "status": "completed",
                "title": script_data["title"],
                "metrics": {
                    "duration_seconds": metrics.total_duration_seconds,
                    "total_cost": float(metrics.total_cost),
                    "quality_score": metrics.quality_score,
                    "phase_durations": metrics.phase_durations
                },
                "personalization": {
                    "used": bool(personalized_content),
                    "confidence": personalized_content['confidence_score'] if personalized_content else 0,
                    "style": personalized_content['style'] if personalized_content else "default"
                },
                "performance_prediction": predicted_performance if predicted_performance else None,
                "files": {
                    "script": script_data.get("script_path", ""),
                    "audio": voice_data.get("audio_path", ""),
                    "thumbnail": thumbnail_data.get("thumbnail_path", "")
                }
            }
            
            # Send completion notification
            if websocket:
                await websocket.send_json({
                    "phase": GenerationPhase.COMPLETED.value,
                    "video_id": video_id,
                    "result": result
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in ML-enhanced video generation: {e}")
            metrics.errors.append(str(e))
            
            # Update phase to failed
            await self._update_phase(video_id, GenerationPhase.FAILED, websocket)
            
            # Return error response
            return {
                "video_id": video_id,
                "status": "failed",
                "error": str(e),
                "metrics": {
                    "duration_seconds": time.time() - metrics.start_time.timestamp(),
                    "total_cost": float(metrics.total_cost),
                    "errors": metrics.errors
                }
            }
    
    async def _get_channel_info(
        self,
        channel_id: str,
        db: AsyncSession
    ) -> Optional[Channel]:
        """Get channel information from database"""
        try:
            query = select(Channel).where(Channel.id == channel_id)
            result = await db.execute(query)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error fetching channel info: {e}")
            return None
    
    async def _get_historical_videos(
        self,
        channel_id: str,
        db: AsyncSession,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get historical videos for a channel"""
        try:
            query = select(Video).where(
                Video.channel_id == channel_id,
                Video.status == VideoStatus.PUBLISHED
            ).order_by(Video.created_at.desc()).limit(limit)
            
            result = await db.execute(query)
            videos = result.scalars().all()
            
            return [
                {
                    'id': video.id,
                    'title': video.title,
                    'description': video.description,
                    'views': video.views or 0,
                    'likes': video.likes or 0,
                    'comments': video.comments or 0,
                    'duration': video.duration or 600,
                    'published_at': video.created_at.isoformat() if video.created_at else datetime.now().isoformat(),
                    'engagement_rate': ((video.likes or 0) + (video.comments or 0)) / max(video.views or 1, 1)
                }
                for video in videos
            ]
        except Exception as e:
            logger.error(f"Error fetching historical videos: {e}")
            return []
    
    async def batch_generate_with_ml(
        self,
        channel_ids: List[str],
        topics: Optional[List[str]] = None,
        db: AsyncSession = None,
        use_personalization: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple videos in batch with ML optimization
        """
        results = []
        
        # Sort channels by predicted performance if ML is available
        if use_personalization and ml_service.automl_pipeline:
            # Get channel insights and sort by potential
            channel_scores = []
            for channel_id in channel_ids:
                insights = await ml_service.get_channel_insights(channel_id, db)
                avg_engagement = insights.get('performance', {}).get('avg_engagement', 0)
                channel_scores.append((channel_id, avg_engagement))
            
            # Sort by engagement potential (highest first)
            channel_scores.sort(key=lambda x: x[1], reverse=True)
            channel_ids = [c[0] for c in channel_scores]
            
            logger.info(f"Optimized batch order based on ML predictions")
        
        # Generate videos in optimized order
        for i, channel_id in enumerate(channel_ids):
            topic = topics[i] if topics and i < len(topics) else None
            
            result = await self.generate_video_with_ml(
                channel_id=channel_id,
                topic=topic,
                db=db,
                use_personalization=use_personalization
            )
            
            results.append(result)
            
            # Small delay between generations to avoid overload
            await asyncio.sleep(1)
        
        return results


# Create enhanced orchestrator instance
enhanced_orchestrator = EnhancedVideoGenerationOrchestrator()


# Helper functions for easy integration
async def generate_personalized_video(
    channel_id: str,
    topic: Optional[str] = None,
    db: AsyncSession = None,
    websocket: Optional[WebSocket] = None
) -> Dict[str, Any]:
    """Generate a video with full ML personalization"""
    return await enhanced_orchestrator.generate_video_with_ml(
        channel_id=channel_id,
        topic=topic,
        db=db,
        websocket=websocket,
        use_personalization=True,
        use_performance_prediction=True
    )


async def generate_batch_videos_with_ml(
    channel_ids: List[str],
    topics: Optional[List[str]] = None,
    db: AsyncSession = None
) -> List[Dict[str, Any]]:
    """Generate multiple videos with ML optimization"""
    return await enhanced_orchestrator.batch_generate_with_ml(
        channel_ids=channel_ids,
        topics=topics,
        db=db,
        use_personalization=True
    )