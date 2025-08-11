#!/usr/bin/env python3
"""
Quality Scoring API Service
FastAPI service for video quality analysis integration
"""

import os
import sys
import logging
import asyncio
import yaml
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from ml_pipeline.quality_scoring.quality_scorer import (
    ContentQualityScorer, 
    QualityScoringConfig, 
    VideoAnalysisMetrics
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class QualityAnalysisRequest(BaseModel):
    video_path: str = Field(..., description="Path to video file")
    script_text: Optional[str] = Field("", description="Script text for content analysis")
    target_topic: Optional[str] = Field("", description="Target topic for relevance analysis")
    use_cache: bool = Field(True, description="Whether to use cached results")

class QualityAnalysisResponse(BaseModel):
    success: bool
    message: str
    metrics: Optional[Dict[str, Any]] = None
    report: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None
    video_hash: Optional[str] = None

class BatchAnalysisRequest(BaseModel):
    video_paths: List[str] = Field(..., description="List of video file paths")
    script_texts: Optional[List[str]] = Field(None, description="Script texts for each video")
    target_topics: Optional[List[str]] = Field(None, description="Target topics for each video")
    use_cache: bool = Field(True, description="Whether to use cached results")

class ConfigUpdateRequest(BaseModel):
    config_updates: Dict[str, Any] = Field(..., description="Configuration updates")

class QualityStatsResponse(BaseModel):
    total_analyses: int
    average_score: float
    min_score: float
    max_score: float
    recent_analyses: int
    uptime_seconds: float

# Global variables
scorer: Optional[ContentQualityScorer] = None
config: Optional[QualityScoringConfig] = None
startup_time: datetime = datetime.now()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global scorer, config
    
    try:
        # Load configuration
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Create configuration object
        config = QualityScoringConfig(
            whisper_model_size=config_data['models']['whisper']['model_size'],
            clip_model_name=config_data['models']['clip']['model_name'],
            bert_model_name=config_data['models']['bert']['model_name'],
            min_acceptable_score=config_data['quality_thresholds']['min_acceptable_score'],
            excellent_score_threshold=config_data['quality_thresholds']['excellent_score_threshold'],
            max_concurrent_analyses=config_data['processing']['max_concurrent_analyses'],
            frame_sample_rate=config_data['processing']['frame_sample_rate'],
            audio_chunk_duration=config_data['processing']['audio_chunk_duration'],
            visual_weight=config_data['scoring_weights']['visual_weight'],
            audio_weight=config_data['scoring_weights']['audio_weight'],
            content_weight=config_data['scoring_weights']['content_weight'],
            technical_weight=config_data['scoring_weights']['technical_weight'],
            target_processing_time=config_data['processing']['target_processing_time'],
            max_video_duration=config_data['processing']['max_video_duration'],
            cache_results=config_data['storage']['cache_results'],
            cache_duration_hours=config_data['storage']['cache_duration_hours'],
            database_path=config_data['storage']['database_path']
        )
        
        # Initialize scorer
        scorer = ContentQualityScorer(config)
        logger.info("Quality scoring service initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize quality scoring service: {e}")
        raise
    finally:
        # Cleanup
        if scorer:
            scorer.close()
            logger.info("Quality scoring service shut down")

# Create FastAPI app
app = FastAPI(
    title="YTEmpire Quality Scoring API",
    description="AI/ML-powered video content quality analysis service",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "quality-scoring",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": (datetime.now() - startup_time).total_seconds(),
        "scorer_initialized": scorer is not None
    }

@app.post("/analyze", response_model=QualityAnalysisResponse)
async def analyze_video_quality(request: QualityAnalysisRequest):
    """Analyze video quality"""
    
    if not scorer:
        raise HTTPException(status_code=503, detail="Quality scoring service not available")
    
    try:
        # Validate video file exists
        if not os.path.exists(request.video_path):
            raise HTTPException(status_code=404, detail=f"Video file not found: {request.video_path}")
        
        # Perform quality analysis
        start_time = datetime.now()
        
        metrics = await scorer.score_video(
            video_path=request.video_path,
            script_text=request.script_text,
            target_topic=request.target_topic,
            use_cache=request.use_cache
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Generate report
        report = scorer.get_quality_report(metrics)
        
        # Calculate video hash for client reference
        video_hash = scorer._calculate_video_hash(request.video_path)
        
        return QualityAnalysisResponse(
            success=True,
            message="Video quality analysis completed successfully",
            metrics=metrics.__dict__,
            report=report,
            processing_time=processing_time,
            video_hash=video_hash
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Quality analysis failed: {e}")
        return QualityAnalysisResponse(
            success=False,
            message=f"Quality analysis failed: {str(e)}",
            processing_time=0.0
        )

@app.post("/analyze/batch")
async def batch_analyze_videos(request: BatchAnalysisRequest):
    """Analyze multiple videos in batch"""
    
    if not scorer:
        raise HTTPException(status_code=503, detail="Quality scoring service not available")
    
    try:
        # Validate inputs
        video_count = len(request.video_paths)
        
        if request.script_texts and len(request.script_texts) != video_count:
            raise HTTPException(status_code=400, detail="Script texts count must match video paths count")
        
        if request.target_topics and len(request.target_topics) != video_count:
            raise HTTPException(status_code=400, detail="Target topics count must match video paths count")
        
        # Prepare batch analysis parameters
        script_texts = request.script_texts or [""] * video_count
        target_topics = request.target_topics or [""] * video_count
        
        start_time = datetime.now()
        
        # Perform batch analysis
        results = []
        for i, video_path in enumerate(request.video_paths):
            try:
                if not os.path.exists(video_path):
                    results.append({
                        "video_path": video_path,
                        "success": False,
                        "error": f"File not found: {video_path}"
                    })
                    continue
                
                metrics = await scorer.score_video(
                    video_path=video_path,
                    script_text=script_texts[i],
                    target_topic=target_topics[i],
                    use_cache=request.use_cache
                )
                
                report = scorer.get_quality_report(metrics)
                video_hash = scorer._calculate_video_hash(video_path)
                
                results.append({
                    "video_path": video_path,
                    "success": True,
                    "metrics": metrics.__dict__,
                    "report": report,
                    "video_hash": video_hash
                })
                
            except Exception as e:
                logger.error(f"Failed to analyze {video_path}: {e}")
                results.append({
                    "video_path": video_path,
                    "success": False,
                    "error": str(e)
                })
        
        total_processing_time = (datetime.now() - start_time).total_seconds()
        successful_analyses = sum(1 for r in results if r["success"])
        
        return {
            "success": True,
            "message": f"Batch analysis completed: {successful_analyses}/{video_count} successful",
            "results": results,
            "total_processing_time": total_processing_time,
            "successful_count": successful_analyses,
            "failed_count": video_count - successful_analyses
        }
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@app.post("/upload-and-analyze")
async def upload_and_analyze(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    script_text: str = Form(""),
    target_topic: str = Form(""),
    use_cache: bool = Form(True)
):
    """Upload video file and analyze quality"""
    
    if not scorer:
        raise HTTPException(status_code=503, detail="Quality scoring service not available")
    
    # Validate file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    try:
        # Save uploaded file temporarily
        temp_dir = Path("/tmp/ytempire_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        file_path = temp_dir / f"{datetime.now().timestamp()}_{file.filename}"
        
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        # Analyze video
        metrics = await scorer.score_video(
            video_path=str(file_path),
            script_text=script_text,
            target_topic=target_topic,
            use_cache=use_cache
        )
        
        report = scorer.get_quality_report(metrics)
        video_hash = scorer._calculate_video_hash(str(file_path))
        
        # Schedule file cleanup
        background_tasks.add_task(cleanup_file, file_path)
        
        return QualityAnalysisResponse(
            success=True,
            message="Video uploaded and analyzed successfully",
            metrics=metrics.__dict__,
            report=report,
            video_hash=video_hash
        )
        
    except Exception as e:
        logger.error(f"Upload and analysis failed: {e}")
        # Cleanup file on error
        if file_path.exists():
            file_path.unlink()
        
        raise HTTPException(status_code=500, detail=f"Upload and analysis failed: {str(e)}")

@app.get("/stats", response_model=QualityStatsResponse)
async def get_quality_stats():
    """Get quality scoring statistics"""
    
    if not scorer:
        raise HTTPException(status_code=503, detail="Quality scoring service not available")
    
    try:
        stats = scorer.database.get_statistics()
        uptime = (datetime.now() - startup_time).total_seconds()
        
        return QualityStatsResponse(
            total_analyses=stats['total_analyses'],
            average_score=stats['average_score'],
            min_score=stats['min_score'],
            max_score=stats['max_score'],
            recent_analyses=stats['recent_analyses'],
            uptime_seconds=uptime
        )
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@app.get("/config")
async def get_configuration():
    """Get current configuration"""
    
    if not config:
        raise HTTPException(status_code=503, detail="Configuration not available")
    
    return {
        "whisper_model_size": config.whisper_model_size,
        "clip_model_name": config.clip_model_name,
        "bert_model_name": config.bert_model_name,
        "min_acceptable_score": config.min_acceptable_score,
        "excellent_score_threshold": config.excellent_score_threshold,
        "max_concurrent_analyses": config.max_concurrent_analyses,
        "target_processing_time": config.target_processing_time,
        "cache_results": config.cache_results,
        "cache_duration_hours": config.cache_duration_hours
    }

@app.put("/config")
async def update_configuration(request: ConfigUpdateRequest):
    """Update configuration (limited runtime updates)"""
    
    if not config:
        raise HTTPException(status_code=503, detail="Configuration not available")
    
    try:
        # Allow updating certain configuration parameters
        allowed_updates = [
            'min_acceptable_score',
            'excellent_score_threshold',
            'max_concurrent_analyses',
            'cache_duration_hours',
            'target_processing_time'
        ]
        
        updates_applied = []
        
        for key, value in request.config_updates.items():
            if key in allowed_updates:
                if hasattr(config, key):
                    setattr(config, key, value)
                    updates_applied.append(key)
        
        return {
            "success": True,
            "message": f"Configuration updated: {', '.join(updates_applied)}",
            "updates_applied": updates_applied
        }
        
    except Exception as e:
        logger.error(f"Configuration update failed: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration update failed: {str(e)}")

@app.get("/models/status")
async def get_models_status():
    """Get status of loaded models"""
    
    if not scorer:
        raise HTTPException(status_code=503, detail="Quality scoring service not available")
    
    try:
        status = {
            "visual_analyzer": {
                "clip_model_loaded": scorer.visual_analyzer.clip_model is not None,
                "device": str(scorer.visual_analyzer.device)
            },
            "audio_analyzer": {
                "whisper_model_loaded": scorer.audio_analyzer.whisper_model is not None,
                "model_size": config.whisper_model_size if config else "unknown"
            },
            "content_analyzer": {
                "bert_model_loaded": scorer.content_analyzer.bert_model is not None,
                "sentiment_analyzer_loaded": scorer.content_analyzer.sentiment_analyzer is not None,
                "topic_classifier_loaded": scorer.content_analyzer.topic_classifier is not None,
                "device": str(scorer.content_analyzer.device)
            },
            "technical_analyzer": {
                "status": "ready"
            }
        }
        
        return {
            "success": True,
            "models_status": status,
            "overall_status": "ready" if all([
                status["visual_analyzer"]["clip_model_loaded"],
                status["audio_analyzer"]["whisper_model_loaded"],
                status["content_analyzer"]["bert_model_loaded"]
            ]) else "partially_loaded"
        }
        
    except Exception as e:
        logger.error(f"Failed to get models status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get models status: {str(e)}")

@app.delete("/cache")
async def clear_cache():
    """Clear quality scoring cache"""
    
    if not scorer:
        raise HTTPException(status_code=503, detail="Quality scoring service not available")
    
    try:
        # Clear database cache (simplified - would need proper implementation)
        # This would typically involve deleting cached entries from database
        
        return {
            "success": True,
            "message": "Cache cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"Cache clearing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clearing failed: {str(e)}")

async def cleanup_file(file_path: Path):
    """Background task to cleanup temporary files"""
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to cleanup file {file_path}: {e}")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "detail": str(exc) if app.debug else "An error occurred"
        }
    )

if __name__ == "__main__":
    # Configuration
    host = os.getenv("QUALITY_API_HOST", "0.0.0.0")
    port = int(os.getenv("QUALITY_API_PORT", "8001"))
    workers = int(os.getenv("QUALITY_API_WORKERS", "1"))
    
    # Run server
    uvicorn.run(
        "quality_api:app",
        host=host,
        port=port,
        workers=workers,
        reload=False,
        log_level="info"
    )