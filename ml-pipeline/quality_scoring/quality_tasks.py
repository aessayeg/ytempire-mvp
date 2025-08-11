#!/usr/bin/env python3
"""
Celery tasks for asynchronous quality scoring
Background processing for video quality analysis
"""

import os
import sys
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

from celery import Celery
from celery.exceptions import Retry
import yaml
import redis
from sqlalchemy import create_engine, Column, String, Float, DateTime, Text, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

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

# Celery configuration
celery_broker_url = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
celery_result_backend = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

# Initialize Celery app
celery_app = Celery(
    'quality_tasks',
    broker=celery_broker_url,
    backend=celery_result_backend,
    include=['quality_tasks']
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
    worker_disable_rate_limits=False,
    task_acks_late=True,
    worker_send_task_events=True,
    task_send_sent_event=True,
    result_expires=3600,  # 1 hour
)

# Database setup for task results
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://quality_user:quality_secure_password_2024@localhost:5433/quality_scores')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class QualityTaskResult(Base):
    """Database model for quality task results"""
    __tablename__ = 'quality_task_results'
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String, unique=True, index=True)
    video_path = Column(String, nullable=False)
    status = Column(String, nullable=False)  # pending, running, completed, failed
    overall_score = Column(Float)
    processing_time = Column(Float)
    error_message = Column(Text)
    metrics_json = Column(Text)
    report_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

# Create tables
Base.metadata.create_all(bind=engine)

# Global scorer instance (initialized per worker)
scorer: Optional[ContentQualityScorer] = None
config: Optional[QualityScoringConfig] = None

def get_scorer():
    """Get or initialize quality scorer"""
    global scorer, config
    
    if scorer is None:
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
            logger.info("Quality scorer initialized in worker")
            
        except Exception as e:
            logger.error(f"Failed to initialize quality scorer: {e}")
            raise
    
    return scorer

def update_task_status(task_id: str, status: str, **kwargs):
    """Update task status in database"""
    try:
        db = SessionLocal()
        task_result = db.query(QualityTaskResult).filter(
            QualityTaskResult.task_id == task_id
        ).first()
        
        if task_result:
            task_result.status = status
            for key, value in kwargs.items():
                if hasattr(task_result, key):
                    setattr(task_result, key, value)
            
            if status in ['completed', 'failed']:
                task_result.completed_at = datetime.utcnow()
            
            db.commit()
        
        db.close()
        
    except Exception as e:
        logger.error(f"Failed to update task status: {e}")

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def analyze_video_quality_task(self, 
                              video_path: str,
                              script_text: str = "",
                              target_topic: str = "",
                              use_cache: bool = True,
                              priority: int = 0):
    """Celery task for video quality analysis"""
    
    task_id = self.request.id
    start_time = datetime.utcnow()
    
    logger.info(f"Starting quality analysis task {task_id} for {video_path}")
    
    # Initialize database record
    db = SessionLocal()
    try:
        task_result = QualityTaskResult(
            task_id=task_id,
            video_path=video_path,
            status='running',
            created_at=start_time
        )
        db.add(task_result)
        db.commit()
        
    except Exception as e:
        logger.error(f"Failed to create task record: {e}")
    finally:
        db.close()
    
    try:
        # Get scorer instance
        quality_scorer = get_scorer()
        
        # Check if video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Update status to running
        update_task_status(task_id, 'running')
        
        # Run quality analysis in async context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            metrics = loop.run_until_complete(
                quality_scorer.score_video(
                    video_path=video_path,
                    script_text=script_text,
                    target_topic=target_topic,
                    use_cache=use_cache
                )
            )
            
            # Generate report
            report = quality_scorer.get_quality_report(metrics)
            
        finally:
            loop.close()
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Prepare result data
        result_data = {
            'task_id': task_id,
            'video_path': video_path,
            'success': True,
            'metrics': metrics.__dict__,
            'report': report,
            'processing_time': processing_time,
            'overall_score': metrics.overall_quality_score,
            'completed_at': datetime.utcnow().isoformat()
        }
        
        # Update database
        update_task_status(
            task_id, 
            'completed',
            overall_score=metrics.overall_quality_score,
            processing_time=processing_time,
            metrics_json=str(metrics.__dict__),
            report_json=str(report)
        )
        
        logger.info(f"Quality analysis task {task_id} completed successfully. Score: {metrics.overall_quality_score:.3f}")
        
        return result_data
        
    except FileNotFoundError as e:
        error_msg = str(e)
        logger.error(f"Task {task_id} failed - file not found: {error_msg}")
        
        update_task_status(task_id, 'failed', error_message=error_msg)
        
        return {
            'task_id': task_id,
            'video_path': video_path,
            'success': False,
            'error': error_msg,
            'error_type': 'FileNotFoundError'
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Task {task_id} failed with error: {error_msg}")
        
        # Retry on certain errors
        if self.request.retries < self.max_retries:
            if "memory" in error_msg.lower() or "timeout" in error_msg.lower():
                logger.info(f"Retrying task {task_id} due to recoverable error")
                raise self.retry(countdown=60 * (self.request.retries + 1))
        
        update_task_status(task_id, 'failed', error_message=error_msg)
        
        return {
            'task_id': task_id,
            'video_path': video_path,
            'success': False,
            'error': error_msg,
            'retry_count': self.request.retries
        }

@celery_app.task(bind=True)
def batch_analyze_videos_task(self, video_data: List[Dict[str, Any]]):
    """Celery task for batch video quality analysis"""
    
    task_id = self.request.id
    start_time = datetime.utcnow()
    
    logger.info(f"Starting batch analysis task {task_id} for {len(video_data)} videos")
    
    results = []
    successful_count = 0
    failed_count = 0
    
    for i, video_info in enumerate(video_data):
        try:
            video_path = video_info['video_path']
            script_text = video_info.get('script_text', '')
            target_topic = video_info.get('target_topic', '')
            use_cache = video_info.get('use_cache', True)
            
            # Submit individual analysis task
            individual_task = analyze_video_quality_task.delay(
                video_path=video_path,
                script_text=script_text,
                target_topic=target_topic,
                use_cache=use_cache,
                priority=1  # Higher priority for batch items
            )
            
            # Wait for result (with timeout)
            try:
                result = individual_task.get(timeout=300)  # 5 minutes timeout
                if result['success']:
                    successful_count += 1
                else:
                    failed_count += 1
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Batch item {i} failed: {e}")
                failed_count += 1
                results.append({
                    'video_path': video_path,
                    'success': False,
                    'error': str(e),
                    'error_type': 'TaskTimeout'
                })
                
        except Exception as e:
            logger.error(f"Failed to process batch item {i}: {e}")
            failed_count += 1
            results.append({
                'video_path': video_info.get('video_path', 'unknown'),
                'success': False,
                'error': str(e),
                'error_type': 'ProcessingError'
            })
    
    total_processing_time = (datetime.utcnow() - start_time).total_seconds()
    
    batch_result = {
        'batch_task_id': task_id,
        'success': True,
        'total_videos': len(video_data),
        'successful_count': successful_count,
        'failed_count': failed_count,
        'results': results,
        'total_processing_time': total_processing_time,
        'completed_at': datetime.utcnow().isoformat()
    }
    
    logger.info(f"Batch analysis task {task_id} completed: {successful_count}/{len(video_data)} successful")
    
    return batch_result

@celery_app.task
def cleanup_old_results():
    """Celery task to cleanup old task results"""
    
    try:
        db = SessionLocal()
        
        # Delete task results older than 7 days
        cutoff_date = datetime.utcnow() - timedelta(days=7)
        
        deleted_count = db.query(QualityTaskResult).filter(
            QualityTaskResult.created_at < cutoff_date
        ).delete()
        
        db.commit()
        db.close()
        
        logger.info(f"Cleaned up {deleted_count} old task results")
        
        return {'deleted_count': deleted_count}
        
    except Exception as e:
        logger.error(f"Cleanup task failed: {e}")
        return {'error': str(e)}

@celery_app.task
def health_check_task():
    """Celery task for health checking"""
    
    try:
        # Test scorer initialization
        quality_scorer = get_scorer()
        
        # Test database connection
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        
        # Test Redis connection
        redis_client = redis.from_url(celery_broker_url)
        redis_client.ping()
        
        return {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'scorer_initialized': quality_scorer is not None,
            'database_connected': True,
            'redis_connected': True
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }

@celery_app.task
def get_quality_statistics():
    """Get quality analysis statistics"""
    
    try:
        db = SessionLocal()
        
        # Get task statistics
        total_tasks = db.query(QualityTaskResult).count()
        completed_tasks = db.query(QualityTaskResult).filter(
            QualityTaskResult.status == 'completed'
        ).count()
        failed_tasks = db.query(QualityTaskResult).filter(
            QualityTaskResult.status == 'failed'
        ).count()
        
        # Get average scores
        avg_score_result = db.query(
            db.func.avg(QualityTaskResult.overall_score),
            db.func.min(QualityTaskResult.overall_score),
            db.func.max(QualityTaskResult.overall_score)
        ).filter(
            QualityTaskResult.status == 'completed'
        ).first()
        
        # Recent tasks (last 24 hours)
        recent_cutoff = datetime.utcnow() - timedelta(hours=24)
        recent_tasks = db.query(QualityTaskResult).filter(
            QualityTaskResult.created_at > recent_cutoff
        ).count()
        
        db.close()
        
        return {
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'success_rate': completed_tasks / max(total_tasks, 1),
            'average_score': float(avg_score_result[0] or 0.0),
            'min_score': float(avg_score_result[1] or 0.0),
            'max_score': float(avg_score_result[2] or 0.0),
            'recent_tasks_24h': recent_tasks,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Statistics query failed: {e}")
        return {'error': str(e)}

# Periodic tasks
from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    'cleanup-old-results': {
        'task': 'quality_tasks.cleanup_old_results',
        'schedule': crontab(hour=2, minute=0),  # Daily at 2 AM
    },
    'health-check': {
        'task': 'quality_tasks.health_check_task',
        'schedule': 60.0,  # Every minute
    },
}

if __name__ == "__main__":
    # Run worker
    celery_app.start()