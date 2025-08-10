"""
Unit Tests for Video Pipeline
Owner: QA Engineer #1
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import json

from app.tasks.video_pipeline import (
    start_video_generation,
    execute_video_pipeline,
    update_video_status,
    finalize_video_pipeline,
    handle_pipeline_failure,
    create_video_pipeline,
    get_pipeline_status
)
from app.models.video import Video, VideoStatus
from app.core.celery_app import celery_app


class TestVideoPipeline:
    """Test suite for video pipeline functionality."""

    @pytest.fixture
    def sample_video_request(self):
        """Sample video generation request."""
        return {
            'id': 'test_video_123',
            'channel_id': 'test_channel_456',
            'topic': 'AI Trends 2025',
            'user_id': 'test_user_789'
        }

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        with patch('app.tasks.video_pipeline.get_db') as mock_get_db:
            mock_session = Mock()
            mock_get_db.return_value = iter([mock_session])
            yield mock_session

    def test_start_video_generation_success(self, sample_video_request, mock_db_session):
        """Test successful video generation initialization."""
        # Mock Celery task
        with patch.object(start_video_generation, 'request') as mock_request:
            mock_request.id = 'task_123'
            
            result = start_video_generation.run(sample_video_request)
            
            # Assertions
            assert result['id'] == sample_video_request['id']
            assert result['channel_id'] == sample_video_request['channel_id']
            assert result['topic'] == sample_video_request['topic']
            assert result['status'] == 'processing'
            assert result['stage'] == 'initialized'
            assert 'pipeline_id' in result
            assert 'started_at' in result
            assert 'costs' in result
            
            # Verify database call
            mock_db_session.add.assert_called_once()
            mock_db_session.commit.assert_called_once()

    def test_start_video_generation_missing_fields(self, mock_db_session):
        """Test video generation with missing required fields."""
        invalid_request = {'id': 'test_123'}  # Missing required fields
        
        with pytest.raises(Exception) as exc_info:
            start_video_generation.run(invalid_request)
        
        assert "Missing required field" in str(exc_info.value)

    def test_update_video_status_success(self, mock_db_session):
        """Test successful video status update."""
        # Mock video in database
        mock_video = Mock(spec=Video)
        mock_video.id = 'test_video_123'
        mock_video.metadata = {}
        mock_db_session.query().filter().first.return_value = mock_video
        
        metadata = {'progress': 50, 'current_task': 'content_generation'}
        result = update_video_status.run('test_video_123', 'processing', 'content_generation', metadata)
        
        # Assertions
        assert result['video_id'] == 'test_video_123'
        assert result['status'] == 'processing'
        assert result['stage'] == 'content_generation'
        assert result['metadata'] == metadata
        assert 'timestamp' in result
        
        # Verify database updates
        assert mock_video.status == VideoStatus.PROCESSING
        assert mock_video.current_stage == 'content_generation'
        mock_db_session.commit.assert_called_once()

    def test_update_video_status_video_not_found(self, mock_db_session):
        """Test status update for non-existent video."""
        mock_db_session.query().filter().first.return_value = None
        
        with pytest.raises(Exception) as exc_info:
            update_video_status.run('nonexistent_video', 'processing', 'test')
        
        assert "Video not found" in str(exc_info.value)

    @patch('app.tasks.video_pipeline.chain')
    def test_execute_video_pipeline_success(self, mock_chain, sample_video_request):
        """Test successful pipeline execution."""
        # Mock chain execution
        mock_result = Mock()
        mock_result.id = 'pipeline_task_123'
        mock_chain.return_value.apply_async.return_value = mock_result
        
        with patch.object(update_video_status, 'delay') as mock_update_status:
            result = execute_video_pipeline.run(sample_video_request)
            
            # Assertions
            assert result['video_id'] == sample_video_request['id']
            assert result['pipeline_task_id'] == 'pipeline_task_123'
            assert result['status'] == 'pipeline_executing'
            assert 'started_at' in result
            
            # Verify status update was called
            mock_update_status.assert_called()

    def test_finalize_video_pipeline_success(self, mock_db_session):
        """Test successful pipeline finalization."""
        # Mock video in database
        mock_video = Mock(spec=Video)
        mock_video.pipeline_started_at = datetime.now()
        mock_video.metadata = {}
        mock_db_session.query().filter().first.return_value = mock_video
        
        upload_result = {
            'video_id': 'test_video_123',
            'youtube_video_id': 'yt_video_456',
            'youtube_url': 'https://youtube.com/watch?v=yt_video_456',
            'content_cost': 0.75,
            'audio_cost': 1.20,
            'visual_cost': 0.45,
            'compilation_cost': 0.10
        }
        
        with patch.object(update_video_status, 'delay') as mock_update_status:
            result = finalize_video_pipeline.run(upload_result)
            
            # Assertions
            assert result['video_id'] == 'test_video_123'
            assert result['status'] == 'completed'
            assert result['youtube_video_id'] == 'yt_video_456'
            assert result['total_cost'] == 2.50  # Sum of all costs
            assert 'completed_at' in result
            
            # Verify database updates
            assert mock_video.status == VideoStatus.COMPLETED
            assert mock_video.youtube_video_id == 'yt_video_456'
            assert mock_video.total_cost == 2.50
            mock_db_session.commit.assert_called_once()
            
            # Verify status update
            mock_update_status.assert_called()

    def test_handle_pipeline_failure(self, mock_db_session):
        """Test pipeline failure handling."""
        # Mock video in database
        mock_video = Mock(spec=Video)
        mock_video.metadata = {}
        mock_db_session.query().filter().first.return_value = mock_video
        
        with patch.object(update_video_status, 'delay') as mock_update_status:
            result = handle_pipeline_failure.run(
                'test_video_123',
                'Content generation failed',
                'content_generation'
            )
            
            # Assertions
            assert result['video_id'] == 'test_video_123'
            assert result['status'] == 'failed'
            assert result['error'] == 'Content generation failed'
            assert result['stage'] == 'content_generation'
            
            # Verify database updates
            assert mock_video.status == VideoStatus.FAILED
            assert mock_video.error_message == 'Content generation failed'
            mock_db_session.commit.assert_called_once()

    @patch('app.tasks.video_pipeline.chain')
    def test_create_video_pipeline(self, mock_chain, sample_video_request):
        """Test pipeline creation."""
        mock_result = Mock()
        mock_result.id = 'pipeline_task_456'
        mock_chain.return_value.apply_async.return_value = mock_result
        
        pipeline_id = create_video_pipeline(sample_video_request)
        
        assert pipeline_id == 'pipeline_task_456'
        mock_chain.assert_called_once()

    def test_get_pipeline_status(self):
        """Test pipeline status retrieval."""
        with patch('app.tasks.video_pipeline.celery_app') as mock_celery:
            mock_task_result = Mock()
            mock_task_result.status = 'SUCCESS'
            mock_task_result.result = {'completed': True}
            mock_task_result.successful.return_value = True
            mock_task_result.failed.return_value = False
            
            mock_celery.AsyncResult.return_value = mock_task_result
            
            status = get_pipeline_status('test_task_123')
            
            assert status['task_id'] == 'test_task_123'
            assert status['status'] == 'SUCCESS'
            assert status['result'] == {'completed': True}
            assert status['error'] is None

    def test_cost_budget_validation(self, sample_video_request, mock_db_session):
        """Test cost budget validation during initialization."""
        # Mock configuration with low budget
        with patch('app.tasks.video_pipeline.settings') as mock_settings:
            mock_settings.MAX_COST_PER_VIDEO = 1.0  # Very low budget
            
            result = start_video_generation.run(sample_video_request)
            
            # Should still initialize but with budget warning
            assert result['costs']['budget_limit'] == 1.0

    @pytest.mark.asyncio
    async def test_pipeline_retry_logic(self, sample_video_request, mock_db_session):
        """Test pipeline retry logic on failures."""
        with patch.object(start_video_generation, 'retry') as mock_retry:
            # Simulate database error
            mock_db_session.add.side_effect = Exception("Database connection failed")
            
            with pytest.raises(Exception):
                start_video_generation.run(sample_video_request)
            
            # Should have attempted retry
            mock_retry.assert_called_once()

    def test_pipeline_status_tracking(self, sample_video_request):
        """Test status tracking throughout pipeline."""
        status_updates = []
        
        def mock_update_status(*args, **kwargs):
            status_updates.append(args)
        
        with patch.object(update_video_status, 'delay', side_effect=mock_update_status):
            with patch('app.tasks.video_pipeline.chain') as mock_chain:
                mock_chain.return_value.apply_async.return_value = Mock(id='task_123')
                
                execute_video_pipeline.run(sample_video_request)
                
                # Should have called status update
                assert len(status_updates) > 0
                assert status_updates[0][1] == 'processing'  # status
                assert status_updates[0][2] == 'content_generation'  # stage


@pytest.mark.integration
class TestVideoPipelineIntegration:
    """Integration tests for video pipeline."""

    @pytest.fixture(scope="class")
    def celery_app(self):
        """Configure Celery for testing."""
        celery_app.conf.update(
            task_always_eager=True,
            task_eager_propagates=True,
            broker_url='memory://',
            result_backend='cache+memory://'
        )
        return celery_app

    def test_full_pipeline_integration(self, celery_app, mock_db_session):
        """Test complete pipeline integration."""
        video_request = {
            'id': 'integration_test_video',
            'channel_id': 'test_channel',
            'topic': 'Integration Test Topic',
            'user_id': 'test_user'
        }
        
        with patch('app.tasks.content_generation.generate_content_task') as mock_content:
            with patch('app.tasks.audio_synthesis.synthesize_audio_task') as mock_audio:
                with patch('app.tasks.video_compilation.compile_video_task') as mock_compile:
                    with patch('app.tasks.youtube_upload.upload_to_youtube_task') as mock_upload:
                        
                        # Mock task results
                        mock_content.s.return_value = Mock()
                        mock_audio.s.return_value = Mock()
                        mock_compile.s.return_value = Mock()
                        mock_upload.s.return_value = Mock()
                        
                        # Execute pipeline
                        pipeline_id = create_video_pipeline(video_request)
                        
                        # Verify pipeline was created
                        assert pipeline_id is not None
                        
                        # Check status
                        status = get_pipeline_status(pipeline_id)
                        assert status['task_id'] == pipeline_id


if __name__ == '__main__':
    pytest.main([__file__, '-v'])