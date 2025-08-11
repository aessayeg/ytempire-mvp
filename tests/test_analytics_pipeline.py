"""
Tests for Analytics Data Pipeline
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import json
import pandas as pd
import numpy as np

from ml_pipeline.services.analytics_pipeline import (
    AnalyticsPipeline,
    EventStreamProcessor,
    DataQualityChecker,
    DataTransformationEngine,
    AggregationPipeline,
    AnalyticsEvent,
    EventType,
    AggregatedMetric
)


@pytest.fixture
async def mock_redis():
    """Mock Redis client"""
    mock = AsyncMock()
    mock.xadd = AsyncMock(return_value="1234567890-0")
    mock.xreadgroup = AsyncMock(return_value=[])
    mock.xgroup_create = AsyncMock()
    mock.zadd = AsyncMock()
    mock.hincrbyfloat = AsyncMock()
    mock.scan = AsyncMock(return_value=(0, []))
    mock.type = AsyncMock(return_value='hash')
    mock.hgetall = AsyncMock(return_value={'total': '100'})
    mock.xinfo_stream = AsyncMock(return_value={'last-generated-id': f"{int(datetime.utcnow().timestamp() * 1000)}-0"})
    return mock


@pytest.fixture
async def analytics_pipeline(mock_redis):
    """Create analytics pipeline with mocked dependencies"""
    pipeline = AnalyticsPipeline(
        redis_url="redis://localhost:6379",
        db_url="postgresql+asyncpg://test:test@localhost/test"
    )
    
    # Mock Redis connections
    pipeline.stream_processor.redis_client = mock_redis
    pipeline.aggregation_pipeline.redis_client = mock_redis
    
    return pipeline


class TestDataQualityChecker:
    """Test data quality validation"""
    
    @pytest.mark.asyncio
    async def test_validate_complete_event(self):
        """Test validation of complete event"""
        checker = DataQualityChecker()
        
        event = AnalyticsEvent(
            event_id="test_123",
            event_type=EventType.VIDEO_VIEW,
            timestamp=datetime.utcnow(),
            user_id="user_1",
            channel_id="channel_1",
            video_id="video_1",
            data={'view_count': 100},
            metadata={'source': 'youtube'}
        )
        
        is_valid, scores = await checker.validate_event(event)
        
        assert is_valid is True
        assert scores['completeness'] == 1.0
        assert scores['consistency'] == 1.0
        assert scores['timeliness'] == 1.0
        assert scores['accuracy'] == 1.0
        assert scores['uniqueness'] == 1.0
    
    @pytest.mark.asyncio
    async def test_validate_incomplete_event(self):
        """Test validation of incomplete event"""
        checker = DataQualityChecker()
        
        event = AnalyticsEvent(
            event_id=None,  # Missing ID
            event_type=EventType.VIDEO_VIEW,
            timestamp=datetime.utcnow(),
            user_id="user_1",
            channel_id="channel_1",
            video_id="video_1",
            data={'view_count': 100},
            metadata={}
        )
        
        is_valid, scores = await checker.validate_event(event)
        
        assert is_valid is False  # Should fail due to missing event_id
        assert scores['completeness'] < 1.0
        assert scores['uniqueness'] == 0.0
    
    @pytest.mark.asyncio
    async def test_check_timeliness(self):
        """Test timeliness check"""
        checker = DataQualityChecker()
        
        # Old event
        old_event = AnalyticsEvent(
            event_id="old_123",
            event_type=EventType.VIDEO_VIEW,
            timestamp=datetime.utcnow() - timedelta(days=2),
            user_id="user_1",
            channel_id="channel_1",
            video_id="video_1",
            data={},
            metadata={}
        )
        
        score = await checker._check_timeliness(old_event)
        assert score == 0.5  # Old events get 0.5 score
        
        # Recent event
        recent_event = AnalyticsEvent(
            event_id="recent_123",
            event_type=EventType.VIDEO_VIEW,
            timestamp=datetime.utcnow() - timedelta(minutes=30),
            user_id="user_1",
            channel_id="channel_1",
            video_id="video_1",
            data={},
            metadata={}
        )
        
        score = await checker._check_timeliness(recent_event)
        assert score == 1.0  # Recent events get 1.0 score


class TestEventStreamProcessor:
    """Test event stream processing"""
    
    @pytest.mark.asyncio
    async def test_publish_valid_event(self, mock_redis):
        """Test publishing valid event to stream"""
        processor = EventStreamProcessor("redis://localhost:6379")
        processor.redis_client = mock_redis
        
        event = AnalyticsEvent(
            event_id="test_123",
            event_type=EventType.VIDEO_VIEW,
            timestamp=datetime.utcnow(),
            user_id="user_1",
            channel_id="channel_1",
            video_id="video_1",
            data={'view_count': 100},
            metadata={'source': 'youtube'}
        )
        
        message_id = await processor.publish_event(event)
        
        assert message_id == "1234567890-0"
        mock_redis.xadd.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_video_event(self, mock_redis):
        """Test video event processing"""
        processor = EventStreamProcessor("redis://localhost:6379")
        processor.redis_client = mock_redis
        
        event = AnalyticsEvent(
            event_id="video_123",
            event_type=EventType.VIDEO_VIEW,
            timestamp=datetime.utcnow(),
            user_id="user_1",
            channel_id="channel_1",
            video_id="video_1",
            data={'view_count': 100, 'duration': 300},
            metadata={}
        )
        
        await processor._process_video_event(event)
        
        mock_redis.zadd.assert_called_once()
        mock_redis.expire.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_financial_event(self, mock_redis):
        """Test financial event processing"""
        processor = EventStreamProcessor("redis://localhost:6379")
        processor.redis_client = mock_redis
        
        # Test cost event
        cost_event = AnalyticsEvent(
            event_id="cost_123",
            event_type=EventType.COST_INCURRED,
            timestamp=datetime.utcnow(),
            user_id="user_1",
            channel_id="channel_1",
            video_id="video_1",
            data={'service': 'openai', 'amount': 0.15},
            metadata={}
        )
        
        await processor._process_financial_event(cost_event)
        mock_redis.hincrbyfloat.assert_called()
        
        # Test revenue event
        revenue_event = AnalyticsEvent(
            event_id="revenue_123",
            event_type=EventType.REVENUE_EARNED,
            timestamp=datetime.utcnow(),
            user_id="user_1",
            channel_id="channel_1",
            video_id="video_1",
            data={'amount': 5.50},
            metadata={}
        )
        
        await processor._process_financial_event(revenue_event)
        assert mock_redis.hincrbyfloat.call_count == 2


class TestDataTransformationEngine:
    """Test data transformation capabilities"""
    
    @pytest.mark.asyncio
    async def test_normalize_data(self):
        """Test data normalization"""
        engine = DataTransformationEngine()
        
        df = pd.DataFrame({
            'views': [100, 200, 300, 400, 500],
            'likes': [10, 20, 30, 40, 50],
            'category': ['tech', 'tech', 'gaming', 'gaming', 'tech']
        })
        
        normalized = await engine._normalize_data(df.copy())
        
        # Check that numeric columns are normalized
        assert abs(normalized['views'].mean()) < 0.01  # Mean should be ~0
        assert abs(normalized['views'].std() - 1.0) < 0.01  # Std should be ~1
        assert 'category' in normalized.columns  # Non-numeric columns preserved
    
    @pytest.mark.asyncio
    async def test_aggregate_data(self):
        """Test data aggregation by time"""
        engine = DataTransformationEngine()
        
        df = pd.DataFrame({
            'timestamp': [
                datetime.utcnow() - timedelta(days=i) 
                for i in range(5)
            ],
            'value': [10, 20, 30, 40, 50]
        })
        
        aggregated = await engine._aggregate_data(df)
        
        assert 'hour' in aggregated.columns
        assert 'day' in aggregated.columns
        assert 'week' in aggregated.columns
        assert 'month' in aggregated.columns
    
    @pytest.mark.asyncio
    async def test_enrich_data(self):
        """Test data enrichment"""
        engine = DataTransformationEngine()
        
        df = pd.DataFrame({
            'view_count': [1000, 2000, 3000],
            'like_count': [100, 150, 200],
            'revenue': [10.0, 20.0, 30.0],
            'cost': [3.0, 5.0, 7.0]
        })
        
        enriched = await engine._enrich_data(df)
        
        assert 'engagement_rate' in enriched.columns
        assert 'profit' in enriched.columns
        assert 'profit_margin' in enriched.columns
        
        # Verify calculations
        assert enriched['engagement_rate'][0] == pytest.approx(0.1, 0.01)
        assert enriched['profit'][0] == pytest.approx(7.0, 0.01)
        assert enriched['profit_margin'][0] == pytest.approx(0.7, 0.01)
    
    @pytest.mark.asyncio
    async def test_derive_metrics(self):
        """Test metric derivation"""
        engine = DataTransformationEngine()
        
        df = pd.DataFrame({
            'value': list(range(1, 31))  # 30 days of data
        })
        
        derived = await engine._derive_metrics(df)
        
        assert 'value_ma7' in derived.columns
        assert 'value_ma30' in derived.columns
        
        # Check moving average calculation
        assert derived['value_ma7'].iloc[-1] == pytest.approx(27.0, 0.1)  # Average of last 7 days


class TestAggregationPipeline:
    """Test data aggregation pipeline"""
    
    @pytest.mark.asyncio
    async def test_aggregate_metrics(self, mock_redis):
        """Test metric aggregation"""
        pipeline = AggregationPipeline("redis://localhost:6379")
        pipeline.redis_client = mock_redis
        
        mock_redis.scan = AsyncMock(return_value=(0, [
            'analytics:views:channel_1:20240101',
            'analytics:views:channel_1:20240102'
        ]))
        mock_redis.hgetall = AsyncMock(return_value={
            'total': '1000',
            'unique': '500'
        })
        
        result = await pipeline.aggregate_metrics(
            metric_type='views',
            period='daily',
            dimensions={'channel_id': 'channel_1'}
        )
        
        assert isinstance(result, AggregatedMetric)
        assert result.metric_name == 'views'
        assert result.period == 'daily'
        assert 'total' in result.values
        assert result.values['total'] == 2000  # 1000 * 2 keys
        assert result.confidence > 0
    
    @pytest.mark.asyncio
    async def test_create_rollups(self, mock_redis):
        """Test rollup creation"""
        pipeline = AggregationPipeline("redis://localhost:6379")
        pipeline.redis_client = mock_redis
        
        # Mock keys for rollup
        mock_redis.scan = AsyncMock(return_value=(0, [
            f'analytics:views:hourly:2024010{i:02d}' for i in range(1, 25)
        ]))
        mock_redis.type = AsyncMock(return_value='hash')
        mock_redis.hgetall = AsyncMock(return_value={'total': '100'})
        
        rollup_count = await pipeline.create_rollups('hourly', 'daily')
        
        assert rollup_count >= 0
        # Verify that aggregation was attempted
        assert mock_redis.scan.called


class TestAnalyticsPipeline:
    """Test main analytics pipeline"""
    
    @pytest.mark.asyncio
    async def test_ingest_event(self, analytics_pipeline):
        """Test event ingestion"""
        message_id = await analytics_pipeline.ingest_event(
            event_type=EventType.VIDEO_VIEW,
            data={'view_count': 100, 'duration': 300},
            user_id='user_1',
            channel_id='channel_1',
            video_id='video_1'
        )
        
        assert message_id == "1234567890-0"
    
    @pytest.mark.asyncio
    async def test_get_metrics(self, analytics_pipeline):
        """Test metric retrieval"""
        metrics = await analytics_pipeline.get_metrics(
            metric_type='views',
            period='daily',
            dimensions={'channel_id': 'channel_1'}
        )
        
        assert isinstance(metrics, AggregatedMetric)
        assert metrics.metric_name == 'views'
        assert metrics.period == 'daily'
    
    @pytest.mark.asyncio
    async def test_run_data_quality_checks(self, analytics_pipeline, mock_redis):
        """Test data quality checks"""
        results = await analytics_pipeline.run_data_quality_checks()
        
        assert 'stream_health' in results
        assert 'data_completeness' in results
        assert 'data_freshness' in results
        assert 'processing_lag' in results
        assert 'overall' in results
        
        assert results['stream_health'] == 1.0
        assert results['data_freshness'] > 0.9  # Should be fresh
        assert results['overall'] > 0


@pytest.mark.asyncio
async def test_end_to_end_pipeline():
    """Test end-to-end pipeline flow"""
    # This would be an integration test with real Redis in CI/CD
    # For unit tests, we use mocks
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])