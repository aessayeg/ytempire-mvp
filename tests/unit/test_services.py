"""
Unit tests for service layers
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
import json
from app.services.youtube_service import YouTubeService
from app.services.video_generation_service import VideoGenerationService
from app.services.cost_tracking_service import CostTrackingService
from app.services.email_service import EmailService
from app.services.optimized_queries import OptimizedQueryService

class TestYouTubeService:
    """Test YouTube service"""
    
    @pytest.fixture
    def youtube_service(self):
        """Create YouTube service instance"""
        with patch('app.services.youtube_service.build'):
            service = YouTubeService()
            service.youtube = Mock()
            return service
    
    @pytest.mark.asyncio
    async def test_upload_video_success(self, youtube_service):
        """Test successful video upload"""
        # Arrange
        youtube_service.youtube.videos().insert().execute.return_value = {
            'id': 'video_123',
            'snippet': {'title': 'Test Video'}
        }
        
        # Act
        result = await youtube_service.upload_video(
            title="Test Video",
            description="Test Description",
            video_path="/path/to/video.mp4",
            tags=["test", "video"]
        )
        
        # Assert
        assert result['id'] == 'video_123'
        youtube_service.youtube.videos().insert.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_channel_stats(self, youtube_service):
        """Test getting channel statistics"""
        # Arrange
        youtube_service.youtube.channels().list().execute.return_value = {
            'items': [{
                'statistics': {
                    'viewCount': '1000',
                    'subscriberCount': '100',
                    'videoCount': '10'
                }
            }]
        }
        
        # Act
        stats = await youtube_service.get_channel_stats('channel_123')
        
        # Assert
        assert stats['viewCount'] == '1000'
        assert stats['subscriberCount'] == '100'
        assert stats['videoCount'] == '10'
    
    @pytest.mark.asyncio
    async def test_check_quota_usage(self, youtube_service):
        """Test quota usage checking"""
        # Arrange
        youtube_service.quota_used = 5000
        youtube_service.daily_quota_limit = 10000
        
        # Act
        has_quota = youtube_service.has_quota_available(1000)
        remaining = youtube_service.get_remaining_quota()
        
        # Assert
        assert has_quota is True
        assert remaining == 5000

class TestVideoGenerationService:
    """Test video generation service"""
    
    @pytest.fixture
    def video_service(self):
        """Create video generation service instance"""
        service = VideoGenerationService()
        return service
    
    @pytest.mark.asyncio
    async def test_generate_script(self, video_service):
        """Test script generation"""
        # Arrange
        with patch('app.services.video_generation_service.openai') as mock_openai:
            mock_openai.ChatCompletion.create.return_value = {
                'choices': [{'message': {'content': 'Generated script content'}}]
            }
            
            # Act
            script = await video_service.generate_script(
                topic="Test Topic",
                style="educational",
                duration=300
            )
            
            # Assert
            assert script == 'Generated script content'
            mock_openai.ChatCompletion.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_synthesize_voice(self, video_service):
        """Test voice synthesis"""
        # Arrange
        with patch('app.services.video_generation_service.elevenlabs') as mock_elevenlabs:
            mock_elevenlabs.generate.return_value = b'audio_data'
            
            # Act
            audio_data = await video_service.synthesize_voice(
                script="Test script",
                voice_id="voice_123"
            )
            
            # Assert
            assert audio_data == b'audio_data'
            mock_elevenlabs.generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_thumbnail(self, video_service):
        """Test thumbnail creation"""
        # Arrange
        with patch('app.services.video_generation_service.openai') as mock_openai:
            mock_openai.Image.create.return_value = {
                'data': [{'url': 'https://example.com/thumbnail.png'}]
            }
            
            # Act
            thumbnail_url = await video_service.create_thumbnail(
                title="Test Video",
                style="modern"
            )
            
            # Assert
            assert thumbnail_url == 'https://example.com/thumbnail.png'
            mock_openai.Image.create.assert_called_once()

class TestCostTrackingService:
    """Test cost tracking service"""
    
    @pytest.fixture
    def cost_service(self):
        """Create cost tracking service instance"""
        service = CostTrackingService()
        service.db = AsyncMock()
        return service
    
    @pytest.mark.asyncio
    async def test_track_cost(self, cost_service):
        """Test cost tracking"""
        # Arrange
        cost_service.db.add = Mock()
        cost_service.db.commit = AsyncMock()
        
        # Act
        await cost_service.track_cost(
            service="openai",
            operation="script_generation",
            amount=0.05,
            video_id=1
        )
        
        # Assert
        cost_service.db.add.assert_called_once()
        cost_service.db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_video_cost(self, cost_service):
        """Test getting total cost for a video"""
        # Arrange
        mock_costs = [
            Mock(amount=0.05, service="openai"),
            Mock(amount=0.10, service="elevenlabs"),
            Mock(amount=0.02, service="dalle")
        ]
        cost_service.db.execute.return_value.scalars.return_value.all.return_value = mock_costs
        
        # Act
        total_cost = await cost_service.get_video_cost(video_id=1)
        
        # Assert
        assert total_cost == 0.17
    
    @pytest.mark.asyncio
    async def test_get_daily_cost(self, cost_service):
        """Test getting daily cost"""
        # Arrange
        cost_service.db.execute.return_value.scalar.return_value = 25.50
        
        # Act
        daily_cost = await cost_service.get_daily_cost()
        
        # Assert
        assert daily_cost == 25.50
        cost_service.db.execute.assert_called_once()

class TestEmailService:
    """Test email service"""
    
    @pytest.fixture
    def email_service(self):
        """Create email service instance"""
        service = EmailService()
        return service
    
    @pytest.mark.asyncio
    async def test_send_email(self, email_service):
        """Test sending email"""
        # Arrange
        with patch('app.services.email_service.smtplib.SMTP') as mock_smtp:
            mock_server = Mock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            # Act
            result = await email_service.send_email(
                to_email="test@example.com",
                subject="Test Email",
                body="Test content"
            )
            
            # Assert
            assert result is True
            mock_server.send_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_verification_email(self, email_service):
        """Test sending verification email"""
        # Arrange
        with patch.object(email_service, 'send_email', return_value=True) as mock_send:
            # Act
            result = await email_service.send_verification_email(
                email="test@example.com",
                token="verify_token_123"
            )
            
            # Assert
            assert result is True
            mock_send.assert_called_once()
            call_args = mock_send.call_args[1]
            assert "Verify Your Email" in call_args['subject']
            assert "verify_token_123" in call_args['body']

class TestOptimizedQueries:
    """Test optimized query service"""
    
    @pytest.fixture
    def query_service(self):
        """Create query service instance"""
        return OptimizedQueryService()
    
    @pytest.fixture
    def mock_db(self):
        """Mock database session"""
        db = AsyncMock()
        return db
    
    @pytest.mark.asyncio
    async def test_get_user_with_channels(self, query_service, mock_db):
        """Test getting user with channels eagerly loaded"""
        # Arrange
        mock_user = Mock()
        mock_user.id = 1
        mock_user.channels = [Mock(), Mock()]
        
        mock_db.execute.return_value.scalar_one_or_none.return_value = mock_user
        
        # Act
        result = await query_service.get_user_with_channels(mock_db, user_id=1)
        
        # Assert
        assert result == mock_user
        assert len(result.channels) == 2
        mock_db.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_dashboard_data(self, query_service, mock_db):
        """Test getting optimized dashboard data"""
        # Arrange
        mock_row = (
            Mock(id=1, email="test@example.com"),  # User
            5,  # channel_count
            50,  # video_count
            10000,  # total_views
            500,  # total_likes
            150.0  # total_cost
        )
        mock_db.execute.return_value.first.return_value = mock_row
        
        # Act
        result = await query_service.get_dashboard_data(mock_db, user_id=1)
        
        # Assert
        assert result['channel_count'] == 5
        assert result['video_count'] == 50
        assert result['total_views'] == 10000
        assert result['total_likes'] == 500
        assert result['total_cost'] == 150.0
    
    @pytest.mark.asyncio
    async def test_get_channel_performance_metrics(self, query_service, mock_db):
        """Test getting channel performance metrics"""
        # Arrange
        mock_row = Mock(
            total_videos=20,
            total_views=5000,
            total_likes=200,
            total_comments=50,
            avg_views=250,
            avg_duration=300,
            last_video_date=datetime.now(),
            total_cost=100.0
        )
        mock_db.execute.return_value.first.return_value = mock_row
        
        # Act
        result = await query_service.get_channel_performance_metrics(mock_db, channel_id=1)
        
        # Assert
        assert result['total_videos'] == 20
        assert result['total_views'] == 5000
        assert result['total_cost'] == 100.0
        assert 'engagement_rate' not in result  # Calculated in endpoint, not service