"""
Unit tests for database models
"""
import pytest
from datetime import datetime
from app.models.user import User
from app.models.channel import Channel
from app.models.video import Video
from app.models.cost import Cost
from app.core.security import get_password_hash

class TestUserModel:
    """Test User model"""
    
    def test_user_creation(self):
        """Test creating a user instance"""
        user = User(
            email="test@example.com",
            full_name="Test User",
            hashed_password=get_password_hash("testpass123"),
            is_active=True,
            is_verified=False
        )
        
        assert user.email == "test@example.com"
        assert user.full_name == "Test User"
        assert user.is_active is True
        assert user.is_verified is False
        assert user.hashed_password != "testpass123"
    
    def test_user_str_representation(self):
        """Test user string representation"""
        user = User(email="test@example.com", full_name="Test User")
        assert str(user) == "User(email=test@example.com)"
    
    def test_user_default_values(self):
        """Test user default values"""
        user = User(
            email="test@example.com",
            hashed_password="hashed"
        )
        
        assert user.is_active is True
        assert user.is_verified is False
        assert user.is_superuser is False
        assert user.channels_limit == 5

class TestChannelModel:
    """Test Channel model"""
    
    def test_channel_creation(self):
        """Test creating a channel instance"""
        channel = Channel(
            name="Test Channel",
            description="Test Description",
            owner_id=1,
            youtube_channel_id="UC123456",
            is_active=True
        )
        
        assert channel.name == "Test Channel"
        assert channel.description == "Test Description"
        assert channel.owner_id == 1
        assert channel.youtube_channel_id == "UC123456"
        assert channel.is_active is True
    
    def test_channel_health_score_default(self):
        """Test channel health score default"""
        channel = Channel(
            name="Test Channel",
            owner_id=1
        )
        
        assert channel.health_score == 1.0
        assert channel.is_active is True
    
    def test_channel_relationships(self):
        """Test channel relationships"""
        channel = Channel(
            name="Test Channel",
            owner_id=1
        )
        
        # Videos relationship should be empty initially
        assert channel.videos == []

class TestVideoModel:
    """Test Video model"""
    
    def test_video_creation(self):
        """Test creating a video instance"""
        video = Video(
            title="Test Video",
            description="Test Description",
            channel_id=1,
            status="pending",
            duration=300,
            topic="Test Topic"
        )
        
        assert video.title == "Test Video"
        assert video.description == "Test Description"
        assert video.channel_id == 1
        assert video.status == "pending"
        assert video.duration == 300
        assert video.topic == "Test Topic"
    
    def test_video_default_values(self):
        """Test video default values"""
        video = Video(
            title="Test Video",
            channel_id=1
        )
        
        assert video.status == "pending"
        assert video.views == 0
        assert video.likes == 0
        assert video.comments == 0
        assert video.revenue == 0.0
    
    def test_video_status_transitions(self):
        """Test video status transitions"""
        video = Video(
            title="Test Video",
            channel_id=1,
            status="pending"
        )
        
        # Valid status values
        valid_statuses = ["pending", "processing", "published", "failed", "deleted"]
        
        for status in valid_statuses:
            video.status = status
            assert video.status == status

class TestCostModel:
    """Test Cost model"""
    
    def test_cost_creation(self):
        """Test creating a cost instance"""
        cost = Cost(
            video_id=1,
            service="openai",
            operation="script_generation",
            amount=0.05,
            tokens_used=1000
        )
        
        assert cost.video_id == 1
        assert cost.service == "openai"
        assert cost.operation == "script_generation"
        assert cost.amount == 0.05
        assert cost.tokens_used == 1000
    
    def test_cost_calculations(self):
        """Test cost calculations"""
        costs = [
            Cost(video_id=1, service="openai", amount=0.05),
            Cost(video_id=1, service="elevenlabs", amount=0.10),
            Cost(video_id=1, service="dalle", amount=0.02)
        ]
        
        total = sum(c.amount for c in costs)
        assert total == 0.17
    
    def test_cost_by_service(self):
        """Test grouping costs by service"""
        costs = [
            Cost(video_id=1, service="openai", amount=0.05),
            Cost(video_id=1, service="openai", amount=0.03),
            Cost(video_id=1, service="elevenlabs", amount=0.10)
        ]
        
        service_costs = {}
        for cost in costs:
            if cost.service not in service_costs:
                service_costs[cost.service] = 0
            service_costs[cost.service] += cost.amount
        
        assert service_costs["openai"] == 0.08
        assert service_costs["elevenlabs"] == 0.10

class TestModelRelationships:
    """Test model relationships"""
    
    def test_user_channel_relationship(self):
        """Test user-channel relationship"""
        user = User(
            id=1,
            email="test@example.com",
            hashed_password="hashed"
        )
        
        channel1 = Channel(name="Channel 1", owner_id=1, owner=user)
        channel2 = Channel(name="Channel 2", owner_id=1, owner=user)
        
        user.channels = [channel1, channel2]
        
        assert len(user.channels) == 2
        assert channel1.owner == user
        assert channel2.owner == user
    
    def test_channel_video_relationship(self):
        """Test channel-video relationship"""
        channel = Channel(
            id=1,
            name="Test Channel",
            owner_id=1
        )
        
        video1 = Video(title="Video 1", channel_id=1, channel=channel)
        video2 = Video(title="Video 2", channel_id=1, channel=channel)
        
        channel.videos = [video1, video2]
        
        assert len(channel.videos) == 2
        assert video1.channel == channel
        assert video2.channel == channel
    
    def test_video_cost_relationship(self):
        """Test video-cost relationship"""
        video = Video(
            id=1,
            title="Test Video",
            channel_id=1
        )
        
        cost1 = Cost(video_id=1, service="openai", amount=0.05, video=video)
        cost2 = Cost(video_id=1, service="elevenlabs", amount=0.10, video=video)
        
        video.costs = [cost1, cost2]
        
        assert len(video.costs) == 2
        assert cost1.video == video
        assert cost2.video == video
        
        total_cost = sum(c.amount for c in video.costs)
        assert total_cost == 0.15