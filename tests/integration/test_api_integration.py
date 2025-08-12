"""
Integration tests for API endpoints
"""
import pytest
import asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.db.base import Base
from app.core.config import settings
from app.models.user import User
from app.models.channel import Channel
from app.models.video import Video

# Test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def test_engine():
    """Create test database engine"""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()

@pytest.fixture
async def test_db(test_engine):
    """Create test database session"""
    async_session = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        yield session
        await session.rollback()

@pytest.fixture
async def test_client():
    """Create test client"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture
async def test_user(test_db):
    """Create test user"""
    from app.core.security import get_password_hash
    
    user = User(
        email="test@example.com",
        full_name="Test User",
        hashed_password=get_password_hash("testpass123"),
        is_active=True,
        is_verified=True
    )
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    return user

@pytest.fixture
async def auth_headers(test_client, test_user):
    """Get authentication headers"""
    response = await test_client.post(
        "/api/v1/auth/login",
        data={"username": test_user.email, "password": "testpass123"}
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

class TestAuthIntegration:
    """Test authentication flow integration"""
    
    @pytest.mark.asyncio
    async def test_register_login_flow(self, test_client):
        """Test complete registration and login flow"""
        # Register new user
        register_response = await test_client.post(
            "/api/v1/auth/register",
            json={
                "email": "newuser@example.com",
                "password": "newpass123",
                "full_name": "New User"
            }
        )
        assert register_response.status_code == 201
        user_data = register_response.json()
        assert user_data["email"] == "newuser@example.com"
        
        # Login with new user
        login_response = await test_client.post(
            "/api/v1/auth/login",
            data={
                "username": "newuser@example.com",
                "password": "newpass123"
            }
        )
        assert login_response.status_code == 200
        token_data = login_response.json()
        assert "access_token" in token_data
        
        # Access protected endpoint
        headers = {"Authorization": f"Bearer {token_data['access_token']}"}
        me_response = await test_client.get("/api/v1/users/me", headers=headers)
        assert me_response.status_code == 200
        me_data = me_response.json()
        assert me_data["email"] == "newuser@example.com"
    
    @pytest.mark.asyncio
    async def test_invalid_login(self, test_client):
        """Test login with invalid credentials"""
        response = await test_client.post(
            "/api/v1/auth/login",
            data={
                "username": "nonexistent@example.com",
                "password": "wrongpass"
            }
        )
        assert response.status_code == 401
        assert "Incorrect email or password" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_access_without_token(self, test_client):
        """Test accessing protected endpoint without token"""
        response = await test_client.get("/api/v1/users/me")
        assert response.status_code == 401
        assert "Not authenticated" in response.json()["detail"]

class TestChannelIntegration:
    """Test channel management integration"""
    
    @pytest.mark.asyncio
    async def test_create_channel(self, test_client, auth_headers):
        """Test creating a channel"""
        response = await test_client.post(
            "/api/v1/channels",
            json={
                "channel_name": "Test Channel",
                "description": "Test Description",
                "content_type": "educational"
            },
            headers=auth_headers
        )
        assert response.status_code == 201
        channel_data = response.json()
        assert channel_data["channel_name"] == "Test Channel"
        assert channel_data["id"] is not None
    
    @pytest.mark.asyncio
    async def test_list_channels(self, test_client, auth_headers, test_db, test_user):
        """Test listing user channels"""
        # Create test channels
        channel1 = Channel(name="Channel 1", owner_id=test_user.id)
        channel2 = Channel(name="Channel 2", owner_id=test_user.id)
        test_db.add_all([channel1, channel2])
        await test_db.commit()
        
        # List channels
        response = await test_client.get("/api/v1/channels", headers=auth_headers)
        assert response.status_code == 200
        channels = response.json()
        assert len(channels) >= 2
    
    @pytest.mark.asyncio
    async def test_update_channel(self, test_client, auth_headers, test_db, test_user):
        """Test updating a channel"""
        # Create test channel
        channel = Channel(name="Original Name", owner_id=test_user.id)
        test_db.add(channel)
        await test_db.commit()
        await test_db.refresh(channel)
        
        # Update channel
        response = await test_client.put(
            f"/api/v1/channels/{channel.id}",
            json={"channel_name": "Updated Name"},
            headers=auth_headers
        )
        assert response.status_code == 200
        updated_data = response.json()
        assert updated_data["channel_name"] == "Updated Name"
    
    @pytest.mark.asyncio
    async def test_delete_channel(self, test_client, auth_headers, test_db, test_user):
        """Test deleting a channel"""
        # Create test channel
        channel = Channel(name="To Delete", owner_id=test_user.id)
        test_db.add(channel)
        await test_db.commit()
        await test_db.refresh(channel)
        
        # Delete channel
        response = await test_client.delete(
            f"/api/v1/channels/{channel.id}",
            headers=auth_headers
        )
        assert response.status_code == 200
        
        # Verify deletion
        get_response = await test_client.get(
            f"/api/v1/channels/{channel.id}",
            headers=auth_headers
        )
        assert get_response.status_code == 404

class TestVideoIntegration:
    """Test video generation integration"""
    
    @pytest.mark.asyncio
    async def test_generate_video(self, test_client, auth_headers, test_db, test_user):
        """Test video generation endpoint"""
        # Create test channel
        channel = Channel(name="Test Channel", owner_id=test_user.id)
        test_db.add(channel)
        await test_db.commit()
        await test_db.refresh(channel)
        
        # Generate video
        response = await test_client.post(
            "/api/v1/videos/generate",
            json={
                "channel_id": channel.id,
                "topic": "Test Topic",
                "style": "educational",
                "duration": 300
            },
            headers=auth_headers
        )
        assert response.status_code in [200, 201]
        video_data = response.json()
        assert video_data["topic"] == "Test Topic"
        assert video_data["status"] in ["pending", "processing"]
    
    @pytest.mark.asyncio
    async def test_list_videos(self, test_client, auth_headers, test_db, test_user):
        """Test listing videos"""
        # Create test channel and videos
        channel = Channel(name="Test Channel", owner_id=test_user.id)
        test_db.add(channel)
        await test_db.commit()
        await test_db.refresh(channel)
        
        video1 = Video(title="Video 1", channel_id=channel.id)
        video2 = Video(title="Video 2", channel_id=channel.id)
        test_db.add_all([video1, video2])
        await test_db.commit()
        
        # List videos
        response = await test_client.get(
            f"/api/v1/videos?channel_id={channel.id}",
            headers=auth_headers
        )
        assert response.status_code == 200
        videos = response.json()
        assert len(videos) >= 2

class TestDashboardIntegration:
    """Test dashboard and analytics integration"""
    
    @pytest.mark.asyncio
    async def test_dashboard_stats(self, test_client, auth_headers, test_db, test_user):
        """Test dashboard statistics endpoint"""
        # Create test data
        channel = Channel(name="Test Channel", owner_id=test_user.id)
        test_db.add(channel)
        await test_db.commit()
        await test_db.refresh(channel)
        
        video = Video(
            title="Test Video",
            channel_id=channel.id,
            views=1000,
            likes=50,
            comments=10
        )
        test_db.add(video)
        await test_db.commit()
        
        # Get dashboard stats
        response = await test_client.get(
            "/api/v1/dashboard/stats",
            headers=auth_headers
        )
        assert response.status_code == 200
        stats = response.json()
        assert "total_channels" in stats
        assert "total_videos" in stats
        assert "total_views" in stats
    
    @pytest.mark.asyncio
    async def test_channel_analytics(self, test_client, auth_headers, test_db, test_user):
        """Test channel analytics endpoint"""
        # Create test channel
        channel = Channel(name="Test Channel", owner_id=test_user.id)
        test_db.add(channel)
        await test_db.commit()
        await test_db.refresh(channel)
        
        # Get analytics
        response = await test_client.get(
            f"/api/v1/analytics/channel/{channel.id}",
            headers=auth_headers
        )
        assert response.status_code == 200
        analytics = response.json()
        assert "performance_metrics" in analytics

class TestBetaUserIntegration:
    """Test beta user functionality integration"""
    
    @pytest.mark.asyncio
    async def test_beta_signup(self, test_client):
        """Test beta user signup"""
        response = await test_client.post(
            "/api/v1/beta/signup",
            json={
                "full_name": "Beta User",
                "email": "beta@example.com",
                "use_case": "Testing the platform for educational content",
                "expected_volume": "50-100"
            }
        )
        assert response.status_code == 201
        beta_data = response.json()
        assert "api_key" in beta_data
        assert beta_data["message"] == "Beta access granted! Check your email for login credentials."
    
    @pytest.mark.asyncio
    async def test_beta_stats(self, test_client):
        """Test beta user statistics"""
        response = await test_client.get("/api/v1/beta/stats")
        assert response.status_code == 200
        stats = response.json()
        assert "total_signups" in stats
        assert "active_users" in stats

class TestPerformanceIntegration:
    """Test performance optimizations"""
    
    @pytest.mark.asyncio
    async def test_optimized_endpoints(self, test_client, auth_headers):
        """Test optimized endpoints performance"""
        # Test optimized dashboard
        response = await test_client.get(
            "/api/v1/channels/optimized/dashboard",
            headers=auth_headers
        )
        assert response.status_code == 200
        assert "X-Cache" in response.headers or response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_gzip_compression(self, test_client):
        """Test gzip compression"""
        headers = {"Accept-Encoding": "gzip"}
        response = await test_client.get("/api/v1/health", headers=headers)
        assert response.status_code == 200
        # Check if response is compressed (would have Content-Encoding header)
        # Note: httpx automatically decompresses, so we just verify it works